"""
Streaming writers for OME-Zarr files.

Provides memory-efficient streaming writers for large tiles that don't
fit in RAM, using block-by-block processing.
"""

from __future__ import annotations

import logging
import math
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import dask.array as da
import numpy as np
import zarr
from ngff_zarr import to_multiscales, to_ngff_image, to_ngff_zarr

from .metadata import _prepare_ngff_writer_metadata, default_channel_colors
from .zarr_compat import create_group_array, open_group_v2

logger = logging.getLogger(__name__)


def _normalize_progress_mode(progress_mode: str | bool | None) -> str:
    if progress_mode is True:
        return "both"
    if progress_mode is False or progress_mode is None:
        return "none"
    normalized = str(progress_mode).strip().lower().replace("_", "-")
    aliases = {
        "true": "both",
        "yes": "both",
        "1": "both",
        "false": "none",
        "no": "none",
        "0": "none",
        "off": "none",
        "log": "log",
        "logs": "log",
        "logging": "log",
        "tqdm": "tqdm",
        "bar": "tqdm",
        "both": "both",
        "none": "none",
    }
    if normalized not in aliases:
        raise ValueError("progress_mode must be one of 'none', 'log', 'tqdm', or 'both'.")
    return aliases[normalized]


def _omero_version(root_attrs: dict[str, Any], schema: str) -> str:
    """Choose an OMERO version string consistent with the emitted root attrs."""
    try:
        version = root_attrs["multiscales"][0].get("version")
    except (KeyError, IndexError, TypeError):
        version = None
    if version is not None:
        return str(version)
    return "latest" if schema == "latest" else "0.4"


def _build_omero_block(
    *,
    name: str,
    version: str,
    channel_labels: list[str],
    channel_colors: list[str] | None,
) -> dict[str, Any]:
    """Build the standard OMERO display block used by the streaming writers."""
    colors = channel_colors or default_channel_colors(len(channel_labels))
    if len(colors) != len(channel_labels):
        raise ValueError(
            f"Writer received {len(colors)} channel colors for {len(channel_labels)} channels."
        )
    return {
        "name": name,
        "version": version,
        "rdefs": {"model": "color", "defaultZ": 0, "defaultT": 0},
        "channels": [
            {
                "label": channel_labels[idx],
                "color": colors[idx],
                "window": {"start": 0.0, "end": 255.0, "min": 0.0, "max": 255.0},
                "active": True,
                "inverted": False,
                "coefficient": 1.0,
                "family": "linear",
            }
            for idx in range(len(channel_labels))
        ],
    }


def _create_group_array(group: Any, name: str, **kwargs: Any):
    """Create an array on a Zarr group across v2/v3 API differences."""
    if "chunks" not in kwargs:
        create_array = getattr(group, "create_array", None)
        if callable(create_array):
            return create_array(name, **kwargs)
        return group.create_dataset(name, **kwargs)
    return create_group_array(group, name, **kwargs)


def _add_translation_to_multiscales(
    root_attrs: dict[str, Any],
    *,
    translation: list[float],
) -> dict[str, Any]:
    """Append NGFF translation transforms to every dataset when requested."""
    if not translation:
        return root_attrs
    attrs = dict(root_attrs)
    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, list):
        return attrs
    attrs["multiscales"] = [dict(ms) if isinstance(ms, dict) else ms for ms in multiscales]
    for multiscale in attrs["multiscales"]:
        if not isinstance(multiscale, dict):
            continue
        datasets = multiscale.get("datasets")
        if not isinstance(datasets, list):
            continue
        new_datasets = []
        for dataset in datasets:
            if not isinstance(dataset, dict):
                new_datasets.append(dataset)
                continue
            ds = dict(dataset)
            transforms = list(ds.get("coordinateTransformations") or [])
            transforms = [
                t
                for t in transforms
                if not (isinstance(t, dict) and t.get("type") == "translation")
            ]
            transforms.append({"type": "translation", "translation": [float(v) for v in translation]})
            ds["coordinateTransformations"] = transforms
            new_datasets.append(ds)
        multiscale["datasets"] = new_datasets
    return attrs


def _max_pool_2x(mask: np.ndarray) -> np.ndarray:
    """Conservative 2x downsampling for binary label masks."""
    h, w = mask.shape
    if h == 1 and w == 1:
        return mask
    out_h = max(1, h >> 1)
    out_w = max(1, w >> 1)
    cropped = mask[: out_h * 2, : out_w * 2]
    if cropped.size == 0:
        return mask[::2, ::2]
    return cropped.reshape(out_h, 2, out_w, 2).max(axis=(1, 3)).astype(mask.dtype)


def write_ngff_from_tile_ts(
    tile_yxc: np.ndarray | da.Array,
    out_path: str | Path,
    base_px_um_xy: tuple[float, float] | None = None,
    *,
    chunks_xy: int = 512,
    num_mips: int = 8,
    name: str = "image",
    version: str = "0.4",
    channel_labels: list[str] | None = None,
    channel_colors: list[str] | None = None,
    ngff_metadata: dict[str, Any] | None = None,
    metadata_schema: Literal["latest", "v0.4", "0.4"] | None = None,
) -> None:
    """
    Stream a large (Y,X,C) tile to an OME-Zarr multiscale using ngff-zarr
    with the TensorStore backend (no full in-RAM pyramid).

    Parameters
    ----------
    tile_yxc : np.ndarray or da.Array
        Input tile (Y, X, C).
    out_path : str or Path
        Output path for OME-Zarr.
    base_px_um_xy : tuple, optional
        Physical pixel size (px_um, py_um) in micrometers.
    chunks_xy : int
        Chunk size in X and Y dimensions.
    num_mips : int
        Number of MIP levels to generate.
    name : str
        Name for the image in metadata.
    version : str
        OME-NGFF version (default "0.4").
    channel_labels : list of str, optional
        Labels for each channel.
    channel_colors : list of str, optional
        Colors for each channel (hex strings).
    ngff_metadata : dict, optional
        Full ``get_vsi_metadata()`` payload or a direct NGFF root-attrs payload.
    metadata_schema : {"latest", "v0.4", "0.4"}, optional
        Schema alias used when ``ngff_metadata`` provides dual projections.
    """
    out_path = str(Path(out_path))
    if len(tile_yxc.shape) != 3:
        raise ValueError(f"tile_yxc must be (Y,X,C), got shape {tile_yxc.shape}")
    Y, X, C = map(int, tile_yxc.shape)
    dataset_count = max(1, num_mips)
    prepared = _prepare_ngff_writer_metadata(
        dataset_count=dataset_count,
        channel_count=C,
        name=name,
        fallback_phys_xy_um=base_px_um_xy,
        ngff_metadata=ngff_metadata,
        metadata_schema=metadata_schema,
        channel_labels=channel_labels,
    )
    root_attrs = prepared["root_attrs"]
    resolved_name = prepared["resolved_name"]
    resolved_labels = prepared["resolved_channel_labels"]
    resolved_phys_xy_um = prepared["resolved_phys_xy_um"] or (1.0, 1.0)
    omero_version = _omero_version(root_attrs, prepared["schema"])
    px_um, py_um = resolved_phys_xy_um

    # Ensure Dask with reasonable chunking
    if not isinstance(tile_yxc, da.Array):
        tile_yxc = da.from_array(tile_yxc, chunks=(chunks_xy, chunks_xy, C))
    tile_cyx = da.moveaxis(tile_yxc, -1, 0).rechunk((C, chunks_xy, chunks_xy))

    # Base image (dims c,y,x with micrometer units)
    img = to_ngff_image(
        data=tile_cyx,
        dims=("c", "y", "x"),
        scale={"c": 1.0, "y": float(py_um), "x": float(px_um)},
        name=resolved_name,
        axes_units={"y": "micrometer", "x": "micrometer"},
    )

    # Ask ngff-zarr to build a pyramid lazily; 2x downsampling per level
    levels = [2] * (max(1, num_mips) - 1)
    ms = to_multiscales(
        img,
        scale_factors=levels,
        chunks={"c": C, "y": chunks_xy, "x": chunks_xy},
    )

    # Write via TensorStore for out-of-core, chunked IO
    to_ngff_zarr(
        store=out_path,
        multiscales=ms,
        version=version,
        overwrite=True,
        use_tensorstore=True,
    )

    root = zarr.open_group(out_path, mode="r+")
    attrs = dict(root_attrs)
    attrs["omero"] = _build_omero_block(
        name=resolved_name,
        version=omero_version,
        channel_labels=resolved_labels,
        channel_colors=channel_colors,
    )
    root.attrs.put(attrs)


def write_ngff_from_tile_streaming_ome(
    tile_yxc_da: da.Array,
    out_dir: Path | str,
    phys_xy_um: tuple[float, float] | None = None,
    *,
    block_xy: int = 512,
    num_mips: int,
    name: str = "image",
    compressor=None,
    dtype: str = "uint8",
    channel_labels: list[str] | None = None,
    channel_colors: list[str] | None = None,
    ngff_metadata: dict[str, Any] | None = None,
    metadata_schema: Literal["latest", "v0.4", "0.4"] | None = None,
    fill_value: int | float | None = 0,
    sparse_zero_chunks: bool = False,
    coordinate_translation_yx_um: tuple[float, float] | None = None,
    run_manifest: dict[str, Any] | None = None,
    progress_mode: str | bool | None = "none",
    progress_interval_s: float = 30.0,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Constant-memory, blockwise OME-Zarr writer (v0.4), no ngff-zarr.

    Streams each mip directly; attaches multiscales + OMERO metadata.

    Parameters
    ----------
    tile_yxc_da : da.Array
        Lazy Dask array (Y, X, C).
    out_dir : Path or str
        Output directory for OME-Zarr.
    phys_xy_um : tuple, optional
        Physical pixel size (px_um, py_um) in micrometers.
    block_xy : int
        Block size for streaming.
    num_mips : int
        Number of MIP levels to generate.
    name : str
        Name for the image in metadata.
    compressor : optional
        Zarr compressor (e.g., zarr.Blosc).
    dtype : str
        Output dtype.
    channel_labels : list of str, optional
        Labels for each channel.
    channel_colors : list of str, optional
        Colors for each channel (hex strings).
    ngff_metadata : dict, optional
        Full ``get_vsi_metadata()`` payload or a direct NGFF root-attrs payload.
    metadata_schema : {"latest", "v0.4", "0.4"}, optional
        Schema alias used when ``ngff_metadata`` provides dual projections.
    progress_mode : {"none", "log", "tqdm", "both"} or bool
        Progress reporting mode. ``True`` is treated as ``"both"``.
    progress_interval_s : float
        Minimum seconds between log progress updates.
    progress_callback : callable, optional
        Called after each base block with progress details.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = open_group_v2(str(out_dir), mode="w")

    # Shapes / scales
    if tile_yxc_da.ndim != 3:
        raise ValueError(f"tile_yxc_da must be (Y,X,C), got shape {tile_yxc_da.shape}")
    Y, X, C = map(int, tile_yxc_da.shape)
    prepared = _prepare_ngff_writer_metadata(
        dataset_count=num_mips,
        channel_count=C,
        name=name,
        fallback_phys_xy_um=phys_xy_um,
        ngff_metadata=ngff_metadata,
        metadata_schema=metadata_schema,
        channel_labels=channel_labels,
    )
    root_attrs = prepared["root_attrs"]
    if coordinate_translation_yx_um is not None:
        trans_y, trans_x = map(float, coordinate_translation_yx_um)
        root_attrs = _add_translation_to_multiscales(
            root_attrs,
            translation=[0.0, trans_y, trans_x],
        )
    resolved_name = prepared["resolved_name"]
    resolved_labels = prepared["resolved_channel_labels"]
    omero_version = _omero_version(root_attrs, prepared["schema"])

    shapes = [(C, max(1, Y >> m), max(1, X >> m)) for m in range(num_mips)]
    chunks = [(C, min(block_xy, sy), min(block_xy, sx)) for (_c, sy, sx) in shapes]

    # Create per-scale arrays (C,Y,X) with the same chunking
    arrays = []
    for m, (sc, sy, sx) in enumerate(shapes):
        arr = _create_group_array(
            root,
            f"s{m}",
            shape=(sc, sy, sx),
            chunks=chunks[m],
            dtype=dtype,
            compressor=compressor,
            fill_value=fill_value,
            overwrite=True,
            zarr_format=2,
        )
        arrays.append(arr)

    # Stream blocks from the lazy Dask source and write all mips
    normalized_progress = _normalize_progress_mode(progress_mode)
    total_blocks_y = max(1, math.ceil(Y / block_xy))
    total_blocks_x = max(1, math.ceil(X / block_xy))
    total_blocks = total_blocks_y * total_blocks_x
    blocks_done = 0
    started = time.monotonic()
    last_log = started
    progress_bar = None
    stats: dict[str, Any] = {
        "name": name,
        "shape_yxc": [Y, X, C],
        "chunks_xy": int(block_xy),
        "sparse_zero_chunks": bool(sparse_zero_chunks),
        "rgb_chunks_written": 0,
        "rgb_chunks_skipped": 0,
        "fill_value": fill_value,
        "pyramid_generation_policy": "downsample_streamed_s0",
    }
    if run_manifest is not None:
        run_manifest.update(stats)

    if normalized_progress in {"tqdm", "both"}:
        try:
            from tqdm.auto import tqdm

            progress_bar = tqdm(total=total_blocks, desc=f"{name} s0 blocks", unit="block")
        except Exception as exc:
            logger.warning("tqdm progress requested but unavailable (%s); continuing with logs.", exc)
            normalized_progress = "log" if normalized_progress == "tqdm" else "both"

    logger.info(
        "Streaming OME-Zarr %s to %s: shape=(%d, %d, %d), blocks=%d, block_xy=%d, mips=%d.",
        name,
        out_dir,
        Y,
        X,
        C,
        total_blocks,
        block_xy,
        num_mips,
    )

    try:
        for y0 in range(0, Y, block_xy):
            y1 = min(Y, y0 + block_xy)
            for x0 in range(0, X, block_xy):
                x1 = min(X, x0 + block_xy)
                # Read a small block lazily -> NumPy
                blk = tile_yxc_da[y0:y1, x0:x1, :].astype(dtype).compute()

                # Write mip 0
                if sparse_zero_chunks and not np.any(blk):
                    stats["rgb_chunks_skipped"] += 1
                else:
                    arrays[0][:, y0:y1, x0:x1] = np.moveaxis(blk, -1, 0)
                    stats["rgb_chunks_written"] += 1

                # Downsample and write higher mips by stride slicing
                src = blk
                for m in range(1, num_mips):
                    # Integer decimation by 2 per level
                    src = src[::2, ::2, :]
                    if src.size == 0:
                        break
                    ym0 = y0 >> m
                    xm0 = x0 >> m
                    ym1 = ym0 + src.shape[0]
                    xm1 = xm0 + src.shape[1]
                    if sparse_zero_chunks and not np.any(src):
                        continue
                    arrays[m][:, ym0:ym1, xm0:xm1] = np.moveaxis(src, -1, 0)

                blocks_done += 1
                now = time.monotonic()
                elapsed = now - started
                blocks_per_sec = blocks_done / elapsed if elapsed > 0 else 0.0
                blocks_left = total_blocks - blocks_done
                eta_s = blocks_left / blocks_per_sec if blocks_per_sec > 0 else None
                event = {
                    "name": name,
                    "out_dir": str(out_dir),
                    "blocks_done": blocks_done,
                    "total_blocks": total_blocks,
                    "elapsed_s": elapsed,
                    "blocks_per_sec": blocks_per_sec,
                    "eta_s": eta_s,
                    "y0": y0,
                    "y1": y1,
                    "x0": x0,
                    "x1": x1,
                }
                if progress_callback is not None:
                    progress_callback(event)
                if run_manifest is not None:
                    run_manifest.update(stats)
                    run_manifest["blocks_done"] = int(blocks_done)
                    run_manifest["total_blocks"] = int(total_blocks)
                    run_manifest["elapsed_s"] = float(elapsed)
                if progress_bar is not None:
                    progress_bar.update(1)
                should_log = (
                    normalized_progress in {"log", "both"}
                    and progress_interval_s >= 0
                    and (now - last_log >= progress_interval_s or blocks_done == total_blocks)
                )
                if should_log:
                    eta_text = f"{eta_s:.1f}s" if eta_s is not None else "unknown"
                    logger.info(
                        "Streaming OME-Zarr progress [%s]: %d/%d blocks, elapsed %.1fs, "
                        "%.2f blocks/sec, ETA %s, out=%s.",
                        name,
                        blocks_done,
                        total_blocks,
                        elapsed,
                        blocks_per_sec,
                        eta_text,
                        out_dir,
                    )
                    last_log = now
    finally:
        if progress_bar is not None:
            progress_bar.close()

    root.attrs.update(root_attrs)

    # Optional OMERO display metadata (napari will use it)
    root.attrs["omero"] = _build_omero_block(
        name=resolved_name,
        version=omero_version,
        channel_labels=resolved_labels,
        channel_colors=channel_colors,
    )
    stats["elapsed_s"] = time.monotonic() - started
    return stats


def write_tissue_mask_label_pyramid(
    mask_yx_da: da.Array,
    out_dir: Path | str,
    phys_xy_um: tuple[float, float],
    *,
    block_xy: int,
    num_mips: int,
    compressor=None,
    sparse_zero_chunks: bool = True,
    coordinate_translation_yx_um: tuple[float, float] | None = None,
    label_name: str = "tissue_mask",
    metadata_schema: Literal["latest", "v0.4", "0.4"] | None = "v0.4",
) -> dict[str, Any]:
    """Write a spatial Y/X OME-Zarr image-label pyramid under labels/tissue_mask."""
    out_dir = Path(out_dir)
    root = open_group_v2(str(out_dir), mode="r+")
    labels_group = root.create_group("labels", overwrite=True)
    labels_group.attrs["labels"] = [label_name]
    label_group = labels_group.create_group(label_name, overwrite=True)

    if mask_yx_da.ndim != 2:
        raise ValueError(f"mask_yx_da must be (Y,X), got shape {mask_yx_da.shape}")
    y_size, x_size = map(int, mask_yx_da.shape)
    py_um = float(phys_xy_um[1])
    px_um = float(phys_xy_um[0])
    trans_y = float(coordinate_translation_yx_um[0]) if coordinate_translation_yx_um else 0.0
    trans_x = float(coordinate_translation_yx_um[1]) if coordinate_translation_yx_um else 0.0

    datasets = []
    arrays = []
    for m in range(num_mips):
        sy = max(1, y_size >> m)
        sx = max(1, x_size >> m)
        scale = [py_um * (2**m), px_um * (2**m)]
        datasets.append(
            {
                "path": f"s{m}",
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale},
                    {"type": "translation", "translation": [trans_y, trans_x]},
                ],
            }
        )
        arr = _create_group_array(
            label_group,
            f"s{m}",
            shape=(sy, sx),
            chunks=(min(block_xy, sy), min(block_xy, sx)),
            dtype="uint8",
            compressor=compressor,
            fill_value=0,
            overwrite=True,
            zarr_format=2,
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
        arrays.append(arr)

    label_group.attrs["multiscales"] = [
        {
            "version": "0.4" if metadata_schema in {"v0.4", "0.4", None} else str(metadata_schema),
            "name": label_name,
            "axes": [
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets,
        }
    ]
    label_group.attrs["image-label"] = {
        "version": "0.4",
        "colors": [
            {"label-value": 0, "rgba": [0, 0, 0, 0]},
            {"label-value": 1, "rgba": [0, 255, 0, 128]},
        ],
        "properties": [
            {"label-value": 0, "class": "background"},
            {"label-value": 1, "class": "tissue"},
        ],
        "source": {"image": "../../"},
    }
    label_group.attrs["mask_pyramid_policy"] = "max_pool"
    label_group.attrs["mask_pyramid_semantics"] = "conservative_visualization"

    total_blocks_y = max(1, math.ceil(y_size / block_xy))
    total_blocks_x = max(1, math.ceil(x_size / block_xy))
    stats = {
        "mask_chunks_written": 0,
        "mask_chunks_skipped": 0,
        "total_blocks": total_blocks_y * total_blocks_x,
        "mask_pyramid_policy": "max_pool",
        "mask_pyramid_semantics": "conservative_visualization",
    }
    for y0 in range(0, y_size, block_xy):
        y1 = min(y_size, y0 + block_xy)
        for x0 in range(0, x_size, block_xy):
            x1 = min(x_size, x0 + block_xy)
            blk = mask_yx_da[y0:y1, x0:x1].astype("uint8").compute()
            if sparse_zero_chunks and not np.any(blk):
                stats["mask_chunks_skipped"] += 1
            else:
                arrays[0][y0:y1, x0:x1] = blk
                stats["mask_chunks_written"] += 1
            src = blk
            for m in range(1, num_mips):
                src = _max_pool_2x(src)
                if src.size == 0:
                    break
                ym0 = y0 >> m
                xm0 = x0 >> m
                ym1 = ym0 + src.shape[0]
                xm1 = xm0 + src.shape[1]
                if sparse_zero_chunks and not np.any(src):
                    continue
                arrays[m][ym0:ym1, xm0:xm1] = src
    return stats
