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
    progress_mode: str | bool | None = "none",
    progress_interval_s: float = 30.0,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
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
                arrays[0][:, y0:y1, x0:x1] = np.moveaxis(blk, -1, 0)

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
