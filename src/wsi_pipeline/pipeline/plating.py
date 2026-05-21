# ---------------------------
# Per slide processing with per-tissue dual write
# ---------------------------
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import dask.array as da
import dask.config
import numpy as np
import zarr
from numcodecs import Blosc

from ..config import SegmentationConfig, TileConfig
from ..omezarr.metadata import (
    _get_multiscales_paths,
    _phys_xy_um,
    _project_source_metadata_for_tile_writes,
    default_channel_colors,
    default_channel_labels,
)
from ..omezarr.pyramid import build_mips_from_yxc, compute_num_mips_min_side
from ..omezarr.streaming import write_ngff_from_tile_streaming_ome, write_ngff_from_tile_ts
from ..omezarr.writers import write_ngff_from_mips_ngffzarr
from ..precomputed.plate_writer import PlatePrecomputedWriter
from ..segmentation.segmenter import make_lowres_mask
from ..tiles.generator import TissueTileRecord, generate_tissue_tile_records
from ..wsi_processing import segment_tissue

logger = logging.getLogger(__name__)


def _normalize_compression_mode(compression: str | None) -> str:
    normalized = str(compression or "none").strip().lower().replace("_", "-")
    aliases = {
        "none": "none",
        "uncompressed": "none",
        "raw": "none",
        "false": "none",
        "0": "none",
        "lossless": "lossless",
        "compressed": "lossless",
        "zstd": "lossless",
        "true": "lossless",
        "1": "lossless",
    }
    if normalized not in aliases:
        raise ValueError("compression must be one of 'none' or 'lossless'.")
    return aliases[normalized]


def _compressor_for_compression_mode(compression: str):
    compression = _normalize_compression_mode(compression)
    if compression == "none":
        return None
    return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


def _is_big_tile(tile_da, bytes_per_px=1, min_side=8192, max_bytes=1_500_000_000):
    y, x, c = map(int, tile_da.shape)
    return (max(y, x) >= min_side) or (y * x * c * bytes_per_px >= max_bytes)


def _safe_close_existing_client():
    # If a client already exists (e.g., from a previous run), close it to avoid port 8787 conflicts.
    from dask.distributed import get_client

    try:
        c = get_client()
        try:
            c.shutdown()  # politely stop scheduler + workers if we own them
        except Exception:
            c.close()  # at least drop our client
    except ValueError:
        pass  # no existing client


def _resolve_source_ngff_metadata(
    source_context: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, str]:
    """
    Resolve optional source metadata context for plating writes.

    Returns
    -------
    tuple
        ``(metadata_payload, metadata_schema)`` where payload is either a rich
        NGFF metadata dict or ``None``.
    """
    metadata_schema = "v0.4"
    if source_context is None:
        return None, metadata_schema

    metadata_schema = str(source_context.get("metadata_schema") or "v0.4")
    supplied_metadata = source_context.get("ngff_metadata")
    if isinstance(supplied_metadata, dict):
        return supplied_metadata, metadata_schema

    source_kind = str(source_context.get("source_kind") or "unknown").strip().lower()
    source_path = source_context.get("source_path")
    if source_kind == "vsi" and source_path:
        metadata_backend = str(source_context.get("metadata_backend") or "auto")
        try:
            from ..vsi_converter import get_vsi_metadata

            metadata = get_vsi_metadata(
                source_path,
                metadata_backend=metadata_backend,
                target_schema="latest",
            )
            if isinstance(metadata, dict) and metadata:
                return metadata, metadata_schema
            logger.warning(
                "VSI metadata extraction returned empty metadata for %s; continuing with phys_xy_um fallback.",
                source_path,
            )
        except Exception as exc:
            logger.warning(
                "Unable to resolve VSI metadata for %s (%s); continuing with phys_xy_um fallback.",
                source_path,
                exc,
            )
    return None, metadata_schema


def _tile_ngff_metadata_or_none(
    source_metadata: dict[str, Any] | None,
    *,
    dataset_count: int,
    name: str,
    phys_xy_um: tuple[float, float] | None = None,
) -> dict[str, Any] | None:
    """Project resolved source metadata to tile-compatible NGFF payloads."""
    if source_metadata is None:
        return None
    try:
        return _project_source_metadata_for_tile_writes(
            source_metadata,
            dataset_count=dataset_count,
            name=name,
            phys_xy_um=phys_xy_um,
        )
    except Exception as exc:
        logger.warning(
            "Failed to project source metadata for tile '%s' (%s); continuing with phys_xy_um fallback.",
            name,
            exc,
        )
        return None


def _channel_labels_for_count(channel_count: int) -> list[str]:
    """Return default brightfield labels for writer calls."""
    return default_channel_labels(int(channel_count))


def _channel_colors_for_count(channel_count: int) -> list[str]:
    """Return default brightfield colors for writer calls."""
    return default_channel_colors(int(channel_count))


def _clean_ome_zarr_stem(path: Path) -> str:
    """Return a readable source name, stripping the compound OME-Zarr suffix."""
    name = path.name
    return name[: -len(".ome.zarr")] if name.endswith(".ome.zarr") else path.stem


def _resolve_level_path(
    ds_paths: list[str],
    level: int | str | None,
    *,
    default_index: int,
    label: str,
) -> tuple[int, str]:
    """Resolve an integer multiscales index or dataset path to ``(index, path)``."""
    if not ds_paths:
        raise ValueError("OME-Zarr source does not contain any multiscales datasets.")
    if level is None:
        idx = default_index
    elif isinstance(level, str):
        if level not in ds_paths:
            raise ValueError(
                f"{label} dataset path {level!r} was not found in multiscales datasets {ds_paths}."
            )
        idx = ds_paths.index(level)
    else:
        idx = int(level)
        if idx < 0:
            idx = len(ds_paths) + idx
    if idx < 0 or idx >= len(ds_paths):
        raise ValueError(f"{label} level {level!r} is out of range for {len(ds_paths)} datasets.")
    return idx, ds_paths[idx]


def _segmentation_kwargs(
    segmentation_config: SegmentationConfig | None,
    *,
    min_size: int,
    struct_elem_px: int,
) -> dict[str, Any]:
    """Translate notebook-style segmentation config into the low-res mask API."""
    if segmentation_config is None:
        return {
            "dynamic_threshold": True,
            "fixed_threshold": 0.7,
            "min_size": min_size,
            "struct_elem_px": struct_elem_px,
            "additional_smooth": False,
            "output_images": False,
        }

    return {
        "dynamic_threshold": True,
        "fixed_threshold": 0.7,
        "min_size": segmentation_config.min_area_px,
        "struct_elem_px": segmentation_config.struct_elem_px,
        "additional_smooth": False,
        "output_images": False,
        "keep_top_k": segmentation_config.keep_top_k,
        "stain_gate": segmentation_config.stain_gate,
        "stain_gate_mode": segmentation_config.stain_gate_mode,
        "stain_min_saturation": segmentation_config.stain_min_saturation,
        "stain_min_od": segmentation_config.stain_min_od,
        "stain_min_he_signal": segmentation_config.stain_min_he_signal,
        "stain_od_bg_percentile": segmentation_config.stain_od_bg_percentile,
        "stain_od_mad_multiplier": segmentation_config.stain_od_mad_multiplier,
        "stain_pre_open_px": segmentation_config.stain_pre_open_px,
        "split_touching": segmentation_config.split_touching,
        "r_split": segmentation_config.r_split,
        "appendage_refinement_enabled": segmentation_config.appendage_refinement_enabled,
        "appendage_refinement_mode": segmentation_config.appendage_refinement_mode,
        "appendage_refinement_profile": segmentation_config.appendage_refinement_profile,
        "diagnostics": segmentation_config.diagnostics,
        "return_diag": segmentation_config.diagnostics,
    }


def _segment_for_plating(
    image: np.ndarray | da.Array,
    *,
    segment_fn,
    segmentation_config: SegmentationConfig | None,
    min_size: int,
    struct_elem_px: int,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """
    Segment a low-resolution plating image.

    When a notebook-style SegmentationConfig is supplied and no explicit
    segment_fn override is present, use the same segment_tissue() path as
    notebook 01. Passing segment_fn remains the compatibility shim for the
    legacy make_lowres_mask/preprocess API.
    """
    if segment_fn is not None:
        return segment_fn(
            image,
            **_segmentation_kwargs(
                segmentation_config,
                min_size=min_size,
                struct_elem_px=struct_elem_px,
            ),
        )

    if segmentation_config is not None:
        mask, info = segment_tissue(
            image,
            backend=segmentation_config.backend,
            target_long_side=segmentation_config.target_long_side,
            min_area_px=segmentation_config.min_area_px,
            struct_elem_px=segmentation_config.struct_elem_px,
            stain_gate=segmentation_config.stain_gate,
            stain_gate_mode=segmentation_config.stain_gate_mode,
            stain_min_saturation=segmentation_config.stain_min_saturation,
            stain_min_od=segmentation_config.stain_min_od,
            stain_min_he_signal=segmentation_config.stain_min_he_signal,
            stain_od_bg_percentile=segmentation_config.stain_od_bg_percentile,
            stain_od_mad_multiplier=segmentation_config.stain_od_mad_multiplier,
            stain_pre_open_px=segmentation_config.stain_pre_open_px,
            split_touching=segmentation_config.split_touching,
            r_split=segmentation_config.r_split,
            keep_top_k=segmentation_config.keep_top_k,
            appendage_refinement_enabled=segmentation_config.appendage_refinement_enabled,
            appendage_refinement_mode=segmentation_config.appendage_refinement_mode,
            appendage_refinement_profile=segmentation_config.appendage_refinement_profile,
            diagnostics=segmentation_config.diagnostics,
        )
        return mask, info

    return make_lowres_mask(
        image,
        **_segmentation_kwargs(
            None,
            min_size=min_size,
            struct_elem_px=struct_elem_px,
        ),
    )


def _source_context_path(
    source_context: Mapping[str, Any] | None,
    *keys: str,
) -> str | None:
    if source_context is None:
        return None
    for key in keys:
        value = source_context.get(key)
        if value is not None:
            return str(value)
    return None


def _source_vsi_from_context(source_context: Mapping[str, Any] | None) -> str | None:
    if source_context is None:
        return None
    for key in ("source_vsi", "vsi_path"):
        value = source_context.get(key)
        if value is not None:
            return str(value)
    source_kind = str(source_context.get("source_kind") or "").lower()
    source_path = source_context.get("source_path")
    if source_kind == "vsi" and source_path is not None:
        return str(source_path)
    return None


def _debug_bounds_xyxy(frame_debug: Mapping[str, Any], key: str) -> list[int] | None:
    """Return a frame-debug YX bounds payload as export-friendly XYXY bounds."""
    payload = frame_debug.get(key)
    if not isinstance(payload, Mapping):
        return None
    try:
        return [
            int(payload["x0"]),
            int(payload["y0"]),
            int(payload["x1"]),
            int(payload["y1"]),
        ]
    except KeyError:
        return None


def _write_tissue_manifest(
    ngff_dir: Path,
    *,
    record: TissueTileRecord,
    source_context: Mapping[str, Any] | None,
    source_ome_zarr: Path | None,
    source_level: int,
    segmentation_level: int,
    phys_xy_um: tuple[float, float],
) -> Path:
    """Write the chopped-derivative manifest next to a tissue OME-Zarr."""
    px_um, py_um = map(float, phys_xy_um)
    payload = {
        "role": "derivative",
        "derivative_type": "tissue_crop_ome_zarr",
        "source_vsi": _source_vsi_from_context(source_context),
        "source_ets": _source_context_path(source_context, "source_ets", "ets_path"),
        "source_ome_zarr": _source_context_path(source_context, "source_ome_zarr")
        or (str(source_ome_zarr) if source_ome_zarr is not None else None),
        "source_level": int(source_level),
        "segmentation_level": int(segmentation_level),
        "tissue_index": int(record.tissue_index),
        "crop_bounds_source_level": list(record.crop_bounds_source_level),
        "crop_bounds_segmentation_level": list(record.crop_bounds_segmentation_level),
        "physical_pixel_size": {"x": px_um, "y": py_um, "unit": "micrometer"},
        "operations": [
            "read_ets_pyramid",
            "segment_lowres",
            "extract_tissue",
            "write_ome_zarr",
        ],
    }
    if record.frame_debug is not None:
        frame_debug = record.frame_debug
        logical_source = _debug_bounds_xyxy(frame_debug, "logical_canvas_source_yx")
        clipped_source = _debug_bounds_xyxy(frame_debug, "clipped_source_yx")
        logical_segmentation = _debug_bounds_xyxy(frame_debug, "logical_frame_seg_yx")
        clipped_segmentation = _debug_bounds_xyxy(frame_debug, "clipped_frame_seg_yx")
        if logical_source is not None:
            payload["logical_crop_bounds_source_level"] = logical_source
        if clipped_source is not None:
            payload["clipped_crop_bounds_source_level"] = clipped_source
        if logical_segmentation is not None:
            payload["logical_crop_bounds_segmentation_level"] = logical_segmentation
        if clipped_segmentation is not None:
            payload["clipped_crop_bounds_segmentation_level"] = clipped_segmentation
        padding = frame_debug.get("padding_source_level")
        if isinstance(padding, Mapping):
            payload["padding_source_level"] = {
                "top": int(padding.get("top", 0)),
                "bottom": int(padding.get("bottom", 0)),
                "left": int(padding.get("left", 0)),
                "right": int(padding.get("right", 0)),
            }
        if frame_debug.get("tile_frame_level") is not None:
            payload["tile_frame_level"] = str(frame_debug["tile_frame_level"])

    manifest_path = ngff_dir / "tissue_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _write_tissue_frame_debug(
    ngff_dir: Path,
    *,
    record: TissueTileRecord,
    source_level: int,
    segmentation_level: int,
) -> Path | None:
    """Write optional rich coordinate diagnostics as a sidecar JSON."""
    if record.frame_debug is None:
        return None
    payload = {
        "source_level": int(source_level),
        "segmentation_level": int(segmentation_level),
        "tissue_index": int(record.tissue_index),
        "label_id": int(record.label_id),
        "frame_debug": record.frame_debug,
    }
    debug_path = ngff_dir / "tissue_frame_debug.json"
    debug_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return debug_path


def _record_written_tissue(
    out_paths: list[Path],
    ngff_dir: Path,
    *,
    record: TissueTileRecord,
    source_context: Mapping[str, Any] | None,
    source_ome_zarr: Path | None,
    source_level: int,
    segmentation_level: int,
    phys_xy_um: tuple[float, float],
) -> None:
    _write_tissue_manifest(
        ngff_dir,
        record=record,
        source_context=source_context,
        source_ome_zarr=source_ome_zarr,
        source_level=source_level,
        segmentation_level=segmentation_level,
        phys_xy_um=phys_xy_um,
    )
    _write_tissue_frame_debug(
        ngff_dir,
        record=record,
        source_level=source_level,
        segmentation_level=segmentation_level,
    )
    out_paths.append(ngff_dir)


def process_slide_with_plating(
    zarr_root_path: os.PathLike,
    out_ngff_dir: os.PathLike,  # where to write tissue_region_*.zarr
    *,
    # Coarse segmentation function preprocess() options
    segment_fn=None,  # callable(dask_arr)->(filled_mask_bool, _)
    source_level: int | str = 0,
    segmentation_level: int | str | None = None,
    tile_frame_level: str = "segmentation",
    segmentation_config: SegmentationConfig | None = None,
    tile_config: TileConfig | None = None,
    struct_elem_px: int = 9,  # structuring element radius in pixels
    min_size: int = 2000,
    # Plate options
    precomputed_plate_path: str | None = None,  # "file:///.../plate_precomp"
    plate_backend: str = "tensorstore",  # or "cloudvolume"
    plate_chunk_xy: int = 512,
    parallel: bool = False,
    fill_missing: bool = False,
    # mips options
    min_side_for_mips: int | None = None,  # default to chunk size in writers
    downscale: int = 2,  # default downsampling rate for mips (not used yet)
    tile_extra_margin_px: int = 0,
    # dtype policy
    dtype: np.dtype | None = "uint8",  # cast ROI to uint8 before mip (recommended for imagery)
    source_context: Mapping[str, Any] | None = None,
    compression: str = "none",
    progress_mode: str | bool | None = "none",
    progress_interval_s: float = 30.0,
) -> list[Path]:
    """
    Pipeline for ONE slide root (OME-Zarr):
        1) Run segmentation on the selected pyramid level -> boolean mask with N labels after filling.
        2) Build **Dask** tiles at the selected source level with ROI upsampling & HR masking.
        3) For each tile: compute -> build tinybrain mips (once) -> write per-tissue NGFF.
        4) Optionally append each tissue as a Z-slice into a single Precomputed plate.
    Returns: list of per-tissue NGFF output directories.

    ``source_level`` and ``segmentation_level`` accept either multiscales indices
    or dataset paths such as ``"s7"``. ``segmentation_config`` and
    ``tile_config`` mirror the notebook configuration objects. ``tile_frame_level``
    controls whether crop dimensions/margins are finalized at the segmentation
    level (notebook-equivalent default) or at the source level (legacy behavior).

    Returns
    -------
    region_paths : list of paths. No large arrays are held in memory.
    """
    # Ensure the zarr_root_path is a Path object
    zarr_root_path = Path(zarr_root_path)

    # Make the output directory if it doesn't already exist
    out_ngff_dir = Path(out_ngff_dir)
    out_ngff_dir.mkdir(parents=True, exist_ok=True)
    compression = _normalize_compression_mode(compression)
    compressor = _compressor_for_compression_mode(compression)
    if tile_config is not None:
        plate_chunk_xy = tile_config.chunk_size
        tile_extra_margin_px = tile_config.extra_margin_px
        tile_pad_multiple = tile_config.pad_multiple
    else:
        tile_pad_multiple = plate_chunk_xy

    # Step 1: Load the NGFF root, find image and metadata for the levels
    logger.info("Loading the NGFF root.")

    root = zarr.open_group(str(zarr_root_path), mode="r")
    ds_paths = _get_multiscales_paths(root)  # e.g. ["s0","s1","s2","s3"]
    L_idx, L_path = _resolve_level_path(
        ds_paths,
        source_level,
        default_index=0,
        label="source",
    )
    sL_idx, sL_path = _resolve_level_path(
        ds_paths,
        segmentation_level,
        default_index=len(ds_paths) - 1,
        label="segmentation",
    )
    logger.info(
        "Resolved OME-Zarr levels for %s: source level %d -> %s; segmentation level %d -> %s.",
        zarr_root_path.name,
        L_idx,
        L_path,
        sL_idx,
        sL_path,
    )

    # Load arrays which are stored (C,Y,X) with parallelization; give unique and stable names to prevent collisions in the graph
    RUN_UID = f"{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    dataset_id = zarr_root_path.name.replace(".", "_")
    base_name = f"ngff-{dataset_id}-{L_path}-{RUN_UID}"
    coarse_name = f"ngff-{dataset_id}-{sL_path}-{RUN_UID}"
    base_cyx = da.from_zarr(str(zarr_root_path / L_path), name=base_name)  # (C, H, W)
    sL = da.from_zarr(str(zarr_root_path / sL_path), name=coarse_name)
    logger.info(
        "Using source shape %s and segmentation shape %s.",
        tuple(map(int, base_cyx.shape)),
        tuple(map(int, sL.shape)),
    )

    # # If the source is of modest size, persist once so that all dowstream tiles share one graph; otherwise, avoid entirely, and slice lazily.
    # base_cyx = base_cyx.persist()
    # wait(base_cyx)
    # sL = sL.persist()
    # wait(sL)

    # Should I rechunk to force the arrays to align with the tile size?
    # base_cyx = base_cyx.rechunk((3, plate_chunk_xy, plate_chunk_xy))
    # sL = sL.rechunk((3, plate_chunk_xy, plate_chunk_xy))

    # Double check high res image shape info; channel information is either grayscale or RGB
    if base_cyx.ndim != 3 or base_cyx.shape[0] not in (1, 3):
        raise ValueError("Expected (C,Y,X) at base")

    # Precompute shapes and physical pixel sizes at the highest resolution
    C, H0, W0 = base_cyx.shape
    px_um, py_um = _phys_xy_um(root, L_idx)  # this is the base s0 scale now
    source_ngff_metadata, source_metadata_schema = _resolve_source_ngff_metadata(source_context)
    if source_ngff_metadata is not None:
        logger.info(
            "Resolved source metadata context for plating writes (schema=%s).",
            source_metadata_schema,
        )

    # Step 2: Segment at the coarsest level using the segment_fn to get a tissue region mask
    logger.info("Generating tissue masks at the coarsest resolution.")

    # Ensure channel-first for the grayscale() function within segment_fn
    if sL.ndim == 3 and sL.shape[-1] in (1, 3) and sL.shape[0] not in (1, 3):
        sL = da.moveaxis(sL, -1, 0)  # (C, Y, X)

    filled_lr, _ = _segment_for_plating(
        sL,
        segment_fn=segment_fn,
        segmentation_config=segmentation_config,
        min_size=min_size,
        struct_elem_px=struct_elem_px,
    )

    # Step 2: upsample the low-resolution mask to the high-resolution image
    # Build LIST of HR tile records (Y,X,C), ordered left->right
    tile_records, tile_dim = generate_tissue_tile_records(
        s0_cyx=base_cyx,
        low_res_filled=filled_lr.astype(bool),
        chunk=plate_chunk_xy,
        pad_multiple=tile_pad_multiple,
        extra_margin_px=tile_extra_margin_px,
        tile_frame_level=tile_frame_level,
    )
    # Break out early if there are no tissue sections
    if not tile_records:
        logger.warning("[%s] no tissue regions found.", zarr_root_path.name)
        return []

    plate = None
    if precomputed_plate_path:
        # Z is the number of tiles; voxel_size_z is arbitrary (1.0 um) for 2D plates
        plate = PlatePrecomputedWriter(
            precomp_path=precomputed_plate_path,
            width=tile_dim,
            height=tile_dim,
            z_slices=len(tile_records),
            voxel_size_um=(px_um, py_um, 1.0),
            chunk_xy=plate_chunk_xy,
            min_side_for_mips=min_side_for_mips,
            backend=plate_backend,
            dtype=dtype if dtype else str(base_cyx.dtype),
            encoding="raw",
            parallel=parallel,
            fill_missing=fill_missing,
        )

    # 3) iterate tiles
    out_paths: list[Path] = []

    big_tile_threshold = 8192  # 2^13=8192; tune threshold
    item_size_threshold = 1_500_000_000
    bytes_per_px = np.dtype(dtype if dtype else np.uint8).itemsize
    # nbytes_est = np.prod(np.squeeze(tiles_yxc[0].shape)) * bytes_per_px
    any_big = any(
        _is_big_tile(
            tile_da=record.tile,
            bytes_per_px=bytes_per_px,
            min_side=big_tile_threshold,
            max_bytes=item_size_threshold,
        )
        for record in tile_records
    )
    n_tiles_threshold = 16
    n_tiles = len(tile_records)
    # If any tile is huge, avoid distributed 'compute-then-write' path
    use_distributed = bool(parallel) and (n_tiles >= n_tiles_threshold) and not any_big

    if use_distributed:
        from dask.distributed import Client, LocalCluster, as_completed

        _safe_close_existing_client()

        # Create a local cluster so that we can stream compute tiles. This makes us parallelized across multiple CPUs but not memory-exploding
        # Upper Limit: Each image pyramid should be ~2.5GB so we should be good. Note that agregate_memory = n_workers * memory_limit (e.g. 10 workers, "8GB" -> 80GB)
        # Use context managers so we always shut down cleanly
        n_workers = min(10, os.cpu_count())
        with (
            LocalCluster(
                n_workers=n_workers,
                threads_per_worker=1,
                # processes=False,        # False w/threads may stabilize WSL2 runs
                processes=True,  # separate processes (nanny restarts workers)
                memory_limit="auto",  # let Dask size to process, safer in WSL2
                scheduler_port=0,  # random free port, avoids conflicts
                dashboard_address=None,  # disable dashboard to avoid port issues
            ) as cluster,
            Client(cluster, set_as_default=True) as client,
        ):
            # Set distributed configuration
            dask.config.set(
                {
                    "array.slicing.split_large_chunks": True,  # useful when there are oversized chunks
                    "distributed.worker.memory.target": 0.6,
                    "distributed.worker.memory.spill": 0.7,
                    "distributed.worker.memory.pause": 0.85,  # helps prevent out of memory
                    "distributed.worker.memory.terminate": 0.95,
                    "distributed.comm.timeouts.connect": "20s",
                    "distributed.comm.timeouts.tcp": "120s",
                }
            )

            client.wait_for_workers(1)  # don't block waiting for all workers

            # Submit in batches so we do not flood scheduler/RAM
            batch_size = 8
            for start in range(0, n_tiles, batch_size):
                batch = tile_records[start : start + batch_size]
                fmap = {client.compute(record.tile): (start + i) for i, record in enumerate(batch)}

                for fut in as_completed(fmap):
                    z_idx = fmap.pop(fut)
                    record = tile_records[z_idx]
                    tile = fut.result()  # stores (Y,X,C) numpy array as soon as one tile finishes
                    # dtype policy (optional)
                    if dtype and tile.dtype != np.uint8:
                        logger.debug("Enforcing dtype policy.")
                        if np.issubdtype(tile.dtype, np.integer):
                            maxv = np.iinfo(tile.dtype).max
                            tile = (
                                (tile.astype(np.float32) / max(1, maxv) * 255.0)
                                .clip(0, 255)
                                .astype(np.uint8)
                            )
                        else:
                            tile = (tile * 255.0).clip(0, 255).astype(np.uint16)

                    # Write per-tissue NGFF
                    name = (
                        f"{_clean_ome_zarr_stem(zarr_root_path)}_tissue_{record.tissue_index:02d}"
                    )
                    ngff_dir = out_ngff_dir / f"{name}.ome.zarr"
                    if any_big:
                        # STREAM the NGFF pyramid from base tile via TensorStore
                        num_mips = compute_num_mips_min_side(
                            tile.shape[1],
                            tile.shape[0],
                            min_side_for_mips or plate_chunk_xy,
                        )
                        tile_ngff_metadata = _tile_ngff_metadata_or_none(
                            source_ngff_metadata,
                            dataset_count=num_mips,
                            name=name,
                            phys_xy_um=(px_um, py_um),
                        )
                        write_ngff_from_tile_ts(
                            tile,
                            ngff_dir,
                            (px_um, py_um),
                            chunks_xy=plate_chunk_xy,
                            num_mips=num_mips,
                            name=name,
                            version="0.4",
                            channel_labels=_channel_labels_for_count(tile.shape[2]),
                            channel_colors=_channel_colors_for_count(tile.shape[2]),
                            ngff_metadata=tile_ngff_metadata,
                            metadata_schema=source_metadata_schema,
                        )
                        # Plate: let your writer auto-stream based on its own gate
                        if plate is not None:
                            plate.write_slice(z_idx, tile)
                        _record_written_tissue(
                            out_paths,
                            ngff_dir,
                            record=record,
                            source_context=source_context,
                            source_ome_zarr=zarr_root_path,
                            source_level=L_idx,
                            segmentation_level=sL_idx,
                            phys_xy_um=(px_um, py_um),
                        )
                    else:
                        # Small/medium: keep your fast path (build mips once in RAM)
                        ms = compute_num_mips_min_side(
                            tile.shape[1], tile.shape[0], min_side_for_mips or plate_chunk_xy
                        )
                        tile_ngff_metadata = _tile_ngff_metadata_or_none(
                            source_ngff_metadata,
                            dataset_count=ms,
                            name=name,
                            phys_xy_um=(px_um, py_um),
                        )
                        mips = build_mips_from_yxc(tile, ms)
                        write_ngff_from_mips_ngffzarr(
                            mips_yxc=mips,
                            out_dir=ngff_dir,
                            phys_xy_um=(px_um, py_um),
                            name=name,
                            chunks_xy=plate_chunk_xy,
                            version="0.4",  # keep 0.4 unless you want sharded v0.5
                            overwrite=True,
                            channel_labels=_channel_labels_for_count(mips[0].shape[2]),
                            channel_colors=_channel_colors_for_count(mips[0].shape[2]),
                            add_omero=True,
                            ngff_metadata=tile_ngff_metadata,
                            metadata_schema=source_metadata_schema,
                        )
                        # # This function uses zarr and manually specifies the metadata
                        # write_ngff_from_mips(mips,
                        # ngff_dir,
                        # (px_um, py_um),
                        # name=name,
                        # chunks_xy=plate_chunk_xy,
                        # dtype=mips[0].dtype)
                        _record_written_tissue(
                            out_paths,
                            ngff_dir,
                            record=record,
                            source_context=source_context,
                            source_ome_zarr=zarr_root_path,
                            source_level=L_idx,
                            segmentation_level=sL_idx,
                            phys_xy_um=(px_um, py_um),
                        )
                        if plate is not None:
                            plate.write_slice(z_idx, mips)

                    del tile

                #     # Release worker memory early
                #     fut.release()

                # client.run(gc.collect)
    else:
        # Set distributed configuration
        dask.config.set(
            {
                "array.slicing.split_large_chunks": True  # useful when there are oversized chunks
            }
        )

        # For small jobs, threads are faster and simpler (no scheduler ports, no heartbeats)
        _safe_close_existing_client()

        with dask.config.set(scheduler="threads"):
            for z_idx, record in enumerate(tile_records, start=0):
                tile_dask = record.tile
                # Write per-tissue NGFF
                name = f"{_clean_ome_zarr_stem(zarr_root_path)}_tissue_{record.tissue_index:02d}"
                ngff_dir = out_ngff_dir / f"{name}.ome.zarr"
                logger.debug(
                    "tile %d: %s, big=%s",
                    z_idx,
                    tuple(map(int, tile_dask.shape)),
                    _is_big_tile(tile_dask, bytes_per_px),
                )
                tile = None

                # For big tiles, don't build mips_yxc in memory. Instead, write directly from the base tile with the streaming writer
                if any_big:
                    # DO NOT compute() -- keep it lazy and (optionally) cast lazily
                    # tlazy = tile_dask.astype(np.uint8) if dtype and tile_dask.dtype != np.uint8 else tile_dask
                    num_mips = compute_num_mips_min_side(
                        tile_dask.shape[1],
                        tile_dask.shape[0],
                        min_side_for_mips or plate_chunk_xy,
                    )
                    tile_ngff_metadata = _tile_ngff_metadata_or_none(
                        source_ngff_metadata,
                        dataset_count=num_mips,
                        name=name,
                        phys_xy_um=(px_um, py_um),
                    )

                    # write_ngff_from_tile_ts(
                    #     tlazy,
                    #     ngff_dir,
                    #     (px_um, py_um),
                    #     chunks_xy=plate_chunk_xy,
                    #     num_mips=compute_num_mips_min_side(tlazy.shape[1], tlazy.shape[0],
                    #                                     min_side_for_mips or plate_chunk_xy),
                    #     name=name, version="0.4",
                    #     channel_labels=_channel_labels_for_count(tlazy.shape[2]),
                    #     channel_colors=_channel_colors_for_count(tlazy.shape[2]),
                    # )
                    write_ngff_from_tile_streaming_ome(
                        tile_yxc_da=tile_dask.astype(np.uint8)
                        if tile_dask.dtype != np.uint8
                        else tile_dask,
                        out_dir=ngff_dir,
                        phys_xy_um=(px_um, py_um),
                        block_xy=plate_chunk_xy,
                        num_mips=num_mips,
                        name=name,
                        compressor=compressor,
                        channel_labels=_channel_labels_for_count(int(tile_dask.shape[2])),
                        channel_colors=_channel_colors_for_count(int(tile_dask.shape[2])),
                        ngff_metadata=tile_ngff_metadata,
                        metadata_schema=source_metadata_schema,
                        progress_mode=progress_mode,
                        progress_interval_s=progress_interval_s,
                    )

                    # Append to precomputed plate (optional)
                    if plate is not None:
                        # plate.write_slice(z_idx, tlazy)   # accepts dask arrays, streams blocks
                        plate.write_slice(
                            z_idx,
                            tile_dask.astype(np.uint8)
                            if tile_dask.dtype != np.uint8
                            else tile_dask,
                        )  # accepts dask arrays, streams blocks
                    _record_written_tissue(
                        out_paths,
                        ngff_dir,
                        record=record,
                        source_context=source_context,
                        source_ome_zarr=zarr_root_path,
                        source_level=L_idx,
                        segmentation_level=sL_idx,
                        phys_xy_um=(px_um, py_um),
                    )

                else:
                    tile = tile_dask.compute()  # numpy (Y,X,C); ok for small/medium arrays
                    # dtype policy (optional)
                    if dtype and tile.dtype != np.uint8:
                        logger.debug("Enforcing dtype policy.")
                        if np.issubdtype(tile.dtype, np.integer):
                            maxv = np.iinfo(tile.dtype).max
                            tile = (
                                (tile.astype(np.float32) / max(1, maxv) * 255.0)
                                .clip(0, 255)
                                .astype(np.uint8)
                            )
                        else:
                            tile = (tile * 255.0).clip(0, 255).astype(np.uint16)

                    # Compute mips once
                    ms = compute_num_mips_min_side(
                        tile.shape[1], tile.shape[0], min_side_for_mips or plate_chunk_xy
                    )
                    tile_ngff_metadata = _tile_ngff_metadata_or_none(
                        source_ngff_metadata,
                        dataset_count=ms,
                        name=name,
                        phys_xy_um=(px_um, py_um),
                    )
                    mips = build_mips_from_yxc(tile, ms)
                    # print(f"mips[0].dtype: {mips[0].dtype}")

                    write_ngff_from_mips_ngffzarr(
                        mips_yxc=mips,
                        out_dir=ngff_dir,
                        phys_xy_um=(px_um, py_um),
                        name=name,
                        chunks_xy=plate_chunk_xy,
                        version="0.4",  # keep 0.4 unless you want sharded v0.5
                        overwrite=True,
                        channel_labels=_channel_labels_for_count(mips[0].shape[2]),
                        channel_colors=_channel_colors_for_count(mips[0].shape[2]),
                        add_omero=True,
                        ngff_metadata=tile_ngff_metadata,
                        metadata_schema=source_metadata_schema,
                    )
                    # This function uses zarr and manually specifies the metadata
                    # write_ngff_from_mips(mips,
                    #                     ngff_dir,
                    #                     (px_um, py_um), # base scale physical pixel size
                    #                     name=name,
                    #                     chunks_xy=plate_chunk_xy,
                    #                     dtype=mips[0].dtype)
                    _record_written_tissue(
                        out_paths,
                        ngff_dir,
                        record=record,
                        source_context=source_context,
                        source_ome_zarr=zarr_root_path,
                        source_level=L_idx,
                        segmentation_level=sL_idx,
                        phys_xy_um=(px_um, py_um),
                    )

                    # Append to precomputed plate (optional)
                    if plate is not None:
                        plate.write_slice(z_idx, mips)

                if tile is not None:
                    del tile

    # # Optional: extra cleanup in notebooks so reruns do not collide
    # gc.collect()

    logger.info("Wrote %d tissue OME-Zarrs to %s", len(out_paths), out_ngff_dir)

    try:
        # Shut down the cluster
        client.close()
        client.shutdown()
    except Exception:
        pass

    return out_paths
