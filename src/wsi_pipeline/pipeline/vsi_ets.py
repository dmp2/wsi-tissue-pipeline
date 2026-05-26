"""VSI/ETS entry points for source and tissue OME-Zarr generation."""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dask.array as da
import dask.config
import numpy as np
from numcodecs import Blosc
from scipy.ndimage import binary_fill_holes
from skimage import measure

from ..config import SegmentationConfig, TileConfig
from ..etsfile import ETSFile
from ..omezarr.ets_writer import write_ets_pyramid_to_ngff_zarr
from ..omezarr.metadata import default_channel_colors, default_channel_labels
from ..omezarr.pyramid import build_mips_from_yxc, compute_num_mips_min_side
from ..omezarr.streaming import write_ngff_from_tile_streaming_ome, write_tissue_mask_label_pyramid
from ..omezarr.writers import write_ngff_from_mips_ngffzarr
from ..omezarr.zarr_compat import create_group_array, open_group_v2
from ..precomputed.plate_writer import PlatePrecomputedWriter
from ..tiles.generator import (
    BoundsYX,
    TissueTileRecord,
    _build_tissue_frame_specs,
    _normalize_crop_shape_policy,
    _normalize_tile_frame_level,
    project_label_mask_to_source_region,
)
from ..vsi_converter import find_ets_file, get_vsi_metadata
from .plating import (
    _is_big_tile,
    _record_written_tissue,
    _safe_close_existing_client,
    _segment_for_plating,
    _tile_ngff_metadata_or_none,
    process_slide_with_plating,
)

logger = logging.getLogger(__name__)


def _expected_dataset_paths_from_metadata(metadata: dict[str, Any]) -> list[str]:
    canonical = metadata.get("canonical_metadata")
    if isinstance(canonical, dict):
        dataset_paths = canonical.get("dataset_paths")
        if isinstance(dataset_paths, list) and dataset_paths:
            return [str(path) for path in dataset_paths]

    for key in ("ngff", "ngff_v04", "ngff_latest"):
        ngff = metadata.get(key)
        if not isinstance(ngff, dict):
            continue
        multiscales = ngff.get("multiscales")
        if not isinstance(multiscales, list) or not multiscales:
            continue
        datasets = multiscales[0].get("datasets")
        if isinstance(datasets, list) and datasets:
            paths = [dataset.get("path") for dataset in datasets if isinstance(dataset, dict)]
            if all(path is not None for path in paths):
                return [str(path) for path in paths]

    num_levels = metadata.get("num_levels")
    if num_levels is not None:
        return [f"s{level}" for level in range(int(num_levels))]

    return []


def _missing_source_ome_zarr_arrays(output_path: Path, metadata: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for dataset_path in _expected_dataset_paths_from_metadata(metadata):
        array_path = output_path / dataset_path
        if not ((array_path / ".zarray").is_file() or (array_path / "zarr.json").is_file()):
            missing.append(dataset_path)
    return missing


def _source_ome_zarr_shape_errors(
    output_path: Path,
    metadata: dict[str, Any],
    ets_path: Path,
) -> list[str]:
    """Return dataset paths whose cached array shapes do not match the ETS pyramid."""
    dataset_paths = _expected_dataset_paths_from_metadata(metadata)
    if not dataset_paths:
        return []

    channel_count = 3
    canonical = metadata.get("canonical_metadata")
    if isinstance(canonical, dict) and canonical.get("channel_count") is not None:
        channel_count = int(canonical["channel_count"])
    elif metadata.get("channel_count") is not None:
        channel_count = int(metadata["channel_count"])

    errors: list[str] = []
    with ETSFile(ets_path) as ets:
        for level, dataset_path in enumerate(dataset_paths):
            zarray_path = output_path / dataset_path / ".zarray"
            if not zarray_path.is_file():
                continue
            try:
                zarray = json.loads(zarray_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                errors.append(f"{dataset_path} (unreadable .zarray)")
                continue

            height, width = ets.level_shape(level)
            expected_shape = [channel_count, int(height), int(width)]
            actual_shape = zarray.get("shape")
            if list(actual_shape) != expected_shape:
                errors.append(f"{dataset_path} shape={actual_shape!r}, expected={expected_shape!r}")
    return errors


def _temporary_source_path(output_path: Path) -> Path:
    """Return the hidden incomplete output path used for source conversion."""
    return output_path.with_name(f".{output_path.name}.incomplete")


def _promote_completed_source(temp_path: Path, output_path: Path) -> None:
    """Replace a source OME-Zarr only after a complete temp tree exists."""
    if output_path.exists():
        shutil.rmtree(output_path)
    temp_path.rename(output_path)


def _physical_xy_from_metadata(metadata: dict[str, Any]) -> tuple[float, float] | None:
    physical = metadata.get("physical_pixel_size_um")
    if isinstance(physical, dict):
        x_um = physical.get("x")
        y_um = physical.get("y")
        if x_um is not None and y_um is not None:
            return float(x_um), float(y_um)
    canonical = metadata.get("canonical_metadata")
    if isinstance(canonical, dict):
        canonical_physical = canonical.get("physical_pixel_size_um")
        if isinstance(canonical_physical, dict):
            x_um = canonical_physical.get("x")
            y_um = canonical_physical.get("y")
            if x_um is not None and y_um is not None:
                return float(x_um), float(y_um)
    return None


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


def _normalize_pyramid_generation_policy(policy: str | None) -> str:
    normalized = str(policy or "downsample_streamed_s0").strip().lower().replace("-", "_")
    aliases = {
        "downsample_streamed_s0": "downsample_streamed_s0",
        "streamed_s0": "downsample_streamed_s0",
        "current_streamed_s0_downsample": "downsample_streamed_s0",
        "native_source_pyramid_crop": "native_source_pyramid_crop",
        "native": "native_source_pyramid_crop",
        "native_source": "native_source_pyramid_crop",
    }
    if normalized not in aliases:
        raise ValueError(
            "pyramid_generation_policy must be one of "
            "['downsample_streamed_s0', 'native_source_pyramid_crop']."
        )
    return aliases[normalized]


def _normalize_primary_rgb_mode(primary_rgb_mode: str | None) -> str:
    normalized = str(primary_rgb_mode or "masked_rgb").strip().lower().replace("-", "_")
    aliases = {
        "masked_rgb": "masked_rgb",
        "masked": "masked_rgb",
        "mask": "masked_rgb",
        "unmasked_rgb": "unmasked_rgb",
        "unmasked": "unmasked_rgb",
        "raw_rgb": "unmasked_rgb",
        "raw": "unmasked_rgb",
    }
    if normalized not in aliases:
        raise ValueError("primary_rgb_mode must be one of 'masked_rgb' or 'unmasked_rgb'.")
    return aliases[normalized]


def _resolve_primary_rgb_options(
    *,
    primary_rgb_mode: str | None,
    materialize_masked_rgb: bool | None,
    masked_rgb_fill_value: int | None,
    store_unmasked_rgb: bool | None,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    if bool(store_unmasked_rgb):
        raise NotImplementedError(
            "store_unmasked_rgb=True is not implemented in this pass; "
            "use primary_rgb_mode='unmasked_rgb' for debug/unmasked artifacts."
        )

    resolution_source = "primary_rgb_mode"
    legacy_used = False
    if primary_rgb_mode is None:
        if materialize_masked_rgb is not None:
            primary_rgb_mode = "masked_rgb" if bool(materialize_masked_rgb) else "unmasked_rgb"
            resolution_source = "materialize_masked_rgb"
            legacy_used = True
        elif defaults.get("primary_rgb_mode") is not None:
            primary_rgb_mode = str(defaults["primary_rgb_mode"])
            resolution_source = "profile_default"
        else:
            primary_rgb_mode = (
                "masked_rgb"
                if bool(defaults.get("materialize_masked_rgb", True))
                else "unmasked_rgb"
            )
            resolution_source = "profile_default_materialize_masked_rgb"
    resolved_mode = _normalize_primary_rgb_mode(primary_rgb_mode)
    fill_value = (
        int(masked_rgb_fill_value)
        if masked_rgb_fill_value is not None
        else int(defaults.get("masked_rgb_fill_value", 0))
    )
    if not 0 <= fill_value <= 255:
        raise ValueError("masked_rgb_fill_value must be between 0 and 255 for uint8 RGB output.")
    return {
        "primary_rgb_mode": resolved_mode,
        "materialize_masked_rgb": resolved_mode == "masked_rgb",
        "masked_rgb_fill_value": fill_value,
        "store_unmasked_rgb": False,
        "primary_rgb_mode_resolution_source": resolution_source,
        "materialize_masked_rgb_deprecated_alias_used": bool(legacy_used),
    }


def _normalize_output_profile(output_profile: str | None) -> str:
    normalized = str(output_profile or "validation").strip().lower().replace("-", "_")
    aliases = {
        "validation": "validation",
        "notebook": "validation",
        "notebook_square": "validation",
        "production": "production",
        "prod": "production",
        "upload": "upload_staging",
        "upload_staging": "upload_staging",
        "database": "upload_staging",
    }
    if normalized not in aliases:
        raise ValueError(
            "output_profile must be one of 'validation', 'production', or 'upload_staging'."
        )
    return aliases[normalized]


def _profile_defaults(output_profile: str) -> dict[str, Any]:
    profile = _normalize_output_profile(output_profile)
    if profile == "production":
        return {
            "crop_shape_policy": "compact_rectangle",
            "tile_frame_level": "segmentation",
            "extra_margin_px": 0,
            "compression": "lossless",
            "store_tissue_mask": True,
            "primary_rgb_mode": "masked_rgb",
            "masked_rgb_fill_value": 0,
            "store_unmasked_rgb": False,
            "materialize_masked_rgb": True,
            "sparse_zero_chunks": True,
            "pyramid_generation_policy": "native_source_pyramid_crop",
            "source_tile_aligned_canvas": True,
            "native_mip_stop_policy": "segmentation_level",
            "native_mip_stop_level": "segmentation_level",
        }
    if profile == "upload_staging":
        return {
            "crop_shape_policy": "compact_rectangle",
            "tile_frame_level": "segmentation",
            "extra_margin_px": 0,
            "compression": "none",
            "store_tissue_mask": True,
            "primary_rgb_mode": "masked_rgb",
            "masked_rgb_fill_value": 0,
            "store_unmasked_rgb": False,
            "materialize_masked_rgb": True,
            "sparse_zero_chunks": False,
            "pyramid_generation_policy": "native_source_pyramid_crop",
            "source_tile_aligned_canvas": True,
            "native_mip_stop_policy": "segmentation_level",
            "native_mip_stop_level": "segmentation_level",
        }
    return {
        "crop_shape_policy": "notebook_square",
        "tile_frame_level": "segmentation",
        "extra_margin_px": 0,
        "compression": "none",
        "store_tissue_mask": False,
        "primary_rgb_mode": "masked_rgb",
        "masked_rgb_fill_value": 0,
        "store_unmasked_rgb": False,
        "materialize_masked_rgb": True,
        "sparse_zero_chunks": False,
        "pyramid_generation_policy": "downsample_streamed_s0",
        "source_tile_aligned_canvas": False,
        "native_mip_stop_policy": "segmentation_level",
        "native_mip_stop_level": "segmentation_level",
    }


def _compression_descriptor(compression: str) -> dict[str, Any]:
    compression = _normalize_compression_mode(compression)
    if compression == "none":
        return {"mode": "none", "codec": None}
    return {
        "mode": "lossless",
        "codec": "blosc",
        "cname": "zstd",
        "clevel": 5,
        "shuffle": "bitshuffle",
    }


def _human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _pyramid_shapes_yxc(
    *,
    base_shape_yxc: tuple[int, int, int],
    num_mips: int,
) -> list[tuple[int, int, int]]:
    y, x, c = map(int, base_shape_yxc)
    return [(max(1, y >> level), max(1, x >> level), c) for level in range(int(num_mips))]


def _chunk_count_yx(*, y: int, x: int, chunk_xy: int) -> int:
    return int(math.ceil(int(y) / int(chunk_xy)) * math.ceil(int(x) / int(chunk_xy)))


def _estimate_mask_s0_chunk_activity(
    *,
    lr_labels: np.ndarray,
    label_id: int,
    label_crop_seg_yx: BoundsYX,
    source_shape_yx: tuple[int, int],
    canvas: BoundsYX,
    chunk_xy: int,
) -> dict[str, int]:
    empty = 0
    positive = 0
    for y0 in range(0, canvas.h, int(chunk_xy)):
        y1 = min(canvas.h, y0 + int(chunk_xy))
        for x0 in range(0, canvas.w, int(chunk_xy)):
            x1 = min(canvas.w, x0 + int(chunk_xy))
            mask = _project_native_mask_block(
                lr_labels=lr_labels,
                label_id=int(label_id),
                label_crop_seg_yx=label_crop_seg_yx,
                level_shape_yx=source_shape_yx,
                canvas=canvas,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
            )
            if np.any(mask):
                positive += 1
            else:
                empty += 1
    return {
        "mask_empty_chunks": int(empty),
        "mask_positive_chunks": int(positive),
        "rgb_chunks_skippable_before_decode": int(empty),
    }


def _estimate_pyramid_bytes(
    *,
    shapes_yxc: list[tuple[int, int, int]],
    bytes_per_pixel: int,
) -> int:
    return int(sum(y * x * c * int(bytes_per_pixel) for y, x, c in shapes_yxc))


def _estimate_warning(*, bytes_all_mips: int, s0_chunks: int) -> str | None:
    gib = bytes_all_mips / float(2**30)
    if gib >= 250.0:
        return "very_large_output_over_250_gib"
    if gib >= 100.0:
        return "large_output_over_100_gib"
    if s0_chunks >= 100_000:
        return "many_s0_chunks_over_100k"
    return None


_VALIDATED_SOURCE_LEVEL2_NATIVE_BASELINE: dict[str, Any] = {
    "source_level": 2,
    "segmentation_level": 7,
    "elapsed_s": 228.3,
    "disk_actual_bytes": 703 * 1024**2,
    "disk_apparent_bytes": 688 * 1024**2,
    "file_count": 5171,
    "config": {
        "primary_rgb_mode": "masked_rgb",
        "masked_rgb_fill_value": 0,
        "extra_margin_px": 0,
        "pyramid_generation_policy": "native_source_pyramid_crop",
        "source_tile_aligned_canvas": True,
        "native_mip_stop_policy": "segmentation_level",
        "native_mip_stop_level": "segmentation_level",
        "store_tissue_mask": True,
        "sparse_zero_chunks": True,
        "compression_mode": "lossless",
    },
}


@dataclass(frozen=True)
class NativeOutputLevelPlan:
    source_levels: tuple[int, ...]
    native_mip_stop_policy: str
    native_mip_stop_level: int
    mip_stop_reason: str
    coarsest_segmentation_level_not_written: bool
    warnings: tuple[str, ...] = ()


def _normalize_native_mip_stop_policy(
    native_mip_stop_policy: str | None,
    *,
    native_mip_stop_level: int | str | None = None,
) -> str:
    if native_mip_stop_policy is None:
        if isinstance(native_mip_stop_level, str):
            normalized_level = native_mip_stop_level.strip().lower().replace("-", "_")
            if normalized_level in {"none", "all", "available", "available_source_levels"}:
                return "available_source_levels"
            if normalized_level not in {"", "segmentation", "segmentation_level"}:
                return "explicit_level"
        return "segmentation_level"
    normalized = str(native_mip_stop_policy).strip().lower().replace("-", "_")
    aliases = {
        "segmentation": "segmentation_level",
        "segmentation_level": "segmentation_level",
        "min_side": "min_side_for_mips",
        "min_side_for_mips": "min_side_for_mips",
        "available": "available_source_levels",
        "available_source_levels": "available_source_levels",
        "all": "available_source_levels",
        "requested": "requested_mips",
        "requested_mips": "requested_mips",
        "explicit": "explicit_level",
        "explicit_level": "explicit_level",
        "native_mip_stop_level": "explicit_level",
    }
    if normalized not in aliases:
        raise ValueError(
            "native_mip_stop_policy must be one of "
            "'segmentation_level', 'min_side_for_mips', "
            "'available_source_levels', or 'requested_mips'."
        )
    return aliases[normalized]


def _explicit_native_mip_stop_level(
    native_mip_stop_level: int | str | None,
    *,
    segmentation_level: int,
) -> int | None:
    if native_mip_stop_level is None:
        return None
    if isinstance(native_mip_stop_level, str):
        normalized = native_mip_stop_level.strip().lower().replace("-", "_")
        if normalized in {"", "segmentation", "segmentation_level"}:
            return int(segmentation_level)
        if normalized in {"none", "all", "available", "available_source_levels"}:
            return None
        return int(normalized)
    return int(native_mip_stop_level)


def resolve_native_output_levels(
    source_level: int,
    segmentation_level: int,
    n_ets_levels: int,
    native_mip_stop_policy: str | None,
    native_mip_stop_level: int | str | None,
    min_side_for_mips: int | None = None,
    requested_mips: int | None = None,
    *,
    min_side_mip_count: int | None = None,
) -> NativeOutputLevelPlan:
    """Resolve native ETS levels used for an output OME-Zarr pyramid."""
    del min_side_for_mips  # The caller computes min_side_mip_count from the mapped FOV.
    source_idx = int(source_level)
    segmentation_idx = int(segmentation_level)
    nlevels = int(n_ets_levels)
    if nlevels <= 0:
        raise ValueError("n_ets_levels must be positive.")
    if source_idx < 0 or source_idx >= nlevels:
        raise ValueError(f"source_level {source_idx} is outside ETS levels 0..{nlevels - 1}.")
    if segmentation_idx < source_idx:
        raise ValueError(
            f"segmentation_level {segmentation_idx} must be >= source_level {source_idx}."
        )
    if segmentation_idx >= nlevels:
        raise ValueError(
            f"segmentation_level {segmentation_idx} is outside ETS levels 0..{nlevels - 1}."
        )

    policy = _normalize_native_mip_stop_policy(
        native_mip_stop_policy,
        native_mip_stop_level=native_mip_stop_level,
    )
    explicit_level = _explicit_native_mip_stop_level(
        native_mip_stop_level,
        segmentation_level=segmentation_idx,
    )
    available_stop_level = nlevels - 1
    warnings: list[str] = []

    if policy == "segmentation_level":
        final_source_level = segmentation_idx
        reason = "segmentation_level"
    elif policy == "min_side_for_mips":
        count = int(min_side_mip_count) if min_side_mip_count is not None else None
        if count is None:
            count = int(requested_mips) if requested_mips is not None else nlevels - source_idx
        final_source_level = source_idx + max(1, int(count)) - 1
        reason = "min_side_for_mips"
    elif policy == "requested_mips":
        count = int(requested_mips) if requested_mips is not None else nlevels - source_idx
        final_source_level = source_idx + max(1, int(count)) - 1
        reason = "requested_mips"
    elif policy == "available_source_levels":
        final_source_level = available_stop_level
        reason = "available_source_levels"
    else:
        if explicit_level is None:
            raise ValueError("native_mip_stop_policy='explicit_level' requires a stop level.")
        final_source_level = int(explicit_level)
        reason = "native_mip_stop_level"

    if policy not in {"segmentation_level", "explicit_level"} and explicit_level is not None:
        final_source_level = min(int(final_source_level), int(explicit_level))
        if final_source_level == int(explicit_level):
            reason = "native_mip_stop_level"

    final_source_level = max(source_idx, min(int(final_source_level), available_stop_level))
    coarsest_segmentation_level_not_written = final_source_level < segmentation_idx
    if coarsest_segmentation_level_not_written:
        warnings.append(
            "coarsest_segmentation_level_not_written:"
            f"final_source_level={final_source_level}:segmentation_level={segmentation_idx}"
        )

    return NativeOutputLevelPlan(
        source_levels=tuple(range(source_idx, final_source_level + 1)),
        native_mip_stop_policy=policy,
        native_mip_stop_level=int(final_source_level),
        mip_stop_reason=reason,
        coarsest_segmentation_level_not_written=bool(
            coarsest_segmentation_level_not_written
        ),
        warnings=tuple(warnings),
    )


def _normalize_native_mip_stop_level(
    native_mip_stop_level: int | str | None,
    *,
    segmentation_level: int,
) -> tuple[int | None, str]:
    """Backward-compatible normalizer retained for older call sites/tests."""
    if native_mip_stop_level is None:
        return int(segmentation_level), "segmentation_level"
    if isinstance(native_mip_stop_level, str):
        normalized = native_mip_stop_level.strip().lower().replace("-", "_")
        if normalized in {"", "segmentation", "segmentation_level"}:
            return int(segmentation_level), "segmentation_level"
        if normalized in {"none", "all", "available", "available_source_levels"}:
            return None, "available_source_levels"
        return int(normalized), "explicit_level"
    return int(native_mip_stop_level), "explicit_level"


def _native_mip_stop_reason(
    *,
    specs: list[_NativePyramidLevelSpec],
    requested_mips: int | None,
    native_mip_stop_level: int | None,
    native_mip_stop_source: str,
    source_level: int,
    available_mips: int,
    min_side_mip_count: int,
) -> str:
    if not specs:
        return "no_mips"
    num_mips = len(specs)
    if (
        native_mip_stop_source == "segmentation_level"
        and native_mip_stop_level is not None
        and specs[-1].source_level == int(native_mip_stop_level)
    ):
        return "segmentation_level"
    if native_mip_stop_level is not None and specs[-1].source_level == int(native_mip_stop_level):
        return "native_mip_stop_level"
    if requested_mips is not None and num_mips == max(1, int(requested_mips)):
        return "requested_mips"
    if num_mips == int(min_side_mip_count):
        return "min_side_for_mips"
    if num_mips == int(available_mips):
        return "available_source_levels"
    _ = source_level
    return "combined_limits"


def _bounds_expansion_yx(outer: BoundsYX, inner: BoundsYX) -> dict[str, int]:
    return {
        "top": int(inner.y0 - outer.y0),
        "bottom": int(outer.y1 - inner.y1),
        "left": int(inner.x0 - outer.x0),
        "right": int(outer.x1 - inner.x1),
    }


def _projection_from_validated_source_level2(
    *,
    estimate_totals: dict[str, Any],
    baseline_totals: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline = _VALIDATED_SOURCE_LEVEL2_NATIVE_BASELINE
    if not baseline_totals:
        return {
            "baseline": baseline,
            "method": "unavailable_no_baseline_estimate",
            "warnings": ["projection_baseline_estimate_unavailable"],
        }
    baseline_chunks = int(baseline_totals.get("combined_logical_chunks", 0))
    estimate_chunks = int(estimate_totals.get("combined_logical_chunks", 0))
    baseline_bytes = int(baseline_totals.get("total_uncompressed_bytes_rgb_plus_mask", 0))
    estimate_bytes = int(estimate_totals.get("total_uncompressed_bytes_rgb_plus_mask", 0))
    chunk_ratio = float(estimate_chunks / baseline_chunks) if baseline_chunks else None
    byte_ratio = float(estimate_bytes / baseline_bytes) if baseline_bytes else None
    runtime_ratio = chunk_ratio if chunk_ratio is not None else byte_ratio
    projected_elapsed = (
        float(baseline["elapsed_s"] * runtime_ratio) if runtime_ratio is not None else None
    )
    actual_disk = (
        int(round(int(baseline["disk_actual_bytes"]) * byte_ratio))
        if byte_ratio is not None
        else None
    )
    apparent_disk = (
        int(round(int(baseline["disk_apparent_bytes"]) * byte_ratio))
        if byte_ratio is not None
        else None
    )
    file_count = (
        int(math.ceil(int(baseline["file_count"]) * chunk_ratio))
        if chunk_ratio is not None
        else None
    )
    return {
        "baseline": baseline,
        "method": "validated_source_level2_native_masked_ratio",
        "logical_chunk_ratio_to_baseline": chunk_ratio,
        "logical_byte_ratio_to_baseline": byte_ratio,
        "projected_elapsed_s": projected_elapsed,
        "projected_elapsed_min": (
            projected_elapsed / 60.0 if projected_elapsed is not None else None
        ),
        "projected_disk_actual_bytes": actual_disk,
        "projected_disk_actual_size": (
            _human_bytes(actual_disk) if actual_disk is not None else None
        ),
        "projected_disk_apparent_bytes": apparent_disk,
        "projected_disk_apparent_size": (
            _human_bytes(apparent_disk) if apparent_disk is not None else None
        ),
        "projected_file_count": file_count,
    }


def _baseline_config_mismatch_warnings(
    *,
    source_level: int,
    segmentation_level: int,
    primary_rgb_mode: str,
    masked_rgb_fill_value: int,
    extra_margin_px: int,
    pyramid_generation_policy: str,
    source_tile_aligned_canvas: bool,
    native_mip_stop_policy: str,
    store_tissue_mask: bool,
    sparse_zero_chunks: bool,
    compression: str,
) -> list[str]:
    baseline = _VALIDATED_SOURCE_LEVEL2_NATIVE_BASELINE
    expected = baseline["config"]
    warnings: list[str] = []
    checks = {
        "segmentation_level": (int(segmentation_level), int(baseline["segmentation_level"])),
        "primary_rgb_mode": (primary_rgb_mode, expected["primary_rgb_mode"]),
        "masked_rgb_fill_value": (int(masked_rgb_fill_value), expected["masked_rgb_fill_value"]),
        "extra_margin_px": (int(extra_margin_px), expected["extra_margin_px"]),
        "pyramid_generation_policy": (
            pyramid_generation_policy,
            expected["pyramid_generation_policy"],
        ),
        "source_tile_aligned_canvas": (
            bool(source_tile_aligned_canvas),
            expected["source_tile_aligned_canvas"],
        ),
        "native_mip_stop_policy": (
            native_mip_stop_policy,
            expected["native_mip_stop_policy"],
        ),
        "store_tissue_mask": (bool(store_tissue_mask), expected["store_tissue_mask"]),
        "sparse_zero_chunks": (bool(sparse_zero_chunks), expected["sparse_zero_chunks"]),
        "compression_mode": (
            _normalize_compression_mode(compression),
            expected["compression_mode"],
        ),
    }
    for key, (actual, want) in checks.items():
        if actual != want:
            warnings.append(f"baseline_config_mismatch:{key}:actual={actual}:baseline={want}")
    if int(source_level) == int(baseline["source_level"]):
        warnings.append("estimate_source_level_matches_projection_baseline")
    return warnings


def _resolve_ets_level(
    level: int | str | None,
    *,
    default_index: int,
    nlevels: int,
    label: str,
) -> int:
    """Resolve an ETS pyramid level from an integer index or an sN dataset name."""
    if level is None:
        idx = default_index
    elif isinstance(level, str):
        normalized = level.strip()
        if normalized.startswith("s") and normalized[1:].isdigit():
            idx = int(normalized[1:])
        else:
            idx = int(normalized)
    else:
        idx = int(level)
    if idx < 0:
        idx = nlevels + idx
    if idx < 0 or idx >= nlevels:
        raise ValueError(f"{label} level {level!r} is out of range for {nlevels} ETS levels.")
    return idx


def _filled_lr_labels(low_res_filled: np.ndarray) -> np.ndarray:
    """Return connected-component labels after filling each low-res tissue island."""
    lr_lbl, n_lr = measure.label(low_res_filled.astype(bool), connectivity=2, return_num=True)
    filled_lr_lbl = np.zeros_like(lr_lbl, dtype=np.int32)
    for lid in range(1, n_lr + 1):
        comp = lr_lbl == lid
        if comp.any():
            filled_lr_lbl[binary_fill_holes(comp)] = lid
    return filled_lr_lbl


def _read_ets_region_yxc(
    ets_path: str | Path,
    *,
    level: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> np.ndarray:
    """Read an arbitrary ETS level region as a YXC uint8 array."""
    x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    with ETSFile(ets_path) as ets:
        height, width = map(int, ets.level_shape(level))
        rx0 = max(0, min(width, x0))
        rx1 = max(0, min(width, x1))
        ry0 = max(0, min(height, y0))
        ry1 = max(0, min(height, y1))
        out = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        if rx1 <= rx0 or ry1 <= ry0:
            return out

        tx = int(ets.tile_xsize)
        ty = int(ets.tile_ysize)
        col0 = rx0 // tx
        col1 = (rx1 - 1) // tx
        row0 = ry0 // ty
        row1 = (ry1 - 1) // ty

        for tile_row in range(row0, row1 + 1):
            tile_y0 = tile_row * ty
            tile_y1 = tile_y0 + ty
            iy0 = max(ry0, tile_y0)
            iy1 = min(ry1, tile_y1)
            if iy1 <= iy0:
                continue
            for tile_col in range(col0, col1 + 1):
                tile_x0 = tile_col * tx
                tile_x1 = tile_x0 + tx
                ix0 = max(rx0, tile_x0)
                ix1 = min(rx1, tile_x1)
                if ix1 <= ix0:
                    continue
                tile = ets.get_tile_decoded(level, tile_col, tile_row)
                src = tile[iy0 - tile_y0 : iy1 - tile_y0, ix0 - tile_x0 : ix1 - tile_x0, :]
                out[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0, :] = src
        return out


@dataclass(frozen=True)
class _NativePyramidLevelSpec:
    output_index: int
    source_level: int
    source_shape_yx: tuple[int, int]
    canonical_canvas_source_yx: BoundsYX
    canvas_source_yx: BoundsYX
    clipped_source_yx: BoundsYX
    source_read_envelope_yx: BoundsYX
    clipped_source_read_envelope_yx: BoundsYX
    output_shape_yx: tuple[int, int]
    phys_xy_um: tuple[float, float]
    translation_yx_um: tuple[float, float]
    scale_from_parent_yx: tuple[float, float]


def _bounds_from_frame_debug(frame_debug: dict[str, Any] | None, key: str) -> BoundsYX | None:
    if not isinstance(frame_debug, dict):
        return None
    value = frame_debug.get(key)
    if not isinstance(value, dict):
        return None
    try:
        return BoundsYX(
            y0=int(value["y0"]),
            x0=int(value["x0"]),
            y1=int(value["y1"]),
            x1=int(value["x1"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _record_logical_canvas_source_yx(record: Any) -> BoundsYX:
    direct_canvas = getattr(record, "logical_canvas_source_yx", None)
    if isinstance(direct_canvas, BoundsYX):
        return direct_canvas
    canvas = _bounds_from_frame_debug(
        getattr(record, "frame_debug", None), "logical_canvas_source_yx"
    )
    if canvas is not None:
        return canvas
    x0, y0, x1, y1 = map(int, record.crop_bounds_source_level)
    return BoundsYX(y0=y0, x0=x0, y1=y1, x1=x1)


def _record_label_crop_seg_yx(record: Any) -> BoundsYX:
    direct_label_crop = getattr(record, "label_crop_seg_yx", None)
    if isinstance(direct_label_crop, BoundsYX):
        return direct_label_crop
    label_crop = _bounds_from_frame_debug(getattr(record, "frame_debug", None), "label_crop_seg_yx")
    if label_crop is not None:
        return label_crop
    x0, y0, x1, y1 = map(int, record.crop_bounds_segmentation_level)
    return BoundsYX(y0=y0, x0=x0, y1=y1, x1=x1)


def _map_parent_bounds_to_level(
    bounds: BoundsYX,
    *,
    parent_shape_yx: tuple[int, int],
    level_shape_yx: tuple[int, int],
) -> BoundsYX:
    parent_h, parent_w = map(int, parent_shape_yx)
    level_h, level_w = map(int, level_shape_yx)
    scale_y = parent_h / max(1, level_h)
    scale_x = parent_w / max(1, level_w)
    return BoundsYX(
        y0=int(math.floor(bounds.y0 / scale_y)),
        x0=int(math.floor(bounds.x0 / scale_x)),
        y1=int(math.ceil(bounds.y1 / scale_y)),
        x1=int(math.ceil(bounds.x1 / scale_x)),
    )


def _align_bounds_outward_to_grid(bounds: BoundsYX, *, tile_size_yx: tuple[int, int]) -> BoundsYX:
    tile_h, tile_w = map(int, tile_size_yx)
    return BoundsYX(
        y0=math.floor(bounds.y0 / tile_h) * tile_h,
        x0=math.floor(bounds.x0 / tile_w) * tile_w,
        y1=math.ceil(bounds.y1 / tile_h) * tile_h,
        x1=math.ceil(bounds.x1 / tile_w) * tile_w,
    )


def _canonical_canvas_in_source_level_coordinates(
    record: TissueTileRecord,
    *,
    source_tile_aligned_canvas: bool,
    source_tile_size_yx: tuple[int, int],
) -> BoundsYX:
    """Return the single half-open source-level FOV used by all native mips."""
    canvas = _record_logical_canvas_source_yx(record)
    if not source_tile_aligned_canvas:
        return canvas
    return _align_bounds_outward_to_grid(canvas, tile_size_yx=source_tile_size_yx)


def _native_pyramid_level_specs(
    *,
    record: Any,
    source_level: int,
    source_shape_yx: tuple[int, int],
    ets_level_shapes_yx: list[tuple[int, int]],
    source_phys_xy_um: tuple[float, float],
    block_xy: int,
    min_side_for_mips: int | None,
    source_tile_aligned_canvas: bool,
    source_tile_size_yx: tuple[int, int],
    requested_mips: int | None = None,
    segmentation_level: int | None = None,
    native_mip_stop_policy: str | None = None,
    native_mip_stop_level: int | str | None = None,
) -> list[_NativePyramidLevelSpec]:
    canonical_canvas = _canonical_canvas_in_source_level_coordinates(
        record,
        source_tile_aligned_canvas=source_tile_aligned_canvas,
        source_tile_size_yx=source_tile_size_yx,
    )
    min_side_mip_count = compute_num_mips_min_side(
        canonical_canvas.w,
        canonical_canvas.h,
        int(min_side_for_mips or block_xy),
    )
    policy = native_mip_stop_policy
    if policy is None:
        policy = (
            "segmentation_level"
            if segmentation_level is not None
            else ("requested_mips" if requested_mips is not None else "min_side_for_mips")
        )
    explicit_stop_level = _explicit_native_mip_stop_level(
        native_mip_stop_level,
        segmentation_level=len(ets_level_shapes_yx) - 1,
    )
    effective_segmentation_level = (
        int(segmentation_level)
        if segmentation_level is not None
        else (
            int(explicit_stop_level)
            if explicit_stop_level is not None
            else len(ets_level_shapes_yx) - 1
        )
    )
    level_plan = resolve_native_output_levels(
        int(source_level),
        int(effective_segmentation_level),
        len(ets_level_shapes_yx),
        policy,
        native_mip_stop_level,
        min_side_for_mips=min_side_for_mips,
        requested_mips=requested_mips,
        min_side_mip_count=min_side_mip_count,
    )
    source_h, source_w = map(int, source_shape_yx)
    source_px, source_py = map(float, source_phys_xy_um)
    specs: list[_NativePyramidLevelSpec] = []
    for output_index, ets_level in enumerate(level_plan.source_levels):
        ets_level = int(ets_level)
        level_shape = tuple(map(int, ets_level_shapes_yx[ets_level]))
        level_h, level_w = level_shape
        level_canvas = _map_parent_bounds_to_level(
            canonical_canvas,
            parent_shape_yx=source_shape_yx,
            level_shape_yx=level_shape,
        )
        source_read_envelope = _align_bounds_outward_to_grid(
            level_canvas,
            tile_size_yx=source_tile_size_yx,
        )
        clipped = level_canvas.clip(level_shape)
        clipped_read_envelope = source_read_envelope.clip(level_shape)
        scale_y = source_h / max(1, level_h)
        scale_x = source_w / max(1, level_w)
        phys_x = source_px * scale_x
        phys_y = source_py * scale_y
        specs.append(
            _NativePyramidLevelSpec(
                output_index=output_index,
                source_level=ets_level,
                source_shape_yx=level_shape,
                canonical_canvas_source_yx=canonical_canvas,
                canvas_source_yx=level_canvas,
                clipped_source_yx=clipped,
                source_read_envelope_yx=source_read_envelope,
                clipped_source_read_envelope_yx=clipped_read_envelope,
                output_shape_yx=(max(1, level_canvas.h), max(1, level_canvas.w)),
                phys_xy_um=(float(phys_x), float(phys_y)),
                translation_yx_um=(
                    float(level_canvas.y0 * phys_y),
                    float(level_canvas.x0 * phys_x),
                ),
                scale_from_parent_yx=(float(scale_y), float(scale_x)),
            )
        )
    return specs


def _parent_bounds_from_ngff_transform_source_yx(
    dataset: dict[str, Any],
    *,
    shape_yx: tuple[int, int],
    source_phys_xy_um: tuple[float, float],
) -> dict[str, float]:
    """Recover source-level parent bounds from NGFF scale/translation metadata."""
    transforms = dataset.get("coordinateTransformations")
    if not isinstance(transforms, list):
        raise ValueError("NGFF dataset is missing coordinateTransformations")
    scale_values: list[float] | None = None
    translation_values: list[float] | None = None
    for transform in transforms:
        if not isinstance(transform, dict):
            continue
        if transform.get("type") == "scale":
            scale_values = [float(value) for value in transform.get("scale", [])]
        elif transform.get("type") == "translation":
            translation_values = [float(value) for value in transform.get("translation", [])]
    if not scale_values:
        raise ValueError("NGFF dataset is missing a scale transform")
    if translation_values is None:
        translation_values = [0.0] * len(scale_values)
    if len(scale_values) == 3:
        scale_y_um = float(scale_values[1])
        scale_x_um = float(scale_values[2])
        translation_y_um = float(translation_values[1])
        translation_x_um = float(translation_values[2])
    elif len(scale_values) == 2:
        scale_y_um = float(scale_values[0])
        scale_x_um = float(scale_values[1])
        translation_y_um = float(translation_values[0])
        translation_x_um = float(translation_values[1])
    else:
        raise ValueError("NGFF scale transform must have 2 or 3 entries")
    source_px_um, source_py_um = map(float, source_phys_xy_um)
    h, w = map(int, shape_yx)
    y0 = translation_y_um / source_py_um
    x0 = translation_x_um / source_px_um
    return {
        "y0": float(y0),
        "x0": float(x0),
        "y1": float((translation_y_um + h * scale_y_um) / source_py_um),
        "x1": float((translation_x_um + w * scale_x_um) / source_px_um),
    }


def _read_ets_region_yxc_open(
    ets: ETSFile,
    *,
    level: int,
    canvas: BoundsYX,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    stats: dict[str, Any],
) -> np.ndarray:
    """Read an output-space region from one ETS level into a padded Y/X/C block."""
    out = np.zeros((int(y1 - y0), int(x1 - x0), 3), dtype=np.uint8)
    source_y0 = int(canvas.y0 + y0)
    source_y1 = int(canvas.y0 + y1)
    source_x0 = int(canvas.x0 + x0)
    source_x1 = int(canvas.x0 + x1)
    level_h, level_w = map(int, ets.level_shape(level))
    valid_y0 = max(0, source_y0)
    valid_y1 = min(level_h, source_y1)
    valid_x0 = max(0, source_x0)
    valid_x1 = min(level_w, source_x1)
    if valid_y1 <= valid_y0 or valid_x1 <= valid_x0:
        return out

    tile_h = int(ets.tile_ysize)
    tile_w = int(ets.tile_xsize)
    row0 = valid_y0 // tile_h
    row1 = (valid_y1 - 1) // tile_h
    col0 = valid_x0 // tile_w
    col1 = (valid_x1 - 1) // tile_w
    stats["source_tile_decode_calls"] = int(stats.get("source_tile_decode_calls", 0))
    unique = stats.setdefault("unique_source_tiles_touched", set())
    for tile_row in range(row0, row1 + 1):
        tile_y0 = tile_row * tile_h
        tile_y1 = tile_y0 + tile_h
        iy0 = max(valid_y0, tile_y0)
        iy1 = min(valid_y1, tile_y1)
        if iy1 <= iy0:
            continue
        for tile_col in range(col0, col1 + 1):
            tile_x0 = tile_col * tile_w
            tile_x1 = tile_x0 + tile_w
            ix0 = max(valid_x0, tile_x0)
            ix1 = min(valid_x1, tile_x1)
            if ix1 <= ix0:
                continue
            tile = ets.get_tile_decoded(level, tile_col, tile_row)
            stats["source_tile_decode_calls"] += 1
            unique.add((int(level), int(tile_col), int(tile_row)))
            src = tile[iy0 - tile_y0 : iy1 - tile_y0, ix0 - tile_x0 : ix1 - tile_x0, :]
            dst_y0 = iy0 - source_y0
            dst_x0 = ix0 - source_x0
            out[dst_y0 : dst_y0 + src.shape[0], dst_x0 : dst_x0 + src.shape[1], :] = src
    return out


def _project_native_mask_block(
    *,
    lr_labels: np.ndarray,
    label_id: int,
    label_crop_seg_yx: BoundsYX,
    level_shape_yx: tuple[int, int],
    canvas: BoundsYX,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    out = np.zeros((int(y1 - y0), int(x1 - x0)), dtype=np.uint8)
    source_y0 = int(canvas.y0 + y0)
    source_y1 = int(canvas.y0 + y1)
    source_x0 = int(canvas.x0 + x0)
    source_x1 = int(canvas.x0 + x1)
    level_h, level_w = map(int, level_shape_yx)
    valid = BoundsYX(
        y0=max(0, source_y0),
        x0=max(0, source_x0),
        y1=min(level_h, source_y1),
        x1=min(level_w, source_x1),
    )
    if valid.h <= 0 or valid.w <= 0:
        return out
    scale_y = level_h / lr_labels.shape[0]
    scale_x = level_w / lr_labels.shape[1]
    projected = project_label_mask_to_source_region(
        lr_labels,
        label_id=int(label_id),
        source_region_yx=valid,
        label_crop_seg_yx=label_crop_seg_yx.clip(lr_labels.shape),
        scale_y=scale_y,
        scale_x=scale_x,
    )
    dst_y0 = valid.y0 - source_y0
    dst_x0 = valid.x0 - source_x0
    out[dst_y0 : dst_y0 + projected.shape[0], dst_x0 : dst_x0 + projected.shape[1]] = (
        projected.astype(np.uint8)
    )
    return out


def _write_native_group_metadata(
    root: Any,
    label_group: Any | None,
    *,
    name: str,
    specs: list[_NativePyramidLevelSpec],
    channel_count: int,
    channel_labels: list[str],
    channel_colors: list[str],
    metadata_schema: str,
    output_scale_to_source_level: dict[str, int],
    source_tile_aligned_canvas: bool,
    primary_rgb_mode: str,
    masked_rgb_fill_value: int,
    masked_rgb_pyramid_semantics: str,
) -> None:
    version = "0.4" if metadata_schema in {"v0.4", "0.4", None} else str(metadata_schema)
    datasets = []
    for spec in specs:
        scale = [1.0, float(spec.phys_xy_um[1]), float(spec.phys_xy_um[0])]
        translation = [0.0, *map(float, spec.translation_yx_um)]
        datasets.append(
            {
                "path": f"s{spec.output_index}",
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale},
                    {"type": "translation", "translation": translation},
                ],
            }
        )
    root.attrs["multiscales"] = [
        {
            "version": version,
            "name": name,
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets,
        }
    ]
    root.attrs["omero"] = {
        "name": name,
        "version": version,
        "channels": [
            {"label": label, "color": color, "active": True}
            for label, color in zip(channel_labels, channel_colors, strict=True)
        ],
    }
    root.attrs["native_source_pyramid"] = {
        "pyramid_generation_policy": "native_source_pyramid_crop",
        "rgb_pyramid_semantics": "native_scanner_pyramid",
        "reference_policy": "downsample_streamed_s0",
        "source_tile_aligned_canvas": bool(source_tile_aligned_canvas),
        "output_scale_to_source_level": output_scale_to_source_level,
        "channel_count": int(channel_count),
        "primary_rgb_mode": primary_rgb_mode,
        "masked_rgb_fill_value": int(masked_rgb_fill_value),
        "mask_applied_to_primary_rgb": primary_rgb_mode == "masked_rgb",
        "masked_rgb_pyramid_semantics": masked_rgb_pyramid_semantics,
        "store_unmasked_rgb": False,
    }
    root.attrs["primary_rgb_mode"] = primary_rgb_mode
    root.attrs["masked_rgb_fill_value"] = int(masked_rgb_fill_value)
    root.attrs["mask_applied_to_primary_rgb"] = primary_rgb_mode == "masked_rgb"
    root.attrs["store_unmasked_rgb"] = False
    if label_group is not None:
        label_datasets = []
        for spec in specs:
            label_datasets.append(
                {
                    "path": f"s{spec.output_index}",
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [float(spec.phys_xy_um[1]), float(spec.phys_xy_um[0])],
                        },
                        {
                            "type": "translation",
                            "translation": list(map(float, spec.translation_yx_um)),
                        },
                    ],
                }
            )
        label_group.attrs["multiscales"] = [
            {
                "version": version,
                "name": "tissue_mask",
                "axes": [
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": label_datasets,
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
        label_group.attrs["mask_generation_policy"] = "project_segmentation_per_scale"
        label_group.attrs["mask_pyramid_semantics"] = "label_safe_nearest"


def write_native_ets_tissue_pyramid_ome(
    *,
    ets_path: str | Path,
    out_dir: str | Path,
    record: TissueTileRecord,
    lr_labels: np.ndarray,
    source_level: int,
    source_shape_yx: tuple[int, int],
    source_phys_xy_um: tuple[float, float],
    block_xy: int,
    name: str,
    compressor=None,
    sparse_zero_chunks: bool = True,
    store_tissue_mask: bool = True,
    metadata_schema: str = "v0.4",
    min_side_for_mips: int | None = None,
    requested_mips: int | None = None,
    segmentation_level: int | None = None,
    native_mip_stop_policy: str | None = None,
    native_mip_stop_level: int | str | None = None,
    native_mip_stop_source: str = "available_source_levels",
    max_chunks_per_level: int | None = None,
    source_tile_aligned_canvas: bool = False,
    primary_rgb_mode: str = "unmasked_rgb",
    masked_rgb_fill_value: int = 0,
    store_unmasked_rgb: bool = False,
    channel_labels: list[str] | None = None,
    channel_colors: list[str] | None = None,
    run_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one tissue OME-Zarr by cropping each output scale from native ETS levels."""
    primary_rgb_mode = _normalize_primary_rgb_mode(primary_rgb_mode)
    if store_unmasked_rgb:
        raise NotImplementedError(
            "store_unmasked_rgb=True is not implemented in this pass; "
            "use primary_rgb_mode='unmasked_rgb' for debug/unmasked artifacts."
        )
    masked_rgb_fill_value = int(masked_rgb_fill_value)
    if not 0 <= masked_rgb_fill_value <= 255:
        raise ValueError("masked_rgb_fill_value must be between 0 and 255.")
    mask_applied_to_primary_rgb = primary_rgb_mode == "masked_rgb"
    masked_rgb_pyramid_semantics = (
        "mask_projected_per_scale" if mask_applied_to_primary_rgb else "not_applicable"
    )
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_limit = max(0, int(max_chunks_per_level)) if max_chunks_per_level is not None else None
    channel_count = 3
    channel_labels = channel_labels or default_channel_labels(channel_count)
    channel_colors = channel_colors or default_channel_colors(channel_count)
    stats: dict[str, Any] = {
        "pyramid_generation_policy": "native_source_pyramid_crop",
        "rgb_pyramid_semantics": "native_scanner_pyramid",
        "reference_policy": "downsample_streamed_s0",
        "source_tile_aligned_canvas": bool(source_tile_aligned_canvas),
        "native_mip_stop_policy": (
            native_mip_stop_policy
            if native_mip_stop_policy is not None
            else native_mip_stop_source
        ),
        "primary_rgb_mode": primary_rgb_mode,
        "masked_rgb_fill_value": int(masked_rgb_fill_value),
        "mask_applied_to_primary_rgb": bool(mask_applied_to_primary_rgb),
        "store_tissue_mask": bool(store_tissue_mask),
        "store_unmasked_rgb": False,
        "masked_rgb_pyramid_semantics": masked_rgb_pyramid_semantics,
        "mask_generation_policy": "project_segmentation_per_scale",
        "mask_pyramid_semantics": "label_safe_nearest",
        "rgb_chunk_write_calls": 0,
        "rgb_chunks_written": 0,
        "rgb_chunks_skipped": 0,
        "rgb_chunks_skipped_before_decode": 0,
        "mask_chunk_write_calls": 0,
        "mask_chunks_written": 0,
        "mask_chunks_skipped": 0,
        "mask_empty_chunks": 0,
        "mask_positive_chunks": 0,
        "source_tile_decode_calls": 0,
        "max_chunks_per_level": chunk_limit,
    }
    unique_rgb_chunks: set[tuple[int, int, int]] = set()
    unique_mask_chunks: set[tuple[int, int, int]] = set()

    with ETSFile(ets_path) as ets:
        ets_level_shapes = [
            tuple(map(int, ets.level_shape(idx))) for idx in range(int(ets.nlevels))
        ]
        source_tile_size_yx = (int(ets.tile_ysize), int(ets.tile_xsize))
        specs = _native_pyramid_level_specs(
            record=record,
            source_level=int(source_level),
            source_shape_yx=source_shape_yx,
            ets_level_shapes_yx=ets_level_shapes,
            source_phys_xy_um=source_phys_xy_um,
            block_xy=int(block_xy),
            min_side_for_mips=min_side_for_mips,
            requested_mips=requested_mips,
            segmentation_level=segmentation_level,
            native_mip_stop_policy=native_mip_stop_policy,
            native_mip_stop_level=native_mip_stop_level,
            source_tile_aligned_canvas=bool(source_tile_aligned_canvas),
            source_tile_size_yx=source_tile_size_yx,
        )
        min_side_mip_count = compute_num_mips_min_side(
            specs[0].canonical_canvas_source_yx.w,
            specs[0].canonical_canvas_source_yx.h,
            int(min_side_for_mips or block_xy),
        )
        explicit_stop_level = _explicit_native_mip_stop_level(
            native_mip_stop_level,
            segmentation_level=int(specs[-1].source_level),
        )
        effective_segmentation_level = (
            int(segmentation_level)
            if segmentation_level is not None
            else (
                int(explicit_stop_level)
                if explicit_stop_level is not None
                else int(specs[-1].source_level)
            )
        )
        level_plan = resolve_native_output_levels(
            int(source_level),
            int(effective_segmentation_level),
            len(ets_level_shapes),
            native_mip_stop_policy
            if native_mip_stop_policy is not None
            else (
                "segmentation_level"
                if segmentation_level is not None
                else ("requested_mips" if requested_mips is not None else "min_side_for_mips")
            ),
            native_mip_stop_level,
            min_side_for_mips=min_side_for_mips,
            requested_mips=requested_mips,
            min_side_mip_count=min_side_mip_count,
        )
        root = open_group_v2(str(out_dir), mode="w")
        rgb_arrays = []
        mask_arrays = []
        labels_group = None
        mask_group = None
        for spec in specs:
            out_h, out_w = map(int, spec.output_shape_yx)
            arr = create_group_array(
                root,
                f"s{spec.output_index}",
                shape=(channel_count, out_h, out_w),
                chunks=(channel_count, min(block_xy, out_h), min(block_xy, out_w)),
                dtype="uint8",
                compressor=compressor,
                fill_value=masked_rgb_fill_value if mask_applied_to_primary_rgb else 0,
                overwrite=True,
                zarr_format=2,
            )
            arr.attrs["_ARRAY_DIMENSIONS"] = ["c", "y", "x"]
            rgb_arrays.append(arr)
        if store_tissue_mask:
            labels_group = root.create_group("labels", overwrite=True)
            labels_group.attrs["labels"] = ["tissue_mask"]
            mask_group = labels_group.create_group("tissue_mask", overwrite=True)
            for spec in specs:
                out_h, out_w = map(int, spec.output_shape_yx)
                arr = create_group_array(
                    mask_group,
                    f"s{spec.output_index}",
                    shape=(out_h, out_w),
                    chunks=(min(block_xy, out_h), min(block_xy, out_w)),
                    dtype="uint8",
                    compressor=compressor,
                    fill_value=masked_rgb_fill_value if primary_rgb_mode == "masked_rgb" else 0,
                    overwrite=True,
                    zarr_format=2,
                )
                arr.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
                mask_arrays.append(arr)

        label_crop = _record_label_crop_seg_yx(record)
        per_scale: list[dict[str, Any]] = []
        for spec, rgb_arr in zip(specs, rgb_arrays, strict=True):
            out_h, out_w = map(int, spec.output_shape_yx)
            level_expected_full = max(1, math.ceil(out_h / block_xy)) * max(
                1, math.ceil(out_w / block_xy)
            )
            level_expected = (
                min(level_expected_full, chunk_limit)
                if chunk_limit is not None
                else level_expected_full
            )
            stats["rgb_chunks_expected"] = int(stats.get("rgb_chunks_expected", 0)) + int(
                level_expected
            )
            if store_tissue_mask:
                stats["mask_chunks_expected"] = int(stats.get("mask_chunks_expected", 0)) + int(
                    level_expected
                )
            level_processed = 0
            for y0 in range(0, out_h, block_xy):
                if chunk_limit is not None and level_processed >= chunk_limit:
                    break
                y1 = min(out_h, y0 + block_xy)
                chunk_y = y0 // block_xy
                for x0 in range(0, out_w, block_xy):
                    if chunk_limit is not None and level_processed >= chunk_limit:
                        break
                    level_processed += 1
                    x1 = min(out_w, x0 + block_xy)
                    chunk_x = x0 // block_xy
                    mask = None
                    mask_has_pixels = False
                    if mask_applied_to_primary_rgb or store_tissue_mask:
                        mask = _project_native_mask_block(
                            lr_labels=lr_labels,
                            label_id=int(record.label_id),
                            label_crop_seg_yx=label_crop,
                            level_shape_yx=spec.source_shape_yx,
                            canvas=spec.canvas_source_yx,
                            y0=y0,
                            y1=y1,
                            x0=x0,
                            x1=x1,
                        )
                        mask_has_pixels = bool(np.any(mask))
                        if mask_has_pixels:
                            stats["mask_positive_chunks"] += 1
                        else:
                            stats["mask_empty_chunks"] += 1

                    if (
                        mask_applied_to_primary_rgb
                        and masked_rgb_fill_value == 0
                        and mask is not None
                        and not mask_has_pixels
                    ):
                        stats["rgb_chunks_skipped_before_decode"] += 1
                        stats["rgb_chunks_skipped"] += 1
                    else:
                        rgb = _read_ets_region_yxc_open(
                            ets,
                            level=spec.source_level,
                            canvas=spec.canvas_source_yx,
                            y0=y0,
                            y1=y1,
                            x0=x0,
                            x1=x1,
                            stats=stats,
                        )
                        if mask_applied_to_primary_rgb and mask is not None:
                            rgb = np.where(mask[..., None].astype(bool), rgb, masked_rgb_fill_value)
                        if sparse_zero_chunks and not np.any(rgb):
                            stats["rgb_chunks_skipped"] += 1
                        else:
                            rgb_arr[:, y0:y1, x0:x1] = np.moveaxis(rgb, -1, 0)
                            stats["rgb_chunk_write_calls"] += 1
                            stats["rgb_chunks_written"] += 1
                            unique_rgb_chunks.add((spec.output_index, chunk_y, chunk_x))
                    if store_tissue_mask and mask is not None:
                        if sparse_zero_chunks and not np.any(mask):
                            stats["mask_chunks_skipped"] += 1
                        else:
                            mask_arrays[spec.output_index][y0:y1, x0:x1] = mask
                            stats["mask_chunk_write_calls"] += 1
                            stats["mask_chunks_written"] += 1
                            unique_mask_chunks.add((spec.output_index, chunk_y, chunk_x))
            per_scale.append(
                {
                    "path": f"s{spec.output_index}",
                    "output_index": int(spec.output_index),
                    "source_level": int(spec.source_level),
                    "source_shape_yx": list(map(int, spec.source_shape_yx)),
                    "canonical_canvas_in_source_level_coordinates": (
                        spec.canonical_canvas_source_yx.as_dict()
                    ),
                    "parent_space_canvas_source_yx": spec.canonical_canvas_source_yx.as_dict(),
                    "output_canvas_source_yx": spec.canvas_source_yx.as_dict(),
                    "native_source_level_canvas_yx": spec.canvas_source_yx.as_dict(),
                    "native_source_level_clipped_yx": spec.clipped_source_yx.as_dict(),
                    "source_read_envelope_yx": spec.source_read_envelope_yx.as_dict(),
                    "source_read_envelope_clipped_yx": (
                        spec.clipped_source_read_envelope_yx.as_dict()
                    ),
                    "output_shape_yx": list(map(int, spec.output_shape_yx)),
                    "phys_xy_um": {"x": spec.phys_xy_um[0], "y": spec.phys_xy_um[1]},
                    "translation_yx_um": list(map(float, spec.translation_yx_um)),
                    "scale_from_parent_yx": list(map(float, spec.scale_from_parent_yx)),
                }
            )

    output_scale_to_source_level = {
        f"s{spec.output_index}": int(spec.source_level) for spec in specs
    }
    _write_native_group_metadata(
        root,
        mask_group,
        name=name,
        specs=specs,
        channel_count=channel_count,
        channel_labels=channel_labels,
        channel_colors=channel_colors,
        metadata_schema=metadata_schema,
        output_scale_to_source_level=output_scale_to_source_level,
        source_tile_aligned_canvas=source_tile_aligned_canvas,
        primary_rgb_mode=primary_rgb_mode,
        masked_rgb_fill_value=masked_rgb_fill_value,
        masked_rgb_pyramid_semantics=masked_rgb_pyramid_semantics,
    )
    native_metadata = dict(root.attrs.get("native_source_pyramid", {}))
    native_metadata["levels"] = per_scale
    native_metadata["requested_mips"] = int(requested_mips) if requested_mips is not None else None
    native_metadata["native_mip_stop_policy"] = level_plan.native_mip_stop_policy
    native_metadata["native_mip_stop_level"] = int(level_plan.native_mip_stop_level)
    native_metadata["native_mip_stop_level_source"] = level_plan.native_mip_stop_policy
    native_metadata["mip_stop_reason"] = level_plan.mip_stop_reason
    native_metadata["coarsest_segmentation_level_not_written"] = bool(
        level_plan.coarsest_segmentation_level_not_written
    )
    native_metadata["mip_stop_warnings"] = list(level_plan.warnings)
    native_metadata["min_side_for_mips"] = (
        int(min_side_for_mips) if min_side_for_mips is not None else None
    )
    native_metadata["canonical_canvas_in_source_level_coordinates"] = specs[
        0
    ].canonical_canvas_source_yx.as_dict()
    root.attrs["native_source_pyramid"] = native_metadata
    unique_source_tiles = stats.pop("unique_source_tiles_touched", set())
    stats["unique_source_tiles_touched"] = len(unique_source_tiles)
    stats["unique_rgb_chunks_written"] = len(unique_rgb_chunks)
    stats["unique_mask_chunks_written"] = len(unique_mask_chunks)
    stats["rgb_write_amplification"] = (
        float(stats["rgb_chunk_write_calls"] / len(unique_rgb_chunks)) if unique_rgb_chunks else 0.0
    )
    stats["mask_write_amplification"] = (
        float(stats["mask_chunk_write_calls"] / len(unique_mask_chunks))
        if unique_mask_chunks
        else 0.0
    )
    stats["num_mips"] = len(specs)
    stats["requested_mips"] = int(requested_mips) if requested_mips is not None else None
    stats["native_mip_stop_policy"] = level_plan.native_mip_stop_policy
    stats["native_mip_stop_level"] = int(level_plan.native_mip_stop_level)
    stats["native_mip_stop_level_source"] = level_plan.native_mip_stop_policy
    stats["available_native_mips"] = max(1, len(ets_level_shapes) - int(source_level))
    stats["min_side_for_mips"] = int(min_side_for_mips) if min_side_for_mips is not None else None
    stats["mip_stop_reason"] = level_plan.mip_stop_reason
    stats["coarsest_segmentation_level_not_written"] = bool(
        level_plan.coarsest_segmentation_level_not_written
    )
    stats["mip_stop_warnings"] = list(level_plan.warnings)
    stats["rgb_s0_chunks_expected"] = int(
        max(1, math.ceil(specs[0].output_shape_yx[0] / block_xy))
        * max(1, math.ceil(specs[0].output_shape_yx[1] / block_xy))
    )
    stats["mask_s0_chunks_expected"] = stats["rgb_s0_chunks_expected"] if store_tissue_mask else 0
    stats["combined_s0_chunks_expected"] = int(
        stats["rgb_s0_chunks_expected"] + stats["mask_s0_chunks_expected"]
    )
    stats["output_scale_to_source_level"] = output_scale_to_source_level
    stats["native_pyramid_levels"] = per_scale
    stats["canonical_canvas_in_source_level_coordinates"] = specs[
        0
    ].canonical_canvas_source_yx.as_dict()
    if run_manifest is not None:
        run_manifest.update(_json_ready(stats))
    return _json_ready(stats)


def _read_masked_ets_block(
    block: np.ndarray,
    *,
    ets_path: str,
    source_level: int,
    source_shape_yx: tuple[int, int],
    tile_origin_yx: tuple[int, int],
    logical_canvas_source_yx: tuple[int, int, int, int],
    label_crop_seg_yx: tuple[int, int, int, int],
    lr_labels: np.ndarray,
    label_id: int,
    output_kind: str = "masked_rgb",
    block_info: dict | None = None,
) -> np.ndarray:
    """Dask block callback for direct ETS tissue tile extraction."""
    if block_info is None:
        return np.zeros_like(block)

    loc = block_info[None]["array-location"]
    out_y0, out_y1 = map(int, loc[0])
    out_x0, out_x1 = map(int, loc[1])
    block_h = out_y1 - out_y0
    block_w = out_x1 - out_x0
    if output_kind == "mask":
        out = np.zeros((block_h, block_w), dtype=np.uint8)
    else:
        out = np.zeros((block_h, block_w, block.shape[2]), dtype=block.dtype)

    tile_y0, tile_x0 = map(int, tile_origin_yx)
    canvas = BoundsYX(*map(int, logical_canvas_source_yx))
    source_y0 = tile_y0 + out_y0
    source_y1 = tile_y0 + out_y1
    source_x0 = tile_x0 + out_x0
    source_x1 = tile_x0 + out_x1
    source_h, source_w = map(int, source_shape_yx)

    valid_y0 = max(0, source_y0)
    valid_y1 = min(source_h, source_y1)
    valid_x0 = max(0, source_x0)
    valid_x1 = min(source_w, source_x1)
    if valid_y1 <= valid_y0 or valid_x1 <= valid_x0:
        return out

    region = None
    if output_kind != "mask":
        region = _read_ets_region_yxc(
            ets_path,
            level=source_level,
            x0=valid_x0,
            y0=valid_y0,
            x1=valid_x1,
            y1=valid_y1,
        )

    if canvas.y0 != tile_y0 or canvas.x0 != tile_x0:
        raise ValueError("logical_canvas_source_yx origin must match tile_origin_yx.")

    yr = source_h / lr_labels.shape[0]
    xr = source_w / lr_labels.shape[1]
    valid_region = BoundsYX(valid_y0, valid_x0, valid_y1, valid_x1)
    label_crop = BoundsYX(*map(int, label_crop_seg_yx)).clip(lr_labels.shape)
    mask = project_label_mask_to_source_region(
        lr_labels,
        label_id=int(label_id),
        source_region_yx=valid_region,
        label_crop_seg_yx=label_crop,
        scale_y=yr,
        scale_x=xr,
    )
    dst_y0 = valid_y0 - source_y0
    dst_x0 = valid_x0 - source_x0
    if output_kind == "mask":
        out[dst_y0 : dst_y0 + mask.shape[0], dst_x0 : dst_x0 + mask.shape[1]] = mask.astype(
            np.uint8
        )
    elif output_kind == "rgb":
        assert region is not None
        out[dst_y0 : dst_y0 + region.shape[0], dst_x0 : dst_x0 + region.shape[1], :] = region
    else:
        assert region is not None
        masked = np.where(mask[..., None], region, 0)
        out[dst_y0 : dst_y0 + masked.shape[0], dst_x0 : dst_x0 + masked.shape[1], :] = masked
    return out


def _direct_ets_tissue_tile_records(
    *,
    ets_path: Path,
    source_level: int,
    source_shape_yx: tuple[int, int],
    low_res_filled: np.ndarray,
    chunk: int,
    pad_multiple: int | None,
    extra_margin_px: int,
    tile_frame_level: str,
    crop_shape_policy: str = "notebook_square",
    materialize_masked_rgb: bool = True,
    masked_rgb_fill_value: int = 0,
) -> tuple[list[TissueTileRecord], int]:
    """Build lazy per-tissue tile records that read directly from ETS source blocks."""
    tile_frame_level = _normalize_tile_frame_level(tile_frame_level)
    source_h, source_w = map(int, source_shape_yx)
    lr_labels = _filled_lr_labels(low_res_filled)
    if lr_labels.max() == 0:
        return [], 0

    if pad_multiple is None:
        pad_multiple = chunk
    frame_specs, tile_dim = _build_tissue_frame_specs(
        lr_labels,
        source_shape_yx=(source_h, source_w),
        tile_frame_level=tile_frame_level,
        pad_multiple=pad_multiple,
        extra_margin_px=extra_margin_px,
        crop_shape_policy=crop_shape_policy,
    )

    records: list[TissueTileRecord] = []
    for spec in frame_specs:
        canvas = spec.logical_canvas_source_yx
        clipped = spec.clipped_source_yx
        if clipped.h <= 0 or clipped.w <= 0:
            continue

        tile_h, tile_w = spec.source_canvas_shape_yx
        rgb_chunks = (min(chunk, tile_h), min(chunk, tile_w), 3)
        dummy_rgb = da.zeros((tile_h, tile_w, 3), chunks=rgb_chunks, dtype=np.uint8)
        rgb = dummy_rgb.map_blocks(
            _read_masked_ets_block,
            dtype=np.uint8,
            ets_path=str(ets_path),
            source_level=int(source_level),
            source_shape_yx=(source_h, source_w),
            tile_origin_yx=(canvas.y0, canvas.x0),
            logical_canvas_source_yx=canvas.as_yx(),
            label_crop_seg_yx=spec.label_crop_seg_yx.as_yx(),
            lr_labels=lr_labels,
            label_id=int(spec.label_id),
            output_kind="rgb",
        )
        mask_chunks = (min(chunk, tile_h), min(chunk, tile_w))
        dummy_mask = da.zeros((tile_h, tile_w), chunks=mask_chunks, dtype=np.uint8)
        mask = dummy_mask.map_blocks(
            _read_masked_ets_block,
            dtype=np.uint8,
            ets_path=str(ets_path),
            source_level=int(source_level),
            source_shape_yx=(source_h, source_w),
            tile_origin_yx=(canvas.y0, canvas.x0),
            logical_canvas_source_yx=canvas.as_yx(),
            label_crop_seg_yx=spec.label_crop_seg_yx.as_yx(),
            lr_labels=lr_labels,
            label_id=int(spec.label_id),
            output_kind="mask",
        )
        tile = (
            da.where(mask[..., None].astype(bool), rgb, int(masked_rgb_fill_value))
            if materialize_masked_rgb
            else rgb
        )
        record_tile_dim = int(tile_h) if int(tile_h) == int(tile_w) else int(max(tile_h, tile_w))
        records.append(
            TissueTileRecord(
                tile=tile,
                tissue_index=int(spec.tissue_index),
                label_id=int(spec.label_id),
                crop_bounds_source_level=spec.clipped_source_yx.as_xyxy(),
                crop_bounds_segmentation_level=spec.clipped_frame_seg_yx.as_xyxy(),
                tile_dim=record_tile_dim,
                tile_shape_yx=(int(tile_h), int(tile_w)),
                mask=mask,
                tile_frame_level=tile_frame_level,
                crop_shape_policy=crop_shape_policy,
                source_tile_dim=record_tile_dim,
                segmentation_tile_dim=int(spec.segmentation_tile_dim),
                scale_y=float(spec.scale_y),
                scale_x=float(spec.scale_x),
                frame_debug=spec.debug_dict(),
            )
        )
    return records, int(tile_dim)


def _mask_component_boxes(mask: np.ndarray) -> list[dict[str, int]]:
    """Return left-to-right component boxes for a boolean mask."""
    lbl = measure.label(np.asarray(mask, dtype=bool), connectivity=2)
    records: list[dict[str, int]] = []
    for prop in sorted(measure.regionprops(lbl), key=lambda item: item.bbox[1]):
        y0, x0, y1, x1 = map(int, prop.bbox)
        records.append(
            {
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "area_px": int(prop.area),
            }
        )
    return records


def _mask_iou(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Return IoU over the common crop of two masks."""
    h = min(int(a.shape[0]), int(b.shape[0]))
    w = min(int(a.shape[1]), int(b.shape[1]))
    aa = np.asarray(a[:h, :w], dtype=bool)
    bb = np.asarray(b[:h, :w], dtype=bool)
    union = int(np.logical_or(aa, bb).sum())
    intersection = int(np.logical_and(aa, bb).sum())
    return {
        "shape": [h, w],
        "intersection_px": intersection,
        "union_px": union,
        "iou": float(intersection / union) if union else 1.0,
    }


def _to_overlay_rgb(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return an RGB image with component boundaries overlaid in red."""
    from skimage.segmentation import find_boundaries

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            arr = (arr.astype(np.float32) / max(1, info.max) * 255.0).clip(0, 255)
        else:
            arr = arr.astype(np.float32)
            if np.nanmax(arr) <= 1.0:
                arr = arr * 255.0
            arr = np.nan_to_num(arr, nan=0.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    else:
        arr = arr.copy()

    boundary = find_boundaries(np.asarray(mask, dtype=bool), mode="outer")
    h = min(boundary.shape[0], arr.shape[0])
    w = min(boundary.shape[1], arr.shape[1])
    out = arr[:h, :w, :].copy()
    out[boundary[:h, :w]] = np.array([255, 0, 0], dtype=np.uint8)
    return out


def _json_ready(value: Any) -> Any:
    """Convert common NumPy/Python containers to JSON-serializable values."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _bounds_from_debug(debug: dict[str, Any], key: str) -> BoundsYX:
    payload = debug[key]
    return BoundsYX(
        y0=int(payload["y0"]),
        x0=int(payload["x0"]),
        y1=int(payload["y1"]),
        x1=int(payload["x1"]),
    )


def _array_stats(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr)
    if arr.size == 0:
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    stats: dict[str, Any] = {
        "shape": list(map(int, arr.shape)),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }
    channel_axis: int | None = None
    if arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4):
            channel_axis = -1
        elif arr.shape[0] in (1, 3, 4):
            channel_axis = 0

    if channel_axis is not None:
        channel_arr = np.moveaxis(arr, channel_axis, -1)
        channels = min(int(channel_arr.shape[-1]), 4)
        flat = channel_arr[..., :channels].reshape(-1, channels).astype(np.float64)
        stats["channel_means"] = [float(v) for v in flat.mean(axis=0)]
        stats["channel_stds"] = [float(v) for v in flat.std(axis=0)]
        stats["channel_mins"] = [float(v) for v in flat.min(axis=0)]
        stats["channel_maxs"] = [float(v) for v in flat.max(axis=0)]
        if channels >= 3:
            absdiffs = {
                "rg": float(np.mean(np.abs(flat[:, 0] - flat[:, 1]))),
                "rb": float(np.mean(np.abs(flat[:, 0] - flat[:, 2]))),
                "gb": float(np.mean(np.abs(flat[:, 1] - flat[:, 2]))),
            }
            stats["channel_absdiff_means"] = absdiffs
            stats["channels_nearly_identical"] = bool(max(absdiffs.values()) <= 2.0)
            corr_flat = flat
            if corr_flat.shape[0] > 100_000:
                step = max(1, corr_flat.shape[0] // 100_000)
                corr_flat = corr_flat[::step]
            correlations: dict[str, float | None] = {}
            for name, a_idx, b_idx in (("rg", 0, 1), ("rb", 0, 2), ("gb", 1, 2)):
                a = corr_flat[:, a_idx]
                b = corr_flat[:, b_idx]
                if float(a.std()) == 0.0 or float(b.std()) == 0.0:
                    correlations[name] = None
                else:
                    correlations[name] = float(np.corrcoef(a, b)[0, 1])
            stats["channel_correlations"] = correlations
    return stats


def _read_ome_zarr_s0_yxc(
    path: Path,
    *,
    max_debug_pixels: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Read an OME-Zarr ``s0`` array as YXC for diagnostics when small enough."""
    import zarr

    zarr_path = Path(path)
    s0_path = zarr_path / "s0"
    try:
        arr = zarr.open_array(str(s0_path), mode="r")
    except Exception as exc:
        return None, {"skipped": f"could not open {s0_path}: {exc}"}

    shape = tuple(map(int, arr.shape))
    if len(shape) == 2:
        y, x = shape
        channel_axis = None
    elif len(shape) == 3 and shape[0] in (1, 3, 4):
        y, x = shape[1], shape[2]
        channel_axis = 0
    elif len(shape) == 3 and shape[-1] in (1, 3, 4):
        y, x = shape[0], shape[1]
        channel_axis = -1
    else:
        return None, {"skipped": f"unsupported readback shape {shape}", "shape": list(shape)}

    if int(y) * int(x) > int(max_debug_pixels):
        return None, {
            "skipped": "readback s0 exceeds max_debug_pixels",
            "shape": list(shape),
            "pixels": int(y) * int(x),
            "max_debug_pixels": int(max_debug_pixels),
        }

    data = np.asarray(arr[...])
    if channel_axis == 0:
        data = np.moveaxis(data, 0, -1)
    elif channel_axis is None:
        data = np.repeat(data[..., None], 3, axis=2)
    if data.ndim == 3 and data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=2)
    data = data[..., :3]
    return data, _array_stats(data)


def _resolve_readback_ome_zarr_path(
    readback_path: Path | None,
    *,
    tissue_index: int,
) -> Path | None:
    """Resolve a readback tissue OME-Zarr from either a direct path or parent directory."""
    if readback_path is None:
        return None
    path = Path(readback_path)
    if path.name.endswith(".ome.zarr"):
        return path if path.exists() else None
    if not path.exists():
        return None
    exact_matches = sorted(path.glob(f"*tissue_{int(tissue_index):02d}.ome.zarr"))
    if exact_matches:
        return exact_matches[0]
    all_matches = sorted(path.glob("*.ome.zarr"))
    if 0 <= int(tissue_index) < len(all_matches):
        return all_matches[int(tissue_index)]
    return None


def _place_region_in_canvas(
    *,
    canvas_shape_yx: tuple[int, int],
    canvas_bounds: BoundsYX,
    clipped_bounds: BoundsYX,
    region: np.ndarray,
) -> np.ndarray:
    out = np.zeros(
        (int(canvas_shape_yx[0]), int(canvas_shape_yx[1]), region.shape[2]), dtype=region.dtype
    )
    dst_y0 = int(clipped_bounds.y0 - canvas_bounds.y0)
    dst_x0 = int(clipped_bounds.x0 - canvas_bounds.x0)
    out[dst_y0 : dst_y0 + region.shape[0], dst_x0 : dst_x0 + region.shape[1], :] = region
    return out


def _resize_panel_yxc(arr: np.ndarray, output_shape_yx: tuple[int, int]) -> np.ndarray:
    from skimage.transform import resize

    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    resized = resize(
        arr,
        (*map(int, output_shape_yx), arr.shape[2]),
        order=0 if arr.dtype == np.uint8 and np.isin(arr, [0, 255]).all() else 1,
        preserve_range=True,
        anti_aliasing=arr.ndim == 3 and max(arr.shape[:2]) > max(output_shape_yx),
    )
    return np.clip(resized, 0, 255).astype(np.uint8)


def _reference_segmentation_crop(
    image_yxc: np.ndarray,
    mask: np.ndarray,
    *,
    frame_debug: dict[str, Any],
) -> np.ndarray:
    logical = _bounds_from_debug(frame_debug, "logical_frame_seg_yx")
    clipped = _bounds_from_debug(frame_debug, "clipped_frame_seg_yx")
    side = int(frame_debug["segmentation_tile_dim"])
    out = np.zeros((side, side, 3), dtype=np.uint8)
    if clipped.h <= 0 or clipped.w <= 0:
        return out
    region = image_yxc[clipped.y0 : clipped.y1, clipped.x0 : clipped.x1, :3]
    region_mask = mask[clipped.y0 : clipped.y1, clipped.x0 : clipped.x1]
    region = np.where(region_mask[..., None], region, 0)
    dst_y0 = int(clipped.y0 - logical.y0)
    dst_x0 = int(clipped.x0 - logical.x0)
    out[dst_y0 : dst_y0 + region.shape[0], dst_x0 : dst_x0 + region.shape[1], :] = region
    return out


def _direct_record_debug_arrays(
    *,
    ets_path: Path,
    source_level: int,
    source_shape_yx: tuple[int, int],
    lr_labels: np.ndarray,
    record: TissueTileRecord,
    full_source_yxc: np.ndarray | None,
    readback_yxc: np.ndarray | None,
    readback_stats: dict[str, Any] | None,
    max_debug_pixels: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Materialize one direct ETS record for debug stats and panels when small enough."""
    if record.frame_debug is None:
        return {"skipped": "record has no frame_debug"}, {}
    frame_debug = record.frame_debug
    canvas = _bounds_from_debug(frame_debug, "logical_canvas_source_yx")
    clipped = _bounds_from_debug(frame_debug, "clipped_source_yx")
    label_crop = _bounds_from_debug(frame_debug, "label_crop_seg_yx")
    canvas_pixels = int(canvas.h * canvas.w)
    if canvas_pixels > int(max_debug_pixels):
        return {
            "skipped": "source canvas exceeds max_debug_pixels",
            "canvas_pixels": canvas_pixels,
            "max_debug_pixels": int(max_debug_pixels),
        }, {}

    region = _read_ets_region_yxc(
        ets_path,
        level=int(source_level),
        x0=clipped.x0,
        y0=clipped.y0,
        x1=clipped.x1,
        y1=clipped.y1,
    )
    unmasked = _place_region_in_canvas(
        canvas_shape_yx=(canvas.h, canvas.w),
        canvas_bounds=canvas,
        clipped_bounds=clipped,
        region=region,
    )
    mask_region = project_label_mask_to_source_region(
        lr_labels,
        label_id=int(record.label_id),
        source_region_yx=clipped,
        label_crop_seg_yx=label_crop,
        scale_y=float(frame_debug["scale_y"]),
        scale_x=float(frame_debug["scale_x"]),
    )
    mask_canvas = np.zeros((canvas.h, canvas.w), dtype=bool)
    dst_y0 = int(clipped.y0 - canvas.y0)
    dst_x0 = int(clipped.x0 - canvas.x0)
    mask_canvas[dst_y0 : dst_y0 + mask_region.shape[0], dst_x0 : dst_x0 + mask_region.shape[1]] = (
        mask_region
    )
    masked = np.where(mask_canvas[..., None], unmasked, 0)
    streamed_tile = record.tile.compute()

    stats: dict[str, Any] = {
        "frame_debug": frame_debug,
        "unmasked_source_crop": _array_stats(unmasked),
        "projected_mask": {
            **_array_stats(mask_canvas.astype(np.uint8)),
            "sum": int(mask_canvas.sum()),
            "mean": float(mask_canvas.mean()) if mask_canvas.size else 0.0,
        },
        "masked_source_crop": _array_stats(masked),
        "tile_before_write": _array_stats(streamed_tile),
        "streamed_vs_materialized_absdiff": _array_stats(
            np.abs(streamed_tile.astype(np.int16) - masked.astype(np.int16))
        ),
    }
    if full_source_yxc is not None:
        full_region = full_source_yxc[clipped.y0 : clipped.y1, clipped.x0 : clipped.x1, :3]
        full_canvas = _place_region_in_canvas(
            canvas_shape_yx=(canvas.h, canvas.w),
            canvas_bounds=canvas,
            clipped_bounds=clipped,
            region=full_region,
        )
        stats["full_read_source_crop"] = _array_stats(full_canvas)
        stats["direct_vs_full_unmasked_absdiff"] = _array_stats(
            np.abs(unmasked.astype(np.int16) - full_canvas.astype(np.int16))
        )
    if readback_yxc is not None:
        stats["ome_zarr_readback_s0"] = readback_stats or _array_stats(readback_yxc)
        if readback_yxc.shape == streamed_tile.shape:
            stats["readback_vs_tile_absdiff"] = _array_stats(
                np.abs(readback_yxc.astype(np.int16) - streamed_tile.astype(np.int16))
            )
        else:
            stats["readback_vs_tile_absdiff"] = {
                "skipped": "shape mismatch",
                "readback_shape": list(map(int, readback_yxc.shape)),
                "tile_shape": list(map(int, streamed_tile.shape)),
            }

    arrays = {
        "unmasked": unmasked,
        "mask": mask_canvas,
        "masked": masked,
        "streamed_tile": streamed_tile,
    }
    if full_source_yxc is not None:
        arrays["full_read_source_crop"] = full_canvas
    if readback_yxc is not None:
        arrays["ome_zarr_readback_s0"] = readback_yxc
    return stats, arrays


def _write_four_panel_debug(
    output_path: Path,
    *,
    tissue_index: int,
    reference: np.ndarray,
    unmasked: np.ndarray,
    mask: np.ndarray,
    masked: np.ndarray,
) -> Path:
    import imageio.v3 as iio

    target_shape = reference.shape[:2]
    panels = [
        reference.astype(np.uint8),
        _resize_panel_yxc(unmasked, target_shape),
        _resize_panel_yxc(mask.astype(np.uint8) * 255, target_shape),
        _resize_panel_yxc(masked, target_shape),
    ]
    spacer = np.ones((target_shape[0], 8, 3), dtype=np.uint8) * 255
    panel = np.concatenate(
        [panels[0], spacer, panels[1], spacer, panels[2], spacer, panels[3]], axis=1
    )
    out_path = output_path / f"tissue_{int(tissue_index):02d}_four_panel_debug.png"
    iio.imwrite(out_path, panel)
    return out_path


def _write_pixel_path_debug(
    output_path: Path,
    *,
    tissue_index: int,
    full_level: np.ndarray | None,
    crop_from_full: np.ndarray | None,
    direct_crop: np.ndarray,
    readback_or_streamed: np.ndarray,
    target_shape_yx: tuple[int, int],
) -> Path:
    """Write A/B/C/D pixel-path panels for source/readback isolation."""
    import imageio.v3 as iio

    target_shape = tuple(map(int, target_shape_yx))
    blank = np.zeros((*target_shape, 3), dtype=np.uint8)
    full_panel = _resize_panel_yxc(full_level, target_shape) if full_level is not None else blank
    full_crop_panel = (
        _resize_panel_yxc(crop_from_full, target_shape) if crop_from_full is not None else blank
    )
    panels = [
        full_panel,
        full_crop_panel,
        _resize_panel_yxc(direct_crop, target_shape),
        _resize_panel_yxc(readback_or_streamed, target_shape),
    ]
    spacer = np.ones((target_shape[0], 8, 3), dtype=np.uint8) * 255
    panel = np.concatenate(
        [panels[0], spacer, panels[1], spacer, panels[2], spacer, panels[3]], axis=1
    )
    out_path = output_path / f"tissue_{int(tissue_index):02d}_pixel_path_debug.png"
    iio.imwrite(out_path, panel)
    return out_path


def diagnose_vsi_replating(
    vsi_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    flat_image_path: str | Path | None = None,
    readback_ome_zarr: str | Path | None = None,
    source_level: int | str = 7,
    segmentation_level: int | str | None = 7,
    tile_frame_level: str = "segmentation",
    segmentation_config: SegmentationConfig | None = None,
    tile_config: TileConfig | None = None,
    max_debug_pixels: int = 50_000_000,
) -> dict[str, Any]:
    """
    Read only the requested ETS levels and report segmentation/crop diagnostics.

    This is a no-full-rerun smoke helper for comparing the notebook flat-image
    route with the VSI/ETS replating route. When ``output_dir`` is supplied, it
    writes ``diagnostics.json`` plus mask-boundary overlay PNGs.
    """
    vsi_path = Path(vsi_path)
    output_path = Path(output_dir) if output_dir is not None else None
    readback_root = Path(readback_ome_zarr) if readback_ome_zarr is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    config = segmentation_config or SegmentationConfig()
    tile_cfg = tile_config or TileConfig()
    tile_frame_level = _normalize_tile_frame_level(tile_frame_level)

    ets_path = find_ets_file(vsi_path)
    if ets_path is None:
        raise FileNotFoundError(f"No ETS file found for VSI {vsi_path}")
    ets_path = Path(ets_path)

    with ETSFile(ets_path) as ets:
        source_idx = _resolve_ets_level(
            source_level,
            default_index=0,
            nlevels=ets.nlevels,
            label="source",
        )
        segmentation_idx = _resolve_ets_level(
            segmentation_level,
            default_index=ets.nlevels - 1,
            nlevels=ets.nlevels,
            label="segmentation",
        )
        source_shape_yx = tuple(map(int, ets.level_shape(source_idx)))
        segmentation_yxc = ets.read_level(segmentation_idx)

    ets_mask, ets_info = _segment_for_plating(
        segmentation_yxc,
        segment_fn=None,
        segmentation_config=config,
        min_size=config.min_area_px,
        struct_elem_px=config.struct_elem_px,
    )
    ets_mask = np.asarray(ets_mask, dtype=bool)
    lr_labels = _filled_lr_labels(ets_mask)
    tile_records, tile_dim = _direct_ets_tissue_tile_records(
        ets_path=ets_path,
        source_level=source_idx,
        source_shape_yx=source_shape_yx,
        low_res_filled=ets_mask,
        chunk=tile_cfg.chunk_size,
        pad_multiple=tile_cfg.pad_multiple,
        extra_margin_px=tile_cfg.extra_margin_px,
        tile_frame_level=tile_frame_level,
    )
    scale_y = float(source_shape_yx[0] / ets_mask.shape[0])
    scale_x = float(source_shape_yx[1] / ets_mask.shape[1])
    nominal_scale = float(2 ** (int(segmentation_idx) - int(source_idx)))
    first_record = tile_records[0] if tile_records else None
    source_tile_dim = (
        int(first_record.source_tile_dim or first_record.tile_dim)
        if first_record
        else int(tile_dim)
    )
    segmentation_tile_dim = (
        int(first_record.segmentation_tile_dim)
        if first_record and first_record.segmentation_tile_dim is not None
        else int(np.ceil(source_tile_dim / max(scale_y, scale_x))) if source_tile_dim else 0
    )
    effective_segmentation_tile_dim = (
        float(source_tile_dim / max(scale_y, scale_x)) if source_tile_dim else 0.0
    )
    full_source_yxc: np.ndarray | None = None
    if int(source_shape_yx[0]) * int(source_shape_yx[1]) <= int(max_debug_pixels):
        with ETSFile(ets_path) as ets:
            full_source_yxc = ets.read_level(source_idx)

    flat_summary: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    reference_image_yxc = segmentation_yxc
    reference_mask = ets_mask
    if flat_image_path is not None:
        import imageio.v3 as iio

        flat_path = Path(flat_image_path)
        flat_yxc = iio.imread(flat_path)
        flat_mask, flat_info = _segment_for_plating(
            flat_yxc,
            segment_fn=None,
            segmentation_config=config,
            min_size=config.min_area_px,
            struct_elem_px=config.struct_elem_px,
        )
        flat_mask = np.asarray(flat_mask, dtype=bool)
        reference_image_yxc = flat_yxc
        reference_mask = flat_mask
        flat_summary = {
            "path": str(flat_path),
            "shape": list(map(int, flat_yxc.shape)),
            "dtype": str(flat_yxc.dtype),
            "component_count": int(measure.label(flat_mask, connectivity=2).max()),
            "component_boxes": _mask_component_boxes(flat_mask),
            "segmentation_info": flat_info,
        }
        comparison = {"flat_vs_ets_mask": _mask_iou(flat_mask, ets_mask)}
        if output_path is not None:
            iio.imwrite(output_path / "flat_overlay.png", _to_overlay_rgb(flat_yxc, flat_mask))

    debug_rows: list[dict[str, Any]] = []
    four_panel_paths: list[str] = []
    pixel_path_panel_paths: list[str] = []
    for record in tile_records:
        readback_yxc: np.ndarray | None = None
        readback_stats: dict[str, Any] | None = None
        readback_path = _resolve_readback_ome_zarr_path(
            readback_root,
            tissue_index=record.tissue_index,
        )
        if readback_path is not None:
            readback_yxc, readback_stats = _read_ome_zarr_s0_yxc(
                readback_path,
                max_debug_pixels=max_debug_pixels,
            )
            if readback_stats is not None:
                readback_stats["path"] = str(readback_path)

        debug_stats, debug_arrays = _direct_record_debug_arrays(
            ets_path=ets_path,
            source_level=source_idx,
            source_shape_yx=source_shape_yx,
            lr_labels=lr_labels,
            record=record,
            full_source_yxc=full_source_yxc,
            readback_yxc=readback_yxc,
            readback_stats=readback_stats,
            max_debug_pixels=max_debug_pixels,
        )
        if readback_root is not None and readback_path is None:
            debug_stats["ome_zarr_readback_s0"] = {
                "skipped": f"no readback OME-Zarr found for tissue {int(record.tissue_index)}",
                "readback_root": str(readback_root),
            }
        debug_rows.append(
            {
                "tissue_index": int(record.tissue_index),
                "label_id": int(record.label_id),
                **debug_stats,
            }
        )
        if output_path is not None and record.tissue_index == 0 and debug_arrays:
            reference = _reference_segmentation_crop(
                reference_image_yxc,
                reference_mask,
                frame_debug=record.frame_debug or {},
            )
            four_panel_path = _write_four_panel_debug(
                output_path,
                tissue_index=record.tissue_index,
                reference=reference,
                unmasked=debug_arrays["unmasked"],
                mask=debug_arrays["mask"],
                masked=debug_arrays["masked"],
            )
            four_panel_paths.append(str(four_panel_path))
            pixel_reference_shape = reference.shape[:2]
            pixel_path = _write_pixel_path_debug(
                output_path,
                tissue_index=record.tissue_index,
                full_level=full_source_yxc,
                crop_from_full=debug_arrays.get("full_read_source_crop"),
                direct_crop=debug_arrays["unmasked"],
                readback_or_streamed=debug_arrays.get(
                    "ome_zarr_readback_s0",
                    debug_arrays["streamed_tile"],
                ),
                target_shape_yx=pixel_reference_shape,
            )
            pixel_path_panel_paths.append(str(pixel_path))

    result: dict[str, Any] = {
        "vsi_path": str(vsi_path),
        "ets_path": str(ets_path),
        "source_level": int(source_idx),
        "segmentation_level": int(segmentation_idx),
        "tile_frame_level": str(tile_frame_level),
        "source_shape_yx": list(source_shape_yx),
        "source_full_read_level": (
            _array_stats(full_source_yxc)
            if full_source_yxc is not None
            else {
                "skipped": "source level exceeds max_debug_pixels",
                "source_pixels": int(source_shape_yx[0]) * int(source_shape_yx[1]),
                "max_debug_pixels": int(max_debug_pixels),
            }
        ),
        "readback_ome_zarr": str(readback_root) if readback_root is not None else None,
        "scale_y": scale_y,
        "scale_x": scale_x,
        "nominal_power_of_two_scale": nominal_scale,
        "source_tile_dim": source_tile_dim,
        "segmentation_tile_dim": segmentation_tile_dim,
        "effective_segmentation_tile_dim": effective_segmentation_tile_dim,
        "notebook_equivalent_frame": str(tile_frame_level).strip().lower().replace("_", "-")
        == "segmentation",
        "ets_segmentation_input": {
            "shape": list(map(int, segmentation_yxc.shape)),
            "dtype": str(segmentation_yxc.dtype),
            "component_count": int(measure.label(ets_mask, connectivity=2).max()),
            "component_boxes": _mask_component_boxes(ets_mask),
            "segmentation_info": ets_info,
        },
        "flat_input": flat_summary,
        "comparison": comparison,
        "tile_dim": int(tile_dim),
        "debug_sidecars": {
            "diagnostics_json": (
                str(output_path / "diagnostics.json") if output_path is not None else None
            ),
            "four_panel_pngs": four_panel_paths,
            "pixel_path_pngs": pixel_path_panel_paths,
        },
        "debug_rows": debug_rows,
        "tile_records": [
            {
                "tissue_index": int(record.tissue_index),
                "label_id": int(record.label_id),
                "crop_bounds_source_level": list(record.crop_bounds_source_level),
                "crop_bounds_segmentation_level": list(record.crop_bounds_segmentation_level),
                "tile_dim": int(record.tile_dim),
                "frame_debug": record.frame_debug,
                "source_tile_dim": int(record.source_tile_dim or record.tile_dim),
                "segmentation_tile_dim": int(
                    record.segmentation_tile_dim
                    if record.segmentation_tile_dim is not None
                    else segmentation_tile_dim
                ),
                "effective_segmentation_tile_dim": float(
                    (record.source_tile_dim or record.tile_dim) / max(scale_y, scale_x)
                ),
            }
            for record in tile_records
        ],
    }

    result = _json_ready(result)

    if output_path is not None:
        import imageio.v3 as iio

        iio.imwrite(output_path / "ets_overlay.png", _to_overlay_rgb(segmentation_yxc, ets_mask))
        (output_path / "diagnostics.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

    return result


def estimate_vsi_direct_plating(
    vsi_path: str | Path,
    *,
    segment_fn=None,
    source_level: int | str = 0,
    segmentation_level: int | str | None = 7,
    output_profile: str = "validation",
    tile_frame_level: str = "segmentation",
    crop_shape_policy: str | None = None,
    segmentation_config: SegmentationConfig | None = None,
    tile_config: TileConfig | None = None,
    metadata_backend: str = "auto",
    metadata_schema: str = "v0.4",
    struct_elem_px: int = 9,
    min_size: int = 2000,
    plate_chunk_xy: int = 512,
    min_side_for_mips: int | None = None,
    dtype: np.dtype | str | None = "uint8",
    compression: str | None = None,
    store_tissue_mask: bool | None = None,
    primary_rgb_mode: str | None = None,
    masked_rgb_fill_value: int | None = None,
    store_unmasked_rgb: bool | None = None,
    materialize_masked_rgb: bool | None = None,
    sparse_zero_chunks: bool | None = None,
    pyramid_generation_policy: str | None = None,
    source_tile_aligned_canvas: bool | None = None,
    native_mip_stop_policy: str | None = None,
    native_mip_stop_level: int | str | None = None,
    config_source: str | None = None,
) -> dict[str, Any]:
    """
    Estimate a direct VSI/ETS plating run without writing tissue OME-Zarrs.

    This performs the same metadata read, level resolution, segmentation, and
    crop-frame construction as :func:`process_vsi_with_direct_plating`, but it
    avoids building the full-resolution Dask tile graph or writing any arrays.
    """
    started = time.monotonic()
    vsi_path = Path(vsi_path)
    output_profile = _normalize_output_profile(output_profile)
    defaults = _profile_defaults(output_profile)
    if crop_shape_policy is None:
        crop_shape_policy = str(defaults["crop_shape_policy"])
    if compression is None:
        compression = str(defaults["compression"])
    if store_tissue_mask is None:
        store_tissue_mask = bool(defaults["store_tissue_mask"])
    rgb_options = _resolve_primary_rgb_options(
        primary_rgb_mode=primary_rgb_mode,
        materialize_masked_rgb=materialize_masked_rgb,
        masked_rgb_fill_value=masked_rgb_fill_value,
        store_unmasked_rgb=store_unmasked_rgb,
        defaults=defaults,
    )
    primary_rgb_mode = str(rgb_options["primary_rgb_mode"])
    materialize_masked_rgb = bool(rgb_options["materialize_masked_rgb"])
    masked_rgb_fill_value = int(rgb_options["masked_rgb_fill_value"])
    store_unmasked_rgb = bool(rgb_options["store_unmasked_rgb"])
    if sparse_zero_chunks is None:
        sparse_zero_chunks = bool(defaults["sparse_zero_chunks"])
    if pyramid_generation_policy is None:
        pyramid_generation_policy = str(
            defaults.get("pyramid_generation_policy", "downsample_streamed_s0")
        )
    if source_tile_aligned_canvas is None:
        source_tile_aligned_canvas = bool(defaults.get("source_tile_aligned_canvas", False))
    if native_mip_stop_policy is None:
        native_mip_stop_policy = str(defaults.get("native_mip_stop_policy", "segmentation_level"))
    if native_mip_stop_level is None:
        native_mip_stop_level = defaults.get("native_mip_stop_level", "segmentation_level")
    if tile_frame_level == "segmentation" and output_profile in {"production", "upload_staging"}:
        tile_frame_level = str(defaults["tile_frame_level"])
    tile_frame_level = _normalize_tile_frame_level(tile_frame_level)
    crop_shape_policy = _normalize_crop_shape_policy(crop_shape_policy)
    compression = _normalize_compression_mode(compression)
    pyramid_generation_policy = _normalize_pyramid_generation_policy(pyramid_generation_policy)
    native_mip_stop_policy = _normalize_native_mip_stop_policy(
        native_mip_stop_policy,
        native_mip_stop_level=native_mip_stop_level,
    )

    ets_path = find_ets_file(vsi_path)
    if ets_path is None:
        raise FileNotFoundError(f"No ETS file found for VSI {vsi_path}")
    ets_path = Path(ets_path)

    metadata = get_vsi_metadata(
        vsi_path,
        metadata_backend=metadata_backend,
        target_schema="latest",
    )
    if not metadata:
        raise RuntimeError(f"Unable to extract structural metadata for VSI {vsi_path}")

    if tile_config is not None:
        plate_chunk_xy = int(tile_config.chunk_size)
        tile_extra_margin_px = int(tile_config.extra_margin_px)
        tile_pad_multiple = int(tile_config.pad_multiple)
        if output_profile in {"production", "upload_staging"} and tile_extra_margin_px == 0:
            tile_extra_margin_px = int(defaults["extra_margin_px"])
        if crop_shape_policy == "notebook_square":
            crop_shape_policy = _normalize_crop_shape_policy(tile_config.crop_shape_policy)
    else:
        tile_extra_margin_px = int(defaults["extra_margin_px"])
        tile_pad_multiple = int(plate_chunk_xy)

    with ETSFile(ets_path) as ets:
        source_idx = _resolve_ets_level(
            source_level,
            default_index=0,
            nlevels=ets.nlevels,
            label="source",
        )
        segmentation_idx = _resolve_ets_level(
            segmentation_level,
            default_index=ets.nlevels - 1,
            nlevels=ets.nlevels,
            label="segmentation",
        )
        source_shape_yx = tuple(map(int, ets.level_shape(source_idx)))
        ets_level_shapes_yx = [tuple(map(int, ets.level_shape(idx))) for idx in range(ets.nlevels)]
        source_tile_size_yx = (
            int(getattr(ets, "tile_ysize", plate_chunk_xy)),
            int(getattr(ets, "tile_xsize", plate_chunk_xy)),
        )
        segmentation_yxc = ets.read_level(segmentation_idx)
    seg_cyx = da.from_array(
        np.moveaxis(segmentation_yxc, -1, 0),
        chunks=(
            int(segmentation_yxc.shape[2]),
            min(plate_chunk_xy, int(segmentation_yxc.shape[0])),
            min(plate_chunk_xy, int(segmentation_yxc.shape[1])),
        ),
    )
    filled_lr, segmentation_info = _segment_for_plating(
        seg_cyx,
        segment_fn=segment_fn,
        segmentation_config=segmentation_config,
        min_size=segmentation_config.min_area_px if segmentation_config is not None else min_size,
        struct_elem_px=(
            segmentation_config.struct_elem_px
            if segmentation_config is not None
            else struct_elem_px
        ),
    )
    lr_labels = _filled_lr_labels(np.asarray(filled_lr, dtype=bool))
    if lr_labels.max() == 0:
        return _json_ready(
            {
                "vsi_path": str(vsi_path),
                "ets_path": str(ets_path),
                "metadata_backend": metadata_backend,
                "metadata_schema": metadata_schema,
                "source_level": int(source_idx),
                "segmentation_level": int(segmentation_idx),
                "tile_frame_level": tile_frame_level,
                "crop_shape_policy": crop_shape_policy,
                "output_profile": output_profile,
                "primary_rgb_mode": primary_rgb_mode,
                "masked_rgb_fill_value": int(masked_rgb_fill_value),
                "mask_applied_to_primary_rgb": primary_rgb_mode == "masked_rgb",
                "store_tissue_mask": bool(store_tissue_mask),
                "store_unmasked_rgb": bool(store_unmasked_rgb),
                "pyramid_generation_policy": pyramid_generation_policy,
                "source_tile_aligned_canvas": bool(source_tile_aligned_canvas),
                "native_mip_stop_policy": native_mip_stop_policy,
                "native_mip_stop_level": int(segmentation_idx),
                "native_mip_stop_level_source": native_mip_stop_policy,
                "mip_stop_reason": "segmentation_level",
                "coarsest_segmentation_level_not_written": False,
                "context_margin_px": int(tile_extra_margin_px),
                "effective_extra_margin_px": int(tile_extra_margin_px),
                "source_shape_yx": list(source_shape_yx),
                "segmentation_shape_yx": list(map(int, lr_labels.shape)),
                "tissue_count": 0,
                "tissues": [],
                "totals": {
                    "s0_chunks": 0,
                    "mask_empty_chunks": 0,
                    "mask_positive_chunks": 0,
                    "rgb_chunks_skippable_before_decode": 0,
                    "expected_sparse_zero_behavior": (
                        "empty_mask_chunks_skip_rgb_decode_and_write"
                        if primary_rgb_mode == "masked_rgb" and masked_rgb_fill_value == 0
                        else "post_decode_sparse_zero_only"
                    ),
                    "all_mip_chunks": 0,
                    "mask_all_mip_chunks": 0,
                    "combined_logical_chunks": 0,
                    "rgb_uncompressed_bytes_all_mips": 0,
                    "mask_uncompressed_bytes_all_mips": 0,
                    "total_uncompressed_bytes_rgb_plus_mask": 0,
                    "rgb_uncompressed_size_all_mips": _human_bytes(0),
                    "mask_uncompressed_size_all_mips": _human_bytes(0),
                    "total_uncompressed_size_rgb_plus_mask": _human_bytes(0),
                    "uncompressed_bytes_all_mips": 0,
                    "uncompressed_bytes_estimate": 0,
                    "uncompressed_size_all_mips": _human_bytes(0),
                    "compressed_bytes_sample_estimate": None,
                    "compressed_estimate_method": None,
                    "warnings": ["no_tissue_regions_found"],
                },
                "segmentation_info": segmentation_info,
                "elapsed_s": time.monotonic() - started,
            }
        )

    frame_specs, tile_dim = _build_tissue_frame_specs(
        lr_labels,
        source_shape_yx=source_shape_yx,
        tile_frame_level=tile_frame_level,
        pad_multiple=tile_pad_multiple,
        extra_margin_px=tile_extra_margin_px,
        crop_shape_policy=crop_shape_policy,
    )

    dtype_obj = np.dtype(dtype if dtype else np.uint8)
    bytes_per_pixel = int(dtype_obj.itemsize)
    min_side = int(min_side_for_mips or plate_chunk_xy)
    base_phys_xy_um = _physical_xy_from_metadata(metadata) or (1.0, 1.0)
    source_phys_xy_um = (
        float(base_phys_xy_um[0]) * (2**source_idx),
        float(base_phys_xy_um[1]) * (2**source_idx),
    )
    tissues: list[dict[str, Any]] = []
    total_s0_chunks = 0
    total_all_mip_chunks = 0
    total_mask_all_mip_chunks = 0
    total_bytes = 0
    total_mask_bytes = 0
    total_mask_empty_chunks = 0
    total_mask_positive_chunks = 0
    total_rgb_chunks_skippable_before_decode = 0
    total_projection_baseline_chunks = 0
    total_projection_baseline_bytes = 0
    warnings: list[str] = []

    for spec in frame_specs:
        canvas = spec.logical_canvas_source_yx
        clipped = spec.clipped_source_yx
        if clipped.h <= 0 or clipped.w <= 0:
            continue
        channels = 3
        native_specs: list[_NativePyramidLevelSpec] = []
        output_scale_to_source_level: dict[str, int] | None = None
        per_scale_shapes_yx: list[list[int]] | None = None
        per_scale_native_levels: list[dict[str, Any]] | None = None
        mip_stop_reason = "min_side_for_mips"
        tissue_native_mip_stop_policy: str | None = None
        tissue_native_mip_stop_level: int | None = None
        coarsest_segmentation_level_not_written = False
        mip_stop_warnings: list[str] = []
        context_margin_px = int(tile_extra_margin_px)
        source_tile_alignment_expansion_yx = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        projection_baseline_chunks = 0
        projection_baseline_bytes = 0
        if pyramid_generation_policy == "native_source_pyramid_crop":
            native_specs = _native_pyramid_level_specs(
                record=spec,
                source_level=int(source_idx),
                source_shape_yx=source_shape_yx,
                ets_level_shapes_yx=ets_level_shapes_yx,
                source_phys_xy_um=source_phys_xy_um,
                block_xy=plate_chunk_xy,
                min_side_for_mips=min_side_for_mips,
                source_tile_aligned_canvas=bool(source_tile_aligned_canvas),
                source_tile_size_yx=source_tile_size_yx,
                requested_mips=None,
                segmentation_level=int(segmentation_idx),
                native_mip_stop_policy=native_mip_stop_policy,
                native_mip_stop_level=native_mip_stop_level,
            )
            shapes_yxc = [
                (int(level.output_shape_yx[0]), int(level.output_shape_yx[1]), channels)
                for level in native_specs
            ]
            output_scale_to_source_level = {
                f"s{level.output_index}": int(level.source_level) for level in native_specs
            }
            per_scale_shapes_yx = [
                [int(level.output_shape_yx[0]), int(level.output_shape_yx[1])]
                for level in native_specs
            ]
            per_scale_native_levels = [
                {
                    "path": f"s{level.output_index}",
                    "source_level": int(level.source_level),
                    "output_shape_yx": list(map(int, level.output_shape_yx)),
                    "output_canvas_source_yx": level.canvas_source_yx.as_dict(),
                    "source_read_envelope_yx": level.source_read_envelope_yx.as_dict(),
                    "source_read_envelope_clipped_yx": (
                        level.clipped_source_read_envelope_yx.as_dict()
                    ),
                    "scale_from_parent_yx": list(map(float, level.scale_from_parent_yx)),
                    "translation_yx_um": list(map(float, level.translation_yx_um)),
                }
                for level in native_specs
            ]
            source_tile_alignment_expansion_yx = _bounds_expansion_yx(
                native_specs[0].canonical_canvas_source_yx,
                spec.logical_canvas_source_yx,
            )
            min_side_mip_count = compute_num_mips_min_side(
                native_specs[0].canonical_canvas_source_yx.w,
                native_specs[0].canonical_canvas_source_yx.h,
                min_side,
            )
            level_plan = resolve_native_output_levels(
                int(source_idx),
                int(segmentation_idx),
                len(ets_level_shapes_yx),
                native_mip_stop_policy,
                native_mip_stop_level,
                min_side_for_mips=min_side_for_mips,
                requested_mips=None,
                min_side_mip_count=min_side_mip_count,
            )
            tissue_native_mip_stop_policy = level_plan.native_mip_stop_policy
            tissue_native_mip_stop_level = int(level_plan.native_mip_stop_level)
            mip_stop_reason = level_plan.mip_stop_reason
            coarsest_segmentation_level_not_written = bool(
                level_plan.coarsest_segmentation_level_not_written
            )
            mip_stop_warnings = list(level_plan.warnings)
            warnings.extend(f"tissue_{spec.tissue_index:02d}:{w}" for w in mip_stop_warnings)
            for level, shape_yxc in zip(native_specs, shapes_yxc, strict=True):
                if level.source_level >= int(
                    _VALIDATED_SOURCE_LEVEL2_NATIVE_BASELINE["source_level"]
                ):
                    level_rgb_chunks = _chunk_count_yx(
                        y=shape_yxc[0],
                        x=shape_yxc[1],
                        chunk_xy=plate_chunk_xy,
                    )
                    projection_baseline_chunks += level_rgb_chunks
                    projection_baseline_bytes += int(
                        shape_yxc[0] * shape_yxc[1] * channels * bytes_per_pixel
                    )
                    if store_tissue_mask:
                        projection_baseline_chunks += level_rgb_chunks
                        projection_baseline_bytes += int(shape_yxc[0] * shape_yxc[1])
        else:
            num_mips = compute_num_mips_min_side(canvas.w, canvas.h, min_side)
            shapes_yxc = _pyramid_shapes_yxc(
                base_shape_yxc=(canvas.h, canvas.w, channels),
                num_mips=num_mips,
            )
            mip_stop_reason = "min_side_for_mips"
        num_mips = len(shapes_yxc)
        s0_canvas = native_specs[0].canvas_source_yx if native_specs else canvas
        mask_shapes_yx = [(y, x) for y, x, _c in shapes_yxc]
        s0_chunks = _chunk_count_yx(y=shapes_yxc[0][0], x=shapes_yxc[0][1], chunk_xy=plate_chunk_xy)
        all_mip_chunks = sum(
            _chunk_count_yx(y=y, x=x, chunk_xy=plate_chunk_xy) for y, x, _c in shapes_yxc
        )
        mask_mip_chunks = (
            sum(_chunk_count_yx(y=y, x=x, chunk_xy=plate_chunk_xy) for y, x in mask_shapes_yx)
            if store_tissue_mask
            else 0
        )
        bytes_all_mips = _estimate_pyramid_bytes(
            shapes_yxc=shapes_yxc,
            bytes_per_pixel=bytes_per_pixel,
        )
        mask_bytes_all_mips = int(sum(y * x for y, x in mask_shapes_yx)) if store_tissue_mask else 0
        if primary_rgb_mode == "masked_rgb" or store_tissue_mask:
            mask_activity = {
                "mask_empty_chunks": 0,
                "mask_positive_chunks": 0,
                "rgb_chunks_skippable_before_decode": 0,
            }
            mask_levels = native_specs if native_specs else [None]
            for native_level in mask_levels:
                activity = _estimate_mask_s0_chunk_activity(
                    lr_labels=lr_labels,
                    label_id=int(spec.label_id),
                    label_crop_seg_yx=spec.label_crop_seg_yx,
                    source_shape_yx=(
                        native_level.source_shape_yx
                        if native_level is not None
                        else source_shape_yx
                    ),
                    canvas=native_level.canvas_source_yx if native_level is not None else canvas,
                    chunk_xy=plate_chunk_xy,
                )
                for key in mask_activity:
                    mask_activity[key] += int(activity[key])
        else:
            mask_activity = {
                "mask_empty_chunks": 0,
                "mask_positive_chunks": 0,
                "rgb_chunks_skippable_before_decode": 0,
            }
        if not (primary_rgb_mode == "masked_rgb" and masked_rgb_fill_value == 0):
            mask_activity["rgb_chunks_skippable_before_decode"] = 0
        warning = _estimate_warning(bytes_all_mips=bytes_all_mips, s0_chunks=s0_chunks)
        if warning is not None:
            warnings.append(f"tissue_{spec.tissue_index:02d}:{warning}")

        total_s0_chunks += s0_chunks
        total_all_mip_chunks += all_mip_chunks
        total_mask_all_mip_chunks += mask_mip_chunks
        total_bytes += bytes_all_mips
        total_mask_bytes += mask_bytes_all_mips
        total_projection_baseline_chunks += int(projection_baseline_chunks)
        total_projection_baseline_bytes += int(projection_baseline_bytes)
        total_mask_empty_chunks += int(mask_activity["mask_empty_chunks"])
        total_mask_positive_chunks += int(mask_activity["mask_positive_chunks"])
        total_rgb_chunks_skippable_before_decode += int(
            mask_activity["rgb_chunks_skippable_before_decode"]
        )
        total_combined_bytes = bytes_all_mips + mask_bytes_all_mips
        tissues.append(
            {
                "tissue_index": int(spec.tissue_index),
                "label_id": int(spec.label_id),
                "logical_canvas_source_yx": spec.logical_canvas_source_yx.as_dict(),
                "clipped_source_yx": spec.clipped_source_yx.as_dict(),
                "padding_source_level": spec.padding_source_level.as_dict(),
                "logical_frame_segmentation_yx": spec.logical_frame_seg_yx.as_dict(),
                "clipped_frame_segmentation_yx": spec.clipped_frame_seg_yx.as_dict(),
                "crop_shape_policy": crop_shape_policy,
                "context_margin_px": int(context_margin_px),
                "source_tile_aligned_canvas": bool(source_tile_aligned_canvas),
                "source_tile_alignment_expansion_yx": source_tile_alignment_expansion_yx,
                "tile_shape_yx": [int(shapes_yxc[0][0]), int(shapes_yxc[0][1])],
                "s0_shape_yx": [int(shapes_yxc[0][0]), int(shapes_yxc[0][1])],
                "child_origin_in_parent_source_level": [int(s0_canvas.x0), int(s0_canvas.y0)],
                "source_tile_dim": int(spec.source_canvas_dim),
                "segmentation_tile_dim": int(spec.segmentation_tile_dim),
                "effective_segmentation_tile_dim": float(
                    spec.source_canvas_dim / max(spec.scale_y, spec.scale_x)
                ),
                "scale_y": float(spec.scale_y),
                "scale_x": float(spec.scale_x),
                "s0_shape_yxc": [int(shapes_yxc[0][0]), int(shapes_yxc[0][1]), channels],
                "mask_s0_shape_yx": (
                    [int(shapes_yxc[0][0]), int(shapes_yxc[0][1])] if store_tissue_mask else None
                ),
                "s0_chunks": int(s0_chunks),
                "mask_empty_chunks": int(mask_activity["mask_empty_chunks"]),
                "mask_positive_chunks": int(mask_activity["mask_positive_chunks"]),
                "rgb_chunks_skippable_before_decode": int(
                    mask_activity["rgb_chunks_skippable_before_decode"]
                ),
                "expected_sparse_zero_behavior": (
                    "empty_mask_chunks_skip_rgb_decode_and_write"
                    if primary_rgb_mode == "masked_rgb" and masked_rgb_fill_value == 0
                    else "post_decode_sparse_zero_only"
                ),
                "num_mips": int(num_mips),
                "per_scale_shapes_yx": per_scale_shapes_yx,
                "output_scale_to_source_level": output_scale_to_source_level,
                "native_pyramid_levels": per_scale_native_levels,
                "native_mip_stop_policy": tissue_native_mip_stop_policy,
                "native_mip_stop_level": tissue_native_mip_stop_level,
                "mip_stop_reason": mip_stop_reason,
                "coarsest_segmentation_level_not_written": bool(
                    coarsest_segmentation_level_not_written
                ),
                "mip_stop_warnings": mip_stop_warnings,
                "mip_shapes_yxc": [list(map(int, shape)) for shape in shapes_yxc],
                "all_mip_chunks": int(all_mip_chunks),
                "mask_all_mip_chunks": int(mask_mip_chunks),
                "combined_logical_chunks": int(all_mip_chunks + mask_mip_chunks),
                "rgb_uncompressed_bytes_all_mips": int(bytes_all_mips),
                "mask_uncompressed_bytes_all_mips": int(mask_bytes_all_mips),
                "total_uncompressed_bytes_rgb_plus_mask": int(total_combined_bytes),
                "rgb_uncompressed_size_all_mips": _human_bytes(bytes_all_mips),
                "mask_uncompressed_size_all_mips": _human_bytes(mask_bytes_all_mips),
                "total_uncompressed_size_rgb_plus_mask": _human_bytes(total_combined_bytes),
                # Legacy compatibility: RGB-only bytes.
                "uncompressed_bytes_all_mips": int(bytes_all_mips),
                # Legacy compatibility: combined RGB + mask bytes.
                "uncompressed_bytes_estimate": int(total_combined_bytes),
                "uncompressed_size_all_mips": _human_bytes(bytes_all_mips),
                "compressed_bytes_sample_estimate": None,
                "compressed_estimate_method": None,
                "warning": warning,
            }
        )

    total_warning = _estimate_warning(
        bytes_all_mips=total_bytes,
        s0_chunks=total_s0_chunks,
    )
    if total_warning is not None:
        warnings.append(f"total:{total_warning}")
    warnings.extend(
        _baseline_config_mismatch_warnings(
            source_level=int(source_idx),
            segmentation_level=int(segmentation_idx),
            primary_rgb_mode=primary_rgb_mode,
            masked_rgb_fill_value=int(masked_rgb_fill_value),
            extra_margin_px=int(tile_extra_margin_px),
            pyramid_generation_policy=pyramid_generation_policy,
            source_tile_aligned_canvas=bool(source_tile_aligned_canvas),
            native_mip_stop_policy=native_mip_stop_policy,
            store_tissue_mask=bool(store_tissue_mask),
            sparse_zero_chunks=bool(sparse_zero_chunks),
            compression=compression,
        )
    )
    projection_baseline_totals = (
        {
            "combined_logical_chunks": int(total_projection_baseline_chunks),
            "total_uncompressed_bytes_rgb_plus_mask": int(total_projection_baseline_bytes),
        }
        if pyramid_generation_policy == "native_source_pyramid_crop"
        and total_projection_baseline_chunks > 0
        else None
    )
    projection = _projection_from_validated_source_level2(
        estimate_totals={
            "combined_logical_chunks": int(total_all_mip_chunks + total_mask_all_mip_chunks),
            "total_uncompressed_bytes_rgb_plus_mask": int(total_bytes + total_mask_bytes),
        },
        baseline_totals=projection_baseline_totals,
    )
    native_tissues = [t for t in tissues if t.get("native_mip_stop_level") is not None]
    top_native_mip_stop_level = (
        int(native_tissues[0]["native_mip_stop_level"]) if native_tissues else None
    )
    top_mip_stop_reason = native_tissues[0]["mip_stop_reason"] if native_tissues else None
    if native_tissues and any(
        int(t["native_mip_stop_level"]) != top_native_mip_stop_level
        or t["mip_stop_reason"] != top_mip_stop_reason
        for t in native_tissues
    ):
        top_mip_stop_reason = "mixed"
    top_coarsest_segmentation_level_not_written = any(
        bool(t.get("coarsest_segmentation_level_not_written")) for t in native_tissues
    )

    result = {
        "vsi_path": str(vsi_path),
        "ets_path": str(ets_path),
        "metadata_backend": metadata_backend,
        "metadata_schema": metadata_schema,
        "source_level": int(source_idx),
        "segmentation_level": int(segmentation_idx),
        "output_profile": output_profile,
        "tile_frame_level": tile_frame_level,
        "crop_shape_policy": crop_shape_policy,
        "compression": _compression_descriptor(compression),
        "pyramid_generation_policy": pyramid_generation_policy,
        "source_tile_aligned_canvas": bool(source_tile_aligned_canvas),
        "native_mip_stop_policy": native_mip_stop_policy,
        "native_mip_stop_level": top_native_mip_stop_level,
        "native_mip_stop_level_source": native_mip_stop_policy,
        "mip_stop_reason": top_mip_stop_reason,
        "coarsest_segmentation_level_not_written": bool(
            top_coarsest_segmentation_level_not_written
        ),
        "store_tissue_mask": bool(store_tissue_mask),
        "primary_rgb_mode": primary_rgb_mode,
        "masked_rgb_fill_value": int(masked_rgb_fill_value),
        "mask_applied_to_primary_rgb": primary_rgb_mode == "masked_rgb",
        "store_unmasked_rgb": bool(store_unmasked_rgb),
        "materialize_masked_rgb": bool(materialize_masked_rgb),
        "primary_rgb_mode_resolution_source": rgb_options["primary_rgb_mode_resolution_source"],
        "materialize_masked_rgb_deprecated_alias_used": bool(
            rgb_options["materialize_masked_rgb_deprecated_alias_used"]
        ),
        "masked_rgb_pyramid_semantics": (
            "mask_projected_per_scale"
            if pyramid_generation_policy == "native_source_pyramid_crop"
            and primary_rgb_mode == "masked_rgb"
            else (
                "masked_s0_then_downsampled"
                if primary_rgb_mode == "masked_rgb"
                else "not_applicable"
            )
        ),
        "sparse_zero_chunks": bool(sparse_zero_chunks),
        "config_source": config_source or "programmatic/default",
        "context_margin_px": int(tile_extra_margin_px),
        "effective_extra_margin_px": int(tile_extra_margin_px),
        "source_shape_yx": list(source_shape_yx),
        "segmentation_shape_yx": list(map(int, lr_labels.shape)),
        "source_physical_pixel_size": {
            "x": float(base_phys_xy_um[0]) * (2**source_idx),
            "y": float(base_phys_xy_um[1]) * (2**source_idx),
            "unit": "micrometer",
        },
        "chunk_xy": int(plate_chunk_xy),
        "pad_multiple": int(tile_pad_multiple),
        "extra_margin_px": int(tile_extra_margin_px),
        "dtype": str(dtype_obj),
        "bytes_per_pixel": bytes_per_pixel,
        "tissue_count": len(tissues),
        "tile_dim": int(tile_dim),
        "tissues": tissues,
        "totals": {
            "s0_chunks": int(total_s0_chunks),
            "mask_empty_chunks": int(total_mask_empty_chunks),
            "mask_positive_chunks": int(total_mask_positive_chunks),
            "rgb_chunks_skippable_before_decode": int(total_rgb_chunks_skippable_before_decode),
            "expected_sparse_zero_behavior": (
                "empty_mask_chunks_skip_rgb_decode_and_write"
                if primary_rgb_mode == "masked_rgb" and masked_rgb_fill_value == 0
                else "post_decode_sparse_zero_only"
            ),
            "all_mip_chunks": int(total_all_mip_chunks),
            "mask_all_mip_chunks": int(total_mask_all_mip_chunks),
            "combined_logical_chunks": int(total_all_mip_chunks + total_mask_all_mip_chunks),
            "rgb_uncompressed_bytes_all_mips": int(total_bytes),
            "mask_uncompressed_bytes_all_mips": int(total_mask_bytes),
            "total_uncompressed_bytes_rgb_plus_mask": int(total_bytes + total_mask_bytes),
            "rgb_uncompressed_size_all_mips": _human_bytes(total_bytes),
            "mask_uncompressed_size_all_mips": _human_bytes(total_mask_bytes),
            "total_uncompressed_size_rgb_plus_mask": _human_bytes(total_bytes + total_mask_bytes),
            # Legacy compatibility: RGB-only bytes.
            "uncompressed_bytes_all_mips": int(total_bytes),
            # Legacy compatibility: combined RGB + mask bytes.
            "uncompressed_bytes_estimate": int(total_bytes + total_mask_bytes),
            "uncompressed_size_all_mips": _human_bytes(total_bytes),
            "compressed_bytes_sample_estimate": None,
            "compressed_estimate_method": None,
            "projection_baseline_logical_chunks": int(total_projection_baseline_chunks),
            "projection_baseline_uncompressed_bytes_rgb_plus_mask": int(
                total_projection_baseline_bytes
            ),
            "projected_source_level0_from_validated_source_level2": projection,
            "warnings": warnings,
        },
        "projection": projection,
        "validated_source_level2_baseline": _VALIDATED_SOURCE_LEVEL2_NATIVE_BASELINE,
        "segmentation_info": segmentation_info,
        "elapsed_s": time.monotonic() - started,
    }
    return _json_ready(result)


def vsi_to_source_ome_zarr(
    vsi_path: str | Path,
    output_path: str | Path,
    *,
    metadata_backend: str = "auto",
    metadata_schema: str = "v0.4",
    chunks_xy: int = 512,
    overwrite: bool = False,
    source_writer: str = "direct",
) -> tuple[Path, Path, dict[str, Any]]:
    """
    Convert a VSI/ETS pyramid to a source OME-Zarr pyramid without flat files.

    Returns ``(source_ome_zarr_path, ets_path, vsi_metadata)``.
    """
    vsi_path = Path(vsi_path)
    output_path = Path(output_path)
    ets_path = find_ets_file(vsi_path)
    if ets_path is None:
        raise FileNotFoundError(f"No ETS file found for VSI {vsi_path}")

    metadata = get_vsi_metadata(
        vsi_path,
        metadata_backend=metadata_backend,
        target_schema="latest",
    )
    if not metadata:
        raise RuntimeError(f"Unable to extract structural metadata for VSI {vsi_path}")

    if output_path.exists() and not overwrite:
        missing = _missing_source_ome_zarr_arrays(output_path, metadata)
        if missing:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview += f", ... ({len(missing)} missing total)"
            raise RuntimeError(
                f"Existing source OME-Zarr at {output_path} appears incomplete; "
                f"missing dataset array(s): {preview}. "
                "Rerun with overwrite_source=True or choose a fresh output directory."
            )
        shape_errors = _source_ome_zarr_shape_errors(output_path, metadata, Path(ets_path))
        if shape_errors:
            preview = "; ".join(shape_errors[:3])
            if len(shape_errors) > 3:
                preview += f"; ... ({len(shape_errors)} shape mismatches total)"
            raise RuntimeError(
                f"Existing source OME-Zarr at {output_path} does not match the ETS pyramid; "
                f"{preview}. Rerun with overwrite_source=True or choose a fresh output directory."
            )
        return output_path, Path(ets_path), metadata

    channel_labels = default_channel_labels(3)
    channel_colors = default_channel_colors(3)
    phys_xy_um = _physical_xy_from_metadata(metadata)
    if phys_xy_um is None:
        logger.warning(
            "Physical pixel size unavailable for %s; source OME-Zarr will use 1.0 um fallback scales.",
            vsi_path,
        )
        phys_xy_um = (1.0, 1.0)

    writer_name = source_writer.strip().lower().replace("_", "-")
    if writer_name not in {"direct", "ngff-zarr"}:
        raise ValueError("source_writer must be one of ['direct', 'ngff-zarr'].")

    temp_output_path = _temporary_source_path(output_path)
    if temp_output_path.exists():
        shutil.rmtree(temp_output_path)

    if writer_name == "direct":
        write_ets_pyramid_to_ngff_zarr(
            ets_path,
            temp_output_path,
            phys_xy_um=phys_xy_um,
            name=vsi_path.stem,
            chunks_xy=chunks_xy,
            overwrite=True,
            channel_labels=channel_labels,
            channel_colors=channel_colors,
            add_omero=True,
            ngff_metadata=metadata,
            metadata_schema=metadata_schema,
        )
    else:
        ets = ETSFile(ets_path)
        try:
            levels_yxc = [ets.to_dask(level) for level in range(ets.nlevels)]
            write_ngff_from_mips_ngffzarr(
                mips_yxc=levels_yxc,
                out_dir=temp_output_path,
                phys_xy_um=phys_xy_um,
                name=vsi_path.stem,
                chunks_xy=chunks_xy,
                version="0.4",
                overwrite=True,
                channel_labels=channel_labels,
                channel_colors=channel_colors,
                add_omero=True,
                ngff_metadata=metadata,
                metadata_schema=metadata_schema,
            )
        finally:
            ets.close()

    _promote_completed_source(temp_output_path, output_path)

    return output_path, Path(ets_path), metadata


def process_vsi_with_direct_plating(
    vsi_path: str | Path,
    out_ngff_dir: str | Path,
    *,
    segment_fn=None,
    source_level: int | str = 0,
    segmentation_level: int | str | None = None,
    output_profile: str = "validation",
    tile_frame_level: str = "segmentation",
    crop_shape_policy: str | None = None,
    segmentation_config: SegmentationConfig | None = None,
    tile_config: TileConfig | None = None,
    metadata_backend: str = "auto",
    metadata_schema: str = "v0.4",
    struct_elem_px: int = 9,
    min_size: int = 2000,
    precomputed_plate_path: str | None = None,
    plate_backend: str = "tensorstore",
    plate_chunk_xy: int = 512,
    parallel: bool = False,
    fill_missing: bool = False,
    min_side_for_mips: int | None = None,
    requested_mips: int | None = None,
    tile_extra_margin_px: int = 0,
    dtype: np.dtype | str | None = "uint8",
    compression: str | None = None,
    store_tissue_mask: bool | None = None,
    primary_rgb_mode: str | None = None,
    masked_rgb_fill_value: int | None = None,
    store_unmasked_rgb: bool | None = None,
    materialize_masked_rgb: bool | None = None,
    sparse_zero_chunks: bool | None = None,
    pyramid_generation_policy: str | None = None,
    source_tile_aligned_canvas: bool | None = None,
    native_mip_stop_policy: str | None = None,
    native_mip_stop_level: int | str | None = None,
    resume: bool = False,
    progress_mode: str | bool | None = "none",
    progress_interval_s: float = 30.0,
) -> list[Path]:
    """
    Segment a VSI/ETS pyramid and write per-tissue OME-Zarr derivatives without
    materializing a full-slide source OME-Zarr.
    """
    del parallel  # Direct ETS block reads are deliberately scheduled locally.
    vsi_path = Path(vsi_path)
    out_ngff_dir = Path(out_ngff_dir)
    out_ngff_dir.mkdir(parents=True, exist_ok=True)
    output_profile = _normalize_output_profile(output_profile)
    defaults = _profile_defaults(output_profile)
    if crop_shape_policy is None:
        crop_shape_policy = str(defaults["crop_shape_policy"])
    if compression is None:
        compression = str(defaults["compression"])
    if store_tissue_mask is None:
        store_tissue_mask = bool(defaults["store_tissue_mask"])
    rgb_options = _resolve_primary_rgb_options(
        primary_rgb_mode=primary_rgb_mode,
        materialize_masked_rgb=materialize_masked_rgb,
        masked_rgb_fill_value=masked_rgb_fill_value,
        store_unmasked_rgb=store_unmasked_rgb,
        defaults=defaults,
    )
    primary_rgb_mode = str(rgb_options["primary_rgb_mode"])
    materialize_masked_rgb = bool(rgb_options["materialize_masked_rgb"])
    masked_rgb_fill_value = int(rgb_options["masked_rgb_fill_value"])
    store_unmasked_rgb = bool(rgb_options["store_unmasked_rgb"])
    if sparse_zero_chunks is None:
        sparse_zero_chunks = bool(defaults["sparse_zero_chunks"])
    if pyramid_generation_policy is None:
        pyramid_generation_policy = str(
            defaults.get("pyramid_generation_policy", "downsample_streamed_s0")
        )
    if source_tile_aligned_canvas is None:
        source_tile_aligned_canvas = bool(defaults.get("source_tile_aligned_canvas", False))
    if native_mip_stop_policy is None:
        native_mip_stop_policy = str(defaults.get("native_mip_stop_policy", "segmentation_level"))
    if native_mip_stop_level is None:
        native_mip_stop_level = defaults.get("native_mip_stop_level", "segmentation_level")
    tile_frame_level = _normalize_tile_frame_level(tile_frame_level)
    crop_shape_policy = _normalize_crop_shape_policy(crop_shape_policy)
    compression = _normalize_compression_mode(compression)
    pyramid_generation_policy = _normalize_pyramid_generation_policy(pyramid_generation_policy)
    native_mip_stop_policy = _normalize_native_mip_stop_policy(
        native_mip_stop_policy,
        native_mip_stop_level=native_mip_stop_level,
    )
    compressor = _compressor_for_compression_mode(compression)

    ets_path = find_ets_file(vsi_path)
    if ets_path is None:
        raise FileNotFoundError(f"No ETS file found for VSI {vsi_path}")
    ets_path = Path(ets_path)

    metadata = get_vsi_metadata(
        vsi_path,
        metadata_backend=metadata_backend,
        target_schema="latest",
    )
    if not metadata:
        raise RuntimeError(f"Unable to extract structural metadata for VSI {vsi_path}")

    if tile_config is not None:
        plate_chunk_xy = tile_config.chunk_size
        tile_extra_margin_px = tile_config.extra_margin_px
        tile_pad_multiple = tile_config.pad_multiple
        if output_profile in {"production", "upload_staging"} and tile_extra_margin_px == 0:
            tile_extra_margin_px = int(defaults["extra_margin_px"])
        if crop_shape_policy == "notebook_square":
            crop_shape_policy = _normalize_crop_shape_policy(tile_config.crop_shape_policy)
    else:
        tile_extra_margin_px = int(defaults["extra_margin_px"])
        tile_pad_multiple = plate_chunk_xy

    if precomputed_plate_path and crop_shape_policy == "compact_rectangle":
        raise ValueError(
            "compact_rectangle outputs are variable-shape and cannot be written as a single plate stack."
        )
    if precomputed_plate_path and pyramid_generation_policy == "native_source_pyramid_crop":
        raise ValueError(
            "native_source_pyramid_crop does not currently support precomputed plate output."
        )

    with ETSFile(ets_path) as ets:
        source_idx = _resolve_ets_level(
            source_level,
            default_index=0,
            nlevels=ets.nlevels,
            label="source",
        )
        segmentation_idx = _resolve_ets_level(
            segmentation_level,
            default_index=ets.nlevels - 1,
            nlevels=ets.nlevels,
            label="segmentation",
        )
        source_shape_yx = tuple(map(int, ets.level_shape(source_idx)))
        segmentation_yxc = ets.read_level(segmentation_idx)

    logger.info(
        "Resolved ETS levels for %s: source level %d shape=%s; segmentation level %d shape=%s.",
        vsi_path.name,
        source_idx,
        source_shape_yx,
        segmentation_idx,
        tuple(map(int, segmentation_yxc.shape)),
    )

    seg_cyx = da.from_array(
        np.moveaxis(segmentation_yxc, -1, 0),
        chunks=(
            3,
            min(plate_chunk_xy, segmentation_yxc.shape[0]),
            min(plate_chunk_xy, segmentation_yxc.shape[1]),
        ),
    )
    filled_lr, _ = _segment_for_plating(
        seg_cyx,
        segment_fn=segment_fn,
        segmentation_config=segmentation_config,
        min_size=min_size,
        struct_elem_px=struct_elem_px,
    )

    tile_records, tile_dim = _direct_ets_tissue_tile_records(
        ets_path=ets_path,
        source_level=source_idx,
        source_shape_yx=source_shape_yx,
        low_res_filled=filled_lr.astype(bool),
        chunk=plate_chunk_xy,
        pad_multiple=tile_pad_multiple,
        extra_margin_px=tile_extra_margin_px,
        tile_frame_level=tile_frame_level,
        crop_shape_policy=crop_shape_policy,
        materialize_masked_rgb=bool(materialize_masked_rgb),
        masked_rgb_fill_value=int(masked_rgb_fill_value),
    )
    if not tile_records:
        logger.warning("[%s] no tissue regions found.", vsi_path.name)
        return []

    base_phys_xy_um = _physical_xy_from_metadata(metadata)
    if base_phys_xy_um is None:
        logger.warning(
            "Physical pixel size unavailable for %s; tissue manifests will use 1.0 um fallback scales.",
            vsi_path,
        )
        base_phys_xy_um = (1.0, 1.0)
    px_um = float(base_phys_xy_um[0]) * (2**source_idx)
    py_um = float(base_phys_xy_um[1]) * (2**source_idx)

    source_context = {
        "source_kind": "vsi",
        "source_path": str(vsi_path),
        "source_vsi": str(vsi_path),
        "source_ets": str(ets_path),
        "source_ome_zarr": None,
        "ngff_metadata": metadata,
        "metadata_schema": metadata_schema,
        "metadata_backend": metadata_backend,
        "output_profile": output_profile,
        "crop_shape_policy": crop_shape_policy,
        "pyramid_generation_policy": pyramid_generation_policy,
        "source_tile_aligned_canvas": bool(source_tile_aligned_canvas),
        "primary_rgb_mode": primary_rgb_mode,
        "masked_rgb_fill_value": int(masked_rgb_fill_value),
        "mask_applied_to_primary_rgb": primary_rgb_mode == "masked_rgb",
        "store_tissue_mask": bool(store_tissue_mask),
        "store_unmasked_rgb": bool(store_unmasked_rgb),
        "primary_rgb_mode_resolution_source": rgb_options["primary_rgb_mode_resolution_source"],
        "materialize_masked_rgb_deprecated_alias_used": bool(
            rgb_options["materialize_masked_rgb_deprecated_alias_used"]
        ),
        "masked_rgb_pyramid_semantics": (
            "mask_projected_per_scale"
            if pyramid_generation_policy == "native_source_pyramid_crop"
            and primary_rgb_mode == "masked_rgb"
            else (
                "masked_s0_then_downsampled"
                if primary_rgb_mode == "masked_rgb"
                else "not_applicable"
            )
        ),
        "requested_mips": int(requested_mips) if requested_mips is not None else None,
        "native_mip_stop_policy": native_mip_stop_policy,
        "native_mip_stop_level": native_mip_stop_level,
        "native_mip_stop_level_source": native_mip_stop_policy,
    }
    source_ngff_metadata = metadata
    source_metadata_schema = metadata_schema

    plate = None
    if precomputed_plate_path:
        plate = PlatePrecomputedWriter(
            precomp_path=precomputed_plate_path,
            width=tile_dim,
            height=tile_dim,
            z_slices=len(tile_records),
            voxel_size_um=(px_um, py_um, 1.0),
            chunk_xy=plate_chunk_xy,
            min_side_for_mips=min_side_for_mips,
            backend=plate_backend,
            dtype=dtype if dtype else "uint8",
            encoding="raw",
            parallel=False,
            fill_missing=fill_missing,
        )

    out_paths: list[Path] = []
    big_tile_threshold = 8192
    item_size_threshold = 1_500_000_000
    bytes_per_px = np.dtype(dtype if dtype else np.uint8).itemsize
    any_big = any(
        _is_big_tile(
            tile_da=record.tile,
            bytes_per_px=bytes_per_px,
            min_side=big_tile_threshold,
            max_bytes=item_size_threshold,
        )
        for record in tile_records
    )

    _safe_close_existing_client()
    run_started = time.monotonic()
    with dask.config.set({"scheduler": "threads", "array.slicing.split_large_chunks": True}):
        for z_idx, record in enumerate(tile_records):
            tile_dask = record.tile
            name = f"{vsi_path.stem}_tissue_{record.tissue_index:02d}"
            ngff_dir = out_ngff_dir / f"{name}.ome.zarr"
            tissue_started = time.monotonic()
            tissue_source_context = source_context
            if resume and ngff_dir.exists():
                logger.info("Skipping completed tissue output because resume=True: %s", ngff_dir)
                _record_written_tissue(
                    out_paths,
                    ngff_dir,
                    record=record,
                    source_context=source_context,
                    source_ome_zarr=None,
                    source_level=source_idx,
                    segmentation_level=segmentation_idx,
                    phys_xy_um=(px_um, py_um),
                )
                continue
            logger.info(
                "Writing direct ETS tissue %d/%d: %s shape=%s compression=%s.",
                z_idx + 1,
                len(tile_records),
                name,
                tuple(map(int, tile_dask.shape)),
                compression,
            )
            force_streaming = bool(
                store_tissue_mask or sparse_zero_chunks or crop_shape_policy != "notebook_square"
            )
            if pyramid_generation_policy == "native_source_pyramid_crop":
                work_dir = ngff_dir.with_name(f".{ngff_dir.name}.incomplete")
                if work_dir.exists() and not resume:
                    shutil.rmtree(work_dir)
                if ngff_dir.exists() and not resume:
                    shutil.rmtree(ngff_dir)
                work_dir.mkdir(parents=True, exist_ok=True)
                origin_y = int(
                    (record.frame_debug or {}).get("logical_canvas_source_yx", {}).get("y0", 0)
                )
                origin_x = int(
                    (record.frame_debug or {}).get("logical_canvas_source_yx", {}).get("x0", 0)
                )
                run_manifest = {
                    "output_profile": output_profile,
                    "crop_shape_policy": crop_shape_policy,
                    "compression": _compression_descriptor(compression),
                    "sparse_zero_chunks": bool(sparse_zero_chunks),
                    "store_tissue_mask": bool(store_tissue_mask),
                    "materialize_masked_rgb": bool(materialize_masked_rgb),
                    "primary_rgb_mode": primary_rgb_mode,
                    "masked_rgb_fill_value": int(masked_rgb_fill_value),
                    "mask_applied_to_primary_rgb": primary_rgb_mode == "masked_rgb",
                    "store_unmasked_rgb": bool(store_unmasked_rgb),
                    "primary_rgb_mode_resolution_source": rgb_options[
                        "primary_rgb_mode_resolution_source"
                    ],
                    "materialize_masked_rgb_deprecated_alias_used": bool(
                        rgb_options["materialize_masked_rgb_deprecated_alias_used"]
                    ),
                    "pyramid_generation_policy": "native_source_pyramid_crop",
                    "rgb_pyramid_semantics": "native_scanner_pyramid",
                    "reference_policy": "downsample_streamed_s0",
                    "masked_rgb_pyramid_semantics": (
                        "mask_projected_per_scale"
                        if primary_rgb_mode == "masked_rgb"
                        else "not_applicable"
                    ),
                    "source_tile_aligned_canvas": bool(source_tile_aligned_canvas),
                    "requested_mips": int(requested_mips) if requested_mips is not None else None,
                    "native_mip_stop_policy": native_mip_stop_policy,
                    "native_mip_stop_level": native_mip_stop_level,
                    "native_mip_stop_level_source": native_mip_stop_policy,
                    "status": "running",
                    "started_at_unix": time.time(),
                    "source_level_origin_yx": [origin_y, origin_x],
                }
                (work_dir / "run_manifest.json").write_text(
                    json.dumps(_json_ready(run_manifest), indent=2),
                    encoding="utf-8",
                )
                native_stats = write_native_ets_tissue_pyramid_ome(
                    ets_path=ets_path,
                    out_dir=work_dir,
                    record=record,
                    lr_labels=_filled_lr_labels(np.asarray(filled_lr, dtype=bool)),
                    source_level=source_idx,
                    source_shape_yx=source_shape_yx,
                    source_phys_xy_um=(px_um, py_um),
                    block_xy=plate_chunk_xy,
                    name=name,
                    compressor=compressor,
                    sparse_zero_chunks=bool(sparse_zero_chunks),
                    store_tissue_mask=bool(store_tissue_mask),
                    metadata_schema=source_metadata_schema,
                    min_side_for_mips=min_side_for_mips,
                    requested_mips=requested_mips,
                    segmentation_level=int(segmentation_idx),
                    native_mip_stop_policy=native_mip_stop_policy,
                    native_mip_stop_level=native_mip_stop_level,
                    native_mip_stop_source=native_mip_stop_policy,
                    source_tile_aligned_canvas=bool(source_tile_aligned_canvas),
                    primary_rgb_mode=primary_rgb_mode,
                    masked_rgb_fill_value=int(masked_rgb_fill_value),
                    store_unmasked_rgb=bool(store_unmasked_rgb),
                    channel_labels=default_channel_labels(int(tile_dask.shape[2])),
                    channel_colors=default_channel_colors(int(tile_dask.shape[2])),
                    run_manifest=run_manifest,
                )
                run_manifest.update(native_stats)
                tissue_source_context = {
                    **source_context,
                    "pyramid_generation_policy": "native_source_pyramid_crop",
                    "rgb_pyramid_semantics": "native_scanner_pyramid",
                    "reference_policy": "downsample_streamed_s0",
                    "source_tile_aligned_canvas": bool(source_tile_aligned_canvas),
                    "canonical_canvas_in_source_level_coordinates": native_stats.get(
                        "canonical_canvas_in_source_level_coordinates"
                    ),
                    "output_scale_to_source_level": native_stats.get(
                        "output_scale_to_source_level"
                    ),
                    "native_mip_stop_policy": native_stats.get("native_mip_stop_policy"),
                    "native_mip_stop_level": native_stats.get("native_mip_stop_level"),
                    "mip_stop_reason": native_stats.get("mip_stop_reason"),
                    "coarsest_segmentation_level_not_written": native_stats.get(
                        "coarsest_segmentation_level_not_written"
                    ),
                    "native_pyramid_levels": native_stats.get("native_pyramid_levels"),
                    "mask_generation_policy": native_stats.get("mask_generation_policy"),
                    "mask_pyramid_semantics": native_stats.get("mask_pyramid_semantics"),
                    "primary_rgb_mode": native_stats.get("primary_rgb_mode"),
                    "masked_rgb_fill_value": native_stats.get("masked_rgb_fill_value"),
                    "mask_applied_to_primary_rgb": native_stats.get("mask_applied_to_primary_rgb"),
                    "masked_rgb_pyramid_semantics": native_stats.get(
                        "masked_rgb_pyramid_semantics"
                    ),
                    "store_tissue_mask": native_stats.get("store_tissue_mask"),
                    "store_unmasked_rgb": native_stats.get("store_unmasked_rgb"),
                    "primary_rgb_mode_resolution_source": rgb_options[
                        "primary_rgb_mode_resolution_source"
                    ],
                    "materialize_masked_rgb_deprecated_alias_used": bool(
                        rgb_options["materialize_masked_rgb_deprecated_alias_used"]
                    ),
                    "mask_empty_chunks": native_stats.get("mask_empty_chunks"),
                    "mask_positive_chunks": native_stats.get("mask_positive_chunks"),
                    "rgb_chunks_skipped_before_decode": native_stats.get(
                        "rgb_chunks_skipped_before_decode"
                    ),
                    "rgb_write_amplification": native_stats.get("rgb_write_amplification"),
                    "mask_write_amplification": native_stats.get("mask_write_amplification"),
                    "rgb_chunk_write_calls": native_stats.get("rgb_chunk_write_calls"),
                    "unique_rgb_chunks_written": native_stats.get("unique_rgb_chunks_written"),
                    "mask_chunk_write_calls": native_stats.get("mask_chunk_write_calls"),
                    "unique_mask_chunks_written": native_stats.get("unique_mask_chunks_written"),
                }
                run_manifest["status"] = "complete"
                run_manifest["finished_at_unix"] = time.time()
                (work_dir / "run_manifest.json").write_text(
                    json.dumps(_json_ready(run_manifest), indent=2),
                    encoding="utf-8",
                )
                if ngff_dir.exists():
                    shutil.rmtree(ngff_dir)
                work_dir.rename(ngff_dir)
            elif any_big or force_streaming:
                num_mips = compute_num_mips_min_side(
                    int(tile_dask.shape[1]),
                    int(tile_dask.shape[0]),
                    min_side_for_mips or plate_chunk_xy,
                )
                tile_ngff_metadata = _tile_ngff_metadata_or_none(
                    source_ngff_metadata,
                    dataset_count=num_mips,
                    name=name,
                    phys_xy_um=(px_um, py_um),
                )
                tlazy = tile_dask.astype(np.uint8) if tile_dask.dtype != np.uint8 else tile_dask
                work_dir = ngff_dir.with_name(f".{ngff_dir.name}.incomplete")
                if work_dir.exists() and not resume:
                    shutil.rmtree(work_dir)
                if ngff_dir.exists() and not resume:
                    shutil.rmtree(ngff_dir)
                work_dir.mkdir(parents=True, exist_ok=True)
                run_manifest = {
                    "output_profile": output_profile,
                    "crop_shape_policy": crop_shape_policy,
                    "compression": _compression_descriptor(compression),
                    "sparse_zero_chunks": bool(sparse_zero_chunks),
                    "store_tissue_mask": bool(store_tissue_mask),
                    "materialize_masked_rgb": bool(materialize_masked_rgb),
                    "primary_rgb_mode": primary_rgb_mode,
                    "masked_rgb_fill_value": int(masked_rgb_fill_value),
                    "mask_applied_to_primary_rgb": primary_rgb_mode == "masked_rgb",
                    "store_unmasked_rgb": bool(store_unmasked_rgb),
                    "primary_rgb_mode_resolution_source": rgb_options[
                        "primary_rgb_mode_resolution_source"
                    ],
                    "materialize_masked_rgb_deprecated_alias_used": bool(
                        rgb_options["materialize_masked_rgb_deprecated_alias_used"]
                    ),
                    "pyramid_generation_policy": "downsample_streamed_s0",
                    "masked_rgb_pyramid_semantics": (
                        "masked_s0_then_downsampled"
                        if primary_rgb_mode == "masked_rgb"
                        else "not_applicable"
                    ),
                    "source_tile_aligned_canvas": False,
                    "status": "running",
                    "started_at_unix": time.time(),
                }
                (work_dir / "run_manifest.json").write_text(
                    json.dumps(_json_ready(run_manifest), indent=2),
                    encoding="utf-8",
                )
                origin_y = int(
                    (record.frame_debug or {}).get("logical_canvas_source_yx", {}).get("y0", 0)
                )
                origin_x = int(
                    (record.frame_debug or {}).get("logical_canvas_source_yx", {}).get("x0", 0)
                )
                write_ngff_from_tile_streaming_ome(
                    tile_yxc_da=tlazy,
                    out_dir=work_dir,
                    phys_xy_um=(px_um, py_um),
                    block_xy=plate_chunk_xy,
                    num_mips=num_mips,
                    name=name,
                    compressor=compressor,
                    channel_labels=default_channel_labels(int(tile_dask.shape[2])),
                    channel_colors=default_channel_colors(int(tile_dask.shape[2])),
                    ngff_metadata=tile_ngff_metadata,
                    metadata_schema=source_metadata_schema,
                    fill_value=0,
                    sparse_zero_chunks=bool(sparse_zero_chunks),
                    coordinate_translation_yx_um=(origin_y * py_um, origin_x * px_um),
                    run_manifest=run_manifest,
                    progress_mode=progress_mode,
                    progress_interval_s=progress_interval_s,
                )
                if store_tissue_mask and record.mask is not None:
                    mask_stats = write_tissue_mask_label_pyramid(
                        record.mask,
                        work_dir,
                        (px_um, py_um),
                        block_xy=plate_chunk_xy,
                        num_mips=num_mips,
                        compressor=compressor,
                        sparse_zero_chunks=bool(sparse_zero_chunks),
                        coordinate_translation_yx_um=(origin_y * py_um, origin_x * px_um),
                        metadata_schema=source_metadata_schema,
                    )
                    run_manifest.update(mask_stats)
                run_manifest["combined_s0_chunks_expected"] = int(
                    run_manifest.get("rgb_s0_chunks_expected", 0)
                    + run_manifest.get("mask_s0_chunks_expected", 0)
                )
                run_manifest["status"] = "complete"
                run_manifest["finished_at_unix"] = time.time()
                (work_dir / "run_manifest.json").write_text(
                    json.dumps(_json_ready(run_manifest), indent=2),
                    encoding="utf-8",
                )
                if ngff_dir.exists():
                    shutil.rmtree(ngff_dir)
                work_dir.rename(ngff_dir)
                if plate is not None:
                    plate.write_slice(z_idx, tlazy)
            else:
                tile = tile_dask.compute()
                if dtype and tile.dtype != np.uint8:
                    if np.issubdtype(tile.dtype, np.integer):
                        maxv = np.iinfo(tile.dtype).max
                        tile = (
                            (tile.astype(np.float32) / max(1, maxv) * 255.0)
                            .clip(0, 255)
                            .astype(np.uint8)
                        )
                    else:
                        tile = (tile * 255.0).clip(0, 255).astype(np.uint16)

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
                mips = build_mips_from_yxc(tile, num_mips)
                write_ngff_from_mips_ngffzarr(
                    mips_yxc=mips,
                    out_dir=ngff_dir,
                    phys_xy_um=(px_um, py_um),
                    name=name,
                    chunks_xy=plate_chunk_xy,
                    version="0.4",
                    overwrite=True,
                    channel_labels=default_channel_labels(mips[0].shape[2]),
                    channel_colors=default_channel_colors(mips[0].shape[2]),
                    add_omero=True,
                    ngff_metadata=tile_ngff_metadata,
                    metadata_schema=source_metadata_schema,
                )
                if plate is not None:
                    plate.write_slice(z_idx, mips)

            _record_written_tissue(
                out_paths,
                ngff_dir,
                record=record,
                source_context=tissue_source_context,
                source_ome_zarr=None,
                source_level=source_idx,
                segmentation_level=segmentation_idx,
                phys_xy_um=(px_um, py_um),
            )
            logger.info(
                "Finished direct ETS tissue %d/%d: %s in %.1fs.",
                z_idx + 1,
                len(tile_records),
                name,
                time.monotonic() - tissue_started,
            )

    logger.info(
        "Wrote %d direct ETS tissue OME-Zarrs to %s in %.1fs.",
        len(out_paths),
        out_ngff_dir,
        time.monotonic() - run_started,
    )
    return out_paths


def process_vsi_directory_with_plating(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "*.vsi",
    source_ome_zarr_dir: str | Path | None = None,
    per_tissue_dir: str | Path | None = None,
    source_level: int | str = 0,
    segmentation_level: int | str | None = 7,
    output_profile: str = "validation",
    tile_frame_level: str = "segmentation",
    crop_shape_policy: str | None = None,
    segmentation_config: SegmentationConfig | None = None,
    tile_config: TileConfig | None = None,
    metadata_backend: str = "auto",
    metadata_schema: str = "v0.4",
    overwrite_source: bool = False,
    source_writer: str = "direct",
    materialize_source: bool = False,
    parallel: bool = False,
    min_side_for_mips: int | None = None,
    requested_mips: int | None = None,
    dtype: np.dtype | str | None = "uint8",
    compression: str | None = None,
    store_tissue_mask: bool | None = None,
    primary_rgb_mode: str | None = None,
    masked_rgb_fill_value: int | None = None,
    store_unmasked_rgb: bool | None = None,
    materialize_masked_rgb: bool | None = None,
    sparse_zero_chunks: bool | None = None,
    pyramid_generation_policy: str | None = None,
    source_tile_aligned_canvas: bool | None = None,
    native_mip_stop_policy: str | None = None,
    native_mip_stop_level: int | str | None = None,
    resume: bool = False,
    progress_mode: str | bool | None = "none",
    progress_interval_s: float = 30.0,
) -> dict[str, list[Path]]:
    """
    Process all matching VSI files into per-tissue OME-Zarr derivatives.

    The returned mapping mirrors ``process_directory()`` style:
    ``{input_vsi_path: [tissue_ome_zarr_paths...]}``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    source_ome_zarr_dir = (
        Path(source_ome_zarr_dir) if source_ome_zarr_dir else output_dir / "source_ome_zarr"
    )
    per_tissue_dir = Path(per_tissue_dir) if per_tissue_dir else output_dir / "per_tissue_ngff"
    if materialize_source:
        source_ome_zarr_dir.mkdir(parents=True, exist_ok=True)
    per_tissue_dir.mkdir(parents=True, exist_ok=True)
    output_profile = _normalize_output_profile(output_profile)
    defaults = _profile_defaults(output_profile)
    if crop_shape_policy is None:
        crop_shape_policy = str(defaults["crop_shape_policy"])
    if compression is None:
        compression = str(defaults["compression"])
    if store_tissue_mask is None:
        store_tissue_mask = bool(defaults["store_tissue_mask"])
    rgb_options = _resolve_primary_rgb_options(
        primary_rgb_mode=primary_rgb_mode,
        materialize_masked_rgb=materialize_masked_rgb,
        masked_rgb_fill_value=masked_rgb_fill_value,
        store_unmasked_rgb=store_unmasked_rgb,
        defaults=defaults,
    )
    primary_rgb_mode = str(rgb_options["primary_rgb_mode"])
    materialize_masked_rgb = bool(rgb_options["materialize_masked_rgb"])
    masked_rgb_fill_value = int(rgb_options["masked_rgb_fill_value"])
    store_unmasked_rgb = bool(rgb_options["store_unmasked_rgb"])
    if sparse_zero_chunks is None:
        sparse_zero_chunks = bool(defaults["sparse_zero_chunks"])
    if pyramid_generation_policy is None:
        pyramid_generation_policy = str(
            defaults.get("pyramid_generation_policy", "downsample_streamed_s0")
        )
    if source_tile_aligned_canvas is None:
        source_tile_aligned_canvas = bool(defaults.get("source_tile_aligned_canvas", False))
    if native_mip_stop_policy is None:
        native_mip_stop_policy = str(defaults.get("native_mip_stop_policy", "segmentation_level"))
    if native_mip_stop_level is None:
        native_mip_stop_level = defaults.get("native_mip_stop_level", "segmentation_level")
    tile_frame_level = _normalize_tile_frame_level(tile_frame_level)
    crop_shape_policy = _normalize_crop_shape_policy(crop_shape_policy)
    compression = _normalize_compression_mode(compression)
    pyramid_generation_policy = _normalize_pyramid_generation_policy(pyramid_generation_policy)
    native_mip_stop_policy = _normalize_native_mip_stop_policy(
        native_mip_stop_policy,
        native_mip_stop_level=native_mip_stop_level,
    )

    results: dict[str, list[Path]] = {}
    vsi_paths = sorted(input_dir.glob(pattern))
    if not vsi_paths:
        logger.warning("No VSI files matched %s in %s", pattern, input_dir)
        return results

    chunk_xy = tile_config.chunk_size if tile_config is not None else 512
    for vsi_path in vsi_paths:
        if materialize_source:
            source_ome_zarr = source_ome_zarr_dir / f"{vsi_path.stem}.ome.zarr"
            source_ome_zarr, ets_path, metadata = vsi_to_source_ome_zarr(
                vsi_path,
                source_ome_zarr,
                metadata_backend=metadata_backend,
                metadata_schema=metadata_schema,
                chunks_xy=chunk_xy,
                overwrite=overwrite_source,
                source_writer=source_writer,
            )
            tissue_paths = process_slide_with_plating(
                source_ome_zarr,
                per_tissue_dir,
                source_level=source_level,
                segmentation_level=segmentation_level,
                output_profile=output_profile,
                tile_frame_level=tile_frame_level,
                crop_shape_policy=crop_shape_policy,
                segmentation_config=segmentation_config,
                tile_config=tile_config,
                parallel=parallel,
                min_side_for_mips=min_side_for_mips,
                requested_mips=requested_mips,
                dtype=dtype,
                compression=compression,
                store_tissue_mask=store_tissue_mask,
                primary_rgb_mode=primary_rgb_mode,
                masked_rgb_fill_value=masked_rgb_fill_value,
                store_unmasked_rgb=store_unmasked_rgb,
                materialize_masked_rgb=materialize_masked_rgb,
                sparse_zero_chunks=sparse_zero_chunks,
                progress_mode=progress_mode,
                progress_interval_s=progress_interval_s,
                source_context={
                    "source_kind": "vsi",
                    "source_path": str(vsi_path),
                    "source_vsi": str(vsi_path),
                    "source_ets": str(ets_path),
                    "source_ome_zarr": str(source_ome_zarr),
                    "ngff_metadata": metadata,
                    "metadata_backend": metadata_backend,
                    "metadata_schema": metadata_schema,
                },
            )
        else:
            tissue_paths = process_vsi_with_direct_plating(
                vsi_path,
                per_tissue_dir,
                source_level=source_level,
                segmentation_level=segmentation_level,
                output_profile=output_profile,
                tile_frame_level=tile_frame_level,
                crop_shape_policy=crop_shape_policy,
                segmentation_config=segmentation_config,
                tile_config=tile_config,
                metadata_backend=metadata_backend,
                metadata_schema=metadata_schema,
                parallel=parallel,
                min_side_for_mips=min_side_for_mips,
                requested_mips=requested_mips,
                dtype=dtype,
                compression=compression,
                store_tissue_mask=store_tissue_mask,
                primary_rgb_mode=primary_rgb_mode,
                masked_rgb_fill_value=masked_rgb_fill_value,
                store_unmasked_rgb=store_unmasked_rgb,
                materialize_masked_rgb=materialize_masked_rgb,
                sparse_zero_chunks=sparse_zero_chunks,
                pyramid_generation_policy=pyramid_generation_policy,
                source_tile_aligned_canvas=source_tile_aligned_canvas,
                native_mip_stop_policy=native_mip_stop_policy,
                native_mip_stop_level=native_mip_stop_level,
                resume=resume,
                progress_mode=progress_mode,
                progress_interval_s=progress_interval_s,
            )
        results[str(vsi_path)] = tissue_paths

    return results
