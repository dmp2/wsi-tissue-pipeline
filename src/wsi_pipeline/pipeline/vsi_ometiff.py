"""Native per-tissue VSI/ETS to OME-TIFF diagnostic benchmark."""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import subprocess
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..bioformats_runtime import ensure_bioformats_jnius
from ..config import PipelineConfig, SegmentationConfig, TileConfig
from ..etsfile import ETSFile
from ..tiles.generator import (
    TissueFrameSpec,
    TissueTileRecord,
    _build_tissue_frame_specs,
    _normalize_crop_shape_policy,
    _normalize_tile_frame_level,
)
from ..vsi_converter import find_ets_file, get_vsi_metadata
from .plating import _segment_for_plating
from .vsi_ets import (
    _VALIDATED_SOURCE_LEVEL2_NATIVE_BASELINE,
    _filled_lr_labels,
    _human_bytes,
    _json_ready,
    _native_pyramid_level_specs,
    _normalize_native_mip_stop_policy,
    _normalize_output_profile,
    _physical_xy_from_metadata,
    _profile_defaults,
    _project_native_mask_block,
    _read_ets_region_yxc_open,
    _record_label_crop_seg_yx,
    _resolve_ets_level,
    _resolve_primary_rgb_options,
    resolve_native_output_levels,
)

# Tests may monkeypatch this object. Runtime imports are lazy so importing the
# pipeline package does not require tifffile until OME-TIFF work is requested.
tifffile: Any | None = None

TIFF_BENCHMARK_MODES = (
    "synthetic-zero-pyramidal-ometiff-write",
    "synthetic-random-pyramidal-ometiff-write",
    "replay-real-rgb-tile-ometiff-write",
    "mask-only-ometiff-write",
    "native-per-tissue-ometiff-export",
)
TILE_SAMPLING_CHOICES = ("first", "random", "stratified")
TISSUE_SELECTION_CHOICES = ("first", "heaviest-tiles", "largest-area")
TIFF_SIZE_LIMIT_BYTES = 4 * 1024**3
TIFF_VALIDATION_WINDOW_SIZE = 512
TIFF_VALIDATION_WINDOW_GRID = 3
SOURCE_LEVEL0_OME_ZARR_PRODUCTION_BASELINE = {
    "source_level": 0,
    "elapsed_s": 3001.8,
    "disk_actual_size": "13G",
    "disk_apparent_size": "13G",
    "file_count": 69519,
    "output_scale_to_source_level": {f"s{i}": i for i in range(8)},
    "notes": "validated source-level-0 masked/native/aligned OME-Zarr production output",
}
SOURCE_LEVEL2_OMETIFF_FALLBACK = {
    "source_level": 2,
    "elapsed_s": 127.4233925580047,
    "physical_bytes": 728555520,
    "apparent_bytes": 646626367,
    "estimated_tiff_tile_payload_count": 5377,
    "positive_rgb_tiles_written": 2530,
    "zero_rgb_tiles_written": 2847,
    "rgb_tiles_skipped_before_decode": 2847,
}


@dataclass(frozen=True)
class OmetiffTissue:
    record: TissueTileRecord
    frame_spec: TissueFrameSpec


@dataclass(frozen=True)
class OmetiffGeometry:
    vsi_path: Path
    ets_path: Path
    metadata: dict[str, Any]
    source_level: int
    segmentation_level: int
    source_shape_yx: tuple[int, int]
    segmentation_shape_yxc: tuple[int, int, int]
    ets_level_shapes_yx: list[tuple[int, int]]
    ets_tile_size_yx: tuple[int, int]
    output_profile: str
    tile_frame_level: str
    crop_shape_policy: str
    chunk_xy: int
    pad_multiple: int
    extra_margin_px: int
    primary_rgb_mode: str
    masked_rgb_fill_value: int
    source_tile_aligned_canvas: bool
    native_mip_stop_policy: str
    native_mip_stop_level: int | str | None
    phys_xy_um: tuple[float, float]
    lr_labels: np.ndarray
    tissues: list[OmetiffTissue]
    all_tissues: list[OmetiffTissue]
    tissue_selection: str
    tissue_index: int | None
    tissue_selection_reason: str
    tissue_selection_candidates: list[dict[str, Any]]
    config_source: str


@dataclass(frozen=True)
class TiffTileTask:
    output_index: int
    source_level: int
    y0: int
    y1: int
    x0: int
    x1: int
    tile_y: int
    tile_x: int
    mask_has_pixels: bool

    @property
    def key(self) -> tuple[int, int, int]:
        return (int(self.output_index), int(self.tile_y), int(self.tile_x))


def _get_tifffile():
    if tifffile is not None:
        return tifffile
    import importlib

    return importlib.import_module("tifffile")


def _normalize_tiff_compression(compression: str | None) -> str | None:
    normalized = str(compression or "deflate").strip().lower().replace("_", "-")
    aliases: dict[str, str | None] = {
        "none": None,
        "raw": None,
        "uncompressed": None,
        "false": None,
        "0": None,
        "deflate": "deflate",
        "adobe-deflate": "deflate",
        "zip": "deflate",
        "zlib": "deflate",
        "lossless": "deflate",
        "zstd": "deflate",
        "lzw": "lzw",
        "jpeg": "jpeg",
    }
    if normalized not in aliases:
        raise ValueError("TIFF compression must be one of deflate, lzw, jpeg, or none.")
    return aliases[normalized]


def _compression_label(compression: str | None) -> str:
    normalized = _normalize_tiff_compression(compression)
    return normalized if normalized is not None else "none"


def _normalize_tile_sampling(tile_sampling: str | None) -> str:
    normalized = str(tile_sampling or "first").strip().lower().replace("_", "-")
    if normalized not in TILE_SAMPLING_CHOICES:
        raise ValueError(
            "tile_sampling must be one of: " + ", ".join(TILE_SAMPLING_CHOICES)
        )
    return normalized


def _normalize_tissue_selection(tissue_selection: str | None) -> str:
    normalized = str(tissue_selection or "first").strip().lower().replace("_", "-")
    if normalized not in TISSUE_SELECTION_CHOICES:
        raise ValueError(
            "tissue_selection must be one of: " + ", ".join(TISSUE_SELECTION_CHOICES)
        )
    return normalized


def _file_sizes(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"apparent_bytes": 0, "physical_bytes": 0, "file_count": 0}
    if path.is_file():
        stat = path.stat()
        physical = int(getattr(stat, "st_blocks", 0) * 512) or int(stat.st_size)
        return {
            "apparent_bytes": int(stat.st_size),
            "physical_bytes": physical,
            "file_count": 1,
        }
    apparent = 0
    physical = 0
    file_count = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        file_count += 1
        stat = item.stat()
        apparent += int(stat.st_size)
        physical += int(getattr(stat, "st_blocks", 0) * 512)
    if physical == 0 and apparent:
        physical = apparent
    return {
        "apparent_bytes": int(apparent),
        "physical_bytes": int(physical),
        "file_count": int(file_count),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(flat)


def _record_from_frame_spec(spec: TissueFrameSpec) -> TissueTileRecord:
    return TissueTileRecord(
        tile=None,  # type: ignore[arg-type]
        tissue_index=int(spec.tissue_index),
        label_id=int(spec.label_id),
        crop_bounds_source_level=spec.clipped_source_yx.as_xyxy(),
        crop_bounds_segmentation_level=spec.clipped_frame_seg_yx.as_xyxy(),
        tile_dim=int(max(spec.source_canvas_shape_yx)),
        tile_shape_yx=tuple(map(int, spec.source_canvas_shape_yx)),
        mask=None,
        tile_frame_level=spec.tile_frame_level,
        crop_shape_policy=spec.crop_shape_policy,
        source_tile_dim=int(max(spec.source_canvas_shape_yx)),
        segmentation_tile_dim=int(spec.segmentation_tile_dim),
        scale_y=float(spec.scale_y),
        scale_x=float(spec.scale_x),
        frame_debug=spec.debug_dict(),
    )


def _tissue_area(tissue: OmetiffTissue) -> int:
    shape = tuple(map(int, tissue.record.tile_shape_yx or (0, 0)))
    if len(shape) != 2:
        return 0
    return int(shape[0] * shape[1])


def _tissue_selection_candidates(
    tissues: list[OmetiffTissue],
    *,
    source_level: int,
    source_shape_yx: tuple[int, int],
    ets_level_shapes_yx: list[tuple[int, int]],
    source_phys_xy_um: tuple[float, float],
    tile_size: int,
    segmentation_level: int,
    native_mip_stop_policy: str,
    native_mip_stop_level: int | str | None,
    source_tile_aligned_canvas: bool,
    source_tile_size_yx: tuple[int, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tissue in tissues:
        specs = _native_pyramid_level_specs(
            record=tissue.record,
            source_level=int(source_level),
            source_shape_yx=source_shape_yx,
            ets_level_shapes_yx=ets_level_shapes_yx,
            source_phys_xy_um=source_phys_xy_um,
            block_xy=int(tile_size),
            min_side_for_mips=None,
            requested_mips=None,
            segmentation_level=int(segmentation_level),
            native_mip_stop_policy=native_mip_stop_policy,
            native_mip_stop_level=native_mip_stop_level,
            source_tile_aligned_canvas=bool(source_tile_aligned_canvas),
            source_tile_size_yx=source_tile_size_yx,
        )
        payload = _tile_payload_estimate(specs, int(tile_size))
        row = {
            "tissue_index": int(tissue.record.tissue_index),
            "label_id": int(tissue.record.label_id),
            "area_px": _tissue_area(tissue),
            "s0_shape_yx": list(map(int, specs[0].output_shape_yx)) if specs else [0, 0],
            **payload,
        }
        rows.append(row)
    return rows


def _select_tissues(
    tissues: list[OmetiffTissue],
    candidates: list[dict[str, Any]],
    *,
    tissue_index: int | None,
    tissue_selection: str,
    max_tissues: int | None,
) -> tuple[list[OmetiffTissue], str]:
    by_index = {int(tissue.record.tissue_index): tissue for tissue in tissues}
    if tissue_index is not None:
        requested = int(tissue_index)
        if requested not in by_index:
            available = sorted(by_index)
            raise ValueError(
                f"Requested tissue_index={requested} was not found; available tissues: {available}"
            )
        selected = [by_index[requested]]
        if max_tissues is not None and int(max_tissues) < 1:
            selected = []
        if len(selected) != 1 or int(selected[0].record.tissue_index) != requested:
            raise RuntimeError(f"Deterministic tissue selection failed for tissue_index={requested}")
        if requested == 0:
            reason = (
                "explicit tissue-index request; source-level-0 estimate identifies tissue 0 as "
                "highest estimated_tiff_tile_payload_count for the E241 pilot"
            )
        else:
            reason = f"explicit tissue-index request for tissue {requested}"
        return selected, reason

    selection = _normalize_tissue_selection(tissue_selection)
    ordered = list(tissues)
    if selection == "heaviest-tiles":
        rank = {
            int(row["tissue_index"]): (
                int(row.get("estimated_tiff_tile_payload_count", 0)),
                int(row.get("area_px", 0)),
                -int(row["tissue_index"]),
            )
            for row in candidates
        }
        ordered.sort(key=lambda tissue: rank[int(tissue.record.tissue_index)], reverse=True)
        reason = "tissue-selection heaviest-tiles: largest estimated TIFF tile payload count"
    elif selection == "largest-area":
        ordered.sort(key=lambda tissue: (_tissue_area(tissue), -int(tissue.record.tissue_index)), reverse=True)
        reason = "tissue-selection largest-area: largest source-level output area"
    else:
        reason = "tissue-selection first: original segmentation order"

    if max_tissues is not None:
        ordered = ordered[: max(0, int(max_tissues))]
    return ordered, reason


def _resolve_ometiff_geometry(
    *,
    vsi_path: str | Path,
    source_level: int | str,
    segmentation_level: int | str | None,
    output_profile: str,
    tile_frame_level: str,
    crop_shape_policy: str | None,
    segmentation_config: SegmentationConfig,
    tile_config: TileConfig,
    metadata_backend: str,
    primary_rgb_mode: str | None,
    masked_rgb_fill_value: int | None,
    store_unmasked_rgb: bool | None,
    materialize_masked_rgb: bool | None,
    source_tile_aligned_canvas: bool | None,
    native_mip_stop_policy: str | None,
    native_mip_stop_level: int | str | None,
    max_tissues: int | None,
    tissue_index: int | None,
    tissue_selection: str,
    config_source: str,
) -> OmetiffGeometry:
    vsi_path = Path(vsi_path)
    output_profile = _normalize_output_profile(output_profile)
    defaults = _profile_defaults(output_profile)
    resolved_crop_shape_policy = _normalize_crop_shape_policy(
        crop_shape_policy or str(defaults["crop_shape_policy"])
    )
    resolved_tile_frame_level = _normalize_tile_frame_level(tile_frame_level)
    if source_tile_aligned_canvas is None:
        source_tile_aligned_canvas = bool(defaults["source_tile_aligned_canvas"])
    if native_mip_stop_policy is None:
        native_mip_stop_policy = str(defaults["native_mip_stop_policy"])
    if native_mip_stop_level is None:
        native_mip_stop_level = defaults.get("native_mip_stop_level")
    native_mip_stop_policy = _normalize_native_mip_stop_policy(
        native_mip_stop_policy,
        native_mip_stop_level=native_mip_stop_level,
    )
    rgb_options = _resolve_primary_rgb_options(
        primary_rgb_mode=primary_rgb_mode,
        materialize_masked_rgb=materialize_masked_rgb,
        masked_rgb_fill_value=masked_rgb_fill_value,
        store_unmasked_rgb=store_unmasked_rgb,
        defaults=defaults,
    )
    resolved_primary_rgb_mode = str(rgb_options["primary_rgb_mode"])
    resolved_fill = int(rgb_options["masked_rgb_fill_value"])
    if resolved_primary_rgb_mode != "masked_rgb":
        raise NotImplementedError("OME-TIFF benchmark currently targets masked_rgb output.")
    if resolved_fill != 0:
        raise NotImplementedError("OME-TIFF benchmark currently targets fill value 0.")
    ets_path = find_ets_file(vsi_path)
    if ets_path is None:
        raise FileNotFoundError(f"No ETS file found for VSI {vsi_path}")
    ets_path = Path(ets_path)
    metadata = get_vsi_metadata(vsi_path, metadata_backend=metadata_backend, target_schema="latest")
    if not metadata:
        raise RuntimeError(f"Unable to extract structural metadata for VSI {vsi_path}")

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
        ets_level_shapes_yx = [
            tuple(map(int, ets.level_shape(idx))) for idx in range(int(ets.nlevels))
        ]
        ets_tile_size_yx = (int(ets.tile_ysize), int(ets.tile_xsize))
        segmentation_yxc = ets.read_level(segmentation_idx)

    filled_lr, _segmentation_info = _segment_for_plating(
        np.moveaxis(segmentation_yxc, -1, 0),
        segment_fn=None,
        segmentation_config=segmentation_config,
        min_size=segmentation_config.min_area_px,
        struct_elem_px=segmentation_config.struct_elem_px,
    )
    lr_labels = _filled_lr_labels(np.asarray(filled_lr, dtype=bool))
    frame_specs, _tile_dim = _build_tissue_frame_specs(
        lr_labels,
        source_shape_yx=source_shape_yx,
        tile_frame_level=resolved_tile_frame_level,
        pad_multiple=int(tile_config.pad_multiple),
        extra_margin_px=int(tile_config.extra_margin_px),
        crop_shape_policy=resolved_crop_shape_policy,
    )
    all_tissues: list[OmetiffTissue] = []
    for spec in frame_specs:
        if spec.clipped_source_yx.h <= 0 or spec.clipped_source_yx.w <= 0:
            continue
        all_tissues.append(OmetiffTissue(record=_record_from_frame_spec(spec), frame_spec=spec))

    base_phys_xy_um = _physical_xy_from_metadata(metadata) or (1.0, 1.0)
    phys_xy_um = (
        float(base_phys_xy_um[0]) * (2**int(source_idx)),
        float(base_phys_xy_um[1]) * (2**int(source_idx)),
    )
    normalized_tissue_selection = _normalize_tissue_selection(tissue_selection)
    selection_candidates = _tissue_selection_candidates(
        all_tissues,
        source_level=int(source_idx),
        source_shape_yx=source_shape_yx,
        ets_level_shapes_yx=ets_level_shapes_yx,
        source_phys_xy_um=phys_xy_um,
        tile_size=int(tile_config.chunk_size),
        segmentation_level=int(segmentation_idx),
        native_mip_stop_policy=native_mip_stop_policy,
        native_mip_stop_level=native_mip_stop_level,
        source_tile_aligned_canvas=bool(source_tile_aligned_canvas),
        source_tile_size_yx=ets_tile_size_yx,
    )
    tissues, selection_reason = _select_tissues(
        all_tissues,
        selection_candidates,
        tissue_index=tissue_index,
        tissue_selection=normalized_tissue_selection,
        max_tissues=max_tissues,
    )
    if not tissues:
        raise ValueError("OME-TIFF benchmark tissue selection resolved to zero tissues.")

    return OmetiffGeometry(
        vsi_path=vsi_path,
        ets_path=ets_path,
        metadata=metadata,
        source_level=int(source_idx),
        segmentation_level=int(segmentation_idx),
        source_shape_yx=source_shape_yx,
        segmentation_shape_yxc=tuple(map(int, segmentation_yxc.shape)),
        ets_level_shapes_yx=ets_level_shapes_yx,
        ets_tile_size_yx=ets_tile_size_yx,
        output_profile=output_profile,
        tile_frame_level=resolved_tile_frame_level,
        crop_shape_policy=resolved_crop_shape_policy,
        chunk_xy=int(tile_config.chunk_size),
        pad_multiple=int(tile_config.pad_multiple),
        extra_margin_px=int(tile_config.extra_margin_px),
        primary_rgb_mode=resolved_primary_rgb_mode,
        masked_rgb_fill_value=resolved_fill,
        source_tile_aligned_canvas=bool(source_tile_aligned_canvas),
        native_mip_stop_policy=native_mip_stop_policy,
        native_mip_stop_level=native_mip_stop_level,
        phys_xy_um=phys_xy_um,
        lr_labels=lr_labels,
        tissues=tissues,
        all_tissues=all_tissues,
        tissue_selection=normalized_tissue_selection,
        tissue_index=int(tissue_index) if tissue_index is not None else None,
        tissue_selection_reason=selection_reason,
        tissue_selection_candidates=selection_candidates,
        config_source=str(config_source),
    )


def _native_specs_for_record(
    geometry: OmetiffGeometry,
    record: TissueTileRecord,
    *,
    requested_mips: int | None = None,
):
    return _native_pyramid_level_specs(
        record=record,
        source_level=int(geometry.source_level),
        source_shape_yx=geometry.source_shape_yx,
        ets_level_shapes_yx=geometry.ets_level_shapes_yx,
        source_phys_xy_um=geometry.phys_xy_um,
        block_xy=int(geometry.chunk_xy),
        min_side_for_mips=None,
        requested_mips=requested_mips,
        segmentation_level=int(geometry.segmentation_level),
        native_mip_stop_policy=geometry.native_mip_stop_policy,
        native_mip_stop_level=geometry.native_mip_stop_level,
        source_tile_aligned_canvas=bool(geometry.source_tile_aligned_canvas),
        source_tile_size_yx=geometry.ets_tile_size_yx,
    )


def _iter_tile_tasks(
    *,
    specs: Iterable[Any],
    lr_labels: np.ndarray,
    record: TissueTileRecord,
    tile_size: int,
) -> Iterator[TiffTileTask]:
    label_crop = _record_label_crop_seg_yx(record)
    for spec in specs:
        out_h, out_w = map(int, spec.output_shape_yx)
        for y0 in range(0, out_h, int(tile_size)):
            y1 = min(out_h, y0 + int(tile_size))
            tile_y = y0 // int(tile_size)
            for x0 in range(0, out_w, int(tile_size)):
                x1 = min(out_w, x0 + int(tile_size))
                tile_x = x0 // int(tile_size)
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
                yield TiffTileTask(
                    output_index=int(spec.output_index),
                    source_level=int(spec.source_level),
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                    tile_y=tile_y,
                    tile_x=tile_x,
                    mask_has_pixels=bool(np.any(mask)),
                )


def _select_tile_keys(
    tasks: list[TiffTileTask],
    *,
    max_tiles: int | None,
    tile_sampling: str,
    tile_random_seed: int,
) -> tuple[set[tuple[int, int, int]] | None, dict[str, Any]]:
    total = len(tasks)
    if max_tiles is None or int(max_tiles) >= total:
        return None, {
            "tile_sampling": tile_sampling,
            "max_tiles": max_tiles,
            "candidate_tiles": total,
            "selected_tiles": total,
            "benchmark_only_partial_payload": False,
        }
    limit = max(0, int(max_tiles))
    sampling = _normalize_tile_sampling(tile_sampling)
    if sampling == "first":
        selected = tasks[:limit]
    elif sampling == "random":
        rng = random.Random(int(tile_random_seed))
        indexes = set(rng.sample(range(total), k=limit))
        selected = [task for idx, task in enumerate(tasks) if idx in indexes]
    else:
        positive = [task for task in tasks if task.mask_has_pixels]
        empty = [task for task in tasks if not task.mask_has_pixels]
        pos_target = min(len(positive), max(1, limit // 2)) if limit else 0
        empty_target = min(len(empty), limit - pos_target)
        remaining = limit - pos_target - empty_target
        rng = random.Random(int(tile_random_seed))
        selected_pos = rng.sample(positive, k=pos_target) if pos_target else []
        selected_empty = rng.sample(empty, k=empty_target) if empty_target else []
        selected_keys = {task.key for task in (*selected_pos, *selected_empty)}
        if remaining > 0:
            leftovers = [task for task in tasks if task.key not in selected_keys]
            selected_keys.update(task.key for task in rng.sample(leftovers, k=remaining))
        selected = [task for task in tasks if task.key in selected_keys]
    selected_key_set = {task.key for task in selected}
    summary = {
        "tile_sampling": sampling,
        "tile_random_seed": int(tile_random_seed),
        "max_tiles": int(max_tiles),
        "candidate_tiles": int(total),
        "selected_tiles": int(len(selected_key_set)),
        "selected_positive_tiles": int(sum(1 for task in selected if task.mask_has_pixels)),
        "selected_empty_tiles": int(sum(1 for task in selected if not task.mask_has_pixels)),
        "candidate_positive_tiles": int(sum(1 for task in tasks if task.mask_has_pixels)),
        "candidate_empty_tiles": int(sum(1 for task in tasks if not task.mask_has_pixels)),
        "benchmark_only_partial_payload": True,
    }
    return selected_key_set, summary


def _pad_tile(arr: np.ndarray, *, tile_size: int, channels: int | None) -> np.ndarray:
    if channels is None:
        out = np.zeros((int(tile_size), int(tile_size)), dtype=np.uint8)
        out[: arr.shape[0], : arr.shape[1]] = arr
        return out
    out = np.zeros((int(tile_size), int(tile_size), int(channels)), dtype=np.uint8)
    out[: arr.shape[0], : arr.shape[1], :] = arr
    return out


def _ome_rgb_metadata(name: str, spec: Any) -> dict[str, Any]:
    return {
        "axes": "YXS",
        "Name": name,
        "PhysicalSizeX": float(spec.phys_xy_um[0]),
        "PhysicalSizeXUnit": "um",
        "PhysicalSizeY": float(spec.phys_xy_um[1]),
        "PhysicalSizeYUnit": "um",
        "Channel": {"Name": ["red", "green", "blue"]},
    }


def _ome_mask_metadata(name: str, spec: Any) -> dict[str, Any]:
    return {
        "axes": "YX",
        "Name": name,
        "PhysicalSizeX": float(spec.phys_xy_um[0]),
        "PhysicalSizeXUnit": "um",
        "PhysicalSizeY": float(spec.phys_xy_um[1]),
        "PhysicalSizeYUnit": "um",
    }


def _shape_yxc(spec: Any) -> tuple[int, int, int]:
    out_h, out_w = map(int, spec.output_shape_yx)
    return (out_h, out_w, 3)


def _shape_yx(spec: Any) -> tuple[int, int]:
    out_h, out_w = map(int, spec.output_shape_yx)
    return (out_h, out_w)


def _write_level(
    writer: Any,
    data: Iterable[np.ndarray],
    *,
    spec: Any,
    tile_size: int,
    compression: str | None,
    is_rgb: bool,
    is_first: bool,
    subifds: int,
    metadata: dict[str, Any] | None,
) -> None:
    kwargs = {
        "tile": (int(tile_size), int(tile_size)),
        "compression": compression,
        "dtype": np.uint8,
        "metadata": metadata if is_first else None,
    }
    if is_rgb:
        kwargs.update({"shape": _shape_yxc(spec), "photometric": "rgb"})
    else:
        kwargs.update({"shape": _shape_yx(spec), "photometric": "minisblack"})
    if is_first and subifds:
        kwargs["subifds"] = int(subifds)
    if not is_first:
        kwargs["subfiletype"] = 1
    writer.write(data, **kwargs)


def _rgb_tile_iterator(
    *,
    ets: ETSFile,
    specs: list[Any],
    lr_labels: np.ndarray,
    record: TissueTileRecord,
    selected_keys: set[tuple[int, int, int]] | None,
    tile_size: int,
    stats: dict[str, Any],
) -> Iterator[tuple[Any, Iterable[np.ndarray]]]:
    label_crop = _record_label_crop_seg_yx(record)
    for spec in specs:
        def _tiles_for_spec(spec=spec):
            out_h, out_w = map(int, spec.output_shape_yx)
            for y0 in range(0, out_h, int(tile_size)):
                y1 = min(out_h, y0 + int(tile_size))
                tile_y = y0 // int(tile_size)
                for x0 in range(0, out_w, int(tile_size)):
                    x1 = min(out_w, x0 + int(tile_size))
                    tile_x = x0 // int(tile_size)
                    key = (int(spec.output_index), int(tile_y), int(tile_x))
                    if selected_keys is not None and key not in selected_keys:
                        started = time.perf_counter()
                        tile = _pad_tile(
                            np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8),
                            tile_size=tile_size,
                            channels=3,
                        )
                        stats["sample_unprocessed_zero_tiles_written"] += 1
                        stats["zero_rgb_tiles_written"] += 1
                        try:
                            yield tile
                        finally:
                            stats["zero_tile_write_compress_s"] += time.perf_counter() - started
                        continue

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
                        stats["mask_positive_tiles"] += 1
                    else:
                        stats["mask_empty_tiles"] += 1

                    started = time.perf_counter()
                    if not mask_has_pixels:
                        stats["rgb_tiles_skipped_before_decode"] += 1
                        tile = _pad_tile(
                            np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8),
                            tile_size=tile_size,
                            channels=3,
                        )
                        stats["zero_rgb_tiles_written"] += 1
                        try:
                            yield tile
                        finally:
                            stats["zero_tile_write_compress_s"] += time.perf_counter() - started
                    else:
                        rgb = _read_ets_region_yxc_open(
                            ets,
                            level=int(spec.source_level),
                            canvas=spec.canvas_source_yx,
                            y0=y0,
                            y1=y1,
                            x0=x0,
                            x1=x1,
                            stats=stats,
                        )
                        rgb = np.where(mask[..., None].astype(bool), rgb, 0).astype(np.uint8)
                        tile = _pad_tile(rgb, tile_size=tile_size, channels=3)
                        stats["positive_rgb_tiles_written"] += 1
                        try:
                            yield tile
                        finally:
                            stats["positive_tile_read_mask_write_s"] += (
                                time.perf_counter() - started
                            )

        yield spec, _tiles_for_spec()


def _mask_tile_iterator(
    *,
    specs: list[Any],
    lr_labels: np.ndarray,
    record: TissueTileRecord,
    selected_keys: set[tuple[int, int, int]] | None,
    tile_size: int,
    stats: dict[str, Any],
) -> Iterator[tuple[Any, Iterable[np.ndarray]]]:
    label_crop = _record_label_crop_seg_yx(record)
    for spec in specs:
        def _tiles_for_spec(spec=spec):
            out_h, out_w = map(int, spec.output_shape_yx)
            for y0 in range(0, out_h, int(tile_size)):
                y1 = min(out_h, y0 + int(tile_size))
                tile_y = y0 // int(tile_size)
                for x0 in range(0, out_w, int(tile_size)):
                    x1 = min(out_w, x0 + int(tile_size))
                    tile_x = x0 // int(tile_size)
                    key = (int(spec.output_index), int(tile_y), int(tile_x))
                    if selected_keys is not None and key not in selected_keys:
                        yield _pad_tile(
                            np.zeros((y1 - y0, x1 - x0), dtype=np.uint8),
                            tile_size=tile_size,
                            channels=None,
                        )
                        stats["zero_mask_tiles_written"] += 1
                        continue
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
                    if np.any(mask):
                        stats["positive_mask_tiles_written"] += 1
                    else:
                        stats["zero_mask_tiles_written"] += 1
                    yield _pad_tile(mask.astype(np.uint8), tile_size=tile_size, channels=None)

        yield spec, _tiles_for_spec()


def _level_records(specs: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "path": f"s{int(spec.output_index)}",
            "output_index": int(spec.output_index),
            "source_level": int(spec.source_level),
            "output_shape_yx": list(map(int, spec.output_shape_yx)),
            "source_shape_yx": list(map(int, spec.source_shape_yx)),
            "canonical_canvas_in_source_level_coordinates": (
                spec.canonical_canvas_source_yx.as_dict()
            ),
            "output_canvas_source_yx": spec.canvas_source_yx.as_dict(),
            "source_read_envelope_yx": spec.source_read_envelope_yx.as_dict(),
            "source_read_envelope_clipped_yx": spec.clipped_source_read_envelope_yx.as_dict(),
            "phys_xy_um": {"x": float(spec.phys_xy_um[0]), "y": float(spec.phys_xy_um[1])},
            "translation_yx_um": list(map(float, spec.translation_yx_um)),
            "scale_from_parent_yx": list(map(float, spec.scale_from_parent_yx)),
        }
        for spec in specs
    ]


def _tile_payload_estimate(specs: list[Any], tile_size: int) -> dict[str, int]:
    tile_count = 0
    for spec in specs:
        out_h, out_w = map(int, spec.output_shape_yx)
        tile_count += math.ceil(out_h / int(tile_size)) * math.ceil(out_w / int(tile_size))
    rgb_bytes = int(tile_count * int(tile_size) * int(tile_size) * 3)
    mask_bytes = int(tile_count * int(tile_size) * int(tile_size))
    return {
        "estimated_tiff_tile_payload_count": int(tile_count),
        "projected_rgb_ometiff_bytes": rgb_bytes,
        "projected_mask_ometiff_bytes": mask_bytes,
        "projected_combined_ometiff_bytes": rgb_bytes + mask_bytes,
    }


def _tile_activity_estimate(
    *,
    specs: list[Any],
    lr_labels: np.ndarray,
    record: TissueTileRecord,
    tile_size: int,
) -> dict[str, int]:
    tasks = list(
        _iter_tile_tasks(
            specs=specs,
            lr_labels=lr_labels,
            record=record,
            tile_size=tile_size,
        )
    )
    positive = sum(1 for task in tasks if task.mask_has_pixels)
    empty = len(tasks) - positive
    return {
        "estimated_positive_tile_count": int(positive),
        "estimated_zero_tile_count": int(empty),
        "estimated_tiff_tile_payload_count": int(len(tasks)),
    }


def write_native_ets_tissue_pyramid_ometiff(
    *,
    ets_path: str | Path,
    rgb_path: str | Path,
    mask_path: str | Path,
    record: TissueTileRecord,
    lr_labels: np.ndarray,
    source_level: int,
    source_shape_yx: tuple[int, int],
    source_phys_xy_um: tuple[float, float],
    tile_size: int,
    name: str,
    compression: str | None = "deflate",
    segmentation_level: int | None = None,
    native_mip_stop_policy: str | None = "segmentation_level",
    native_mip_stop_level: int | str | None = "segmentation_level",
    source_tile_aligned_canvas: bool = True,
    source_tile_size_yx: tuple[int, int] | None = None,
    max_tiles: int | None = None,
    tile_sampling: str = "first",
    tile_random_seed: int = 0,
) -> dict[str, Any]:
    """Write one tissue as companion RGB/mask pyramidal OME-TIFF files."""
    compression_label = _compression_label(compression)
    compression = _normalize_tiff_compression(compression)
    rgb_path = Path(rgb_path)
    mask_path = Path(mask_path)
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if rgb_path.exists():
        rgb_path.unlink()
    if mask_path.exists():
        mask_path.unlink()

    stats: dict[str, Any] = {
        "format": "ome-tiff",
        "mask_representation": "companion_mask_ome_tiff",
        "pyramid_generation_policy": "native_source_pyramid_crop",
        "primary_rgb_mode": "masked_rgb",
        "masked_rgb_fill_value": 0,
        "source_tile_aligned_canvas": bool(source_tile_aligned_canvas),
        "native_mip_stop_policy": native_mip_stop_policy,
        "tile_size": int(tile_size),
        "compression": compression_label,
        "bigtiff_enabled": True,
        "rgb_tiles_skipped_before_decode": 0,
        "zero_rgb_tiles_written": 0,
        "positive_rgb_tiles_written": 0,
        "sample_unprocessed_zero_tiles_written": 0,
        "zero_mask_tiles_written": 0,
        "positive_mask_tiles_written": 0,
        "mask_empty_tiles": 0,
        "mask_positive_tiles": 0,
        "zero_tile_write_compress_s": 0.0,
        "positive_tile_read_mask_write_s": 0.0,
        "source_tile_decode_calls": 0,
        "unique_source_tiles_touched": set(),
    }
    tf = _get_tifffile()
    started = time.perf_counter()
    with ETSFile(ets_path) as ets:
        ets_level_shapes = [
            tuple(map(int, ets.level_shape(idx))) for idx in range(int(ets.nlevels))
        ]
        resolved_source_tile_size_yx = source_tile_size_yx or (
            int(ets.tile_ysize),
            int(ets.tile_xsize),
        )
        specs = _native_pyramid_level_specs(
            record=record,
            source_level=int(source_level),
            source_shape_yx=source_shape_yx,
            ets_level_shapes_yx=ets_level_shapes,
            source_phys_xy_um=source_phys_xy_um,
            block_xy=int(tile_size),
            min_side_for_mips=None,
            requested_mips=None,
            segmentation_level=segmentation_level,
            native_mip_stop_policy=native_mip_stop_policy,
            native_mip_stop_level=native_mip_stop_level,
            source_tile_aligned_canvas=bool(source_tile_aligned_canvas),
            source_tile_size_yx=resolved_source_tile_size_yx,
        )
        min_side_mip_count = len(specs)
        level_plan = resolve_native_output_levels(
            int(source_level),
            int(segmentation_level if segmentation_level is not None else specs[-1].source_level),
            len(ets_level_shapes),
            native_mip_stop_policy,
            native_mip_stop_level,
            min_side_mip_count=min_side_mip_count,
        )
        all_tasks = list(
            _iter_tile_tasks(
                specs=specs,
                lr_labels=lr_labels,
                record=record,
                tile_size=int(tile_size),
            )
        )
        selected_keys, sampling_summary = _select_tile_keys(
            all_tasks,
            max_tiles=max_tiles,
            tile_sampling=tile_sampling,
            tile_random_seed=tile_random_seed,
        )

        with tf.TiffWriter(str(rgb_path), bigtiff=True) as writer:
            for spec, data in _rgb_tile_iterator(
                ets=ets,
                specs=specs,
                lr_labels=lr_labels,
                record=record,
                selected_keys=selected_keys,
                tile_size=int(tile_size),
                stats=stats,
            ):
                is_first = int(spec.output_index) == 0
                _write_level(
                    writer,
                    data,
                    spec=spec,
                    tile_size=int(tile_size),
                    compression=compression,
                    is_rgb=True,
                    is_first=is_first,
                    subifds=len(specs) - 1,
                    metadata=_ome_rgb_metadata(name, specs[0]),
                )

        with tf.TiffWriter(str(mask_path), bigtiff=True) as writer:
            for spec, data in _mask_tile_iterator(
                specs=specs,
                lr_labels=lr_labels,
                record=record,
                selected_keys=selected_keys,
                tile_size=int(tile_size),
                stats=stats,
            ):
                is_first = int(spec.output_index) == 0
                _write_level(
                    writer,
                    data,
                    spec=spec,
                    tile_size=int(tile_size),
                    compression=compression,
                    is_rgb=False,
                    is_first=is_first,
                    subifds=len(specs) - 1,
                    metadata=_ome_mask_metadata(f"{name}_mask", specs[0]),
                )

    stats["elapsed_s"] = time.perf_counter() - started
    stats["num_mips"] = len(specs)
    stats["native_mip_stop_level"] = int(level_plan.native_mip_stop_level)
    stats["mip_stop_reason"] = level_plan.mip_stop_reason
    stats["coarsest_segmentation_level_not_written"] = bool(
        level_plan.coarsest_segmentation_level_not_written
    )
    stats["output_scale_to_source_level"] = {
        f"s{int(spec.output_index)}": int(spec.source_level) for spec in specs
    }
    stats["native_pyramid_levels"] = _level_records(specs)
    stats.update(_tile_payload_estimate(specs, int(tile_size)))
    stats["bigtiff_required_by_estimate"] = bool(
        stats["projected_rgb_ometiff_bytes"] >= TIFF_SIZE_LIMIT_BYTES
        or stats["projected_mask_ometiff_bytes"] >= TIFF_SIZE_LIMIT_BYTES
    )
    stats["tile_sampling_summary"] = sampling_summary
    stats["benchmark_only_partial_payload"] = bool(
        sampling_summary.get("benchmark_only_partial_payload")
    )
    unique_source_tiles = stats.pop("unique_source_tiles_touched", set())
    stats["unique_source_tiles_touched"] = len(unique_source_tiles)
    stats["rgb_path"] = str(rgb_path)
    stats["mask_path"] = str(mask_path)
    stats["rgb_file"] = _file_sizes(rgb_path)
    stats["mask_file"] = _file_sizes(mask_path)
    stats["combined_file"] = {
        "apparent_bytes": int(stats["rgb_file"]["apparent_bytes"])
        + int(stats["mask_file"]["apparent_bytes"]),
        "physical_bytes": int(stats["rgb_file"]["physical_bytes"])
        + int(stats["mask_file"]["physical_bytes"]),
        "file_count": int(stats["rgb_file"]["file_count"]) + int(stats["mask_file"]["file_count"]),
    }
    return _json_ready(stats)


def _write_synthetic_or_replay_mode(
    *,
    mode: str,
    out_path: Path,
    geometry: OmetiffGeometry,
    tissue: OmetiffTissue,
    specs: list[Any],
    compression: str | None,
    tile_size: int,
    max_tiles: int | None,
    tile_sampling: str,
    tile_random_seed: int,
) -> dict[str, Any]:
    tf = _get_tifffile()
    compression_label = _compression_label(compression)
    compression = _normalize_tiff_compression(compression)
    tasks = list(
        _iter_tile_tasks(
            specs=specs,
            lr_labels=geometry.lr_labels,
            record=tissue.record,
            tile_size=tile_size,
        )
    )
    selected_keys, sampling_summary = _select_tile_keys(
        tasks,
        max_tiles=max_tiles,
        tile_sampling=tile_sampling,
        tile_random_seed=tile_random_seed,
    )
    cached_tile: np.ndarray | None = None
    cached_decode_calls = 0
    if mode == "replay-real-rgb-tile-ometiff-write":
        stats_for_read: dict[str, Any] = {"source_tile_decode_calls": 0}
        with ETSFile(geometry.ets_path) as ets:
            for spec in specs:
                first = next((task for task in tasks if task.output_index == spec.output_index), None)
                if first is None:
                    continue
                tile = _read_ets_region_yxc_open(
                    ets,
                    level=int(spec.source_level),
                    canvas=spec.canvas_source_yx,
                    y0=first.y0,
                    y1=first.y1,
                    x0=first.x0,
                    x1=first.x1,
                    stats=stats_for_read,
                )
                cached_tile = _pad_tile(tile, tile_size=tile_size, channels=3)
                break
        cached_decode_calls = int(stats_for_read.get("source_tile_decode_calls", 0))
    rng = np.random.default_rng(int(tile_random_seed))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    started = time.perf_counter()
    tile_counter = 0
    with tf.TiffWriter(str(out_path), bigtiff=True) as writer:
        for spec in specs:
            def _tiles_for_spec(spec=spec):
                nonlocal tile_counter
                out_h, out_w = map(int, spec.output_shape_yx)
                for y0 in range(0, out_h, int(tile_size)):
                    tile_y = y0 // int(tile_size)
                    y1 = min(out_h, y0 + int(tile_size))
                    for x0 in range(0, out_w, int(tile_size)):
                        tile_x = x0 // int(tile_size)
                        x1 = min(out_w, x0 + int(tile_size))
                        key = (int(spec.output_index), int(tile_y), int(tile_x))
                        if selected_keys is not None and key not in selected_keys:
                            tile_counter += 1
                            yield np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                            continue
                        tile_counter += 1
                        if mode == "synthetic-random-pyramidal-ometiff-write":
                            yield rng.integers(
                                0,
                                256,
                                size=(tile_size, tile_size, 3),
                                dtype=np.uint8,
                            )
                        elif mode == "replay-real-rgb-tile-ometiff-write" and cached_tile is not None:
                            yield cached_tile
                        elif mode == "mask-only-ometiff-write":
                            mask = _project_native_mask_block(
                                lr_labels=geometry.lr_labels,
                                label_id=int(tissue.record.label_id),
                                label_crop_seg_yx=_record_label_crop_seg_yx(tissue.record),
                                level_shape_yx=spec.source_shape_yx,
                                canvas=spec.canvas_source_yx,
                                y0=y0,
                                y1=y1,
                                x0=x0,
                                x1=x1,
                            )
                            yield _pad_tile(
                                np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=2),
                                tile_size=tile_size,
                                channels=3,
                            )
                        else:
                            yield np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

            is_first = int(spec.output_index) == 0
            _write_level(
                writer,
                _tiles_for_spec(),
                spec=spec,
                tile_size=tile_size,
                compression=compression,
                is_rgb=True,
                is_first=is_first,
                subifds=len(specs) - 1,
                metadata=_ome_rgb_metadata(f"{out_path.stem}", specs[0]),
            )
    elapsed = time.perf_counter() - started
    sizes = _file_sizes(out_path)
    return _json_ready(
        {
            "mode": mode,
            "tissue_index": int(tissue.record.tissue_index),
            "path": str(out_path),
            "elapsed_s": float(elapsed),
            "tile_payloads_written": int(tile_counter),
            "cached_source_tile_decode_calls": int(cached_decode_calls),
            "compression": compression_label,
            "bigtiff_enabled": True,
            "tile_sampling_summary": sampling_summary,
            **sizes,
        }
    )


def _tiff_yx_from_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[0] in (1, 3, 4) and shape[-1] not in (1, 3, 4):
        return int(shape[1]), int(shape[2])
    if len(shape) == 3:
        return int(shape[0]), int(shape[1])
    raise ValueError(f"Unsupported TIFF level shape {shape}")


def _tiff_window_key(
    shape: tuple[int, ...],
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> tuple[Any, ...]:
    if len(shape) == 2:
        return (slice(y0, y1), slice(x0, x1))
    if len(shape) == 3 and shape[0] in (1, 3, 4) and shape[-1] not in (1, 3, 4):
        return (slice(None), slice(y0, y1), slice(x0, x1))
    if len(shape) == 3:
        return (slice(y0, y1), slice(x0, x1), slice(None))
    raise ValueError(f"Unsupported TIFF level shape {shape}")


def _validation_windows(
    y: int,
    x: int,
    *,
    window_size: int = TIFF_VALIDATION_WINDOW_SIZE,
    grid: int = TIFF_VALIDATION_WINDOW_GRID,
) -> list[dict[str, int]]:
    y = int(y)
    x = int(x)
    win_y = min(int(window_size), y)
    win_x = min(int(window_size), x)
    ys = np.linspace(0, max(0, y - win_y), num=max(1, int(grid)), dtype=int)
    xs = np.linspace(0, max(0, x - win_x), num=max(1, int(grid)), dtype=int)
    windows: list[dict[str, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for y0 in ys:
        for x0 in xs:
            window = (int(y0), int(y0 + win_y), int(x0), int(x0 + win_x))
            if window in seen:
                continue
            seen.add(window)
            windows.append({"y0": window[0], "y1": window[1], "x0": window[2], "x1": window[3]})
    return windows


def _to_yxc_tiff_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        return np.moveaxis(arr, 0, -1)
    return arr


def _read_tiff_level_window(series: Any, level_index: int, window: dict[str, int]) -> np.ndarray:
    import zarr

    store = series.aszarr(level=int(level_index))
    try:
        arr = zarr.open(store, mode="r")
        shape = tuple(map(int, arr.shape))
        key = _tiff_window_key(
            shape,
            y0=int(window["y0"]),
            y1=int(window["y1"]),
            x0=int(window["x0"]),
            x1=int(window["x1"]),
        )
        return _to_yxc_tiff_array(np.asarray(arr[key]))
    finally:
        store.close()


def _zero_outside_mask_sample_check(
    *,
    rgb_series: Any,
    mask_series: Any,
    level_index: int,
    expected_yx: tuple[int, int],
) -> dict[str, Any]:
    windows = _validation_windows(expected_yx[0], expected_yx[1])
    for window in windows:
        rgb = _read_tiff_level_window(rgb_series, level_index, window)
        mask = np.asarray(_read_tiff_level_window(mask_series, level_index, window)).astype(bool)
        if mask.ndim == 3:
            mask = mask[..., 0]
        outside = rgb[~mask]
        if outside.size and np.any(outside):
            return {
                "level": int(level_index),
                "status": "failed",
                "method": "stratified_windows",
                "window": window,
            }
    return {
        "level": int(level_index),
        "status": "passed",
        "method": "stratified_windows",
        "window_count": len(windows),
        "windows": windows,
    }


def _validate_ometiff_pair(
    *,
    rgb_path: Path,
    mask_path: Path,
    expected_levels: list[dict[str, Any]],
    max_pixels: int = 50_000_000,
) -> dict[str, Any]:
    try:
        tf = _get_tifffile()
    except Exception as exc:
        return {"status": "unavailable", "reason": f"tifffile unavailable: {exc}"}
    try:
        with tf.TiffFile(str(rgb_path)) as rgb_tif, tf.TiffFile(str(mask_path)) as mask_tif:
            rgb_ome = getattr(rgb_tif, "ome_metadata", None)
            mask_ome = getattr(mask_tif, "ome_metadata", None)
            rgb_series = rgb_tif.series[0]
            mask_series = mask_tif.series[0]
            rgb_levels = list(getattr(rgb_series, "levels", [rgb_series]))
            mask_levels = list(getattr(mask_series, "levels", [mask_series]))
            if len(rgb_levels) != len(expected_levels) or len(mask_levels) != len(expected_levels):
                return {
                    "status": "failed",
                    "reason": "pyramid_level_count_mismatch",
                    "rgb_levels": len(rgb_levels),
                    "mask_levels": len(mask_levels),
                    "expected_levels": len(expected_levels),
                }
            zero_checks = []
            shape_checks = []
            for idx, expected in enumerate(expected_levels):
                expected_yx = tuple(map(int, expected["output_shape_yx"]))
                rgb_shape = tuple(map(int, rgb_levels[idx].shape))
                mask_shape = tuple(map(int, mask_levels[idx].shape))
                rgb_yx = _tiff_yx_from_shape(rgb_shape)
                mask_yx = _tiff_yx_from_shape(mask_shape)
                shape_check = {
                    "level": idx,
                    "status": "passed" if rgb_yx == expected_yx and mask_yx == expected_yx else "failed",
                    "rgb_shape": rgb_shape,
                    "mask_shape": mask_shape,
                    "expected_yx": expected_yx,
                }
                shape_checks.append(shape_check)
                if shape_check["status"] != "passed":
                    return {
                        "status": "failed",
                        "reason": "shape_mismatch",
                        "level": idx,
                        "rgb_shape": rgb_shape,
                        "mask_shape": mask_shape,
                        "expected_yx": expected_yx,
                        "shape_agreement_by_level": shape_checks,
                    }
                pixels = int(expected_yx[0] * expected_yx[1])
                if pixels > int(max_pixels):
                    check = _zero_outside_mask_sample_check(
                        rgb_series=rgb_series,
                        mask_series=mask_series,
                        level_index=idx,
                        expected_yx=expected_yx,
                    )
                    zero_checks.append(check)
                    if check["status"] != "passed":
                        return {
                            "status": "failed",
                            "reason": "nonzero_rgb_outside_mask",
                            "level": idx,
                            "zero_outside_mask_checks": zero_checks,
                            "shape_agreement_by_level": shape_checks,
                        }
                    continue
                rgb = _to_yxc_tiff_array(np.asarray(rgb_levels[idx].asarray()))
                mask = np.asarray(mask_levels[idx].asarray()).astype(bool)
                if mask.ndim == 3:
                    mask = mask[..., 0]
                outside = rgb[~mask]
                zero_ok = bool(outside.size == 0 or not np.any(outside))
                zero_checks.append(
                    {
                        "level": idx,
                        "status": "passed" if zero_ok else "failed",
                        "method": "full_level",
                    }
                )
                if not zero_ok:
                    return {
                        "status": "failed",
                        "reason": "nonzero_rgb_outside_mask",
                        "level": idx,
                        "zero_outside_mask_checks": zero_checks,
                        "shape_agreement_by_level": shape_checks,
                    }
        return {
            "status": "passed",
            "ome_metadata_readable": bool(rgb_ome and mask_ome),
            "level_count": len(expected_levels),
            "shape_agreement_by_level": shape_checks,
            "zero_outside_mask_checks": zero_checks,
        }
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}

def _validate_bioformats(path: Path) -> dict[str, Any]:
    try:
        jnius = ensure_bioformats_jnius()
        reader_cls = jnius.autoclass("loci.formats.ImageReader")
        reader = reader_cls()
        try:
            reader.setId(str(path))
            return {
                "status": "passed",
                "series_count": int(reader.getSeriesCount()),
            }
        finally:
            reader.close()
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}


def _write_tissue_manifest(
    path: Path,
    *,
    geometry: OmetiffGeometry,
    tissue: OmetiffTissue,
    stats: dict[str, Any],
) -> None:
    payload = {
        "role": "derivative",
        "format": "ome-tiff",
        "source_vsi": str(geometry.vsi_path),
        "source_ets": str(geometry.ets_path),
        "tissue_index": int(tissue.record.tissue_index),
        "label_id": int(tissue.record.label_id),
        "rgb_path": stats.get("rgb_path"),
        "mask_path": stats.get("mask_path"),
        "primary_rgb_mode": geometry.primary_rgb_mode,
        "masked_rgb_fill_value": int(geometry.masked_rgb_fill_value),
        "mask_representation": "companion_mask_ome_tiff",
        "pyramid_generation_policy": "native_source_pyramid_crop",
        "source_tile_aligned_canvas": bool(geometry.source_tile_aligned_canvas),
        "native_mip_stop_policy": geometry.native_mip_stop_policy,
        "native_mip_stop_level": stats.get("native_mip_stop_level"),
        "output_scale_to_source_level": stats.get("output_scale_to_source_level"),
        "native_pyramid_levels": stats.get("native_pyramid_levels"),
        "tile_size": int(geometry.chunk_xy),
        "compression": stats.get("compression"),
        "benchmark_only_partial_payload": bool(stats.get("benchmark_only_partial_payload")),
        "stats": stats,
    }
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _write_qc_manifest(per_tissue_dir: Path, tissue_rows: list[dict[str, Any]]) -> Path:
    records = []
    for idx, row in enumerate(tissue_rows, start=1):
        rgb_path = Path(str(row["rgb_path"]))
        shape = row.get("native_pyramid_levels", [{}])[0].get("output_shape_yx", [0, 0])
        records.append(
            {
                "relative_path": str(rgb_path.relative_to(per_tissue_dir)),
                "filename": rgb_path.name,
                "source_image": Path(str(row.get("source_vsi", "Image_01.vsi"))).name,
                "tile_index_on_source": int(row["tissue_index"]),
                "overall_index": idx,
                "overall_label": f"{idx:04d}",
                "width": int(shape[1]),
                "height": int(shape[0]),
            }
        )
    manifest = {
        "generated_by": "wsi_pipeline.pipeline.vsi_ometiff",
        "records": records,
    }
    manifest_path = per_tissue_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2), encoding="utf-8")
    return manifest_path


def _run_qc(
    per_tissue_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    *,
    qc_masked_background: str,
) -> dict[str, Any]:
    try:
        from ..qc_grid import run_qc_workflow

        result = run_qc_workflow(
            per_tissue_dir,
            output_dir / "_qc_ometiff",
            manifest_path=manifest_path,
            qc_display_mode="auto",
            qc_masked_background=qc_masked_background,
        )
        return {
            "status": "passed" if result.records else "failed",
            "record_count": len(result.records),
            "master_contact_sheet": (
                str(result.artifacts.master_contact_sheet)
                if result.artifacts.master_contact_sheet is not None
                else None
            ),
            "stats_csv": (
                str(result.artifacts.stats_csv) if result.artifacts.stats_csv is not None else None
            ),
            "qc_masked_background": qc_masked_background,
        }
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}


def _comparison_and_decision(
    *,
    totals: dict[str, Any],
    validations: list[dict[str, Any]],
    bioformats: list[dict[str, Any]],
    complete_export: bool,
) -> dict[str, Any]:
    baseline = _VALIDATED_SOURCE_LEVEL2_NATIVE_BASELINE
    elapsed = float(totals.get("elapsed_s", 0.0))
    size = int(totals.get("combined_file", {}).get("physical_bytes", 0))
    readback_ok = complete_export and validations and all(
        item.get("status") == "passed" for item in validations
    )
    interchange = "acceptable" if readback_ok else "not_validated"
    baseline_elapsed = float(baseline["elapsed_s"])
    baseline_size = int(baseline["disk_actual_bytes"])
    competitive = (
        readback_ok
        and elapsed <= baseline_elapsed * 1.5
        and size <= baseline_size * 1.5
    )
    return {
        "comparison_table": [
            {
                "format": "OME-Zarr",
                "source_level": 2,
                "elapsed_s": baseline_elapsed,
                "disk_actual_bytes": baseline_size,
                "disk_actual_size": _human_bytes(baseline_size),
                "notes": "validated masked native baseline; sparse zero chunks retained",
            },
            {
                "format": "OME-TIFF",
                "source_level": totals.get("source_level"),
                "elapsed_s": elapsed,
                "disk_actual_bytes": size,
                "disk_actual_size": _human_bytes(size),
                "notes": (
                    "companion mask OME-TIFF; TIFF writes zero tile payloads even when "
                    "RGB decode is skipped"
                ),
            },
        ],
        "reader_compatibility": {
            "tifffile": "passed" if readback_ok else "not_validated",
            "bioformats": bioformats,
        },
        "mask_handling": "companion mask OME-TIFF per tissue",
        "resume_behavior": "coarse tissue-pair skip only; no sparse chunk/tile resume",
        "database_upload_suitability": (
            "interchange/export" if not competitive else "candidate_production_staging"
        ),
        "decision": {
            "interchange_export": interchange,
            "production_database_staging": "competitive" if competitive else "not_competitive",
            "reason": (
                "OME-Zarr keeps a structural advantage for sparse masked RGB because zero "
                "chunks can be omitted; OME-TIFF remains useful when reader compatibility "
                "matters."
            ),
        },
    }


def _payload_only_estimate(
    geometry: OmetiffGeometry,
    *,
    tissues: list[OmetiffTissue],
) -> dict[str, Any]:
    tissue_rows = []
    totals = {
        "projected_rgb_ometiff_bytes": 0,
        "projected_mask_ometiff_bytes": 0,
        "projected_combined_ometiff_bytes": 0,
        "estimated_zero_tile_count": 0,
        "estimated_positive_tile_count": 0,
        "estimated_tiff_tile_payload_count": 0,
    }
    for tissue in tissues:
        specs = _native_specs_for_record(geometry, tissue.record)
        payload = _tile_payload_estimate(specs, int(geometry.chunk_xy))
        row = {
            "tissue_index": int(tissue.record.tissue_index),
            "label_id": int(tissue.record.label_id),
            "output_scale_to_source_level": {
                f"s{int(spec.output_index)}": int(spec.source_level) for spec in specs
            },
            "native_pyramid_levels": _level_records(specs),
            "estimated_zero_tile_count": 0,
            "estimated_positive_tile_count": 0,
            **payload,
        }
        for key in totals:
            totals[key] += int(row.get(key, 0))
        tissue_rows.append(row)
    totals["bigtiff_required_by_estimate"] = bool(
        totals["projected_rgb_ometiff_bytes"] >= TIFF_SIZE_LIMIT_BYTES
        or totals["projected_mask_ometiff_bytes"] >= TIFF_SIZE_LIMIT_BYTES
    )
    totals["projected_rgb_ometiff_size"] = _human_bytes(totals["projected_rgb_ometiff_bytes"])
    totals["projected_mask_ometiff_size"] = _human_bytes(totals["projected_mask_ometiff_bytes"])
    totals["projected_combined_ometiff_size"] = _human_bytes(
        totals["projected_combined_ometiff_bytes"]
    )
    return {"tissues": tissue_rows, "totals": totals}


def _estimate_geometry(
    geometry: OmetiffGeometry,
    *,
    tissues: list[OmetiffTissue] | None = None,
) -> dict[str, Any]:
    tissue_rows = []
    totals = {
        "projected_rgb_ometiff_bytes": 0,
        "projected_mask_ometiff_bytes": 0,
        "projected_combined_ometiff_bytes": 0,
        "estimated_zero_tile_count": 0,
        "estimated_positive_tile_count": 0,
        "estimated_tiff_tile_payload_count": 0,
    }
    estimate_tissues = geometry.tissues if tissues is None else tissues
    for tissue in estimate_tissues:
        specs = _native_specs_for_record(geometry, tissue.record)
        payload = _tile_payload_estimate(specs, int(geometry.chunk_xy))
        activity = _tile_activity_estimate(
            specs=specs,
            lr_labels=geometry.lr_labels,
            record=tissue.record,
            tile_size=int(geometry.chunk_xy),
        )
        row = {
            "tissue_index": int(tissue.record.tissue_index),
            "label_id": int(tissue.record.label_id),
            "output_scale_to_source_level": {
                f"s{int(spec.output_index)}": int(spec.source_level) for spec in specs
            },
            "native_pyramid_levels": _level_records(specs),
            **payload,
            **activity,
        }
        for key in totals:
            totals[key] += int(row.get(key, 0))
        tissue_rows.append(row)
    totals["bigtiff_required_by_estimate"] = bool(
        totals["projected_rgb_ometiff_bytes"] >= TIFF_SIZE_LIMIT_BYTES
        or totals["projected_mask_ometiff_bytes"] >= TIFF_SIZE_LIMIT_BYTES
    )
    totals["projected_rgb_ometiff_size"] = _human_bytes(totals["projected_rgb_ometiff_bytes"])
    totals["projected_mask_ometiff_size"] = _human_bytes(totals["projected_mask_ometiff_bytes"])
    totals["projected_combined_ometiff_size"] = _human_bytes(
        totals["projected_combined_ometiff_bytes"]
    )
    return {"tissues": tissue_rows, "totals": totals}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path.cwd(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "nogit"


def _source_level2_ometiff_reference() -> dict[str, Any]:
    path = Path("output/ometiff_source_level2_full/benchmark.json")
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            totals = payload.get("totals", {})
            summary = payload.get("summary", {})
            return {
                "source_level": totals.get("source_level", 2),
                "elapsed_s": totals.get("elapsed_s"),
                "summary_elapsed_s": summary.get("elapsed_s"),
                "combined_file": totals.get("combined_file"),
                "estimated_tiff_tile_payload_count": totals.get("estimated_tiff_tile_payload_count"),
                "positive_rgb_tiles_written": totals.get("positive_rgb_tiles_written"),
                "zero_rgb_tiles_written": totals.get("zero_rgb_tiles_written"),
                "rgb_tiles_skipped_before_decode": totals.get("rgb_tiles_skipped_before_decode"),
                "source": str(path),
            }
        except Exception:
            pass
    return {"source": "fallback_constant", **SOURCE_LEVEL2_OMETIFF_FALLBACK}


def _pilot_tissue_row(tissue_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(tissue_rows) != 1:
        return None
    row = tissue_rows[0]
    return row if row.get("mode") == "native-per-tissue-ometiff-export" else None


def _tile_ratio_projection(
    *,
    pilot_row: dict[str, Any],
    all_tissue_estimate: dict[str, Any],
) -> dict[str, Any]:
    all_tiles = int(all_tissue_estimate.get("totals", {}).get("estimated_tiff_tile_payload_count", 0))
    pilot_tiles = int(pilot_row.get("estimated_tiff_tile_payload_count", 0))
    ratio = float(all_tiles / pilot_tiles) if pilot_tiles else 0.0
    pilot_elapsed = float(pilot_row.get("elapsed_s", 0.0))
    pilot_physical = int(pilot_row.get("combined_file", {}).get("physical_bytes", 0))
    pilot_apparent = int(pilot_row.get("combined_file", {}).get("apparent_bytes", 0))
    return {
        "method": "estimate_tile_ratio",
        "all_tissue_estimated_tiles": all_tiles,
        "pilot_tissue_estimated_tiles": pilot_tiles,
        "tile_ratio": ratio,
        "projected_elapsed_s": pilot_elapsed * ratio,
        "projected_elapsed_min": pilot_elapsed * ratio / 60.0,
        "projected_physical_bytes": int(pilot_physical * ratio),
        "projected_physical_size": _human_bytes(int(pilot_physical * ratio)),
        "projected_apparent_bytes": int(pilot_apparent * ratio),
        "projected_apparent_size": _human_bytes(int(pilot_apparent * ratio)),
    }


def _write_pilot_summary(
    *,
    output_dir: Path,
    result: dict[str, Any],
    tissue_rows: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    bioformats: list[dict[str, Any]],
    all_tissue_estimate: dict[str, Any],
    command_used: str | None,
    git_commit: str | None,
) -> dict[str, Any] | None:
    pilot_row = _pilot_tissue_row(tissue_rows)
    if pilot_row is None:
        return None
    validation = validations[0] if validations else {"status": "not_run"}
    bio = bioformats[0] if bioformats else {"status": "not_run"}
    summary = result["summary"]
    qc = result["validation"].get("qc", {})
    projection = _tile_ratio_projection(
        pilot_row=pilot_row,
        all_tissue_estimate=all_tissue_estimate,
    )
    payload = {
        "output_directory": str(output_dir),
        "git_commit": git_commit or _git_commit(),
        "command_used": command_used,
        "source_vsi": summary.get("vsi_path"),
        "resolved_ets": summary.get("ets_path"),
        "selected_tissue_index": int(pilot_row.get("tissue_index")),
        "selected_label_id": int(pilot_row.get("label_id", -1)),
        "selection_reason": summary.get("tissue_selection_reason"),
        "elapsed_wall_s": summary.get("elapsed_s"),
        "export_elapsed_s": pilot_row.get("elapsed_s"),
        "rgb_ometiff_size": pilot_row.get("rgb_file"),
        "mask_ometiff_size": pilot_row.get("mask_file"),
        "combined_disk_usage": pilot_row.get("combined_file"),
        "actual_disk_usage": pilot_row.get("combined_file", {}).get("physical_bytes"),
        "apparent_disk_usage": pilot_row.get("combined_file", {}).get("apparent_bytes"),
        "file_count": pilot_row.get("combined_file", {}).get("file_count"),
        "bigtiff_enabled": pilot_row.get("bigtiff_enabled"),
        "bigtiff_required_by_estimate": pilot_row.get("bigtiff_required_by_estimate"),
        "compression": pilot_row.get("compression"),
        "tile_size": pilot_row.get("tile_size"),
        "pyramid_level_count": pilot_row.get("num_mips"),
        "output_scale_to_source_level": pilot_row.get("output_scale_to_source_level"),
        "rgb_mask_shape_agreement_by_level": validation.get("shape_agreement_by_level"),
        "sampled_zero_outside_mask_validation": validation.get("zero_outside_mask_checks"),
        "qc_artifacts": qc,
        "bioformats_status": bio,
        "tifffile_status": validation,
        "source_level2_ometiff_comparison": _source_level2_ometiff_reference(),
        "source_level0_omezarr_comparison": SOURCE_LEVEL0_OME_ZARR_PRODUCTION_BASELINE,
        "three_tissue_source_level0_ometiff_projection": projection,
    }
    json_path = output_dir / "pilot_summary.json"
    md_path = output_dir / "pilot_summary.md"
    json_path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    md = [
        "# Source-Level-0 OME-TIFF Tissue-00 Pilot Summary",
        "",
        f"- Output directory: `{payload['output_directory']}`",
        f"- Git commit: `{payload['git_commit']}`",
        f"- Source VSI: `{payload['source_vsi']}`",
        f"- Resolved ETS: `{payload['resolved_ets']}`",
        f"- Selected tissue: `{payload['selected_tissue_index']}`; label `{payload['selected_label_id']}`",
        f"- Selection reason: {payload['selection_reason']}",
        f"- Wall elapsed: {float(payload['elapsed_wall_s'] or 0.0):.1f}s",
        f"- Export elapsed: {float(payload['export_elapsed_s'] or 0.0):.1f}s",
        f"- Compression: `{payload['compression']}`; tile size: `{payload['tile_size']}`; BigTIFF enabled: `{payload['bigtiff_enabled']}`; required by estimate: `{payload['bigtiff_required_by_estimate']}`",
        f"- Pyramid mapping: `{payload['output_scale_to_source_level']}`",
        f"- Combined physical size: {_human_bytes(int(payload['actual_disk_usage'] or 0))}; apparent size: {_human_bytes(int(payload['apparent_disk_usage'] or 0))}; file count: `{payload['file_count']}`",
        f"- tifffile status: `{validation.get('status')}`; Bio-Formats status: `{bio.get('status')}`; QC status: `{qc.get('status')}`",
        f"- Three-tissue tile-ratio projected elapsed: {projection['projected_elapsed_min']:.1f} min; projected physical size: {projection['projected_physical_size']}",
        "",
        "Stored RGB remains `primary_rgb_mode=masked_rgb` with `masked_rgb_fill_value=0`; QC background changes are visualization-only.",
    ]
    if command_used:
        md.extend(["", "## Command", "", "```bash", command_used, "```"])
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    payload["pilot_summary_json"] = str(json_path)
    payload["pilot_summary_md"] = str(md_path)
    return _json_ready(payload)


def run_vsi_ometiff_benchmark(
    vsi_path: str | Path,
    output_dir: str | Path,
    *,
    source_level: int | str = 2,
    segmentation_level: int | str | None = 7,
    output_profile: str = "production",
    tile_frame_level: str = "segmentation",
    crop_shape_policy: str | None = None,
    segmentation_config: SegmentationConfig | None = None,
    tile_config: TileConfig | None = None,
    pipeline_config: PipelineConfig | None = None,
    metadata_backend: str = "bioformats",
    metadata_schema: str = "v0.4",
    compression: str | None = "deflate",
    max_tissues: int | None = None,
    max_tiles: int | None = None,
    tile_sampling: str = "first",
    tile_random_seed: int = 0,
    estimate_only: bool = False,
    tissue_index: int | None = None,
    tissue_selection: str = "first",
    no_synthetic_benchmarks: bool = False,
    qc_masked_background: str = "black",
    command_used: str | None = None,
    git_commit: str | None = None,
    primary_rgb_mode: str | None = None,
    masked_rgb_fill_value: int | None = None,
    store_unmasked_rgb: bool | None = None,
    materialize_masked_rgb: bool | None = None,
    source_tile_aligned_canvas: bool | None = None,
    native_mip_stop_policy: str | None = None,
    native_mip_stop_level: int | str | None = None,
    config_source: str | None = None,
) -> dict[str, Any]:
    del metadata_schema
    started = time.perf_counter()
    compression_label = _compression_label(compression)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = pipeline_config or PipelineConfig()
    seg_cfg = segmentation_config or config.segmentation
    tile_cfg = tile_config or config.tiles
    geometry = _resolve_ometiff_geometry(
        vsi_path=vsi_path,
        source_level=source_level,
        segmentation_level=segmentation_level,
        output_profile=output_profile,
        tile_frame_level=tile_frame_level,
        crop_shape_policy=crop_shape_policy,
        segmentation_config=seg_cfg,
        tile_config=tile_cfg,
        metadata_backend=metadata_backend,
        primary_rgb_mode=primary_rgb_mode,
        masked_rgb_fill_value=masked_rgb_fill_value,
        store_unmasked_rgb=store_unmasked_rgb,
        materialize_masked_rgb=materialize_masked_rgb,
        source_tile_aligned_canvas=source_tile_aligned_canvas,
        native_mip_stop_policy=native_mip_stop_policy,
        native_mip_stop_level=native_mip_stop_level,
        max_tissues=max_tissues,
        tissue_index=tissue_index,
        tissue_selection=tissue_selection,
        config_source=config_source or "programmatic/default",
    )
    if estimate_only or not no_synthetic_benchmarks:
        estimate = _estimate_geometry(geometry)
        all_tissue_estimate = _estimate_geometry(geometry, tissues=geometry.all_tissues)
    else:
        estimate = _payload_only_estimate(geometry, tissues=geometry.tissues)
        all_tissue_estimate = _payload_only_estimate(geometry, tissues=geometry.all_tissues)
    complete_export = not estimate_only and max_tiles is None
    rows: list[dict[str, Any]] = []
    tissue_rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    bioformats: list[dict[str, Any]] = []
    per_tissue_dir = output_dir / "per_tissue_ometiff"

    if estimate_only:
        rows.append(
            {
                "mode": "estimate-only",
                "source_level": int(geometry.source_level),
                "elapsed_s": 0.0,
                **estimate["totals"],
            }
        )
    else:
        if per_tissue_dir.exists():
            shutil.rmtree(per_tissue_dir)
        per_tissue_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = output_dir / "artifacts"
        if not no_synthetic_benchmarks:
            artifact_dir.mkdir(parents=True, exist_ok=True)
        for tissue in geometry.tissues:
            name = f"{geometry.vsi_path.stem}_tissue_{int(tissue.record.tissue_index):02d}"
            specs = _native_specs_for_record(geometry, tissue.record)
            if not no_synthetic_benchmarks:
                for mode in TIFF_BENCHMARK_MODES[:-1]:
                    synthetic_path = artifact_dir / f"{name}_{mode}.ome.tif"
                    row = _write_synthetic_or_replay_mode(
                        mode=mode,
                        out_path=synthetic_path,
                        geometry=geometry,
                        tissue=tissue,
                        specs=specs,
                        compression=compression,
                        tile_size=int(geometry.chunk_xy),
                        max_tiles=max_tiles,
                        tile_sampling=tile_sampling,
                        tile_random_seed=int(tile_random_seed) + int(tissue.record.tissue_index),
                    )
                    rows.append(row)

            rgb_path = per_tissue_dir / f"{name}_rgb.ome.tif"
            mask_path = per_tissue_dir / f"{name}_mask.ome.tif"
            stats = write_native_ets_tissue_pyramid_ometiff(
                ets_path=geometry.ets_path,
                rgb_path=rgb_path,
                mask_path=mask_path,
                record=tissue.record,
                lr_labels=geometry.lr_labels,
                source_level=int(geometry.source_level),
                source_shape_yx=geometry.source_shape_yx,
                source_phys_xy_um=geometry.phys_xy_um,
                tile_size=int(geometry.chunk_xy),
                name=name,
                compression=compression,
                segmentation_level=int(geometry.segmentation_level),
                native_mip_stop_policy=geometry.native_mip_stop_policy,
                native_mip_stop_level=geometry.native_mip_stop_level,
                source_tile_aligned_canvas=bool(geometry.source_tile_aligned_canvas),
                source_tile_size_yx=geometry.ets_tile_size_yx,
                max_tiles=max_tiles,
                tile_sampling=tile_sampling,
                tile_random_seed=int(tile_random_seed) + int(tissue.record.tissue_index),
            )
            tissue_manifest = per_tissue_dir / f"{name}_manifest.json"
            _write_tissue_manifest(tissue_manifest, geometry=geometry, tissue=tissue, stats=stats)
            row = {
                "mode": "native-per-tissue-ometiff-export",
                "tissue_index": int(tissue.record.tissue_index),
                "label_id": int(tissue.record.label_id),
                "source_vsi": str(geometry.vsi_path),
                "source_ets": str(geometry.ets_path),
                "manifest_path": str(tissue_manifest),
                **stats,
            }
            rows.append(row)
            tissue_rows.append(row)
            if complete_export:
                validation = _validate_ometiff_pair(
                    rgb_path=rgb_path,
                    mask_path=mask_path,
                    expected_levels=stats["native_pyramid_levels"],
                )
                validation["tissue_index"] = int(tissue.record.tissue_index)
                validations.append(validation)
                bio = _validate_bioformats(rgb_path)
                bio["tissue_index"] = int(tissue.record.tissue_index)
                bioformats.append(bio)

    aggregate_sizes = _file_sizes(per_tissue_dir)
    export_rows = [row for row in tissue_rows if row.get("mode") == "native-per-tissue-ometiff-export"]
    totals = {
        "source_level": int(geometry.source_level),
        "segmentation_level": int(geometry.segmentation_level),
        "tissue_count": len(export_rows) if export_rows else len(geometry.tissues),
        "elapsed_s": float(sum(float(row.get("elapsed_s", 0.0)) for row in export_rows)),
        "combined_file": aggregate_sizes,
        "rgb_tiles_skipped_before_decode": int(
            sum(int(row.get("rgb_tiles_skipped_before_decode", 0)) for row in export_rows)
        ),
        "zero_rgb_tiles_written": int(
            sum(int(row.get("zero_rgb_tiles_written", 0)) for row in export_rows)
        ),
        "positive_rgb_tiles_written": int(
            sum(int(row.get("positive_rgb_tiles_written", 0)) for row in export_rows)
        ),
        "mask_empty_tiles": int(sum(int(row.get("mask_empty_tiles", 0)) for row in export_rows)),
        "mask_positive_tiles": int(
            sum(int(row.get("mask_positive_tiles", 0)) for row in export_rows)
        ),
        "source_tile_decode_calls": int(
            sum(int(row.get("source_tile_decode_calls", 0)) for row in export_rows)
        ),
        **estimate["totals"],
    }
    if tissue_rows:
        manifest_path = _write_qc_manifest(per_tissue_dir, tissue_rows)
    else:
        manifest_path = per_tissue_dir / "manifest.json"
    qc = (
        _run_qc(
            per_tissue_dir,
            output_dir,
            manifest_path,
            qc_masked_background=qc_masked_background,
        )
        if complete_export and tissue_rows
        else {"status": "skipped", "reason": "estimate_only_or_sampled_tiles"}
    )
    comparison = _comparison_and_decision(
        totals=totals,
        validations=validations,
        bioformats=bioformats,
        complete_export=complete_export,
    )
    result = {
        "schema_version": "v0.1",
        "summary": {
            "vsi_path": str(geometry.vsi_path),
            "ets_path": str(geometry.ets_path),
            "source_level": int(geometry.source_level),
            "segmentation_level": int(geometry.segmentation_level),
            "output_profile": geometry.output_profile,
            "tile_frame_level": geometry.tile_frame_level,
            "crop_shape_policy": geometry.crop_shape_policy,
            "extra_margin_px": int(geometry.extra_margin_px),
            "primary_rgb_mode": geometry.primary_rgb_mode,
            "masked_rgb_fill_value": int(geometry.masked_rgb_fill_value),
            "pyramid_generation_policy": "native_source_pyramid_crop",
            "source_tile_aligned_canvas": bool(geometry.source_tile_aligned_canvas),
            "native_mip_stop_policy": geometry.native_mip_stop_policy,
            "native_mip_stop_level": geometry.native_mip_stop_level,
            "tile_size": int(geometry.chunk_xy),
            "compression": compression_label,
            "bigtiff_enabled": True,
            "max_tissues": max_tissues,
            "max_tiles": max_tiles,
            "tile_sampling": _normalize_tile_sampling(tile_sampling),
            "tile_random_seed": int(tile_random_seed),
            "tissue_index": int(tissue_index) if tissue_index is not None else None,
            "tissue_selection": geometry.tissue_selection,
            "tissue_selection_reason": geometry.tissue_selection_reason,
            "tissue_selection_candidates": geometry.tissue_selection_candidates,
            "no_synthetic_benchmarks": bool(no_synthetic_benchmarks),
            "qc_masked_background": qc_masked_background,
            "estimate_only": bool(estimate_only),
            "complete_export": bool(complete_export),
            "config_source": geometry.config_source,
            "elapsed_s": time.perf_counter() - started,
        },
        "estimate": estimate,
        "all_tissue_estimate": all_tissue_estimate,
        "totals": totals,
        "tissues": tissue_rows if tissue_rows else estimate["tissues"],
        "rows": rows,
        "validation": {
            "tifffile": validations,
            "bioformats": bioformats,
            "qc": qc,
        },
        "comparison": comparison,
        "decision_rules": comparison["decision"],
        "artifacts": {
            "per_tissue_dir": str(per_tissue_dir),
            "manifest_json": str(manifest_path),
            "benchmark_json": str(output_dir / "benchmark.json"),
            "benchmark_csv": str(output_dir / "benchmark.csv"),
            "qc_dir": str(output_dir / "_qc_ometiff"),
        },
    }
    pilot_summary = _write_pilot_summary(
        output_dir=output_dir,
        result=result,
        tissue_rows=tissue_rows,
        validations=validations,
        bioformats=bioformats,
        all_tissue_estimate=all_tissue_estimate,
        command_used=command_used,
        git_commit=git_commit,
    )
    if pilot_summary is not None:
        result["pilot_summary"] = pilot_summary
        result["artifacts"]["pilot_summary_json"] = str(output_dir / "pilot_summary.json")
        result["artifacts"]["pilot_summary_md"] = str(output_dir / "pilot_summary.md")
    result = _json_ready(result)
    (output_dir / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_csv(output_dir / "benchmark.csv", rows)
    return result


__all__ = [
    "TIFF_BENCHMARK_MODES",
    "TILE_SAMPLING_CHOICES",
    "run_vsi_ometiff_benchmark",
    "write_native_ets_tissue_pyramid_ometiff",
]
