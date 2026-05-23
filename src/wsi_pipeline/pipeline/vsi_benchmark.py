"""Sampleable VSI/ETS transcode benchmark harness."""

from __future__ import annotations

import cProfile
import csv
import importlib.metadata
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numcodecs import Blosc

from ..config import PipelineConfig, SegmentationConfig, TileConfig
from ..etsfile import ETSFile
from ..omezarr.metadata import default_channel_colors, default_channel_labels
from ..omezarr.pyramid import compute_num_mips_min_side
from ..omezarr.streaming import _max_pool_2x
from ..omezarr.zarr_compat import create_group_array, open_group_v2
from ..tiles.generator import (
    BoundsYX,
    TissueTileRecord,
    _build_tissue_frame_specs,
    _normalize_crop_shape_policy,
    _normalize_tile_frame_level,
    project_label_mask_to_source_region,
)
from ..vsi_converter import find_ets_file, get_vsi_metadata
from .plating import _segment_for_plating
from .vsi_ets import (
    _filled_lr_labels,
    _json_ready,
    _normalize_output_profile,
    _physical_xy_from_metadata,
    _profile_defaults,
    _resolve_ets_level,
    write_native_ets_tissue_pyramid_ome,
)

logger = logging.getLogger(__name__)

REQUIRED_BENCHMARK_MODES = (
    "ets-read-only",
    "synthetic-zero-write",
    "synthetic-random-write",
    "replay-cached-rgb-blocks-write",
    "direct-ets-rgb-no-mask-no-mips",
    "direct-ets-rgb-plus-mask-no-mips",
    "direct-ets-rgb-plus-mask-mips",
    "native-source-pyramid-rgb-plus-mask-mips",
    "native-source-pyramid-rgb-plus-mask-mips-aligned",
    "materialized-source-crop-write",
)

BENCHMARK_CODEC_CHOICES = (
    "none",
    "lz4-byte-shuffle",
    "zstd-1-no-shuffle",
    "zstd-1-byte-shuffle",
    "zstd-1-bitshuffle",
    "zstd-5-bitshuffle",
)

DEFAULT_BENCHMARK_CODEC = "zstd-5-bitshuffle"
BLOCK_SAMPLING_CHOICES = ("first", "random", "tissue", "mixed", "stratified")
COARSE_BLOCK_STRATA = ("padding", "background", "mixed", "tissue")
DETAILED_BLOCK_STRATA = (
    "padding",
    "background",
    "positive_any",
    "low_positive",
    "moderate_positive",
    "high_positive",
)
MiB = float(2**20)


@dataclass(frozen=True)
class CodecSpec:
    name: str
    compressor: Blosc | None
    descriptor: dict[str, Any]


@dataclass
class StageTimers:
    values: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, name: str, elapsed_s: float) -> None:
        self.values[name] += float(elapsed_s)

    def as_dict(self) -> dict[str, float]:
        expected = (
            "geometry_crop_setup_s",
            "ets_tile_lookup_s",
            "ets_tile_decode_s",
            "source_block_assembly_s",
            "mask_projection_s",
            "rgb_write_s",
            "mask_write_s",
            "s0_rgb_write_compress_s",
            "s0_mask_write_compress_s",
            "s0_only_s",
            "mip_generation_s",
            "mask_pyramid_s",
            "rgb_mip_downsample_s",
            "rgb_mip_write_compress_s",
            "mask_mip_downsample_s",
            "mask_mip_write_compress_s",
            "pyramid_write_path_s",
            "compression_write_s",
            "metadata_finalization_s",
            "artifact_cleanup_s",
            "materialized_full_level_read_s",
            "materialized_crop_write_s",
            "materialized_total_s",
            "replay_sample_read_s",
        )
        result = {name: float(self.values.get(name, 0.0)) for name in expected}
        for name, value in sorted(self.values.items()):
            result.setdefault(name, float(value))
        return result


@dataclass
class TileAccounting:
    tile_decode_calls: int = 0
    unique_tiles: set[tuple[int, int, int]] = field(default_factory=set)
    output_chunks_processed: int = 0
    output_chunks_skipped_before_read: int = 0
    output_chunks_requiring_source_pixels: int = 0
    source_tiles_per_output_chunk: list[int] = field(default_factory=list)
    exact_chunks_accounted: int = 0
    lightweight_chunks_accounted: int = 0

    def note_chunk(
        self,
        *,
        source_tiles: Iterable[tuple[int, int, int]],
        requires_source: bool,
        skipped_before_read: bool,
        exact: bool,
    ) -> None:
        tiles = list(source_tiles)
        self.output_chunks_processed += 1
        if skipped_before_read:
            self.output_chunks_skipped_before_read += 1
        if requires_source:
            self.output_chunks_requiring_source_pixels += 1
        self.source_tiles_per_output_chunk.append(len(tiles))
        self.unique_tiles.update(tiles)
        if exact:
            self.exact_chunks_accounted += 1
        else:
            self.lightweight_chunks_accounted += 1

    def note_decode(self, level: int, col: int, row: int) -> None:
        self.tile_decode_calls += 1
        self.unique_tiles.add((int(level), int(col), int(row)))

    def as_dict(self, *, tile_size_yx: tuple[int, int], chunk_xy: int) -> dict[str, Any]:
        unique_count = len(self.unique_tiles)
        repeated_decode_factor = (
            float(self.tile_decode_calls / unique_count) if unique_count else 0.0
        )
        counts = np.asarray(self.source_tiles_per_output_chunk, dtype=np.float64)
        if counts.size:
            mean_tiles = float(counts.mean())
            p50 = float(np.percentile(counts, 50))
            p95 = float(np.percentile(counts, 95))
            max_tiles = int(counts.max())
        else:
            mean_tiles = p50 = p95 = 0.0
            max_tiles = 0
        tile_h, tile_w = map(int, tile_size_yx)
        ideal = max(1, math.ceil(int(chunk_xy) / tile_h) * math.ceil(int(chunk_xy) / tile_w))
        potential_alignment_win = float(mean_tiles / ideal) if ideal else 0.0
        return {
            "unique_ets_source_tiles_touched": int(unique_count),
            "total_ets_tile_decode_calls": int(self.tile_decode_calls),
            "estimated_repeated_decode_factor": repeated_decode_factor,
            "output_chunks_processed": int(self.output_chunks_processed),
            "output_chunks_skipped_before_read": int(self.output_chunks_skipped_before_read),
            "output_chunks_requiring_source_pixels": int(
                self.output_chunks_requiring_source_pixels
            ),
            "source_tiles_per_output_chunk_mean": mean_tiles,
            "source_tiles_per_output_chunk_p50": p50,
            "source_tiles_per_output_chunk_p95": p95,
            "source_tiles_per_output_chunk_max": max_tiles,
            "ideal_source_tiles_per_output_chunk": int(ideal),
            "potential_alignment_win": potential_alignment_win,
            "exact_chunks_accounted": int(self.exact_chunks_accounted),
            "lightweight_chunks_accounted": int(self.lightweight_chunks_accounted),
        }


@dataclass(frozen=True)
class BenchmarkBlock:
    tissue_index: int
    block_index: int
    y0: int
    y1: int
    x0: int
    x1: int
    source_y0: int
    source_y1: int
    source_x0: int
    source_x1: int
    valid_y0: int
    valid_y1: int
    valid_x0: int
    valid_x1: int
    source_tiles: tuple[tuple[int, int, int], ...]
    requires_source: bool
    skipped_before_read: bool
    ideal_source_tile_count: int
    mask_fraction: float = 0.0
    coarse_stratum: str = "unclassified"
    detailed_stratum: str = "unclassified"

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def raw_output_bytes_rgb(self) -> int:
        return int(self.h * self.w * 3)

    @property
    def decoded_source_bytes(self) -> int:
        if not self.requires_source:
            return 0
        return int((self.valid_y1 - self.valid_y0) * (self.valid_x1 - self.valid_x0) * 3)


@dataclass
class BenchmarkTissue:
    tissue_index: int
    label_id: int
    logical_canvas_source_yx: BoundsYX
    clipped_source_yx: BoundsYX
    label_crop_seg_yx: BoundsYX
    tile_shape_yx: tuple[int, int]
    crop_bounds_source_level: tuple[int, int, int, int]
    crop_bounds_segmentation_level: tuple[int, int, int, int]
    num_mips: int
    blocks: list[BenchmarkBlock]
    all_block_count: int
    sampling_summary: dict[str, Any]
    frame_debug: dict[str, Any]


@dataclass
class BenchmarkGeometry:
    vsi_path: Path
    ets_path: Path
    metadata: dict[str, Any]
    source_level: int
    segmentation_level: int
    source_shape_yx: tuple[int, int]
    segmentation_shape_yxc: tuple[int, int, int]
    tile_size_yx: tuple[int, int]
    chunk_xy: int
    output_profile: str
    tile_frame_level: str
    crop_shape_policy: str
    profile_defaults: dict[str, Any]
    configured_extra_margin_px: int
    effective_extra_margin_px: int
    config_source: str
    block_sampling: str
    block_random_seed: int
    sampling_summary: dict[str, Any]
    compression_default: str
    store_tissue_mask: bool
    materialize_masked_rgb: bool
    sparse_zero_chunks: bool
    phys_xy_um: tuple[float, float]
    lr_labels: np.ndarray
    tissues: list[BenchmarkTissue]


def _compressor_for_benchmark_codec(codec: str | None) -> CodecSpec:
    normalized = str(codec or DEFAULT_BENCHMARK_CODEC).strip().lower().replace("_", "-")
    if normalized == "none":
        return CodecSpec(
            name="none",
            compressor=None,
            descriptor={"codec": None, "mode": "none"},
        )
    specs = {
        "lz4-byte-shuffle": ("lz4", 5, Blosc.SHUFFLE),
        "zstd-1-no-shuffle": ("zstd", 1, Blosc.NOSHUFFLE),
        "zstd-1-byte-shuffle": ("zstd", 1, Blosc.SHUFFLE),
        "zstd-1-bitshuffle": ("zstd", 1, Blosc.BITSHUFFLE),
        "zstd-5-bitshuffle": ("zstd", 5, Blosc.BITSHUFFLE),
    }
    if normalized not in specs:
        raise ValueError(
            "codec must be one of: " + ", ".join(("none", *sorted(specs.keys())))
        )
    cname, clevel, shuffle = specs[normalized]
    shuffle_label = {
        Blosc.NOSHUFFLE: "none",
        Blosc.SHUFFLE: "byte",
        Blosc.BITSHUFFLE: "bit",
    }.get(shuffle, str(shuffle))
    return CodecSpec(
        name=normalized,
        compressor=Blosc(cname=cname, clevel=clevel, shuffle=shuffle),
        descriptor={
            "mode": "lossless",
            "codec": "blosc",
            "cname": cname,
            "clevel": clevel,
            "shuffle": shuffle_label,
        },
    )


def _benchmark_codec_from_production_compression(compression: str) -> str:
    return "none" if str(compression).strip().lower() == "none" else DEFAULT_BENCHMARK_CODEC


def _safe_package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _available_ram_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            return page_size * pages
        except (OSError, ValueError):
            return None
    return None


def _git_commit_hash(repo_root: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root or Path.cwd()),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _capture_environment(repo_root: Path | None = None) -> dict[str, Any]:
    packages = {
        name: _safe_package_version(name)
        for name in (
            "numpy",
            "dask",
            "zarr",
            "numcodecs",
            "opencv-python-headless",
            "opencv-python",
            "scikit-image",
            "tinybrain",
        )
    }
    try:
        import cv2

        cv2_version = getattr(cv2, "__version__", None)
    except Exception:
        cv2_version = None
    if cv2_version is not None:
        packages["cv2"] = str(cv2_version)
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "available_ram_bytes": _available_ram_bytes(),
        "packages": packages,
        "git_commit": _git_commit_hash(repo_root),
    }


def _directory_sizes(path: Path) -> dict[str, int]:
    apparent = 0
    physical = 0
    if not path.exists():
        return {"apparent_bytes": 0, "physical_bytes": 0, "file_count": 0}
    file_count = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        file_count += 1
        try:
            stat = item.stat()
        except OSError:
            continue
        apparent += int(stat.st_size)
        physical += int(getattr(stat, "st_blocks", 0) * 512)
    if physical == 0 and apparent:
        physical = apparent
    return {
        "apparent_bytes": int(apparent),
        "physical_bytes": int(physical),
        "file_count": int(file_count),
    }


def _percentiles(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _normalize_block_sampling(block_sampling: str | None) -> str:
    normalized = str(block_sampling or "first").strip().lower().replace("_", "-")
    if normalized not in BLOCK_SAMPLING_CHOICES:
        raise ValueError(
            "block_sampling must be one of: " + ", ".join(BLOCK_SAMPLING_CHOICES)
        )
    return normalized


def _block_strata_for_mask_fraction(
    *,
    requires_source: bool,
    mask_fraction: float,
) -> tuple[str, str]:
    if not requires_source:
        return "padding", "padding"
    fraction = float(mask_fraction)
    if fraction <= 0.0:
        return "background", "background"
    if fraction < 0.05:
        return "mixed", "low_positive"
    if fraction < 0.5:
        return "mixed", "moderate_positive"
    return "tissue", "high_positive"


def _count_block_strata(blocks: Iterable[BenchmarkBlock]) -> dict[str, dict[str, int]]:
    coarse = dict.fromkeys(COARSE_BLOCK_STRATA, 0)
    detailed = dict.fromkeys(DETAILED_BLOCK_STRATA, 0)
    for block in blocks:
        coarse[block.coarse_stratum] = coarse.get(block.coarse_stratum, 0) + 1
        detailed[block.detailed_stratum] = detailed.get(block.detailed_stratum, 0) + 1
        if block.mask_fraction > 0.0:
            detailed["positive_any"] = detailed.get("positive_any", 0) + 1
    return {"coarse": coarse, "detailed": detailed}


def _mask_fraction_summary(blocks: Iterable[BenchmarkBlock]) -> dict[str, float]:
    return _percentiles([float(block.mask_fraction) for block in blocks])


def _sampling_warnings(selected_counts: dict[str, dict[str, int]]) -> list[str]:
    warnings: list[str] = []
    detailed = selected_counts.get("detailed", {})
    coarse = selected_counts.get("coarse", {})
    if int(detailed.get("positive_any", 0)) == 0:
        warnings.append("no_positive_any_blocks_selected")
    if int(coarse.get("mixed", 0)) == 0 and int(coarse.get("tissue", 0)) == 0:
        warnings.append("no_mixed_or_tissue_blocks_selected")
    if int(detailed.get("high_positive", 0)) == 0:
        warnings.append("no_high_positive_blocks_selected")
    return warnings


def _random_subset_preserve_order(
    blocks: list[BenchmarkBlock],
    *,
    count: int,
    rng: np.random.Generator,
) -> list[BenchmarkBlock]:
    if count >= len(blocks):
        return list(blocks)
    if count <= 0:
        return []
    selected = set(rng.choice(len(blocks), size=count, replace=False).tolist())
    return [block for idx, block in enumerate(blocks) if idx in selected]


def _select_blocks_for_sampling(
    blocks: list[BenchmarkBlock],
    *,
    max_blocks: int | None,
    block_sampling: str,
    block_random_seed: int,
) -> tuple[list[BenchmarkBlock], dict[str, Any]]:
    sampling = _normalize_block_sampling(block_sampling)
    candidate_counts = _count_block_strata(blocks)
    limit = len(blocks) if max_blocks is None else max(0, min(int(max_blocks), len(blocks)))
    requested_counts: dict[str, int] = {}
    rng = np.random.default_rng(int(block_random_seed))

    if max_blocks is None and sampling in {"first", "random", "stratified"}:
        selected = list(blocks)
    elif sampling == "first":
        selected = list(blocks[:limit])
    elif sampling == "random":
        selected = _random_subset_preserve_order(blocks, count=limit, rng=rng)
    elif sampling == "tissue":
        positives = [block for block in blocks if block.mask_fraction > 0.0]
        ranked = sorted(positives, key=lambda block: (-block.mask_fraction, block.block_index))
        selected_ids = {block.block_index for block in ranked[:limit]}
        selected = [block for block in blocks if block.block_index in selected_ids]
    elif sampling == "mixed":
        mixed = [block for block in blocks if block.coarse_stratum == "mixed"]
        selected = list(mixed[:limit])
    else:
        by_stratum = {
            stratum: [block for block in blocks if block.coarse_stratum == stratum]
            for stratum in COARSE_BLOCK_STRATA
        }
        base = limit // len(COARSE_BLOCK_STRATA) if COARSE_BLOCK_STRATA else 0
        remainder = limit % len(COARSE_BLOCK_STRATA) if COARSE_BLOCK_STRATA else 0
        for idx, stratum in enumerate(COARSE_BLOCK_STRATA):
            requested_counts[stratum] = base + (1 if idx < remainder else 0)
        selected_ids: set[int] = set()
        for stratum in COARSE_BLOCK_STRATA:
            picked = _random_subset_preserve_order(
                by_stratum[stratum],
                count=min(requested_counts[stratum], len(by_stratum[stratum])),
                rng=rng,
            )
            selected_ids.update(block.block_index for block in picked)
        remaining = limit - len(selected_ids)
        if remaining > 0:
            leftovers = [block for block in blocks if block.block_index not in selected_ids]
            picked = _random_subset_preserve_order(leftovers, count=remaining, rng=rng)
            selected_ids.update(block.block_index for block in picked)
        selected = [block for block in blocks if block.block_index in selected_ids]

    selected_counts = _count_block_strata(selected)
    summary = {
        "block_sampling": sampling,
        "block_random_seed": int(block_random_seed),
        "max_blocks": max_blocks,
        "total_candidate_blocks": int(len(blocks)),
        "selected_blocks": int(len(selected)),
        "requested_coarse_counts": requested_counts,
        "candidate_counts": candidate_counts,
        "selected_counts": selected_counts,
        "candidate_mask_fraction": _mask_fraction_summary(blocks),
        "selected_mask_fraction": _mask_fraction_summary(selected),
        "warnings": _sampling_warnings(selected_counts),
    }
    return selected, summary


def _merge_sampling_summaries(tissues: Iterable[BenchmarkTissue]) -> dict[str, Any]:
    tissue_list = list(tissues)
    summaries = [tissue.sampling_summary for tissue in tissue_list]
    total_candidates = sum(int(item.get("total_candidate_blocks", 0)) for item in summaries)
    total_selected = sum(int(item.get("selected_blocks", 0)) for item in summaries)
    candidate_coarse = dict.fromkeys(COARSE_BLOCK_STRATA, 0)
    selected_coarse = dict.fromkeys(COARSE_BLOCK_STRATA, 0)
    candidate_detailed = dict.fromkeys(DETAILED_BLOCK_STRATA, 0)
    selected_detailed = dict.fromkeys(DETAILED_BLOCK_STRATA, 0)
    warnings: set[str] = set()
    for summary in summaries:
        counts = summary.get("candidate_counts", {})
        for key, value in (counts.get("coarse") or {}).items():
            candidate_coarse[key] = candidate_coarse.get(key, 0) + int(value)
        for key, value in (counts.get("detailed") or {}).items():
            candidate_detailed[key] = candidate_detailed.get(key, 0) + int(value)
        counts = summary.get("selected_counts", {})
        for key, value in (counts.get("coarse") or {}).items():
            selected_coarse[key] = selected_coarse.get(key, 0) + int(value)
        for key, value in (counts.get("detailed") or {}).items():
            selected_detailed[key] = selected_detailed.get(key, 0) + int(value)
        warnings.update(str(item) for item in summary.get("warnings", []))
    selected_counts = {"coarse": selected_coarse, "detailed": selected_detailed}
    warnings.update(_sampling_warnings(selected_counts))
    return {
        "total_candidate_blocks": int(total_candidates),
        "selected_blocks": int(total_selected),
        "candidate_counts": {"coarse": candidate_coarse, "detailed": candidate_detailed},
        "selected_counts": selected_counts,
        "selected_mask_fraction": _mask_fraction_summary(
            block for tissue in tissue_list for block in tissue.blocks
        ),
        "warnings": sorted(warnings),
    }


def _iter_source_tiles_for_region(
    *,
    level: int,
    valid_y0: int,
    valid_y1: int,
    valid_x0: int,
    valid_x1: int,
    tile_h: int,
    tile_w: int,
) -> tuple[tuple[int, int, int], ...]:
    if valid_y1 <= valid_y0 or valid_x1 <= valid_x0:
        return ()
    row0 = valid_y0 // tile_h
    row1 = (valid_y1 - 1) // tile_h
    col0 = valid_x0 // tile_w
    col1 = (valid_x1 - 1) // tile_w
    return tuple(
        (int(level), int(col), int(row))
        for row in range(row0, row1 + 1)
        for col in range(col0, col1 + 1)
    )


def _build_blocks_for_tissue(
    *,
    tissue_index: int,
    canvas: BoundsYX,
    tile_shape_yx: tuple[int, int],
    source_shape_yx: tuple[int, int],
    source_level: int,
    chunk_xy: int,
    tile_size_yx: tuple[int, int],
) -> list[BenchmarkBlock]:
    tile_h, tile_w = map(int, tile_size_yx)
    source_h, source_w = map(int, source_shape_yx)
    tile_h_out, tile_w_out = map(int, tile_shape_yx)
    blocks: list[BenchmarkBlock] = []
    block_index = 0
    for y0 in range(0, tile_h_out, chunk_xy):
        y1 = min(tile_h_out, y0 + chunk_xy)
        for x0 in range(0, tile_w_out, chunk_xy):
            x1 = min(tile_w_out, x0 + chunk_xy)
            source_y0 = int(canvas.y0 + y0)
            source_y1 = int(canvas.y0 + y1)
            source_x0 = int(canvas.x0 + x0)
            source_x1 = int(canvas.x0 + x1)
            valid_y0 = max(0, source_y0)
            valid_y1 = min(source_h, source_y1)
            valid_x0 = max(0, source_x0)
            valid_x1 = min(source_w, source_x1)
            requires_source = valid_y1 > valid_y0 and valid_x1 > valid_x0
            source_tiles = _iter_source_tiles_for_region(
                level=source_level,
                valid_y0=valid_y0,
                valid_y1=valid_y1,
                valid_x0=valid_x0,
                valid_x1=valid_x1,
                tile_h=tile_h,
                tile_w=tile_w,
            )
            ideal = max(1, math.ceil((y1 - y0) / tile_h) * math.ceil((x1 - x0) / tile_w))
            blocks.append(
                BenchmarkBlock(
                    tissue_index=int(tissue_index),
                    block_index=int(block_index),
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                    source_y0=source_y0,
                    source_y1=source_y1,
                    source_x0=source_x0,
                    source_x1=source_x1,
                    valid_y0=valid_y0,
                    valid_y1=valid_y1,
                    valid_x0=valid_x0,
                    valid_x1=valid_x1,
                    source_tiles=source_tiles,
                    requires_source=requires_source,
                    skipped_before_read=not requires_source,
                    ideal_source_tile_count=ideal,
                )
            )
            block_index += 1
    return blocks


def _mask_fraction_for_block(
    *,
    lr_labels: np.ndarray,
    source_shape_yx: tuple[int, int],
    label_id: int,
    label_crop_seg_yx: BoundsYX,
    block: BenchmarkBlock,
) -> float:
    if not block.requires_source:
        return 0.0
    source_h, source_w = map(int, source_shape_yx)
    scale_y = source_h / lr_labels.shape[0]
    scale_x = source_w / lr_labels.shape[1]
    projected = project_label_mask_to_source_region(
        lr_labels,
        label_id=int(label_id),
        source_region_yx=_valid_region_for_block(block),
        label_crop_seg_yx=label_crop_seg_yx.clip(lr_labels.shape),
        scale_y=scale_y,
        scale_x=scale_x,
    )
    denominator = max(1, int(block.h * block.w))
    return float(np.count_nonzero(projected) / denominator)


def _classify_blocks_for_tissue(
    blocks: list[BenchmarkBlock],
    *,
    lr_labels: np.ndarray,
    source_shape_yx: tuple[int, int],
    label_id: int,
    label_crop_seg_yx: BoundsYX,
) -> list[BenchmarkBlock]:
    classified: list[BenchmarkBlock] = []
    for block in blocks:
        mask_fraction = _mask_fraction_for_block(
            lr_labels=lr_labels,
            source_shape_yx=source_shape_yx,
            label_id=label_id,
            label_crop_seg_yx=label_crop_seg_yx,
            block=block,
        )
        coarse, detailed = _block_strata_for_mask_fraction(
            requires_source=block.requires_source,
            mask_fraction=mask_fraction,
        )
        classified.append(
            replace(
                block,
                mask_fraction=mask_fraction,
                coarse_stratum=coarse,
                detailed_stratum=detailed,
            )
        )
    return classified


def _alignment_diagnostics(tissue: BenchmarkTissue, *, tile_size_yx: tuple[int, int]) -> dict[str, Any]:
    tile_h, tile_w = map(int, tile_size_yx)
    canvas = tissue.logical_canvas_source_yx
    aligned = [
        block
        for block in tissue.blocks
        if block.source_y0 % tile_h == 0 and block.source_x0 % tile_w == 0
    ]
    source_tile_counts = [len(block.source_tiles) for block in tissue.blocks]
    ideal_counts = [block.ideal_source_tile_count for block in tissue.blocks]
    mean_current = _percentiles(source_tile_counts)["mean"]
    mean_ideal = max(_percentiles(ideal_counts)["mean"], 1.0)
    return {
        "tissue_index": int(tissue.tissue_index),
        "canvas_origin_mod_ets_tile_yx": [int(canvas.y0 % tile_h), int(canvas.x0 % tile_w)],
        "output_chunk_origin_mod_ets_tile_yx_samples": [
            [int(block.source_y0 % tile_h), int(block.source_x0 % tile_w)]
            for block in tissue.blocks[:16]
        ],
        "fraction_output_chunks_aligned_to_ets_tile_boundaries": (
            float(len(aligned) / len(tissue.blocks)) if tissue.blocks else 0.0
        ),
        "source_tiles_per_output_chunk": _percentiles(source_tile_counts),
        "ideal_source_tiles_per_output_chunk": _percentiles(ideal_counts),
        "potential_alignment_win": float(mean_current / mean_ideal) if mean_ideal else 0.0,
    }


def _resolve_benchmark_geometry(
    *,
    vsi_path: Path,
    source_level: int | str,
    segmentation_level: int | str | None,
    output_profile: str,
    tile_frame_level: str,
    crop_shape_policy: str | None,
    segmentation_config: SegmentationConfig,
    tile_config: TileConfig,
    metadata_backend: str,
    metadata_schema: str,
    max_tissues: int | None,
    max_blocks: int | None,
    block_sampling: str,
    block_random_seed: int,
    config_source: str,
    min_side_for_mips: int | None,
) -> BenchmarkGeometry:
    del metadata_schema
    started = time.perf_counter()
    output_profile = _normalize_output_profile(output_profile)
    defaults = _profile_defaults(output_profile)
    resolved_crop_shape_policy = crop_shape_policy or str(defaults["crop_shape_policy"])
    resolved_crop_shape_policy = _normalize_crop_shape_policy(resolved_crop_shape_policy)
    resolved_tile_frame_level = _normalize_tile_frame_level(tile_frame_level)
    chunk_xy = int(tile_config.chunk_size)
    pad_multiple = int(tile_config.pad_multiple)
    configured_extra_margin_px = int(tile_config.extra_margin_px)
    extra_margin_px = configured_extra_margin_px
    resolved_block_sampling = _normalize_block_sampling(block_sampling)

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
        segmentation_yxc = ets.read_level(segmentation_idx)
        tile_size_yx = (int(ets.tile_ysize), int(ets.tile_xsize))

    filled_lr, segmentation_info = _segment_for_plating(
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
        pad_multiple=pad_multiple,
        extra_margin_px=extra_margin_px,
        crop_shape_policy=resolved_crop_shape_policy,
    )
    tissues: list[BenchmarkTissue] = []
    for spec in frame_specs:
        if max_tissues is not None and len(tissues) >= int(max_tissues):
            break
        clipped = spec.clipped_source_yx
        if clipped.h <= 0 or clipped.w <= 0:
            continue
        tile_shape = tuple(map(int, spec.source_canvas_shape_yx))
        num_mips = compute_num_mips_min_side(
            tile_shape[1],
            tile_shape[0],
            int(min_side_for_mips or chunk_xy),
        )
        candidate_blocks = _build_blocks_for_tissue(
            tissue_index=int(spec.tissue_index),
            canvas=spec.logical_canvas_source_yx,
            tile_shape_yx=tile_shape,
            source_shape_yx=source_shape_yx,
            source_level=source_idx,
            chunk_xy=chunk_xy,
            tile_size_yx=tile_size_yx,
        )
        classified_blocks = _classify_blocks_for_tissue(
            candidate_blocks,
            lr_labels=lr_labels,
            source_shape_yx=source_shape_yx,
            label_id=int(spec.label_id),
            label_crop_seg_yx=spec.label_crop_seg_yx,
        )
        selected_blocks, sampling_summary = _select_blocks_for_sampling(
            classified_blocks,
            max_blocks=max_blocks,
            block_sampling=resolved_block_sampling,
            block_random_seed=int(block_random_seed) + int(spec.tissue_index),
        )
        tissues.append(
            BenchmarkTissue(
                tissue_index=int(spec.tissue_index),
                label_id=int(spec.label_id),
                logical_canvas_source_yx=spec.logical_canvas_source_yx,
                clipped_source_yx=spec.clipped_source_yx,
                label_crop_seg_yx=spec.label_crop_seg_yx,
                tile_shape_yx=tile_shape,
                crop_bounds_source_level=spec.clipped_source_yx.as_xyxy(),
                crop_bounds_segmentation_level=spec.clipped_frame_seg_yx.as_xyxy(),
                num_mips=int(num_mips),
                blocks=selected_blocks,
                all_block_count=len(candidate_blocks),
                sampling_summary=sampling_summary,
                frame_debug={
                    **spec.debug_dict(),
                    "segmentation_info": segmentation_info,
                    "geometry_setup_elapsed_s": time.perf_counter() - started,
                },
            )
        )

    phys_xy = _physical_xy_from_metadata(metadata) or (1.0, 1.0)
    phys_xy_level = (float(phys_xy[0]) * (2**source_idx), float(phys_xy[1]) * (2**source_idx))
    return BenchmarkGeometry(
        vsi_path=vsi_path,
        ets_path=ets_path,
        metadata=metadata,
        source_level=int(source_idx),
        segmentation_level=int(segmentation_idx),
        source_shape_yx=source_shape_yx,
        segmentation_shape_yxc=tuple(map(int, segmentation_yxc.shape)),
        tile_size_yx=tile_size_yx,
        chunk_xy=chunk_xy,
        output_profile=output_profile,
        tile_frame_level=resolved_tile_frame_level,
        crop_shape_policy=resolved_crop_shape_policy,
        profile_defaults=dict(defaults),
        configured_extra_margin_px=int(configured_extra_margin_px),
        effective_extra_margin_px=int(extra_margin_px),
        config_source=str(config_source),
        block_sampling=resolved_block_sampling,
        block_random_seed=int(block_random_seed),
        sampling_summary=_merge_sampling_summaries(tissues),
        compression_default=str(defaults["compression"]),
        store_tissue_mask=bool(defaults["store_tissue_mask"]),
        materialize_masked_rgb=bool(defaults["materialize_masked_rgb"]),
        sparse_zero_chunks=bool(defaults["sparse_zero_chunks"]),
        phys_xy_um=phys_xy_level,
        lr_labels=lr_labels,
        tissues=tissues,
    )


def _geometry_to_json(geometry: BenchmarkGeometry) -> dict[str, Any]:
    return {
        "vsi_path": str(geometry.vsi_path),
        "ets_path": str(geometry.ets_path),
        "source_level": int(geometry.source_level),
        "segmentation_level": int(geometry.segmentation_level),
        "source_shape_yx": list(map(int, geometry.source_shape_yx)),
        "segmentation_shape_yxc": list(map(int, geometry.segmentation_shape_yxc)),
        "ets_tile_size_yx": list(map(int, geometry.tile_size_yx)),
        "chunk_xy": int(geometry.chunk_xy),
        "output_profile": geometry.output_profile,
        "tile_frame_level": geometry.tile_frame_level,
        "crop_shape_policy": geometry.crop_shape_policy,
        "profile_defaults": geometry.profile_defaults,
        "configured_extra_margin_px": int(geometry.configured_extra_margin_px),
        "effective_extra_margin_px": int(geometry.effective_extra_margin_px),
        "config_source": geometry.config_source,
        "block_sampling": geometry.block_sampling,
        "block_random_seed": int(geometry.block_random_seed),
        "sampling_summary": geometry.sampling_summary,
        "store_tissue_mask": bool(geometry.store_tissue_mask),
        "materialize_masked_rgb": bool(geometry.materialize_masked_rgb),
        "sparse_zero_chunks": bool(geometry.sparse_zero_chunks),
        "source_physical_pixel_size_um": {
            "x": float(geometry.phys_xy_um[0]),
            "y": float(geometry.phys_xy_um[1]),
        },
        "tissue_count": len(geometry.tissues),
        "tissues": [
            {
                "tissue_index": int(tissue.tissue_index),
                "label_id": int(tissue.label_id),
                "tile_shape_yx": list(map(int, tissue.tile_shape_yx)),
                "crop_bounds_source_level": list(map(int, tissue.crop_bounds_source_level)),
                "crop_bounds_segmentation_level": list(
                    map(int, tissue.crop_bounds_segmentation_level)
                ),
                "num_mips": int(tissue.num_mips),
                "candidate_blocks": int(tissue.all_block_count),
                "sampled_blocks": len(tissue.blocks),
                "sampling_summary": tissue.sampling_summary,
                "logical_canvas_source_yx": tissue.logical_canvas_source_yx.as_dict(),
                "clipped_source_yx": tissue.clipped_source_yx.as_dict(),
                "source_tile_alignment": _alignment_diagnostics(
                    tissue,
                    tile_size_yx=geometry.tile_size_yx,
                ),
            }
            for tissue in geometry.tissues
        ],
    }


def _warm_filesystem_cache(geometry: BenchmarkGeometry) -> dict[str, Any]:
    warmed_tiles = 0
    started = time.perf_counter()
    with ETSFile(geometry.ets_path) as ets:
        for tissue in geometry.tissues[:1]:
            for block in tissue.blocks[:4]:
                for _level, col, row in block.source_tiles[:2]:
                    ets.get_tile_decoded(geometry.source_level, col, row)
                    warmed_tiles += 1
    return {"warmed_tiles": warmed_tiles, "elapsed_s": time.perf_counter() - started}


def _create_benchmark_arrays(
    out_dir: Path,
    tissue: BenchmarkTissue,
    *,
    num_mips: int,
    include_mask: bool,
    chunk_xy: int,
    codec: CodecSpec,
) -> tuple[Any, list[Any], list[Any]]:
    root = open_group_v2(str(out_dir), mode="w")
    arrays = []
    mask_arrays = []
    height, width = map(int, tissue.tile_shape_yx)
    for m in range(num_mips):
        sy = max(1, height >> m)
        sx = max(1, width >> m)
        arr = create_group_array(
            root,
            f"s{m}",
            shape=(3, sy, sx),
            chunks=(3, min(chunk_xy, sy), min(chunk_xy, sx)),
            dtype="uint8",
            compressor=codec.compressor,
            fill_value=0,
            overwrite=True,
            zarr_format=2,
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = ["c", "y", "x"]
        arrays.append(arr)
    if include_mask:
        labels_group = root.create_group("labels", overwrite=True)
        labels_group.attrs["labels"] = ["tissue_mask"]
        mask_group = labels_group.create_group("tissue_mask", overwrite=True)
        for m in range(num_mips):
            sy = max(1, height >> m)
            sx = max(1, width >> m)
            arr = create_group_array(
                mask_group,
                f"s{m}",
                shape=(sy, sx),
                chunks=(min(chunk_xy, sy), min(chunk_xy, sx)),
                dtype="uint8",
                compressor=codec.compressor,
                fill_value=0,
                overwrite=True,
                zarr_format=2,
            )
            arr.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
            mask_arrays.append(arr)
    return root, arrays, mask_arrays


def _finalize_minimal_metadata(
    root: Any,
    *,
    tissue: BenchmarkTissue,
    mode: str,
    codec: CodecSpec,
    num_mips: int,
    phys_xy_um: tuple[float, float],
) -> None:
    datasets = []
    for m in range(num_mips):
        scale = [1.0, float(phys_xy_um[1]) * (2**m), float(phys_xy_um[0]) * (2**m)]
        datasets.append(
            {
                "path": f"s{m}",
                "coordinateTransformations": [{"type": "scale", "scale": scale}],
            }
        )
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": f"benchmark_tissue_{tissue.tissue_index:02d}",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets,
        }
    ]
    root.attrs["omero"] = {
        "name": f"benchmark_tissue_{tissue.tissue_index:02d}",
        "version": "0.4",
        "channels": [
            {"label": label, "color": color, "active": True}
            for label, color in zip(default_channel_labels(3), default_channel_colors(3), strict=True)
        ],
    }
    root.attrs["benchmark"] = {
        "mode": mode,
        "codec": codec.descriptor,
        "tissue_index": int(tissue.tissue_index),
    }


def _valid_region_for_block(block: BenchmarkBlock) -> BoundsYX:
    return BoundsYX(block.valid_y0, block.valid_x0, block.valid_y1, block.valid_x1)


def _read_ets_block_yxc(
    ets: ETSFile,
    block: BenchmarkBlock,
    *,
    level: int,
    tile_size_yx: tuple[int, int],
    timers: StageTimers,
    accounting: TileAccounting,
) -> np.ndarray:
    out = np.zeros((block.h, block.w, 3), dtype=np.uint8)
    if not block.requires_source:
        return out

    lookup_started = time.perf_counter()
    tile_h, tile_w = map(int, tile_size_yx)
    source_tiles = block.source_tiles
    timers.add("ets_tile_lookup_s", time.perf_counter() - lookup_started)
    for _level, col, row in source_tiles:
        tile_y0 = row * tile_h
        tile_x0 = col * tile_w
        iy0 = max(block.valid_y0, tile_y0)
        iy1 = min(block.valid_y1, tile_y0 + tile_h)
        ix0 = max(block.valid_x0, tile_x0)
        ix1 = min(block.valid_x1, tile_x0 + tile_w)
        if iy1 <= iy0 or ix1 <= ix0:
            continue
        decode_started = time.perf_counter()
        tile = ets.get_tile_decoded(level, col, row)
        timers.add("ets_tile_decode_s", time.perf_counter() - decode_started)
        accounting.note_decode(level, col, row)

        assembly_started = time.perf_counter()
        src = tile[iy0 - tile_y0 : iy1 - tile_y0, ix0 - tile_x0 : ix1 - tile_x0, :]
        dst_y0 = iy0 - block.source_y0
        dst_x0 = ix0 - block.source_x0
        out[dst_y0 : dst_y0 + src.shape[0], dst_x0 : dst_x0 + src.shape[1], :] = src
        timers.add("source_block_assembly_s", time.perf_counter() - assembly_started)
    return out


def _read_materialized_block_yxc(source_yxc: np.ndarray, block: BenchmarkBlock) -> np.ndarray:
    out = np.zeros((block.h, block.w, 3), dtype=np.uint8)
    if not block.requires_source:
        return out
    src = source_yxc[block.valid_y0 : block.valid_y1, block.valid_x0 : block.valid_x1, :3]
    dst_y0 = block.valid_y0 - block.source_y0
    dst_x0 = block.valid_x0 - block.source_x0
    out[dst_y0 : dst_y0 + src.shape[0], dst_x0 : dst_x0 + src.shape[1], :] = src
    return out


def _project_mask_block(
    geometry: BenchmarkGeometry,
    tissue: BenchmarkTissue,
    block: BenchmarkBlock,
    timers: StageTimers,
) -> np.ndarray:
    mask = np.zeros((block.h, block.w), dtype=np.uint8)
    if not block.requires_source:
        return mask
    started = time.perf_counter()
    source_h, source_w = map(int, geometry.source_shape_yx)
    scale_y = source_h / geometry.lr_labels.shape[0]
    scale_x = source_w / geometry.lr_labels.shape[1]
    projected = project_label_mask_to_source_region(
        geometry.lr_labels,
        label_id=int(tissue.label_id),
        source_region_yx=_valid_region_for_block(block),
        label_crop_seg_yx=tissue.label_crop_seg_yx.clip(geometry.lr_labels.shape),
        scale_y=scale_y,
        scale_x=scale_x,
    )
    dst_y0 = block.valid_y0 - block.source_y0
    dst_x0 = block.valid_x0 - block.source_x0
    mask[dst_y0 : dst_y0 + projected.shape[0], dst_x0 : dst_x0 + projected.shape[1]] = (
        projected.astype(np.uint8)
    )
    timers.add("mask_projection_s", time.perf_counter() - started)
    return mask


def _write_rgb_block(arr: Any, block: BenchmarkBlock, rgb: np.ndarray, timers: StageTimers) -> bool:
    if not np.any(rgb):
        return False
    started = time.perf_counter()
    arr[:, block.y0 : block.y1, block.x0 : block.x1] = np.moveaxis(rgb, -1, 0)
    elapsed = time.perf_counter() - started
    timers.add("s0_rgb_write_compress_s", elapsed)
    timers.add("rgb_write_s", elapsed)
    timers.add("compression_write_s", elapsed)
    return True


def _write_mask_block(arr: Any, block: BenchmarkBlock, mask: np.ndarray, timers: StageTimers) -> bool:
    if not np.any(mask):
        return False
    started = time.perf_counter()
    arr[block.y0 : block.y1, block.x0 : block.x1] = mask
    elapsed = time.perf_counter() - started
    timers.add("s0_mask_write_compress_s", elapsed)
    timers.add("mask_write_s", elapsed)
    timers.add("compression_write_s", elapsed)
    return True


def _write_rgb_mips(
    arrays: list[Any],
    block: BenchmarkBlock,
    rgb: np.ndarray,
    timers: StageTimers,
) -> int:
    writes = 0
    src = rgb
    for m in range(1, len(arrays)):
        started = time.perf_counter()
        src = src[::2, ::2, :]
        elapsed = time.perf_counter() - started
        timers.add("rgb_mip_downsample_s", elapsed)
        timers.add("mip_generation_s", elapsed)
        timers.add("pyramid_write_path_s", elapsed)
        if src.size == 0 or not np.any(src):
            continue
        ym0 = block.y0 >> m
        xm0 = block.x0 >> m
        ym1 = ym0 + src.shape[0]
        xm1 = xm0 + src.shape[1]
        write_started = time.perf_counter()
        arrays[m][:, ym0:ym1, xm0:xm1] = np.moveaxis(src, -1, 0)
        elapsed = time.perf_counter() - write_started
        timers.add("rgb_mip_write_compress_s", elapsed)
        timers.add("rgb_write_s", elapsed)
        timers.add("compression_write_s", elapsed)
        timers.add("pyramid_write_path_s", elapsed)
        writes += 1
    return writes


def _write_mask_mips(
    arrays: list[Any],
    block: BenchmarkBlock,
    mask: np.ndarray,
    timers: StageTimers,
) -> int:
    writes = 0
    src = mask
    for m in range(1, len(arrays)):
        started = time.perf_counter()
        src = _max_pool_2x(src)
        elapsed = time.perf_counter() - started
        timers.add("mask_mip_downsample_s", elapsed)
        timers.add("mask_pyramid_s", elapsed)
        timers.add("pyramid_write_path_s", elapsed)
        if src.size == 0 or not np.any(src):
            continue
        ym0 = block.y0 >> m
        xm0 = block.x0 >> m
        ym1 = ym0 + src.shape[0]
        xm1 = xm0 + src.shape[1]
        write_started = time.perf_counter()
        arrays[m][ym0:ym1, xm0:xm1] = src
        elapsed = time.perf_counter() - write_started
        timers.add("mask_mip_write_compress_s", elapsed)
        timers.add("mask_write_s", elapsed)
        timers.add("compression_write_s", elapsed)
        timers.add("pyramid_write_path_s", elapsed)
        writes += 1
    return writes


def _mode_uses_zarr(mode: str) -> bool:
    return mode != "ets-read-only"


def _mode_uses_source(mode: str) -> bool:
    return mode in {
        "ets-read-only",
        "replay-cached-rgb-blocks-write",
        "direct-ets-rgb-no-mask-no-mips",
        "direct-ets-rgb-plus-mask-no-mips",
        "direct-ets-rgb-plus-mask-mips",
        "native-source-pyramid-rgb-plus-mask-mips",
        "native-source-pyramid-rgb-plus-mask-mips-aligned",
    }


def _mode_includes_mask(mode: str) -> bool:
    return mode in {
        "direct-ets-rgb-plus-mask-no-mips",
        "direct-ets-rgb-plus-mask-mips",
        "native-source-pyramid-rgb-plus-mask-mips",
        "native-source-pyramid-rgb-plus-mask-mips-aligned",
        "materialized-source-crop-write",
    }


def _mode_includes_mips(mode: str) -> bool:
    return mode in {
        "direct-ets-rgb-plus-mask-mips",
        "native-source-pyramid-rgb-plus-mask-mips",
        "native-source-pyramid-rgb-plus-mask-mips-aligned",
        "materialized-source-crop-write",
    }


def _mode_is_native_source_pyramid(mode: str) -> bool:
    return mode in {
        "native-source-pyramid-rgb-plus-mask-mips",
        "native-source-pyramid-rgb-plus-mask-mips-aligned",
    }


def _native_mode_uses_aligned_canvas(mode: str) -> bool:
    return mode == "native-source-pyramid-rgb-plus-mask-mips-aligned"


def _record_for_benchmark_tissue(tissue: BenchmarkTissue) -> TissueTileRecord:
    return TissueTileRecord(
        tile=np.zeros((*tissue.tile_shape_yx, 3), dtype=np.uint8),
        tissue_index=int(tissue.tissue_index),
        label_id=int(tissue.label_id),
        crop_bounds_source_level=tuple(map(int, tissue.crop_bounds_source_level)),
        crop_bounds_segmentation_level=tuple(map(int, tissue.crop_bounds_segmentation_level)),
        tile_dim=int(max(tissue.tile_shape_yx)),
        tile_shape_yx=tuple(map(int, tissue.tile_shape_yx)),
        mask=None,
        tile_frame_level="source",
        crop_shape_policy="compact_rectangle",
        source_tile_dim=int(max(tissue.tile_shape_yx)),
        segmentation_tile_dim=None,
        frame_debug=tissue.frame_debug,
    )


def _sample_replay_blocks(
    geometry: BenchmarkGeometry,
    *,
    sample_count: int = 8,
    timers: StageTimers,
    accounting: TileAccounting,
) -> list[np.ndarray]:
    samples: list[np.ndarray] = []
    started = time.perf_counter()
    with ETSFile(geometry.ets_path) as ets:
        for tissue in geometry.tissues:
            for block in tissue.blocks:
                if not block.requires_source:
                    continue
                samples.append(
                    _read_ets_block_yxc(
                        ets,
                        block,
                        level=geometry.source_level,
                        tile_size_yx=geometry.tile_size_yx,
                        timers=timers,
                        accounting=accounting,
                    )
                )
                if len(samples) >= sample_count:
                    timers.add("replay_sample_read_s", time.perf_counter() - started)
                    return samples
    timers.add("replay_sample_read_s", time.perf_counter() - started)
    if not samples:
        samples.append(np.zeros((geometry.chunk_xy, geometry.chunk_xy, 3), dtype=np.uint8))
    return samples


def _run_single_mode(
    *,
    geometry: BenchmarkGeometry,
    benchmark_dir: Path,
    run_order: int,
    mode: str,
    codec: CodecSpec,
    keep_artifacts: bool,
    materialized_read_max_gib: float,
    cache_state: str,
) -> dict[str, Any]:
    if mode not in REQUIRED_BENCHMARK_MODES:
        raise ValueError(f"Unknown benchmark mode: {mode}")

    timers = StageTimers()
    accounting = TileAccounting()
    mode_started = time.perf_counter()
    artifact_dir = benchmark_dir / "artifacts" / f"{run_order:03d}_{mode}_{codec.name}"
    deleted_before_run = False
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
        deleted_before_run = True
    artifact_dir.mkdir(parents=True, exist_ok=True)

    skipped_reason: str | None = None
    chunks_written = 0
    chunks_skipped = 0
    mask_chunks_written = 0
    mask_chunks_skipped = 0
    mip_chunks_written = 0
    mask_mip_chunks_written = 0
    raw_input_bytes = 0
    raw_output_bytes = 0
    materialized_source: np.ndarray | None = None
    native_accounting_override: dict[str, Any] | None = None
    native_writer_metrics: dict[str, Any] = {}

    if mode == "materialized-source-crop-write":
        source_h, source_w = map(int, geometry.source_shape_yx)
        estimated_gib = source_h * source_w * 3 / float(2**30)
        if estimated_gib > float(materialized_read_max_gib):
            skipped_reason = (
                f"materialized source level estimate {estimated_gib:.2f} GiB exceeds "
                f"cap {float(materialized_read_max_gib):.2f} GiB"
            )
        else:
            read_started = time.perf_counter()
            with ETSFile(geometry.ets_path) as ets:
                materialized_source = ets.read_level(geometry.source_level)
            timers.add("materialized_full_level_read_s", time.perf_counter() - read_started)

    replay_blocks: list[np.ndarray] | None = None
    if skipped_reason is None and mode == "replay-cached-rgb-blocks-write":
        replay_blocks = _sample_replay_blocks(geometry, timers=timers, accounting=accounting)

    sizes = {"apparent_bytes": 0, "physical_bytes": 0, "file_count": 0}
    if skipped_reason is None and _mode_is_native_source_pyramid(mode):
        native_unique_tiles = 0
        native_decode_calls = 0
        native_expected_chunks = 0
        for tissue in geometry.tissues:
            tissue_dir = artifact_dir / f"tissue_{tissue.tissue_index:02d}.ome.zarr"
            record = _record_for_benchmark_tissue(tissue)
            native_chunk_limit = (
                len(tissue.blocks) if len(tissue.blocks) < int(tissue.all_block_count) else None
            )
            stats = write_native_ets_tissue_pyramid_ome(
                ets_path=geometry.ets_path,
                out_dir=tissue_dir,
                record=record,
                lr_labels=geometry.lr_labels,
                source_level=geometry.source_level,
                source_shape_yx=geometry.source_shape_yx,
                source_phys_xy_um=geometry.phys_xy_um,
                block_xy=geometry.chunk_xy,
                name=f"benchmark_tissue_{tissue.tissue_index:02d}",
                compressor=codec.compressor,
                sparse_zero_chunks=True,
                store_tissue_mask=True,
                metadata_schema="v0.4",
                requested_mips=tissue.num_mips,
                max_chunks_per_level=native_chunk_limit,
                source_tile_aligned_canvas=_native_mode_uses_aligned_canvas(mode),
                channel_labels=default_channel_labels(3),
                channel_colors=default_channel_colors(3),
            )
            chunks_written += int(stats.get("rgb_chunks_written", 0))
            chunks_skipped += int(stats.get("rgb_chunks_skipped", 0))
            mask_chunks_written += int(stats.get("mask_chunks_written", 0))
            mask_chunks_skipped += int(stats.get("mask_chunks_skipped", 0))
            native_expected_chunks += int(stats.get("rgb_chunks_expected", 0))
            native_decode_calls += int(stats.get("source_tile_decode_calls", 0))
            native_unique_tiles += int(stats.get("unique_source_tiles_touched", 0))
            for key in (
                "rgb_chunk_write_calls",
                "unique_rgb_chunks_written",
                "mask_chunk_write_calls",
                "unique_mask_chunks_written",
            ):
                native_writer_metrics[key] = int(native_writer_metrics.get(key, 0)) + int(
                    stats.get(key, 0)
                )
            for level_info in stats.get("native_pyramid_levels", []):
                shape = level_info.get("output_shape_yx") or [0, 0]
                h, w = map(int, shape)
                raw_output_bytes += int(h * w * 4)
                raw_input_bytes += int(h * w * 3)
        accounting.output_chunks_processed = int(native_expected_chunks)
        accounting.output_chunks_requiring_source_pixels = int(native_expected_chunks)
        accounting.tile_decode_calls = int(native_decode_calls)
        native_accounting_override = {
            "unique_ets_source_tiles_touched": int(native_unique_tiles),
            "total_ets_tile_decode_calls": int(native_decode_calls),
            "estimated_repeated_decode_factor": (
                float(native_decode_calls / native_unique_tiles) if native_unique_tiles else 0.0
            ),
            "output_chunks_processed": int(native_expected_chunks),
            "output_chunks_skipped_before_read": 0,
            "output_chunks_requiring_source_pixels": int(native_expected_chunks),
            "source_tiles_per_output_chunk_mean": 0.0,
            "source_tiles_per_output_chunk_p50": 0.0,
            "source_tiles_per_output_chunk_p95": 0.0,
            "source_tiles_per_output_chunk_max": 0,
            "ideal_source_tiles_per_output_chunk": 1,
            "potential_alignment_win": 0.0,
            "exact_chunks_accounted": int(native_expected_chunks),
            "lightweight_chunks_accounted": 0,
        }
        sizes = _directory_sizes(artifact_dir)
        native_writer_metrics["rgb_write_amplification"] = (
            float(
                native_writer_metrics.get("rgb_chunk_write_calls", 0)
                / native_writer_metrics.get("unique_rgb_chunks_written", 0)
            )
            if native_writer_metrics.get("unique_rgb_chunks_written", 0)
            else 0.0
        )
        native_writer_metrics["mask_write_amplification"] = (
            float(
                native_writer_metrics.get("mask_chunk_write_calls", 0)
                / native_writer_metrics.get("unique_mask_chunks_written", 0)
            )
            if native_writer_metrics.get("unique_mask_chunks_written", 0)
            else 0.0
        )
    elif skipped_reason is None:
        with ETSFile(geometry.ets_path) as ets:
            for tissue in geometry.tissues:
                tissue_dir = artifact_dir / f"tissue_{tissue.tissue_index:02d}.ome.zarr"
                include_mask = _mode_includes_mask(mode)
                include_mips = _mode_includes_mips(mode)
                num_mips = tissue.num_mips if include_mips else 1
                root = None
                arrays: list[Any] = []
                mask_arrays: list[Any] = []
                if _mode_uses_zarr(mode):
                    tissue_dir.mkdir(parents=True, exist_ok=True)
                    root, arrays, mask_arrays = _create_benchmark_arrays(
                        tissue_dir,
                        tissue,
                        num_mips=num_mips,
                        include_mask=include_mask,
                        chunk_xy=geometry.chunk_xy,
                        codec=codec,
                    )

                for block_index, block in enumerate(tissue.blocks):
                    exact = True
                    accounting.note_chunk(
                        source_tiles=block.source_tiles,
                        requires_source=block.requires_source,
                        skipped_before_read=block.skipped_before_read,
                        exact=exact,
                    )
                    raw_input_bytes += block.decoded_source_bytes
                    raw_output_bytes += block.raw_output_bytes_rgb
                    if mode == "ets-read-only":
                        if not block.requires_source:
                            chunks_skipped += 1
                            continue
                        _read_ets_block_yxc(
                            ets,
                            block,
                            level=geometry.source_level,
                            tile_size_yx=geometry.tile_size_yx,
                            timers=timers,
                            accounting=accounting,
                        )
                        chunks_written += 1
                        continue

                    if mode == "synthetic-zero-write":
                        rgb = np.zeros((block.h, block.w, 3), dtype=np.uint8)
                    elif mode == "synthetic-random-write":
                        seed = (
                            geometry.source_level * 1_000_003
                            + tissue.tissue_index * 10_007
                            + block_index
                        )
                        rng = np.random.default_rng(seed)
                        rgb = rng.integers(0, 256, size=(block.h, block.w, 3), dtype=np.uint8)
                    elif mode == "replay-cached-rgb-blocks-write":
                        assert replay_blocks is not None
                        rgb = replay_blocks[block_index % len(replay_blocks)]
                        if rgb.shape[:2] != (block.h, block.w):
                            padded = np.zeros((block.h, block.w, 3), dtype=np.uint8)
                            h = min(block.h, rgb.shape[0])
                            w = min(block.w, rgb.shape[1])
                            padded[:h, :w, :] = rgb[:h, :w, :]
                            rgb = padded
                    elif mode == "materialized-source-crop-write":
                        assert materialized_source is not None
                        crop_started = time.perf_counter()
                        rgb = _read_materialized_block_yxc(materialized_source, block)
                        timers.add("materialized_crop_write_s", time.perf_counter() - crop_started)
                    else:
                        if not block.requires_source:
                            rgb = np.zeros((block.h, block.w, 3), dtype=np.uint8)
                        else:
                            rgb = _read_ets_block_yxc(
                                ets,
                                block,
                                level=geometry.source_level,
                                tile_size_yx=geometry.tile_size_yx,
                                timers=timers,
                                accounting=accounting,
                            )

                    s0_started = time.perf_counter()
                    if arrays:
                        if _write_rgb_block(arrays[0], block, rgb, timers):
                            chunks_written += 1
                        else:
                            chunks_skipped += 1
                    if include_mask:
                        mask = _project_mask_block(geometry, tissue, block, timers)
                        raw_output_bytes += int(block.h * block.w)
                        if mask_arrays:
                            if _write_mask_block(mask_arrays[0], block, mask, timers):
                                mask_chunks_written += 1
                            else:
                                mask_chunks_skipped += 1
                    else:
                        mask = None
                    timers.add("s0_only_s", time.perf_counter() - s0_started)
                    if include_mips:
                        if include_mask and mask is not None:
                            mask_mip_chunks_written += _write_mask_mips(mask_arrays, block, mask, timers)
                        mip_chunks_written += _write_rgb_mips(arrays, block, rgb, timers)
                    del mask

                if root is not None:
                    meta_started = time.perf_counter()
                    _finalize_minimal_metadata(
                        root,
                        tissue=tissue,
                        mode=mode,
                        codec=codec,
                        num_mips=num_mips,
                        phys_xy_um=geometry.phys_xy_um,
                    )
                    timers.add("metadata_finalization_s", time.perf_counter() - meta_started)

        sizes = _directory_sizes(artifact_dir)

    elapsed_s = time.perf_counter() - mode_started
    if mode == "materialized-source-crop-write":
        timers.values["materialized_total_s"] = float(elapsed_s)
        if skipped_reason is None:
            full_read_s = float(timers.values.get("materialized_full_level_read_s", 0.0))
            timers.values["materialized_crop_write_s"] = max(0.0, float(elapsed_s - full_read_s))
    if not keep_artifacts and artifact_dir.exists():
        cleanup_started = time.perf_counter()
        shutil.rmtree(artifact_dir)
        timers.add("artifact_cleanup_s", time.perf_counter() - cleanup_started)
        artifact_deleted_after_run = True
    else:
        artifact_deleted_after_run = False

    accounting_dict = native_accounting_override or accounting.as_dict(
        tile_size_yx=geometry.tile_size_yx,
        chunk_xy=geometry.chunk_xy,
    )
    throughput = {
        "raw_input_MiB_per_sec": (raw_input_bytes / MiB / elapsed_s) if elapsed_s else 0.0,
        "raw_output_MiB_per_sec": (raw_output_bytes / MiB / elapsed_s) if elapsed_s else 0.0,
        "physical_write_MiB_per_sec": (
            sizes["physical_bytes"] / MiB / elapsed_s if elapsed_s else 0.0
        ),
    }
    row = {
        "mode": mode,
        "codec": codec.name if _mode_uses_zarr(mode) else "n/a",
        "run_order": int(run_order),
        "cache_state": cache_state,
        "source_level": int(geometry.source_level),
        "segmentation_level": int(geometry.segmentation_level),
        "output_profile": geometry.output_profile,
        "crop_shape_policy": geometry.crop_shape_policy,
        "configured_extra_margin_px": int(geometry.configured_extra_margin_px),
        "effective_extra_margin_px": int(geometry.effective_extra_margin_px),
        "profile_default_extra_margin_px": int(
            geometry.profile_defaults.get("extra_margin_px", 0)
        ),
        "config_source": geometry.config_source,
        "block_sampling": geometry.block_sampling,
        "block_random_seed": int(geometry.block_random_seed),
        "sampling_summary": geometry.sampling_summary,
        "sample_bias_warnings": geometry.sampling_summary.get("warnings", []),
        "source_tile_aligned_canvas": (
            bool(_native_mode_uses_aligned_canvas(mode)) if _mode_is_native_source_pyramid(mode) else False
        ),
        "number_of_tissues": len(geometry.tissues),
        "number_of_blocks_processed": int(accounting.output_chunks_processed),
        "artifact_deleted_before_run": bool(deleted_before_run),
        "artifact_deleted_after_run": bool(artifact_deleted_after_run),
        "elapsed_s": float(elapsed_s),
        "blocks_per_sec": (
            accounting.output_chunks_processed / elapsed_s if elapsed_s > 0 else 0.0
        ),
        "chunks_written": int(chunks_written),
        "chunks_skipped": int(chunks_skipped),
        "mask_chunks_written": int(mask_chunks_written),
        "mask_chunks_skipped": int(mask_chunks_skipped),
        "mip_chunks_written": int(mip_chunks_written),
        "mask_mip_chunks_written": int(mask_mip_chunks_written),
        "apparent_bytes": int(sizes["apparent_bytes"]),
        "physical_bytes": int(sizes["physical_bytes"]),
        "file_count": int(sizes["file_count"]),
        "raw_input_bytes": int(raw_input_bytes),
        "raw_output_bytes": int(raw_output_bytes),
        "codec_descriptor": codec.descriptor if _mode_uses_zarr(mode) else None,
        "skipped": skipped_reason is not None,
        "skip_reason": skipped_reason,
        "stage_timers": timers.as_dict(),
        "source_tile_accounting": accounting_dict,
        "native_writer_metrics": native_writer_metrics,
        **throughput,
    }
    return _json_ready(row)


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    flat[f"{key}.{subkey}"] = json.dumps(_json_ready(subvalue), sort_keys=True)
                else:
                    flat[f"{key}.{subkey}"] = subvalue
        elif isinstance(value, list):
            flat[key] = json.dumps(_json_ready(value), sort_keys=True)
        else:
            flat[key] = value
    return flat


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = [_flatten_row(row) for row in rows]
    fieldnames: list[str] = []
    for row in flat_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b <= 0:
        return None
    return float(a / b)


def _derive_bottleneck(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if not row.get("skipped")]

    def fastest(mode: str) -> dict[str, Any] | None:
        candidates = [row for row in usable if row.get("mode") == mode]
        if not candidates:
            return None
        return min(candidates, key=lambda item: float(item.get("elapsed_s") or float("inf")))

    materialized = fastest("materialized-source-crop-write")
    direct_mips = fastest("direct-ets-rgb-plus-mask-mips")
    direct_no_mips = fastest("direct-ets-rgb-plus-mask-no-mips")
    replay = fastest("replay-cached-rgb-blocks-write")
    synthetic_zero = fastest("synthetic-zero-write")
    synthetic_random = fastest("synthetic-random-write")
    ets_read = fastest("ets-read-only")
    zstd5 = [
        row
        for row in usable
        if row.get("codec") == "zstd-5-bitshuffle" and "direct-ets" in str(row.get("mode"))
    ]
    lz4 = [
        row
        for row in usable
        if row.get("codec") == "lz4-byte-shuffle" and "direct-ets" in str(row.get("mode"))
    ]
    rules = {
        "materialized_much_faster_than_direct": False,
        "synthetic_replay_writes_slow": False,
        "no_mips_much_faster_than_mips": False,
        "pyramid_write_path_dominates_downsample": False,
        "zstd5_much_slower_than_lz4": False,
        "ets_read_fast_direct_slow": False,
        "repeated_decode_factor_high": False,
        "potential_alignment_win_high": False,
    }
    labels: list[str] = []
    observed: dict[str, Any] = {}
    direct_elapsed = float(direct_mips.get("elapsed_s", 0.0)) if direct_mips else None
    materialized_elapsed = (
        float(materialized.get("elapsed_s", 0.0)) if materialized and not materialized.get("skipped") else None
    )
    if (ratio := _ratio(direct_elapsed, materialized_elapsed)) is not None and ratio >= 1.5:
        rules["materialized_much_faster_than_direct"] = True
        labels.append("ETS decode/access/redecode")
    if replay and synthetic_random:
        replay_bps = float(replay.get("blocks_per_sec", 0.0))
        random_bps = float(synthetic_random.get("blocks_per_sec", 0.0))
        zero_bps = float(synthetic_zero.get("blocks_per_sec", 0.0)) if synthetic_zero else 0.0
        if replay_bps < 5.0 and random_bps < 5.0 and zero_bps < 20.0:
            rules["synthetic_replay_writes_slow"] = True
            labels.append("writer/compression/filesystem")
    if direct_mips and direct_no_mips:
        ratio = _ratio(
            float(direct_mips.get("elapsed_s", 0.0)),
            float(direct_no_mips.get("elapsed_s", 0.0)),
        )
        observed["mips_vs_no_mips_elapsed_ratio"] = ratio
        if ratio is not None and ratio >= 1.25:
            rules["no_mips_much_faster_than_mips"] = True
            timers = direct_mips.get("stage_timers") or {}
            downsample_s = float(timers.get("rgb_mip_downsample_s") or 0.0) + float(
                timers.get("mask_mip_downsample_s") or 0.0
            )
            write_s = float(timers.get("rgb_mip_write_compress_s") or 0.0) + float(
                timers.get("mask_mip_write_compress_s") or 0.0
            )
            observed["pyramid_downsample_s"] = downsample_s
            observed["pyramid_write_compress_s"] = write_s
            if write_s >= downsample_s:
                rules["pyramid_write_path_dominates_downsample"] = True
                labels.append("pyramid write path")
            else:
                labels.append("pyramid downsampling")
    if zstd5 and lz4:
        z_elapsed = min(float(row.get("elapsed_s", 0.0)) for row in zstd5)
        l_elapsed = min(float(row.get("elapsed_s", 0.0)) for row in lz4)
        if (ratio := _ratio(z_elapsed, l_elapsed)) is not None and ratio >= 1.25:
            rules["zstd5_much_slower_than_lz4"] = True
            labels.append("compression")
    if ets_read and direct_mips:
        read_bps = float(ets_read.get("blocks_per_sec", 0.0))
        direct_bps = float(direct_mips.get("blocks_per_sec", 0.0))
        if read_bps > direct_bps * 1.5:
            rules["ets_read_fast_direct_slow"] = True
            labels.append("block assembly/scheduling/writer interaction")
    max_redecode = 0.0
    max_alignment_win = 0.0
    sample_bias_warnings: set[str] = set()
    for row in usable:
        accounting = row.get("source_tile_accounting") or {}
        max_redecode = max(
            max_redecode,
            float(accounting.get("estimated_repeated_decode_factor") or 0.0),
        )
        max_alignment_win = max(
            max_alignment_win,
            float(accounting.get("potential_alignment_win") or 0.0),
        )
        sample_bias_warnings.update(str(item) for item in row.get("sample_bias_warnings", []))
    if max_redecode >= 2.0:
        rules["repeated_decode_factor_high"] = True
        labels.append("missing ETS tile cache candidate")
    if max_alignment_win >= 1.75:
        rules["potential_alignment_win_high"] = True
        labels.append("source-tile-aligned canvas candidate")
    label = labels[0] if labels else "inconclusive"
    return {
        "top_bottleneck_category": label,
        "bottleneck_candidates": list(dict.fromkeys(labels)),
        "rules": rules,
        "thresholds": {
            "materialized_vs_direct_ratio": 1.5,
            "mips_vs_no_mips_ratio": 1.25,
            "zstd5_vs_lz4_ratio": 1.25,
            "repeated_decode_factor": 2.0,
            "potential_alignment_win": 1.75,
        },
        "observed": {
            "max_repeated_decode_factor": max_redecode,
            "max_potential_alignment_win": max_alignment_win,
            "sample_bias_warnings": sorted(sample_bias_warnings),
            **observed,
        },
    }


def _normalize_modes(modes: Iterable[str] | None) -> list[str]:
    selected = [str(mode).strip() for mode in (modes or ()) if str(mode).strip()]
    if not selected:
        return list(REQUIRED_BENCHMARK_MODES)
    invalid = sorted(set(selected) - set(REQUIRED_BENCHMARK_MODES))
    if invalid:
        raise ValueError(f"Unknown benchmark mode(s): {', '.join(invalid)}")
    return selected


def _normalize_codecs(codecs: Iterable[str] | None, *, default_codec: str) -> list[str]:
    selected = [str(codec).strip().lower().replace("_", "-") for codec in (codecs or ()) if str(codec).strip()]
    if not selected:
        selected = [default_codec]
    invalid = sorted(set(selected) - set(BENCHMARK_CODEC_CHOICES))
    if invalid:
        raise ValueError(f"Unknown benchmark codec(s): {', '.join(invalid)}")
    return selected


def _read_external_baselines(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        raise ValueError("external baseline JSON must be an object, a list, or {'rows': [...]}.")
    out = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"external baseline row {idx} is not an object.")
        out.append({"mode": row.get("mode", "external-baseline"), "external_baseline": True, **row})
    return out


def run_vsi_transcode_benchmark(
    vsi_path: str | Path,
    benchmark_dir: str | Path,
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
    modes: Iterable[str] | None = None,
    codecs: Iterable[str] | None = None,
    max_tissues: int | None = None,
    max_blocks: int | None = None,
    block_sampling: str = "first",
    block_random_seed: int = 0,
    keep_artifacts: bool = False,
    materialized_read_max_gib: float = 8.0,
    warm_cache: bool = False,
    profile_cpu: bool = False,
    progress_interval_s: float = 30.0,
    external_baseline_json: str | Path | None = None,
    min_side_for_mips: int | None = None,
    config_source: str | None = None,
) -> dict[str, Any]:
    del progress_interval_s
    vsi_path = Path(vsi_path)
    benchmark_dir = Path(benchmark_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    config = pipeline_config or PipelineConfig()
    seg_cfg = segmentation_config or config.segmentation
    tile_cfg = tile_config or config.tiles
    geometry_started = time.perf_counter()
    geometry = _resolve_benchmark_geometry(
        vsi_path=vsi_path,
        source_level=source_level,
        segmentation_level=segmentation_level,
        output_profile=output_profile,
        tile_frame_level=tile_frame_level,
        crop_shape_policy=crop_shape_policy,
        segmentation_config=seg_cfg,
        tile_config=tile_cfg,
        metadata_backend=metadata_backend,
        metadata_schema=metadata_schema,
        max_tissues=max_tissues,
        max_blocks=max_blocks,
        block_sampling=block_sampling,
        block_random_seed=block_random_seed,
        config_source=config_source or "programmatic/default",
        min_side_for_mips=min_side_for_mips,
    )
    geometry_elapsed = time.perf_counter() - geometry_started
    default_codec = _benchmark_codec_from_production_compression(geometry.compression_default)
    selected_modes = _normalize_modes(modes)
    selected_codecs = _normalize_codecs(codecs, default_codec=default_codec)

    resolved_config = {
        "pipeline_config": config.to_dict(),
        "segmentation_config": seg_cfg.model_dump(),
        "tile_config": tile_cfg.model_dump(),
        "benchmark_options": {
            "vsi_path": str(vsi_path),
            "benchmark_dir": str(benchmark_dir),
            "source_level": int(geometry.source_level),
            "segmentation_level": int(geometry.segmentation_level),
            "output_profile": geometry.output_profile,
            "tile_frame_level": geometry.tile_frame_level,
            "crop_shape_policy": geometry.crop_shape_policy,
            "metadata_backend": metadata_backend,
            "metadata_schema": metadata_schema,
            "modes": selected_modes,
            "codecs": selected_codecs,
            "max_tissues": max_tissues,
            "max_blocks": max_blocks,
            "block_sampling": geometry.block_sampling,
            "block_random_seed": int(geometry.block_random_seed),
            "keep_artifacts": keep_artifacts,
            "materialized_read_max_gib": materialized_read_max_gib,
            "warm_cache": warm_cache,
            "profile_cpu": profile_cpu,
            "min_side_for_mips": min_side_for_mips,
            "config_source": geometry.config_source,
            "profile_defaults": geometry.profile_defaults,
            "configured_extra_margin_px": int(geometry.configured_extra_margin_px),
            "effective_extra_margin_px": int(geometry.effective_extra_margin_px),
        },
    }

    environment = _capture_environment(Path.cwd())
    geometry_json = _geometry_to_json(geometry)
    geometry_json["geometry_setup_elapsed_s"] = float(geometry_elapsed)

    (benchmark_dir / "resolved_config.json").write_text(
        json.dumps(_json_ready(resolved_config), indent=2),
        encoding="utf-8",
    )
    (benchmark_dir / "geometry.json").write_text(
        json.dumps(_json_ready(geometry_json), indent=2),
        encoding="utf-8",
    )
    (benchmark_dir / "environment.json").write_text(
        json.dumps(_json_ready(environment), indent=2),
        encoding="utf-8",
    )

    warm_cache_info = None
    if warm_cache:
        warm_cache_info = _warm_filesystem_cache(geometry)
    cache_state = "warm-cache-requested" if warm_cache else "cache-state-not-controlled"

    rows: list[dict[str, Any]] = []
    run_order = 1
    for mode in selected_modes:
        mode_codecs = selected_codecs if _mode_uses_zarr(mode) else ["none"]
        for codec_name in mode_codecs:
            codec = _compressor_for_benchmark_codec(codec_name)
            logger.info(
                "Running VSI transcode benchmark mode=%s codec=%s source_level=%s.",
                mode,
                codec.name,
                geometry.source_level,
            )
            if profile_cpu:
                profile_path = benchmark_dir / f"profile_{run_order:03d}_{mode}_{codec.name}.prof"
                profiler = cProfile.Profile()
                row = profiler.runcall(
                    _run_single_mode,
                    geometry=geometry,
                    benchmark_dir=benchmark_dir,
                    run_order=run_order,
                    mode=mode,
                    codec=codec,
                    keep_artifacts=keep_artifacts,
                    materialized_read_max_gib=materialized_read_max_gib,
                    cache_state=cache_state,
                )
                profiler.dump_stats(str(profile_path))
                row["cpu_profile_path"] = str(profile_path)
            else:
                row = _run_single_mode(
                    geometry=geometry,
                    benchmark_dir=benchmark_dir,
                    run_order=run_order,
                    mode=mode,
                    codec=codec,
                    keep_artifacts=keep_artifacts,
                    materialized_read_max_gib=materialized_read_max_gib,
                    cache_state=cache_state,
                )
            rows.append(row)
            run_order += 1

    rows.extend(_read_external_baselines(Path(external_baseline_json) if external_baseline_json else None))
    decision = _derive_bottleneck(rows)
    result = {
        "schema_version": "v0.2",
        "summary": {
            "vsi_path": str(vsi_path),
            "ets_path": str(geometry.ets_path),
            "source_level": int(geometry.source_level),
            "segmentation_level": int(geometry.segmentation_level),
            "output_profile": geometry.output_profile,
            "crop_shape_policy": geometry.crop_shape_policy,
            "profile_defaults": geometry.profile_defaults,
            "configured_extra_margin_px": int(geometry.configured_extra_margin_px),
            "effective_extra_margin_px": int(geometry.effective_extra_margin_px),
            "config_source": geometry.config_source,
            "modes": selected_modes,
            "codecs": selected_codecs,
            "max_tissues": max_tissues,
            "max_blocks": max_blocks,
            "block_sampling": geometry.block_sampling,
            "block_random_seed": int(geometry.block_random_seed),
            "sampling_summary": geometry.sampling_summary,
            "warm_cache": bool(warm_cache),
            "warm_cache_info": warm_cache_info,
            "keep_artifacts": bool(keep_artifacts),
            "geometry_setup_elapsed_s": float(geometry_elapsed),
        },
        "decision_rules": decision,
        "rows": rows,
        "artifacts": {
            "resolved_config_json": str(benchmark_dir / "resolved_config.json"),
            "geometry_json": str(benchmark_dir / "geometry.json"),
            "environment_json": str(benchmark_dir / "environment.json"),
            "benchmark_json": str(benchmark_dir / "benchmark.json"),
            "benchmark_csv": str(benchmark_dir / "benchmark.csv"),
        },
    }
    result = _json_ready(result)
    (benchmark_dir / "benchmark.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    _write_csv(benchmark_dir / "benchmark.csv", rows)
    return result


__all__ = [
    "BENCHMARK_CODEC_CHOICES",
    "REQUIRED_BENCHMARK_MODES",
    "_compressor_for_benchmark_codec",
    "_derive_bottleneck",
    "run_vsi_transcode_benchmark",
]
