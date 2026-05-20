"""VSI/ETS entry points for source and tissue OME-Zarr generation."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import dask.array as da
import dask.config
import numpy as np
from scipy.ndimage import affine_transform, binary_fill_holes
from skimage import measure

from ..config import SegmentationConfig, TileConfig
from ..etsfile import ETSFile
from ..omezarr.ets_writer import write_ets_pyramid_to_ngff_zarr
from ..omezarr.metadata import default_channel_colors, default_channel_labels
from ..omezarr.pyramid import build_mips_from_yxc, compute_num_mips_min_side
from ..omezarr.streaming import write_ngff_from_tile_streaming_ome
from ..omezarr.writers import write_ngff_from_mips_ngffzarr
from ..precomputed.plate_writer import PlatePrecomputedWriter
from ..tiles.generator import TissueTileRecord, sort_labels_left_to_right
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


def _read_masked_ets_block(
    block: np.ndarray,
    *,
    ets_path: str,
    source_level: int,
    source_shape_yx: tuple[int, int],
    tile_origin_yx: tuple[int, int],
    source_crop_bounds_yx: tuple[int, int, int, int],
    lr_labels: np.ndarray,
    label_id: int,
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
    out = np.zeros((block_h, block_w, block.shape[2]), dtype=block.dtype)

    tile_y0, tile_x0 = map(int, tile_origin_yx)
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

    region = _read_ets_region_yxc(
        ets_path,
        level=source_level,
        x0=valid_x0,
        y0=valid_y0,
        x1=valid_x1,
        y1=valid_y1,
    )

    yr = source_h / lr_labels.shape[0]
    xr = source_w / lr_labels.shape[1]
    crop_y0, crop_x0, crop_y1, crop_x1 = map(int, source_crop_bounds_yx)
    lr_crop_y0 = max(0, int(np.floor(crop_y0 / yr)))
    lr_crop_x0 = max(0, int(np.floor(crop_x0 / xr)))
    lr_crop_y1 = min(lr_labels.shape[0], int(np.ceil(crop_y1 / yr)))
    lr_crop_x1 = min(lr_labels.shape[1], int(np.ceil(crop_x1 / xr)))
    lr_crop = lr_labels[lr_crop_y0:lr_crop_y1, lr_crop_x0:lr_crop_x1].astype(np.int32)
    matrix = np.array([[1.0 / yr, 0.0], [0.0, 1.0 / xr]], dtype=float)
    offset = np.array(
        [(valid_y0 / yr) - lr_crop_y0, (valid_x0 / xr) - lr_crop_x0],
        dtype=float,
    )
    mask = (
        affine_transform(
            lr_crop,
            matrix=matrix,
            offset=offset,
            output_shape=(valid_y1 - valid_y0, valid_x1 - valid_x0),
            order=0,
        )
        == int(label_id)
    )
    region = np.where(mask[..., None], region, 0)

    dst_y0 = valid_y0 - source_y0
    dst_x0 = valid_x0 - source_x0
    out[dst_y0 : dst_y0 + region.shape[0], dst_x0 : dst_x0 + region.shape[1], :] = region
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
) -> tuple[list[TissueTileRecord], int]:
    """Build lazy per-tissue tile records that read directly from ETS source blocks."""
    source_h, source_w = map(int, source_shape_yx)
    lr_labels = _filled_lr_labels(low_res_filled)
    if lr_labels.max() == 0:
        return [], 0

    if pad_multiple is None:
        pad_multiple = chunk
    yr = source_h / low_res_filled.shape[0]
    xr = source_w / low_res_filled.shape[1]

    roi_specs: list[tuple[int, int, int, int, int, int, int, int]] = []
    max_side = 0
    for lid in sort_labels_left_to_right(lr_labels):
        lr_mask = lr_labels == lid
        rows, cols = np.any(lr_mask, axis=1), np.any(lr_mask, axis=0)
        yi = np.where(rows)[0]
        xi = np.where(cols)[0]
        if yi.size == 0 or xi.size == 0:
            continue

        y0_lr, y1_lr = yi[[0, -1]]
        x0_lr, x1_lr = xi[[0, -1]]
        y0_src = max(0, min(source_h, int(np.floor(y0_lr * yr))))
        y1_src = max(0, min(source_h, int(np.ceil((y1_lr + 1) * yr))))
        x0_src = max(0, min(source_w, int(np.floor(x0_lr * xr))))
        x1_src = max(0, min(source_w, int(np.ceil((x1_lr + 1) * xr))))
        max_side = max(
            max_side,
            (y1_src - y0_src) + (2 * extra_margin_px),
            (x1_src - x0_src) + (2 * extra_margin_px),
        )
        roi_specs.append((int(lid), int(y0_lr), int(y1_lr), int(x0_lr), int(x1_lr), y0_src, y1_src, x0_src, x1_src))

    if not roi_specs:
        return [], 0

    tile_dim = ((max_side + pad_multiple - 1) // pad_multiple) * pad_multiple
    records: list[TissueTileRecord] = []
    for tissue_index, (lid, _y0_lr, _y1_lr, _x0_lr, _x1_lr, y0_src, y1_src, x0_src, x1_src) in enumerate(roi_specs):
        center_y = (y0_src + y1_src) / 2.0
        center_x = (x0_src + x1_src) / 2.0
        tile_y0 = int(np.floor(center_y - tile_dim / 2.0))
        tile_x0 = int(np.floor(center_x - tile_dim / 2.0))
        tile_y1 = tile_y0 + tile_dim
        tile_x1 = tile_x0 + tile_dim

        src_y0 = max(0, tile_y0)
        src_x0 = max(0, tile_x0)
        src_y1 = min(source_h, tile_y1)
        src_x1 = min(source_w, tile_x1)
        if src_y1 <= src_y0 or src_x1 <= src_x0:
            continue

        lr_crop_y0 = max(0, int(np.floor(src_y0 / yr)))
        lr_crop_x0 = max(0, int(np.floor(src_x0 / xr)))
        lr_crop_y1 = min(lr_labels.shape[0], int(np.ceil(src_y1 / yr)))
        lr_crop_x1 = min(lr_labels.shape[1], int(np.ceil(src_x1 / xr)))

        chunks = (min(chunk, tile_dim), min(chunk, tile_dim), 3)
        dummy = da.zeros((tile_dim, tile_dim, 3), chunks=chunks, dtype=np.uint8)
        tile = dummy.map_blocks(
            _read_masked_ets_block,
            dtype=np.uint8,
            ets_path=str(ets_path),
            source_level=int(source_level),
            source_shape_yx=(source_h, source_w),
            tile_origin_yx=(tile_y0, tile_x0),
            source_crop_bounds_yx=(src_y0, src_x0, src_y1, src_x1),
            lr_labels=lr_labels,
            label_id=int(lid),
        )
        records.append(
            TissueTileRecord(
                tile=tile,
                tissue_index=int(tissue_index),
                label_id=int(lid),
                crop_bounds_source_level=(int(src_x0), int(src_y0), int(src_x1), int(src_y1)),
                crop_bounds_segmentation_level=(
                    int(lr_crop_x0),
                    int(lr_crop_y0),
                    int(lr_crop_x1),
                    int(lr_crop_y1),
                ),
                tile_dim=int(tile_dim),
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


def diagnose_vsi_replating(
    vsi_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    flat_image_path: str | Path | None = None,
    source_level: int | str = 7,
    segmentation_level: int | str | None = 7,
    segmentation_config: SegmentationConfig | None = None,
    tile_config: TileConfig | None = None,
) -> dict[str, Any]:
    """
    Read only the requested ETS levels and report segmentation/crop diagnostics.

    This is a no-full-rerun smoke helper for comparing the notebook flat-image
    route with the VSI/ETS replating route. When ``output_dir`` is supplied, it
    writes ``diagnostics.json`` plus mask-boundary overlay PNGs.
    """
    vsi_path = Path(vsi_path)
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    config = segmentation_config or SegmentationConfig()
    tile_cfg = tile_config or TileConfig()

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
    tile_records, tile_dim = _direct_ets_tissue_tile_records(
        ets_path=ets_path,
        source_level=source_idx,
        source_shape_yx=source_shape_yx,
        low_res_filled=ets_mask,
        chunk=tile_cfg.chunk_size,
        pad_multiple=tile_cfg.pad_multiple,
        extra_margin_px=tile_cfg.extra_margin_px,
    )

    flat_summary: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
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

    result: dict[str, Any] = {
        "vsi_path": str(vsi_path),
        "ets_path": str(ets_path),
        "source_level": int(source_idx),
        "segmentation_level": int(segmentation_idx),
        "source_shape_yx": list(source_shape_yx),
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
        "tile_records": [
            {
                "tissue_index": int(record.tissue_index),
                "label_id": int(record.label_id),
                "crop_bounds_source_level": list(record.crop_bounds_source_level),
                "crop_bounds_segmentation_level": list(record.crop_bounds_segmentation_level),
                "tile_dim": int(record.tile_dim),
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
    tile_extra_margin_px: int = 0,
    dtype: np.dtype | str | None = "uint8",
) -> list[Path]:
    """
    Segment a VSI/ETS pyramid and write per-tissue OME-Zarr derivatives without
    materializing a full-slide source OME-Zarr.
    """
    del parallel  # Direct ETS block reads are deliberately scheduled locally.
    vsi_path = Path(vsi_path)
    out_ngff_dir = Path(out_ngff_dir)
    out_ngff_dir.mkdir(parents=True, exist_ok=True)

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
    else:
        tile_pad_multiple = plate_chunk_xy

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
        chunks=(3, min(plate_chunk_xy, segmentation_yxc.shape[0]), min(plate_chunk_xy, segmentation_yxc.shape[1])),
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
    with dask.config.set({"scheduler": "threads", "array.slicing.split_large_chunks": True}):
        for z_idx, record in enumerate(tile_records):
            tile_dask = record.tile
            name = f"{vsi_path.stem}_tissue_{record.tissue_index:02d}"
            ngff_dir = out_ngff_dir / f"{name}.ome.zarr"
            if any_big:
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
                write_ngff_from_tile_streaming_ome(
                    tile_yxc_da=tlazy,
                    out_dir=ngff_dir,
                    phys_xy_um=(px_um, py_um),
                    block_xy=plate_chunk_xy,
                    num_mips=num_mips,
                    name=name,
                    compressor=None,
                    channel_labels=default_channel_labels(int(tile_dask.shape[2])),
                    channel_colors=default_channel_colors(int(tile_dask.shape[2])),
                    ngff_metadata=tile_ngff_metadata,
                    metadata_schema=source_metadata_schema,
                )
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
                source_context=source_context,
                source_ome_zarr=None,
                source_level=source_idx,
                segmentation_level=segmentation_idx,
                phys_xy_um=(px_um, py_um),
            )

    logger.info("Wrote %d direct ETS tissue OME-Zarrs to %s", len(out_paths), out_ngff_dir)
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
    segmentation_config: SegmentationConfig | None = None,
    tile_config: TileConfig | None = None,
    metadata_backend: str = "auto",
    metadata_schema: str = "v0.4",
    overwrite_source: bool = False,
    source_writer: str = "direct",
    materialize_source: bool = False,
    parallel: bool = False,
    min_side_for_mips: int | None = None,
    dtype: np.dtype | str | None = "uint8",
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
                segmentation_config=segmentation_config,
                tile_config=tile_config,
                parallel=parallel,
                min_side_for_mips=min_side_for_mips,
                dtype=dtype,
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
                segmentation_config=segmentation_config,
                tile_config=tile_config,
                metadata_backend=metadata_backend,
                metadata_schema=metadata_schema,
                parallel=parallel,
                min_side_for_mips=min_side_for_mips,
                dtype=dtype,
            )
        results[str(vsi_path)] = tissue_paths

    return results
