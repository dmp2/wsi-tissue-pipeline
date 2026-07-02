"""
Quality Control Grid Generation

Creates contact sheets and thumbnail grids for visual quality control.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
from PIL import Image, ImageDraw, ImageFont, ImageOps

logger = logging.getLogger(__name__)

NGFF_THUMBNAIL_FULL_RES_MAX_PIXELS = 25_000_000
NGFF_STATS_FULL_RES_MAX_PIXELS = 50_000_000
OMETIFF_QC_MAX_PIXELS = 4_000_000
OMETIFF_QC_WINDOW_SIZE = 512
OMETIFF_QC_WINDOW_GRID = 3

# Try to import torch for faster grid generation
try:
    import torch
    from torchvision.utils import make_grid

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Filename pattern for parsed tissue images.
# The slide and slice fields are usually zero-padded, but some local/demo
# workflows can emit wider numeric fields (for example, slide 001).
DEFAULT_PATTERN = re.compile(
    r"^(?P<prefix>.+?)_(?P<slide>\d+)_(?P<slice_on_slide>\d+)_(?P<overall>\d+)"
    r"(?:\.(?P<ext>tif|tiff|png|jpg|jpeg))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QCRecord:
    relative_path: str
    filename: str
    source_image: str
    tile_index_on_source: int
    overall_index: int
    overall_label: str
    width: int
    height: int
    component_qc: dict[str, Any] | None = None

    def path(self, input_dir: str | Path) -> Path:
        return Path(input_dir) / self.relative_path

    @property
    def artifact_likely(self) -> bool:
        if not self.component_qc:
            return False
        return bool(self.component_qc.get("artifact_likely", False))

    @property
    def artifact_reason(self) -> str:
        if not self.component_qc:
            return ""
        return str(self.component_qc.get("artifact_reason", ""))


def _make_qc_record(**kwargs: Any) -> QCRecord:
    """
    Construct a QCRecord, tolerating stale notebook kernels during autoreload.

    IPython can occasionally reload functions while leaving an older dataclass
    object in module globals.  In that state, the old QCRecord constructor does
    not accept ``component_qc`` even though the loader function now passes it.
    """
    try:
        return QCRecord(**kwargs)
    except TypeError as exc:
        if "component_qc" not in str(exc):
            raise
        component_qc = kwargs.pop("component_qc", None)
        record = QCRecord(**kwargs)
        object.__setattr__(record, "component_qc", component_qc)
        return record


def _record_component_qc(record: QCRecord) -> dict[str, Any] | None:
    value = getattr(record, "component_qc", None)
    return value if isinstance(value, dict) else None


def _record_artifact_likely(record: QCRecord) -> bool:
    component_qc = _record_component_qc(record)
    if component_qc is None:
        return False
    return bool(component_qc.get("artifact_likely", False))


def _record_artifact_reason(record: QCRecord) -> str:
    component_qc = _record_component_qc(record)
    if component_qc is None:
        return ""
    return str(component_qc.get("artifact_reason", ""))


@dataclass(frozen=True)
class QCArtifacts:
    master_contact_sheet: Path | None
    per_slide_grids: list[Path]
    stats_csv: Path | None
    records_manifest: Path | None


@dataclass(frozen=True)
class QCWorkflowResult:
    records: list[QCRecord]
    artifacts: QCArtifacts


@dataclass(frozen=True)
class NGFFLevelSelection:
    dataset_path: str
    axes: tuple[str, ...] | None
    raw_shape: tuple[int, ...]
    array_yxc: np.ndarray


@dataclass(frozen=True)
class OmeTiffQCSelection:
    array_yxc: np.ndarray
    method: str
    level_index: int
    level_shape_yx: tuple[int, int]
    raw_shape: tuple[int, ...]
    windows: list[dict[str, int]]


def find_images(
    root: Path,
    pattern: re.Pattern | None = None,
    extensions: tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
) -> list[Path]:
    """
    Find all images matching a pattern in a directory.

    Parameters
    ----------
    root : Path
        Directory to search.
    pattern : re.Pattern, optional
        Regex pattern for filename matching.
    extensions : tuple
        Valid file extensions.

    Returns
    -------
    list of Path
        Matching image files.
    """
    root = Path(root)
    if pattern is None:
        pattern = DEFAULT_PATTERN

    images = []
    for ext in extensions:
        for p in root.glob(f"*{ext}"):
            if pattern.match(p.name):
                images.append(p)

    return sorted(set(images))


def parse_filename(
    fname: str,
    pattern: re.Pattern | None = None,
) -> dict[str, str | int] | None:
    """
    Parse tissue image filename into components.

    Parameters
    ----------
    fname : str
        Filename to parse.
    pattern : re.Pattern, optional
        Regex pattern with named groups.

    Returns
    -------
    dict or None
        Parsed components or None if no match.
    """
    if pattern is None:
        pattern = DEFAULT_PATTERN

    m = pattern.match(fname)
    if not m:
        return None

    groups = m.groupdict()
    return {
        "prefix": groups.get("prefix", ""),
        "slide": int(groups["slide"]) if "slide" in groups else 0,
        "slide_raw": groups.get("slide", ""),
        "slice_on_slide": int(groups["slice_on_slide"]) if "slice_on_slide" in groups else 0,
        "slice_on_slide_raw": groups.get("slice_on_slide", ""),
        "overall": int(groups["overall"]) if "overall" in groups else 0,
        "overall_raw": groups.get("overall", ""),
        "ext": (groups.get("ext") or "").lower(),
    }


def group_by_slide(
    paths: list[Path],
    pattern: re.Pattern | None = None,
) -> dict[int, list[tuple[int, int, Path]]]:
    """
    Group images by slide number.

    Parameters
    ----------
    paths : list of Path
        Image file paths.
    pattern : re.Pattern, optional
        Filename pattern.

    Returns
    -------
    dict
        {slide_num: [(slice_idx, overall_idx, Path), ...]}
    """
    grouped: dict[int, list[tuple[int, int, Path]]] = {}

    for p in paths:
        parsed = parse_filename(p.name, pattern)
        if parsed is None:
            continue

        slide = int(parsed["slide"])
        grouped.setdefault(slide, []).append(
            (int(parsed["slice_on_slide"]), int(parsed["overall"]), p)
        )

    # Sort within each slide
    for slide in grouped:
        grouped[slide].sort(key=lambda x: (x[0], x[1]))

    return dict(sorted(grouped.items()))


def load_thumbnail(
    path: Path,
    size: int = 256,
    mode: str = "RGB",
    qc_display_mode: str = "auto",
    qc_masked_background: str = "black",
) -> Image.Image:
    """
    Load and resize image to thumbnail.

    Parameters
    ----------
    path : Path
        Image path.
    size : int
        Maximum size for shortest edge.
    mode : str
        PIL image mode.

    Returns
    -------
    PIL.Image
        Thumbnail image.
    """
    if _is_ome_zarr_path(path):
        return _load_ngff_thumbnail(
            path,
            size=size,
            mode=mode,
            qc_display_mode=qc_display_mode,
            qc_masked_background=qc_masked_background,
        )
    if _is_ome_tiff_path(path):
        return _load_ometiff_thumbnail(
            path,
            size=size,
            mode=mode,
            qc_display_mode=qc_display_mode,
            qc_masked_background=qc_masked_background,
        )

    with Image.open(path) as img:
        if img.mode != mode:
            img = img.convert(mode)
        img.thumbnail((size, size), Image.BICUBIC)
        return img.copy()


def _is_ome_zarr_path(path: Path) -> bool:
    return path.is_dir() and path.name.endswith(".ome.zarr")


def _is_ome_tiff_path(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and (name.endswith(".ome.tif") or name.endswith(".ome.tiff"))


def _ngff_dataset_paths(path: Path) -> list[str]:
    root = zarr.open_group(str(path), mode="r")
    multiscales = root.attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError(f"{path} is not a readable OME-Zarr image group.")
    datasets = multiscales[0].get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"{path} does not contain NGFF datasets.")
    return [str(dataset["path"]) for dataset in datasets]


def _ngff_axes(path: Path) -> tuple[str, ...] | None:
    """Return NGFF axis names for the first multiscales entry when available."""
    root = zarr.open_group(str(path), mode="r")
    multiscales = root.attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        return None
    axes = multiscales[0].get("axes")
    if not isinstance(axes, list):
        return None
    names: list[str] = []
    for axis in axes:
        if isinstance(axis, dict):
            name = axis.get("name")
        else:
            name = axis
        if name is None:
            return None
        names.append(str(name).lower())
    return tuple(names)


def _array_yx_shape(shape: tuple[int, ...], axes: tuple[str, ...] | None = None) -> tuple[int, int]:
    if axes is not None and len(axes) == len(shape) and "y" in axes and "x" in axes:
        return int(shape[axes.index("y")]), int(shape[axes.index("x")])
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) != 3:
        raise ValueError(f"Expected 2D or 3D image array, got shape {shape}.")
    if shape[0] in (1, 3, 4) and shape[-1] not in (1, 3, 4):
        return int(shape[1]), int(shape[2])
    return int(shape[0]), int(shape[1])


def _to_yxc_array(arr: np.ndarray, axes: tuple[str, ...] | None = None) -> np.ndarray:
    """Normalize NGFF/image arrays to YXC or YX before display/statistics."""
    arr = np.asarray(arr)
    if axes is not None and len(axes) == arr.ndim and "y" in axes and "x" in axes:
        y_axis = axes.index("y")
        x_axis = axes.index("x")
        c_axis = axes.index("c") if "c" in axes else None
        if c_axis is None:
            return np.moveaxis(arr, (y_axis, x_axis), (0, 1))
        return np.moveaxis(arr, (y_axis, x_axis, c_axis), (0, 1, 2))

    if arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4) and not (
            arr.shape[0] in (1, 3, 4) and arr.shape[1] > 4
        ):
            return arr
        if arr.shape[0] in (1, 3, 4):
            return np.moveaxis(arr, 0, -1)
    return arr


def _to_display_uint8(arr: np.ndarray, axes: tuple[str, ...] | None = None) -> np.ndarray:
    arr = _to_yxc_array(arr, axes)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return (arr.astype(np.float32) / max(1, info.max) * 255.0).clip(0, 255).astype(np.uint8)

    arr_f = arr.astype(np.float32)
    finite = np.isfinite(arr_f)
    if finite.any() and float(np.nanmax(arr_f)) <= 1.0:
        arr_f = arr_f * 255.0
    return np.nan_to_num(arr_f, nan=0.0).clip(0, 255).astype(np.uint8)


def _normalize_qc_display_mode(qc_display_mode: str | None) -> str:
    normalized = str(qc_display_mode or "auto").strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "raw": "raw_rgb",
        "raw_rgb": "raw_rgb",
        "rgb": "raw_rgb",
        "mask": "mask",
        "tissue_mask": "mask",
        "masked": "masked_rgb",
        "masked_rgb": "masked_rgb",
        "triptych": "triptych",
    }
    if normalized not in aliases:
        raise ValueError(
            "qc_display_mode must be one of 'auto', 'raw_rgb', 'mask', 'masked_rgb', or 'triptych'."
        )
    return aliases[normalized]


def _normalize_qc_masked_background(qc_masked_background: str | None) -> str:
    normalized = str(qc_masked_background or "black").strip().lower().replace("-", "_")
    aliases = {"black": "black", "zero": "black", "0": "black", "white": "white", "255": "white"}
    if normalized not in aliases:
        raise ValueError("qc_masked_background must be one of 'black' or 'white'.")
    return aliases[normalized]


def _ngff_primary_rgb_mode(path: Path) -> str | None:
    try:
        root = zarr.open_group(str(path), mode="r")
        mode = root.attrs.get("primary_rgb_mode")
        if mode is not None:
            return str(mode)
    except Exception:
        pass
    manifest_path = path / "tissue_manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            mode = payload.get("primary_rgb_mode")
            if mode is not None:
                return str(mode)
        except Exception:
            return None
    return None


def _load_ngff_mask_for_dataset(path: Path, dataset_path: str) -> np.ndarray | None:
    mask_path = path / "labels" / "tissue_mask" / dataset_path
    if not mask_path.exists():
        return None
    mask_group_path = path / "labels" / "tissue_mask"
    try:
        axes = _ngff_axes(mask_group_path)
    except Exception:
        axes = ("y", "x")
    arr = zarr.open_array(str(mask_path), mode="r")
    mask = _to_yxc_array(np.asarray(arr[...]), axes)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(bool)


def _mask_rgb_for_display(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    *,
    qc_masked_background: str = "black",
) -> np.ndarray:
    if mask is None:
        return rgb
    background = 255 if _normalize_qc_masked_background(qc_masked_background) == "white" else 0
    if rgb.ndim == 2:
        return np.where(mask, rgb, background).astype(rgb.dtype)
    return np.where(mask[..., None], rgb, background).astype(rgb.dtype)


def _thumbnail_array(arr: np.ndarray, *, size: int, mode: str) -> Image.Image:
    img = Image.fromarray(_to_display_uint8(arr))
    if img.mode != mode:
        img = img.convert(mode)
    img.thumbnail((size, size), Image.BICUBIC)
    return img.copy()



def _import_tifffile_for_qc():
    import importlib

    return importlib.import_module("tifffile")


def _ome_tiff_levels(path: Path):
    tifffile_mod = _import_tifffile_for_qc()
    tif = tifffile_mod.TiffFile(str(path))
    series = tif.series[0]
    levels = list(getattr(series, "levels", [series]))
    return tif, levels


def _ome_tiff_yx_from_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[0] in (1, 3, 4) and shape[-1] not in (1, 3, 4):
        return int(shape[1]), int(shape[2])
    if len(shape) == 3:
        return int(shape[0]), int(shape[1])
    raise ValueError(f"Unsupported OME-TIFF shape {shape}.")


def _ome_tiff_yx_shape(path: Path) -> tuple[int, int]:
    tif, levels = _ome_tiff_levels(path)
    try:
        return _ome_tiff_yx_from_shape(tuple(levels[0].shape))
    finally:
        tif.close()


def _ome_tiff_window_key(
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
    raise ValueError(f"Unsupported OME-TIFF shape {shape}.")


def _stratified_windows(
    y: int,
    x: int,
    *,
    window_size: int = OMETIFF_QC_WINDOW_SIZE,
    grid: int = OMETIFF_QC_WINDOW_GRID,
) -> list[dict[str, int]]:
    y = int(y)
    x = int(x)
    win_y = min(int(window_size), y)
    win_x = min(int(window_size), x)
    if y <= 0 or x <= 0:
        return []
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


def _mosaic_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    arrays = [_to_yxc_array(np.asarray(arr)) for arr in arrays]
    max_h = max(int(arr.shape[0]) for arr in arrays)
    max_w = max(int(arr.shape[1]) for arr in arrays)
    channels = max((int(arr.shape[2]) if arr.ndim == 3 else 1) for arr in arrays)
    cols = int(math.ceil(math.sqrt(len(arrays))))
    rows = int(math.ceil(len(arrays) / cols))
    if channels == 1:
        out = np.zeros((rows * max_h, cols * max_w), dtype=arrays[0].dtype)
    else:
        out = np.zeros((rows * max_h, cols * max_w, channels), dtype=arrays[0].dtype)
    for idx, arr in enumerate(arrays):
        row = idx // cols
        col = idx % cols
        y0 = row * max_h
        x0 = col * max_w
        if channels == 1 and arr.ndim == 3:
            arr = arr[..., 0]
        if channels > 1 and arr.ndim == 2:
            arr = np.repeat(arr[..., None], channels, axis=2)
        out[y0 : y0 + arr.shape[0], x0 : x0 + arr.shape[1], ...] = arr
    return out


def _read_ome_tiff_windows(
    path: Path,
    *,
    level_index: int,
    windows: list[dict[str, int]],
) -> list[np.ndarray]:
    tifffile_mod = _import_tifffile_for_qc()
    arrays: list[np.ndarray] = []
    with tifffile_mod.TiffFile(str(path)) as tif:
        series = tif.series[0]
        store = series.aszarr(level=int(level_index))
        try:
            arr = zarr.open(store, mode="r")
            shape = tuple(map(int, arr.shape))
            for window in windows:
                key = _ome_tiff_window_key(
                    shape,
                    y0=int(window["y0"]),
                    y1=int(window["y1"]),
                    x0=int(window["x0"]),
                    x1=int(window["x1"]),
                )
                arrays.append(np.asarray(arr[key]))
        finally:
            store.close()
    return arrays


def _load_ome_tiff_qc_selection(
    path: Path,
    *,
    preferred_size: int = 512,
    max_pixels: int = OMETIFF_QC_MAX_PIXELS,
) -> OmeTiffQCSelection:
    tif, levels = _ome_tiff_levels(path)
    try:
        chosen_index: int | None = None
        for idx, level in enumerate(levels):
            y, x = _ome_tiff_yx_from_shape(tuple(level.shape))
            if y * x <= int(max_pixels):
                chosen_index = idx
                break
        if chosen_index is not None:
            level = levels[chosen_index]
            arr = _to_yxc_array(np.asarray(level.asarray()))
            level_yx = _ome_tiff_yx_from_shape(tuple(level.shape))
            return OmeTiffQCSelection(
                array_yxc=arr,
                method="pyramid_level_under_pixel_cap",
                level_index=int(chosen_index),
                level_shape_yx=level_yx,
                raw_shape=tuple(map(int, level.shape)),
                windows=[],
            )
        level_shape_yx = _ome_tiff_yx_from_shape(tuple(levels[0].shape))
    finally:
        tif.close()

    windows = _stratified_windows(
        level_shape_yx[0],
        level_shape_yx[1],
        window_size=max(int(preferred_size), OMETIFF_QC_WINDOW_SIZE),
        grid=OMETIFF_QC_WINDOW_GRID,
    )
    arrays = _read_ome_tiff_windows(path, level_index=0, windows=windows)
    mosaic = _mosaic_arrays(arrays)
    return OmeTiffQCSelection(
        array_yxc=mosaic,
        method="stratified_windows",
        level_index=0,
        level_shape_yx=level_shape_yx,
        raw_shape=tuple(map(int, mosaic.shape)),
        windows=windows,
    )


def _companion_mask_path_for_ome_tiff(path: Path) -> Path | None:
    name = path.name
    for suffix in ("_rgb.ome.tif", "_rgb.ome.tiff"):
        if name.lower().endswith(suffix):
            candidate = path.with_name(name[: -len(suffix)] + suffix.replace("_rgb", "_mask"))
            return candidate if candidate.exists() else None
    return None


def _load_ome_tiff_level_array(path: Path, *, preferred_size: int = 512) -> np.ndarray:
    return _load_ome_tiff_qc_selection(path, preferred_size=preferred_size).array_yxc


def _load_ometiff_thumbnail(
    path: Path,
    *,
    size: int,
    mode: str,
    qc_display_mode: str = "auto",
    qc_masked_background: str = "black",
) -> Image.Image:
    display_mode = _normalize_qc_display_mode(qc_display_mode)
    background = _normalize_qc_masked_background(qc_masked_background)
    raw = _to_display_uint8(_load_ome_tiff_qc_selection(path, preferred_size=size).array_yxc)
    mask_path = _companion_mask_path_for_ome_tiff(path)
    mask = None
    if mask_path is not None and mask_path.exists():
        mask_arr = _load_ome_tiff_qc_selection(mask_path, preferred_size=size).array_yxc
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        mask = mask_arr.astype(bool)

    if display_mode == "auto":
        display_mode = "masked_rgb"
    if display_mode == "mask":
        if mask is None:
            raise ValueError("QC mask display requested, but companion OME-TIFF mask is not present.")
        return _thumbnail_array(mask.astype(np.uint8) * 255, size=size, mode=mode)
    if display_mode == "triptych":
        if mask is None:
            raise ValueError("Triptych QC requires a companion OME-TIFF mask.")
        return _triptych_thumbnail(
            raw=raw,
            mask=mask,
            size=size,
            mode=mode,
            qc_masked_background=background,
        )
    if display_mode == "raw_rgb":
        return _thumbnail_array(raw, size=size, mode=mode)
    return _thumbnail_array(
        _mask_rgb_for_display(raw, mask, qc_masked_background=background),
        size=size,
        mode=mode,
    )


def _triptych_thumbnail(
    *,
    raw: np.ndarray,
    mask: np.ndarray | None,
    size: int,
    mode: str,
    qc_masked_background: str = "black",
) -> Image.Image:
    if mask is None:
        raise ValueError("triptych QC requires labels/tissue_mask.")
    raw_img = _thumbnail_array(raw, size=size, mode=mode)
    mask_img = _thumbnail_array((mask.astype(np.uint8) * 255), size=size, mode=mode)
    masked_img = _thumbnail_array(
        _mask_rgb_for_display(raw, mask, qc_masked_background=qc_masked_background),
        size=size,
        mode=mode,
    )
    w = max(raw_img.width, mask_img.width, masked_img.width)
    h = max(raw_img.height, mask_img.height, masked_img.height)
    panels = [ImageOps.pad(img, (w, h), color=(255, 255, 255)) for img in (raw_img, mask_img, masked_img)]
    out = Image.new(mode, (w * 3 + 2, h), (255, 255, 255))
    for idx, panel in enumerate(panels):
        out.paste(panel, (idx * (w + 1), 0))
    return out


def _choose_ngff_dataset_path(
    path: Path,
    *,
    preferred_size: int,
    prefer_full_resolution: bool = False,
    max_full_res_pixels: int | None = None,
) -> str:
    dataset_paths = _ngff_dataset_paths(path)
    axes = _ngff_axes(path)
    if prefer_full_resolution and dataset_paths:
        first = dataset_paths[0]
        arr = zarr.open_array(str(path / first), mode="r")
        y, x = _array_yx_shape(tuple(arr.shape), axes)
        if max_full_res_pixels is None or int(y) * int(x) <= int(max_full_res_pixels):
            return first

    chosen = dataset_paths[-1]
    for dataset_path in reversed(dataset_paths):
        arr = zarr.open_array(str(path / dataset_path), mode="r")
        y, x = _array_yx_shape(tuple(arr.shape), axes)
        if max(y, x) >= preferred_size:
            chosen = dataset_path
            break
    return chosen


def _load_ngff_level(
    path: Path,
    *,
    preferred_size: int = 512,
    prefer_full_resolution: bool = False,
    max_full_res_pixels: int | None = None,
) -> NGFFLevelSelection:
    axes = _ngff_axes(path)
    chosen = _choose_ngff_dataset_path(
        path,
        preferred_size=preferred_size,
        prefer_full_resolution=prefer_full_resolution,
        max_full_res_pixels=max_full_res_pixels,
    )
    arr = zarr.open_array(str(path / chosen), mode="r")
    raw = np.asarray(arr[...])
    return NGFFLevelSelection(
        dataset_path=chosen,
        axes=axes,
        raw_shape=tuple(map(int, raw.shape)),
        array_yxc=_to_yxc_array(raw, axes),
    )


def _load_ngff_level_array(path: Path, *, preferred_size: int = 512) -> np.ndarray:
    return _load_ngff_level(path, preferred_size=preferred_size).array_yxc


def _load_ngff_thumbnail(
    path: Path,
    *,
    size: int,
    mode: str,
    qc_display_mode: str = "auto",
    qc_masked_background: str = "black",
) -> Image.Image:
    selection = _load_ngff_level(
        path,
        preferred_size=size,
        prefer_full_resolution=True,
        max_full_res_pixels=NGFF_THUMBNAIL_FULL_RES_MAX_PIXELS,
    )
    display_mode = _normalize_qc_display_mode(qc_display_mode)
    primary_rgb_mode = (_ngff_primary_rgb_mode(path) or "unmasked_rgb").strip().lower()
    mask = _load_ngff_mask_for_dataset(path, selection.dataset_path)
    raw = _to_display_uint8(selection.array_yxc)

    if display_mode == "auto":
        if primary_rgb_mode == "masked_rgb":
            display_mode = "masked_rgb"
        elif mask is not None:
            display_mode = "masked_rgb"
        else:
            display_mode = "raw_rgb"

    if display_mode == "raw_rgb":
        if primary_rgb_mode == "masked_rgb":
            raise ValueError(
                "Raw RGB is not stored in this masked-only production artifact. "
                "Use a debug/unmasked profile or regenerate from the source VSI/ETS."
            )
        return _thumbnail_array(raw, size=size, mode=mode)
    if display_mode == "mask":
        if mask is None:
            raise ValueError("QC mask display requested, but labels/tissue_mask is not present.")
        return _thumbnail_array(mask.astype(np.uint8) * 255, size=size, mode=mode)
    if display_mode == "triptych":
        if primary_rgb_mode == "masked_rgb":
            raise ValueError(
                "Triptych QC requires raw RGB, but this artifact stores masked RGB as primary. "
                "Use a debug/unmasked profile or regenerate from the source VSI/ETS."
            )
        return _triptych_thumbnail(
            raw=raw,
            mask=mask,
            size=size,
            mode=mode,
            qc_masked_background=qc_masked_background,
        )

    if primary_rgb_mode == "masked_rgb":
        if _normalize_qc_masked_background(qc_masked_background) == "white" and mask is not None:
            return _thumbnail_array(
                _mask_rgb_for_display(raw, mask, qc_masked_background=qc_masked_background),
                size=size,
                mode=mode,
            )
        return _thumbnail_array(raw, size=size, mode=mode)
    return _thumbnail_array(
        _mask_rgb_for_display(raw, mask, qc_masked_background=qc_masked_background),
        size=size,
        mode=mode,
    )


def _channel_stats(arr: np.ndarray) -> dict[str, float | bool]:
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return {}
    rgb = arr[..., :3].reshape(-1, 3).astype(np.float64)
    absdiffs = {
        "rg": float(np.mean(np.abs(rgb[:, 0] - rgb[:, 1]))),
        "rb": float(np.mean(np.abs(rgb[:, 0] - rgb[:, 2]))),
        "gb": float(np.mean(np.abs(rgb[:, 1] - rgb[:, 2]))),
    }
    return {
        "image_mean_red": float(rgb[:, 0].mean()),
        "image_mean_green": float(rgb[:, 1].mean()),
        "image_mean_blue": float(rgb[:, 2].mean()),
        "image_std_red": float(rgb[:, 0].std()),
        "image_std_green": float(rgb[:, 1].std()),
        "image_std_blue": float(rgb[:, 2].std()),
        "channel_absdiff_rg": absdiffs["rg"],
        "channel_absdiff_rb": absdiffs["rb"],
        "channel_absdiff_gb": absdiffs["gb"],
        "channels_nearly_identical": bool(max(absdiffs.values()) <= 2.0),
    }


def annotate_image(
    img: Image.Image,
    text: str,
    position: str = "top-left",
    font_size: int = 12,
) -> Image.Image:
    """
    Add text annotation to image.

    Parameters
    ----------
    img : PIL.Image
        Image to annotate.
    text : str
        Annotation text.
    position : str
        Position: 'top-left', 'top-right', 'bottom-left', 'bottom-right'.
    font_size : int
        Font size.

    Returns
    -------
    PIL.Image
        Annotated image.
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Calculate position
    w, h = img.size
    pad = 2

    if "top" in position:
        y = pad
    else:
        y = h - th - pad * 2

    if "left" in position:
        x = pad
    else:
        x = w - tw - pad * 2

    # Draw background box
    draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return img


def mark_artifact_image(img: Image.Image) -> Image.Image:
    """Draw a red border around a likely artifact thumbnail."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for offset in range(3):
        draw.rectangle(
            [offset, offset, w - 1 - offset, h - 1 - offset],
            outline=(220, 20, 20),
        )
    return img


def create_grid_pil(
    images: list[Image.Image],
    columns: int,
    padding: int = 1,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Create image grid using PIL.

    Parameters
    ----------
    images : list of PIL.Image
        Images to arrange.
    columns : int
        Number of columns.
    padding : int
        Padding between images.
    background : tuple
        Background color.

    Returns
    -------
    PIL.Image
        Grid image.
    """
    if not images:
        raise ValueError("No images provided")

    # Normalize sizes to the largest thumbnail dimensions so mixed-aspect
    # thumbnails do not get cropped down to the size of the first image.
    w = max(img.size[0] for img in images)
    h = max(img.size[1] for img in images)
    images = [ImageOps.pad(img, (w, h), color=background) for img in images]

    rows = math.ceil(len(images) / columns)
    grid_w = columns * w + (columns - 1) * padding
    grid_h = rows * h + (rows - 1) * padding

    grid = Image.new("RGB", (grid_w, grid_h), background)

    for idx, img in enumerate(images):
        r = idx // columns
        c = idx % columns
        x = c * (w + padding)
        y = r * (h + padding)
        grid.paste(img, (x, y))

    return grid


def create_grid_torch(
    images: list[Image.Image],
    columns: int,
    padding: int = 1,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Create image grid using PyTorch (faster for large grids).

    Parameters
    ----------
    images : list of PIL.Image
        Images to arrange.
    columns : int
        Number of columns.
    padding : int
        Padding between images.

    Returns
    -------
    PIL.Image
        Grid image.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")

    if not images:
        raise ValueError("No images provided")

    # Normalize sizes so aspect-ratio-preserving thumbnails can still be
    # stacked into a single tensor batch, using the largest thumbnail size
    # rather than the first image's size.
    w = max(img.size[0] for img in images)
    h = max(img.size[1] for img in images)
    images = [ImageOps.pad(img, (w, h), color=background) for img in images]

    # Convert to tensors
    tensors = []
    for img in images:
        arr = np.array(img, copy=True)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        tensors.append(t)

    batch = torch.stack(tensors)
    grid = make_grid(batch, nrow=columns, padding=padding)

    # Convert back to PIL
    grid = (grid.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(grid, mode="RGB")


def _default_manifest_path(input_dir: str | Path) -> Path:
    return Path(input_dir) / "tile_manifest.json"


def _resolve_manifest_path(input_dir: str | Path, manifest_path: str | Path | None) -> Path | None:
    if manifest_path is not None:
        chosen = Path(manifest_path)
        return chosen if chosen.exists() else None

    default_manifest = _default_manifest_path(input_dir)
    if default_manifest.exists():
        return default_manifest
    derivative_manifest = Path(input_dir) / "manifest.json"
    return derivative_manifest if derivative_manifest.exists() else None


def _normalize_backend(backend: str) -> str:
    if backend == "auto":
        return "torch" if TORCH_AVAILABLE else "pil"
    if backend not in {"pil", "torch"}:
        raise ValueError("backend must be one of 'pil', 'torch', or 'auto'")
    return backend


def _image_size(path: Path) -> tuple[int, int]:
    if _is_ome_zarr_path(path):
        axes = _ngff_axes(path)
        dataset_path = _ngff_dataset_paths(path)[0]
        arr = zarr.open_array(str(path / dataset_path), mode="r")
        y, x = _array_yx_shape(tuple(arr.shape), axes)
        return x, y
    if _is_ome_tiff_path(path):
        y, x = _ome_tiff_yx_shape(path)
        return x, y
    with Image.open(path) as img:
        return img.width, img.height


def _legacy_source_image(parsed: dict[str, str | int]) -> str:
    prefix = str(parsed["prefix"])
    slide_label = str(parsed.get("slide_raw") or parsed["slide"])
    return f"{prefix}_{slide_label}"


def _derivative_source_image(payload: dict[str, Any]) -> str:
    for key in ("source_vsi", "source_ome_zarr", "source_ets"):
        value = payload.get(key)
        if value:
            return Path(str(value)).name
    return "unknown_source"


def _record_from_derivative_manifest(
    input_dir: Path,
    manifest_path: Path,
    payload: dict[str, Any],
    *,
    fallback_overall_index: int,
) -> QCRecord | None:
    tissue_dir = manifest_path.parent
    if not tissue_dir.exists():
        logger.warning(
            "Skipping derivative manifest with missing tissue directory: %s", manifest_path
        )
        return None
    try:
        relative_path = str(tissue_dir.relative_to(input_dir))
    except ValueError:
        relative_path = str(tissue_dir)
    if relative_path == "":
        relative_path = "."

    try:
        tissue_index = int(payload.get("tissue_index", fallback_overall_index - 1))
    except (TypeError, ValueError):
        tissue_index = fallback_overall_index - 1
    width, height = _image_size(tissue_dir)
    overall_index = tissue_index + 1
    return _make_qc_record(
        relative_path=relative_path,
        filename=tissue_dir.name,
        source_image=_derivative_source_image(payload),
        tile_index_on_source=tissue_index,
        overall_index=overall_index,
        overall_label=f"{overall_index:04d}",
        width=width,
        height=height,
        component_qc=None,
    )


def _load_manifest_records(input_dir: Path, manifest_path: Path) -> list[QCRecord]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("role") == "derivative":
        record = _record_from_derivative_manifest(
            input_dir,
            manifest_path,
            payload,
            fallback_overall_index=1,
        )
        return [record] if record is not None else []

    records = []
    for item in payload.get("records", []):
        record = _make_qc_record(
            relative_path=str(item["relative_path"]),
            filename=str(item["filename"]),
            source_image=str(item["source_image"]),
            tile_index_on_source=int(item["tile_index_on_source"]),
            overall_index=int(item["overall_index"]),
            overall_label=str(item["overall_label"]),
            width=int(item["width"]),
            height=int(item["height"]),
            component_qc=item.get("component_qc"),
        )
        if record.path(input_dir).exists():
            records.append(record)
        else:
            logger.warning(
                "Skipping manifest record for missing QC image: %s", record.relative_path
            )
    return records


def _load_derivative_manifest_records(input_dir: Path) -> list[QCRecord]:
    if _is_ome_zarr_path(input_dir):
        manifest_paths = [input_dir / "tissue_manifest.json"]
    else:
        manifest_paths = sorted(input_dir.glob("*.ome.zarr/tissue_manifest.json"))

    records: list[QCRecord] = []
    for ordinal, manifest_path in enumerate(manifest_paths, start=1):
        if not manifest_path.exists():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed derivative manifest %s: %s", manifest_path, exc)
            continue
        if payload.get("role") != "derivative":
            continue
        record = _record_from_derivative_manifest(
            input_dir,
            manifest_path,
            payload,
            fallback_overall_index=ordinal,
        )
        if record is not None and record.path(input_dir).exists():
            records.append(record)

    records.sort(
        key=lambda record: (
            record.source_image,
            record.tile_index_on_source,
            record.filename,
        )
    )
    return records


def _load_processing_metadata_records(input_dir: Path) -> list[QCRecord]:
    """
    Load records from per-slide metadata emitted by ``process_wsi``.

    Notebook 01 writes files like ``level_7_Image_00_00.tif`` plus
    ``level_7_Image_00_metadata.json`` before any global-index renaming.  These
    files do not match the legacy renamed-tile pattern, so QC needs this
    metadata fallback when review happens before artifact deletion/renaming.
    """
    records: list[QCRecord] = []
    metadata_paths = sorted(input_dir.glob("*_metadata.json"))
    for metadata_path in metadata_paths:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed processing metadata %s: %s", metadata_path, exc)
            continue

        tile_records = payload.get("tile_records") or []
        if not tile_records:
            output_paths = payload.get("output_paths") or []
            tile_records = [
                {
                    "path": path,
                    "source_image": Path(payload.get("input_path", metadata_path.stem)).name,
                    "tile_index_on_source": idx,
                }
                for idx, path in enumerate(output_paths)
            ]

        for item in tile_records:
            path = Path(str(item.get("path", "")))
            if not path.is_absolute():
                path = input_dir / path
            if not path.exists() or path.suffix.lower() not in {
                ".tif",
                ".tiff",
                ".png",
                ".jpg",
                ".jpeg",
            }:
                continue

            width = int(item.get("width", 0) or 0)
            height = int(item.get("height", 0) or 0)
            if width <= 0 or height <= 0:
                width, height = _image_size(path)

            records.append(
                _make_qc_record(
                    relative_path=str(path.relative_to(input_dir)),
                    filename=path.name,
                    source_image=str(
                        item.get("source_image") or payload.get("input_path") or metadata_path.stem
                    ),
                    tile_index_on_source=int(item.get("tile_index_on_source", 0)),
                    overall_index=0,
                    overall_label="",
                    width=width,
                    height=height,
                    component_qc=item.get("component_qc"),
                )
            )

    records.sort(
        key=lambda record: (
            record.source_image,
            record.tile_index_on_source,
            record.filename,
        )
    )
    for idx, record in enumerate(records, start=1):
        records[idx - 1] = _make_qc_record(
            relative_path=record.relative_path,
            filename=record.filename,
            source_image=record.source_image,
            tile_index_on_source=record.tile_index_on_source,
            overall_index=idx,
            overall_label=f"{idx:04d}",
            width=record.width,
            height=record.height,
            component_qc=_record_component_qc(record),
        )
    return records


def _load_legacy_qc_records(
    input_dir: Path,
    pattern: re.Pattern | None = None,
) -> list[QCRecord]:
    records: list[QCRecord] = []
    for path in find_images(input_dir, pattern=pattern):
        parsed = parse_filename(path.name, pattern)
        if parsed is None:
            continue
        width, height = _image_size(path)
        records.append(
            _make_qc_record(
                relative_path=str(path.relative_to(input_dir)),
                filename=path.name,
                source_image=_legacy_source_image(parsed),
                tile_index_on_source=int(parsed["slice_on_slide"]),
                overall_index=int(parsed["overall"]),
                overall_label=str(parsed.get("overall_raw") or parsed["overall"]),
                width=width,
                height=height,
            )
        )
    records.sort(
        key=lambda record: (
            record.source_image,
            record.tile_index_on_source,
            record.overall_index,
            record.filename,
        )
    )
    return records


def load_qc_records(
    input_dir: str | Path,
    manifest_path: str | Path | None = None,
    pattern: re.Pattern | None = None,
) -> list[QCRecord]:
    input_dir = Path(input_dir)
    chosen_manifest = _resolve_manifest_path(input_dir, manifest_path)
    if chosen_manifest is not None:
        return _load_manifest_records(input_dir, chosen_manifest)
    metadata_records = _load_processing_metadata_records(input_dir)
    if metadata_records:
        return metadata_records
    derivative_records = _load_derivative_manifest_records(input_dir)
    if derivative_records:
        return derivative_records
    return _load_legacy_qc_records(input_dir, pattern=pattern)


def _sorted_groups(records: list[QCRecord]) -> list[tuple[int, str, list[QCRecord]]]:
    grouped: dict[str, list[QCRecord]] = {}
    for record in records:
        grouped.setdefault(record.source_image, []).append(record)

    ordered_groups: list[tuple[int, str, list[QCRecord]]] = []
    ordered_source_images = sorted(
        grouped,
        key=lambda source_image: (
            min(record.overall_index for record in grouped[source_image]),
            source_image,
        ),
    )
    for ordinal, source_image in enumerate(ordered_source_images, start=1):
        group_records = sorted(
            grouped[source_image],
            key=lambda record: (
                record.tile_index_on_source,
                record.overall_index,
                record.filename,
            ),
        )
        ordered_groups.append((ordinal, source_image, group_records))
    return ordered_groups


def compute_qc_stats(
    records: list[QCRecord],
    input_dir: str | Path,
    *,
    qc_masked_background: str = "black",
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    rows: list[dict[str, Any]] = []
    for record in records:
        path = record.path(input_dir)
        if _is_ome_zarr_path(path):
            selection = _load_ngff_level(
                path,
                preferred_size=512,
                prefer_full_resolution=True,
                max_full_res_pixels=NGFF_STATS_FULL_RES_MAX_PIXELS,
            )
            arr = _to_display_uint8(selection.array_yxc)
            row = {
                "relative_path": record.relative_path,
                "filename": record.filename,
                "source_image": record.source_image,
                "tile_index_on_source": record.tile_index_on_source,
                "overall_index": record.overall_index,
                "overall_label": record.overall_label,
                "width": record.width,
                "height": record.height,
                "area_px": record.width * record.height,
                "image_mean_intensity": float(arr.mean()),
                "image_std_intensity": float(arr.std()),
                "ngff_dataset_path": selection.dataset_path,
                "ngff_axes": ",".join(selection.axes) if selection.axes is not None else "",
                "ngff_raw_shape": "x".join(str(dim) for dim in selection.raw_shape),
                "artifact_likely": _record_artifact_likely(record),
                "artifact_reason": _record_artifact_reason(record),
            }
            row.update(_channel_stats(arr))
            rows.append(row)
            continue

        if _is_ome_tiff_path(path):
            selection = _load_ome_tiff_qc_selection(path, preferred_size=512)
            arr = _to_display_uint8(selection.array_yxc)
            background = _normalize_qc_masked_background(qc_masked_background)
            row = {
                "relative_path": record.relative_path,
                "filename": record.filename,
                "source_image": record.source_image,
                "tile_index_on_source": record.tile_index_on_source,
                "overall_index": record.overall_index,
                "overall_label": record.overall_label,
                "width": record.width,
                "height": record.height,
                "area_px": record.width * record.height,
                "image_mean_intensity": float(arr.mean()),
                "image_std_intensity": float(arr.std()),
                "ometiff_stats_level_shape": "x".join(str(dim) for dim in arr.shape),
                "ometiff_qc_method": selection.method,
                "ometiff_qc_level_index": selection.level_index,
                "ometiff_qc_level_shape_yx": "x".join(str(dim) for dim in selection.level_shape_yx),
                "ometiff_qc_raw_shape": "x".join(str(dim) for dim in selection.raw_shape),
                "ometiff_qc_windows": json.dumps(selection.windows, sort_keys=True),
                "qc_masked_background": background,
                "artifact_likely": _record_artifact_likely(record),
                "artifact_reason": _record_artifact_reason(record),
            }
            row.update(_channel_stats(arr))
            rows.append(row)
            continue

        with Image.open(path) as img:
            arr = np.asarray(img)
            row = {
                "relative_path": record.relative_path,
                "filename": record.filename,
                "source_image": record.source_image,
                "tile_index_on_source": record.tile_index_on_source,
                "overall_index": record.overall_index,
                "overall_label": record.overall_label,
                "width": img.width,
                "height": img.height,
                "area_px": img.width * img.height,
                "image_mean_intensity": float(arr.mean()),
                "image_std_intensity": float(arr.std()),
                "artifact_likely": _record_artifact_likely(record),
                "artifact_reason": _record_artifact_reason(record),
            }
            row.update(_channel_stats(arr))
            rows.append(row)
            component_qc = _record_component_qc(record)
            if component_qc:
                rows[-1].update(
                    {
                        key: value
                        for key, value in component_qc.items()
                        if key not in {"artifact_likely", "artifact_reason"}
                    }
                )
    return pd.DataFrame(rows)

def _calculate_columns(count: int, columns: int | str) -> int:
    if columns == "auto":
        return max(1, int(math.sqrt(count) + 0.5))
    return max(1, int(columns))


def _build_label(
    record: QCRecord,
    label_mode: str,
    *,
    group_ordinal: int | None = None,
    master: bool = False,
) -> str | None:
    if label_mode == "slice":
        label = f"{record.tile_index_on_source:02d}"
        return f"{label}|ART?" if _record_artifact_likely(record) else label
    if label_mode == "overall":
        label = record.overall_label
        return f"{label}|ART?" if _record_artifact_likely(record) else label
    if label_mode == "both":
        if master:
            label = (
                f"s{group_ordinal:02d}:t{record.tile_index_on_source:02d}|g{record.overall_label}"
            )
        else:
            label = f"t{record.tile_index_on_source:02d}|g{record.overall_label}"
        return f"{label}|ART?" if _record_artifact_likely(record) else label
    if label_mode == "none":
        return None
    raise ValueError("label_mode must be one of 'slice', 'overall', 'both', or 'none'")


def _create_grid(
    thumbnails: list[Image.Image],
    columns: int,
    padding: int,
    backend: str,
) -> Image.Image:
    if backend == "torch":
        return create_grid_torch(thumbnails, columns, padding)
    return create_grid_pil(thumbnails, columns, padding)


def render_qc_grids(
    records: list[QCRecord],
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    thumb_size: int = 256,
    padding: int = 1,
    columns: int | str = "auto",
    label_mode: str = "slice",
    qc_display_mode: str = "auto",
    qc_masked_background: str = "black",
    backend: str = "pil",
    write_master: bool = True,
    write_per_slide: bool = True,
) -> QCArtifacts:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_backend = _normalize_backend(backend)

    per_slide_paths: list[Path] = []
    grouped = _sorted_groups(records)

    if write_per_slide:
        for group_ordinal, _source_image, group_records in grouped:
            thumbs = []
            for record in group_records:
                thumb = load_thumbnail(
                    record.path(input_dir),
                    thumb_size,
                    qc_display_mode=qc_display_mode,
                    qc_masked_background=qc_masked_background,
                )
                label = _build_label(record, label_mode)
                if label:
                    thumb = annotate_image(thumb, label)
                if _record_artifact_likely(record):
                    thumb = mark_artifact_image(thumb)
                thumbs.append(thumb)

            grid = _create_grid(
                thumbs,
                _calculate_columns(len(thumbs), columns),
                padding,
                normalized_backend,
            )
            out_path = output_dir / f"slide_{group_ordinal:02d}_grid.png"
            grid.save(out_path, dpi=(300, 300))
            per_slide_paths.append(out_path)

    master_path: Path | None = None
    if write_master and grouped:
        all_thumbs = []
        for group_ordinal, _source_image, group_records in grouped:
            for record in group_records:
                thumb = load_thumbnail(
                    record.path(input_dir),
                    thumb_size,
                    qc_display_mode=qc_display_mode,
                    qc_masked_background=qc_masked_background,
                )
                label = _build_label(record, label_mode, group_ordinal=group_ordinal, master=True)
                if label:
                    thumb = annotate_image(thumb, label)
                if _record_artifact_likely(record):
                    thumb = mark_artifact_image(thumb)
                all_thumbs.append(thumb)

        master = _create_grid(
            all_thumbs,
            _calculate_columns(len(all_thumbs), columns),
            padding,
            normalized_backend,
        )
        master_path = output_dir / "master_contact_sheet.png"
        master.save(master_path, dpi=(300, 300))

    return QCArtifacts(
        master_contact_sheet=master_path,
        per_slide_grids=per_slide_paths,
        stats_csv=None,
        records_manifest=None,
    )


def run_qc_workflow(
    input_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    thumb_size: int = 256,
    padding: int = 1,
    columns: int | str = "auto",
    label_mode: str = "both",
    qc_display_mode: str = "auto",
    qc_masked_background: str = "black",
    backend: str = "pil",
    write_master: bool = True,
    write_per_slide: bool = True,
    write_stats: bool = True,
) -> QCWorkflowResult:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records_manifest = _resolve_manifest_path(input_dir, manifest_path)
    records = load_qc_records(input_dir, manifest_path=manifest_path)
    if not records:
        logger.warning("No QC inputs found in %s", input_dir)
        return QCWorkflowResult(
            records=[],
            artifacts=QCArtifacts(
                master_contact_sheet=None,
                per_slide_grids=[],
                stats_csv=None,
                records_manifest=records_manifest,
            ),
        )

    rendered = render_qc_grids(
        records,
        input_dir,
        output_dir,
        thumb_size=thumb_size,
        padding=padding,
        columns=columns,
        label_mode=label_mode,
        qc_display_mode=qc_display_mode,
        qc_masked_background=qc_masked_background,
        backend=backend,
        write_master=write_master,
        write_per_slide=write_per_slide,
    )

    stats_csv: Path | None = None
    if write_stats:
        stats_df = compute_qc_stats(
            records,
            input_dir,
            qc_masked_background=qc_masked_background,
        )
        stats_csv = output_dir / "image_statistics.csv"
        stats_df.to_csv(stats_csv, index=False)

    artifacts = QCArtifacts(
        master_contact_sheet=rendered.master_contact_sheet,
        per_slide_grids=rendered.per_slide_grids,
        stats_csv=stats_csv,
        records_manifest=records_manifest,
    )
    logger.info(
        "Created %d QC records and %d per-slide grids", len(records), len(artifacts.per_slide_grids)
    )
    return QCWorkflowResult(records=records, artifacts=artifacts)


def build_qc_grids(
    input_dir: str | Path,
    output_dir: str | Path,
    thumb_size: int = 256,
    padding: int = 1,
    columns: int | str = "auto",
    label_mode: str = "slice",
    qc_display_mode: str = "auto",
    qc_masked_background: str = "black",
    backend: str = "pil",
    create_master: bool = True,
) -> list[Path]:
    """
    Build QC grids for all images in a directory.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing tissue images.
    output_dir : str or Path
        Directory for output grid images.
    thumb_size : int
        Thumbnail size.
    padding : int
        Padding between thumbnails.
    columns : int or 'auto'
        Number of columns ('auto' for square-ish layout).
    label_mode : str
        Label mode: 'slice', 'overall', 'both', or 'none'.
    backend : str
        Grid backend: 'pil', 'torch', or compatibility alias 'auto'.
    create_master : bool
        Create master contact sheet with all images.

    Returns
    -------
    list of Path
        Created grid image paths.
    """
    result = run_qc_workflow(
        input_dir=input_dir,
        output_dir=output_dir,
        thumb_size=thumb_size,
        padding=padding,
        columns=columns,
        label_mode=label_mode,
        qc_display_mode=qc_display_mode,
        qc_masked_background=qc_masked_background,
        backend=backend,
        write_master=create_master,
        write_per_slide=True,
        write_stats=False,
    )
    output_paths = list(result.artifacts.per_slide_grids)
    if result.artifacts.master_contact_sheet is not None:
        output_paths.append(result.artifacts.master_contact_sheet)
    logger.info("Created %d QC grids in %s", len(output_paths), output_dir)
    return output_paths


class QCGridBuilder:
    """
    Builder class for QC grids with configurable options.

    Convenience wrapper around :func:`build_qc_grids`. Stores configuration
    and delegates to the function on each :meth:`build` call. Prefer
    :func:`build_qc_grids` for one-off calls.

    Examples
    --------
    >>> builder = QCGridBuilder(thumb_size=256, padding=2)
    >>> grids = builder.build(input_dir="tissues", output_dir="qc")
    """

    def __init__(
        self,
        thumb_size: int = 256,
        padding: int = 1,
        columns: int | str = "auto",
        label_mode: str = "slice",
        qc_display_mode: str = "auto",
        qc_masked_background: str = "black",
        backend: str = "pil",
    ):
        self.thumb_size = thumb_size
        self.padding = padding
        self.columns = columns
        self.label_mode = label_mode
        self.qc_display_mode = qc_display_mode
        self.qc_masked_background = qc_masked_background
        self.backend = backend

    def build(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        create_master: bool = True,
    ) -> list[Path]:
        """Build QC grids for a directory."""
        return build_qc_grids(
            input_dir,
            output_dir,
            thumb_size=self.thumb_size,
            padding=self.padding,
            columns=self.columns,
            label_mode=self.label_mode,
            qc_display_mode=self.qc_display_mode,
            qc_masked_background=self.qc_masked_background,
            backend=self.backend,
            create_master=create_master,
        )
