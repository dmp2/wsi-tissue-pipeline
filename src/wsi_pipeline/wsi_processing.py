"""
WSI Processing Module

Main processing pipeline for whole-slide image tissue segmentation and extraction.
Supports VSI/ETS, TIFF, JPG, and other common image formats.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import dask.array as da
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk

from .config import PipelineConfig, SegmentationConfig, TileConfig, load_config
from .segmentation.appendage import refine_appendages
from .segmentation.component_qc import filter_mask_by_labels, score_components
from .segmentation.core import (
    create_thumbnail as _shared_create_thumbnail,
)
from .segmentation.core import (
    upsample_mask as _shared_upsample_mask,
)
from .segmentation.morphology import (
    keep_largest_components as _shared_keep_largest_components,
)
from .segmentation.morphology import (
    split_touching_components as _shared_split_touching_components,
)
from .segmentation.stain import he_stain_mask
from .tiles.generator import generate_tissue_tiles

logger = logging.getLogger(__name__)

# Supported image file extensions (lowercase)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".vsi", ".ets"}

# Backend type
Backend = Literal[
    "local-entropy",
    "local-otsu",
    "tiatoolbox-otsu",
    "tiatoolbox-morph",
    "pathml-he",
]


def _to_gray(img: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale (float32).

    Handles both channel-first (C,H,W) and channel-last (H,W,C) formats.
    """
    if img.ndim == 2:
        return img.astype(np.float32)

    # Determine channel axis
    if img.shape[0] == 3:
        r, g, b = img[0], img[1], img[2]
    elif img.shape[-1] == 3:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
    else:
        return np.squeeze(img).astype(np.float32)

    # Standard grayscale conversion (ITU-R BT.709)
    return (0.2125 * r + 0.7154 * g + 0.0721 * b).astype(np.float32)


def _create_thumbnail(
    img: np.ndarray | da.Array,
    target_long_side: int,
) -> tuple[np.ndarray, float]:
    """
    Create a thumbnail and return the scale factor.

    Returns
    -------
    thumbnail : np.ndarray
        Resized image.
    scale : float
        Scale factor (thumbnail_size / original_size).
    """
    return _shared_create_thumbnail(img, target_long_side)


def _upsample_mask(
    mask_thumbnail: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Upsample boolean mask using nearest-neighbor interpolation."""
    return _shared_upsample_mask(mask_thumbnail, target_shape)


def _entropy_mask(
    gray: np.ndarray | da.Array,
    struct_elem_px: int,
    min_area: int,
    *,
    stain_mask: np.ndarray | None = None,
    pre_open_px: int = 0,
) -> np.ndarray:
    """
    Create tissue mask using local entropy.

    Parameters
    ----------
    gray : array
        Grayscale image.
    struct_elem_px : int
        Structuring element radius.
    min_area : int
        Minimum object area in pixels.
    stain_mask : np.ndarray, optional
        Boolean stain-confidence mask to apply before morphology.
    pre_open_px : int
        Optional opening radius applied after stain gating and before closing.

    Returns
    -------
    mask : np.ndarray
        Boolean tissue mask.
    """
    fp = disk(max(1, int(struct_elem_px)))

    if isinstance(gray, da.Array):
        # Compute global stats for consistent scaling
        gmin = float(gray.min().compute())
        gmax = float(gray.max().compute())
        gscale = 255.0 / (gmax - gmin + 1e-6)

        def _entropy_chunk(chunk, gmin=gmin, gscale=gscale, struct_elem_px=struct_elem_px):
            u8 = ((chunk.astype(np.float32) - gmin) * gscale).clip(0, 255).astype(np.uint8)
            return rank_entropy(u8, disk(struct_elem_px))

        ent = gray.map_overlap(
            _entropy_chunk,
            depth=(struct_elem_px, struct_elem_px),
            boundary="reflect",
            dtype=np.uint8,
        )
        ent_np = ent.astype(np.float32).compute()
    else:
        # NumPy path
        g = gray.astype(np.float32)
        g = (g - g.min()) / (g.max() - g.min() + 1e-6)
        u8 = (g * 255).astype(np.uint8)
        ent_np = rank_entropy(u8, fp).astype(np.float32)

    # Global Otsu threshold on entropy
    thr = filters.threshold_otsu(ent_np)
    bw = ent_np > thr

    if stain_mask is not None:
        bw = bw & np.asarray(stain_mask, dtype=bool)
    if pre_open_px > 0:
        bw = morphology.opening(bw, footprint=disk(int(pre_open_px)))

    # Morphological cleanup
    bw = morphology.binary_closing(bw, footprint=fp)
    bw = ndi.binary_fill_holes(bw)
    bw = morphology.remove_small_objects(bw, min_size=min_area)
    bw = morphology.remove_small_holes(bw, area_threshold=min_area)

    return bw


def _otsu_mask(
    gray: np.ndarray,
    struct_elem_px: int,
    min_area: int,
    *,
    stain_mask: np.ndarray | None = None,
    pre_open_px: int = 0,
) -> np.ndarray:
    """Create tissue mask using global Otsu threshold."""
    # Gaussian blur
    sm = filters.gaussian(gray, sigma=1.0, preserve_range=True)

    # Otsu threshold (tissue is darker than background)
    thr = filters.threshold_otsu(sm)
    bw = sm < thr

    if stain_mask is not None:
        bw = bw & np.asarray(stain_mask, dtype=bool)
    if pre_open_px > 0:
        bw = morphology.opening(bw, footprint=disk(int(pre_open_px)))

    # Morphological cleanup
    fp = disk(struct_elem_px)
    bw = morphology.binary_closing(bw, footprint=fp)
    bw = morphology.remove_small_holes(bw, area_threshold=min_area)
    bw = morphology.remove_small_objects(bw, min_size=min_area)

    return bw


def _split_touching_components(
    mask: np.ndarray,
    r_split: int = 3,
    min_area: int = 256,
) -> np.ndarray:
    """
    Split touching tissue components using watershed.

    Parameters
    ----------
    mask : np.ndarray
        Binary tissue mask.
    r_split : int
        Erosion radius for seed generation.
    min_area : int
        Minimum component area.

    Returns
    -------
    mask : np.ndarray
        Mask with touching components separated.
    """
    return _shared_split_touching_components(mask, r_split=r_split, min_area=min_area)


def _load_image(input_path: Path) -> np.ndarray:
    """
    Load an image from file, auto-detecting VSI/ETS format.

    Parameters
    ----------
    input_path : Path
        Path to the image file (.vsi, .ets, .jpg, .tiff, etc.).

    Returns
    -------
    np.ndarray
        Image array (H, W, C) or (H, W).

    Raises
    ------
    FileNotFoundError
        If the file cannot be read or VSI has no associated ETS data.
    """
    suffix = input_path.suffix.lower()

    if suffix == ".vsi":
        from .vsi_converter import vsi_to_flat_image
        img = vsi_to_flat_image(input_path, level=0)
        if img is None:
            raise FileNotFoundError(
                f"Could not read VSI file (no ETS data found): {input_path}"
            )
        return img

    if suffix == ".ets":
        from .vsi_converter import ets_to_flat_image
        img = ets_to_flat_image(input_path, level=0)
        if img is None:
            raise FileNotFoundError(f"Could not read ETS file: {input_path}")
        return img

    # Standard image formats
    import imageio.v3 as iio
    return iio.imread(input_path)


def segment_tissue(
    img: np.ndarray | da.Array,
    backend: Backend = "local-entropy",
    target_long_side: int = 1800,
    min_area_px: int = 3000,
    struct_elem_px: int = 4,
    stain_gate: bool = False,
    stain_gate_mode: str = "fixed",
    stain_min_saturation: float = 0.08,
    stain_min_od: float = 0.35,
    stain_min_he_signal: float = 0.0,
    stain_od_bg_percentile: float = 0.80,
    stain_od_mad_multiplier: float = 4.0,
    stain_pre_open_px: int = 0,
    split_touching: bool = True,
    r_split: int = 3,
    keep_top_k: int | None = None,
    appendage_refinement_enabled: bool = False,
    appendage_refinement_mode: str = "trim",
    appendage_refinement_profile: str = "he_sections",
    diagnostics: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Segment tissue regions from a whole-slide image.

    Parameters
    ----------
    img : array-like
        Input image (H, W, C) or (C, H, W).
    backend : str
        Segmentation algorithm.
    target_long_side : int
        Target size for thumbnail processing.
    min_area_px : int
        Minimum tissue area in pixels (at thumbnail scale).
    struct_elem_px : int
        Structuring element radius.
    stain_gate : bool
        Apply an H&E stain-confidence gate before morphology.
    stain_gate_mode : str
        H&E gate mode: "fixed", "adaptive-od", or "adaptive-he".
    stain_min_saturation : float
        Minimum HSV saturation for fixed stain-confidence gating.
    stain_min_od : float
        Minimum summed optical density, or adaptive mode floor.
    stain_min_he_signal : float
        Optional minimum HED hematoxylin+eosin signal for stain rescue.
    stain_od_bg_percentile : float
        Brightness percentile for adaptive background optical-density estimation.
    stain_od_mad_multiplier : float
        Robust MAD multiplier for adaptive optical-density gating.
    stain_pre_open_px : int
        Optional opening radius applied after stain gating and before closing.
    split_touching : bool
        Whether to split touching tissue sections.
    r_split : int
        Radius for splitting.
    keep_top_k : int, optional
        Keep only the K largest connected components after splitting.
    appendage_refinement_enabled : bool
        Trim or annotate weakly stained peripheral appendages before tile extraction.
    appendage_refinement_mode : str
        Appendage refinement mode: "trim" or "annotate".
    appendage_refinement_profile : str
        Appendage refinement profile. Currently only "he_sections" is supported.
    diagnostics : bool
        Enable diagnostic output.

    Returns
    -------
    mask : np.ndarray
        Binary mask at full resolution (H, W).
    info : dict
        Diagnostic information.
    """
    # Ensure channel-last format
    if isinstance(img, da.Array):
        if img.shape[0] == 3 and img.shape[-1] != 3:
            img = da.moveaxis(img, 0, -1)
        H, W = int(img.shape[0]), int(img.shape[1])
    else:
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
            img = np.moveaxis(img, 0, -1)
        H, W = img.shape[:2]

    # Create thumbnail for processing
    thumb, scale = _create_thumbnail(img, target_long_side)
    Ht, Wt = thumb.shape[:2]

    # Segmentation parameters are defined at thumbnail scale.  Keeping them
    # stable across source sizes avoids admitting tiny background components
    # when a large slide is downsampled for mask generation.
    min_area_scaled = max(64, int(min_area_px))
    struct_elem_scaled = max(2, int(struct_elem_px))

    # Convert to grayscale
    gray = _to_gray(thumb)
    stain_mask_t = None
    stain_info: dict[str, Any] = {}
    if stain_gate:
        stain_mask_t, stain_info = he_stain_mask(
            thumb,
            mode=stain_gate_mode,
            min_saturation=stain_min_saturation,
            min_od=stain_min_od,
            min_he_signal=stain_min_he_signal,
            od_bg_percentile=stain_od_bg_percentile,
            od_mad_multiplier=stain_od_mad_multiplier,
            return_info=True,
        )

    # Segment based on backend
    if backend == "local-entropy":
        mask_t = _entropy_mask(
            gray,
            struct_elem_scaled,
            min_area_scaled,
            stain_mask=stain_mask_t,
            pre_open_px=stain_pre_open_px,
        )
    elif backend == "local-otsu":
        mask_t = _otsu_mask(
            gray,
            struct_elem_scaled,
            min_area_scaled,
            stain_mask=stain_mask_t,
            pre_open_px=stain_pre_open_px,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Split touching components
    if split_touching:
        mask_t = _split_touching_components(mask_t, r_split, min_area_scaled)

    appendage_info: dict[str, Any] = {
        "enabled": appendage_refinement_enabled,
        "mode": appendage_refinement_mode,
        "profile": appendage_refinement_profile,
        "n_appendages_flagged": 0,
        "flagged_area_px": 0,
        "n_appendages_trimmed": 0,
        "trimmed_area_px": 0,
        "trimmed_fraction": 0.0,
        "appendage_reason": "",
        "records": [],
    }
    if appendage_refinement_enabled:
        mask_t, appendage_info = refine_appendages(
            thumb,
            mask_t,
            mode=appendage_refinement_mode,
            profile=appendage_refinement_profile,
            min_area_px=min_area_scaled,
            he_mask=stain_mask_t if stain_gate and stain_gate_mode == "adaptive-he" else None,
        )

    mask_t = _shared_keep_largest_components(mask_t, keep_top_k)

    # Count components
    labeled, n_components = measure.label(mask_t, return_num=True, connectivity=2)

    info = {
        "backend": backend,
        "thumbnail_shape": (Ht, Wt),
        "original_shape": (H, W),
        "scale": scale,
        "struct_elem_px": struct_elem_scaled,
        "min_area": min_area_scaled,
        "stain_gate": stain_gate,
        "stain_gate_mode": stain_gate_mode,
        "stain_min_saturation": stain_min_saturation,
        "stain_min_od": stain_min_od,
        "stain_min_he_signal": stain_min_he_signal,
        "stain_od_bg_percentile": stain_od_bg_percentile,
        "stain_od_mad_multiplier": stain_od_mad_multiplier,
        "stain_od_threshold": stain_info.get("od_threshold"),
        "stain_he_threshold": stain_info.get("he_threshold"),
        "stain_pre_open_px": stain_pre_open_px,
        "appendage_refinement": appendage_info,
        "n_components": n_components,
        "keep_top_k": keep_top_k,
    }

    if diagnostics:
        logger.debug(
            "[segmenter] backend=%s size_t=(%d, %d) struct_elem_px=%s min_area=%s "
            "stain_gate=%s r_split=%s CCs: %s",
            backend, Ht, Wt, struct_elem_scaled, min_area_scaled, stain_gate,
            r_split, n_components
        )

    # Upsample mask to full resolution
    mask = _upsample_mask(mask_t, (H, W))

    return mask, info


def extract_tissue_tiles(
    img: np.ndarray | da.Array,
    mask: np.ndarray,
    chunk_size: int = 512,
    pad_multiple: int = 512,
    extra_margin_px: int = 0,
) -> list[da.Array]:
    """
    Extract individual tissue tiles from image using a segmentation mask.

    This is the notebook/CLI-facing wrapper around
    :func:`wsi_pipeline.tiles.generator.generate_tissue_tiles`.  The mask may be
    a full-resolution upsampled mask or a lower-resolution thumbnail mask; the
    output tiles are centered, square, and share one common side length for the
    source image.

    Parameters
    ----------
    img : array-like
        Input image (H, W, C).
    mask : np.ndarray
        Binary tissue mask.
    chunk_size : int
        Chunk size for Dask arrays.
    pad_multiple : int
        Pad tiles to multiple of this value.
    extra_margin_px : int
        Extra margin around each tissue region.

    Returns
    -------
    tiles : list of dask.array.Array
        List of centered square tissue tiles in ``(Y, X, C)`` order.
    """
    if isinstance(img, da.Array):
        img_da = img
    else:
        img_da = da.from_array(img, chunks=(chunk_size, chunk_size, -1))

    if img_da.ndim != 3:
        raise ValueError("Expected image with shape (Y, X, C) or (C, Y, X).")

    if img_da.shape[0] in (1, 3) and img_da.shape[-1] not in (1, 3):
        s0_cyx = img_da
    else:
        s0_cyx = da.moveaxis(img_da, -1, 0)

    tiles, _tile_dim = generate_tissue_tiles(
        s0_cyx=s0_cyx,
        low_res_filled=np.asarray(mask, dtype=bool),
        chunk=chunk_size,
        pad_multiple=pad_multiple,
        extra_margin_px=extra_margin_px,
    )
    return tiles


def process_wsi(
    input_path: str | Path,
    output_dir: str | Path,
    config: PipelineConfig | None = None,
    segmentation_config: SegmentationConfig | None = None,
    tile_config: TileConfig | None = None,
) -> dict[str, Any]:
    """
    Process a single whole-slide image.

    Automatically detects VSI/ETS files and converts them. Also supports
    standard formats (JPG, TIFF, PNG, etc.) via imageio.

    Parameters
    ----------
    input_path : str or Path
        Path to input image (.vsi, .ets, .jpg, .tiff, etc.).
    output_dir : str or Path
        Directory for output files.
    config : PipelineConfig, optional
        Full pipeline configuration.
    segmentation_config : SegmentationConfig, optional
        Segmentation-specific configuration.
    tile_config : TileConfig, optional
        Tile extraction configuration.

    Returns
    -------
    result : dict
        Processing results including output paths and metadata.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    if config is None:
        config = PipelineConfig()
    if segmentation_config is None:
        segmentation_config = config.segmentation
    if tile_config is None:
        tile_config = config.tiles

    # Load image (auto-detects VSI/ETS)
    img = _load_image(input_path)

    # Ensure channel-last
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.moveaxis(img, 0, -1)

    # Convert to Dask array for large images
    img_da = da.from_array(img, chunks=(tile_config.chunk_size, tile_config.chunk_size, -1))

    # Segment tissue
    mask, seg_info = segment_tissue(
        img_da,
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

    component_qc_records: list[dict[str, Any]] = []
    component_qc_by_tile: dict[int, dict[str, Any]] = {}
    if segmentation_config.component_qc_enabled:
        component_qc_records = [
            record.to_dict()
            for record in score_components(
                img,
                mask,
                profile=segmentation_config.component_qc_profile,
            )
        ]

        if segmentation_config.component_qc_mode == "drop_artifacts":
            labels_to_keep = {
                int(record["component_label"])
                for record in component_qc_records
                if not bool(record["artifact_likely"])
            }
            mask = filter_mask_by_labels(mask, labels_to_keep)
            component_qc_records = [
                record for record in component_qc_records
                if int(record["component_label"]) in labels_to_keep
            ]
            for new_idx, record in enumerate(component_qc_records):
                record["tile_index_on_source"] = new_idx

        component_qc_by_tile = {
            int(record["tile_index_on_source"]): record
            for record in component_qc_records
        }

    # Extract tiles
    tiles = extract_tissue_tiles(
        img_da,
        mask,
        chunk_size=tile_config.chunk_size,
        pad_multiple=tile_config.pad_multiple,
        extra_margin_px=tile_config.extra_margin_px,
    )

    # Save tiles
    import imageio.v3 as iio

    output_paths = []
    tile_records = []
    for i, tile in enumerate(tiles):
        if isinstance(tile, da.Array):
            tile_np = tile.compute()
        else:
            tile_np = tile

        # Convert dtype if needed
        if config.output.convert_to_uint8 and tile_np.dtype != np.uint8:
            if np.issubdtype(tile_np.dtype, np.integer):
                maxv = np.iinfo(tile_np.dtype).max
                tile_np = (tile_np.astype(np.float32) / maxv * 255).clip(0, 255).astype(np.uint8)
            else:
                tile_np = (tile_np * 255).clip(0, 255).astype(np.uint8)

        # Save
        output_name = f"{input_path.stem}_{i:02d}.tif"
        output_path = output_dir / output_name
        iio.imwrite(output_path, tile_np)
        output_paths.append(output_path)
        tile_records.append(
            {
                "source_image": input_path.name,
                "tile_index_on_source": i,
                "path": str(output_path),
                "width": int(tile_np.shape[1]),
                "height": int(tile_np.shape[0]),
                "component_qc": component_qc_by_tile.get(i),
            }
        )

    # Save metadata
    metadata = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "n_tiles": len(tiles),
        "segmentation": seg_info,
        "appendage_refinement": seg_info.get("appendage_refinement"),
        "component_qc": {
            "enabled": segmentation_config.component_qc_enabled,
            "mode": segmentation_config.component_qc_mode,
            "profile": segmentation_config.component_qc_profile,
            "records": component_qc_records,
        },
        "output_paths": [str(p) for p in output_paths],
        "tile_records": tile_records,
    }

    metadata_path = output_dir / f"{input_path.stem}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("[%s] tiles=%d", input_path.name, len(tiles))

    return metadata


def _is_image_file(path: Path) -> bool:
    """Check if a file has a recognized image extension."""
    return path.suffix.lower() in _IMAGE_EXTENSIONS


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.vsi",
    config: PipelineConfig | None = None,
    **kwargs,
) -> dict[Path, list[Path]]:
    """
    Process all images in a directory.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing input images.
    output_dir : str or Path
        Directory for output files.
    pattern : str
        Glob pattern for input files (e.g. '*.vsi', '*.jpg', '*').
    config : PipelineConfig, optional
        Pipeline configuration.
    **kwargs
        Additional arguments passed to process_wsi.

    Returns
    -------
    results : dict
        Mapping from input paths to output paths.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = PipelineConfig()

    # If pattern is '*', filter to image files only
    candidates = sorted(input_dir.glob(pattern))
    if pattern == "*":
        candidates = [p for p in candidates if _is_image_file(p)]

    results = {}
    for input_path in candidates:
        try:
            metadata = process_wsi(
                input_path,
                output_dir,
                config=config,
                **kwargs,
            )
            results[input_path] = [Path(p) for p in metadata["output_paths"]]
        except Exception as e:
            logger.error("Error processing %s: %s", input_path, e)
            results[input_path] = []

    return results


def process_specimen(
    input_dir: str | Path,
    output_dir: str | Path,
    config: str | Path | PipelineConfig | None = None,
    pattern: str = "*.vsi",
    specimen_name: str | None = None,
) -> dict[str, Any]:
    """
    Process a complete specimen (directory of WSI files).

    This is the main entry point for processing with MLflow tracking.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing specimen WSI files.
    output_dir : str or Path
        Output directory for processed files.
    config : str, Path, or PipelineConfig, optional
        Configuration file path or config object.
    pattern : str
        Glob pattern for input files (e.g. '*.vsi', '*.jpg', '*').
    specimen_name : str, optional
        Name for the specimen (defaults to directory name).

    Returns
    -------
    results : dict
        Processing results and metadata.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Load config
    if config is None:
        cfg = PipelineConfig()
    elif isinstance(config, (str, Path)):
        cfg = load_config(config)
    else:
        cfg = config

    # Specimen name
    if specimen_name is None:
        specimen_name = input_dir.name

    # Process
    results = process_directory(
        input_dir,
        output_dir,
        pattern=pattern,
        config=cfg,
    )

    return {
        "specimen_name": specimen_name,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_inputs": len(results),
        "n_outputs": sum(len(v) for v in results.values()),
        "file_results": {str(k): [str(p) for p in v] for k, v in results.items()},
    }


class WSIProcessor:
    """
    High-level WSI processing class with configuration management.

    Convenience wrapper around the standalone functions :func:`process_wsi`,
    :func:`process_directory`, and :func:`process_specimen`. Prefer the
    standalone functions for scripting; use this class when you want to
    configure once and call multiple times.

    Examples
    --------
    >>> processor = WSIProcessor("configs/default.yaml")
    >>> results = processor.process_directory("/data/specimen", "/output/specimen")
    """

    def __init__(
        self,
        config: str | Path | PipelineConfig | None = None,
    ):
        if config is None:
            self.config = PipelineConfig()
        elif isinstance(config, (str, Path)):
            self.config = load_config(config)
        else:
            self.config = config

    def process_wsi(
        self,
        input_path: str | Path,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        """Process a single WSI file."""
        return process_wsi(input_path, output_dir, config=self.config)

    def process_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        pattern: str = "*.vsi",
    ) -> dict[Path, list[Path]]:
        """Process all images in a directory."""
        return process_directory(
            input_dir, output_dir, pattern=pattern, config=self.config
        )

    def process_specimen(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        pattern: str = "*.vsi",
        specimen_name: str | None = None,
    ) -> dict[str, Any]:
        """Process a complete specimen."""
        return process_specimen(
            input_dir,
            output_dir,
            config=self.config,
            pattern=pattern,
            specimen_name=specimen_name,
        )
