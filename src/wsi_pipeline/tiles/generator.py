"""
Tile generation utilities for WSI processing.

Provides functions for extracting tissue tiles from whole-slide images,
including ROI upsampling and masking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import dask.array as da
import numpy as np
from dask_image.ndinterp import affine_transform as dask_affine_transform
from scipy.ndimage import affine_transform as scipy_affine_transform
from scipy.ndimage import binary_fill_holes
from skimage import measure

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundsYX:
    """Half-open bounds in array order: y0, x0, y1, x1."""

    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def h(self) -> int:
        return int(self.y1 - self.y0)

    @property
    def w(self) -> int:
        return int(self.x1 - self.x0)

    @property
    def area(self) -> int:
        return max(0, self.h) * max(0, self.w)

    def as_yx(self) -> tuple[int, int, int, int]:
        return (int(self.y0), int(self.x0), int(self.y1), int(self.x1))

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (int(self.x0), int(self.y0), int(self.x1), int(self.y1))

    def as_dict(self) -> dict[str, int]:
        return {
            "y0": int(self.y0),
            "x0": int(self.x0),
            "y1": int(self.y1),
            "x1": int(self.x1),
            "h": int(self.h),
            "w": int(self.w),
        }

    def clip(self, shape_yx: tuple[int, int]) -> BoundsYX:
        h, w = map(int, shape_yx)
        return BoundsYX(
            y0=max(0, min(h, int(self.y0))),
            x0=max(0, min(w, int(self.x0))),
            y1=max(0, min(h, int(self.y1))),
            x1=max(0, min(w, int(self.x1))),
        )

    def halo(self, pixels: int, shape_yx: tuple[int, int]) -> BoundsYX:
        pixels = int(pixels)
        return BoundsYX(
            self.y0 - pixels,
            self.x0 - pixels,
            self.y1 + pixels,
            self.x1 + pixels,
        ).clip(shape_yx)


@dataclass(frozen=True)
class PaddingYX:
    """Padding needed to place clipped source data in a logical source canvas."""

    top: int
    bottom: int
    left: int
    right: int

    def as_dict(self) -> dict[str, int]:
        return {
            "top": int(self.top),
            "bottom": int(self.bottom),
            "left": int(self.left),
            "right": int(self.right),
        }


@dataclass(frozen=True)
class TissueFrameSpec:
    """Coordinate bookkeeping for one tissue tile."""

    tissue_index: int
    label_id: int
    tile_frame_level: str
    component_bbox_seg_yx: BoundsYX
    logical_frame_seg_yx: BoundsYX
    clipped_frame_seg_yx: BoundsYX
    mapped_source_frame_yx: BoundsYX
    logical_canvas_source_yx: BoundsYX
    clipped_source_yx: BoundsYX
    label_crop_seg_yx: BoundsYX
    padding_source_level: PaddingYX
    segmentation_tile_dim: int
    source_tile_dim: int
    scale_y: float
    scale_x: float

    @property
    def mapped_source_h(self) -> int:
        return int(self.mapped_source_frame_yx.h)

    @property
    def mapped_source_w(self) -> int:
        return int(self.mapped_source_frame_yx.w)

    @property
    def source_canvas_dim(self) -> int:
        return int(self.source_tile_dim)

    def debug_dict(self) -> dict[str, Any]:
        return {
            "tissue_index": int(self.tissue_index),
            "label_id": int(self.label_id),
            "tile_frame_level": self.tile_frame_level,
            "component_bbox_seg_yx": self.component_bbox_seg_yx.as_dict(),
            "logical_frame_seg_yx": self.logical_frame_seg_yx.as_dict(),
            "clipped_frame_seg_yx": self.clipped_frame_seg_yx.as_dict(),
            "mapped_source_frame_yx": self.mapped_source_frame_yx.as_dict(),
            "logical_canvas_source_yx": self.logical_canvas_source_yx.as_dict(),
            "clipped_source_yx": self.clipped_source_yx.as_dict(),
            "label_crop_seg_yx": self.label_crop_seg_yx.as_dict(),
            "padding_source_level": self.padding_source_level.as_dict(),
            "segmentation_tile_dim": int(self.segmentation_tile_dim),
            "source_tile_dim": int(self.source_tile_dim),
            "mapped_source_h": int(self.mapped_source_h),
            "mapped_source_w": int(self.mapped_source_w),
            "source_canvas_dim": int(self.source_canvas_dim),
            "scale_y": float(self.scale_y),
            "scale_x": float(self.scale_x),
        }


@dataclass(frozen=True)
class TissueTileRecord:
    """Lazy tissue tile plus its parent-coordinate crop metadata."""

    tile: da.Array
    tissue_index: int
    label_id: int
    crop_bounds_source_level: tuple[int, int, int, int]
    crop_bounds_segmentation_level: tuple[int, int, int, int]
    tile_dim: int
    tile_frame_level: str = "source"
    source_tile_dim: int | None = None
    segmentation_tile_dim: int | None = None
    scale_y: float | None = None
    scale_x: float | None = None
    frame_debug: dict[str, Any] | None = None


TileFrameLevel = Literal["source", "segmentation"]


def _normalize_tile_frame_level(tile_frame_level: str) -> TileFrameLevel:
    normalized = str(tile_frame_level).strip().lower().replace("_", "-")
    if normalized not in {"source", "segmentation"}:
        raise ValueError("tile_frame_level must be one of ['segmentation', 'source'].")
    return normalized  # type: ignore[return-value]


def _round_up(v: int, m: int) -> int:
    return ((int(v) + int(m) - 1) // int(m)) * int(m)


def _centered_square_bounds(center_y: float, center_x: float, side: int) -> BoundsYX:
    y0 = int(np.floor(center_y - side / 2.0))
    x0 = int(np.floor(center_x - side / 2.0))
    return BoundsYX(y0=y0, x0=x0, y1=y0 + int(side), x1=x0 + int(side))


def _expand_bounds_to_square(bounds: BoundsYX, side: int) -> BoundsYX:
    side = int(side)
    extra_y = max(0, side - bounds.h)
    extra_x = max(0, side - bounds.w)
    top = extra_y // 2
    left = extra_x // 2
    return BoundsYX(
        y0=bounds.y0 - top,
        x0=bounds.x0 - left,
        y1=bounds.y0 - top + side,
        x1=bounds.x0 - left + side,
    )


def _map_seg_bounds_to_source(bounds: BoundsYX, scale_y: float, scale_x: float) -> BoundsYX:
    return BoundsYX(
        y0=int(np.floor(bounds.y0 * scale_y)),
        x0=int(np.floor(bounds.x0 * scale_x)),
        y1=int(np.ceil(bounds.y1 * scale_y)),
        x1=int(np.ceil(bounds.x1 * scale_x)),
    )


def _map_source_bounds_to_seg(bounds: BoundsYX, scale_y: float, scale_x: float) -> BoundsYX:
    return BoundsYX(
        y0=int(np.floor(bounds.y0 / scale_y)),
        x0=int(np.floor(bounds.x0 / scale_x)),
        y1=int(np.ceil(bounds.y1 / scale_y)),
        x1=int(np.ceil(bounds.x1 / scale_x)),
    )


def _padding_for_canvas(canvas: BoundsYX, clipped: BoundsYX) -> PaddingYX:
    return PaddingYX(
        top=int(clipped.y0 - canvas.y0),
        bottom=int(canvas.y1 - clipped.y1),
        left=int(clipped.x0 - canvas.x0),
        right=int(canvas.x1 - clipped.x1),
    )


def _build_tissue_frame_specs(
    lr_labels: np.ndarray,
    *,
    source_shape_yx: tuple[int, int],
    tile_frame_level: str,
    pad_multiple: int,
    extra_margin_px: int,
    label_halo_px: int = 1,
) -> tuple[list[TissueFrameSpec], int]:
    """Build explicit frame/canvas specs for each labeled tissue."""
    tile_frame_level = _normalize_tile_frame_level(tile_frame_level)
    source_h, source_w = map(int, source_shape_yx)
    seg_h, seg_w = map(int, lr_labels.shape)
    scale_y = source_h / seg_h
    scale_x = source_w / seg_w

    component_specs: list[tuple[int, BoundsYX, BoundsYX, BoundsYX, BoundsYX]] = []
    source_max_side = 0
    segmentation_max_side = 0
    for lid in sort_labels_left_to_right(lr_labels):
        lr_mask = lr_labels == lid
        rows, cols = np.any(lr_mask, axis=1), np.any(lr_mask, axis=0)
        yi = np.where(rows)[0]
        xi = np.where(cols)[0]
        if yi.size == 0 or xi.size == 0:
            continue

        component_seg = BoundsYX(
            y0=int(yi[0]),
            x0=int(xi[0]),
            y1=int(yi[-1] + 1),
            x1=int(xi[-1] + 1),
        )
        mapped_component = _map_seg_bounds_to_source(component_seg, scale_y, scale_x).clip(
            (source_h, source_w)
        )
        segmentation_max_side = max(
            segmentation_max_side,
            component_seg.h + (2 * extra_margin_px),
            component_seg.w + (2 * extra_margin_px),
        )
        source_max_side = max(
            source_max_side,
            mapped_component.h + (2 * extra_margin_px),
            mapped_component.w + (2 * extra_margin_px),
        )
        component_specs.append((int(lid), component_seg, mapped_component, component_seg, mapped_component))

    if not component_specs:
        return [], 0

    raw_specs: list[dict[str, Any]] = []
    if tile_frame_level == "source":
        source_tile_dim = _round_up(source_max_side, pad_multiple)
        segmentation_tile_dim = int(np.ceil(source_tile_dim / max(scale_y, scale_x)))
        source_canvas_dim = int(source_tile_dim)
        for lid, component_seg, mapped_component, _unused_seg, _unused_source in component_specs:
            source_center_y = (mapped_component.y0 + mapped_component.y1) / 2.0
            source_center_x = (mapped_component.x0 + mapped_component.x1) / 2.0
            canvas = _centered_square_bounds(source_center_y, source_center_x, source_canvas_dim)
            logical_frame_seg = _map_source_bounds_to_seg(canvas, scale_y, scale_x)
            mapped_source_frame = canvas
            raw_specs.append(
                {
                    "lid": lid,
                    "component_seg": component_seg,
                    "logical_frame_seg": logical_frame_seg,
                    "mapped_source_frame": mapped_source_frame,
                }
            )
    else:
        segmentation_tile_dim = _round_up(segmentation_max_side, pad_multiple)
        mapped_source_frames: list[BoundsYX] = []
        for lid, component_seg, _mapped_component, _unused_seg, _unused_source in component_specs:
            seg_center_y = (component_seg.y0 + component_seg.y1) / 2.0
            seg_center_x = (component_seg.x0 + component_seg.x1) / 2.0
            logical_frame_seg = _centered_square_bounds(
                seg_center_y,
                seg_center_x,
                segmentation_tile_dim,
            )
            mapped_source_frame = _map_seg_bounds_to_source(logical_frame_seg, scale_y, scale_x)
            mapped_source_frames.append(mapped_source_frame)
            raw_specs.append(
                {
                    "lid": lid,
                    "component_seg": component_seg,
                    "logical_frame_seg": logical_frame_seg,
                    "mapped_source_frame": mapped_source_frame,
                }
            )
        source_canvas_dim = max(
            max(frame.h, frame.w)
            for frame in mapped_source_frames
        )

    specs: list[TissueFrameSpec] = []
    for tissue_index, raw in enumerate(raw_specs):
        component_seg = raw["component_seg"]
        logical_frame_seg = raw["logical_frame_seg"]
        mapped_source_frame = raw["mapped_source_frame"]
        logical_canvas_source = _expand_bounds_to_square(mapped_source_frame, source_canvas_dim)
        clipped_source = logical_canvas_source.clip((source_h, source_w))
        clipped_frame_seg = logical_frame_seg.clip((seg_h, seg_w))
        label_crop_seg = clipped_frame_seg.halo(label_halo_px, (seg_h, seg_w))
        specs.append(
            TissueFrameSpec(
                tissue_index=int(tissue_index),
                label_id=int(raw["lid"]),
                tile_frame_level=tile_frame_level,
                component_bbox_seg_yx=component_seg,
                logical_frame_seg_yx=logical_frame_seg,
                clipped_frame_seg_yx=clipped_frame_seg,
                mapped_source_frame_yx=mapped_source_frame,
                logical_canvas_source_yx=logical_canvas_source,
                clipped_source_yx=clipped_source,
                label_crop_seg_yx=label_crop_seg,
                padding_source_level=_padding_for_canvas(logical_canvas_source, clipped_source),
                segmentation_tile_dim=int(segmentation_tile_dim),
                source_tile_dim=int(source_canvas_dim),
                scale_y=float(scale_y),
                scale_x=float(scale_x),
            )
        )
    return specs, int(source_canvas_dim)


def project_label_mask_to_source_region(
    lr_labels: np.ndarray,
    *,
    label_id: int,
    source_region_yx: BoundsYX,
    label_crop_seg_yx: BoundsYX,
    scale_y: float,
    scale_x: float,
) -> np.ndarray:
    """Project a segmentation label mask into a source-level region."""
    label_crop = lr_labels[
        label_crop_seg_yx.y0 : label_crop_seg_yx.y1,
        label_crop_seg_yx.x0 : label_crop_seg_yx.x1,
    ].astype(np.int32)
    if label_crop.size == 0 or source_region_yx.h <= 0 or source_region_yx.w <= 0:
        return np.zeros((max(0, source_region_yx.h), max(0, source_region_yx.w)), dtype=bool)

    matrix = np.array([[1.0 / scale_y, 0.0], [0.0, 1.0 / scale_x]], dtype=float)
    offset = np.array(
        [
            ((source_region_yx.y0 + 0.5) / scale_y) - 0.5 - label_crop_seg_yx.y0,
            ((source_region_yx.x0 + 0.5) / scale_x) - 0.5 - label_crop_seg_yx.x0,
        ],
        dtype=float,
    )
    projected = scipy_affine_transform(
        label_crop,
        matrix=matrix,
        offset=offset,
        output_shape=(source_region_yx.h, source_region_yx.w),
        order=0,
        mode="constant",
        cval=0,
    )
    return projected == int(label_id)


def center_crop_pad_dask(yxc: da.Array, target_side: int) -> da.Array:
    """
    Center-crop then pad to square target_side (works with Dask).

    Parameters
    ----------
    yxc : da.Array
        Input array (Y, X, C).
    target_side : int
        Target square dimension.

    Returns
    -------
    da.Array
        Cropped/padded array (target_side, target_side, C).
    """
    H, W = int(yxc.shape[0]), int(yxc.shape[1])

    # Center-crop if larger than target
    ys, ye = (max(0, (H - target_side) // 2), min(H, (H + target_side) // 2))
    xs, xe = (max(0, (W - target_side) // 2), min(W, (W + target_side) // 2))
    cropped = yxc[ys:ye, xs:xe, :]

    # Pad if smaller than target
    ph = target_side - int(cropped.shape[0])
    pw = target_side - int(cropped.shape[1])
    top = ph // 2 if ph > 0 else 0
    bottom = ph - top if ph > 0 else 0
    left = pw // 2 if pw > 0 else 0
    right = pw - left if pw > 0 else 0

    if ph > 0 or pw > 0:
        cropped = da.pad(cropped, ((top, bottom), (left, right), (0, 0)), mode="constant")

    # Final safety crop (no-op unless an odd/even rounding mismatch)
    return cropped[:target_side, :target_side, :]


def crop_and_pad(
    image: np.ndarray,
    mask: np.ndarray,
    target_shape: np.ndarray | int,
) -> np.ndarray:
    """
    Crop the image to the bounding box of the mask and then pad it to target_shape.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C).
    mask : np.ndarray
        Binary mask (H, W).
    target_shape : int or array-like
        Target shape. If int, creates square output.

    Returns
    -------
    np.ndarray
        Cropped and padded image.
    """
    # Find the bounding box of the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Crop the image to the bounding box
    cropped_image = image[ymin : ymax + 1, xmin : xmax + 1]

    if isinstance(target_shape, int):
        # Calculate padding amounts for square image
        height, width = cropped_image.shape[:2]
        y_pad = max(0, (target_shape - height) // 2)
        x_pad = max(0, (target_shape - width) // 2)
        cropped_image = np.pad(
            cropped_image,
            (
                (y_pad, target_shape - height - y_pad),
                (x_pad, target_shape - width - x_pad),
                (0, 0),
            ),
            mode="constant",
        )
        padded_image = cropped_image[:target_shape, :target_shape, :]
    else:
        # Calculate padding amounts for specified (rectangular) image
        y_pad = (target_shape[0] - cropped_image.shape[0]) // 2
        x_pad = (target_shape[1] - cropped_image.shape[1]) // 2

        # Pad the cropped image to the target shape
        padded_image = np.pad(
            cropped_image,
            (
                (y_pad, target_shape[0] - cropped_image.shape[0] - y_pad),
                (x_pad, target_shape[1] - cropped_image.shape[1] - x_pad),
                (0, 0),
            ),
            mode="constant",
            constant_values=0,
        )

    return padded_image


def generate_tissue_images(
    filled_img: np.ndarray,
    original_img: np.ndarray | da.Array,
    display_images: bool = False,
) -> da.Array:
    """
    Generate images for each tissue region, cropped and padded to original image size.

    Parameters
    ----------
    filled_img : np.ndarray
        Filled binary mask with tissue regions.
    original_img : np.ndarray or da.Array
        Original image (C, Y, X) or (Y, X, C).
    display_images : bool
        Whether to display images using matplotlib.

    Returns
    -------
    da.Array
        Stacked tissue images (num_slices, height, width, channels).
    """
    if original_img.shape[0] == 3:  # Check if channels are first
        original_img = da.moveaxis(original_img, 0, -1)

    # Label connected components in the segmentation mask
    labeled_mask, num_labels = measure.label(filled_img, return_num=True, connectivity=2)
    logger.info("Number of tissue regions detected: %d", num_labels)
    target_shape = original_img.shape[:2]

    # List to store the processed tissue images as Dask arrays
    tissue_images = []

    for i in range(1, num_labels + 1):
        # Create and fill each tissue region
        tissue_mask = labeled_mask == i
        filled_tissue_mask = binary_fill_holes(tissue_mask)

        # Apply the filled mask to the original image and crop/pad it
        tissue_image = np.where(filled_tissue_mask[..., None], original_img, 0)
        processed_image = crop_and_pad(tissue_image, filled_tissue_mask, target_shape)

        if display_images:
            try:
                import matplotlib.pyplot as plt

                plt.figure()
                plt.imshow(processed_image)
                plt.title(f"Tissue Region {i}")
                plt.axis("off")
                plt.show()
            except ImportError:
                pass

        # Convert processed image to a Dask array and add it to the list
        tissue_images.append(da.from_array(processed_image, chunks=original_img.shape))

    # Stack the images along a new dimension
    tissue_image_stack = da.stack(tissue_images, axis=0)

    return tissue_image_stack


def sort_labels_left_to_right(filled_lr_lbl: np.ndarray) -> list[int]:
    """
    Return label IDs sorted by low-res centroid x (left->right).

    Parameters
    ----------
    filled_lr_lbl : np.ndarray
        Labeled image with filled regions.

    Returns
    -------
    list of int
        Label IDs sorted by x-coordinate of centroid.
    """
    props = measure.regionprops(filled_lr_lbl)
    return [p.label for p in sorted(props, key=lambda p: p.centroid[1])]


def generate_tissue_tile_records(
    s0_cyx: da.Array,
    low_res_filled: np.ndarray,
    *,
    tile_frame_level: TileFrameLevel,
    chunk: int = 512,
    pad_multiple: int | None = None,
    extra_margin_px: int = 0,
) -> tuple[list[TissueTileRecord], int]:
    """
    Build one high-res (Y,X,C) Dask tile record per tissue.

    ``low_res_filled`` is segmented at thumbnail/coarse scale, then labels are
    mapped back onto the high-resolution image only for each output window.  All
    output windows use the same square dimension, are centered on their tissue
    bounding box, and are padded when the centered square extends beyond the
    source image.

    Parameters
    ----------
    s0_cyx : da.Array
        High-resolution image (C, Y, X).
    low_res_filled : np.ndarray
        Boolean mask at coarsest level.
    tile_frame_level : {"source", "segmentation"}
        Coordinate level where tile size, padding, and margin are defined.
        ``"source"`` preserves the historical behavior. ``"segmentation"``
        finalizes notebook-style crop frames at the segmentation mask level
        before mapping them to source pixels.
    chunk : int
        Writer chunk size for output tiles.
    pad_multiple : int, optional
        Round tile dimension to multiple of this value. Defaults to chunk.
    extra_margin_px : int
        Extra margin around each tissue region, in ``tile_frame_level`` pixels.
    Returns
    -------
    records : list of TissueTileRecord
        Per-tissue tiles and source/segmentation crop bounds. Bounds are
        ``(x0, y0, x1, y1)`` exclusive in their parent level.
    tile_dim : int
        The common square side length used for padding/cropping.
    """
    assert s0_cyx.ndim == 3
    C, Yh, Xh = map(int, s0_cyx.shape)
    tile_frame_level = _normalize_tile_frame_level(tile_frame_level)

    if pad_multiple is None:
        pad_multiple = chunk

    # -------- Pass 0: label & fill at segmentation level --------
    lr_lbl, n_lr = measure.label(low_res_filled.astype(bool), connectivity=2, return_num=True)
    if n_lr == 0:
        return [], 0

    filled_lr_lbl = np.zeros_like(lr_lbl, dtype=np.int32)
    for lid in range(1, n_lr + 1):
        comp = lr_lbl == lid
        if comp.any():
            filled_lr_lbl[binary_fill_holes(comp)] = lid
    lr_lbl = filled_lr_lbl

    frame_specs, source_tile_dim = _build_tissue_frame_specs(
        lr_lbl,
        source_shape_yx=(Yh, Xh),
        tile_frame_level=tile_frame_level,
        pad_multiple=pad_multiple,
        extra_margin_px=extra_margin_px,
    )

    records: list[TissueTileRecord] = []
    for spec in frame_specs:
        src = spec.clipped_source_yx
        src_h = src.h
        src_w = src.w
        if src_h <= 0 or src_w <= 0:
            continue

        label_crop = spec.label_crop_seg_yx
        lr_crop_lbl = lr_lbl[label_crop.y0 : label_crop.y1, label_crop.x0 : label_crop.x1].astype(
            np.int32
        )
        matrix = np.array([[1.0 / spec.scale_y, 0.0], [0.0, 1.0 / spec.scale_x]], dtype=float)
        offset = np.array(
            [
                ((src.y0 + 0.5) / spec.scale_y) - 0.5 - label_crop.y0,
                ((src.x0 + 0.5) / spec.scale_x) - 0.5 - label_crop.x0,
            ],
            dtype=float,
        )

        hr_roi_lbl = dask_affine_transform(
            da.from_array(lr_crop_lbl, chunks=lr_crop_lbl.shape),
            matrix=matrix,
            offset=offset,
            output_shape=(src_h, src_w),
            order=0,
        ).astype(np.int32)
        hr_roi_mask = hr_roi_lbl == spec.label_id

        s0_roi_cyx = s0_cyx[:, src.y0 : src.y1, src.x0 : src.x1]
        roi_yxc = da.moveaxis(s0_roi_cyx, 0, -1)

        masked = da.where(hr_roi_mask[..., None], roi_yxc, 0)
        padding = spec.padding_source_level
        pad_top = padding.top
        pad_left = padding.left
        pad_bottom = padding.bottom
        pad_right = padding.right
        if pad_top or pad_bottom or pad_left or pad_right:
            masked = da.pad(
                masked,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
            )
        tile = masked[:source_tile_dim, :source_tile_dim, :].rechunk((chunk, chunk, C))
        records.append(
            TissueTileRecord(
                tile=tile,
                tissue_index=spec.tissue_index,
                label_id=spec.label_id,
                crop_bounds_source_level=spec.clipped_source_yx.as_xyxy(),
                crop_bounds_segmentation_level=spec.clipped_frame_seg_yx.as_xyxy(),
                tile_dim=int(source_tile_dim),
                tile_frame_level=tile_frame_level,
                source_tile_dim=int(source_tile_dim),
                segmentation_tile_dim=int(spec.segmentation_tile_dim),
                scale_y=float(spec.scale_y),
                scale_x=float(spec.scale_x),
                frame_debug=spec.debug_dict(),
            )
        )

    return records, int(source_tile_dim)


def generate_tissue_tiles(
    s0_cyx: da.Array,
    low_res_filled: np.ndarray,
    *,
    tile_frame_level: TileFrameLevel,
    chunk: int = 512,
    pad_multiple: int | None = None,
    extra_margin_px: int = 0,
) -> tuple[list[da.Array], int]:
    """
    Build one high-res (Y,X,C) Dask tile per tissue with a common square side.

    This compatibility wrapper preserves the historical return value. New
    provenance-aware callers should use :func:`generate_tissue_tile_records`.
    """
    records, tile_dim = generate_tissue_tile_records(
        s0_cyx=s0_cyx,
        low_res_filled=low_res_filled,
        chunk=chunk,
        pad_multiple=pad_multiple,
        extra_margin_px=extra_margin_px,
        tile_frame_level=tile_frame_level,
    )
    return [record.tile for record in records], tile_dim
