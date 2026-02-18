"""
Tile generation utilities for WSI processing.

Provides functions for extracting tissue tiles from whole-slide images,
including ROI upsampling and masking.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import dask.array as da
import numpy as np
from dask_image.ndinterp import affine_transform
from scipy.ndimage import binary_fill_holes
from skimage import measure


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
    target_shape: Union[np.ndarray, int],
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
    cropped_image = image[ymin:ymax + 1, xmin:xmax + 1]
    
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
    original_img: Union[np.ndarray, da.Array],
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


def sort_labels_left_to_right(filled_lr_lbl: np.ndarray) -> List[int]:
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


def generate_tissue_tiles(
    s0_cyx: da.Array,
    low_res_filled: np.ndarray,
    *,
    chunk: int = 512,
    pad_multiple: Optional[int] = None,
    extra_margin_px: int = 0,
) -> Tuple[List[da.Array], int]:
    """
    Build one high-res (Y,X,C) Dask tile per tissue with common square side length.
    
    The common side length is derived from the largest high-res bounding box
    across all tissues.
    
    Parameters
    ----------
    s0_cyx : da.Array
        High-resolution image (C, Y, X).
    low_res_filled : np.ndarray
        Boolean mask at coarsest level.
    chunk : int
        Chunk size for output tiles.
    pad_multiple : int, optional
        Round tile dimension to multiple of this value. Defaults to chunk.
    extra_margin_px : int
        Extra margin around each tissue region.
    
    Returns
    -------
    tiles : list of da.Array
        Per-tissue tiles (Y, X, C), rechunked to (chunk, chunk, C).
    tile_dim : int
        The common square side length used for padding/cropping.
    """
    assert s0_cyx.ndim == 3
    C, Yh, Xh = s0_cyx.shape

    if pad_multiple is None:
        pad_multiple = chunk

    # -------- Pass 0: label & fill at low-res --------
    lr_lbl, n_lr = measure.label(low_res_filled.astype(bool), connectivity=2, return_num=True)
    
    # If there are no tissue sections return early
    if n_lr == 0:
        return [], 0

    filled_lr_lbl = np.zeros_like(lr_lbl, dtype=np.int32)
    for lid in range(1, n_lr + 1):
        comp = lr_lbl == lid
        if comp.any():
            comp_filled = binary_fill_holes(comp)
            filled_lr_lbl[comp_filled] = lid
    lr_lbl = filled_lr_lbl

    # Ratios LR -> HR
    Yr = Yh / low_res_filled.shape[0]
    Xr = Xh / low_res_filled.shape[1]

    # -------- Pass 1: compute HR bboxes & find common tile_dim --------
    roi_specs = []
    max_side = 0
    
    for lid in sort_labels_left_to_right(lr_lbl):
        lr_mask = lr_lbl == lid
        rows, cols = np.any(lr_mask, 1), np.any(lr_mask, 0)
        yi = np.where(rows)[0]
        xi = np.where(cols)[0]
        if yi.size == 0 or xi.size == 0:
            continue

        y0_lr, y1_lr = yi[[0, -1]]
        x0_lr, x1_lr = xi[[0, -1]]

        # Map LR bbox -> HR bbox (inclusive)
        y0_hr = int(np.floor(y0_lr * Yr))
        y1_hr = int(np.ceil((y1_lr + 1) * Yr) - 1)
        x0_hr = int(np.floor(x0_lr * Xr))
        x1_hr = int(np.ceil((x1_lr + 1) * Xr) - 1)

        # Expand by optional margin, clamped to image
        y0_hr = max(0, y0_hr - extra_margin_px)
        x0_hr = max(0, x0_hr - extra_margin_px)
        y1_hr = min(Yh - 1, y1_hr + extra_margin_px)
        x1_hr = min(Xh - 1, x1_hr + extra_margin_px)

        H_hr = y1_hr - y0_hr + 1
        W_hr = x1_hr - x0_hr + 1
        max_side = max(max_side, H_hr, W_hr)

        roi_specs.append((lid, y0_lr, y1_lr, x0_lr, x1_lr, y0_hr, y1_hr, x0_hr, x1_hr, H_hr, W_hr))

    if not roi_specs:
        return [], 0

    # Round up to a friendly square dimension
    def _round_up(v: int, m: int) -> int:
        return ((v + m - 1) // m) * m

    tile_dim = _round_up(max_side, pad_multiple)

    # Safety bound: cannot exceed full image
    tile_dim = min(tile_dim, max(Yh, Xh))

    # -------- Pass 2: build tiles lazily with common tile_dim --------
    tiles: List[da.Array] = []
    
    for (lid, y0_lr, y1_lr, x0_lr, x1_lr, y0_hr, y1_hr, x0_hr, x1_hr, H_hr, W_hr) in roi_specs:
        # Upsample LR labels only within the LR ROI to HR ROI size
        lr_crop_lbl = lr_lbl[y0_lr:y1_lr + 1, x0_lr:x1_lr + 1].astype(np.int32)
        
        # 2x2 affine matrix for scaling
        A_roi = np.array([
            [lr_crop_lbl.shape[1] / W_hr, 0],
            [0, lr_crop_lbl.shape[0] / H_hr],
        ], dtype=float)
        
        hr_roi_lbl = affine_transform(
            da.from_array(lr_crop_lbl, chunks=lr_crop_lbl.shape),
            matrix=A_roi,
            output_shape=(H_hr, W_hr),
            order=0,
        ).astype(np.int32)
        hr_roi_mask = hr_roi_lbl == lid

        # Slice s0 in ROI lazily -> (Y,X,C)
        s0_roi_cyx = s0_cyx[:, y0_hr:y1_hr + 1, x0_hr:x1_hr + 1]
        roi_yxc = da.moveaxis(s0_roi_cyx, 0, -1)

        # Apply HR mask and square-crop/pad lazily to the *common* tile_dim
        masked = da.where(hr_roi_mask[..., None], roi_yxc, 0)
        tile = center_crop_pad_dask(masked, tile_dim).rechunk((chunk, chunk, C))
        tiles.append(tile)

    return tiles, tile_dim
