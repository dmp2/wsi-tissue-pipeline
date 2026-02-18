"""
WSI Segmentation classes and functions.

Provides the WSISegmenter class for tissue mask generation with multiple backends.
"""

from __future__ import annotations

import importlib.util
import logging
import warnings

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import dask
import dask.array as da
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage import filters, measure, morphology
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk

# Local imports
from .core import create_thumbnail, to_gray, upsample_mask
from .entropy import entropy_mask
from .morphology import split_touching_components
from .otsu import otsu_mask

# Backend type definition
Backend = Literal[
    "local-entropy",
    "local-otsu",
    "tiatoolbox-otsu",
    "tiatoolbox-morph",
    "pathml-he",
]

# Optional dependency checks
_HAS_TIA = importlib.util.find_spec("tiatoolbox") is not None
_HAS_PATHML = importlib.util.find_spec("pathml") is not None


class WSISegmenter:
    """
    Whole-slide-image tissue mask generator with multiple backend capability.

    The strategy is to generate tissue masks at a constant, low-resolution
    image size (e.g. thumbnail long side length), then upsample the mask to
    the working resolution.

    Parameters
    ----------
    backend : str
        Segmentation backend: 'local-entropy', 'local-otsu', 'tiatoolbox-otsu',
        'tiatoolbox-morph', or 'pathml-he'.
    target_long_side : int
        Target size for thumbnail during segmentation.
    struct_elem_px : int
        Structuring element size in pixels.
    min_area_px : int
        Minimum tissue area in pixels.
    split_touching : bool
        Whether to split touching tissue sections.
    r_split : int
        Radius for splitting touching sections.
    diagnostics : bool
        Enable diagnostic output.

    Examples
    --------
    >>> segmenter = WSISegmenter(backend="local-entropy")
    >>> mask = segmenter(image_array)
    """

    def __init__(
        self,
        backend: Backend = "local-entropy",
        *,
        target_long_side: int = 1800,
        struct_elem_px: int = 9,
        min_area_px: int = 2000,
        split_touching: bool = True,
        r_split: int = 2,
        diagnostics: bool = False,
    ):
        self.backend = backend
        self.target_long_side = int(target_long_side)
        self.struct_elem_px = max(2, int(struct_elem_px))
        self.min_area_px = max(64, int(min_area_px))
        self.split_touching = bool(split_touching)
        self.r_split = int(r_split)
        self.diagnostics = diagnostics

        # Backend checks
        if backend.startswith("tiatoolbox"):
            if not _HAS_TIA:
                raise ImportError(
                    "TIAToolbox not found in this Python. "
                    "Verify the interpreter: `python -c \"import sys; print(sys.executable)\"` "
                    "and install with `python -m pip install tiatoolbox`."
                )
            # Try the specific module and surface the *real* error if it fails
            try:
                from tiatoolbox.tools.tissuemask import (
                    MorphologicalMasker,
                    OtsuTissueMasker,
                )
            except Exception as e:
                raise ImportError(
                    "TIAToolbox is present but `tiatoolbox.tools.tissuemask` failed to import. "
                    "This usually indicates missing OS prerequisites (OpenSlide/OpenJPEG) or a "
                    "mixed environment. See TIAToolbox install guide."
                ) from e

        if backend == "pathml-he" and not _HAS_PATHML:
            raise ImportError("PathML not installed but backend requested.")

    def __call__(self, image: Union[np.ndarray, da.Array]) -> np.ndarray:
        """
        Generate boolean mask for a channel-last image.

        Parameters
        ----------
        image : np.ndarray or dask.array.Array
            Input image (H, W, C) or (H, W).

        Returns
        -------
        np.ndarray
            Boolean mask (H, W).
        """
        # 1) Make thumbnail (NumPy) at constant long side
        thumb, scale = create_thumbnail(image, self.target_long_side)
        if isinstance(image, (np.ndarray, da.Array)):
            H, W = image.shape[0], image.shape[1]
        else:
            H, W = thumb.shape[0], thumb.shape[1]

        # 2) Scale-aware params at thumbnail scale
        struct_r = max(2, int(round(self.struct_elem_px)))
        min_area = max(64, int(round(self.min_area_px)))

        # 3) Compute mask on thumbnail by backend
        if self.backend == "tiatoolbox-otsu":
            from tiatoolbox.tools.tissuemask import OtsuTissueMasker
            masker = OtsuTissueMasker()
            mask_t = masker.fit_transform([thumb])[0].astype(bool)

        elif self.backend == "tiatoolbox-morph":
            from tiatoolbox.tools.tissuemask import MorphologicalMasker
            masker = MorphologicalMasker(
                kernel_size=int(self.struct_elem_px),
                min_region_size=int(self.min_area_px),
            )
            mask_t = masker.fit_transform([thumb])[0].astype(bool)

        elif self.backend == "pathml-he":
            from pathml.preprocessing import TissueDetectionHE
            td = TissueDetectionHE(min_region_size=min_area)
            g = to_gray(thumb)
            mask_t = td._apply(g)
            mask_t = mask_t.astype(bool)

        elif self.backend == "local-otsu":
            g = to_gray(thumb)
            mask_t = otsu_mask(g, struct_r, min_area)

        elif self.backend == "local-entropy":
            g = to_gray(thumb)
            mask_t = entropy_mask(g, struct_r, min_area)

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        # 4) Optional thin-neck split on thumbnail
        lbl_b = measure.label(mask_t, connectivity=2).max()
        if self.split_touching:
            mask_t = split_touching_components(mask_t, self.r_split, min_area)
        lbl_a = measure.label(mask_t, connectivity=2).max()

        if self.diagnostics:
            logger.debug(
                "[segmenter] backend=%s size_t=%s struct_elem_px=%s min_area=%s "
                "r_split=%s CCs: %s->%s",
                self.backend, mask_t.shape, struct_r, min_area, self.r_split, lbl_b, lbl_a
            )

        # 5) Upsample back to full resolution and return
        mask_full = upsample_mask(mask_t, (H, W))
        return mask_full


def segment_mask(
    image: Union[np.ndarray, da.Array],
    *,
    backend: Backend = "local-entropy",
    target_long_side: int = 1800,
    struct_elem_px: int = 9,
    min_area_px: int = 2000,
    split_touching: bool = True,
    r_split: int = 2,
    diagnostics: bool = False,
) -> np.ndarray:
    """
    Generate tissue mask from image using specified backend.

    This is a convenience function that creates a WSISegmenter and runs it.

    Parameters
    ----------
    image : np.ndarray or dask.array.Array
        Input image (H, W, C).
    backend : str
        Segmentation backend.
    target_long_side : int
        Target thumbnail size.
    struct_elem_px : int
        Structuring element size.
    min_area_px : int
        Minimum tissue area.
    split_touching : bool
        Whether to split touching sections.
    r_split : int
        Split radius.
    diagnostics : bool
        Enable diagnostics.

    Returns
    -------
    np.ndarray
        Boolean tissue mask.
    """
    seg = WSISegmenter(
        backend=backend,
        target_long_side=target_long_side,
        struct_elem_px=struct_elem_px,
        min_area_px=min_area_px,
        split_touching=split_touching,
        r_split=r_split,
        diagnostics=diagnostics,
    )
    return seg(image)


def make_lowres_mask(
    dask_img: da.Array,
    *,
    dynamic_threshold: bool = True,
    fixed_threshold: float = 0.7,
    min_size: int = 2000,
    struct_elem_px: int = 9,
    additional_smooth: bool = False,
    output_images: bool = True,
    ref_hw: Tuple[int, int] = (1458, 2814),
    edge_only: bool = True,
    small_switch: int = 1200,
    keep_top_k: Optional[int] = None,
    split_touching: bool = True,
    r_split: int = 2,
    diagnostics: bool = False,
    return_diag: bool = False,
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Create low-resolution tissue mask using entropy-based segmentation.

    This function is designed to work with Dask arrays and provides
    scale-adaptive segmentation parameters.

    Parameters
    ----------
    dask_img : dask.array.Array
        Input image as Dask array (C, Y, X) or (Y, X, C).
    dynamic_threshold : bool
        Use Otsu thresholding if True.
    fixed_threshold : float
        Fixed threshold value if dynamic_threshold is False.
    min_size : int
        Minimum object size at reference scale.
    struct_elem_px : int
        Structuring element size at reference scale.
    additional_smooth : bool
        Apply additional opening operation.
    output_images : bool
        Return mask (always True for backward compat).
    ref_hw : tuple
        Reference (H, W) size for parameter scaling.
    edge_only : bool
        Use only entropy edges (vs. combined with intensity).
    small_switch : int
        Size threshold below which to combine entropy and intensity.
    keep_top_k : int, optional
        Keep only the K largest components.
    split_touching : bool
        Split touching components.
    r_split : int
        Radius for splitting.
    diagnostics : bool
        Print diagnostic info.
    return_diag : bool
        Return diagnostic dict.

    Returns
    -------
    mask : np.ndarray
        Boolean tissue mask.
    diag : dict or None
        Diagnostic information if return_diag is True.
    """
    # Convert to grayscale (channel-last input)
    if dask_img.ndim == 3 and dask_img.shape[0] in (1, 3) and dask_img.shape[-1] not in (1, 3):
        dask_img = da.moveaxis(dask_img, 0, -1)
    gray = to_gray(dask_img).astype(np.float32)

    H, W = int(gray.shape[0]), int(gray.shape[1])
    refH, refW = ref_hw
    len_scale = max(H, W) / max(refH, refW)
    area_scale = (H * W) / max(1, refH * refW)

    se_r = max(2, int(round(struct_elem_px * len_scale)))
    min_area = max(64, int(round(min_size * area_scale)))

    # Entropy with overlap (edge-focused)
    def _entropy_u8(x):
        x = x - x.min()
        rng = x.max() - x.min() + 1e-6
        x = (x / rng * 255.0).astype(np.uint8)
        return rank_entropy(x, disk(se_r))

    ent = da.map_overlap(_entropy_u8, gray, depth=(se_r, se_r), boundary="reflect", dtype=np.uint8)
    ent = da.map_blocks(lambda t: (t - t.min()) / (t.max() - t.min() + 1e-6), ent, dtype=np.float32)

    # Thresholds
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Running on a single-machine scheduler")
        with dask.config.set(scheduler="synchronous"):
            ent_np = ent.compute()
            thr_ent = filters.threshold_otsu(ent_np) if dynamic_threshold else fixed_threshold
            if not edge_only or min(H, W) < small_switch:
                gray_np = gray.compute()
                thr_gray = filters.threshold_otsu(gray_np)

    mask_ent = ent > thr_ent
    if not edge_only or min(H, W) < small_switch:
        mask_int = gray < thr_gray
        bw0 = mask_ent | mask_int
    else:
        bw0 = mask_ent

    # Morphology (with overlap) to connect edges
    nhood = disk(se_r)
    bw1 = da.map_overlap(
        lambda x: morphology.binary_closing(x, nhood),
        bw0,
        depth=(se_r, se_r),
        boundary="reflect",
        dtype=bool,
    )
    if additional_smooth:
        bw1 = da.map_overlap(
            lambda x: morphology.binary_opening(x, nhood),
            bw1,
            depth=(se_r, se_r),
            boundary="reflect",
            dtype=bool,
        )

    # Global ops
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Running on a single-machine scheduler")
        bw = bw1.compute()

    # Fill outlines -> solid islands
    bw = binary_fill_holes(bw)
    bw = morphology.remove_small_objects(bw, min_size=min_area)
    bw = morphology.remove_small_holes(bw, area_threshold=min_area)

    diag: Dict = {}
    if not bw.any():
        return (bw, diag if return_diag else None)

    # Counts before split
    lbl_before = measure.label(bw, connectivity=2)
    n_before = int(lbl_before.max())
    areas_before = [p.area for p in measure.regionprops(lbl_before)]
    areas_before.sort(reverse=True)
    bw_split = bw
    n_after = n_before

    # Split accidental bridges only if present
    if split_touching:
        cand = split_touching_components(bw, r_split=r_split, min_area=min_area)
        lbl_after = measure.label(cand, connectivity=2)
        n_cand = int(lbl_after.max())

        # Accept only if we didn't *lose* sections
        if n_cand >= n_before or n_cand == n_before + 1:
            bw_split = cand
            n_after = n_cand
        else:
            n_after = n_before

    bw = bw_split

    lbl_after = measure.label(bw, connectivity=2)
    n_after = int(lbl_after.max())
    areas_after = [p.area for p in measure.regionprops(lbl_after)]
    areas_after.sort(reverse=True)

    # Optionally keep top-K largest
    n_kept = n_after
    if keep_top_k:
        keep = {
            p.label
            for p in sorted(
                measure.regionprops(lbl_after), key=lambda p: p.area, reverse=True
            )[:keep_top_k]
        }
        bw = np.isin(lbl_after, list(keep))
        n_kept = int(keep_top_k)

    diag = {
        "H": H,
        "W": W,
        "struct_elem_px": se_r,
        "min_area": min_area,
        "r_split": r_split,
        "n_before": n_before,
        "n_after": n_after,
        "n_kept": n_kept,
        "top_areas_before": areas_before[:6],
        "top_areas_after": areas_after[:6],
    }

    if diagnostics:
        logger.debug(
            "[preprocess] size=(%dx%d) struct_elem_px=%s min_area=%s r_split=%s components: %s -> %s",
            H, W, se_r, min_area, r_split, n_before, n_after
            + (f" -> {n_kept}" if keep_top_k else "")
        )

    return (bw, diag if return_diag else None)
