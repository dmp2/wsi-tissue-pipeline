"""
Otsu-based tissue segmentation.

Fast global Otsu threshold for tissue detection.
"""

import numpy as np
from skimage import filters, morphology
from skimage.morphology import disk


def otsu_mask(gray: np.ndarray, struct_elem_px: int, min_area: int) -> np.ndarray:
    """
    Create tissue mask using fast global Otsu threshold on blurred thumbnail.

    This follows the QuPath/TIA style approach.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale thumbnail image.
    struct_elem_px : int
        Structuring element radius for morphological closing.
    min_area : int
        Minimum object area in pixels.

    Returns
    -------
    np.ndarray
        Boolean tissue mask.
    """
    sm = filters.gaussian(gray, sigma=1.0, preserve_range=True)
    thr = filters.threshold_otsu(sm)
    bw = sm < thr
    bw = morphology.binary_closing(bw, footprint=disk(struct_elem_px))
    bw = morphology.remove_small_holes(bw, area_threshold=min_area)
    bw = morphology.remove_small_objects(bw, min_size=min_area)
    return bw


# Backward compatibility
_otsu_mask = otsu_mask
