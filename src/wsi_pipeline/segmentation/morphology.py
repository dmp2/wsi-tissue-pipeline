"""
Morphological operations for tissue segmentation.

Provides functions for splitting touching components and other morphological operations.
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import measure, morphology
from skimage.morphology import disk
from skimage.segmentation import watershed


def remove_small_objects(arr, min_size):
    """
    Remove small objects from boolean or labeled array.

    Parameters
    ----------
    arr : np.ndarray
        Boolean or labeled array.
    min_size : int
        Minimum object size to retain.

    Returns
    -------
    np.ndarray
        Array with small objects removed.
    """
    arr = np.asarray(arr)
    if arr.dtype == bool:
        return morphology.remove_small_objects(arr, min_size=min_size)
    else:
        # labeled array path
        return morphology.remove_small_objects(arr, min_size=min_size).astype(arr.dtype)


# Backward compatibility
_remove_small_objects = remove_small_objects


def binary_closing(binary_image, structure):
    """
    Perform binary closing (dilation followed by erosion).

    Parameters
    ----------
    binary_image : np.ndarray
        Binary image.
    structure : np.ndarray
        Structuring element.

    Returns
    -------
    np.ndarray
        Closed binary image.
    """
    dilated_img = binary_dilation(binary_image, structure=structure)
    closed_img = binary_erosion(dilated_img, structure=structure)
    return closed_img


def split_touching_components(mask_bool: np.ndarray, r_split: int = 2, min_area: int = 256) -> np.ndarray:
    """
    Split thin bridges between touching tissue components using watershed.

    Post-split cleanup is gentler (min_area_post = max(64, min_area//2))
    so small real sections survive.

    Parameters
    ----------
    mask_bool : np.ndarray
        Boolean tissue mask.
    r_split : int
        Erosion radius for seed generation.
    min_area : int
        Minimum object area in pixels.

    Returns
    -------
    np.ndarray
        Mask with touching components separated.
    """
    mask_bool = np.asarray(mask_bool, dtype=bool)
    if not mask_bool.any() or r_split <= 0:
        return morphology.remove_small_objects(mask_bool, min_size=min_area)

    # Markers from a light erosion (interiors)
    seeds = morphology.binary_erosion(mask_bool, footprint=disk(r_split))
    markers = measure.label(seeds, connectivity=2)

    # Fallback: if erosion killed markers for a small object, seed its distance peak
    if markers.max() == 0:
        dist = ndi.distance_transform_edt(mask_bool)
        peak = np.zeros_like(mask_bool, dtype=bool)
        # seed at the global max distance (one marker)
        if dist.max() > 0:
            peak[np.unravel_index(np.argmax(dist), dist.shape)] = True
        markers = measure.label(peak, connectivity=2)

    dist = ndi.distance_transform_edt(mask_bool)
    lbl = watershed(-dist, markers=markers, mask=mask_bool, watershed_line=True)
    out = lbl > 0

    # gentler cleanup
    min_area_post = max(64, min_area // 2)
    out = morphology.remove_small_objects(out, min_size=min_area_post)
    return out


# Backward compatibility
_split_touching_components = split_touching_components
