"""
Core segmentation utilities.

Provides grayscale conversion, thumbnail generation, and mask upsampling.
"""


import dask.array as da
import numpy as np
from skimage.transform import resize


def to_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert channel-last image to grayscale (float32, preserve_range).

    Parameters
    ----------
    image : np.ndarray
        Image array, either (H, W), (C, H, W), or (H, W, C).

    Returns
    -------
    np.ndarray
        Grayscale image as float32.
    """
    if image.ndim == 2:  # already gray
        return image.astype(np.float32)
    if image.shape[0] == 3:
        r, g, b = image[0, ...], image[1, ...], image[2, ...]
    elif image.shape[-1] == 3:
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
    else:
        # single-channel last
        return np.squeeze(image).astype(np.float32)
    return (0.2125 * r + 0.7154 * g + 0.0721 * b).astype(np.float32)


# Backward compatibility
_to_gray = to_gray


def create_thumbnail(image: np.ndarray | da.Array, target_long_side: int) -> tuple[np.ndarray, float]:
    """
    Create a NumPy thumbnail and return the scale factor.

    Parameters
    ----------
    image : np.ndarray or dask.array.Array
        Input image.
    target_long_side : int
        Target size for the longest side.

    Returns
    -------
    thumbnail : np.ndarray
        Resized image.
    scale : float
        Scale factor applied (thumbnail_size / original_size).
    """
    if isinstance(image, da.Array):
        H, W = int(image.shape[0]), int(image.shape[1])
        arr = image.compute()
    else:
        H, W = image.shape[:2]
        arr = image
    if target_long_side and max(H, W) != target_long_side:
        scale = target_long_side / max(H, W)
        Ht, Wt = max(1, int(round(H * scale))), max(1, int(round(W * scale)))
        out = resize(arr, (Ht, Wt), order=1, preserve_range=True, anti_aliasing=True)
        return out.astype(arr.dtype, copy=False), scale
    return arr, 1.0


# Backward compatibility
_thumb = create_thumbnail


def upsample_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Nearest-neighbor upsample of boolean mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask at thumbnail resolution.
    target_shape : tuple
        Target shape (H, W).

    Returns
    -------
    np.ndarray
        Boolean mask at full resolution.
    """
    H, W = target_shape
    return resize(mask.astype(np.uint8), (H, W), order=0, preserve_range=True).astype(bool)


# Backward compatibility
_upsample_bool = upsample_mask
