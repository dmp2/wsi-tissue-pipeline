"""
Stain-aware preprocessing helpers for brightfield histology.

The default tissue segmentation backends are intentionally generic.  These
helpers add an optional H&E-specific gate that removes pixels with weak stain
signal before morphological closing can merge them into tissue components.
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2hed, rgb2hsv


def _as_rgb_float(image: np.ndarray) -> np.ndarray:
    """Return a channel-last RGB image scaled to [0, 1]."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError("Expected an RGB image with shape (Y, X, 3).")

    arr = arr[..., :3].astype(np.float32, copy=False)
    if np.issubdtype(np.asarray(image).dtype, np.integer):
        info = np.iinfo(np.asarray(image).dtype)
        arr = arr / float(info.max)
    elif arr.max(initial=0) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def he_stain_mask(
    image: np.ndarray,
    *,
    min_saturation: float = 0.08,
    min_od: float = 0.35,
    min_he_signal: float = 0.0,
) -> np.ndarray:
    """
    Identify pixels with H&E-like stain signal in a brightfield RGB image.

    The gate combines two robust cues:
    - HSV saturation, which rejects near-neutral bright background.
    - Optical density, which rejects very transparent / weakly stained pixels.

    Optionally, a small positive threshold on the HED hematoxylin+eosin signal
    can rescue legitimate light tissue after color deconvolution.
    """
    rgb = _as_rgb_float(image)
    saturation = rgb2hsv(rgb)[..., 1]
    optical_density = -np.log(np.clip(rgb, 1.0 / 255.0, 1.0)).sum(axis=-1)

    stain = (saturation >= float(min_saturation)) & (optical_density >= float(min_od))
    if min_he_signal > 0:
        hed = rgb2hed(rgb)
        he_signal = np.clip(hed[..., 0], 0, None) + np.clip(hed[..., 1], 0, None)
        stain = stain | (he_signal >= float(min_he_signal))
    return stain


__all__ = ["he_stain_mask"]
