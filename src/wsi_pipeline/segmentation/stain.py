"""
Stain-aware preprocessing helpers for brightfield histology.

The default tissue segmentation backends are intentionally generic.  These
helpers add an optional H&E-specific gate that removes pixels with weak stain
signal before morphological closing can merge them into tissue components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from skimage.color import rgb2hed, rgb2hsv

StainGateMode = Literal["fixed", "adaptive-od", "adaptive-he"]


@dataclass(frozen=True)
class HEFeatures:
    """Reusable H&E-aware per-pixel features for brightfield thumbnails."""

    saturation: np.ndarray
    optical_density: np.ndarray
    hematoxylin: np.ndarray
    eosin: np.ndarray
    he_signal: np.ndarray


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


def he_features(image: np.ndarray) -> HEFeatures:
    """Compute RGB, OD, and color-deconvolved H&E feature images."""
    rgb = _as_rgb_float(image)
    saturation = rgb2hsv(rgb)[..., 1]
    optical_density = -np.log(np.clip(rgb, 1.0 / 255.0, 1.0)).sum(axis=-1)
    hed = rgb2hed(rgb)
    hematoxylin = np.clip(hed[..., 0], 0, None)
    eosin = np.clip(hed[..., 1], 0, None)
    he_signal = hematoxylin + eosin
    return HEFeatures(
        saturation=saturation,
        optical_density=optical_density,
        hematoxylin=hematoxylin,
        eosin=eosin,
        he_signal=he_signal,
    )


def _adaptive_threshold_from_background(
    signal: np.ndarray,
    intensity: np.ndarray,
    *,
    floor: float,
    bg_percentile: float,
    mad_multiplier: float,
) -> float:
    bg_cutoff = np.quantile(intensity, float(bg_percentile))
    bg = intensity >= bg_cutoff
    bg_signal = signal[bg]
    if bg_signal.size:
        bg_median = float(np.median(bg_signal))
        bg_mad = float(np.median(np.abs(bg_signal - bg_median)))
        robust_sigma = 1.4826 * bg_mad
        adaptive = bg_median + (float(mad_multiplier) * robust_sigma)
        return max(float(floor), adaptive)
    return float(floor)


def he_stain_mask(
    image: np.ndarray,
    *,
    mode: StainGateMode = "fixed",
    min_saturation: float = 0.08,
    min_od: float = 0.35,
    min_he_signal: float = 0.0,
    od_bg_percentile: float = 0.80,
    od_mad_multiplier: float = 4.0,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | str]]:
    """
    Identify pixels with H&E-like stain signal in a brightfield RGB image.

    In ``fixed`` mode, the gate combines two cues:
    - HSV saturation, which rejects near-neutral bright background.
    - Optical density, which rejects very transparent / weakly stained pixels.

    In ``adaptive-od`` mode, saturation is not required.  Instead, the optical
    density threshold is estimated from the brightest background-like pixels on
    each image.  ``adaptive-he`` follows the same robust-background strategy,
    but thresholds deconvolved hematoxylin+eosin signal.

    Optionally, a small positive threshold on the HED hematoxylin+eosin signal
    can rescue legitimate light tissue after color deconvolution.
    """
    rgb = _as_rgb_float(image)
    features = he_features(rgb)
    saturation = features.saturation
    optical_density = features.optical_density
    he_signal = features.he_signal

    if mode == "fixed":
        od_threshold = float(min_od)
        he_threshold = float(min_he_signal)
        stain = (saturation >= float(min_saturation)) & (optical_density >= od_threshold)
    elif mode == "adaptive-od":
        intensity = rgb.mean(axis=-1)
        od_threshold = _adaptive_threshold_from_background(
            optical_density,
            intensity,
            floor=float(min_od),
            bg_percentile=od_bg_percentile,
            mad_multiplier=od_mad_multiplier,
        )
        he_threshold = float(min_he_signal)
        stain = optical_density >= od_threshold
    elif mode == "adaptive-he":
        intensity = rgb.mean(axis=-1)
        od_threshold = _adaptive_threshold_from_background(
            optical_density,
            intensity,
            floor=float(min_od),
            bg_percentile=od_bg_percentile,
            mad_multiplier=od_mad_multiplier,
        )
        # Use a small floor so a pure-white background with near-zero MAD does
        # not turn every non-white pixel into stain by thresholding at zero.
        # Keep it low enough for lightly eosin-stained tissue; a weak
        # saturation condition below rejects neutral gray/background artifacts.
        he_floor = float(min_he_signal) if min_he_signal > 0 else 0.012
        he_threshold = _adaptive_threshold_from_background(
            he_signal,
            intensity,
            floor=he_floor,
            bg_percentile=od_bg_percentile,
            mad_multiplier=od_mad_multiplier,
        )
        saturation_floor = float(min_saturation) * 0.5
        he_positive = (he_signal >= he_threshold) & (saturation >= saturation_floor)
        od_positive = (optical_density >= od_threshold) & (saturation >= saturation_floor)
        stain = he_positive | od_positive
    else:
        raise ValueError("mode must be one of 'fixed', 'adaptive-od', or 'adaptive-he'")

    if min_he_signal > 0:
        stain = stain | (he_signal >= float(min_he_signal))

    if return_info:
        return stain, {
            "mode": mode,
            "od_threshold": float(od_threshold),
            "he_threshold": float(he_threshold),
            "min_saturation": float(min_saturation),
            "min_od": float(min_od),
            "min_he_signal": float(min_he_signal),
            "od_bg_percentile": float(od_bg_percentile),
            "od_mad_multiplier": float(od_mad_multiplier),
        }
    return stain


__all__ = ["HEFeatures", "StainGateMode", "he_features", "he_stain_mask"]
