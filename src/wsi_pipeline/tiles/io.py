"""
Tile I/O utilities for saving tissue tiles in various formats.

Provides functions for converting and saving tiles as JPEG, PNG, or TIFF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile as tf
from PIL import Image


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Robust per-channel conversion to uint8 for ubiquitous formats (esp. JPEG).
    Scales each channel independently to 0..255 based on min/max.
    """
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a

    if a.ndim == 2:  # grayscale
        mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return np.zeros_like(a, dtype=np.uint8)
        out = (np.clip((a - mn) / (mx - mn), 0, 1) * 255.0).round().astype(np.uint8)
        return out

    if a.ndim == 3:
        h, w, c = a.shape
        flat = a.reshape(-1, c)
        mn = np.nanmin(flat, axis=0).astype(np.float32)
        mx = np.nanmax(flat, axis=0).astype(np.float32)
        mx = np.where(mx <= mn, mn + 1.0, mx)
        norm = np.clip((a - mn) / (mx - mn), 0, 1)
        out = (norm * 255.0).round().astype(np.uint8)
        return out

    raise ValueError(f"Unexpected array ndim for uint8 conversion: {a.ndim}")


def save_tile(
    tile_np: np.ndarray,
    out_path: Path,
    ext: str,
    *,
    jpeg_quality: int = 95,
    png_compress_level: int = 6,
    tiff_compress: str = "deflate",
    convert_to_uint8: bool = True,
) -> None:
    """
    Save a single tile to disk, handling dtype/shape per format.
    - JPEG: always uint8, grayscale (L) or RGB
    - PNG: typically uint8; if convert_to_uint8=False and data suits Pillow, it will try to preserve dtype.
           (If you really need 16-bit per-channel reliably, prefer TIFF.)
    - TIFF: uses tifffile; preserves dtype if convert_to_uint8=False
    """
    # Drop alpha if any sneaks in
    if tile_np.ndim == 3 and tile_np.shape[-1] == 4:
        tile_np = tile_np[..., :3]

    # Squeeze singleton channel for L images when saving with Pillow
    if tile_np.ndim == 3 and tile_np.shape[-1] == 1:
        tile_np = tile_np[..., 0]

    if ext == "jpg":
        # JPEG needs 8-bit
        if tile_np.dtype != np.uint8:
            tile_np = to_uint8(tile_np)
        im = Image.fromarray(tile_np)  # "L" if 2D, "RGB" if 3D last-dim=3
        im.save(out_path, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)
        return

    if ext == "png":
        if convert_to_uint8 and tile_np.dtype != np.uint8:
            tile_np = to_uint8(tile_np)
        im = Image.fromarray(tile_np)
        im.save(out_path, format="PNG", compress_level=png_compress_level, optimize=True)
        return

    if ext == "tiff":
        if convert_to_uint8 and tile_np.dtype != np.uint8:
            tile_np = to_uint8(tile_np)
        # tifffile supports many dtypes and compressions
        tf.imwrite(str(out_path), tile_np, compression=tiff_compress)
        return

    raise AssertionError("unreachable")
