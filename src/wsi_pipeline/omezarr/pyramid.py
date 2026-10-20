"""
Pyramid building utilities for multiscale image pyramids.

Provides functions for computing the number of MIP levels and
building power-of-two pyramids using efficient downsampling.
"""

from __future__ import annotations

import numpy as np


def compute_num_mips_min_side(width: int, height: int, min_side_for_mips: int) -> int:
    """
    Return number of power-of-two levels (>=1) while min(H,W) >= 2*min_side_for_mips.
    This keeps leaf mips at least ~min_side_for_mips in XY (good with chunk_xy).
    """
    levels = 1
    w, h = width, height
    while min(w, h) >= 2 * min_side_for_mips:
        w //= 2
        h //= 2
        levels += 1
    return levels


def _average_pool_2x2_yxc(img: np.ndarray) -> np.ndarray:
    """2x2 block mean of a (H, W, C) array, ceil-sized; an edge block averages the pixels it has.

    Odd edges are padded by replication, which gives exactly that partial-block mean.
    Integer outputs are rounded to nearest (sums of 4 values are exact in float32 up to
    16-bit inputs) so rounding bias does not accumulate across levels.
    """
    h, w = img.shape[:2]
    pad = ((0, h % 2), (0, w % 2)) + ((0, 0),) * (img.ndim - 2)
    work_dtype = np.float32 if img.dtype.itemsize <= 2 else np.float64
    padded = np.pad(np.asarray(img), pad, mode="edge").astype(work_dtype, copy=False)
    pooled = padded.reshape((h + 1) // 2, 2, (w + 1) // 2, 2, *img.shape[2:]).mean(axis=(1, 3))
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        pooled = np.clip(np.rint(pooled), info.min, info.max)
    return pooled.astype(img.dtype, copy=False)


# Bound conversion buffers independently of image size (4 MiB float32 input tile).
_TINYBRAIN_TILE_SIDE = 1024
_TINYBRAIN_MIN_PIXELS = 4096


def _pool_2x2_hybrid(img: np.ndarray) -> np.ndarray:
    """Accelerate unsigned microscopy data without changing NumPy's rounding contract."""
    h, w = img.shape[:2]
    if (
        img.ndim != 3
        or img.dtype not in (np.dtype("uint8"), np.dtype("uint16"))
        or min(h, w) < 2
        or h * w < _TINYBRAIN_MIN_PIXELS
    ):
        return _average_pool_2x2_yxc(img)

    try:
        from tinybrain.accelerated import average_pooling_2x2
    except ImportError:
        return _average_pool_2x2_yxc(img)

    out = np.empty(((h + 1) // 2, (w + 1) // 2, img.shape[2]), dtype=img.dtype)
    even_h, even_w = h - h % 2, w - w % 2
    # The raw kernel ignores strides. Choose the faster spatial axis first, then
    # make only a bounded plane tile Fortran contiguous. Float32 represents every
    # uint8/uint16 four-pixel sum exactly; integer kernels would truncate the mean.
    transpose = abs(img.strides[1]) < abs(img.strides[0])
    side = _TINYBRAIN_TILE_SIDE  # even: every tile boundary is a pooling boundary
    for c in range(img.shape[2]):
        for y in range(0, even_h, side):
            for x in range(0, even_w, side):
                plane = img[y : min(y + side, even_h), x : min(x + side, even_w), c]
                work = np.array(plane.T if transpose else plane, dtype=np.float32, order="F")
                # One level at a time: multilevel tinybrain retains unrounded sums.
                pooled = average_pooling_2x2(work, 1)[0]
                np.rint(pooled, out=pooled)  # nearest, ties to even, at EVERY level
                if transpose:
                    pooled = pooled.T
                out[y // 2 : y // 2 + pooled.shape[0], x // 2 : x // 2 + pooled.shape[1], c] = (
                    pooled
                )
                del work, pooled

    # Never pass odd dimensions to the native kernel. These thin strips use the
    # reference partial-block mean, including the bottom-right corner.
    if h % 2:
        out[-1:, :, :] = _average_pool_2x2_yxc(img[-1:, :, :])
    if w % 2:
        out[:, -1:, :] = _average_pool_2x2_yxc(img[:, -1:, :])
    return out


def build_mips_from_yxc(base_yxc: np.ndarray, num_mips: int) -> list[np.ndarray]:
    """
    Make a power-of-two mip chain by 2x2 block averaging.
    Returns [mip0, mip1, ...], each (H, W, C); level k has ceil(H / 2**k) x ceil(W / 2**k) pixels.

    Use this only for data without a native pyramid (derived or cropped in-memory
    tiles, masks, synthetic images); a source that already stores a pyramid should be
    copied level by level instead (see ``write_ets_pyramid_to_ngff_zarr``).

    Uint8/uint16 images use tinybrain on bounded even tiles, with exact float32
    sums and NumPy round-to-nearest, ties-to-even after each level. Odd borders
    average only available pixels. Small images, other dtypes (including float32,
    which retains float64 accumulation), and missing tinybrain use NumPy.
    Layout conversion never copies an entire large image. Input strides and
    channel order are preserved; generated levels need not share the base layout.
    """
    if num_mips <= 1:
        return [base_yxc]

    mips = [base_yxc]
    for _ in range(num_mips - 1):
        mips.append(_pool_2x2_hybrid(mips[-1]))
    return mips
