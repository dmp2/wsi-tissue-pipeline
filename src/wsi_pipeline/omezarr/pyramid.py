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


def build_mips_from_yxc(base_yxc: np.ndarray, num_mips: int) -> list[np.ndarray]:
    """
    Make a power-of-two mip chain using tinybrain's 2x2 average pooling.
    Returns [mip0, mip1, ...], each (H, W, C). No recompute later.
    """
    if num_mips <= 1:
        return [base_yxc]

    import tinybrain  # local import so module loads without hard dep

    # Get channel dim
    C = base_yxc.shape[-1]

    # tinybrain operates on 2D; pool channel-by-channel, then restack
    per_channel = []
    for c in range(C):
        first = base_yxc[..., c]               # (H,W)
        downs = list(tinybrain.accelerated.average_pooling_2x2(first, num_mips - 1))
        per_channel.append([first] + downs)         # len = num_mips

    mips = [np.stack([per_channel[c][lvl] for c in range(C)], axis=-1)
            for lvl in range(num_mips)] # list of (H,W,C)

    return mips
