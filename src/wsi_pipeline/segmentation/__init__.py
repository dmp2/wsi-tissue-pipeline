"""
Tissue Segmentation Module

Generates boolean tissue masks from whole-slide image thumbnails using
multiple algorithm backends. The strategy is to segment at a constant
low-resolution thumbnail size, then upsample the mask to the working
resolution — keeping memory use predictable regardless of slide size.

Available backends
------------------
local-entropy    Local entropy thresholding (default, no extra dependencies)
local-otsu       Otsu thresholding with morphological post-processing
tiatoolbox-otsu  Otsu masker from TIAToolbox (requires tiatoolbox)
tiatoolbox-morph Morphological masker from TIAToolbox (requires tiatoolbox)
pathml-he        H&E-aware masker from PathML (requires pathml)

Public API
----------
WSISegmenter     Class interface — configure once, call on multiple images
segment_mask     Functional interface — one-shot convenience wrapper
make_lowres_mask Dask-native segmentation for pipeline integration
"""

from typing import Literal

# Backend type definition
Backend = Literal[
    "local-entropy",
    "local-otsu",
    "tiatoolbox-otsu",
    "tiatoolbox-morph",
    "pathml-he",
]

# Core utilities
from .core import to_gray, create_thumbnail, upsample_mask
# Deprecated aliases — kept for backward compatibility, will be removed in v1.0
from .core import _to_gray, _thumb, _upsample_bool  # deprecated

# Segmentation algorithms
from .entropy import entropy_mask, _entropy_mask  # _entropy_mask deprecated
from .otsu import otsu_mask, _otsu_mask  # _otsu_mask deprecated
from .morphology import remove_small_objects, binary_closing, split_touching_components
# Deprecated aliases — kept for backward compatibility, will be removed in v1.0
from .morphology import _remove_small_objects, _split_touching_components  # deprecated

# High-level segmenter
from .segmenter import WSISegmenter, segment_mask, make_lowres_mask

__all__ = [
    # Type
    "Backend",
    # Core
    "to_gray",
    "create_thumbnail",
    "upsample_mask",
    # Algorithms
    "entropy_mask",
    "otsu_mask",
    "remove_small_objects",
    "binary_closing",
    "split_touching_components",
    # High-level
    "WSISegmenter",
    "segment_mask",
    "make_lowres_mask",
]
