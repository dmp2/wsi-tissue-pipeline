"""
Tile Generation and I/O

Extracts individual tissue sections as square Dask tiles from whole-slide images,
saves them to disk, and manages the hierarchical naming convention used downstream
by the QC grid builder and EM-LDDMM registration pipeline.

Naming convention: ``{PREFIX}_{SLIDE:02d}_{SLICE:02d}_{OVERALL:04d}.{ext}``
- SLIDE:   physical slide number (1-indexed)
- SLICE:   tissue section position on that slide (left-to-right order)
- OVERALL: global section index across all slides (used for Z-registration)

Public API
----------
generate_tissue_tiles   Build list of Dask tile arrays from a segmentation mask
generate_tissue_images  Legacy function (saves tiles to disk directly)
save_tile               Save a single numpy tile to disk
to_uint8                Normalize an array to uint8 range
build_xy_regex_from_pattern   Compile regex from naming pattern string
parse_xx_yy_from_name   Extract slide/slice indices from a filename
overall_label / add_overall_suffix  Utilities for overall-index naming
"""

from __future__ import annotations

# Generator functions
from .generator import (
    center_crop_pad_dask,
    crop_and_pad,
    generate_tissue_images,
    generate_tissue_tiles,
    sort_labels_left_to_right,
)

# I/O functions
from .io import (
    save_tile,
    to_uint8,
)

# Naming utilities
from .naming import (
    add_overall_suffix,
    build_output_glob_from_pattern,
    build_xy_regex_from_pattern,
    normalize_ext,
    overall_label,
    parse_xx_yy_from_name,
    rename_outputs_by_overall_index,
)

__all__ = [
    # Generator
    "generate_tissue_tiles",
    "center_crop_pad_dask",
    "crop_and_pad",
    "generate_tissue_images",
    "sort_labels_left_to_right",
    # I/O
    "to_uint8",
    "save_tile",
    # Naming
    "normalize_ext",
    "build_xy_regex_from_pattern",
    "build_output_glob_from_pattern",
    "parse_xx_yy_from_name",
    "overall_label",
    "add_overall_suffix",
    "rename_outputs_by_overall_index",
]
