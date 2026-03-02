"""
Pipeline Module -- High-Level Orchestration

Coordinates the full tissue extraction workflow: segmentation -> tile extraction
-> OME-Zarr writing -> (optional) Neuroglancer precomputed writing. The term
"plating" refers to the histology practice of laying tissue sections out in a
grid arrangement for downstream 3D registration.

Public API
----------
process_slide_with_plating(slide_path, output_dir, config, ...)
    End-to-end processing for a single WSI slide: segment, extract tiles,
    write OME-Zarr pyramids, and optionally write Neuroglancer precomputed data.
"""

from __future__ import annotations

from .plating import (
    _is_big_tile,
    _safe_close_existing_client,
    process_slide_with_plating,
)

__all__ = [
    "process_slide_with_plating",
    # _is_big_tile and _safe_close_existing_client are internal helpers;
    # imported above for use within the package but not part of the public API.
]
