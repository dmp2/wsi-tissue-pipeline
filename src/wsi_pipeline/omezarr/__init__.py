"""
OME-Zarr / OME-NGFF Module

Writes OME-NGFF (Next Generation File Format) — the open, cloud-friendly
multi-resolution image format used throughout the WSI pipeline. Output files
have the ``.ome.zarr`` extension and can be viewed in Neuroglancer, Napari,
or any OME-Zarr-compatible tool.

Public API
----------
build_mips_from_yxc(base_yxc, num_mips)
    Build a power-of-two resolution pyramid from a (Y, X, C) image.
compute_num_mips_min_side(width, height, min_side_for_mips)
    Compute how many MIP levels fit before the image shrinks below min_side.
write_ngff_from_mips(mips_yxc, out_dir, phys_xy_um, ...)
    Write OME-Zarr using direct Zarr array creation (OME-NGFF v0.4).
write_ngff_from_mips_ngffzarr(mips_yxc, out_dir, phys_xy_um, ...)
    Write OME-Zarr using the ngff-zarr library with TensorStore backend.
write_ngff_from_tile_ts(tile_yxc, out_path, base_px_um_xy, ...)
    Streaming writer for large tiles — no full in-RAM pyramid required.
write_ngff_from_tile_streaming_ome(tile_yxc_da, out_dir, phys_xy_um, ...)
    Constant-memory block-wise OME-Zarr writer (alternative to TensorStore).
"""

from __future__ import annotations

# Metadata utilities (internal — used by writers and pipeline)
from .metadata import (
    _detect_source_ds_factor,
    _get_multiscales_paths,
    _is_ngff_image_group,
    _phys_xy_um,
    _safe_name,
    _sizes_for_mips_xy,
    _voxel_sizes_for_mips_xy,
)

# Pyramid building
from .pyramid import (
    build_mips_from_yxc,
    compute_num_mips_min_side,
)

# Streaming writers (for large images)
from .streaming import (
    write_ngff_from_tile_streaming_ome,
    write_ngff_from_tile_ts,
)

# Standard writers
from .writers import (
    write_ngff_from_mips,
    write_ngff_from_mips_ngffzarr,
)

__all__ = [
    # Pyramid
    "compute_num_mips_min_side",
    "build_mips_from_yxc",
    # Streaming writers
    "write_ngff_from_tile_ts",
    "write_ngff_from_tile_streaming_ome",
    # Standard writers
    "write_ngff_from_mips",
    "write_ngff_from_mips_ngffzarr",
]
