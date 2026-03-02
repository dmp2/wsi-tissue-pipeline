"""
Neuroglancer Precomputed Format Writers

Writes the "Neuroglancer Precomputed" chunked format -- a tiled, multi-resolution
format natively supported by the Neuroglancer browser viewer. Use this module
when you want to serve tissue data directly to Neuroglancer without an
intermediate conversion step.

Backends: CloudVolume (mature, feature-rich) or TensorStore (fast, local).

Public API
----------
PlatePrecomputedWriter
    High-level writer: preallocate a volume, then append tissue slices one at
    a time along Z. Supports streaming for very large tiles.
create_precomputed_cloudvolume / write_slice_cloudvolume
    Low-level CloudVolume backend functions.
create_precomputed_tensorstore / write_slice_tensorstore / write_slice_tensorstore_streaming
    Low-level TensorStore backend functions.
"""

from __future__ import annotations

from .cloudvolume import (
    create_precomputed_cloudvolume,
    write_slice_cloudvolume,
)
from .plate_writer import PlatePrecomputedWriter
from .tensorstore import (
    create_precomputed_tensorstore,
    write_slice_tensorstore,
    write_slice_tensorstore_streaming,
)

__all__ = [
    # CloudVolume backend
    "create_precomputed_cloudvolume",
    "write_slice_cloudvolume",
    # TensorStore backend
    "create_precomputed_tensorstore",
    "write_slice_tensorstore",
    "write_slice_tensorstore_streaming",
    # High-level facade
    "PlatePrecomputedWriter",
]
