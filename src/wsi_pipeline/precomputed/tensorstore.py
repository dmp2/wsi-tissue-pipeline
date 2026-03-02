"""
TensorStore backend for Neuroglancer precomputed format.

Provides functions for creating and writing to precomputed datasets
using the TensorStore library for efficient synchronous I/O.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import dask.array as da
import numpy as np
import tensorstore as ts

from ..omezarr.metadata import _sizes_for_mips_xy, _voxel_sizes_for_mips_xy

logger = logging.getLogger(__name__)


def create_precomputed_tensorstore(
    precomp_path: str,
    W: int,
    H: int,
    Z: int,
    voxel_size_um: tuple[float, float, float],
    num_mips: int,
    chunk_xy: int = 512,
    dtype: str = "uint8",
    encoding: str = "raw",
) -> list[ts.TensorStore]:
    """
    Create TensorStore neuroglancer_precomputed dataset with multiple scales.

    Parameters
    ----------
    precomp_path : str
        Path to the precomputed dataset (may include "file://" prefix).
    W : int
        Width in pixels at full resolution.
    H : int
        Height in pixels at full resolution.
    Z : int
        Number of Z slices (tissue sections).
    voxel_size_um : Tuple[float, float, float]
        Voxel size in micrometers (x, y, z).
    num_mips : int
        Number of MIP levels to create.
    chunk_xy : int
        Chunk size in XY dimensions.
    dtype : str
        Data type for the volume.
    encoding : str
        Encoding format ("raw", "jpeg", etc.).

    Returns
    -------
    List[ts.TensorStore]
        List of TensorStore objects for each MIP level.
    """
    # Convert path to absolute
    if precomp_path.startswith("file://"):
        precomp_path = precomp_path[7:]
    precomp_path = str(Path(precomp_path).absolute())

    # Convert um to nm
    voxel_size_nm = [
        int(round(voxel_size_um[0] * 1000)),
        int(round(voxel_size_um[1] * 1000)),
        int(round(voxel_size_um[2] * 1000))
    ]

    # Generate scales info
    writers = []

    sizes_xy = _sizes_for_mips_xy(W, H, num_mips)
    voxel_sizes_nm_list = _voxel_sizes_for_mips_xy(voxel_size_nm, num_mips)

    # Create info file structure
    info = {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": dtype,
        "num_channels": 3,
        "scales": [],
        "type": "image"
    }

    # Add scales
    for mip in range(num_mips):
        w, h = sizes_xy[mip]
        vx, vy, vz = voxel_sizes_nm_list[mip]

        scale_info = {
            "chunk_sizes": [[min(chunk_xy, w), min(chunk_xy, h), 1]],
            "encoding": encoding,
            "key": str(mip),
            "resolution": [vx, vy, vz],
            "size": [w, h, Z],
            "voxel_offset": [0, 0, 0]
        }
        info["scales"].append(scale_info)

    # Write info file
    info_path = Path(precomp_path) / "info"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info, indent=2))

    logger.info(
        "Created TensorStore info with %d mips | base resolution: %s nm | base size: %dx%dx%d | channels: 3 (RGB)",
        num_mips, voxel_size_nm, W, H, Z,
    )

    # Create TensorStore for each mip level
    for mip in range(num_mips):
        w, h = sizes_xy[mip]

        spec = {
            'driver': 'neuroglancer_precomputed',
            'kvstore': {
                'driver': 'file',
                'path': precomp_path,
            },
            'context': {
                'cache_pool': {'total_bytes_limit': 100_000_000},
                'data_copy_concurrency': {'limit': 4},
            },
            'recheck_cached_data': 'open',
            'scale_index': mip,
            'open': True,
        }

        try:
            dataset = ts.open(spec).result()
            writers.append(dataset)
            logger.debug("Created mip %d: %dx%dx%d", mip, w, h, Z)
        except Exception as e:
            logger.error("Error creating mip %d: %s", mip, e)
            raise

    return writers


def write_slice_tensorstore(
    writers: list[ts.TensorStore],
    z_index: int,
    mips_yxc: list[np.ndarray],
) -> None:
    """
    Write one Z slice for all mip levels using TensorStore.

    Data is written immediately (synchronous) - no caching issues.

    Parameters
    ----------
    writers : List[ts.TensorStore]
        List of TensorStore objects for each MIP level.
    z_index : int
        Z index to write to.
    mips_yxc : List[np.ndarray]
        List of MIP arrays, each (H, W, C).
    """
    for _mip, (dataset, img_yxc) in enumerate(zip(writers, mips_yxc, strict=False)):
        # Transform: (H, W, C) = (Y, X, C) -> (X, Y, C)
        img_xyc = np.transpose(img_yxc, (1, 0, 2))

        # Write synchronously
        dataset[:, :, z_index] = img_xyc


def write_slice_tensorstore_streaming(
    writers: list[ts.TensorStore],
    z_index: int,
    tile_yxc: np.ndarray | da.Array,
    block_xy: int = 512,
) -> None:
    """
    Write one Z slice with streaming (block-by-block) for large tiles.

    Writes the base tile and generates lower resolution MIPs on-the-fly
    using nearest-neighbor decimation.

    Parameters
    ----------
    writers : List[ts.TensorStore]
        List of TensorStore objects for each MIP level (shaped X, Y, Z, C).
    z_index : int
        Z index to write to.
    tile_yxc : np.ndarray or da.Array
        Input tile (Y, X, C), can be Dask or NumPy.
    block_xy : int
        Block size for streaming writes. Should match chunk_xy for best I/O.
    """
    # Work in (C, Y, X) so we can decimate by slicing
    if isinstance(tile_yxc, da.Array):
        tile_cyx = tile_yxc.transpose(2, 0, 1)  # lazy
        Y, X = tile_yxc.shape[0], tile_yxc.shape[1]
    else:
        tile_cyx = np.transpose(tile_yxc, (2, 0, 1))
        Y, X = tile_yxc.shape[0], tile_yxc.shape[1]

    for y0 in range(0, Y, block_xy):
        y1 = min(Y, y0 + block_xy)
        for x0 in range(0, X, block_xy):
            x1 = min(X, x0 + block_xy)

            # Pull full-res source block for this XY window
            src_cyx = tile_cyx[:, y0:y1, x0:x1]
            if isinstance(src_cyx, da.Array):
                src_cyx = src_cyx.compute()  # (C, bh, bw)

            # Write the corresponding block at each mip (nearest-neighbor decimation)
            for level, ds in enumerate(writers):
                step = 1 << level
                y0m, y1m = y0 // step, y1 // step
                x0m, x1m = x0 // step, x1 // step

                if y0m == y1m or x0m == x1m:
                    continue

                blk_yxc = np.transpose(src_cyx[:, ::step, ::step], (1, 2, 0))  # (bh', bw', C)
                slab = np.ascontiguousarray(blk_yxc)

                # Synchronous subscript write: TensorStore handles chunking
                ds[x0m:x1m, y0m:y1m, z_index] = slab
