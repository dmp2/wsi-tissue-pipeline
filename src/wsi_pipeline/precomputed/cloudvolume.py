"""
CloudVolume backend for Neuroglancer precomputed format.

Provides functions for creating and writing to precomputed datasets
using the CloudVolume library.
"""

from __future__ import annotations

import numpy as np

from ..omezarr.metadata import _sizes_for_mips_xy, _voxel_sizes_for_mips_xy


def create_precomputed_cloudvolume(
    precomp_path: str,
    W: int,
    H: int,
    Z: int,
    voxel_size_um: tuple[float, float, float],
    num_mips: int,
    chunk_xy: int = 512,
    dtype: str = "uint8",
    encoding: str = "raw",
    fill_missing: bool = False,
    parallel: bool = False,
):
    """
    Create a precomputed dataset using CloudVolume.

    Parameters
    ----------
    precomp_path : str
        Path to the precomputed dataset (e.g., "file:///path/to/data").
    W : int
        Width in pixels at full resolution.
    H : int
        Height in pixels at full resolution.
    Z : int
        Number of Z slices.
    voxel_size_um : Tuple[float, float, float]
        Voxel size in micrometers (x, y, z).
    num_mips : int
        Number of MIP levels to create.
    chunk_xy : int
        Chunk size in XY dimensions.
    dtype : str
        Data type for the volume.
    encoding : str
        Encoding format ("raw", "jpeg", "compressed_segmentation").
    fill_missing : bool
        Whether to handle missing chunks gracefully.
    parallel : bool
        Whether to enable parallel writes.

    Returns
    -------
    List[CloudVolume]
        List of CloudVolume writers for each MIP level.
    """
    from cloudvolume import CloudVolume
    from cloudvolume.lib import mkdir

    mkdir(precomp_path)

    # Build the scale list explicitly
    scales = []

    # Microns -> nanometers (as ints) as expected by neuroglancer
    voxel_size_nm_int = [
        int(round(voxel_size_um[0] * 1000.0)),
        int(round(voxel_size_um[1] * 1000.0)),
        int(round(voxel_size_um[2] * 1000.0)),
    ]

    # Build XY sizes for each mip, then expand to XYZ
    sizes_xy = _sizes_for_mips_xy(W, H, num_mips)
    sizes_xyz = [[int(w), int(h), int(Z)] for (w, h) in sizes_xy]

    # Build per-mip voxel sizes (nm), already XYZ triples of ints
    voxel_sizes_nm = _voxel_sizes_for_mips_xy(voxel_size_nm_int, num_mips)

    for lvl in range(int(num_mips)):
        scales.append({
            "chunk_sizes": [[chunk_xy, chunk_xy, 1]],
            "encoding": encoding,
            "key": str(lvl),
            "resolution": voxel_sizes_nm[lvl],
            "size": sizes_xyz[lvl],
            "voxel_offset": [0, 0, 0]
        })

    # Create info structure
    info = CloudVolume.create_new_info(
        num_channels=3,
        layer_type="image",
        data_type=dtype,
        encoding=encoding,
        resolution=voxel_size_nm_int,
        voxel_offset=[0, 0, 0],
        chunk_size=[chunk_xy, chunk_xy, 1],
        volume_size=[W, H, Z]
    )

    # Overwrite scales and commit
    info["scales"] = scales

    vol = CloudVolume(
        precomp_path,
        info=info,
        cache=False,
        compress=False,
        parallel=parallel,
        fill_missing=fill_missing
    )
    vol.commit_info()

    # Sanity check: reload and assert scales
    vol2 = CloudVolume(precomp_path, parallel=parallel)
    have = len(vol2.scales)
    need = int(num_mips)
    if have != need:
        raise RuntimeError(
            f"Committed {have} scales but expected {need}. "
            f"Resolutions: {[s['resolution'] for s in vol2.scales]}"
        )

    # Return writers for each mip level
    return [
        CloudVolume(precomp_path, mip=i, parallel=parallel, fill_missing=fill_missing)
        for i in range(num_mips)
    ]


def write_slice_cloudvolume(
    writers,
    z_index: int,
    mips_yxc: list[np.ndarray],
    flush_every: int = 10,
) -> None:
    """
    Write one Z slice (all mips) as XY*C slabs into CloudVolume writers.

    Parameters
    ----------
    writers : List[CloudVolume]
        List of CloudVolume writers for each MIP level.
    z_index : int
        Z index to write to.
    mips_yxc : List[np.ndarray]
        List of MIP arrays, each (H, W, C).
    flush_every : int
        Flush cache every N slices for performance.
    """
    assert len(writers) == len(mips_yxc)

    for _mip, (vol_i, img_yxc) in enumerate(zip(writers, mips_yxc, strict=False)):
        # CloudVolume expects (X, Y, Z, C); input is (H, W, C) = (Y, X, C)
        img_xyc = np.transpose(img_yxc, (1, 0, 2))  # (Y, X, C) -> (X, Y, C)

        # Add Z dimension: (X, Y, C) -> (X, Y, 1, C)
        img_xyzc = img_xyc[:, :, np.newaxis, :]
        vol_i[:, :, z_index:z_index + 1, :] = img_xyzc

        # Force flush if available
        if hasattr(vol_i, 'cache'):
            if hasattr(vol_i.cache, 'flush'):
                vol_i.cache.flush()
            elif hasattr(vol_i.cache, 'flush_cache'):
                vol_i.cache.flush_cache()
