"""
Streaming writers for OME-Zarr files.

Provides memory-efficient streaming writers for large tiles that don't
fit in RAM, using block-by-block processing.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import zarr
from ngff_zarr import to_multiscales, to_ngff_image, to_ngff_zarr

# Handle zarr v2/v3 compatibility for NestedDirectoryStore
try:
    from zarr.storage import NestedDirectoryStore
except ImportError:
    # zarr v3 - use DirectoryStore or LocalStore instead
    try:
        from zarr.storage import DirectoryStore as NestedDirectoryStore
    except ImportError:
        from zarr.storage import LocalStore as NestedDirectoryStore


def write_ngff_from_tile_ts(
    tile_yxc: Union[np.ndarray, da.Array],
    out_path: Union[str, Path],
    base_px_um_xy: Tuple[float, float],
    *,
    chunks_xy: int = 512,
    num_mips: int = 8,
    name: str = "image",
    version: str = "0.4",
    channel_labels: Optional[List[str]] = None,
    channel_colors: Optional[List[str]] = None,
) -> None:
    """
    Stream a large (Y,X,C) tile to an OME-Zarr multiscale using ngff-zarr
    with the TensorStore backend (no full in-RAM pyramid).
    
    Parameters
    ----------
    tile_yxc : np.ndarray or da.Array
        Input tile (Y, X, C).
    out_path : str or Path
        Output path for OME-Zarr.
    base_px_um_xy : tuple
        Physical pixel size (px_um, py_um) in micrometers.
    chunks_xy : int
        Chunk size in X and Y dimensions.
    num_mips : int
        Number of MIP levels to generate.
    name : str
        Name for the image in metadata.
    version : str
        OME-NGFF version (default "0.4").
    channel_labels : list of str, optional
        Labels for each channel.
    channel_colors : list of str, optional
        Colors for each channel (hex strings).
    """
    out_path = str(Path(out_path))
    Y, X, C = map(int, tile_yxc.shape)
    px_um, py_um = base_px_um_xy

    # Ensure Dask with reasonable chunking
    if not isinstance(tile_yxc, da.Array):
        tile_yxc = da.from_array(tile_yxc, chunks=(chunks_xy, chunks_xy, C))

    # Base image (dims y,x,c with micrometer units)
    img = to_ngff_image(
        data=tile_yxc,
        dims=("y", "x", "c"),
        scale={"y": float(py_um), "x": float(px_um), "c": 1.0},
        name=name,
        axes_units={"y": "micrometer", "x": "micrometer"},
    )

    # Ask ngff-zarr to build a pyramid lazily; 2x downsampling per level
    levels = [2] * (max(1, num_mips) - 1)
    ms = to_multiscales(
        img,
        scale_factors=levels,
        chunks={"y": chunks_xy, "x": chunks_xy, "c": C},
    )

    # Write via TensorStore for out-of-core, chunked IO
    to_ngff_zarr(
        store=out_path,
        multiscales=ms,
        version=version,
        overwrite=True,
        use_tensorstore=True,
    )

    # Optional OMERO channels block (napari etc.)
    labels = channel_labels or [f"ch{i}" for i in range(C)]
    colors = channel_colors or ["FFFFFF"] * C
    root = zarr.open_group(out_path, mode="r+")
    attrs = dict(root.attrs)
    attrs["omero"] = {
        "name": name,
        "version": version,
        "rdefs": {"model": "color", "defaultZ": 0, "defaultT": 0},
        "channels": [
            {
                "label": labels[i],
                "color": colors[i],
                "window": {"start": 0.0, "end": 255.0, "min": 0.0, "max": 255.0},
                "active": True,
                "inverted": False,
                "coefficient": 1.0,
                "family": "linear",
            }
            for i in range(C)
        ],
    }
    root.attrs.put(attrs)


def write_ngff_from_tile_streaming_ome(
    tile_yxc_da: da.Array,
    out_dir: Union[Path, str],
    phys_xy_um: Tuple[float, float],
    *,
    block_xy: int = 512,
    num_mips: int,
    name: str = "image",
    compressor=None,
    dtype: str = "uint8",
    channel_labels: Optional[List[str]] = None,
    channel_colors: Optional[List[str]] = None,
) -> None:
    """
    Constant-memory, blockwise OME-Zarr writer (v0.4), no ngff-zarr.
    
    Streams each mip directly; attaches multiscales + OMERO metadata.
    
    Parameters
    ----------
    tile_yxc_da : da.Array
        Lazy Dask array (Y, X, C).
    out_dir : Path or str
        Output directory for OME-Zarr.
    phys_xy_um : tuple
        Physical pixel size (px_um, py_um) in micrometers.
    block_xy : int
        Block size for streaming.
    num_mips : int
        Number of MIP levels to generate.
    name : str
        Name for the image in metadata.
    compressor : optional
        Zarr compressor (e.g., zarr.Blosc).
    dtype : str
        Output dtype.
    channel_labels : list of str, optional
        Labels for each channel.
    channel_colors : list of str, optional
        Colors for each channel (hex strings).
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use NestedDirectoryStore to avoid huge flat dirs
    store = NestedDirectoryStore(str(out_dir))
    root = zarr.group(store=store, overwrite=True)

    # Shapes / scales
    Y, X, C = map(int, tile_yxc_da.shape)
    px_um, py_um = phys_xy_um
    shapes = [(max(1, Y >> m), max(1, X >> m), C) for m in range(num_mips)]
    chunks = [(min(block_xy, s0), min(block_xy, s1), C) for (s0, s1, _) in shapes]

    # Create per-scale arrays (Y,X,C) with the same chunking
    arrays = []
    for m, (sy, sx, sc) in enumerate(shapes):
        g = root.create_group(f"s{m}")
        arr = g.create_dataset(
            "0",
            shape=(sy, sx, sc),
            chunks=chunks[m],
            dtype=dtype,
            compressor=compressor,
            overwrite=True,
        )
        arrays.append(arr)

    # Stream blocks from the lazy Dask source and write all mips
    for y0 in range(0, Y, block_xy):
        y1 = min(Y, y0 + block_xy)
        for x0 in range(0, X, block_xy):
            x1 = min(X, x0 + block_xy)
            # Read a small block lazily -> NumPy
            blk = tile_yxc_da[y0:y1, x0:x1, :].astype(dtype).compute()

            # Write mip 0
            arrays[0][y0:y1, x0:x1, :] = blk

            # Downsample and write higher mips by stride slicing
            src = blk
            for m in range(1, num_mips):
                # Integer decimation by 2 per level
                src = src[::2, ::2, :]
                if src.size == 0:
                    break
                ym0 = y0 >> m
                xm0 = x0 >> m
                ym1 = ym0 + src.shape[0]
                xm1 = xm0 + src.shape[1]
                arrays[m][ym0:ym1, xm0:xm1, :] = src

    # Attach OME-NGFF v0.4 metadata (axes + datasets with scale transforms)
    datasets = []
    for m, (sy, sx, _sc) in enumerate(shapes):
        scale = [1.0, py_um * (2**m), px_um * (2**m)]
        datasets.append({
            "path": f"s{m}",
            "coordinateTransformations": [{"type": "scale", "scale": scale}],
        })

    root.attrs.update({
        "multiscales": [{
            "name": name,
            "version": "0.4",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets,
        }]
    })

    # Optional OMERO display metadata (napari will use it)
    labels = channel_labels or [f"ch{i}" for i in range(C)]
    colors = channel_colors or ["FFFFFF"] * C
    root.attrs["omero"] = {
        "name": name,
        "version": "0.4",
        "rdefs": {"model": "color", "defaultZ": 0, "defaultT": 0},
        "channels": [
            {
                "label": labels[i],
                "color": colors[i],
                "window": {"start": 0.0, "end": 255.0, "min": 0.0, "max": 255.0},
                "active": True,
                "inverted": False,
                "coefficient": 1.0,
                "family": "linear",
            }
            for i in range(C)
        ],
    }
