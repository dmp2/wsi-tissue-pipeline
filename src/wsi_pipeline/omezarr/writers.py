"""
NGFF / OME-Zarr Writers

Write multiscale OME-Zarr files from mip pyramids with proper metadata.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import dask.array as da
import numpy as np
import zarr
from ngff_zarr import Multiscales, to_ngff_image, to_ngff_zarr
from numcodecs import Blosc


def write_ngff_from_mips(
    mips_yxc: List[np.ndarray],
    out_dir: os.PathLike,
    phys_xy_um: Tuple[float, float],
    name: str = "tissue",
    chunks_xy: int = 512,
    compressor: Optional[Blosc] = None,
    dtype: Optional[np.dtype] = None,
) -> None:
    """
    Write multiscale NGFF at out_dir with datasets s0..sN (array layout (C,Y,X)).
    
    Uses manual Zarr array creation with OME-NGFF v0.4 metadata.
    Multiscales v0.4 with axes [c,y,x] and per-level 'scale' transform (µm).
    
    Parameters
    ----------
    mips_yxc : List[np.ndarray]
        List of mip levels, each (H, W, C).
    out_dir : PathLike
        Output directory for the OME-Zarr.
    phys_xy_um : Tuple[float, float]
        Physical pixel size (x_um, y_um) at base level.
    name : str
        Dataset name for metadata.
    chunks_xy : int
        Chunk size for spatial dimensions.
    compressor : Blosc, optional
        Compression settings. Defaults to zstd.
    dtype : np.dtype, optional
        Output dtype. Defaults to input dtype.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if compressor is None:
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

    if dtype is None:
        dtype = mips_yxc[0].dtype

    # Get channel dim
    C = mips_yxc[0].shape[-1]

    # Root markers
    (out_dir / ".zgroup").write_text(json.dumps({"zarr_format": 2}))

    datasets = []
    for lvl, img in enumerate(mips_yxc):
        if img.ndim != 3:
            raise ValueError(f"mips[{lvl}] must be (H,W,C), got shape {img.shape}")
        
        # Preallocate the current image shape
        H, W, _ = img.shape

        # Create the output directory for the current level
        g = out_dir / f"s{lvl}"
        g.mkdir(parents=True, exist_ok=True)

        # Mark each scale as a Zarr group
        (g / ".zgroup").write_text(json.dumps({"zarr_format": 2}))

        # Create the array with the channel first (c,y,x)
        arr = zarr.open_array(
            store=zarr.DirectoryStore(g),
            path="",
            mode="w",
            shape=(C, H, W),
            chunks=(C, min(chunks_xy, H), min(chunks_xy, W)),
            dtype=dtype,
            compressor=compressor
        )
        # Write the data channel-first
        arr[...] = np.moveaxis(img, -1, 0)  # (h, w, c) -> (c, h, w)

        # Mark per-array axis names; use c^ so NG treats it as vector (RGB)
        (g / ".zattrs").write_text(json.dumps({"_ARRAY_DIMENSIONS": ["c^", "y", "x"]}))

        # multiscales dataset entry (scale doubles per level)
        scale = [1.0, phys_xy_um[1] * (2 ** lvl), phys_xy_um[0] * (2 ** lvl)]
        datasets.append({
            "path": f"s{lvl}", 
            "coordinateTransformations": [{"type": "scale", "scale": scale}]
        })

    root_attrs = {
        "multiscales": [{
            "name": name,
            "version": "0.4",
            "axes": [
                {"name": "c^", "type": "channel"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets,
        }],
        "omero": {
            "name": name,
            "version": "0.4",
            "rdefs": {"model": "color", "defaultZ": 0, "defaultT": 0},
            "channels": [
                {
                    "label": f"ch{c}",
                    "color": "FFFFFF",
                    "window": {"start": 0.0, "end": 255.0, "min": 0.0, "max": 255.0},
                    "active": True,
                    "inverted": False,
                    "coefficient": 1.0,
                    "family": "linear",
                } for c in range(C)
            ],
        },
    }
    (out_dir / ".zattrs").write_text(json.dumps(root_attrs, indent=2))


def write_ngff_from_mips_ngffzarr(
    mips_yxc: List[np.ndarray],
    out_dir: Path,
    phys_xy_um: Tuple[float, float],
    *,
    name: str = "image",
    chunks_xy: int = 512,
    version: str = "0.4",
    overwrite: bool = True,
    channel_labels: Optional[List[str]] = None,
    channel_colors: Optional[List[str]] = None,
    add_omero: bool = True
) -> None:
    """
    Write a multiscale OME-Zarr (v0.4 metadata) with TensorStore backend.
    
    Uses ngff-zarr library for standardized metadata and TensorStore
    for efficient chunked writes.
    
    Parameters
    ----------
    mips_yxc : List[np.ndarray]
        List of mip levels, each (Y, X, C).
    out_dir : Path
        Output directory for the OME-Zarr.
    phys_xy_um : Tuple[float, float]
        Physical pixel size (x_um, y_um) at base level.
    name : str
        Dataset name for metadata.
    chunks_xy : int
        Chunk size for spatial dimensions.
    version : str
        OME-NGFF metadata version (0.4 is widely supported).
    overwrite : bool
        Whether to overwrite existing output.
    channel_labels : List[str], optional
        Labels for each channel.
    channel_colors : List[str], optional
        Hex colors for each channel (e.g., ["FFFFFF", "FFFFFF", "FFFFFF"]).
    add_omero : bool
        Whether to add OMERO display metadata.
    """
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    px_um, py_um = phys_xy_um

    # Ensure each level is (C,Y,X) and dask-chunked
    cyx_levels = []
    for lvl, yxc in enumerate(mips_yxc):
        if hasattr(yxc, "chunks"):  # dask array
            arr_yxc = yxc
        else:  # numpy -> dask
            arr_yxc = da.from_array(yxc, chunks=(chunks_xy, chunks_xy, yxc.shape[2]))
        cyx = da.moveaxis(arr_yxc, -1, 0)  # (Y,X,C)->(C,Y,X)
        cyx = cyx.rechunk((cyx.shape[0], chunks_xy, chunks_xy))
        cyx_levels.append(cyx)

    # Build NgffImage objects with correct dims/units and per-level scale
    images = []
    for lvl, cyx in enumerate(cyx_levels):
        scale_map = {"y": py_um * (2 ** lvl), "x": px_um * (2 ** lvl)}
        img = to_ngff_image(
            data=cyx,
            dims=("c", "y", "x"),
            scale=scale_map,
            name=name,
            axes_units={"y": "micrometer", "x": "micrometer"}
        )
        images.append(img)

    # Create Multiscales from precomputed levels
    ms = Multiscales(images=images)
    ms.chunks = (cyx_levels[0].shape[0], chunks_xy, chunks_xy)

    # Write with TensorStore backend
    to_ngff_zarr(
        store=str(out_dir),
        multiscales=ms,
        version=version,
        overwrite=True,
        use_tensorstore=True
    )

    # Add OMERO channel metadata
    if add_omero:
        C = int(cyx_levels[0].shape[0])
        labels = channel_labels or [f"ch{c}" for c in range(C)]
        colors = channel_colors or ["FFFFFF"] * C

        root = zarr.open_group(str(out_dir), mode="r+")
        attrs = dict(root.attrs)

        omero = {
            "name": name,
            "version": version,
            "rdefs": {"model": "color", "defaultZ": 0, "defaultT": 0},
            "channels": [
                {
                    "label": labels[c],
                    "color": colors[c],
                    "window": {"start": 0.0, "end": 255.0, "min": 0.0, "max": 255.0},
                    "active": True,
                    "inverted": False,
                    "coefficient": 1.0,
                    "family": "linear",
                } for c in range(C)
            ],
        }
        attrs["omero"] = omero
        root.attrs.put(attrs)
