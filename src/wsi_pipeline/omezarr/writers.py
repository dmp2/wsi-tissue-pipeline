"""
NGFF / OME-Zarr Writers

Write multiscale OME-Zarr files from mip pyramids with proper metadata.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Literal

import dask.array as da
import numpy as np
import zarr
from ngff_zarr import Multiscales, to_ngff_image, to_ngff_zarr
from ngff_zarr.v04.zarr_metadata import Axis, Dataset, Scale
from ngff_zarr.v04.zarr_metadata import Metadata as V04Metadata
from numcodecs import Blosc

from .metadata import _prepare_ngff_writer_metadata


def _omero_version(root_attrs: dict[str, Any], schema: str) -> str:
    """Choose an OMERO version string consistent with the emitted root attrs."""
    try:
        version = root_attrs["multiscales"][0].get("version")
    except (KeyError, IndexError, TypeError):
        version = None
    if version is not None:
        return str(version)
    return "latest" if schema == "latest" else "0.4"


def _build_omero_block(
    *,
    name: str,
    version: str,
    channel_labels: list[str],
    channel_colors: list[str] | None,
) -> dict[str, Any]:
    """Build the standard OMERO display block used by the writers."""
    colors = channel_colors or ["FFFFFF"] * len(channel_labels)
    if len(colors) != len(channel_labels):
        raise ValueError(
            f"Writer received {len(colors)} channel colors for {len(channel_labels)} channels."
        )
    return {
        "name": name,
        "version": version,
        "rdefs": {"model": "color", "defaultZ": 0, "defaultT": 0},
        "channels": [
            {
                "label": channel_labels[idx],
                "color": colors[idx],
                "window": {"start": 0.0, "end": 255.0, "min": 0.0, "max": 255.0},
                "active": True,
                "inverted": False,
                "coefficient": 1.0,
                "family": "linear",
            }
            for idx in range(len(channel_labels))
        ],
    }


def write_ngff_from_mips(
    mips_yxc: list[np.ndarray],
    out_dir: os.PathLike,
    phys_xy_um: tuple[float, float] | None = None,
    *,
    name: str = "tissue",
    chunks_xy: int = 512,
    compressor: Blosc | None = None,
    dtype: np.dtype | None = None,
    channel_labels: list[str] | None = None,
    channel_colors: list[str] | None = None,
    ngff_metadata: dict[str, Any] | None = None,
    metadata_schema: Literal["latest", "v0.4", "0.4"] | None = None,
) -> None:
    """
    Write multiscale NGFF at out_dir with datasets s0..sN (array layout (C,Y,X)).

    Uses manual Zarr array creation with OME-NGFF v0.4 metadata.
    Multiscales v0.4 with axes [c,y,x] and per-level 'scale' transform (um).

    Parameters
    ----------
    mips_yxc : List[np.ndarray]
        List of mip levels, each (H, W, C).
    out_dir : PathLike
        Output directory for the OME-Zarr.
    phys_xy_um : Tuple[float, float], optional
        Physical pixel size (x_um, y_um) at base level.
    name : str
        Dataset name for metadata.
    chunks_xy : int
        Chunk size for spatial dimensions.
    compressor : Blosc, optional
        Compression settings. Defaults to zstd.
    dtype : np.dtype, optional
        Output dtype. Defaults to input dtype.
    channel_labels : list[str], optional
        Labels for each channel in the OMERO metadata.
    channel_colors : list[str], optional
        Hex colors for each channel in the OMERO metadata.
    ngff_metadata : dict, optional
        Full ``get_vsi_metadata()`` payload or a direct NGFF root-attrs payload.
    metadata_schema : {"latest", "v0.4", "0.4"}, optional
        Schema alias used when ``ngff_metadata`` provides dual projections.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if compressor is None:
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

    if not mips_yxc:
        raise ValueError("mips_yxc must contain at least one level.")

    if dtype is None:
        dtype = mips_yxc[0].dtype

    # Get channel dim
    C = mips_yxc[0].shape[-1]
    prepared = _prepare_ngff_writer_metadata(
        dataset_count=len(mips_yxc),
        channel_count=C,
        name=name,
        fallback_phys_xy_um=phys_xy_um,
        ngff_metadata=ngff_metadata,
        metadata_schema=metadata_schema,
        channel_labels=channel_labels,
        channel_axis_name="c^",
    )
    root_attrs = prepared["root_attrs"]
    resolved_name = prepared["resolved_name"]
    resolved_channel_labels = prepared["resolved_channel_labels"]
    omero_version = _omero_version(root_attrs, prepared["schema"])

    # Root markers
    (out_dir / ".zgroup").write_text(json.dumps({"zarr_format": 2}))

    for lvl, img in enumerate(mips_yxc):
        if img.ndim != 3:
            raise ValueError(f"mips[{lvl}] must be (H,W,C), got shape {img.shape}")
        if img.shape[-1] != C:
            raise ValueError(
                f"mips[{lvl}] has {img.shape[-1]} channels, expected {C}."
            )

        # Preallocate the current image shape
        H, W, _ = img.shape

        # Create the output directory for the current level
        g = out_dir / f"s{lvl}"
        g.mkdir(parents=True, exist_ok=True)

        # Mark each scale as a Zarr group
        (g / ".zgroup").write_text(json.dumps({"zarr_format": 2}))

        # Create the array with the channel first (c,y,x)
        arr = zarr.open_array(
            store=str(g),
            path="",
            mode="w",
            zarr_format=2,
            shape=(C, H, W),
            chunks=(C, min(chunks_xy, H), min(chunks_xy, W)),
            dtype=dtype,
            compressor=compressor
        )
        # Write the data channel-first
        arr[...] = np.moveaxis(img, -1, 0)  # (h, w, c) -> (c, h, w)

        # Mark per-array axis names; use c^ so NG treats it as vector (RGB)
        (g / ".zattrs").write_text(json.dumps({"_ARRAY_DIMENSIONS": ["c^", "y", "x"]}))

    root_attrs["omero"] = _build_omero_block(
        name=resolved_name,
        version=omero_version,
        channel_labels=resolved_channel_labels,
        channel_colors=channel_colors,
    )
    (out_dir / ".zattrs").write_text(json.dumps(root_attrs, indent=2))


def write_ngff_from_mips_ngffzarr(
    mips_yxc: list[np.ndarray],
    out_dir: Path,
    phys_xy_um: tuple[float, float] | None = None,
    *,
    name: str = "image",
    chunks_xy: int = 512,
    version: str = "0.4",
    overwrite: bool = True,
    channel_labels: list[str] | None = None,
    channel_colors: list[str] | None = None,
    add_omero: bool = True,
    ngff_metadata: dict[str, Any] | None = None,
    metadata_schema: Literal["latest", "v0.4", "0.4"] | None = None,
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
    phys_xy_um : Tuple[float, float], optional
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
    ngff_metadata : dict, optional
        Full ``get_vsi_metadata()`` payload or a direct NGFF root-attrs payload.
    metadata_schema : {"latest", "v0.4", "0.4"}, optional
        Schema alias used when ``ngff_metadata`` provides dual projections.
    """
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    if not mips_yxc:
        raise ValueError("mips_yxc must contain at least one level.")

    channel_count = int(mips_yxc[0].shape[-1])
    prepared = _prepare_ngff_writer_metadata(
        dataset_count=len(mips_yxc),
        channel_count=channel_count,
        name=name,
        fallback_phys_xy_um=phys_xy_um,
        ngff_metadata=ngff_metadata,
        metadata_schema=metadata_schema,
        channel_labels=channel_labels,
    )
    root_attrs = prepared["root_attrs"]
    resolved_name = prepared["resolved_name"]
    resolved_channel_labels = prepared["resolved_channel_labels"]
    resolved_phys_xy_um = prepared["resolved_phys_xy_um"] or (1.0, 1.0)
    omero_version = _omero_version(root_attrs, prepared["schema"])
    px_um, py_um = resolved_phys_xy_um

    # Ensure each level is (C,Y,X) and dask-chunked
    cyx_levels = []
    for lvl, yxc in enumerate(mips_yxc):
        if yxc.ndim != 3:
            raise ValueError(f"mips[{lvl}] must be (Y,X,C), got shape {yxc.shape}")
        if yxc.shape[-1] != channel_count:
            raise ValueError(
                f"mips[{lvl}] has {yxc.shape[-1]} channels, expected {channel_count}."
            )
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
            name=resolved_name,
            axes_units={"y": "micrometer", "x": "micrometer"}
        )
        images.append(img)

    # Create Multiscales from precomputed levels
    metadata = V04Metadata(
        axes=[
            Axis(name="c", type="channel"),
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ],
        datasets=[
            Dataset(
                path=f"s{lvl}",
                coordinateTransformations=[
                    Scale(scale=[1.0, py_um * (2 ** lvl), px_um * (2 ** lvl)])
                ],
            )
            for lvl in range(len(images))
        ],
        coordinateTransformations=None,
        name=resolved_name,
        version="0.4",
    )
    ms = Multiscales(images=images, metadata=metadata)
    ms.chunks = (cyx_levels[0].shape[0], chunks_xy, chunks_xy)

    # Write with TensorStore backend
    to_ngff_zarr(
        store=str(out_dir),
        multiscales=ms,
        version=version,
        overwrite=True,
        use_tensorstore=True
    )

    root = zarr.open_group(str(out_dir), mode="r+")
    attrs = dict(root_attrs)
    existing_omero = dict(root.attrs).get("omero")

    if add_omero:
        labels = resolved_channel_labels
        colors = channel_colors or ["FFFFFF"] * channel_count
        attrs["omero"] = _build_omero_block(
            name=resolved_name,
            version=omero_version,
            channel_labels=labels,
            channel_colors=colors,
        )
    elif existing_omero is not None:
        attrs["omero"] = existing_omero

    root.attrs.put(attrs)
