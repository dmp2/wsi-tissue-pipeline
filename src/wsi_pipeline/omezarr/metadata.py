"""
NGFF metadata utilities for reading and writing OME-Zarr metadata.

Provides functions for parsing and generating NGFF-compliant metadata,
including multiscales, physical pixel sizes, and coordinate transformations.
"""

from __future__ import annotations

import json
from pathlib import Path

import zarr


def _is_ngff_image_group(path: Path) -> bool:
    """Basic NGFF image check: requires .zgroup, .zattrs with 'multiscales'."""
    try:
        if not (path / ".zgroup").exists():
            return False
        attrs_p = path / ".zattrs"
        if not attrs_p.exists():
            return False
        attrs = json.loads(attrs_p.read_text())
        return "multiscales" in attrs
    except Exception:
        return False


def _safe_name(s: str) -> str:
    """
    Sanitize for filesystem folder name.
    """
    keep = [c if c.isalnum() or c in ("_", "-", ".") else "_" for c in s.strip()]
    name = "".join(keep).strip("._")
    return name or "untitled"


def _get_multiscales_paths(root: zarr.Group) -> list[str]:
    """
    Get the paths info from the datasets within the zarr group.
    """
    ms = root.attrs["multiscales"][0]
    return [d["path"] for d in ms["datasets"]]


def _phys_xy_um(root: zarr.Group, L: int=0) -> tuple[float,float]:
    """
    Read (phys_x_um, phys_y_um) from the child's NGFF multiscales at Lth resolution.
    L=0 is the highest resolution, used by default
    """
    # dpaths = _get_multiscales_paths(root)
    ms = root.attrs["multiscales"][0]
    scale = ms["datasets"][L]["coordinateTransformations"][0]["scale"]  # NGFF order: [c,y,x]
    phys_y = float(scale[1])
    phys_x = float(scale[2])
    return phys_x, phys_y # (px_um, py_um)


def _detect_source_ds_factor(root: zarr.Group) -> float:
    """
    Detect the source downsample schedule. Usually 2x or 4x.
    """
    ms = root.attrs["multiscales"][0]
    s = [ds["coordinateTransformations"][0]["scale"] for ds in ms["datasets"]]  # [ [1, py, px], ... ]
    ys = [float(si[1]) for si in s]  # use Y only (X should match)
    ratios = [ys[i+1] / ys[i] for i in range(len(ys)-1)]

    # Return the median rounded to 2 decimals (often ~2.0 or ~4.0)
    ratios.sort()

    return round(ratios[len(ratios)//2], 2)




def _sizes_for_mips_xy(W: int, H: int, levels: int) -> list[tuple[int,int]]:
    sizes = []
    w, h = W, H
    for _ in range(levels):
        sizes.append((w, h))
        w = max(1, w // 2)
        h = max(1, h // 2)
    return sizes

def _voxel_sizes_for_mips_xy(phys_xyz: int, levels: int, scale_factor: int = 2) -> list[tuple[int,int,int]]:
    """
    phys_xyz is (x_nm, y_nm, z_nm).
    Double XY per MIP; keep Z fixed.
    Returns [(x0,y0,z0), (x1,y1,z1), ...] as ints.
    """
    voxel_sizes = []
    x, y, z = phys_xyz
    x = int(round(x))
    y = int(round(y))
    z = int(round(z))
    for _ in range(levels):
        voxel_sizes.append((x, y, z))
        x *= scale_factor
        y *= scale_factor
        # Keep z the same since these are 2D plates
    return voxel_sizes

