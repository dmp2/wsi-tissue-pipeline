"""Surface mesh I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "PyVista/VTK are not installed. Install with: "
            'pip install -e ".[visualization]"'
        ) from exc
    return pv


def write_surface(vertices, faces, filename: str | Path) -> Path:
    """Write a triangle surface mesh with PyVista."""

    pv = _require_pyvista()
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    mesh = pv.PolyData(vertices, np.c_[np.full(len(faces), 3), faces])
    mesh.save(filename)
    return filename
