"""Surface plotting helpers."""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

from wsi_pipeline.surface.io import _require_pyvista, read_surface


def _surface_arrays(surface, faces=None) -> tuple[np.ndarray, np.ndarray]:
    if faces is None:
        if isinstance(surface, (str, os.PathLike)):
            return read_surface(surface)
        try:
            surface, faces = surface
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "plot_surface expects either (vertices, faces), a (vertices, faces) tuple, "
                "or a path to a legacy VTK POLYDATA surface"
            ) from exc

    vertices = np.asarray(surface, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices with shape (n, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces with shape (n, 3), got {faces.shape}")
    return vertices, faces


def _plot_surface_matplotlib(vertices, faces, *, show: bool):
    """Display a triangular mesh with Matplotlib."""

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(vertices[faces], facecolor="b", edgecolor="k", alpha=0.2)
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    min_v = np.min(vertices, axis=0)
    max_v = np.max(vertices, axis=0)
    ax.set_xlim3d(min_v[0], max_v[0])
    ax.set_ylim3d(min_v[1], max_v[1])
    ax.set_zlim3d(min_v[2], max_v[2])
    ax.set_aspect("equal")
    if show:
        plt.show()
    return ax


def _plot_surface_pyvista(
    vertices,
    faces,
    *,
    show: bool,
    jupyter_backend: str | None,
):
    """Display a triangular mesh with PyVista."""

    pv = _require_pyvista()
    cells = np.empty((faces.shape[0], 4), dtype=np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = faces
    mesh = pv.PolyData(vertices, cells.ravel())

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="lightsteelblue", opacity=0.55, show_edges=False)
    plotter.add_axes()
    plotter.show_bounds()
    if show:
        show_kwargs = {}
        if jupyter_backend is not None:
            show_kwargs["jupyter_backend"] = jupyter_backend
        plotter.show(**show_kwargs)
    return plotter


def plot_surface(
    surface,
    faces=None,
    *,
    backend: Literal["matplotlib", "pyvista"] = "matplotlib",
    show: bool = True,
    jupyter_backend: str | None = None,
):
    """Display a triangular surface from arrays or a legacy VTK ``POLYDATA`` path."""

    vertices, faces = _surface_arrays(surface, faces)
    if backend == "matplotlib":
        return _plot_surface_matplotlib(vertices, faces, show=show)
    if backend == "pyvista":
        return _plot_surface_pyvista(vertices, faces, show=show, jupyter_backend=jupyter_backend)
    raise ValueError("backend must be either 'matplotlib' or 'pyvista'")
