"""Surface plotting helpers."""

from __future__ import annotations

import numpy as np


def plot_surface(vertices, faces):
    """Display a triangular mesh with Matplotlib."""

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)
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
    plt.show()
    return ax
