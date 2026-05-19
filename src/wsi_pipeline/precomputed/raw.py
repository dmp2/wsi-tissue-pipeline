"""Small raw Neuroglancer-precomputed volume writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np


def write_precomputed_raw_volume(
    precomp_path: str | Path,
    data_czyx: np.ndarray,
    *,
    voxel_size_um: tuple[float, float, float],
    chunk_size_xyz: tuple[int, int, int] = (64, 64, 64),
    layer_type: Literal["image", "segmentation"] = "image",
    encoding: str = "raw",
) -> Path:
    """Write a single-scale raw Neuroglancer-precomputed volume from CZYX data.

    This dependency-light writer is useful when the whole volume already fits
    in memory and a notebook needs a quick viewer-ready layer.
    """

    path = Path(precomp_path)
    data = np.asarray(data_czyx)
    if data.ndim != 4:
        raise ValueError(f"Expected CZYX data, got shape {data.shape}")

    channels, z_size, y_size, x_size = data.shape
    path.mkdir(parents=True, exist_ok=True)
    scale_key = "0"
    scale_dir = path / scale_key
    scale_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": str(data.dtype),
        "num_channels": int(channels),
        "type": layer_type,
        "scales": [
            {
                "chunk_sizes": [list(map(int, chunk_size_xyz))],
                "encoding": encoding,
                "key": scale_key,
                "resolution": [int(round(float(value) * 1000.0)) for value in voxel_size_um],
                "size": [int(x_size), int(y_size), int(z_size)],
                "voxel_offset": [0, 0, 0],
            }
        ],
    }
    (path / "info").write_text(json.dumps(info, indent=2), encoding="utf-8")

    chunk_x, chunk_y, chunk_z = chunk_size_xyz
    for z0 in range(0, z_size, chunk_z):
        z1 = min(z0 + chunk_z, z_size)
        for y0 in range(0, y_size, chunk_y):
            y1 = min(y0 + chunk_y, y_size)
            for x0 in range(0, x_size, chunk_x):
                x1 = min(x0 + chunk_x, x_size)
                chunk_czyx = data[:, z0:z1, y0:y1, x0:x1]
                chunk_xyzc = np.transpose(chunk_czyx, (3, 2, 1, 0))
                chunk_path = scale_dir / f"{x0}-{x1}_{y0}-{y1}_{z0}-{z1}"
                chunk_path.write_bytes(np.asfortranarray(chunk_xyzc).tobytes(order="F"))
    return path
