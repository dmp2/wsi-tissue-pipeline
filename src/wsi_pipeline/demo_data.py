"""
Demo data helpers for notebooks and sample-data scripts.

These helpers intentionally live inside the installed package so Docker and
editable installs can import them from notebooks without relying on the repo
layout or notebook working directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .omezarr import build_mips_from_yxc, compute_num_mips_min_side, write_ngff_from_mips


def _generate_synthetic_wsi_array(
    *,
    width: int,
    height: int,
    seed: int,
) -> np.ndarray:
    """Create a synthetic RGB histology-like image array."""

    from scipy import ndimage

    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 245, dtype=np.uint8)

    noise = rng.normal(0, 3, size=(height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    num_sections = int(rng.integers(3, 8))
    y_coords, x_coords = np.ogrid[:height, :width]

    for _ in range(num_sections):
        cx = int(rng.integers(width // 4, 3 * width // 4))
        cy = int(rng.integers(height // 4, 3 * height // 4))
        rx = int(rng.integers(max(40, width // 16), max(80, width // 7)))
        ry = int(rng.integers(max(30, height // 18), max(60, height // 6)))
        angle = float(rng.uniform(0, 360))

        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))
        x_rot = (x_coords - cx) * cos_a + (y_coords - cy) * sin_a
        y_rot = -(x_coords - cx) * sin_a + (y_coords - cy) * cos_a

        boundary_noise = rng.normal(0, 0.1, size=(height, width))
        boundary_noise = ndimage.gaussian_filter(
            boundary_noise,
            sigma=max(6.0, min(width, height) / 60.0),
        )
        ellipse_mask = ((x_rot / rx) ** 2 + (y_rot / ry) ** 2) <= (1 + boundary_noise)

        tissue_color = np.array(
            [
                int(rng.integers(180, 230)),
                int(rng.integers(140, 200)),
                int(rng.integers(180, 220)),
            ],
            dtype=np.float32,
        )

        for channel in range(3):
            tissue_region = img[:, :, channel].copy()
            local_variation = rng.normal(0, 10, size=(height, width))
            tissue_values = tissue_color[channel] + local_variation
            tissue_region[ellipse_mask] = np.clip(tissue_values[ellipse_mask], 0, 255)
            img[:, :, channel] = tissue_region.astype(np.uint8)

        num_spots = int(rng.integers(50, 200))
        spot_coords = np.where(ellipse_mask)
        if len(spot_coords[0]) > num_spots:
            indices = rng.choice(len(spot_coords[0]), size=num_spots, replace=False)
            for idx in indices:
                sy = int(spot_coords[0][idx])
                sx = int(spot_coords[1][idx])
                spot_radius = int(rng.integers(2, 8))
                spot_mask = (x_coords - sx) ** 2 + (y_coords - sy) ** 2 <= spot_radius**2
                img[spot_mask] = [
                    int(rng.integers(80, 120)),
                    int(rng.integers(60, 100)),
                    int(rng.integers(100, 150)),
                ]

    return img


def create_synthetic_wsi(
    output_path: str | Path,
    seed: int = 42,
    *,
    width: int = 4096,
    height: int = 3072,
) -> Path:
    """Write a synthetic WSI-like PNG for demos and tests."""

    from PIL import Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = _generate_synthetic_wsi_array(width=width, height=height, seed=seed)
    Image.fromarray(img).save(output_path)
    return output_path


def create_demo_ngff_plate(
    plate_root: str | Path,
    *,
    num_tissues: int = 2,
    base_seed: int = 200,
    image_shape: tuple[int, int] = (384, 512),
    phys_xy_um: tuple[float, float] = (8.0, 8.0),
) -> list[Path]:
    """Create a tiny multi-tissue NGFF plate for notebook demos."""

    plate_root = Path(plate_root)
    plate_root.mkdir(parents=True, exist_ok=True)

    height, width = image_shape
    num_mips = max(1, min(3, compute_num_mips_min_side(width, height, min_side_for_mips=64)))

    outputs: list[Path] = []
    for idx in range(num_tissues):
        child = plate_root / f"demo_tissue_{idx + 1:03d}.ome.zarr"
        if child.exists():
            shutil.rmtree(child)

        img = _generate_synthetic_wsi_array(
            width=width,
            height=height,
            seed=base_seed + idx,
        )
        mips = build_mips_from_yxc(img, num_mips=num_mips)
        write_ngff_from_mips(
            mips,
            child,
            phys_xy_um=phys_xy_um,
            name=f"demo_tissue_{idx + 1:03d}",
            chunks_xy=128,
            channel_labels=["R", "G", "B"],
            channel_colors=["FF0000", "00FF00", "0000FF"],
        )
        outputs.append(child)

    return outputs


__all__ = ["create_synthetic_wsi", "create_demo_ngff_plate"]
