"""Registration-derived surface and PyVista helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from wsi_pipeline.registration.visualization import (
    RegistrationNeuroglancerBundle,
    VtkStructuredPoints,
    read_vtk_structured_points,
    resolve_registration_visualization_artifacts,
)
from wsi_pipeline.surface.io import write_surface
from wsi_pipeline.surface.surface_generation import restricted_delaunay_from_image

_PRESENT_STATUSES = {"present", "true", "1"}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _present_mask_from_manifest(manifest_path: Path | None, z_count: int) -> np.ndarray | None:
    if manifest_path is None or not manifest_path.exists():
        return None
    manifest = _load_json(manifest_path)
    entries = list(manifest.get("entries", [])) if manifest else []
    if len(entries) != z_count:
        return None
    return np.array(
        [str(entry.get("status", "")).strip().lower() in _PRESENT_STATUSES for entry in entries],
        dtype=bool,
    )


def _require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "PyVista/VTK are not installed. Install with: "
            'pip install -e ".[visualization]"'
        ) from exc
    return pv


def _intensity_zyx(vtk: VtkStructuredPoints) -> np.ndarray:
    return np.mean(np.asarray(vtk.data_czyx, dtype=np.float32), axis=0)


def _as_pyvista_image_data(vtk: VtkStructuredPoints, scalars_zyx: np.ndarray):
    pv = _require_pyvista()
    x_size, y_size, z_size = vtk.dimensions_xyz
    grid = pv.ImageData(dimensions=(x_size, y_size, z_size))
    grid.origin = vtk.origin_xyz
    grid.spacing = vtk.spacing_xyz
    grid.point_data["intensity"] = np.asarray(
        np.transpose(scalars_zyx, (2, 1, 0)),
        dtype=np.float32,
        order="F",
    ).ravel(order="F")
    return grid


def _resolve_surface_vtk(artifacts, source: str) -> tuple[Path, bool]:
    if source == "best_available":
        if artifacts.filled_vtk is not None:
            return artifacts.filled_vtk, False
        if artifacts.registered_vtk is not None:
            return artifacts.registered_vtk, True
    elif source in {"filled", "filled_volume", "upsampled", "upsampling"}:
        if artifacts.filled_vtk is not None:
            return artifacts.filled_vtk, False
    elif source in {"registered", "registered_slices", "target_registered"}:
        if artifacts.registered_vtk is not None:
            return artifacts.registered_vtk, True
    elif source in {"aligned", "aligned_volume"}:
        return artifacts.aligned_vtk, artifacts.filled_vtk is None
    else:
        raise ValueError(
            "source must be one of: best_available, filled_volume, registered_slices, "
            "or aligned_volume"
        )

    raise FileNotFoundError(
        f"No {source!r} surface source is available for {artifacts.registration_output}"
    )


def _auto_threshold(intensity_zyx: np.ndarray, threshold: float | Literal["auto"]) -> float:
    finite = intensity_zyx[np.isfinite(intensity_zyx)]
    if finite.size == 0:
        raise ValueError("Cannot threshold an empty/non-finite registration volume")
    if threshold != "auto":
        return float(threshold)
    positive = finite[finite > 0]
    if positive.size:
        level = float(np.percentile(positive, 5.0))
    else:
        level = float(np.percentile(finite, 50.0))
    if not np.isfinite(level):
        raise ValueError("Could not compute a finite tissue threshold")
    return level


def _surface_mask_from_vtk(
    vtk: VtkStructuredPoints,
    *,
    threshold: float | Literal["auto"],
    present_mask: np.ndarray | None,
    sparse_stack: bool,
    smooth_sigma: float,
) -> tuple[np.ndarray, float]:
    intensity = _intensity_zyx(vtk)
    level = _auto_threshold(intensity, threshold)
    mask = (intensity > level).astype(np.float64)
    if sparse_stack and present_mask is not None and present_mask.shape == (mask.shape[0],):
        mask[~present_mask] = 0.0

    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter

        sigma = (0.0, float(smooth_sigma), float(smooth_sigma)) if sparse_stack else float(smooth_sigma)
        mask = gaussian_filter(mask, sigma=sigma)
        if sparse_stack and present_mask is not None and present_mask.shape == (mask.shape[0],):
            mask[~present_mask] = 0.0
    if not np.any(mask > 0):
        raise ValueError("No tissue voxels were found for surface generation")
    return mask, level


def _auto_min_distance(spacing_xyz: tuple[float, float, float]) -> float:
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    return float(np.sqrt(np.sum(np.square(spacing))))


def prepare_registration_surface_mesh(
    registration_output: str | Path,
    *,
    source: str = "best_available",
    method: str = "restricted_delaunay",
    output_path: str | Path | None = None,
    threshold: float | Literal["auto"] = "auto",
    min_distance: float | Literal["auto"] = "auto",
    smooth_sigma: float = 1.0,
) -> Path:
    """Create a surface mesh from Step-5 registered slice outputs."""

    artifacts = resolve_registration_visualization_artifacts(registration_output)
    source_vtk, sparse_stack = _resolve_surface_vtk(artifacts, source)
    vtk = read_vtk_structured_points(source_vtk)
    present_mask = (
        _present_mask_from_manifest(artifacts.manifest_path, vtk.data_czyx.shape[1])
        if sparse_stack
        else None
    )
    image, level = _surface_mask_from_vtk(
        vtk,
        threshold=threshold,
        present_mask=present_mask,
        sparse_stack=sparse_stack,
        smooth_sigma=smooth_sigma,
    )
    if method != "restricted_delaunay":
        raise ValueError("Only method='restricted_delaunay' is supported")

    min_distance_value = (
        _auto_min_distance(vtk.spacing_xyz) if min_distance == "auto" else float(min_distance)
    )
    faces, vertices_zyx = restricted_delaunay_from_image(
        image,
        {
            "dx": np.asarray(vtk.spacing_xyz, dtype=np.float64),
            "isoval": 0.5,
            "minD": min_distance_value,
            "threshold": level,
            "verbose": False,
        },
    )
    vertices_xyz = np.asarray(vertices_zyx, dtype=np.float64)[:, [2, 1, 0]]
    vertices_xyz = vertices_xyz + np.asarray(vtk.origin_xyz, dtype=np.float64)

    if output_path is None:
        output_path = artifacts.registration_output / "visualization" / "surface_mesh.vtk"
    output_path = Path(output_path)
    return write_surface(vertices_xyz, faces, output_path)


def _registration_bundle_paths(
    bundle: RegistrationNeuroglancerBundle | str | Path,
) -> tuple[Path, Path | None, Path | None]:
    if isinstance(bundle, RegistrationNeuroglancerBundle):
        if bundle.aligned_vtk is None:
            raise ValueError("Registration bundle does not include an aligned VTK path.")
        return bundle.aligned_vtk, bundle.registered_vtk, bundle.manifest_path

    metadata_path = Path(bundle).expanduser().resolve() / "registration_visualization.json"
    metadata = _load_json(metadata_path)
    if metadata is None:
        raise FileNotFoundError(f"No registration_visualization.json found in {bundle}")
    base_vtk = Path(metadata["aligned_vtk"])
    registered_vtk = Path(metadata["registered_vtk"]) if metadata.get("registered_vtk") else None
    manifest_path = Path(metadata["manifest_path"]) if metadata.get("manifest_path") else None
    return base_vtk, registered_vtk, manifest_path


def show_registration_pyvista_scene(
    bundle: RegistrationNeuroglancerBundle | str | Path,
    *,
    max_slices: int | None = 24,
):
    """Show a compact PyVista volume/slice scene for a prepared registration bundle."""

    pv = _require_pyvista()
    base_vtk, registered_vtk, manifest_path = _registration_bundle_paths(bundle)

    base = read_vtk_structured_points(base_vtk)
    grid = _as_pyvista_image_data(base, _intensity_zyx(base))

    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars="intensity", opacity="sigmoid_5", cmap="gray")
    if registered_vtk is not None:
        registered = read_vtk_structured_points(registered_vtk)
        present_mask = _present_mask_from_manifest(manifest_path, registered.data_czyx.shape[1])
        z_indices = (
            np.flatnonzero(present_mask)
            if present_mask is not None and np.any(present_mask)
            else np.arange(registered.data_czyx.shape[1])
        )
        if max_slices is not None and z_indices.size > max_slices:
            keep = np.linspace(0, z_indices.size - 1, num=max_slices).round().astype(int)
            z_indices = z_indices[keep]
        for z_idx in z_indices:
            z_um = registered.origin_xyz[2] + float(z_idx) * registered.spacing_xyz[2]
            plane = pv.Plane(
                center=(
                    registered.origin_xyz[0]
                    + registered.spacing_xyz[0] * (registered.dimensions_xyz[0] - 1) / 2,
                    registered.origin_xyz[1]
                    + registered.spacing_xyz[1] * (registered.dimensions_xyz[1] - 1) / 2,
                    z_um,
                ),
                direction=(0, 0, 1),
                i_size=registered.spacing_xyz[0] * max(registered.dimensions_xyz[0] - 1, 1),
                j_size=registered.spacing_xyz[1] * max(registered.dimensions_xyz[1] - 1, 1),
            )
            plotter.add_mesh(plane, color="tomato", opacity=0.18)
    plotter.add_axes()
    plotter.show()
    return plotter
