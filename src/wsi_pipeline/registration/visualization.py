"""Notebook-friendly 3D visualization helpers for EM-LDDMM outputs."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import numpy as np

from wsi_pipeline.neuroglancer import RGB_SHADER_PLAIN, start_cors_server

_PRESENT_STATUSES = {"present", "true", "1"}
_VTK_TO_DTYPE: dict[str, str] = {
    "unsigned_char": "u1",
    "char": "i1",
    "unsigned_short": ">u2",
    "short": ">i2",
    "unsigned_int": ">u4",
    "int": ">i4",
    "float": ">f4",
    "double": ">f8",
}

OVERLAY_SHADER = r"""
#uicontrol invlerp contrast
#uicontrol float opacity slider(default=0.55, min=0.0, max=1.0)
void main() {
    vec3 rgb = vec3(
        contrast(toNormalized(getDataValue(0))),
        contrast(toNormalized(getDataValue(1))),
        contrast(toNormalized(getDataValue(2)))
    );
    float a = max(max(rgb.r, rgb.g), rgb.b) * opacity;
    emitRGBA(vec4(rgb, a));
}
"""


@dataclass(frozen=True)
class VtkStructuredPoints:
    """Small in-memory representation of a legacy VTK structured-points volume."""

    path: Path
    dimensions_xyz: tuple[int, int, int]
    origin_xyz: tuple[float, float, float]
    spacing_xyz: tuple[float, float, float]
    scalar_names: tuple[str, ...]
    data_czyx: np.ndarray


@dataclass(frozen=True)
class RegistrationVisualizationArtifacts:
    """Resolved registration artifacts used to build visualization outputs."""

    registration_output: Path
    base_vtk: Path
    registered_vtk: Path | None
    template_vtk: Path | None
    manifest_path: Path | None
    base_kind: str


@dataclass(frozen=True)
class RegistrationNeuroglancerBundle:
    """Paths and metadata for a prepared registration Neuroglancer bundle."""

    root: Path
    base_precomputed: Path
    overlay_precomputed: Path | None
    state_path: Path
    metadata_path: Path
    base_vtk: Path
    registered_vtk: Path | None
    manifest_path: Path | None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _path_from_plan(registration_output: Path, key: str) -> Path | None:
    plan = _load_json(registration_output / "resolved_run_plan.json")
    value = plan.get(key) if plan else None
    if value in (None, ""):
        return None
    return Path(value)


def _registration_manifest_path(registration_output: Path) -> Path | None:
    candidates = [
        _path_from_plan(registration_output, "manifest_path"),
        registration_output.parent / "emlddmm_dataset_manifest.json",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def resolve_registration_visualization_artifacts(
    registration_output: str | Path,
) -> RegistrationVisualizationArtifacts:
    """Resolve the best available Step-5 artifacts for 3D visualization."""

    registration_output = Path(registration_output).expanduser().resolve()
    self_images = registration_output / "self_alignment" / "images"
    filled_vtk = registration_output / "upsampling" / "filled_volume.vtk"
    registered_vtk = self_images / "target_registered.vtk"
    template_vtk = self_images / "atlas_free_template.vtk"

    if filled_vtk.exists():
        base_vtk = filled_vtk
        base_kind = "upsampled_filled_volume"
    elif registered_vtk.exists():
        base_vtk = registered_vtk
        base_kind = "registered_target"
    elif template_vtk.exists():
        base_vtk = template_vtk
        base_kind = "atlas_free_template"
    else:
        raise FileNotFoundError(
            "No visualization-ready EM-LDDMM VTK output was found. Expected one of: "
            f"{filled_vtk}, {registered_vtk}, or {template_vtk}."
        )

    return RegistrationVisualizationArtifacts(
        registration_output=registration_output,
        base_vtk=base_vtk,
        registered_vtk=registered_vtk if registered_vtk.exists() else None,
        template_vtk=template_vtk if template_vtk.exists() else None,
        manifest_path=_registration_manifest_path(registration_output),
        base_kind=base_kind,
    )


def _read_nonempty_line(f) -> str:
    while True:
        line = f.readline()
        if line == b"":
            return ""
        text = line.decode("utf-8", errors="replace").strip()
        if text:
            return text


def read_vtk_structured_points(path: str | Path) -> VtkStructuredPoints:
    """Read a legacy binary VTK ``STRUCTURED_POINTS`` volume.

    The EM-LDDMM writer emits channel scalars as separate VTK scalar arrays.
    This reader keeps the dependency surface small for notebook visualization
    by handling that specific legacy VTK shape directly.
    """

    path = Path(path).expanduser().resolve()
    with open(path, "rb") as f:
        first_line = _read_nonempty_line(f)
        if not first_line.startswith("# vtk DataFile"):
            raise ValueError(f"{path} is not a legacy VTK file")
        _title = _read_nonempty_line(f)
        mode = _read_nonempty_line(f).upper()
        if mode != "BINARY":
            raise ValueError(f"Only binary legacy VTK volumes are supported: {path}")
        dataset = _read_nonempty_line(f).upper()
        if dataset != "DATASET STRUCTURED_POINTS":
            raise ValueError(f"Expected DATASET STRUCTURED_POINTS in {path}, got {dataset!r}")

        dimensions_xyz: tuple[int, int, int] | None = None
        origin_xyz: tuple[float, float, float] | None = None
        spacing_xyz: tuple[float, float, float] | None = None
        point_count: int | None = None

        while point_count is None:
            line = _read_nonempty_line(f)
            if not line:
                raise ValueError(f"VTK header ended before POINT_DATA in {path}")
            parts = line.split()
            key = parts[0].upper()
            if key == "DIMENSIONS" and len(parts) >= 4:
                dimensions_xyz = tuple(int(value) for value in parts[1:4])
            elif key == "ORIGIN" and len(parts) >= 4:
                origin_xyz = tuple(float(value) for value in parts[1:4])
            elif key == "SPACING" and len(parts) >= 4:
                spacing_xyz = tuple(float(value) for value in parts[1:4])
            elif key == "POINT_DATA" and len(parts) >= 2:
                point_count = int(parts[1])

        if dimensions_xyz is None or origin_xyz is None or spacing_xyz is None:
            raise ValueError(f"VTK file is missing DIMENSIONS, ORIGIN, or SPACING: {path}")
        expected_points = math.prod(dimensions_xyz)
        if point_count != expected_points:
            raise ValueError(
                f"POINT_DATA does not match DIMENSIONS in {path}: "
                f"{point_count} != {expected_points}"
            )

        scalar_names: list[str] = []
        scalar_arrays: list[np.ndarray] = []
        while True:
            line = _read_nonempty_line(f)
            if not line:
                break
            parts = line.split()
            if len(parts) < 3 or parts[0].upper() != "SCALARS":
                raise ValueError(f"Expected SCALARS declaration in {path}, got {line!r}")
            name = parts[1]
            vtk_dtype = parts[2].lower()
            dtype_spec = _VTK_TO_DTYPE.get(vtk_dtype)
            if dtype_spec is None:
                raise ValueError(f"Unsupported VTK scalar dtype {vtk_dtype!r} in {path}")
            lookup = _read_nonempty_line(f).upper()
            if not lookup.startswith("LOOKUP_TABLE"):
                raise ValueError(f"Expected LOOKUP_TABLE after SCALARS in {path}")

            dtype = np.dtype(dtype_spec)
            raw = f.read(point_count * dtype.itemsize)
            if len(raw) != point_count * dtype.itemsize:
                raise ValueError(f"VTK scalar array {name!r} is truncated in {path}")
            arr = np.frombuffer(raw, dtype=dtype).astype(np.float32, copy=False)
            x_size, y_size, z_size = dimensions_xyz
            scalar_arrays.append(arr.reshape((z_size, y_size, x_size)))
            scalar_names.append(name)

        if not scalar_arrays:
            raise ValueError(f"No scalar arrays found in {path}")

    return VtkStructuredPoints(
        path=path,
        dimensions_xyz=dimensions_xyz,
        origin_xyz=origin_xyz,
        spacing_xyz=spacing_xyz,
        scalar_names=tuple(scalar_names),
        data_czyx=np.stack(scalar_arrays, axis=0),
    )


def _rgb_channel_order(names: tuple[str, ...]) -> list[int]:
    lower_names = [name.lower() for name in names]
    order: list[int] = []
    for channel in ("r", "g", "b"):
        matches = [idx for idx, name in enumerate(lower_names) if f"({channel})" in name]
        if matches:
            order.append(matches[0])
    if len(order) == 3:
        return order
    return list(range(min(3, len(names))))


def _as_display_uint8(data_czyx: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    order = _rgb_channel_order(names)
    selected = np.asarray(data_czyx[order], dtype=np.float32)
    output = np.zeros(selected.shape, dtype=np.uint8)
    for idx, channel in enumerate(selected):
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            continue
        mn = float(np.min(finite))
        mx = float(np.max(finite))
        if mx <= mn:
            continue
        if mn >= 0.0 and mx <= 1.5:
            scaled = channel * 255.0
        elif mn >= 0.0 and mx <= 255.0:
            scaled = channel
        else:
            lo, hi = np.percentile(finite, [1.0, 99.5])
            if hi <= lo:
                lo, hi = mn, mx
            scaled = (channel - float(lo)) * (255.0 / (float(hi) - float(lo)))
        output[idx] = np.clip(scaled, 0.0, 255.0).round().astype(np.uint8)
    if output.shape[0] == 1:
        output = np.repeat(output, 3, axis=0)
    return output


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


def _mask_to_present_slices(data_czyx: np.ndarray, present_mask: np.ndarray | None) -> np.ndarray:
    masked = np.asarray(data_czyx).copy()
    if present_mask is None:
        return masked
    if present_mask.shape != (masked.shape[1],):
        return masked
    masked[:, ~present_mask, :, :] = 0
    return masked


def _write_precomputed_raw(
    *,
    path: Path,
    data_czyx: np.ndarray,
    spacing_xyz_um: tuple[float, float, float],
    chunk_size_xyz: tuple[int, int, int] = (64, 64, 64),
) -> None:
    data = np.asarray(data_czyx, dtype=np.uint8)
    if data.ndim != 4:
        raise ValueError(f"Expected CZYX data, got shape {data.shape}")
    channels, z_size, y_size, x_size = data.shape

    path.mkdir(parents=True, exist_ok=True)
    key = "0"
    scale_dir = path / key
    scale_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": "uint8",
        "num_channels": int(channels),
        "type": "image",
        "scales": [
            {
                "chunk_sizes": [list(map(int, chunk_size_xyz))],
                "encoding": "raw",
                "key": key,
                "resolution": [int(round(float(value) * 1000.0)) for value in spacing_xyz_um],
                "size": [int(x_size), int(y_size), int(z_size)],
                "voxel_offset": [0, 0, 0],
            }
        ],
    }
    _write_json(path / "info", info)

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


def _write_state(path: Path, *, base_source: str, overlay_source: str | None) -> Path:
    layers: list[dict[str, Any]] = [
        {
            "type": "image",
            "name": "aligned_volume",
            "source": base_source,
            "shader": RGB_SHADER_PLAIN,
        }
    ]
    if overlay_source is not None:
        layers.append(
            {
                "type": "image",
                "name": "registered_slices",
                "source": overlay_source,
                "shader": OVERLAY_SHADER,
                "opacity": 0.75,
            }
        )
    state = {
        "layers": layers,
        "layout": "4panel",
        "showSlices": True,
        "crossSectionScale": 1.0,
        "projectionScale": 200000.0,
        "selectedLayer": {"visible": True, "layer": layers[-1]["name"]},
    }
    return _write_json(path, state)


def prepare_registration_neuroglancer_bundle(
    registration_output: str | Path,
    *,
    overwrite: bool = False,
) -> RegistrationNeuroglancerBundle:
    """Prepare Neuroglancer-ready precomputed layers for a registration run."""

    artifacts = resolve_registration_visualization_artifacts(registration_output)
    bundle_root = artifacts.registration_output / "visualization"
    base_dir = bundle_root / "aligned_volume"
    overlay_dir = bundle_root / "registered_slices"
    metadata_path = bundle_root / "registration_visualization.json"
    state_path = bundle_root / "neuroglancer_state.json"

    if overwrite and bundle_root.exists():
        shutil.rmtree(bundle_root)

    if not base_dir.exists():
        base_vtk = read_vtk_structured_points(artifacts.base_vtk)
        base_uint8 = _as_display_uint8(base_vtk.data_czyx, base_vtk.scalar_names)
        _write_precomputed_raw(
            path=base_dir,
            data_czyx=base_uint8,
            spacing_xyz_um=base_vtk.spacing_xyz,
        )
    else:
        base_vtk = read_vtk_structured_points(artifacts.base_vtk)

    overlay_path: Path | None = None
    if artifacts.registered_vtk is not None:
        if not overlay_dir.exists():
            registered_vtk = read_vtk_structured_points(artifacts.registered_vtk)
            registered_uint8 = _as_display_uint8(
                registered_vtk.data_czyx,
                registered_vtk.scalar_names,
            )
            present_mask = _present_mask_from_manifest(
                artifacts.manifest_path,
                registered_uint8.shape[1],
            )
            registered_uint8 = _mask_to_present_slices(registered_uint8, present_mask)
            _write_precomputed_raw(
                path=overlay_dir,
                data_czyx=registered_uint8,
                spacing_xyz_um=registered_vtk.spacing_xyz,
            )
        overlay_path = overlay_dir

    _write_state(
        state_path,
        base_source=f"precomputed://file://{base_dir.as_posix()}",
        overlay_source=(
            f"precomputed://file://{overlay_dir.as_posix()}" if overlay_path is not None else None
        ),
    )
    _write_json(
        metadata_path,
        {
            "base_vtk": str(artifacts.base_vtk),
            "base_kind": artifacts.base_kind,
            "registered_vtk": str(artifacts.registered_vtk) if artifacts.registered_vtk else None,
            "template_vtk": str(artifacts.template_vtk) if artifacts.template_vtk else None,
            "manifest_path": str(artifacts.manifest_path) if artifacts.manifest_path else None,
            "base_precomputed": str(base_dir),
            "overlay_precomputed": str(overlay_path) if overlay_path else None,
            "state_path": str(state_path),
        },
    )
    return RegistrationNeuroglancerBundle(
        root=bundle_root,
        base_precomputed=base_dir,
        overlay_precomputed=overlay_path,
        state_path=state_path,
        metadata_path=metadata_path,
        base_vtk=artifacts.base_vtk,
        registered_vtk=artifacts.registered_vtk,
        manifest_path=artifacts.manifest_path,
    )


def _rewrite_file_precomputed_source(source: str, *, base_dir: Path, http_base: str) -> str:
    prefix = "precomputed://file://"
    if not source.startswith(prefix):
        return source
    parsed = urlparse(source.replace("precomputed://", ""))
    fs_path = Path(parsed.path).resolve()
    relative = fs_path.relative_to(base_dir)
    return f"precomputed://{http_base.rstrip('/')}/{relative.as_posix()}"


def open_registration_neuroglancer_view(
    bundle: RegistrationNeuroglancerBundle | str | Path,
    *,
    http_host: str = "localhost",
    http_port: int = 8010,
    ng_host: str = "localhost",
    ng_port: int = 9998,
) -> tuple:
    """Open a prepared registration bundle in Neuroglancer."""

    try:
        import neuroglancer as ng
    except ImportError as exc:
        raise ImportError(
            "Neuroglancer is not installed. Install with: "
            'pip install -e ".[visualization]"'
        ) from exc

    if isinstance(bundle, RegistrationNeuroglancerBundle):
        bundle_root = bundle.root
        state_path = bundle.state_path
    else:
        bundle_root = Path(bundle).expanduser().resolve()
        state_path = bundle_root / "neuroglancer_state.json"
    state = _load_json(state_path)
    if state is None:
        raise FileNotFoundError(f"No Neuroglancer state found: {state_path}")

    httpd = start_cors_server(bundle_root, host=http_host, port=http_port)
    http_base = f"http://{http_host}:{http_port}"
    for layer in state.get("layers", []):
        source = layer.get("source")
        if isinstance(source, str):
            layer["source"] = _rewrite_file_precomputed_source(
                source,
                base_dir=bundle_root,
                http_base=http_base,
            )

    ng.set_server_bind_address(bind_address=ng_host, bind_port=ng_port)
    viewer = ng.Viewer()
    with viewer.txn() as s:
        for layer in state.get("layers", []):
            source = layer.get("source")
            name = layer.get("name")
            if isinstance(source, str) and isinstance(name, str):
                s.layers[name] = ng.ImageLayer(
                    source=source,
                    shader=layer.get("shader", RGB_SHADER_PLAIN),
                )
        s.layout = state.get("layout", "4panel")
    return viewer, httpd


def prepare_registration_surface_mesh(
    registration_output: str | Path,
    *,
    output_path: str | Path | None = None,
    threshold: float | Literal["auto"] = "auto",
    smooth: bool = True,
) -> Path:
    """Extract a lightweight surface mesh from the selected registration volume."""

    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "PyVista/VTK are not installed. Install with: "
            'pip install -e ".[visualization]"'
        ) from exc

    artifacts = resolve_registration_visualization_artifacts(registration_output)
    vtk = read_vtk_structured_points(artifacts.base_vtk)
    intensity_zyx = np.mean(np.asarray(vtk.data_czyx, dtype=np.float32), axis=0)
    finite = intensity_zyx[np.isfinite(intensity_zyx)]
    if finite.size == 0:
        raise ValueError(f"Cannot mesh empty volume: {artifacts.base_vtk}")
    positive = finite[finite > 0]
    if threshold == "auto" and positive.size:
        level = float(np.percentile(positive, 5.0))
    else:
        level = float(np.percentile(finite, 50.0)) if threshold == "auto" else float(threshold)
    if not np.isfinite(level) or level <= 0:
        level = float(np.percentile(finite, 50.0))

    # PyVista ImageData expects point scalars in XYZ order.
    x_size, y_size, z_size = vtk.dimensions_xyz
    grid = pv.ImageData(dimensions=(x_size, y_size, z_size))
    grid.origin = vtk.origin_xyz
    grid.spacing = vtk.spacing_xyz
    grid.point_data["intensity"] = np.asarray(
        np.transpose(intensity_zyx, (2, 1, 0)),
        dtype=np.float32,
        order="F",
    ).ravel(order="F")
    mesh = grid.contour([level], scalars="intensity")
    if smooth and mesh.n_points > 0:
        mesh = mesh.smooth(n_iter=20, relaxation_factor=0.05)
    if output_path is None:
        output_path = artifacts.registration_output / "visualization" / "surface_mesh.vtp"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(output_path)
    return output_path


def show_registration_pyvista_scene(
    bundle: RegistrationNeuroglancerBundle | str | Path,
    *,
    max_slices: int | None = 24,
):
    """Show a compact PyVista volume/slice scene for a prepared bundle."""

    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "PyVista/VTK are not installed. Install with: "
            'pip install -e ".[visualization]"'
        ) from exc

    if not isinstance(bundle, RegistrationNeuroglancerBundle):
        metadata = _load_json(Path(bundle) / "registration_visualization.json")
        if metadata is None:
            raise FileNotFoundError(f"No registration_visualization.json found in {bundle}")
        base_vtk = Path(metadata["base_vtk"])
        registered_vtk = Path(metadata["registered_vtk"]) if metadata.get("registered_vtk") else None
        manifest_path = Path(metadata["manifest_path"]) if metadata.get("manifest_path") else None
    else:
        base_vtk = bundle.base_vtk
        registered_vtk = bundle.registered_vtk
        manifest_path = bundle.manifest_path

    base = read_vtk_structured_points(base_vtk)
    intensity_zyx = np.mean(np.asarray(base.data_czyx, dtype=np.float32), axis=0)
    x_size, y_size, z_size = base.dimensions_xyz
    grid = pv.ImageData(dimensions=(x_size, y_size, z_size))
    grid.origin = base.origin_xyz
    grid.spacing = base.spacing_xyz
    grid.point_data["intensity"] = np.asarray(
        np.transpose(intensity_zyx, (2, 1, 0)),
        dtype=np.float32,
        order="F",
    ).ravel(order="F")

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
                    registered.origin_xyz[0] + registered.spacing_xyz[0] * (registered.dimensions_xyz[0] - 1) / 2,
                    registered.origin_xyz[1] + registered.spacing_xyz[1] * (registered.dimensions_xyz[1] - 1) / 2,
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
