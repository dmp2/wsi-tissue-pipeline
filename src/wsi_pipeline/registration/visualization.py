"""Notebook-friendly 3D visualization helpers for EM-LDDMM outputs."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from wsi_pipeline.neuroglancer import RGB_SHADER_PLAIN, start_cors_server
from wsi_pipeline.precomputed import write_precomputed_raw_volume

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

ORIGINAL_SLICE_SHADER = r"""
#uicontrol invlerp contrast
#uicontrol float opacity slider(default=0.55, min=0.0, max=1.0)
void main() {
    float r = contrast(toNormalized(getDataValue(0)));
    float g = contrast(toNormalized(getDataValue(1)));
    float b = contrast(toNormalized(getDataValue(2)));
    float v = max(max(r, g), b);
    emitRGBA(vec4(0.0, v, v, v * opacity));
}
"""

REGISTERED_SLICE_SHADER = r"""
#uicontrol invlerp contrast
#uicontrol float opacity slider(default=0.55, min=0.0, max=1.0)
void main() {
    float r = contrast(toNormalized(getDataValue(0)));
    float g = contrast(toNormalized(getDataValue(1)));
    float b = contrast(toNormalized(getDataValue(2)));
    float v = max(max(r, g), b);
    emitRGBA(vec4(v, 0.0, v, v * opacity));
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
    aligned_vtk: Path
    input_vtk: Path | None
    registered_vtk: Path | None
    template_vtk: Path | None
    filled_vtk: Path | None
    manifest_path: Path | None
    aligned_kind: str


@dataclass(frozen=True)
class RegistrationNeuroglancerBundle:
    """Paths and metadata for a prepared registration Neuroglancer bundle."""

    root: Path
    aligned_precomputed: Path | None
    original_precomputed: Path | None
    registered_precomputed: Path | None
    tissue_mask_precomputed: Path | None
    state_path: Path
    metadata_path: Path
    aligned_vtk: Path | None
    input_vtk: Path | None
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
    input_vtk = self_images / "input_target.vtk"
    registered_vtk = self_images / "target_registered.vtk"
    template_vtk = self_images / "atlas_free_template.vtk"

    if filled_vtk.exists():
        aligned_vtk = filled_vtk
        aligned_kind = "upsampled_filled_volume"
    elif template_vtk.exists():
        aligned_vtk = template_vtk
        aligned_kind = "atlas_free_template"
    elif registered_vtk.exists():
        aligned_vtk = registered_vtk
        aligned_kind = "registered_target_fallback"
    else:
        raise FileNotFoundError(
            "No visualization-ready EM-LDDMM VTK output was found. Expected one of: "
            f"{filled_vtk}, {registered_vtk}, or {template_vtk}."
        )

    return RegistrationVisualizationArtifacts(
        registration_output=registration_output,
        aligned_vtk=aligned_vtk,
        input_vtk=input_vtk if input_vtk.exists() else None,
        registered_vtk=registered_vtk if registered_vtk.exists() else None,
        template_vtk=template_vtk if template_vtk.exists() else None,
        filled_vtk=filled_vtk if filled_vtk.exists() else None,
        manifest_path=_registration_manifest_path(registration_output),
        aligned_kind=aligned_kind,
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


def _as_tissue_mask_uint32(
    data_czyx: np.ndarray,
    *,
    present_mask: np.ndarray | None,
    fill_sparse_stack: bool,
) -> np.ndarray:
    intensity = np.mean(np.asarray(data_czyx, dtype=np.float32), axis=0)
    finite = intensity[np.isfinite(intensity)]
    if finite.size == 0:
        mask_zyx = np.zeros(intensity.shape, dtype=bool)
    else:
        positive = finite[finite > 0]
        threshold = float(np.percentile(positive, 5.0)) if positive.size else float(np.max(finite))
        mask_zyx = intensity > threshold
    if present_mask is not None and present_mask.shape == (mask_zyx.shape[0],):
        mask_zyx[~present_mask] = False
        if fill_sparse_stack and np.any(present_mask):
            present_indices = np.flatnonzero(present_mask)
            filled = np.zeros_like(mask_zyx)
            for z_idx in range(mask_zyx.shape[0]):
                nearest = present_indices[np.argmin(np.abs(present_indices - z_idx))]
                filled[z_idx] = mask_zyx[nearest]
            mask_zyx = filled
    return mask_zyx[None].astype(np.uint32)


def _layer_source(path: Path) -> str:
    return f"precomputed://file://{path.as_posix()}"


def _layer_cache_matches(path: Path, source_vtk: Path, *, layer_type: str) -> bool:
    metadata = _load_json(path / "source.json")
    return bool(
        metadata
        and metadata.get("source_vtk") == str(source_vtk)
        and metadata.get("layer_type") == layer_type
        and (path / "info").exists()
    )


def _write_image_layer_from_vtk(
    *,
    vtk_path: Path,
    output_dir: Path,
    present_mask: np.ndarray | None,
    mask_missing_slices: bool,
    overwrite: bool,
) -> Path:
    if overwrite or not _layer_cache_matches(output_dir, vtk_path, layer_type="image"):
        if output_dir.exists():
            shutil.rmtree(output_dir)
        vtk = read_vtk_structured_points(vtk_path)
        data_uint8 = _as_display_uint8(vtk.data_czyx, vtk.scalar_names)
        if mask_missing_slices:
            data_uint8 = _mask_to_present_slices(data_uint8, present_mask)
        write_precomputed_raw_volume(
            output_dir,
            data_uint8,
            voxel_size_um=vtk.spacing_xyz,
            layer_type="image",
        )
        _write_json(
            output_dir / "source.json",
            {"source_vtk": str(vtk_path), "layer_type": "image"},
        )
    return output_dir


def _write_mask_layer_from_vtk(
    *,
    vtk_path: Path,
    output_dir: Path,
    present_mask: np.ndarray | None,
    fill_sparse_stack: bool,
    overwrite: bool,
) -> Path:
    if overwrite or not _layer_cache_matches(output_dir, vtk_path, layer_type="segmentation"):
        if output_dir.exists():
            shutil.rmtree(output_dir)
        vtk = read_vtk_structured_points(vtk_path)
        mask = _as_tissue_mask_uint32(
            vtk.data_czyx,
            present_mask=present_mask,
            fill_sparse_stack=fill_sparse_stack,
        )
        write_precomputed_raw_volume(
            output_dir,
            mask,
            voxel_size_um=vtk.spacing_xyz,
            layer_type="segmentation",
        )
        _write_json(
            output_dir / "source.json",
            {
                "source_vtk": str(vtk_path),
                "layer_type": "segmentation",
                "fill_sparse_stack": fill_sparse_stack,
            },
        )
    return output_dir


def _write_state(path: Path, *, layers: list[dict[str, Any]]) -> Path:
    state = {
        "layers": layers,
        "layout": "4panel",
        "showSlices": True,
        "crossSectionScale": 1.0,
        "projectionScale": 200000.0,
        "selectedLayer": {"visible": True, "layer": layers[-1]["name"]} if layers else {},
    }
    return _write_json(path, state)


def _image_layer(name: str, source_path: Path, *, visible: bool = True) -> dict[str, Any]:
    return {
        "type": "image",
        "name": name,
        "source": _layer_source(source_path),
        "shader": RGB_SHADER_PLAIN,
        "visible": visible,
    }


def _overlay_layer(
    name: str,
    source_path: Path,
    *,
    shader: str,
    visible: bool = True,
) -> dict[str, Any]:
    return {
        "type": "image",
        "name": name,
        "source": _layer_source(source_path),
        "shader": shader,
        "opacity": 0.75,
        "visible": visible,
    }


def _segmentation_layer(name: str, source_path: Path, *, visible: bool = True) -> dict[str, Any]:
    return {
        "type": "segmentation",
        "name": name,
        "source": _layer_source(source_path),
        "segments": [1],
        "visible": visible,
    }


def prepare_registration_neuroglancer_bundle(
    registration_output: str | Path,
    *,
    overwrite: bool = False,
) -> RegistrationNeuroglancerBundle:
    """Prepare Neuroglancer-ready precomputed layers for a registration run."""

    artifacts = resolve_registration_visualization_artifacts(registration_output)
    bundle_root = artifacts.registration_output / "visualization"
    aligned_dir = bundle_root / "aligned_volume"
    original_dir = bundle_root / "original_slices"
    registered_dir = bundle_root / "registered_slices"
    mask_dir = bundle_root / "tissue_mask"
    metadata_path = bundle_root / "registration_visualization.json"
    state_path = bundle_root / "neuroglancer_state.json"

    aligned_path: Path | None = None
    original_path: Path | None = None
    registered_path: Path | None = None
    mask_path: Path | None = None

    layers: list[dict[str, Any]] = []
    aligned_path = _write_image_layer_from_vtk(
        vtk_path=artifacts.aligned_vtk,
        output_dir=aligned_dir,
        present_mask=None,
        mask_missing_slices=False,
        overwrite=overwrite,
    )
    layers.append(_image_layer("aligned_volume", aligned_path, visible=True))

    z_count = read_vtk_structured_points(artifacts.aligned_vtk).data_czyx.shape[1]
    present_mask = _present_mask_from_manifest(artifacts.manifest_path, z_count)

    if artifacts.input_vtk is not None:
        original_path = _write_image_layer_from_vtk(
            vtk_path=artifacts.input_vtk,
            output_dir=original_dir,
            present_mask=present_mask,
            mask_missing_slices=True,
            overwrite=overwrite,
        )
        layers.append(
            _overlay_layer(
                "original_slices",
                original_path,
                shader=ORIGINAL_SLICE_SHADER,
                visible=True,
            )
        )

    if artifacts.registered_vtk is not None:
        registered_path = _write_image_layer_from_vtk(
            vtk_path=artifacts.registered_vtk,
            output_dir=registered_dir,
            present_mask=present_mask,
            mask_missing_slices=True,
            overwrite=overwrite,
        )
        layers.append(
            _overlay_layer(
                "registered_slices",
                registered_path,
                shader=REGISTERED_SLICE_SHADER,
                visible=True,
            )
        )

    mask_source = artifacts.filled_vtk or artifacts.registered_vtk or artifacts.aligned_vtk
    if mask_source is not None:
        mask_path = _write_mask_layer_from_vtk(
            vtk_path=mask_source,
            output_dir=mask_dir,
            present_mask=present_mask,
            fill_sparse_stack=artifacts.filled_vtk is None,
            overwrite=overwrite,
        )
        layers.append(_segmentation_layer("tissue_mask", mask_path, visible=True))

    _write_state(state_path, layers=layers)
    _write_json(
        metadata_path,
        {
            "aligned_vtk": str(artifacts.aligned_vtk),
            "aligned_kind": artifacts.aligned_kind,
            "input_vtk": str(artifacts.input_vtk) if artifacts.input_vtk else None,
            "registered_vtk": str(artifacts.registered_vtk) if artifacts.registered_vtk else None,
            "template_vtk": str(artifacts.template_vtk) if artifacts.template_vtk else None,
            "filled_vtk": str(artifacts.filled_vtk) if artifacts.filled_vtk else None,
            "manifest_path": str(artifacts.manifest_path) if artifacts.manifest_path else None,
            "aligned_precomputed": str(aligned_path) if aligned_path else None,
            "original_precomputed": str(original_path) if original_path else None,
            "registered_precomputed": str(registered_path) if registered_path else None,
            "tissue_mask_precomputed": str(mask_path) if mask_path else None,
            "state_path": str(state_path),
            "note": (
                "original_slices is available only for runs that wrote "
                "self_alignment/images/input_target.vtk. Rerun Step 6 with the current "
                "pipeline code if this layer is missing."
            ),
        },
    )
    return RegistrationNeuroglancerBundle(
        root=bundle_root,
        aligned_precomputed=aligned_path,
        original_precomputed=original_path,
        registered_precomputed=registered_path,
        tissue_mask_precomputed=mask_path,
        state_path=state_path,
        metadata_path=metadata_path,
        aligned_vtk=artifacts.aligned_vtk,
        input_vtk=artifacts.input_vtk,
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
                if layer.get("type") == "segmentation":
                    s.layers[name] = ng.SegmentationLayer(source=source)
                    s.layers[name].segments = set(layer.get("segments", [1]))
                else:
                    s.layers[name] = ng.ImageLayer(
                        source=source,
                        shader=layer.get("shader", RGB_SHADER_PLAIN),
                    )
                if layer.get("visible") is False:
                    s.layers[name].visible = False
        s.layout = state.get("layout", "4panel")
    return viewer, httpd

