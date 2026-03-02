"""Output helpers for the EM-LDDMM workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .backend import EmlddmmBackend
from .targets import EmlddmmTarget


@dataclass
class RegistrationImage:
    """Minimal image wrapper compatible with legacy EM-LDDMM output writers."""

    x: list[np.ndarray]
    data: np.ndarray
    title: str
    space: str
    name: str
    sample_ids: list[str] = field(default_factory=list)

    def fnames(self) -> list[str]:
        """Return per-slice names expected by legacy writers."""

        if self.sample_ids:
            return [Path(sample_id).stem for sample_id in self.sample_ids]
        return [f"slice_{idx:04d}" for idx in range(self.data.shape[1])]


def _jsonify(obj: Any) -> Any:
    """Convert nested objects into JSON-serializable primitives."""

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {key: _jsonify(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(value) for value in obj]
    return obj


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON payload with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, indent=2)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_kind(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name.endswith("artifacts.json"):
        return "artifact_manifest"
    if name.endswith("config.json") or name.endswith("inputs.json"):
        return "config"
    if name.endswith("summary.json") or name.endswith("metadata.json"):
        return "summary"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "image"
    if suffix == ".vtk":
        return "volume"
    if suffix == ".npy":
        return "payload"
    if suffix == ".txt":
        return "text"
    if suffix == ".html":
        return "report"
    return "other"


def _should_hash_artifact(path: Path, *, kind: str) -> bool:
    if kind in {"config", "summary", "text", "artifact_manifest"}:
        return path.stat().st_size <= 2 * 1024 * 1024
    return False


def build_stage_artifact_manifest(stage_dir: Path) -> dict[str, Any]:
    """Build a structured per-stage artifact manifest."""

    if not stage_dir.exists():
        return {"files": [], "entries": []}
    files = sorted(path for path in stage_dir.rglob("*") if path.is_file())
    entries: list[dict[str, Any]] = []
    for path in files:
        kind = _artifact_kind(path)
        entry: dict[str, Any] = {
            "relative_path": path.relative_to(stage_dir).as_posix(),
            "artifact_kind": kind,
            "size_bytes": path.stat().st_size,
        }
        if _should_hash_artifact(path, kind=kind):
            entry["sha256"] = _sha256(path)
        entries.append(entry)
    return {
        "files": [str(path) for path in files],
        "entries": entries,
    }


def _normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Normalize a 2D array into 8-bit range for preview output."""

    arr = np.asarray(slice_2d, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    mn = float(np.min(finite))
    mx = float(np.max(finite))
    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - mn) / (mx - mn), 0.0, 1.0)
    return (out * 255.0).round().astype(np.uint8)


def _volume_montage(volume: np.ndarray, *, present_mask: np.ndarray | None = None) -> Image.Image:
    """Build a small montage over evenly spaced z slices."""

    if volume.ndim != 4:
        raise ValueError(f"Expected channel-first 4D volume, got {volume.shape}")
    channel_first = np.asarray(volume, dtype=np.float32)
    z_count = channel_first.shape[1]
    available = np.arange(z_count, dtype=int)
    if present_mask is not None and present_mask.shape == (z_count,) and np.any(present_mask):
        available = np.flatnonzero(present_mask)
    if available.size == 0:
        available = np.arange(z_count, dtype=int)

    n_tiles = min(6, int(available.size))
    positions = np.linspace(0, available.size - 1, num=n_tiles).round().astype(int)
    selected = available[positions]
    tiles: list[np.ndarray] = []
    for idx in selected:
        slice_2d = np.mean(channel_first[:, idx], axis=0)
        tiles.append(_normalize_slice(slice_2d))
    montage = np.concatenate(tiles, axis=1)
    return Image.fromarray(montage, mode="L")


def write_self_alignment_outputs(
    *,
    backend: EmlddmmBackend,
    output_dir: Path,
    target: EmlddmmTarget,
    self_alignment: dict[str, Any],
    stage_config: dict[str, Any],
    effective_config: dict[str, Any],
) -> dict[str, str | list[str]]:
    """Write atlas-free self-alignment outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    transforms_dir = output_dir / "transforms"
    qc_dir = output_dir / "qc"
    images_dir = output_dir / "images"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    written_a2d: list[str] = []
    A2d = self_alignment.get("A2d")
    if A2d is not None and backend.write_matrix_data is not None:
        A2d_np = np.asarray(A2d)
        for idx in range(A2d_np.shape[0]):
            matrix_path = transforms_dir / f"A2d_{idx:04d}.txt"
            backend.write_matrix_data(str(matrix_path), A2d_np[idx])
            written_a2d.append(str(matrix_path))

    template_vtk = images_dir / "atlas_free_template.vtk"
    backend.write_vtk_data(
        str(template_vtk),
        target.xJ,
        np.asarray(self_alignment["I"], dtype=np.float32),
        "atlas_free_template",
    )
    registered_vtk = images_dir / "target_registered.vtk"
    backend.write_vtk_data(
        str(registered_vtk),
        target.xJ,
        np.asarray(self_alignment["Jr"], dtype=np.float32),
        "target_registered",
    )

    input_preview = qc_dir / "input_target_overview.png"
    _volume_montage(target.J, present_mask=target.present_mask).save(input_preview)
    registered_preview = qc_dir / "registered_target_overview.png"
    _volume_montage(
        np.asarray(self_alignment["Jr"], dtype=np.float32),
        present_mask=target.present_mask,
    ).save(registered_preview)
    template_preview = qc_dir / "atlas_free_template_overview.png"
    _volume_montage(
        np.asarray(self_alignment["I"], dtype=np.float32),
        present_mask=target.present_mask,
    ).save(template_preview)

    stage_config_path = _write_json(output_dir / "self_alignment_config.json", stage_config)
    effective_config_path = _write_json(output_dir / "effective_config.json", effective_config)
    summary_path = _write_json(
        output_dir / "self_alignment_summary.json",
        {
            "n_slices": int(target.J.shape[1]),
            "present_slices": int(np.count_nonzero(target.present_mask)),
            "source_format": target.source_format,
            "source_path": str(target.source_path),
            "template_vtk": str(template_vtk),
            "registered_vtk": str(registered_vtk),
            "a2d_matrices": written_a2d,
        },
    )

    return {
        "stage_config": str(stage_config_path),
        "effective_config": str(effective_config_path),
        "summary": str(summary_path),
        "a2d_matrices": written_a2d,
        "template_vtk": str(template_vtk),
        "registered_vtk": str(registered_vtk),
        "qc": [str(input_preview), str(registered_preview), str(template_preview)],
    }


def write_atlas_registration_outputs(
    *,
    backend: EmlddmmBackend,
    output_dir: Path,
    atlas_image: RegistrationImage,
    target_image: RegistrationImage,
    registration: dict[str, Any],
    stage_config: dict[str, Any],
    effective_config: dict[str, Any],
    label_axes: list[np.ndarray] | None = None,
    label_data: np.ndarray | None = None,
    transformation_graph: dict[str, Any] | None = None,
    transformation_graph_execution_config: dict[str, Any] | None = None,
    registration_payload: Any | None = None,
) -> dict[str, str | list[str]]:
    """Write atlas-registration outputs using the normalized backend helpers."""

    output_dir.mkdir(parents=True, exist_ok=True)
    backend.write_transform_outputs(str(output_dir), registration, atlas_image, target_image)
    backend.write_qc_outputs(
        str(output_dir),
        registration,
        atlas_image,
        target_image,
        xS=label_axes,
        S=label_data,
    )
    stage_config_path = _write_json(output_dir / "atlas_to_target_config.json", stage_config)
    effective_config_path = _write_json(output_dir / "effective_config.json", effective_config)
    graph_path = None
    graph_execution_path = None
    if transformation_graph is not None:
        graph_path = _write_json(output_dir / "transformation_graph_config.json", transformation_graph)
    if transformation_graph_execution_config is not None:
        graph_execution_path = _write_json(
            output_dir / "transformation_graph_execution_config.json",
            transformation_graph_execution_config,
        )
    registration_payload_path = None
    if registration_payload is not None:
        registration_payload_path = output_dir / "registration_data.npy"
        np.save(
            registration_payload_path,
            np.array(registration_payload, dtype=object),
            allow_pickle=True,
        )

    return {
        "stage_config": str(stage_config_path),
        "effective_config": str(effective_config_path),
        "transformation_graph_config": str(graph_path) if graph_path else "",
        "transformation_graph_execution_config": (
            str(graph_execution_path) if graph_execution_path else ""
        ),
        "registration_data": str(registration_payload_path) if registration_payload_path else "",
    }


def write_upsampling_outputs(
    *,
    backend: EmlddmmBackend,
    output_dir: Path,
    xJ: list[np.ndarray],
    upsampling: dict[str, Any],
    effective_config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Write optional between-slice upsampling outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    effective_config_path = _write_json(output_dir / "effective_config.json", effective_config)
    filled_path = output_dir / "filled_volume.vtk"
    backend.write_vtk_data(
        str(filled_path),
        xJ,
        np.asarray(upsampling["J_filled"], dtype=np.float32),
        "upsampled_target",
    )
    nearest_path = output_dir / "nearest_slice_reference.vtk"
    backend.write_vtk_data(
        str(nearest_path),
        xJ,
        np.asarray(upsampling["J_nearest_bad"], dtype=np.float32),
        "upsampled_target_nearest_reference",
    )
    filled_overview_path = output_dir / "filled_volume_overview.png"
    _volume_montage(np.asarray(upsampling["J_filled"], dtype=np.float32)).save(filled_overview_path)
    nearest_overview_path = output_dir / "nearest_slice_reference_overview.png"
    _volume_montage(np.asarray(upsampling["J_nearest_bad"], dtype=np.float32)).save(
        nearest_overview_path
    )
    metadata_path = _write_json(output_dir / "upsampling_metadata.json", metadata)
    return {
        "effective_config": str(effective_config_path),
        "filled_volume_vtk": str(filled_path),
        "nearest_reference_vtk": str(nearest_path),
        "filled_volume_overview": str(filled_overview_path),
        "nearest_reference_overview": str(nearest_overview_path),
        "metadata": str(metadata_path),
    }
