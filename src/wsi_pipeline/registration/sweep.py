"""Utilities for EM-LDDMM self-alignment parameter sweeps."""

from __future__ import annotations

import csv
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import load_workflow_config_override
from .visualization import read_vtk_structured_points
from .workflow import run_emlddmm_workflow

DEFAULT_EA2D_VALUES: tuple[float, ...] = (5e3, 1e4, 2e4, 5e4, 1e5)
STRESS_EA2D_VALUE = 2e5
DEFAULT_ATLAS_FREE_EXTRA_KWARGS: dict[str, float | int] = {
    "eA": 0.0,
    "ev": 0.0,
    "v_start": 100000,
    "dv": 2000.0,
    "sigmaM": 0.1,
    "sigmaB": 0.2,
    "sigmaA": 0.5,
}

_PRESENT_STATUSES = {"present", "true", "1"}
_IDENTITY_3X3 = np.eye(3, dtype=np.float64)


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, indent=2)
    return path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def format_ea2d_label(value: float) -> str:
    """Return a stable folder label for an ``eA2d`` value."""

    value_float = float(value)
    if value_float == 0.0:
        return "eA2d_0"
    mantissa, exponent = f"{abs(value_float):.6e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
    exponent_int = int(exponent)
    sign = "m" if value_float < 0 else ""
    return f"eA2d_{sign}{mantissa}e{exponent_int}"


def build_ea2d_override(
    ea2d: float,
    *,
    n_steps: int = 10,
    desired_resolution_um: float = 200.0,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a workflow config override for one atlas-free ``eA2d`` run."""

    atlas_free_kwargs = dict(DEFAULT_ATLAS_FREE_EXTRA_KWARGS)
    if extra_kwargs:
        atlas_free_kwargs.update(extra_kwargs)
    return {
        "units": {"desired_resolution_um": float(desired_resolution_um)},
        "self_alignment": {
            "n_steps": int(n_steps),
            "eA2d": float(ea2d),
            "extra_kwargs": atlas_free_kwargs,
        },
    }


def _resolve_path(value: Any, *, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.expanduser().resolve()


def _transform_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"A2d_(\d+)", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def _load_matrix(path: Path) -> np.ndarray:
    try:
        matrix = np.loadtxt(path, delimiter=",", dtype=np.float64)
    except ValueError:
        matrix = np.loadtxt(path, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (3, 3):
        return matrix
    if matrix.shape == (2, 3):
        promoted = np.eye(3, dtype=np.float64)
        promoted[:2, :3] = matrix
        return promoted
    if matrix.ndim == 1 and matrix.size == 9:
        return matrix.reshape(3, 3)
    if matrix.ndim == 1 and matrix.size == 6:
        promoted = np.eye(3, dtype=np.float64)
        promoted[:2, :3] = matrix.reshape(2, 3)
        return promoted
    raise ValueError(f"Expected a 3x3 or 2x3 A2d matrix in {path}, got {matrix.shape}")


def _load_a2d_matrices(registration_output: Path) -> tuple[np.ndarray, list[Path]]:
    transforms_dir = registration_output / "self_alignment" / "transforms"
    matrix_paths = sorted(transforms_dir.glob("A2d_*.txt"), key=_transform_sort_key)
    if not matrix_paths:
        raise FileNotFoundError(f"No A2d_*.txt files found in {transforms_dir}")
    matrices = np.stack([_load_matrix(path) for path in matrix_paths], axis=0)
    return matrices, matrix_paths


def _load_manifest(
    registration_output: Path,
    *,
    manifest_path: str | Path | None,
    plan: dict[str, Any] | None,
) -> tuple[Path | None, list[dict[str, Any]]]:
    candidates: list[Path | None] = [
        _resolve_path(manifest_path, base=registration_output) if manifest_path else None,
        _resolve_path(plan.get("manifest_path"), base=registration_output) if plan else None,
        registration_output.parent / "emlddmm_dataset_manifest.json",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            manifest = _read_json(candidate)
            entries = list(manifest.get("entries", [])) if manifest else []
            return candidate, entries
    return None, []


def _entry_present(entry: dict[str, Any]) -> bool:
    if "present" in entry:
        return bool(entry["present"])
    return str(entry.get("status", "")).strip().lower() in _PRESENT_STATUSES


def _entry_sample_id(entry: dict[str, Any], idx: int) -> str:
    for key in ("sample_id", "section_id", "source_id", "name"):
        value = entry.get(key)
        if value not in (None, ""):
            return str(value)
    path_value = entry.get("path") or entry.get("image_path")
    if path_value not in (None, ""):
        return Path(str(path_value)).stem
    return f"slice_{idx:04d}"


def _metadata_from_manifest(
    entries: list[dict[str, Any]],
    *,
    transform_count: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if len(entries) != transform_count:
        present_mask = np.ones(transform_count, dtype=bool)
        metadata = [
            {
                "grid_index": idx,
                "sample_id": f"slice_{idx:04d}",
                "status": "unknown",
            }
            for idx in range(transform_count)
        ]
        return present_mask, metadata

    present_mask = np.array([_entry_present(entry) for entry in entries], dtype=bool)
    metadata = []
    for idx, entry in enumerate(entries):
        metadata.append(
            {
                "grid_index": int(entry.get("grid_index", idx)),
                "sample_id": _entry_sample_id(entry, idx),
                "status": str(entry.get("status", "present" if present_mask[idx] else "missing")),
            }
        )
    return present_mask, metadata


def _spacing_yx_from_plan(plan: dict[str, Any] | None) -> tuple[float | None, float | None]:
    spacing = None
    if plan:
        pre_plan = plan.get("pre_resampling_plan") or {}
        spacing = pre_plan.get("target_working_spacing_um")
    if not spacing or len(spacing) < 3:
        return None, None
    y_spacing = float(spacing[1])
    x_spacing = float(spacing[2])
    return y_spacing, x_spacing


def _nan_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "median": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _summarize_transforms(
    matrices: np.ndarray,
    *,
    present_mask: np.ndarray,
    metadata: list[dict[str, Any]],
    matrix_paths: list[Path],
    spacing_yx_um: tuple[float | None, float | None],
    top_n: int,
) -> dict[str, Any]:
    translations_um = matrices[:, :2, 2]
    translation_um_norm = np.linalg.norm(translations_um, axis=1)
    y_spacing, x_spacing = spacing_yx_um
    if y_spacing and x_spacing and y_spacing > 0 and x_spacing > 0:
        translation_px_norm = np.sqrt(
            (translations_um[:, 0] / y_spacing) ** 2 + (translations_um[:, 1] / x_spacing) ** 2
        )
    else:
        translation_px_norm = np.full(translation_um_norm.shape, np.nan, dtype=np.float64)

    linear = matrices[:, :2, :2]
    linear_delta = linear - np.eye(2, dtype=np.float64)
    linear_frobenius = np.linalg.norm(linear_delta, axis=(1, 2))
    linear_max_abs = np.max(np.abs(linear_delta), axis=(1, 2))
    determinant = np.linalg.det(linear)
    angle_deg = np.degrees(np.arctan2(linear[:, 1, 0], linear[:, 0, 0]))
    identity_delta = np.max(np.abs(matrices - _IDENTITY_3X3[None, :, :]), axis=(1, 2))
    nonidentity = identity_delta > 1e-6

    present_indices = np.flatnonzero(present_mask)
    missing_mask = ~present_mask
    ranked_indices = present_indices[np.argsort(translation_um_norm[present_indices])[::-1]][:top_n]

    top_moved = []
    for rank, idx in enumerate(ranked_indices, start=1):
        row = {
            "rank": rank,
            "transform_index": int(idx),
            "grid_index": int(metadata[idx]["grid_index"]),
            "sample_id": metadata[idx]["sample_id"],
            "status": metadata[idx]["status"],
            "translation_y_um": float(translations_um[idx, 0]),
            "translation_x_um": float(translations_um[idx, 1]),
            "translation_norm_um": float(translation_um_norm[idx]),
            "translation_norm_working_px": (
                float(translation_px_norm[idx]) if np.isfinite(translation_px_norm[idx]) else None
            ),
            "rotation_deg": float(angle_deg[idx]),
            "linear_deviation_frobenius": float(linear_frobenius[idx]),
            "linear_deviation_max_abs": float(linear_max_abs[idx]),
            "linear_determinant": float(determinant[idx]),
            "matrix_path": str(matrix_paths[idx]),
        }
        top_moved.append(row)

    present_translation_um = translation_um_norm[present_mask]
    present_translation_px = translation_px_norm[present_mask]
    return {
        "translation_yx_um": translations_um,
        "translation_norm_um": translation_um_norm,
        "translation_norm_working_px": translation_px_norm,
        "angle_deg": angle_deg,
        "linear_frobenius": linear_frobenius,
        "linear_max_abs": linear_max_abs,
        "determinant": determinant,
        "nonidentity": nonidentity,
        "summary": {
            "present_translation_um": _nan_stats(present_translation_um),
            "present_translation_working_px": _nan_stats(present_translation_px),
            "all_translation_um": _nan_stats(translation_um_norm),
            "all_translation_working_px": _nan_stats(translation_px_norm),
            "rotation_abs_deg": _nan_stats(np.abs(angle_deg[present_mask])),
            "linear_deviation_frobenius": _nan_stats(linear_frobenius[present_mask]),
            "linear_deviation_max_abs": _nan_stats(linear_max_abs[present_mask]),
            "linear_determinant": _nan_stats(determinant[present_mask]),
            "nonidentity_count": int(np.count_nonzero(nonidentity)),
            "present_nonidentity_count": int(np.count_nonzero(nonidentity & present_mask)),
            "missing_nonidentity_count": int(np.count_nonzero(nonidentity & missing_mask)),
        },
        "top_moved_slices": top_moved,
    }


def _summarize_abs_difference(diff: np.ndarray) -> dict[str, float | int | None]:
    diff = np.asarray(diff, dtype=np.float32)
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean_abs": None,
            "median_abs": None,
            "p95_abs": None,
            "max_abs": None,
            "rms": None,
            "nonzero_fraction": None,
        }
    abs_diff = np.abs(finite)
    return {
        "count": int(finite.size),
        "mean_abs": float(np.mean(abs_diff)),
        "median_abs": float(np.median(abs_diff)),
        "p95_abs": float(np.percentile(abs_diff, 95.0)),
        "max_abs": float(np.max(abs_diff)),
        "rms": float(math.sqrt(float(np.mean(finite * finite)))),
        "nonzero_fraction": float(np.count_nonzero(abs_diff > 1e-6) / finite.size),
    }


def _normalize_pair(
    input_slice: np.ndarray, registered_slice: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate(
        [
            np.asarray(input_slice, dtype=np.float32).ravel(),
            np.asarray(registered_slice, dtype=np.float32).ravel(),
        ]
    )
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return np.zeros_like(input_slice, dtype=np.float32), np.zeros_like(
            registered_slice, dtype=np.float32
        )
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(input_slice, dtype=np.float32), np.zeros_like(
            registered_slice, dtype=np.float32
        )
    input_norm = np.clip((input_slice - lo) / (hi - lo), 0.0, 1.0)
    registered_norm = np.clip((registered_slice - lo) / (hi - lo), 0.0, 1.0)
    return input_norm.astype(np.float32), registered_norm.astype(np.float32)


def _write_overlay_image(
    *,
    input_data_czyx: np.ndarray,
    registered_data_czyx: np.ndarray,
    row: dict[str, Any],
    output_dir: Path,
    max_width: int = 1200,
) -> Path:
    z_idx = int(row["transform_index"])
    input_slice = np.mean(input_data_czyx[:, z_idx], axis=0)
    registered_slice = np.mean(registered_data_czyx[:, z_idx], axis=0)
    input_norm, registered_norm = _normalize_pair(input_slice, registered_slice)
    rgb = np.zeros(input_norm.shape + (3,), dtype=np.float32)
    rgb[..., 0] = registered_norm
    rgb[..., 1] = input_norm
    rgb[..., 2] = np.maximum(input_norm, registered_norm)
    image = Image.fromarray(np.clip(rgb * 255.0, 0.0, 255.0).round().astype(np.uint8), mode="RGB")
    if image.width > max_width:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image.thumbnail((max_width, max_width), resampling)

    font = ImageFont.load_default()
    title = (
        f"{row['sample_id']} z={row['transform_index']} "
        f"move={row['translation_norm_um']:.2f} um "
        f"({row['translation_norm_working_px']:.3f} px)"
        if row.get("translation_norm_working_px") is not None
        else f"{row['sample_id']} z={row['transform_index']} move={row['translation_norm_um']:.2f} um"
    )
    header_h = 24
    canvas = Image.new("RGB", (image.width, image.height + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(20, 20, 20), font=font)
    canvas.paste(image, (0, header_h))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"top_moved_rank{int(row['rank']):02d}_z{z_idx:04d}_overlay.png"
    canvas.save(out_path)
    return out_path


def _write_top_moved_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "transform_index",
        "grid_index",
        "sample_id",
        "status",
        "translation_y_um",
        "translation_x_um",
        "translation_norm_um",
        "translation_norm_working_px",
        "rotation_deg",
        "linear_deviation_frobenius",
        "linear_deviation_max_abs",
        "linear_determinant",
        "matrix_path",
        "overlay_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def diagnose_self_alignment_run(
    registration_output: str | Path,
    *,
    manifest_path: str | Path | None = None,
    top_n: int = 6,
) -> dict[str, Any]:
    """Summarize A2d motion and registered-vs-input differences for one run."""

    registration_output = Path(registration_output).expanduser().resolve()
    diagnostics_dir = registration_output / "self_alignment" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    plan_path = registration_output / "resolved_run_plan.json"
    summary_path = registration_output / "registration_summary.json"
    plan = _read_json(plan_path)
    registration_summary = _read_json(summary_path)
    matrices, matrix_paths = _load_a2d_matrices(registration_output)
    manifest_path_resolved, entries = _load_manifest(
        registration_output,
        manifest_path=manifest_path,
        plan=plan,
    )
    present_mask, metadata = _metadata_from_manifest(entries, transform_count=matrices.shape[0])
    spacing_yx_um = _spacing_yx_from_plan(plan)
    transform_summary = _summarize_transforms(
        matrices,
        present_mask=present_mask,
        metadata=metadata,
        matrix_paths=matrix_paths,
        spacing_yx_um=spacing_yx_um,
        top_n=top_n,
    )
    top_moved = list(transform_summary["top_moved_slices"])

    warnings: list[str] = []
    diff_summary: dict[str, Any] | None = None
    overlay_images: list[str] = []
    images_dir = registration_output / "self_alignment" / "images"
    input_vtk_path = images_dir / "input_target.vtk"
    registered_vtk_path = images_dir / "target_registered.vtk"
    if input_vtk_path.exists() and registered_vtk_path.exists():
        input_vtk = read_vtk_structured_points(input_vtk_path)
        registered_vtk = read_vtk_structured_points(registered_vtk_path)
        if input_vtk.data_czyx.shape != registered_vtk.data_czyx.shape:
            warnings.append(
                "Skipping registered-vs-input difference because input and registered VTK shapes differ: "
                f"{input_vtk.data_czyx.shape} != {registered_vtk.data_czyx.shape}"
            )
        else:
            diff = registered_vtk.data_czyx - input_vtk.data_czyx
            diff_summary = {
                "input_vtk": str(input_vtk_path),
                "registered_vtk": str(registered_vtk_path),
                "shape_czyx": list(diff.shape),
                "all": _summarize_abs_difference(diff),
            }
            if present_mask.shape == (diff.shape[1],):
                diff_summary["present_slices"] = _summarize_abs_difference(diff[:, present_mask])
            else:
                diff_summary["present_slices"] = None

            overlay_dir = diagnostics_dir / "top_moved_overlays"
            for row in top_moved:
                if int(row["transform_index"]) >= input_vtk.data_czyx.shape[1]:
                    continue
                overlay_path = _write_overlay_image(
                    input_data_czyx=input_vtk.data_czyx,
                    registered_data_czyx=registered_vtk.data_czyx,
                    row=row,
                    output_dir=overlay_dir,
                )
                row["overlay_path"] = str(overlay_path)
                overlay_images.append(str(overlay_path))
    else:
        warnings.append(
            "Skipping registered-vs-input difference and overlays because input_target.vtk "
            "or target_registered.vtk is missing."
        )

    diagnostics = {
        "schema_version": "emlddmm-ea2d-diagnostics/v1",
        "registration_output": str(registration_output),
        "plan_path": str(plan_path) if plan_path.exists() else None,
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "manifest_path": str(manifest_path_resolved) if manifest_path_resolved else None,
        "eA2d": (
            ((plan or {}).get("workflow_config") or {}).get("self_alignment", {}).get("eA2d")
            if plan
            else None
        ),
        "n_steps": (
            ((plan or {}).get("workflow_config") or {}).get("self_alignment", {}).get("n_steps")
            if plan
            else None
        ),
        "desired_resolution_um": (
            ((plan or {}).get("workflow_config") or {})
            .get("units", {})
            .get("desired_resolution_um")
            if plan
            else None
        ),
        "working_spacing_um_yx": list(spacing_yx_um),
        "transform_count": int(matrices.shape[0]),
        "present_count": int(np.count_nonzero(present_mask)),
        "missing_count": int(present_mask.size - np.count_nonzero(present_mask)),
        "translation_um": transform_summary["summary"]["present_translation_um"],
        "translation_working_pixels": transform_summary["summary"][
            "present_translation_working_px"
        ],
        "all_translation_um": transform_summary["summary"]["all_translation_um"],
        "all_translation_working_pixels": transform_summary["summary"][
            "all_translation_working_px"
        ],
        "rotation_abs_deg": transform_summary["summary"]["rotation_abs_deg"],
        "linear_deviation": {
            "frobenius": transform_summary["summary"]["linear_deviation_frobenius"],
            "max_abs": transform_summary["summary"]["linear_deviation_max_abs"],
            "determinant": transform_summary["summary"]["linear_determinant"],
        },
        "nonidentity_count": transform_summary["summary"]["nonidentity_count"],
        "present_nonidentity_count": transform_summary["summary"]["present_nonidentity_count"],
        "missing_nonidentity_count": transform_summary["summary"]["missing_nonidentity_count"],
        "top_moved_slices": top_moved,
        "registered_vs_input_difference": diff_summary,
        "overlay_images": overlay_images,
        "registration_summary": registration_summary,
        "warnings": warnings,
    }
    diagnostics_path = diagnostics_dir / "ea2d_diagnostics.json"
    diagnostics["diagnostics_path"] = str(diagnostics_path)
    _write_json(diagnostics_path, diagnostics)
    _write_top_moved_csv(diagnostics_dir / "top_moved_slices.csv", top_moved)
    return diagnostics


def _summary_row(run: dict[str, Any]) -> dict[str, Any]:
    diagnostics = run.get("diagnostics") or {}
    translation_um = diagnostics.get("translation_um") or {}
    translation_px = diagnostics.get("translation_working_pixels") or {}
    rotation_abs = diagnostics.get("rotation_abs_deg") or {}
    linear = diagnostics.get("linear_deviation") or {}
    fro = linear.get("frobenius") or {}
    diff = diagnostics.get("registered_vs_input_difference") or {}
    present_diff = diff.get("present_slices") or {}
    return {
        "label": run.get("label"),
        "eA2d": run.get("eA2d", diagnostics.get("eA2d")),
        "status": run.get("status"),
        "registration_output": run.get("registration_output"),
        "median_translation_um": translation_um.get("median"),
        "p95_translation_um": translation_um.get("p95"),
        "max_translation_um": translation_um.get("max"),
        "median_translation_px": translation_px.get("median"),
        "p95_translation_px": translation_px.get("p95"),
        "max_translation_px": translation_px.get("max"),
        "p95_rotation_abs_deg": rotation_abs.get("p95"),
        "p95_linear_frobenius": fro.get("p95"),
        "present_nonidentity_count": diagnostics.get("present_nonidentity_count"),
        "missing_nonidentity_count": diagnostics.get("missing_nonidentity_count"),
        "present_diff_rms": present_diff.get("rms"),
        "present_diff_mean_abs": present_diff.get("mean_abs"),
        "diagnostics_path": diagnostics.get("diagnostics_path"),
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "eA2d",
        "status",
        "registration_output",
        "median_translation_um",
        "p95_translation_um",
        "max_translation_um",
        "median_translation_px",
        "p95_translation_px",
        "max_translation_px",
        "p95_rotation_abs_deg",
        "p95_linear_frobenius",
        "present_nonidentity_count",
        "missing_nonidentity_count",
        "present_diff_rms",
        "present_diff_mean_abs",
        "diagnostics_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_sweep_summary(sweep_root: Path, runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary_rows = [_summary_row(run) for run in runs]
    summary_path = sweep_root / "ea2d_sweep_summary.json"
    summary_csv = sweep_root / "ea2d_sweep_summary.csv"
    payload = {
        "schema_version": "emlddmm-ea2d-sweep/v1",
        "sweep_root": str(sweep_root),
        "runs": runs,
        "summary_rows": summary_rows,
        "summary_path": str(summary_path),
        "summary_csv": str(summary_csv),
    }
    _write_json(summary_path, payload)
    _write_summary_csv(summary_csv, summary_rows)
    return payload


def run_ea2d_self_alignment_sweep(
    *,
    workflow_kwargs: dict[str, Any],
    sweep_root: str | Path | None = None,
    values: list[float] | tuple[float, ...] = DEFAULT_EA2D_VALUES,
    include_stress_test: bool = False,
    n_steps: int = 10,
    desired_resolution_um: float = 200.0,
    top_n: int = 6,
    dry_run: bool = False,
    force_atlas_free: bool = True,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    """Run the fixed-resolution atlas-free ``eA2d`` ladder."""

    base_kwargs = dict(workflow_kwargs)
    output_root = base_kwargs.get("registration_output")
    if output_root is None:
        dataset_root = base_kwargs.get("dataset_root")
        if dataset_root is None:
            raise ValueError("workflow_kwargs must include registration_output or dataset_root")
        output_root = Path(dataset_root) / "emlddmm"
    sweep_root = (
        Path(sweep_root).expanduser().resolve()
        if sweep_root is not None
        else Path(output_root).expanduser().resolve() / "sweeps" / "ea2d"
    )
    sweep_root.mkdir(parents=True, exist_ok=True)

    sweep_values = [float(value) for value in values]
    if include_stress_test and float(STRESS_EA2D_VALUE) not in sweep_values:
        sweep_values.append(float(STRESS_EA2D_VALUE))

    base_config_override = load_workflow_config_override(base_kwargs.get("emlddmm_config"))
    runs: list[dict[str, Any]] = []
    for ea2d in sweep_values:
        label = format_ea2d_label(ea2d)
        run_dir = sweep_root / label
        run_dir.mkdir(parents=True, exist_ok=True)
        override = build_ea2d_override(
            ea2d,
            n_steps=n_steps,
            desired_resolution_um=desired_resolution_um,
        )
        merged_override = _deep_merge(base_config_override, override)
        config_path = _write_json(run_dir / "ea2d_override.json", merged_override)

        run_kwargs = dict(base_kwargs)
        run_kwargs.pop("plan", None)
        run_kwargs["registration_output"] = run_dir
        run_kwargs["emlddmm_config"] = config_path
        run_kwargs["upsample_between_slices"] = False
        run_kwargs["run_transformation_graph"] = False
        if force_atlas_free:
            run_kwargs["atlas"] = None
            run_kwargs["label"] = None
            run_kwargs["init_affine"] = None
            run_kwargs["orientation_from"] = None
            run_kwargs["orientation_to"] = None

        run_record: dict[str, Any] = {
            "label": label,
            "eA2d": float(ea2d),
            "registration_output": str(run_dir),
            "config_path": str(config_path),
            "status": "pending",
        }
        try:
            result = run_emlddmm_workflow(**run_kwargs, dry_run=dry_run)
            run_record["status"] = "dry_run" if dry_run else "completed"
            run_record["result"] = result.model_dump(mode="python")
            if not dry_run:
                run_record["diagnostics"] = diagnose_self_alignment_run(run_dir, top_n=top_n)
        except Exception as exc:
            run_record["status"] = "failed"
            run_record["error"] = repr(exc)
            runs.append(run_record)
            _write_sweep_summary(sweep_root, runs)
            if not continue_on_error:
                raise
        else:
            runs.append(run_record)

    return _write_sweep_summary(sweep_root, runs)


def summarize_ea2d_sweep(sweep_root: str | Path) -> dict[str, Any] | None:
    """Summarize diagnostics already written under an ``eA2d`` sweep root."""

    sweep_root = Path(sweep_root).expanduser().resolve()
    if not sweep_root.exists():
        return None

    runs: list[dict[str, Any]] = []
    for run_dir in sorted(
        (path for path in sweep_root.iterdir() if path.is_dir()), key=lambda p: p.name
    ):
        diagnostics_path = run_dir / "self_alignment" / "diagnostics" / "ea2d_diagnostics.json"
        if not diagnostics_path.exists():
            continue
        diagnostics = _read_json(diagnostics_path) or {}
        diagnostics["diagnostics_path"] = str(diagnostics_path)
        runs.append(
            {
                "label": run_dir.name,
                "eA2d": diagnostics.get("eA2d"),
                "registration_output": str(run_dir),
                "status": "completed",
                "diagnostics": diagnostics,
            }
        )
    if not runs:
        return None
    return _write_sweep_summary(sweep_root, runs)
