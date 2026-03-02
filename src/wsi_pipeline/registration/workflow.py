"""Notebook-faithful EM-LDDMM workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import importlib.util
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .backend import EmlddmmBackend, resolve_emlddmm_backend
from .config import (
    EmlddmmResolvedPlan,
    EmlddmmStagePlan,
    EmlddmmWorkflowConfig,
    EmlddmmWorkflowResult,
    OrientationResolution,
    PreResamplingPlan,
    SCHEMA_VERSION,
    StageTimelineEntry,
    get_preset_config,
    load_workflow_config_override,
    merge_workflow_config,
)
from .outputs import (
    RegistrationImage,
    build_stage_artifact_manifest,
    write_atlas_registration_outputs,
    write_self_alignment_outputs,
    write_upsampling_outputs,
)
from .orientation import (
    matrix_orientation_resolution,
    none_orientation_resolution,
    resolve_orientation_init,
    validate_orientation_code,
)
from .provenance import build_reproduce_command, build_run_provenance
from .report import build_registration_report_manifest, write_registration_report
from .targets import (
    EmlddmmTarget,
    load_precomputed_target,
    load_prepared_target,
    resolve_target_source_format,
)

logger = logging.getLogger(__name__)
_REGISTRATION_LOGGER_NAME = "wsi_pipeline.registration"


def _upsample_between_slices_impl(*args, **kwargs):
    from .upsample import upsample_between_slices

    return upsample_between_slices(*args, **kwargs)


@dataclass
class AtlasInputs:
    xI: list[np.ndarray]
    I: np.ndarray
    title: str
    names: list[str]
    xS: list[np.ndarray] | None = None
    S: np.ndarray | None = None


@dataclass
class LoadedInputs:
    config: EmlddmmWorkflowConfig
    backend: EmlddmmBackend
    dataset_root: Path
    registration_output: Path
    device_requested: str
    device_used: str
    stage_controls: dict[str, bool]
    skip_reasons: dict[str, str]
    target: EmlddmmTarget
    xJd: list[np.ndarray]
    Jd: np.ndarray
    Wd: np.ndarray
    target_down: list[int]
    pre_resampling_plan: PreResamplingPlan
    atlas_inputs: AtlasInputs | None = None
    atlas_down: list[int] | None = None
    initial_affine: np.ndarray | None = None
    orientation_resolution: OrientationResolution = field(default_factory=none_orientation_resolution)
    warnings: list[str] | None = None
    transformation_graph_script: Path | None = None
    transformation_graph_script_source: str | None = None
    used_legacy_output_alias: bool = False
    config_override_path: Path | None = None
    original_cli_argv: list[str] | None = None


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


def _summarize_debug_object(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {key: _summarize_debug_object(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize_debug_object(item) for item in value]
    return _jsonify(value)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, indent=2)
    return path


def _merge_warnings(*warning_lists: list[str]) -> list[str]:
    merged: list[str] = []
    for warning_list in warning_lists:
        for warning in warning_list:
            if warning and warning not in merged:
                merged.append(warning)
    return merged


def _attach_registration_log_handler(
    registration_output: Path,
    *,
    verbose: bool,
) -> tuple[Path, logging.Logger, logging.Handler]:
    package_logger = logging.getLogger(_REGISTRATION_LOGGER_NAME)
    package_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    package_logger.propagate = True
    log_path = registration_output / "registration.log"
    for existing in list(package_logger.handlers):
        if getattr(existing, "_registration_log_handler", False):
            package_logger.removeHandler(existing)
            existing.close()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler._registration_log_handler = True  # type: ignore[attr-defined]
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    package_logger.addHandler(handler)
    return log_path, package_logger, handler


def _detach_registration_log_handler(
    package_logger: logging.Logger,
    handler: logging.Handler,
) -> None:
    package_logger.removeHandler(handler)
    handler.close()


def _coalesce_dataset_root(
    dataset_root: str | Path | None,
    output_dir: str | Path | None,
) -> Path:
    resolved_dataset = Path(dataset_root) if dataset_root is not None else None
    resolved_output = Path(output_dir) if output_dir is not None else None
    if resolved_dataset is None and resolved_output is None:
        raise ValueError("step5 requires --dataset-root or legacy --output")
    if (
        resolved_dataset is not None
        and resolved_output is not None
        and resolved_dataset.resolve() != resolved_output.resolve()
    ):
        raise ValueError("--dataset-root and --output must match when both are provided")
    return (resolved_dataset or resolved_output).resolve()


def _resolve_used_legacy_output_alias(
    dataset_root: str | Path | None,
    output_dir: str | Path | None,
    used_legacy_output_alias: bool | None,
) -> bool:
    if used_legacy_output_alias is not None:
        return used_legacy_output_alias
    return dataset_root is None and output_dir is not None


def _resolve_device(requested: str | None) -> tuple[str, str]:
    choice = requested or "auto"
    if choice == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return choice, "cuda:0"
        except Exception:
            pass
        return choice, "cpu"
    if choice.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return choice, choice
        except Exception:
            pass
        return choice, "cpu"
    return choice, choice


def _scale_axes(axes: list[np.ndarray], scale: float) -> list[np.ndarray]:
    return [np.asarray(axis, dtype=np.float32) * np.float32(scale) for axis in axes]


def _axis_spacing_um(axis: np.ndarray) -> float:
    axis_np = np.asarray(axis, dtype=np.float32)
    if axis_np.size < 2:
        return 0.0
    return float(np.median(np.abs(np.diff(axis_np))))


def _axis_spacings_um(axes: list[np.ndarray]) -> list[float]:
    return [_axis_spacing_um(axis) for axis in axes]


def _sectioned_stack_target_resolution_um(
    axes: list[np.ndarray],
    desired_resolution_um: float,
) -> list[float]:
    native_spacing = _axis_spacings_um(axes)
    if not native_spacing:
        return []
    target_resolution = [float(desired_resolution_um)] * len(native_spacing)
    target_resolution[0] = native_spacing[0]
    return target_resolution


def _compute_target_downsampling(
    axes: list[np.ndarray],
    desired_resolution_um: float,
    *,
    per_axis_resolution_um: list[float] | None = None,
    locked_axes: set[int] | frozenset[int] = frozenset(),
) -> list[int]:
    if per_axis_resolution_um is not None and len(per_axis_resolution_um) != len(axes):
        raise ValueError(
            "per_axis_resolution_um must match the number of target axes"
        )
    down: list[int] = []
    locked = set(locked_axes)
    for axis_index, axis in enumerate(axes):
        if axis_index in locked:
            down.append(1)
            continue
        spacing = _axis_spacing_um(axis)
        if spacing <= 0:
            down.append(1)
            continue
        requested_resolution = (
            float(per_axis_resolution_um[axis_index])
            if per_axis_resolution_um is not None
            else float(desired_resolution_um)
        )
        down.append(max(1, int(round(requested_resolution / spacing))))
    return down


def _compute_atlas_downsampling(
    atlas_axes: list[np.ndarray],
    target_axes: list[np.ndarray],
    *,
    locked_axes: set[int] | frozenset[int] = frozenset(),
) -> list[int]:
    down: list[int] = []
    locked = set(locked_axes)
    for axis_index, (atlas_axis, target_axis) in enumerate(
        zip(atlas_axes, target_axes, strict=True)
    ):
        if axis_index in locked:
            down.append(1)
            continue
        atlas_spacing = _axis_spacing_um(atlas_axis)
        target_spacing = _axis_spacing_um(target_axis)
        if atlas_spacing <= 0 or target_spacing <= 0:
            down.append(1)
            continue
        down.append(max(1, int(round(target_spacing / atlas_spacing))))
    return down


def _plan_pre_resampling(
    *,
    policy: str,
    target_axes: list[np.ndarray],
    target_downsampling: list[int],
    target_working_axes: list[np.ndarray],
    atlas_axes: list[np.ndarray] | None = None,
    atlas_downsampling: list[int] | None = None,
) -> PreResamplingPlan:
    target_native_spacing = _axis_spacings_um(target_axes)
    target_working_spacing = _axis_spacings_um(target_working_axes)
    notes: list[str] = []
    target_locked_axes: list[int] = []
    atlas_locked_axes: list[int] = []
    atlas_native_spacing: list[float] | None = None
    atlas_reference_spacing: list[float] | None = None

    if policy == "sectioned-stack":
        target_locked_axes = [0] if target_native_spacing else []
        atlas_locked_axes = [0] if atlas_axes else []
        if target_native_spacing:
            notes.append(
                "sectioned-stack policy preserves target axis 0 and applies desired_resolution_um only to in-plane axes."
            )
    elif policy != "legacy-target-first":
        raise ValueError(f"Unknown resampling policy: {policy!r}")

    if atlas_axes is not None:
        atlas_native_spacing = _axis_spacings_um(atlas_axes)
        atlas_reference_spacing = list(target_working_spacing)
        if atlas_native_spacing and atlas_reference_spacing and atlas_downsampling is not None:
            unchanged_axes = [
                axis_index
                for axis_index, (atlas_spacing, reference_spacing, factor) in enumerate(
                    zip(atlas_native_spacing, atlas_reference_spacing, atlas_downsampling, strict=True)
                )
                if axis_index not in atlas_locked_axes
                and atlas_spacing >= reference_spacing
                and factor == 1
            ]
            if unchanged_axes:
                notes.append(
                    "Atlas native spacing is already coarser than or equal to the target working grid on axes "
                    f"{unchanged_axes}; atlas pre-resampling remains unchanged on those axes."
                )

    return PreResamplingPlan(
        policy=policy,
        target_native_spacing_um=target_native_spacing,
        target_working_spacing_um=target_working_spacing,
        target_locked_axes=target_locked_axes,
        target_downsampling=list(target_downsampling),
        atlas_native_spacing_um=atlas_native_spacing,
        atlas_reference_spacing_um=atlas_reference_spacing,
        atlas_locked_axes=atlas_locked_axes,
        atlas_downsampling=list(atlas_downsampling) if atlas_downsampling is not None else None,
        notes=notes,
    )


def _pre_resampling_payload(loaded: LoadedInputs) -> dict[str, Any]:
    return loaded.pre_resampling_plan.model_dump(mode="python")


def _derive_mu_channels(Jd: np.ndarray) -> tuple[list[list[float]], list[list[float]]]:
    Jd_np = np.asarray(Jd, dtype=np.float32)
    channels = int(Jd_np.shape[0])
    mu_b = float(np.min(Jd_np))
    mu_a = float(np.quantile(Jd_np, 0.999))
    return [[mu_b] * channels], [[mu_a] * channels]


def _resolve_stage_controls(
    config: EmlddmmWorkflowConfig,
) -> tuple[dict[str, bool], dict[str, str]]:
    stage_controls = {
        "self_alignment": bool(
            config.stage_controls.self_alignment_enabled and config.self_alignment.enabled
        ),
        "atlas_registration": bool(
            config.mode == "atlas_to_target"
            and config.stage_controls.atlas_registration_enabled
            and config.atlas_registration.enabled
        ),
        "upsampling": bool(
            config.stage_controls.upsampling_enabled and config.upsampling.enabled
        ),
    }
    skip_reasons: dict[str, str] = {}
    if not stage_controls["self_alignment"]:
        skip_reasons["self_alignment"] = "disabled by configuration"
    if config.mode != "atlas_to_target":
        skip_reasons["atlas_registration"] = "no atlas provided"
    elif not stage_controls["atlas_registration"]:
        skip_reasons["atlas_registration"] = "disabled by configuration"
    if not stage_controls["upsampling"]:
        skip_reasons["upsampling"] = "not requested"
    return stage_controls, skip_reasons


def _validate_mode_specific_args(
    config: EmlddmmWorkflowConfig,
    stage_controls: dict[str, bool],
) -> None:
    if config.orientation_from is not None:
        config.orientation_from = validate_orientation_code(config.orientation_from)
    if config.orientation_to is not None:
        config.orientation_to = validate_orientation_code(config.orientation_to)

    if config.mode == "atlas_free":
        if config.label_path is not None:
            raise ValueError("--label requires --atlas")
        if config.init_affine_path is not None or config.orientation_from or config.orientation_to:
            raise ValueError("--init-affine and orientation flags require --atlas")
        if config.transformation_graph.execute:
            raise ValueError("--run-transformation-graph requires atlas registration")
        return

    if config.atlas_path is None:
        raise ValueError("Atlas mode requires --atlas")
    if bool(config.orientation_from) ^ bool(config.orientation_to):
        raise ValueError(
            "Orientation-derived atlas initialization requires both --orientation-from and --orientation-to."
        )
    if stage_controls["atlas_registration"]:
        has_init = config.init_affine_path is not None
        has_orientation = bool(config.orientation_from and config.orientation_to)
        if not has_init and not has_orientation:
            raise ValueError(
                "Atlas registration requires --init-affine or both "
                "--orientation-from and --orientation-to"
            )
    if config.label_path is not None and config.atlas_path is None:
        raise ValueError("--label requires --atlas")
    if config.transformation_graph.execute and not stage_controls["atlas_registration"]:
        raise ValueError("--run-transformation-graph requires atlas registration")


def _resolve_target(
    source_path: Path,
    source_format: str,
    backend: EmlddmmBackend,
    manifest_path: str | Path | None,
) -> EmlddmmTarget:
    resolved_format = resolve_target_source_format(source_path, source_format)
    if resolved_format == "prepared-dir":
        return load_prepared_target(source_path, backend=backend, manifest_path=manifest_path)
    if manifest_path is None:
        raise ValueError("Precomputed targets require --precomputed-manifest")
    return load_precomputed_target(source_path, manifest_path=manifest_path)


def _load_atlas_inputs(
    backend: EmlddmmBackend,
    config: EmlddmmWorkflowConfig,
) -> AtlasInputs | None:
    if config.atlas_path is None:
        return None
    xI, I, title, names = backend.read_data(str(config.atlas_path))
    xI_scaled = _scale_axes([np.asarray(axis, dtype=np.float32) for axis in xI], config.units.atlas_unit_scale)
    labels_axes = None
    labels_data = None
    if config.label_path is not None:
        labels_axes_raw, labels_data_raw, _label_title, _label_names = backend.read_data(str(config.label_path))
        labels_axes = _scale_axes(
            [np.asarray(axis, dtype=np.float32) for axis in labels_axes_raw],
            config.units.atlas_unit_scale,
        )
        labels_data = np.asarray(labels_data_raw, dtype=np.float32)
    return AtlasInputs(
        xI=xI_scaled,
        I=np.asarray(I, dtype=np.float32),
        title=title,
        names=list(names),
        xS=labels_axes,
        S=labels_data,
    )


def _load_init_affine(
    config: EmlddmmWorkflowConfig,
    backend: EmlddmmBackend,
) -> tuple[np.ndarray | None, OrientationResolution]:
    if config.mode != "atlas_to_target":
        return None, none_orientation_resolution()
    if config.init_affine_path is not None:
        if backend.read_matrix_data is not None:
            affine = np.asarray(backend.read_matrix_data(str(config.init_affine_path)), dtype=np.float32)
        else:
            affine = np.loadtxt(config.init_affine_path, dtype=np.float32)
        return affine, matrix_orientation_resolution()
    if config.orientation_from and config.orientation_to:
        return resolve_orientation_init(
            backend,
            config.orientation_from,
            config.orientation_to,
        )
    return None, none_orientation_resolution()


def _build_self_alignment_kwargs(config: EmlddmmWorkflowConfig) -> dict[str, Any]:
    kwargs = {
        "draw": config.self_alignment.draw,
        "n_steps": config.self_alignment.n_steps,
        "eA2d": config.self_alignment.eA2d,
    }
    kwargs.update(config.self_alignment.extra_kwargs)
    return kwargs


def _build_atlas_registration_kwargs(
    config: EmlddmmWorkflowConfig,
    device_used: str,
    Jd: np.ndarray,
    A2d: np.ndarray,
    initial_affine: np.ndarray,
) -> dict[str, Any]:
    muB, muA = _derive_mu_channels(Jd)
    kwargs = {
        "downI": config.atlas_registration.downI,
        "downJ": config.atlas_registration.downJ,
        "n_iter": config.atlas_registration.n_iter,
        "a": config.atlas_registration.a,
        "dv": config.atlas_registration.dv,
        "slice_matching": config.atlas_registration.slice_matching,
        "v_start": config.atlas_registration.v_start,
        "eA": config.atlas_registration.eA,
        "eA2d": config.atlas_registration.eA2d,
        "ev": config.atlas_registration.ev,
        "local_contrast": config.atlas_registration.local_contrast,
        "up_vector": config.atlas_registration.up_vector,
        "sigmaR": config.atlas_registration.sigmaR,
        "muA": muA,
        "muB": muB,
        "A2d": np.asarray(A2d, dtype=np.float32),
        "A": np.asarray(initial_affine, dtype=np.float32),
        "device": device_used,
    }
    kwargs.update(config.atlas_registration.extra_kwargs)
    return kwargs


def _build_upsampling_kwargs(config: EmlddmmWorkflowConfig) -> dict[str, Any]:
    kwargs = {
        "mode": config.upsampling.mode,
        "nt": config.upsampling.nt,
        "downI": config.upsampling.downI,
        "downJ": config.upsampling.downJ,
        "n_iter": config.upsampling.n_iter,
        "a": config.upsampling.a,
        "dv": config.upsampling.dv,
        "slice_matching": config.upsampling.slice_matching,
        "v_start": config.upsampling.v_start,
        "eA": config.upsampling.eA,
        "eA2d": config.upsampling.eA2d,
        "ev": config.upsampling.ev,
        "local_contrast": config.upsampling.local_contrast,
        "up_vector": config.upsampling.up_vector,
        "sigmaR": config.upsampling.sigmaR,
    }
    kwargs.update(config.upsampling.extra_kwargs)
    return kwargs


def _target_sample_ids(target: EmlddmmTarget) -> list[str]:
    manifest = target.manifest or {}
    entries = manifest.get("entries", [])
    if entries:
        return [str(entry.get("sample_id", f"slice_{idx:04d}")) for idx, entry in enumerate(entries)]
    return [f"slice_{idx:04d}" for idx in range(target.J.shape[1])]


def _build_target_image(target: EmlddmmTarget) -> RegistrationImage:
    return RegistrationImage(
        x=target.xJ,
        data=target.J,
        title="target",
        space="target",
        name="target",
        sample_ids=_target_sample_ids(target),
    )


def _build_working_target(loaded: LoadedInputs) -> EmlddmmTarget:
    return EmlddmmTarget(
        xJ=loaded.xJd,
        J=loaded.Jd,
        W0=loaded.Wd,
        manifest=loaded.target.manifest,
        manifest_path=loaded.target.manifest_path,
        source_format=loaded.target.source_format,
        source_path=loaded.target.source_path,
        present_mask=np.any(loaded.Wd > 0, axis=(1, 2)),
    )


def _build_atlas_image(atlas: AtlasInputs) -> RegistrationImage:
    return RegistrationImage(
        x=atlas.xI,
        data=atlas.I,
        title=atlas.title,
        space="atlas",
        name="atlas",
    )


def _build_transformation_graph(
    atlas_path: Path,
    target_source: Path,
    config_path: Path,
    label_path: Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "space_image_path": str(atlas_path),
        "target_space_path": str(target_source),
        "output": "output",
        "configs": [str(config_path)],
    }
    if label_path is not None:
        payload["space_label_path"] = str(label_path)
    return payload


def _build_transformation_graph_execution_config(
    atlas_path: Path,
    target_source: Path,
    config_path: Path,
    label_path: Path | None = None,
) -> dict[str, Any]:
    payload = {
        "output": "output",
        "space_image_path": [
            ["Atlas", "image", str(atlas_path.resolve())],
            ["Target", "image", str(target_source.resolve())],
        ],
        "registrations": [[["Atlas", "image"], ["Target", "image"]]],
        "configs": [str(config_path.resolve())],
        "transform_all": True,
    }
    if label_path is not None:
        payload["space_image_path"].insert(1, ["Atlas", "labels", str(label_path.resolve())])
    return payload


def _apply_atlas_registration_normalization(
    array: np.ndarray,
    config: EmlddmmWorkflowConfig,
) -> np.ndarray:
    array_np = np.asarray(array, dtype=np.float32)
    if config.normalization.atlas_registration_input != "mean_abs":
        return array_np
    denom = float(np.mean(np.abs(array_np)))
    if denom <= 0:
        return array_np
    return array_np / denom


def _stage_dir(registration_output: Path, stage_name: str) -> Path:
    return registration_output / stage_name


def _expected_stage_outputs(
    registration_output: Path,
    include_atlas: bool,
    include_upsampling: bool,
    include_report: bool,
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "root": [
            str(registration_output / "resolved_run_plan.json"),
            str(registration_output / "registration_summary.json"),
            str(registration_output / "registration.log"),
            str(registration_output / "run_provenance.json"),
            str(registration_output / "reproduce_step5_command.txt"),
        ],
        "self_alignment": [
            str(_stage_dir(registration_output, "self_alignment") / "inputs.json"),
            str(_stage_dir(registration_output, "self_alignment") / "self_alignment_config.json"),
            str(_stage_dir(registration_output, "self_alignment") / "effective_config.json"),
            str(_stage_dir(registration_output, "self_alignment") / "artifacts.json"),
        ],
    }
    if include_report:
        expected["root"].extend(
            [
                str(registration_output / "registration_report.json"),
                str(registration_output / "registration_report.html"),
            ]
        )
    if include_atlas:
        expected["atlas_registration"] = [
            str(_stage_dir(registration_output, "atlas_registration") / "inputs.json"),
            str(_stage_dir(registration_output, "atlas_registration") / "atlas_to_target_config.json"),
            str(_stage_dir(registration_output, "atlas_registration") / "effective_config.json"),
            str(_stage_dir(registration_output, "atlas_registration") / "transformation_graph_config.json"),
            str(
                _stage_dir(registration_output, "atlas_registration")
                / "transformation_graph_execution_config.json"
            ),
            str(_stage_dir(registration_output, "atlas_registration") / "registration_data.npy"),
            str(_stage_dir(registration_output, "atlas_registration") / "artifacts.json"),
        ]
    if include_upsampling:
        expected["upsampling"] = [
            str(_stage_dir(registration_output, "upsampling") / "inputs.json"),
            str(_stage_dir(registration_output, "upsampling") / "effective_config.json"),
            str(_stage_dir(registration_output, "upsampling") / "artifacts.json"),
            str(_stage_dir(registration_output, "upsampling") / "filled_volume.vtk"),
            str(_stage_dir(registration_output, "upsampling") / "nearest_slice_reference.vtk"),
            str(_stage_dir(registration_output, "upsampling") / "filled_volume_overview.png"),
            str(
                _stage_dir(registration_output, "upsampling")
                / "nearest_slice_reference_overview.png"
            ),
        ]
    return expected


def _atlas_init_mode(
    config: EmlddmmWorkflowConfig,
    stage_controls: dict[str, bool],
) -> str:
    if not stage_controls["atlas_registration"]:
        return "none"
    if config.init_affine_path is not None:
        return "matrix"
    if config.orientation_from and config.orientation_to:
        return "orientation"
    return "none"


def _enumerate_files(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(str(path) for path in directory.rglob("*") if path.is_file())


def _write_stage_inputs(stage_dir: Path, payload: dict[str, Any]) -> Path:
    return _write_json(stage_dir / "inputs.json", payload)


def _write_stage_artifacts(stage_dir: Path, payload: dict[str, Any]) -> Path:
    return _write_json(stage_dir / "artifacts.json", payload)


def _find_workspace_transformation_graph_script() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "transformation_graph_v01.py"
        if candidate.exists():
            return candidate
    return None


def _resolve_transformation_graph_script(
    config: EmlddmmWorkflowConfig,
    backend: EmlddmmBackend,
) -> tuple[Path | None, list[str], str | None]:
    warnings: list[str] = []
    explicit_path = config.transformation_graph.script_path
    if explicit_path is not None:
        candidate = Path(explicit_path).resolve()
        logger.debug("Checking explicit transformation graph script path: %s", candidate)
        if not candidate.exists():
            raise FileNotFoundError(
                "Transformation graph execution was requested, but the configured script path "
                f"does not exist: {candidate}. Install the external emlddmm package or pass "
                "--transformation-graph-script with a valid path."
            )
        return candidate, warnings, "explicit override path"

    candidates: list[Path] = []
    module_file = getattr(backend.module, "__file__", None)
    if module_file is not None:
        candidates.append(Path(module_file).resolve().parent / "transformation_graph_v01.py")
    module_paths = getattr(backend.module, "__path__", None)
    if module_paths:
        candidates.extend(Path(path).resolve() / "transformation_graph_v01.py" for path in module_paths)

    for module_name in ("emlddmm.transformation_graph_v01", "transformation_graph_v01", "emlddmm"):
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            continue
        if spec.origin and module_name != "emlddmm":
            candidates.append(Path(spec.origin).resolve())
        if spec.origin and module_name == "emlddmm":
            candidates.append(Path(spec.origin).resolve().parent / "transformation_graph_v01.py")
        if spec.submodule_search_locations:
            candidates.extend(
                Path(path).resolve() / "transformation_graph_v01.py"
                for path in spec.submodule_search_locations
            )

    workspace_fallback = _find_workspace_transformation_graph_script()
    if workspace_fallback is not None:
        candidates.append(workspace_fallback.resolve())

    seen: set[Path] = set()
    ordered_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            ordered_candidates.append(candidate)
            seen.add(candidate)
    logger.debug(
        "Transformation graph script candidates: %s",
        [str(candidate) for candidate in ordered_candidates],
    )
    for candidate in ordered_candidates:
        if candidate.exists():
            if workspace_fallback is not None and candidate == workspace_fallback.resolve():
                warnings.append(
                    "Using workspace-local transformation_graph_v01.py fallback; install the external "
                    "emlddmm package for production transformation-graph runs."
                )
                return candidate, warnings, "workspace-local development fallback"
            return candidate, warnings, "external emlddmm package"

    if config.transformation_graph.execute:
        raise FileNotFoundError(
            "Transformation graph execution was requested, but transformation_graph_v01.py could not "
            "be found in the external emlddmm package or via --transformation-graph-script."
        )
    return None, warnings, None


def _execute_transformation_graph(
    script_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, str]:
    if script_path is None:
        raise FileNotFoundError(
            "Transformation graph execution was requested, but transformation_graph_v01.py could not be resolved."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(script_path), "--infile", str(config_path)]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    log_path = output_dir / "outputs.txt"
    log_path.write_text((completed.stdout or "") + (completed.stderr or ""), encoding="utf-8")
    return {"output_dir": str(output_dir), "log": str(log_path)}


def _build_workflow_config(
    *,
    dataset_root: str | Path | None,
    output_dir: str | Path | None,
    target_source: str | Path | None,
    target_source_format: str,
    registration_output: str | Path | None,
    atlas: str | Path | None,
    label: str | Path | None,
    emlddmm_config: str | Path | None,
    preset: str,
    init_affine: str | Path | None,
    orientation_from: str | None,
    orientation_to: str | None,
    atlas_unit_scale: float | None,
    target_unit_scale: float | None,
    device: str | None,
    precomputed_manifest: str | Path | None,
    upsample_between_slices: bool | None,
    upsample_mode: str | None,
    skip_self_alignment: bool,
    run_transformation_graph: bool,
    transformation_graph_script: str | Path | None,
    write_notebook_bundle: bool,
    write_qc_report: bool,
    used_legacy_output_alias: bool | None = None,
) -> tuple[Path, EmlddmmWorkflowConfig, bool]:
    resolved_dataset_root = _coalesce_dataset_root(dataset_root, output_dir)
    resolved_used_legacy_output_alias = _resolve_used_legacy_output_alias(
        dataset_root,
        output_dir,
        used_legacy_output_alias,
    )
    config = get_preset_config(preset)
    config = merge_workflow_config(config, load_workflow_config_override(emlddmm_config))

    if target_source is not None:
        config.target_source.path = Path(target_source)
    elif config.target_source.path is None:
        config.target_source.path = resolved_dataset_root
    config.target_source.format = target_source_format

    if precomputed_manifest is not None:
        config.target_source.manifest_path = Path(precomputed_manifest)
    if registration_output is not None:
        registration_output_path = Path(registration_output)
    else:
        registration_output_path = resolved_dataset_root / "emlddmm"
    if atlas is not None:
        config.atlas_path = Path(atlas)
    else:
        config.atlas_path = None
    if label is not None:
        config.label_path = Path(label)
    else:
        config.label_path = None
    if init_affine is not None:
        config.init_affine_path = Path(init_affine)
    else:
        config.init_affine_path = None
    config.orientation_from = orientation_from
    config.orientation_to = orientation_to
    if atlas_unit_scale is not None:
        config.units.atlas_unit_scale = atlas_unit_scale
    if target_unit_scale is not None:
        config.units.target_unit_scale = target_unit_scale
    if device is not None:
        config.device = device
    if skip_self_alignment:
        config.stage_controls.self_alignment_enabled = False
        config.self_alignment.enabled = False
    if upsample_between_slices is not None:
        config.stage_controls.upsampling_enabled = upsample_between_slices
        config.upsampling.enabled = upsample_between_slices
    if upsample_mode is not None:
        config.upsampling.mode = upsample_mode
    if run_transformation_graph:
        config.transformation_graph.execute = True
    if transformation_graph_script is not None:
        config.transformation_graph.script_path = Path(transformation_graph_script)
    if write_notebook_bundle:
        config.debug.write_notebook_bundle = True
    if write_qc_report:
        config.outputs.write_qc_report = True
    config.mode = "atlas_to_target" if config.atlas_path is not None else "atlas_free"
    logger.debug(
        "Resolved step5 workflow config preset=%s mode=%s target_format=%s",
        config.preset,
        config.mode,
        config.target_source.format,
    )
    return resolved_dataset_root, config, resolved_used_legacy_output_alias


def _load_inputs_for_plan(
    dataset_root: Path,
    registration_output: Path,
    config: EmlddmmWorkflowConfig,
    backend: EmlddmmBackend,
    *,
    used_legacy_output_alias: bool = False,
    config_override_path: Path | None = None,
    original_cli_argv: list[str] | None = None,
) -> LoadedInputs:
    stage_controls, skip_reasons = _resolve_stage_controls(config)
    _validate_mode_specific_args(config, stage_controls)

    device_requested, device_used = _resolve_device(config.device)
    if not stage_controls["atlas_registration"]:
        device_used = "cpu"

    source_path = Path(config.target_source.path or dataset_root).resolve()
    target = _resolve_target(
        source_path,
        config.target_source.format,
        backend,
        config.target_source.manifest_path,
    )
    target.xJ = _scale_axes(target.xJ, config.units.target_unit_scale)
    if config.resampling.policy == "sectioned-stack":
        target_locked_axes = frozenset({0}) if target.xJ else frozenset()
        target_resolution_um = _sectioned_stack_target_resolution_um(
            target.xJ,
            config.units.desired_resolution_um,
        )
    elif config.resampling.policy == "legacy-target-first":
        target_locked_axes = frozenset()
        target_resolution_um = None
    else:
        raise ValueError(f"Unknown resampling policy: {config.resampling.policy!r}")
    target_down = _compute_target_downsampling(
        target.xJ,
        config.units.desired_resolution_um,
        per_axis_resolution_um=target_resolution_um,
        locked_axes=target_locked_axes,
    )
    downsampled = backend.downsample_image_domain(target.xJ, target.J, target_down, W=target.W0)
    xJd, Jd, Wd = downsampled
    xJd = [np.asarray(axis, dtype=np.float32) for axis in xJd]
    Jd = np.asarray(Jd, dtype=np.float32)
    Wd = np.asarray(Wd, dtype=np.float32)
    logger.debug(
        "Loaded target source=%s format=%s shape=%s mask_shape=%s present_slices=%d",
        target.source_path,
        target.source_format,
        tuple(target.J.shape),
        tuple(target.W0.shape),
        int(np.count_nonzero(target.present_mask)),
    )

    atlas_inputs = None
    atlas_down = None
    initial_affine = None
    orientation_resolution = none_orientation_resolution()
    atlas_locked_axes = frozenset()
    if config.atlas_path is not None:
        atlas_inputs = _load_atlas_inputs(backend, config)
        if atlas_inputs is not None:
            atlas_locked_axes = (
                frozenset({0}) if config.resampling.policy == "sectioned-stack" else frozenset()
            )
            atlas_down = _compute_atlas_downsampling(
                atlas_inputs.xI,
                xJd,
                locked_axes=atlas_locked_axes,
            )
        initial_affine, orientation_resolution = _load_init_affine(config, backend)
    pre_resampling_plan = _plan_pre_resampling(
        policy=config.resampling.policy,
        target_axes=target.xJ,
        target_downsampling=target_down,
        target_working_axes=xJd,
        atlas_axes=atlas_inputs.xI if atlas_inputs is not None else None,
        atlas_downsampling=atlas_down,
    )

    warnings: list[str] = []
    default_unit_warnings = []
    if config.atlas_path is not None and config.units.atlas_unit_scale == 1000.0:
        default_unit_warnings.append(
            "Using default atlas_unit_scale=1000.0; atlas coordinates are assumed to be millimeters."
        )
    if config.units.target_unit_scale == 1.0:
        default_unit_warnings.append(
            "Using default target_unit_scale=1.0; target coordinates are assumed to already be micrometers."
        )
    warnings = _merge_warnings(warnings, default_unit_warnings)
    if config.resampling.policy == "legacy-target-first":
        warnings.append(
            "Using legacy-target-first resampling policy; this compatibility mode may downsample the target slice axis and differs from sectioned-stack notebook practice."
        )
    if used_legacy_output_alias:
        warnings.append("Deprecated --output alias was used for step5; prefer --dataset-root.")
    if stage_controls["atlas_registration"] and config.label_path is None:
        warnings.append(
            "Atlas-registration QC overlays will omit labels because no --label path was supplied."
        )
    transformation_graph_script = None
    transformation_graph_script_source = None
    if config.transformation_graph.execute or config.transformation_graph.script_path is not None:
        (
            transformation_graph_script,
            graph_warnings,
            transformation_graph_script_source,
        ) = _resolve_transformation_graph_script(
            config,
            backend,
        )
        warnings = _merge_warnings(warnings, graph_warnings)
        if getattr(backend, "origin_type", None) == "vendored" and config.transformation_graph.execute:
            warnings.append(
                "Transformation-graph execution is using the vendored registration backend; install "
                "the external emlddmm package for the canonical script location."
            )
    logger.debug(
        "Working grid target_down=%s atlas_down=%s transformation_graph_script=%s",
        target_down,
        atlas_down,
        transformation_graph_script,
    )

    return LoadedInputs(
        config=config,
        backend=backend,
        dataset_root=dataset_root,
        registration_output=registration_output,
        device_requested=device_requested,
        device_used=device_used,
        stage_controls=stage_controls,
        skip_reasons=skip_reasons,
        target=target,
        xJd=xJd,
        Jd=Jd,
        Wd=Wd,
        target_down=target_down,
        pre_resampling_plan=pre_resampling_plan,
        atlas_inputs=atlas_inputs,
        atlas_down=atlas_down,
        initial_affine=initial_affine,
        orientation_resolution=orientation_resolution,
        warnings=warnings,
        transformation_graph_script=transformation_graph_script,
        transformation_graph_script_source=transformation_graph_script_source,
        used_legacy_output_alias=used_legacy_output_alias,
        config_override_path=config_override_path,
        original_cli_argv=original_cli_argv,
    )


def _build_resolved_plan(loaded: LoadedInputs, dry_run: bool) -> EmlddmmResolvedPlan:
    registration_output = loaded.registration_output
    expected_outputs = _expected_stage_outputs(
        registration_output,
        include_atlas=loaded.stage_controls["atlas_registration"],
        include_upsampling=loaded.stage_controls["upsampling"],
        include_report=loaded.config.outputs.write_qc_report,
    )
    stage_defs = {
        "self_alignment": EmlddmmStagePlan(
            name="self_alignment",
            enabled=loaded.stage_controls["self_alignment"],
            reason=loaded.skip_reasons.get("self_alignment"),
            output_dir=_stage_dir(registration_output, "self_alignment"),
            expected_outputs=_expected_stage_outputs(
                registration_output,
                False,
                False,
                False,
            )["self_alignment"],
        ),
        "atlas_registration": EmlddmmStagePlan(
            name="atlas_registration",
            enabled=loaded.stage_controls["atlas_registration"],
            reason=loaded.skip_reasons.get("atlas_registration"),
            output_dir=_stage_dir(registration_output, "atlas_registration"),
            expected_outputs=_expected_stage_outputs(
                registration_output,
                True,
                False,
                False,
            ).get("atlas_registration", []),
        ),
        "upsampling": EmlddmmStagePlan(
            name="upsampling",
            enabled=loaded.stage_controls["upsampling"],
            reason=loaded.skip_reasons.get("upsampling"),
            output_dir=_stage_dir(registration_output, "upsampling"),
            expected_outputs=_expected_stage_outputs(
                registration_output,
                False,
                True,
                False,
            ).get("upsampling", []),
        ),
    }
    enabled_stages = [name for name, enabled in loaded.stage_controls.items() if enabled]
    skipped = {
        name: reason
        for name, reason in loaded.skip_reasons.items()
        if not loaded.stage_controls.get(name, False)
    }
    return EmlddmmResolvedPlan(
        schema_version=SCHEMA_VERSION,
        mode=loaded.config.mode or "atlas_free",
        preset=loaded.config.preset,
        dataset_root=loaded.dataset_root,
        registration_output=registration_output,
        target_source=loaded.target.source_path,
        target_source_format=loaded.target.source_format,
        manifest_path=loaded.target.manifest_path,
        atlas_path=loaded.config.atlas_path,
        label_path=loaded.config.label_path,
        device_requested=loaded.device_requested,
        device_used=loaded.device_used,
        backend_name=loaded.backend.name,
        atlas_init_mode=_atlas_init_mode(loaded.config, loaded.stage_controls),
        atlas_unit_scale=loaded.config.units.atlas_unit_scale,
        target_unit_scale=loaded.config.units.target_unit_scale,
        working_resolution_um=loaded.config.units.desired_resolution_um,
        target_downsampling=loaded.target_down,
        atlas_downsampling=loaded.atlas_down,
        pre_resampling_plan=loaded.pre_resampling_plan,
        enabled_stages=enabled_stages,
        skipped_stages=skipped,
        warnings=list(loaded.warnings or []),
        transformation_graph_script=loaded.transformation_graph_script,
        log_path=registration_output / "registration.log",
        report_manifest_path=(
            registration_output / "registration_report.json"
            if loaded.config.outputs.write_qc_report
            else None
        ),
        provenance_path=registration_output / "run_provenance.json",
        orientation_resolution=loaded.orientation_resolution,
        used_legacy_output_alias=loaded.used_legacy_output_alias,
        dry_run=dry_run,
        stages=stage_defs,
        expected_outputs=expected_outputs,
        workflow_config=loaded.config.model_dump(mode="python"),
    )


def _stage_input_payload(loaded: LoadedInputs, stage_name: str) -> dict[str, Any]:
    base = {
        "mode": loaded.config.mode,
        "target_source": loaded.target.source_path,
        "target_source_format": loaded.target.source_format,
        "manifest_path": loaded.target.manifest_path,
        "working_resolution_um": loaded.config.units.desired_resolution_um,
        "target_downsampling": loaded.target_down,
        "pre_resampling_plan": _pre_resampling_payload(loaded),
    }
    if stage_name == "self_alignment":
        base["n_slices"] = int(loaded.Jd.shape[1])
    elif stage_name == "atlas_registration":
        base["atlas_path"] = loaded.config.atlas_path
        base["label_path"] = loaded.config.label_path
        base["atlas_downsampling"] = loaded.atlas_down
        base["atlas_init_mode"] = _atlas_init_mode(loaded.config, loaded.stage_controls)
        base["orientation_resolution"] = loaded.orientation_resolution.model_dump(mode="python")
    elif stage_name == "upsampling":
        base["mode"] = loaded.config.upsampling.mode
    return base


def _timeline_entry(
    stage_name: str,
    *,
    enabled: bool,
    status: str,
    output_dir: Path,
    reason: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    duration_seconds: float | None = None,
) -> StageTimelineEntry:
    return StageTimelineEntry(
        name=stage_name,
        enabled=enabled,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration_seconds,
        reason=reason,
        output_dir=output_dir,
    )


def _artifact_path_from_value(value: Any) -> list[Path]:
    paths: list[Path] = []
    if isinstance(value, str) and value:
        candidate = Path(value)
        if candidate.exists() and candidate.is_file():
            paths.append(candidate)
    elif isinstance(value, Path) and value.exists() and value.is_file():
        paths.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            paths.extend(_artifact_path_from_value(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_artifact_path_from_value(item))
    return paths


def _collect_generated_control_files(
    *,
    registration_output_dir: Path,
    artifacts: dict[str, Any],
) -> list[Path]:
    candidates = [
        registration_output_dir / "resolved_run_plan.json",
        registration_output_dir / "registration_summary.json",
        registration_output_dir / "registration_report.json",
        registration_output_dir / "reproduce_step5_command.txt",
        registration_output_dir / "run_provenance.json",
    ]
    for item in artifacts.values():
        candidates.extend(_artifact_path_from_value(item))
    files: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_file() and resolved not in seen:
            files.append(resolved)
            seen.add(resolved)
    return files


def _write_notebook_bundle(
    registration_output: Path,
    plan: EmlddmmResolvedPlan,
    stage_payloads: dict[str, Any],
) -> Path:
    payload = {
        "resolved_plan": plan.model_dump(mode="python"),
        "stage_payloads": _summarize_debug_object(stage_payloads),
    }
    return _write_json(registration_output / "debug" / "notebook_bundle.json", payload)


def plan_emlddmm_workflow(
    *,
    plan: EmlddmmResolvedPlan | None = None,
    dataset_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    target_source: str | Path | None = None,
    target_source_format: str = "auto",
    registration_output: str | Path | None = None,
    atlas: str | Path | None = None,
    label: str | Path | None = None,
    emlddmm_config: str | Path | None = None,
    preset: str = "macaque-notebook",
    init_affine: str | Path | None = None,
    orientation_from: str | None = None,
    orientation_to: str | None = None,
    atlas_unit_scale: float | None = None,
    target_unit_scale: float | None = None,
    device: str | None = None,
    precomputed_manifest: str | Path | None = None,
    upsample_between_slices: bool | None = None,
    upsample_mode: str | None = None,
    skip_self_alignment: bool = False,
    run_transformation_graph: bool = False,
    transformation_graph_script: str | Path | None = None,
    write_notebook_bundle: bool = False,
    write_qc_report: bool = False,
    used_legacy_output_alias: bool | None = None,
    original_cli_argv: list[str] | None = None,
    backend: EmlddmmBackend | None = None,
    dry_run: bool = False,
) -> EmlddmmResolvedPlan:
    if plan is not None:
        return plan
    resolved_dataset_root, config, resolved_used_legacy_output_alias = _build_workflow_config(
        dataset_root=dataset_root,
        output_dir=output_dir,
        target_source=target_source,
        target_source_format=target_source_format,
        registration_output=registration_output,
        atlas=atlas,
        label=label,
        emlddmm_config=emlddmm_config,
        preset=preset,
        init_affine=init_affine,
        orientation_from=orientation_from,
        orientation_to=orientation_to,
        atlas_unit_scale=atlas_unit_scale,
        target_unit_scale=target_unit_scale,
        device=device,
        precomputed_manifest=precomputed_manifest,
        upsample_between_slices=upsample_between_slices,
        upsample_mode=upsample_mode,
        skip_self_alignment=skip_self_alignment,
        run_transformation_graph=run_transformation_graph,
        transformation_graph_script=transformation_graph_script,
        write_notebook_bundle=write_notebook_bundle,
        write_qc_report=write_qc_report,
        used_legacy_output_alias=used_legacy_output_alias,
    )
    backend_obj = backend or resolve_emlddmm_backend()
    registration_output_path = Path(registration_output) if registration_output is not None else resolved_dataset_root / "emlddmm"
    loaded = _load_inputs_for_plan(
        resolved_dataset_root,
        registration_output_path.resolve(),
        config,
        backend_obj,
        used_legacy_output_alias=resolved_used_legacy_output_alias,
        config_override_path=Path(emlddmm_config).resolve() if emlddmm_config is not None else None,
        original_cli_argv=original_cli_argv,
    )
    return _build_resolved_plan(loaded, dry_run)


def run_emlddmm_workflow(
    *,
    plan: EmlddmmResolvedPlan | None = None,
    dataset_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    target_source: str | Path | None = None,
    target_source_format: str = "auto",
    registration_output: str | Path | None = None,
    atlas: str | Path | None = None,
    label: str | Path | None = None,
    emlddmm_config: str | Path | None = None,
    preset: str = "macaque-notebook",
    init_affine: str | Path | None = None,
    orientation_from: str | None = None,
    orientation_to: str | None = None,
    atlas_unit_scale: float | None = None,
    target_unit_scale: float | None = None,
    device: str | None = None,
    precomputed_manifest: str | Path | None = None,
    upsample_between_slices: bool | None = None,
    upsample_mode: str | None = None,
    skip_self_alignment: bool = False,
    run_transformation_graph: bool = False,
    transformation_graph_script: str | Path | None = None,
    write_notebook_bundle: bool = False,
    write_qc_report: bool = False,
    used_legacy_output_alias: bool | None = None,
    original_cli_argv: list[str] | None = None,
    backend: EmlddmmBackend | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> EmlddmmWorkflowResult:
    started = time.perf_counter()
    backend_obj = backend or resolve_emlddmm_backend()
    if plan is None:
        resolved_dataset_root = _coalesce_dataset_root(dataset_root, output_dir)
        registration_output_dir = (
            Path(registration_output).resolve()
            if registration_output is not None
            else (resolved_dataset_root / "emlddmm").resolve()
        )
    else:
        registration_output_dir = Path(plan.registration_output).resolve()
    registration_output_dir.mkdir(parents=True, exist_ok=True)

    log_path, package_logger, file_handler = _attach_registration_log_handler(
        registration_output_dir,
        verbose=verbose,
    )
    try:
        logger.info("Starting EM-LDDMM workflow planning")
        logger.info("Resolved registration backend: %s", backend_obj.name)
        if _resolve_used_legacy_output_alias(dataset_root, output_dir, used_legacy_output_alias):
            logger.warning("Deprecated --output alias was used for step5; prefer --dataset-root.")

        if plan is None:
            (
                resolved_dataset_root,
                config,
                resolved_used_legacy_output_alias,
            ) = _build_workflow_config(
                dataset_root=dataset_root,
                output_dir=output_dir,
                target_source=target_source,
                target_source_format=target_source_format,
                registration_output=registration_output,
                atlas=atlas,
                label=label,
                emlddmm_config=emlddmm_config,
                preset=preset,
                init_affine=init_affine,
                orientation_from=orientation_from,
                orientation_to=orientation_to,
                atlas_unit_scale=atlas_unit_scale,
                target_unit_scale=target_unit_scale,
                device=device,
                precomputed_manifest=precomputed_manifest,
                upsample_between_slices=upsample_between_slices,
                upsample_mode=upsample_mode,
                skip_self_alignment=skip_self_alignment,
                run_transformation_graph=run_transformation_graph,
                transformation_graph_script=transformation_graph_script,
                write_notebook_bundle=write_notebook_bundle,
                write_qc_report=write_qc_report,
                used_legacy_output_alias=used_legacy_output_alias,
            )
            loaded = _load_inputs_for_plan(
                resolved_dataset_root,
                registration_output_dir,
                config,
                backend_obj,
                used_legacy_output_alias=resolved_used_legacy_output_alias,
                config_override_path=Path(emlddmm_config).resolve() if emlddmm_config is not None else None,
                original_cli_argv=original_cli_argv,
            )
            resolved_plan = _build_resolved_plan(loaded, dry_run)
        else:
            config = EmlddmmWorkflowConfig.model_validate(plan.workflow_config)
            loaded = _load_inputs_for_plan(
                Path(plan.dataset_root),
                registration_output_dir,
                config,
                backend_obj,
                used_legacy_output_alias=plan.used_legacy_output_alias,
                config_override_path=None,
                original_cli_argv=original_cli_argv,
            )
            resolved_plan = _build_resolved_plan(loaded, plan.dry_run or dry_run)

        logger.info(
            "Target source resolved to %s (%s)",
            loaded.target.source_path,
            loaded.target.source_format,
        )
        logger.info(
            "Atlas mode=%s init_mode=%s",
            resolved_plan.mode,
            resolved_plan.atlas_init_mode,
        )
        logger.info(
            "Units atlas=%s target=%s working_resolution_um=%s",
            resolved_plan.atlas_unit_scale,
            resolved_plan.target_unit_scale,
            resolved_plan.working_resolution_um,
        )
        logger.info(
            "Downsampling target=%s atlas=%s",
            resolved_plan.target_downsampling,
            resolved_plan.atlas_downsampling,
        )
        logger.info(
            "Pre-resampling policy=%s target_native_spacing_um=%s target_working_spacing_um=%s target_locked_axes=%s",
            loaded.pre_resampling_plan.policy,
            loaded.pre_resampling_plan.target_native_spacing_um,
            loaded.pre_resampling_plan.target_working_spacing_um,
            loaded.pre_resampling_plan.target_locked_axes,
        )
        if loaded.pre_resampling_plan.atlas_native_spacing_um is not None:
            logger.info(
                "Atlas pre-resampling native_spacing_um=%s reference_spacing_um=%s locked_axes=%s",
                loaded.pre_resampling_plan.atlas_native_spacing_um,
                loaded.pre_resampling_plan.atlas_reference_spacing_um,
                loaded.pre_resampling_plan.atlas_locked_axes,
            )
        for note in loaded.pre_resampling_plan.notes:
            logger.info("Pre-resampling note: %s", note)
        if loaded.transformation_graph_script is not None:
            logger.info(
                "Resolved transformation-graph script: %s (%s)",
                loaded.transformation_graph_script,
                loaded.transformation_graph_script_source,
            )
        for warning in resolved_plan.warnings:
            logger.warning(warning)

        plan_path = _write_json(
            registration_output_dir / "resolved_run_plan.json",
            resolved_plan.model_dump(mode="python"),
        )
        logger.info("Wrote resolved plan -> %s", plan_path)

        stage_results: dict[str, Any] = {}
        artifacts: dict[str, Any] = {}
        stage_payloads: dict[str, Any] = {}
        completed_stages: list[str] = []
        timings: dict[str, float] = {}
        stage_timeline: list[StageTimelineEntry] = []
        self_alignment_result: dict[str, Any] | None = None
        summary_path = registration_output_dir / "registration_summary.json"
        report_manifest_path: Path | None = (
            registration_output_dir / "registration_report.json"
            if loaded.config.outputs.write_qc_report
            else None
        )
        report_path: Path | None = (
            registration_output_dir / "registration_report.html"
            if loaded.config.outputs.write_qc_report
            else None
        )
        provenance_path = registration_output_dir / "run_provenance.json"
        reproduce_command_path = registration_output_dir / "reproduce_step5_command.txt"

        if resolved_plan.dry_run:
            logger.info("Dry run requested; stages will not execute")
            for stage_name, stage_plan in resolved_plan.stages.items():
                status = "planned" if stage_plan.enabled else "skipped"
                stage_results[stage_name] = {
                    "status": status,
                    "reason": stage_plan.reason,
                }
                stage_timeline.append(
                    _timeline_entry(
                        stage_name,
                        enabled=stage_plan.enabled,
                        status=status,
                        output_dir=stage_plan.output_dir or _stage_dir(registration_output_dir, stage_name),
                        reason=stage_plan.reason,
                    )
                )
        else:
            working_target = _build_working_target(loaded)

            if loaded.stage_controls["self_alignment"]:
                logger.info("Starting stage: self_alignment")
                stage_started = time.perf_counter()
                stage_started_at = datetime.now(UTC).isoformat()
                stage_dir = _stage_dir(registration_output_dir, "self_alignment")
                self_kwargs = _build_self_alignment_kwargs(loaded.config)
                stage_payloads["self_alignment"] = self_kwargs
                _write_stage_inputs(stage_dir, _stage_input_payload(loaded, "self_alignment"))
                self_alignment_result = loaded.backend.atlas_free_reconstruction(
                    xJ=loaded.xJd,
                    J=loaded.Jd,
                    W=(loaded.Wd > 0).astype(np.float32),
                    **self_kwargs,
                )
                stage_artifacts = write_self_alignment_outputs(
                    backend=loaded.backend,
                    output_dir=stage_dir,
                    target=working_target,
                    self_alignment=self_alignment_result,
                    stage_config=self_kwargs,
                    effective_config={
                        "mode": loaded.config.mode,
                        "device_used": loaded.device_used,
                        "target_downsampling": loaded.target_down,
                        "resampling_policy": loaded.pre_resampling_plan.policy,
                        "pre_resampling_plan": _pre_resampling_payload(loaded),
                    },
                )
                stage_artifacts["artifacts"] = str(
                    _write_stage_artifacts(stage_dir, build_stage_artifact_manifest(stage_dir))
                )
                artifacts["self_alignment"] = stage_artifacts
                stage_results["self_alignment"] = {
                    "status": "completed",
                    "artifacts": stage_artifacts,
                }
                completed_stages.append("self_alignment")
                timings["self_alignment_seconds"] = round(
                    time.perf_counter() - stage_started,
                    6,
                )
                stage_timeline.append(
                    _timeline_entry(
                        "self_alignment",
                        enabled=True,
                        status="completed",
                        output_dir=stage_dir,
                        started_at=stage_started_at,
                        ended_at=datetime.now(UTC).isoformat(),
                        duration_seconds=timings["self_alignment_seconds"],
                    )
                )
                logger.info(
                    "Completed stage: self_alignment in %.3fs",
                    timings["self_alignment_seconds"],
                )
                logger.info("Wrote self_alignment artifacts -> %s", stage_dir)
            else:
                logger.info(
                    "Skipping stage: self_alignment (%s)",
                    loaded.skip_reasons.get("self_alignment"),
                )
                a2d = np.repeat(np.eye(3, dtype=np.float32)[None], loaded.Jd.shape[1], axis=0)
                self_alignment_result = {"A2d": a2d, "I": loaded.Jd, "Jr": loaded.Jd}
                stage_results["self_alignment"] = {
                    "status": "skipped",
                    "reason": loaded.skip_reasons.get("self_alignment"),
                }
                stage_timeline.append(
                    _timeline_entry(
                        "self_alignment",
                        enabled=False,
                        status="skipped",
                        output_dir=_stage_dir(registration_output_dir, "self_alignment"),
                        reason=loaded.skip_reasons.get("self_alignment"),
                    )
                )

            if loaded.stage_controls["atlas_registration"]:
                logger.info("Starting stage: atlas_registration")
                stage_started = time.perf_counter()
                stage_started_at = datetime.now(UTC).isoformat()
                stage_dir = _stage_dir(registration_output_dir, "atlas_registration")
                atlas_inputs = loaded.atlas_inputs
                assert atlas_inputs is not None
                assert loaded.initial_affine is not None
                assert self_alignment_result is not None
                _write_stage_inputs(stage_dir, _stage_input_payload(loaded, "atlas_registration"))
                xId, Id = loaded.backend.downsample_image_domain(
                    atlas_inputs.xI,
                    atlas_inputs.I,
                    loaded.atlas_down,
                )
                atlas_kwargs = _build_atlas_registration_kwargs(
                    loaded.config,
                    loaded.device_used,
                    loaded.Jd,
                    np.asarray(self_alignment_result["A2d"], dtype=np.float32),
                    loaded.initial_affine,
                )
                stage_payloads["atlas_registration"] = atlas_kwargs
                registration = loaded.backend.emlddmm_multiscale(
                    xI=xId,
                    I=_apply_atlas_registration_normalization(Id, loaded.config),
                    xJ=loaded.xJd,
                    J=_apply_atlas_registration_normalization(loaded.Jd, loaded.config),
                    W0=(loaded.Wd > 0).astype(np.float32),
                    **atlas_kwargs,
                )
                graph_config = None
                if (
                    loaded.config.transformation_graph.write_config
                    and loaded.config.atlas_path is not None
                ):
                    graph_config = _build_transformation_graph(
                        loaded.config.atlas_path,
                        loaded.target.source_path,
                        stage_dir / "effective_config.json",
                        loaded.config.label_path,
                    )
                graph_execution_config = None
                if loaded.config.atlas_path is not None:
                    graph_execution_config = _build_transformation_graph_execution_config(
                        atlas_path=loaded.config.atlas_path,
                        target_source=loaded.target.source_path,
                        config_path=stage_dir / "effective_config.json",
                        label_path=loaded.config.label_path,
                    )
                stage_artifacts = write_atlas_registration_outputs(
                    backend=loaded.backend,
                    output_dir=stage_dir,
                    atlas_image=_build_atlas_image(atlas_inputs),
                    target_image=_build_target_image(working_target),
                    registration=registration,
                    stage_config=atlas_kwargs,
                    effective_config={
                        "mode": loaded.config.mode,
                        "device_used": loaded.device_used,
                        "target_downsampling": loaded.target_down,
                        "atlas_downsampling": loaded.atlas_down,
                        "normalization": loaded.config.normalization.atlas_registration_input,
                        "resampling_policy": loaded.pre_resampling_plan.policy,
                        "pre_resampling_plan": _pre_resampling_payload(loaded),
                    },
                    label_axes=atlas_inputs.xS,
                    label_data=atlas_inputs.S,
                    transformation_graph=graph_config,
                    transformation_graph_execution_config=graph_execution_config,
                    registration_payload=registration,
                )
                if loaded.config.transformation_graph.execute:
                    assert loaded.transformation_graph_script is not None
                    logger.info(
                        "Executing transformation graph via %s",
                        loaded.transformation_graph_script,
                    )
                    graph_artifacts = _execute_transformation_graph(
                        loaded.transformation_graph_script,
                        stage_dir / "transformation_graph_execution_config.json",
                        stage_dir / "transformation_graph",
                    )
                    stage_artifacts["transformation_graph"] = graph_artifacts
                    logger.info(
                        "Completed transformation-graph execution -> %s",
                        graph_artifacts["output_dir"],
                    )
                stage_artifacts["artifacts"] = str(
                    _write_stage_artifacts(stage_dir, build_stage_artifact_manifest(stage_dir))
                )
                artifacts["atlas_registration"] = stage_artifacts
                stage_results["atlas_registration"] = {
                    "status": "completed",
                    "artifacts": stage_artifacts,
                }
                completed_stages.append("atlas_registration")
                timings["atlas_registration_seconds"] = round(
                    time.perf_counter() - stage_started,
                    6,
                )
                stage_timeline.append(
                    _timeline_entry(
                        "atlas_registration",
                        enabled=True,
                        status="completed",
                        output_dir=stage_dir,
                        started_at=stage_started_at,
                        ended_at=datetime.now(UTC).isoformat(),
                        duration_seconds=timings["atlas_registration_seconds"],
                    )
                )
                logger.info(
                    "Completed stage: atlas_registration in %.3fs",
                    timings["atlas_registration_seconds"],
                )
                logger.info("Wrote atlas_registration artifacts -> %s", stage_dir)
            else:
                logger.info(
                    "Skipping stage: atlas_registration (%s)",
                    loaded.skip_reasons.get("atlas_registration"),
                )
                stage_results["atlas_registration"] = {
                    "status": "skipped",
                    "reason": loaded.skip_reasons.get("atlas_registration"),
                }
                stage_timeline.append(
                    _timeline_entry(
                        "atlas_registration",
                        enabled=False,
                        status="skipped",
                        output_dir=_stage_dir(registration_output_dir, "atlas_registration"),
                        reason=loaded.skip_reasons.get("atlas_registration"),
                    )
                )

            if loaded.stage_controls["upsampling"]:
                logger.info("Starting stage: upsampling")
                stage_started = time.perf_counter()
                stage_started_at = datetime.now(UTC).isoformat()
                stage_dir = _stage_dir(registration_output_dir, "upsampling")
                upsampling_kwargs = _build_upsampling_kwargs(loaded.config)
                stage_payloads["upsampling"] = upsampling_kwargs
                _write_stage_inputs(stage_dir, _stage_input_payload(loaded, "upsampling"))
                assert self_alignment_result is not None
                upsampling = _upsample_between_slices_impl(
                    loaded.xJd,
                    np.asarray(self_alignment_result["Jr"], dtype=np.float32),
                    np.asarray(self_alignment_result["A2d"], dtype=np.float32),
                    **upsampling_kwargs,
                )
                stage_artifacts = write_upsampling_outputs(
                    backend=loaded.backend,
                    output_dir=stage_dir,
                    xJ=loaded.xJd,
                    upsampling=upsampling,
                    effective_config={
                        "mode": loaded.config.upsampling.mode,
                        "device_used": loaded.device_used,
                    },
                    metadata={
                        "pairs": upsampling.get("pairs", []),
                        "slices_with_data": upsampling.get("slices_with_data", []),
                    },
                )
                stage_artifacts["artifacts"] = str(
                    _write_stage_artifacts(stage_dir, build_stage_artifact_manifest(stage_dir))
                )
                artifacts["upsampling"] = stage_artifacts
                stage_results["upsampling"] = {
                    "status": "completed",
                    "artifacts": stage_artifacts,
                }
                completed_stages.append("upsampling")
                timings["upsampling_seconds"] = round(
                    time.perf_counter() - stage_started,
                    6,
                )
                stage_timeline.append(
                    _timeline_entry(
                        "upsampling",
                        enabled=True,
                        status="completed",
                        output_dir=stage_dir,
                        started_at=stage_started_at,
                        ended_at=datetime.now(UTC).isoformat(),
                        duration_seconds=timings["upsampling_seconds"],
                    )
                )
                logger.info(
                    "Completed stage: upsampling in %.3fs",
                    timings["upsampling_seconds"],
                )
                logger.info("Wrote upsampling artifacts -> %s", stage_dir)
            else:
                logger.info(
                    "Skipping stage: upsampling (%s)",
                    loaded.skip_reasons.get("upsampling"),
                )
                stage_results["upsampling"] = {
                    "status": "skipped",
                    "reason": loaded.skip_reasons.get("upsampling"),
                }
                stage_timeline.append(
                    _timeline_entry(
                        "upsampling",
                        enabled=False,
                        status="skipped",
                        output_dir=_stage_dir(registration_output_dir, "upsampling"),
                        reason=loaded.skip_reasons.get("upsampling"),
                    )
                )

        if loaded.config.debug.write_notebook_bundle:
            bundle_path = _write_notebook_bundle(
                registration_output_dir,
                resolved_plan,
                stage_payloads,
            )
            artifacts.setdefault("debug", {})["notebook_bundle"] = str(bundle_path)
            logger.info("Wrote notebook debug bundle -> %s", bundle_path)

        timings["total_seconds"] = round(time.perf_counter() - started, 6)
        summary_payload = {
            "schema_version": SCHEMA_VERSION,
            "mode": resolved_plan.mode,
            "backend": loaded.backend.name,
            "backend_name": loaded.backend.name,
            "backend_origin": getattr(loaded.backend, "origin_type", None),
            "target_source": resolved_plan.target_source,
            "target_source_format": resolved_plan.target_source_format,
            "manifest_path": resolved_plan.manifest_path,
            "atlas_path": resolved_plan.atlas_path,
            "label_path": resolved_plan.label_path,
            "device_requested": resolved_plan.device_requested,
            "device_used": resolved_plan.device_used,
            "atlas_init_mode": resolved_plan.atlas_init_mode,
            "atlas_unit_scale": resolved_plan.atlas_unit_scale,
            "target_unit_scale": resolved_plan.target_unit_scale,
            "working_resolution_um": resolved_plan.working_resolution_um,
            "target_downsampling": resolved_plan.target_downsampling,
            "atlas_downsampling": resolved_plan.atlas_downsampling,
            "pre_resampling_plan": resolved_plan.pre_resampling_plan.model_dump(mode="python"),
            "enabled_stages": resolved_plan.enabled_stages,
            "completed_stages": completed_stages,
            "skipped_stages": resolved_plan.skipped_stages,
            "self_alignment_enabled": loaded.stage_controls["self_alignment"],
            "atlas_registration_enabled": loaded.stage_controls["atlas_registration"],
            "upsampling_enabled": loaded.stage_controls["upsampling"],
            "transformation_graph_script": loaded.transformation_graph_script,
            "transformation_graph_script_source": loaded.transformation_graph_script_source,
            "warnings": list(resolved_plan.warnings),
            "log_path": log_path,
            "provenance_path": provenance_path,
            "reproduce_command_path": reproduce_command_path,
            "report_manifest_path": report_manifest_path,
            "report_path": report_path,
            "orientation_resolution": loaded.orientation_resolution.model_dump(mode="python"),
            "used_legacy_output_alias": loaded.used_legacy_output_alias,
            "stage_timeline": [entry.model_dump(mode="python") for entry in stage_timeline],
            "timings": timings,
            "artifacts": artifacts,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        reproduce_command_content = build_reproduce_command(
            plan=resolved_plan,
            config_override_path=loaded.config_override_path,
            used_legacy_output_alias=loaded.used_legacy_output_alias,
        )
        reproduce_command_path.write_text(reproduce_command_content, encoding="utf-8")
        logger.info("Wrote normalized replay command -> %s", reproduce_command_path)

        provisional_provenance = build_run_provenance(
            plan=resolved_plan,
            backend=loaded.backend,
            warnings=summary_payload["warnings"],
            config_override_path=loaded.config_override_path,
            original_cli_argv=loaded.original_cli_argv,
            used_legacy_output_alias=loaded.used_legacy_output_alias,
            report_manifest_path=report_manifest_path,
            report_path=report_path,
            replay_command_path=reproduce_command_path,
            repo_root_hint=Path(__file__).resolve(),
            generated_control_files=_collect_generated_control_files(
                registration_output_dir=registration_output_dir,
                artifacts=artifacts,
            ),
        )
        merged_warnings = _merge_warnings(summary_payload["warnings"], provisional_provenance.warnings)

        if loaded.config.outputs.write_qc_report:
            logger.info("Building registration QC report")
            report_manifest, report_warnings = build_registration_report_manifest(
                registration_output=registration_output_dir,
                plan=resolved_plan,
                summary_payload=summary_payload,
                stage_results=stage_results,
                artifacts=artifacts,
                plan_path=plan_path,
                summary_path=summary_path,
                log_path=log_path,
                provenance_payload=provisional_provenance.model_dump(mode="python"),
                provenance_path=provenance_path,
                reproduce_command_path=reproduce_command_path,
            )
            merged_warnings = _merge_warnings(merged_warnings, report_warnings)
            report_manifest["warnings"] = merged_warnings
            report_manifest_path, report_path = write_registration_report(
                registration_output=registration_output_dir,
                manifest=report_manifest,
            )
            logger.info("Wrote registration QC report -> %s", report_path)

        resolved_plan = resolved_plan.model_copy(
            update={
                "warnings": merged_warnings,
                "report_manifest_path": report_manifest_path,
            }
        )
        summary_payload["warnings"] = merged_warnings
        summary_payload["report_manifest_path"] = report_manifest_path
        summary_payload["report_path"] = report_path

        plan_path = _write_json(
            registration_output_dir / "resolved_run_plan.json",
            resolved_plan.model_dump(mode="python"),
        )
        summary_path = _write_json(summary_path, summary_payload)

        final_provenance = build_run_provenance(
            plan=resolved_plan,
            backend=loaded.backend,
            warnings=merged_warnings,
            config_override_path=loaded.config_override_path,
            original_cli_argv=loaded.original_cli_argv,
            used_legacy_output_alias=loaded.used_legacy_output_alias,
            report_manifest_path=report_manifest_path,
            report_path=report_path,
            replay_command_path=reproduce_command_path,
            repo_root_hint=Path(__file__).resolve(),
            generated_control_files=_collect_generated_control_files(
                registration_output_dir=registration_output_dir,
                artifacts=artifacts,
            ),
        )
        _write_json(provenance_path, final_provenance.model_dump(mode="python"))
        logger.info("Wrote run provenance -> %s", provenance_path)

        if loaded.config.outputs.write_qc_report:
            final_report_manifest, _final_report_warnings = build_registration_report_manifest(
                registration_output=registration_output_dir,
                plan=resolved_plan,
                summary_payload=summary_payload,
                stage_results=stage_results,
                artifacts=artifacts,
                plan_path=plan_path,
                summary_path=summary_path,
                log_path=log_path,
                provenance_payload=final_provenance.model_dump(mode="python"),
                provenance_path=provenance_path,
                reproduce_command_path=reproduce_command_path,
            )
            final_report_manifest["warnings"] = summary_payload["warnings"]
            report_manifest_path, report_path = write_registration_report(
                registration_output=registration_output_dir,
                manifest=final_report_manifest,
            )

        summary_payload["report_manifest_path"] = report_manifest_path
        summary_payload["report_path"] = report_path
        summary_path = _write_json(summary_path, summary_payload)
        logger.info("Wrote registration summary -> %s", summary_path)
        logger.info("Completed EM-LDDMM workflow in %.3fs", timings["total_seconds"])

        output_paths = {
            "plan": str(plan_path),
            "summary": str(summary_path),
            "log": str(log_path),
            "provenance": str(provenance_path),
            "reproduce_command": str(reproduce_command_path),
        }
        if report_manifest_path is not None and report_path is not None:
            output_paths["report_manifest"] = str(report_manifest_path)
            output_paths["report"] = str(report_path)

        return EmlddmmWorkflowResult(
            mode=resolved_plan.mode,
            plan_path=plan_path,
            registration_output=registration_output_dir,
            summary_path=summary_path,
            stage_results=stage_results,
            artifacts=artifacts,
            output_paths=output_paths,
            log_path=log_path,
            provenance_path=provenance_path,
            reproduce_command_path=reproduce_command_path,
            report_path=report_path,
            report_manifest_path=report_manifest_path,
            resolved_plan=resolved_plan,
        )
    finally:
        _detach_registration_log_handler(package_logger, file_handler)
