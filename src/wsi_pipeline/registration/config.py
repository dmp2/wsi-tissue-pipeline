"""Configuration models for the EM-LDDMM workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "emlddmm-step5/v1"


class TargetSourceConfig(BaseModel):
    """Target input configuration."""

    model_config = ConfigDict(extra="ignore")

    path: Path | None = None
    format: Literal["auto", "prepared-dir", "precomputed"] = "auto"
    manifest_path: Path | None = None


class UnitsConfig(BaseModel):
    """Physical-unit and working-resolution configuration."""

    model_config = ConfigDict(extra="ignore")

    atlas_unit_scale: float = Field(default=1000.0)
    target_unit_scale: float = Field(default=1.0)
    desired_resolution_um: float = Field(default=200.0, gt=0.0)


class ResamplingConfig(BaseModel):
    """Outer pre-resampling policy controls."""

    model_config = ConfigDict(extra="ignore")

    policy: Literal["sectioned-stack", "legacy-target-first"] = "sectioned-stack"


class StageControlsConfig(BaseModel):
    """High-level stage enablement controls."""

    model_config = ConfigDict(extra="ignore")

    self_alignment_enabled: bool = True
    atlas_registration_enabled: bool = True
    upsampling_enabled: bool = False


class SelfAlignmentConfig(BaseModel):
    """Atlas-free self-alignment settings."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    draw: bool = False
    n_steps: int = 10
    eA2d: float = 2e4
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class AtlasRegistrationConfig(BaseModel):
    """Atlas-to-target EM-LDDMM settings."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    downI: list[list[int]] = Field(
        default_factory=lambda: [[4, 4, 4], [2, 2, 2], [1, 1, 1]]
    )
    downJ: list[list[int]] = Field(
        default_factory=lambda: [[1, 4, 4], [1, 2, 2], [1, 1, 1]]
    )
    n_iter: list[int] = Field(default_factory=lambda: [100, 50, 25])
    a: list[float] = Field(default_factory=lambda: [500.0])
    dv: list[float] = Field(default_factory=lambda: [1000.0])
    slice_matching: list[bool] = Field(default_factory=lambda: [True])
    v_start: list[int] = Field(default_factory=lambda: [0])
    eA: list[float] = Field(default_factory=lambda: [1e7])
    eA2d: list[float] = Field(default_factory=lambda: [1e5])
    ev: list[float] = Field(default_factory=lambda: [1e-2])
    local_contrast: list[list[int]] = Field(default_factory=lambda: [[1, 16, 16]])
    up_vector: list[list[float]] = Field(default_factory=lambda: [[0.0, 0.0, -1.0]])
    sigmaR: list[float] = Field(default_factory=lambda: [1e4])
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class UpsamplingConfig(BaseModel):
    """Between-slice upsampling settings."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    mode: Literal["seg", "img"] = "seg"
    nt: int = 10
    downI: list[list[int]] = Field(
        default_factory=lambda: [[4, 4, 4], [2, 2, 2], [1, 1, 1]]
    )
    downJ: list[list[int]] = Field(
        default_factory=lambda: [[1, 4, 4], [1, 2, 2], [1, 1, 1]]
    )
    n_iter: list[int] = Field(default_factory=lambda: [100, 50, 25])
    a: list[float] = Field(default_factory=lambda: [500.0])
    dv: list[float] = Field(default_factory=lambda: [1000.0])
    slice_matching: list[bool] = Field(default_factory=lambda: [False])
    v_start: list[int] = Field(default_factory=lambda: [0])
    eA: list[float] = Field(default_factory=lambda: [0.0])
    eA2d: list[float] = Field(default_factory=lambda: [0.0])
    ev: list[float] = Field(default_factory=lambda: [1e-2])
    local_contrast: list[list[int]] = Field(default_factory=lambda: [[1, 16, 16]])
    up_vector: list[list[float]] = Field(default_factory=lambda: [[0.0, 0.0, -1.0]])
    sigmaR: list[float] = Field(default_factory=lambda: [1e4])
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class OutputConfig(BaseModel):
    """Workflow output toggles."""

    model_config = ConfigDict(extra="ignore")

    write_standard_outputs: bool = True
    write_debug_outputs: bool = False
    write_qc_report: bool = False


class NormalizationConfig(BaseModel):
    """Normalization controls for notebook-compatibility and opt-in legacy behavior."""

    model_config = ConfigDict(extra="ignore")

    atlas_registration_input: Literal["none", "mean_abs"] = "none"


class TransformationGraphConfig(BaseModel):
    """Transformation-graph artifact and execution controls."""

    model_config = ConfigDict(extra="ignore")

    write_config: bool = True
    execute: bool = False
    script_path: Path | None = None


class DebugConfig(BaseModel):
    """Debug-only output controls."""

    model_config = ConfigDict(extra="ignore")

    write_notebook_bundle: bool = False


class EmlddmmWorkflowConfig(BaseModel):
    """Main workflow configuration."""

    model_config = ConfigDict(extra="ignore")

    mode: Literal["atlas_free", "atlas_to_target"] | None = None
    preset: str = "macaque-notebook"
    backend: Literal["hybrid"] = "hybrid"
    device: str = "auto"
    target_source: TargetSourceConfig = Field(default_factory=TargetSourceConfig)
    units: UnitsConfig = Field(default_factory=UnitsConfig)
    resampling: ResamplingConfig = Field(default_factory=ResamplingConfig)
    stage_controls: StageControlsConfig = Field(default_factory=StageControlsConfig)
    self_alignment: SelfAlignmentConfig = Field(default_factory=SelfAlignmentConfig)
    atlas_registration: AtlasRegistrationConfig = Field(default_factory=AtlasRegistrationConfig)
    upsampling: UpsamplingConfig = Field(default_factory=UpsamplingConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    transformation_graph: TransformationGraphConfig = Field(default_factory=TransformationGraphConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    atlas_path: Path | None = None
    label_path: Path | None = None
    init_affine_path: Path | None = None
    orientation_from: str | None = None
    orientation_to: str | None = None


class EmlddmmStagePlan(BaseModel):
    """Resolved plan information for a single workflow stage."""

    model_config = ConfigDict(extra="ignore")

    name: str
    enabled: bool
    reason: str | None = None
    output_dir: Path | None = None
    expected_outputs: list[str] = Field(default_factory=list)


class OrientationResolution(BaseModel):
    """Structured atlas-init orientation metadata."""

    model_config = ConfigDict(extra="ignore")

    mode: Literal["none", "matrix", "orientation"] = "none"
    orientation_from: str | None = None
    orientation_to: str | None = None
    is_valid: bool = True
    validation_rule: str | None = None
    resolved_matrix: list[list[float]] | None = None
    source: str | None = None


class StageTimelineEntry(BaseModel):
    """Execution timeline metadata for a single stage."""

    model_config = ConfigDict(extra="ignore")

    name: str
    enabled: bool
    status: Literal["planned", "skipped", "completed", "failed"]
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None
    reason: str | None = None
    output_dir: Path | None = None


class EmlddmmRunProvenance(BaseModel):
    """Machine-readable run provenance written for every step-5 run."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = SCHEMA_VERSION
    generated_at: str
    warnings: list[str] = Field(default_factory=list)
    pipeline: dict[str, Any] = Field(default_factory=dict)
    git: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    backend: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    resolved_cli: dict[str, Any] = Field(default_factory=dict)
    resolved_workflow_config: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    parity: dict[str, Any] = Field(default_factory=dict)


class PreResamplingPlan(BaseModel):
    """Resolved outer pre-resampling metadata."""

    model_config = ConfigDict(extra="ignore")

    policy: Literal["sectioned-stack", "legacy-target-first"]
    target_native_spacing_um: list[float] = Field(default_factory=list)
    target_working_spacing_um: list[float] = Field(default_factory=list)
    target_locked_axes: list[int] = Field(default_factory=list)
    target_downsampling: list[int] = Field(default_factory=list)
    atlas_native_spacing_um: list[float] | None = None
    atlas_reference_spacing_um: list[float] | None = None
    atlas_locked_axes: list[int] = Field(default_factory=list)
    atlas_downsampling: list[int] | None = None
    notes: list[str] = Field(default_factory=list)


class EmlddmmResolvedPlan(BaseModel):
    """Resolved execution plan for a step-5 run."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = SCHEMA_VERSION
    mode: Literal["atlas_free", "atlas_to_target"]
    preset: str
    dataset_root: Path
    registration_output: Path
    target_source: Path
    target_source_format: Literal["prepared-dir", "precomputed"]
    manifest_path: Path | None = None
    atlas_path: Path | None = None
    label_path: Path | None = None
    device_requested: str
    device_used: str
    backend_name: str = ""
    atlas_init_mode: Literal["none", "matrix", "orientation"] = "none"
    atlas_unit_scale: float | None = None
    target_unit_scale: float | None = None
    working_resolution_um: float
    target_downsampling: list[int]
    atlas_downsampling: list[int] | None = None
    pre_resampling_plan: PreResamplingPlan
    enabled_stages: list[str] = Field(default_factory=list)
    skipped_stages: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    transformation_graph_script: Path | None = None
    log_path: Path | None = None
    report_manifest_path: Path | None = None
    provenance_path: Path | None = None
    orientation_resolution: OrientationResolution = Field(default_factory=OrientationResolution)
    used_legacy_output_alias: bool = False
    dry_run: bool = False
    stages: dict[str, EmlddmmStagePlan] = Field(default_factory=dict)
    expected_outputs: dict[str, Any] = Field(default_factory=dict)
    workflow_config: dict[str, Any] = Field(default_factory=dict)


class EmlddmmWorkflowResult(BaseModel):
    """Workflow result."""

    model_config = ConfigDict(extra="ignore")

    mode: Literal["atlas_free", "atlas_to_target"]
    plan_path: Path
    registration_output: Path
    summary_path: Path
    log_path: Path
    provenance_path: Path | None = None
    reproduce_command_path: Path | None = None
    report_path: Path | None = None
    report_manifest_path: Path | None = None
    stage_results: dict[str, Any]
    artifacts: dict[str, Any]
    resolved_plan: EmlddmmResolvedPlan | None = None
    output_paths: dict[str, str | list[str]] = Field(default_factory=dict)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries."""

    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(current, value)
        else:
            merged[key] = value
    return merged


def get_preset_config(name: str = "macaque-notebook") -> EmlddmmWorkflowConfig:
    """Return a preset workflow configuration."""

    if name != "macaque-notebook":
        raise ValueError(f"Unknown EM-LDDMM preset: {name!r}")
    return EmlddmmWorkflowConfig(preset=name)


def merge_workflow_config(
    base: EmlddmmWorkflowConfig,
    override: dict[str, Any] | None = None,
) -> EmlddmmWorkflowConfig:
    """Merge a nested override dictionary into a workflow configuration."""

    if not override:
        return base
    merged = _merge_dicts(base.model_dump(mode="python"), override)
    return EmlddmmWorkflowConfig.model_validate(merged)


def load_workflow_config_override(path: str | Path | None) -> dict[str, Any]:
    """Load a JSON override file for the workflow."""

    if path is None:
        return {}
    override_path = Path(path)
    with open(override_path, encoding="utf-8") as f:
        return json.load(f)
