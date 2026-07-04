"""Novice-facing setup summaries for submission batches."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field

from .models import SubmissionBaseModel
from .preflight import (
    IssueSeverity,
    PreflightIssue,
    PreflightRowResult,
    build_preflight_state,
    run_preflight,
)
from .profiles import RequirementPhase, load_database_profile
from .source_paths import source_location as _source_location
from .source_paths import source_target_for_extension as _source_target_for_extension
from .tissue_plan import build_tissue_detection_plan

SETUP_REPORT_VERSION = "1.0"
DECIMAL_GB = 1_000_000_000
TIFF_SUFFIXES = {".tif", ".tiff"}
OMETIFF_SUFFIXES = (".ome.tif", ".ome.tiff")


class WorkflowMode(str, Enum):
    """Supported setup workflow modes."""

    EXISTING_OMETIFF_UPLOAD = "existing-ometiff-upload"
    CONVERT_SINGLE_TISSUE = "convert-single-tissue"
    EXTRACT_CONVERT_UPLOAD = "extract-convert-upload"


class SetupIssue(SubmissionBaseModel):
    """One setup finding collected from preflight or setup-specific checks."""

    row_number: int | None = None
    row_identifier: str | None = None
    source_path: str | None = None
    source_image_id: str | None = None
    field: str | None = None
    code: str
    message: str
    severity: IssueSeverity
    requirement_phase: RequirementPhase | None = None
    source: str = "setup"


class SetupRowSummary(SubmissionBaseModel):
    """Setup-facing row summary with estimates and compatibility state."""

    row_number: int
    row_identifier: str | None = None
    source_image_id: str | None = None
    source_path: str | None = None
    valid: bool
    blocked: bool
    deferred: bool
    warning: bool
    known_input_bytes: int | None = None
    input_size_status: str
    mode_compatible: bool
    tissue_plan_category: str | None = None
    tissue_plan_reason_code: str | None = None
    issues: list[SetupIssue] = Field(default_factory=list)


class SetupReport(SubmissionBaseModel):
    """Machine-readable setup report for novice batch review and future GUI use."""

    setup_report_version: str = SETUP_REPORT_VERSION
    generated_at: datetime
    workflow_mode: WorkflowMode
    profile_identifier: str
    profile_name: str
    profile_version: str
    profile_path: Path
    manifest_path: Path
    row_count: int
    valid_row_count: int
    blocked_row_count: int
    deferred_row_count: int
    warning_count: int
    deferred_count: int
    error_count: int
    known_input_bytes: int
    unknown_size_row_count: int
    estimated_output_bytes_low: int | None = None
    estimated_output_bytes_high: int | None = None
    estimated_processing_seconds_low: float | None = None
    estimated_processing_seconds_high: float | None = None
    estimated_upload_seconds_low: float | None = None
    estimated_upload_seconds_high: float | None = None
    estimated_total_seconds_low: float | None = None
    estimated_total_seconds_high: float | None = None
    ready_for_next_action: bool
    next_action: str
    recommended_next_action: str
    strict: bool
    upload_mbps: float
    upload_overhead: float
    row_summaries: list[SetupRowSummary] = Field(default_factory=list)
    issues: list[SetupIssue] = Field(default_factory=list)


@dataclass(frozen=True)
class SetupRunResult:
    """Return object for a setup run."""

    report: SetupReport
    setup_report_path: Path | None = None

    def exit_code(self, *, strict: bool | None = None) -> int:
        """Return the CLI exit code for this setup result."""
        strict_mode = self.report.strict if strict is None else strict
        if self.report.error_count:
            return 1
        if strict_mode and (self.report.warning_count or self.report.deferred_count):
            return 1
        return 0


@dataclass(frozen=True)
class _SourceLocation:
    is_local: bool
    path_to_check: Path | None = None


@dataclass(frozen=True)
class _SizeResult:
    known_input_bytes: int | None
    input_size_status: str


def run_setup(
    profile_path: str | Path,
    manifest_path: str | Path,
    workflow_mode: str | WorkflowMode,
    *,
    setup_report_path: str | Path | None = None,
    upload_mbps: float = 100.0,
    upload_overhead: float = 1.25,
    strict: bool = False,
    generated_at: datetime | None = None,
) -> SetupRunResult:
    """Run setup checks and arithmetic-only estimates for a submission batch."""
    profile_path = Path(profile_path)
    manifest_path = Path(manifest_path)
    mode = WorkflowMode(workflow_mode)
    if upload_mbps <= 0:
        raise ValueError("upload_mbps must be greater than 0")
    if upload_overhead <= 0:
        raise ValueError("upload_overhead must be greater than 0")

    profile = load_database_profile(profile_path)
    accepted_extensions = profile.accepted_extensions_for_workflow_mode(mode.value)
    preflight = run_preflight(
        profile_path,
        manifest_path,
        strict=strict,
        accepted_extensions=accepted_extensions,
    )
    state = build_preflight_state(preflight.report)
    tissue_rows = _tissue_plan_rows(state) if mode is WorkflowMode.EXTRACT_CONVERT_UPLOAD else {}

    row_summaries: list[SetupRowSummary] = []
    issues: list[SetupIssue] = [
        _setup_issue_from_preflight(issue) for issue in preflight.report.batch_issues
    ]
    known_input_bytes = 0
    unknown_size_row_count = 0

    for row in preflight.report.row_results:
        row_issues = [_setup_issue_from_preflight(issue) for issue in row.issues]
        row_issues.extend(_mode_compatibility_issues(row, mode, accepted_extensions))
        size_result = _local_input_size(row.source_path, manifest_path)
        if size_result.known_input_bytes is None:
            unknown_size_row_count += 1
        else:
            known_input_bytes += size_result.known_input_bytes

        counts = Counter(issue.severity for issue in row_issues)
        tissue_category, tissue_reason = tissue_rows.get(row.row_number, (None, None))
        row_summaries.append(
            SetupRowSummary(
                row_number=row.row_number,
                row_identifier=row.row_identifier,
                source_image_id=row.source_image_id,
                source_path=row.source_path,
                valid=counts[IssueSeverity.ERROR] == 0,
                blocked=counts[IssueSeverity.ERROR] > 0,
                deferred=counts[IssueSeverity.DEFERRED] > 0,
                warning=counts[IssueSeverity.WARNING] > 0,
                known_input_bytes=size_result.known_input_bytes,
                input_size_status=size_result.input_size_status,
                mode_compatible=not any(issue.source == "setup" for issue in row_issues),
                tissue_plan_category=tissue_category,
                tissue_plan_reason_code=tissue_reason,
                issues=row_issues,
            )
        )
        issues.extend(row_issues)

    issue_counts = Counter(issue.severity for issue in issues)
    estimate = _estimate_batch(
        mode=mode,
        known_input_bytes=known_input_bytes,
        unknown_size_row_count=unknown_size_row_count,
        upload_mbps=upload_mbps,
        upload_overhead=upload_overhead,
    )
    next_action = _next_action(mode)
    ready_for_next_action = issue_counts[IssueSeverity.ERROR] == 0 and not (
        strict and (issue_counts[IssueSeverity.WARNING] or issue_counts[IssueSeverity.DEFERRED])
    )
    report = SetupReport(
        generated_at=generated_at or datetime.now(timezone.utc),
        workflow_mode=mode,
        profile_identifier=preflight.report.profile_identifier,
        profile_name=preflight.report.profile_name,
        profile_version=preflight.report.profile_version,
        profile_path=profile_path,
        manifest_path=manifest_path,
        row_count=preflight.report.total_row_count,
        valid_row_count=sum(1 for row in row_summaries if row.valid),
        blocked_row_count=sum(1 for row in row_summaries if row.blocked),
        deferred_row_count=sum(1 for row in row_summaries if row.deferred),
        warning_count=issue_counts[IssueSeverity.WARNING],
        deferred_count=issue_counts[IssueSeverity.DEFERRED],
        error_count=issue_counts[IssueSeverity.ERROR],
        known_input_bytes=known_input_bytes,
        unknown_size_row_count=unknown_size_row_count,
        ready_for_next_action=ready_for_next_action,
        next_action=next_action,
        recommended_next_action=_recommended_next_action(
            mode=mode,
            ready_for_next_action=ready_for_next_action,
            error_count=issue_counts[IssueSeverity.ERROR],
            warning_count=issue_counts[IssueSeverity.WARNING],
            deferred_count=issue_counts[IssueSeverity.DEFERRED],
            strict=strict,
        ),
        strict=strict,
        upload_mbps=float(upload_mbps),
        upload_overhead=float(upload_overhead),
        row_summaries=row_summaries,
        issues=issues,
        **estimate,
    )

    output_path = Path(setup_report_path) if setup_report_path is not None else None
    if output_path is not None:
        _write_json(output_path, report.to_dict())

    return SetupRunResult(report=report, setup_report_path=output_path)


def _tissue_plan_rows(state: Any) -> dict[int, tuple[str | None, str | None]]:
    plan = build_tissue_detection_plan(state, source_state_path=state.manifest_path)
    rows: dict[int, tuple[str | None, str | None]] = {}
    for job in plan.jobs:
        rows[job.row_number] = (job.category.value, None)
    for summary in [*plan.blocked_rows, *plan.skipped_rows]:
        rows[summary.row_number] = (summary.category.value, summary.reason_code)
    return rows


def _setup_issue_from_preflight(issue: PreflightIssue) -> SetupIssue:
    return SetupIssue(
        row_number=issue.row_number,
        row_identifier=issue.row_identifier,
        source_path=issue.source_path,
        source_image_id=issue.source_image_id,
        field=issue.field,
        code=issue.code,
        message=issue.message,
        severity=issue.severity,
        requirement_phase=issue.requirement_phase,
        source="preflight",
    )


def _mode_compatibility_issues(
    row: PreflightRowResult,
    mode: WorkflowMode,
    accepted_extensions: list[str],
) -> list[SetupIssue]:
    if not row.source_path:
        return []

    source_target = _source_target_for_extension(row.source_path)
    lower_name = Path(source_target).name.lower()
    suffix = Path(source_target).suffix.lower()
    accepted = {extension.lower() for extension in accepted_extensions}
    is_ometiff = lower_name.endswith(OMETIFF_SUFFIXES)
    is_generic_tiff = suffix in TIFF_SUFFIXES and not is_ometiff
    extension_is_profile_accepted = _extension_is_profile_accepted(source_target, accepted)

    if mode is WorkflowMode.EXISTING_OMETIFF_UPLOAD:
        generic_tiff_allowed = is_generic_tiff and suffix in accepted
        if is_ometiff or generic_tiff_allowed:
            return []
        code = (
            "WRONG_MODE_FOR_OMETIFF_UPLOAD"
            if extension_is_profile_accepted or suffix in {".vsi", ".ets"}
            else "MODE_INPUT_EXTENSION_MISMATCH"
        )
        return [
            _mode_issue(
                row,
                code=code,
                message=(
                    "Selected workflow mode expects existing OME-TIFF upload inputs. "
                    f"Source path appears to be '{suffix or '<none>'}' instead."
                ),
            )
        ]

    if mode is WorkflowMode.CONVERT_SINGLE_TISSUE:
        if is_ometiff:
            return [
                _mode_issue(
                    row,
                    code="WRONG_MODE_FOR_SOURCE_CONVERSION",
                    message=(
                        "Selected workflow mode expects source microscopy files for conversion; "
                        "this row appears to already be an OME-TIFF. Use existing-ometiff-upload."
                    ),
                )
            ]
        if not extension_is_profile_accepted:
            return [
                _mode_issue(
                    row,
                    code="MODE_INPUT_EXTENSION_MISMATCH",
                    message=(
                        "Selected workflow mode expects a source microscopy file extension "
                        "accepted by the profile."
                    ),
                )
            ]
        return []

    if is_ometiff:
        return [
            _mode_issue(
                row,
                code="WRONG_MODE_FOR_TISSUE_EXTRACTION",
                message=(
                    "Selected workflow mode expects parent source WSI files for tissue extraction; "
                    "this row appears to already be an OME-TIFF. Use existing-ometiff-upload unless "
                    "a future profile explicitly supports extraction from OME-TIFF WSI inputs."
                ),
            )
        ]
    if not extension_is_profile_accepted:
        return [
            _mode_issue(
                row,
                code="MODE_INPUT_EXTENSION_MISMATCH",
                message=(
                    "Selected workflow mode expects a parent source WSI extension accepted by the profile."
                ),
            )
        ]
    return []


def _mode_issue(row: PreflightRowResult, *, code: str, message: str) -> SetupIssue:
    return SetupIssue(
        row_number=row.row_number,
        row_identifier=row.row_identifier,
        source_path=row.source_path,
        source_image_id=row.source_image_id,
        field="source_path",
        code=code,
        message=message,
        severity=IssueSeverity.ERROR,
        requirement_phase=RequirementPhase.SOURCE_FILE_PREFLIGHT,
        source="setup",
    )


def _extension_is_profile_accepted(source_target: str, accepted_extensions: set[str]) -> bool:
    lower_target = source_target.lower()
    return any(lower_target.endswith(extension) for extension in accepted_extensions)


def _local_input_size(source_path: str | None, manifest_path: Path) -> _SizeResult:
    if not source_path:
        return _SizeResult(known_input_bytes=None, input_size_status="unknown_missing_source_path")

    location = _source_location(source_path, manifest_path)
    if not location.is_local:
        return _SizeResult(known_input_bytes=None, input_size_status="unknown_non_local_source")
    if location.path_to_check is None or not location.path_to_check.exists():
        return _SizeResult(known_input_bytes=None, input_size_status="unknown_missing_local_source")
    return _SizeResult(
        known_input_bytes=location.path_to_check.stat().st_size,
        input_size_status="known",
    )


def _estimate_batch(
    *,
    mode: WorkflowMode,
    known_input_bytes: int,
    unknown_size_row_count: int,
    upload_mbps: float,
    upload_overhead: float,
) -> dict[str, int | float | None]:
    if unknown_size_row_count:
        return {
            "estimated_output_bytes_low": None,
            "estimated_output_bytes_high": None,
            "estimated_processing_seconds_low": None,
            "estimated_processing_seconds_high": None,
            "estimated_upload_seconds_low": None,
            "estimated_upload_seconds_high": None,
            "estimated_total_seconds_low": None,
            "estimated_total_seconds_high": None,
        }

    output_low_multiplier, output_high_multiplier = _output_size_multipliers(mode)
    processing_low_per_gb, processing_high_per_gb = _processing_seconds_per_gb(mode)
    output_low = math.floor(known_input_bytes * output_low_multiplier)
    output_high = math.ceil(known_input_bytes * output_high_multiplier)
    input_gb = known_input_bytes / DECIMAL_GB
    processing_low = input_gb * processing_low_per_gb
    processing_high = input_gb * processing_high_per_gb
    upload_low = _upload_seconds(output_low, upload_mbps, upload_overhead)
    upload_high = _upload_seconds(output_high, upload_mbps, upload_overhead)
    return {
        "estimated_output_bytes_low": output_low,
        "estimated_output_bytes_high": output_high,
        "estimated_processing_seconds_low": processing_low,
        "estimated_processing_seconds_high": processing_high,
        "estimated_upload_seconds_low": upload_low,
        "estimated_upload_seconds_high": upload_high,
        "estimated_total_seconds_low": processing_low + upload_low,
        "estimated_total_seconds_high": processing_high + upload_high,
    }


def _output_size_multipliers(mode: WorkflowMode) -> tuple[float, float]:
    if mode is WorkflowMode.EXISTING_OMETIFF_UPLOAD:
        return 1.0, 1.0
    if mode is WorkflowMode.CONVERT_SINGLE_TISSUE:
        return 0.75, 1.5
    return 0.25, 1.5


def _processing_seconds_per_gb(mode: WorkflowMode) -> tuple[float, float]:
    if mode is WorkflowMode.EXISTING_OMETIFF_UPLOAD:
        return 0.0, 0.0
    if mode is WorkflowMode.CONVERT_SINGLE_TISSUE:
        return 300.0, 1800.0
    return 900.0, 3600.0


def _upload_seconds(bytes_to_upload: int, upload_mbps: float, upload_overhead: float) -> float:
    return bytes_to_upload * 8 / (upload_mbps * 1_000_000) * upload_overhead


def _next_action(mode: WorkflowMode) -> str:
    if mode is WorkflowMode.EXISTING_OMETIFF_UPLOAD:
        return "Validate/package existing OME-TIFF upload inputs."
    if mode is WorkflowMode.CONVERT_SINGLE_TISSUE:
        return "Run future single-tissue conversion stage."
    return "Run future tissue detection and extraction planning stage."


def _recommended_next_action(
    *,
    mode: WorkflowMode,
    ready_for_next_action: bool,
    error_count: int,
    warning_count: int,
    deferred_count: int,
    strict: bool,
) -> str:
    if error_count:
        return "Fix blocking manifest/path/profile/mode errors before continuing."
    if strict and (warning_count or deferred_count):
        return "Resolve warnings or deferred requirements, or rerun setup without strict mode."
    if not ready_for_next_action:
        return "Resolve setup findings before continuing."
    if mode is WorkflowMode.EXISTING_OMETIFF_UPLOAD:
        return "Ready for OME-TIFF validation/package/upload follow-up."
    if mode is WorkflowMode.CONVERT_SINGLE_TISSUE:
        return "Ready for future single-tissue conversion step; source metadata will be validated during conversion."
    return "Ready for future tissue detection stage; estimates are coarse until detection runs."


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "SETUP_REPORT_VERSION",
    "SetupIssue",
    "SetupReport",
    "SetupRowSummary",
    "SetupRunResult",
    "WorkflowMode",
    "run_setup",
]
