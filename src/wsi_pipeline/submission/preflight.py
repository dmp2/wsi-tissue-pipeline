"""Manifest/profile preflight checks for submission batches."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from pydantic import Field, ValidationError

from .manifests import (
    REQUIRED_MANIFEST_COLUMNS,
    SUPPORTED_MANIFEST_COLUMNS,
    _blank_to_none,
    _column_lookup,
    source_extension_for_message,
    source_extension_is_accepted,
)
from .models import SourceSlide, SubmissionBaseModel
from .profiles import DatabaseProfile, RequirementPhase, load_database_profile
from .statuses import BatchStatus

PREFLIGHT_REPORT_VERSION = "1.0"
PREFLIGHT_STATE_VERSION = "1.1"


class IssueSeverity(str, Enum):
    """Preflight issue severity."""

    ERROR = "error"
    WARNING = "warning"
    DEFERRED = "deferred"


class PreflightIssue(SubmissionBaseModel):
    """One batch-level or row-level preflight finding."""

    row_number: int | None = None
    row_identifier: str | None = None
    source_path: str | None = None
    source_image_id: str | None = None
    field: str | None = None
    code: str
    message: str
    severity: IssueSeverity
    requirement_phase: RequirementPhase | None = None


class PreflightRowResult(SubmissionBaseModel):
    """Preflight result for one manifest data row."""

    row_number: int
    row_identifier: str | None = None
    source_path: str | None = None
    source_image_id: str | None = None
    valid: bool
    preflight_status: BatchStatus
    issues: list[PreflightIssue] = Field(default_factory=list)


class PreflightStateIssue(SubmissionBaseModel):
    """Compact issue summary stored with preflight state rows."""

    code: str
    severity: IssueSeverity
    message: str
    field: str | None = None
    requirement_phase: RequirementPhase | None = None


class PreflightReport(SubmissionBaseModel):
    """Machine-readable preflight report."""

    preflight_report_version: str = PREFLIGHT_REPORT_VERSION
    generated_at: datetime
    profile_path: Path
    manifest_path: Path
    profile_identifier: str
    profile_name: str
    profile_version: str
    total_row_count: int
    valid_row_count: int
    rows_with_warnings: int
    rows_with_deferred_requirements: int
    rows_with_errors: int
    warning_count: int
    deferred_count: int
    error_count: int
    batch_status: BatchStatus
    ready_for_next_stage: bool
    strict: bool
    batch_issues: list[PreflightIssue] = Field(default_factory=list)
    row_results: list[PreflightRowResult] = Field(default_factory=list)


class PreflightStateRow(SubmissionBaseModel):
    """Compact row state for later submission workflow stages."""

    row_number: int
    row_identifier: str | None = None
    source_path: str | None = None
    source_image_id: str | None = None
    preflight_status: BatchStatus
    valid: bool
    issue_codes: list[str] = Field(default_factory=list)
    issues: list[PreflightStateIssue] = Field(default_factory=list)


class PreflightState(SubmissionBaseModel):
    """Draft preflight state file for later workflow stages."""

    preflight_state_version: str = PREFLIGHT_STATE_VERSION
    batch_id: str
    generated_at: datetime
    profile_identifier: str
    manifest_path: Path
    preflight_status: BatchStatus
    row_statuses: list[PreflightStateRow] = Field(default_factory=list)
    json_report_path: Path | None = None


@dataclass(frozen=True)
class PreflightRunResult:
    """Return object for a preflight run."""

    report: PreflightReport
    state: PreflightState | None = None
    json_report_path: Path | None = None
    state_out_path: Path | None = None

    def exit_code(self, *, strict: bool | None = None) -> int:
        """Return the CLI exit code for this preflight result."""
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
class _RequirementSpec:
    key: str
    aliases: tuple[str, ...]
    enabled: Callable[[DatabaseProfile], bool]
    default_phase: RequirementPhase
    manifest_fields: tuple[str, ...]
    field: str
    deferred_code: str
    missing_code: str
    deferred_message: str


ROW_REQUIREMENTS: tuple[_RequirementSpec, ...] = (
    _RequirementSpec(
        key="physical_pixel_size",
        aliases=("physical_pixel_size_x", "physical_pixel_size_y"),
        enabled=lambda profile: profile.metadata.require_physical_pixel_size,
        default_phase=RequirementPhase.SOURCE_METADATA_VALIDATION,
        manifest_fields=("physical_pixel_size_x", "physical_pixel_size_y"),
        field="physical_pixel_size",
        deferred_code="DEFERRED_SOURCE_PIXEL_SIZE",
        missing_code="MISSING_SOURCE_PIXEL_SIZE",
        deferred_message=(
            "Physical pixel size is required by the profile and will be validated "
            "during source metadata inspection."
        ),
    ),
    _RequirementSpec(
        key="physical_units",
        aliases=("physical_pixel_size_unit", "units"),
        enabled=lambda profile: profile.metadata.require_units,
        default_phase=RequirementPhase.SOURCE_METADATA_VALIDATION,
        manifest_fields=("physical_pixel_size_unit",),
        field="physical_pixel_size_unit",
        deferred_code="DEFERRED_SOURCE_UNITS",
        missing_code="MISSING_SOURCE_UNITS",
        deferred_message=(
            "Physical units are required by the profile and will be validated "
            "during source metadata inspection."
        ),
    ),
    _RequirementSpec(
        key="channel_metadata",
        aliases=(),
        enabled=lambda profile: profile.metadata.require_channel_metadata,
        default_phase=RequirementPhase.SOURCE_METADATA_VALIDATION,
        manifest_fields=(),
        field="channel_metadata",
        deferred_code="DEFERRED_SOURCE_CHANNEL_METADATA",
        missing_code="MISSING_SOURCE_CHANNEL_METADATA",
        deferred_message=(
            "Channel metadata is required by the profile and is deferred to source "
            "metadata validation."
        ),
    ),
    _RequirementSpec(
        key="parent_source_checksum",
        aliases=("checksum", "source_checksum"),
        enabled=lambda profile: profile.metadata.require_parent_source_checksum,
        default_phase=RequirementPhase.SOURCE_METADATA_VALIDATION,
        manifest_fields=("checksum",),
        field="checksum",
        deferred_code="DEFERRED_SOURCE_CHECKSUM",
        missing_code="MISSING_SOURCE_CHECKSUM",
        deferred_message=(
            "Parent source checksum is required by the profile and is deferred; "
            "this preflight does not compute checksums for large source images."
        ),
    ),
    _RequirementSpec(
        key="crop_bounds_parent_pixels",
        aliases=("parent_crop_bounds",),
        enabled=lambda profile: profile.metadata.require_crop_bounds_parent_pixels,
        default_phase=RequirementPhase.DERIVATIVE_EXPORT,
        manifest_fields=(),
        field="crop_bounds_parent_pixels",
        deferred_code="DEFERRED_DERIVATIVE_PROVENANCE",
        missing_code="MISSING_DERIVATIVE_PROVENANCE",
        deferred_message=(
            "Parent crop bounds are derivative provenance and will exist only after "
            "tissue detection/cropping."
        ),
    ),
    _RequirementSpec(
        key="child_array_to_physical_transform",
        aliases=("array_to_physical_transform",),
        enabled=lambda profile: profile.metadata.require_child_array_to_physical_transform,
        default_phase=RequirementPhase.DERIVATIVE_EXPORT,
        manifest_fields=(),
        field="child_array_to_physical_transform",
        deferred_code="DEFERRED_DERIVATIVE_PROVENANCE",
        missing_code="MISSING_DERIVATIVE_PROVENANCE",
        deferred_message=(
            "Child array-to-physical transform is derivative provenance and will exist "
            "only after conversion."
        ),
    ),
)


BATCH_REQUIREMENTS: tuple[_RequirementSpec, ...] = (
    _RequirementSpec(
        key="ome_metadata_validation",
        aliases=("ome_xml", "ometiff_validation"),
        enabled=lambda profile: profile.ometiff.require_ome_xml
        or profile.validation.validate_ometiff,
        default_phase=RequirementPhase.UPLOAD_VALIDATION,
        manifest_fields=(),
        field="ome_metadata_validation",
        deferred_code="DEFERRED_OME_METADATA_VALIDATION",
        missing_code="MISSING_OME_METADATA_VALIDATION",
        deferred_message=(
            "OME metadata validation is required by the profile and is deferred until "
            "OME-TIFF derivatives exist."
        ),
    ),
    _RequirementSpec(
        key="sidecar_consistency",
        aliases=("validate_sidecar_consistency",),
        enabled=lambda profile: profile.validation.validate_sidecar_consistency,
        default_phase=RequirementPhase.UPLOAD_VALIDATION,
        manifest_fields=(),
        field="sidecar_consistency",
        deferred_code="DEFERRED_UPLOAD_VALIDATION",
        missing_code="MISSING_UPLOAD_VALIDATION",
        deferred_message=(
            "Sidecar consistency validation is required by the profile and is deferred "
            "until derivative sidecars exist."
        ),
    ),
    _RequirementSpec(
        key="output_checksums",
        aliases=("validate_checksums", "emit_checksums"),
        enabled=lambda profile: profile.validation.validate_checksums
        or profile.output.emit_checksums,
        default_phase=RequirementPhase.UPLOAD_VALIDATION,
        manifest_fields=(),
        field="output_checksums",
        deferred_code="DEFERRED_UPLOAD_CHECKSUM_VALIDATION",
        missing_code="MISSING_UPLOAD_CHECKSUM_VALIDATION",
        deferred_message=(
            "Output checksum validation is required by the profile and is deferred "
            "until derivative outputs exist."
        ),
    ),
    _RequirementSpec(
        key="tissue_detection_qc",
        aliases=("qc_overlay",),
        enabled=lambda profile: profile.qc.require_tissue_detection_qc,
        default_phase=RequirementPhase.DERIVATIVE_EXPORT,
        manifest_fields=(),
        field="tissue_detection_qc",
        deferred_code="DEFERRED_TISSUE_DETECTION_QC",
        missing_code="MISSING_TISSUE_DETECTION_QC",
        deferred_message=(
            "Tissue detection QC is required by the profile and is deferred until "
            "tissue detection runs."
        ),
    ),
)


def run_preflight(
    profile_path: str | Path,
    manifest_path: str | Path,
    *,
    json_report_path: str | Path | None = None,
    state_out_path: str | Path | None = None,
    strict: bool = False,
    accepted_extensions: Iterable[str] | None = None,
) -> PreflightRunResult:
    """Validate a submission profile and manifest before image processing."""
    profile_path = Path(profile_path)
    manifest_path = Path(manifest_path)
    profile = load_database_profile(profile_path)
    source_extensions = tuple(accepted_extensions or profile.input.accepted_extensions)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Submission manifest not found: {manifest_path}")

    generated_at = datetime.now(timezone.utc)
    batch_issues: list[PreflightIssue] = []
    row_results: list[PreflightRowResult] = []

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            batch_issues.append(
                PreflightIssue(
                    code="EMPTY_MANIFEST",
                    message=f"Submission manifest is empty: {manifest_path}",
                    severity=IssueSeverity.ERROR,
                    requirement_phase=RequirementPhase.PREFLIGHT_MANIFEST,
                )
            )
        else:
            column_lookup = _column_lookup(reader.fieldnames)
            missing_columns = sorted(REQUIRED_MANIFEST_COLUMNS - set(column_lookup))
            if missing_columns:
                batch_issues.append(
                    PreflightIssue(
                        field=", ".join(missing_columns),
                        code="MISSING_MANIFEST_COLUMNS",
                        message=(
                            "Submission manifest is missing required columns: "
                            + ", ".join(missing_columns)
                        ),
                        severity=IssueSeverity.ERROR,
                        requirement_phase=RequirementPhase.PREFLIGHT_MANIFEST,
                    )
                )
            else:
                seen_identifiers: dict[tuple[str, str], int] = {}
                for row in reader:
                    row_results.append(
                        _preflight_row(
                            row,
                            row_number=reader.line_num,
                            column_lookup=column_lookup,
                            manifest_path=manifest_path,
                            profile=profile,
                            accepted_extensions=source_extensions,
                            seen_identifiers=seen_identifiers,
                        )
                    )

    batch_issues.extend(_batch_requirement_issues(profile))
    report = _build_report(
        profile=profile,
        profile_path=profile_path,
        manifest_path=manifest_path,
        generated_at=generated_at,
        strict=strict,
        batch_issues=batch_issues,
        row_results=row_results,
    )

    json_path = Path(json_report_path) if json_report_path is not None else None
    if json_path is not None:
        _write_json(json_path, report.to_dict())

    state = None
    state_path = Path(state_out_path) if state_out_path is not None else None
    if state_path is not None:
        state = _build_state(report, json_path)
        _write_json(state_path, state.to_dict())

    return PreflightRunResult(
        report=report,
        state=state,
        json_report_path=json_path,
        state_out_path=state_path,
    )


def _preflight_row(
    row: Mapping[str, str | None],
    *,
    row_number: int,
    column_lookup: Mapping[str, str],
    manifest_path: Path,
    profile: DatabaseProfile,
    accepted_extensions: Iterable[str],
    seen_identifiers: dict[tuple[str, str], int],
) -> PreflightRowResult:
    specimen_id = _row_value(row, column_lookup, "specimen_id")
    slide_id = _row_value(row, column_lookup, "slide_id")
    source_value = _row_value(row, column_lookup, "source_path")
    row_identifier = _row_identifier(specimen_id, slide_id)
    source_image_id = row_identifier
    issues: list[PreflightIssue] = []

    for column in sorted(REQUIRED_MANIFEST_COLUMNS):
        if _row_value(row, column_lookup, column) is None:
            issues.append(
                _row_issue(
                    row_number=row_number,
                    row_identifier=row_identifier,
                    source_path=source_value,
                    source_image_id=source_image_id,
                    field=column,
                    code="MISSING_REQUIRED_FIELD",
                    message=f"Required manifest field '{column}' is blank.",
                    severity=IssueSeverity.ERROR,
                    phase=RequirementPhase.PREFLIGHT_MANIFEST,
                )
            )

    if specimen_id is not None and slide_id is not None:
        identifier_key = (specimen_id, slide_id)
        first_row = seen_identifiers.get(identifier_key)
        if first_row is None:
            seen_identifiers[identifier_key] = row_number
        else:
            issues.append(
                _row_issue(
                    row_number=row_number,
                    row_identifier=row_identifier,
                    source_path=source_value,
                    source_image_id=source_image_id,
                    field="specimen_id,slide_id",
                    code="DUPLICATE_SOURCE_SLIDE_IDENTIFIER",
                    message=(
                        "Duplicate specimen_id/slide_id pair also appears on row "
                        f"{first_row}; this is a warning until an explicit unique "
                        "source-slide identifier is defined."
                    ),
                    severity=IssueSeverity.WARNING,
                    phase=RequirementPhase.PREFLIGHT_MANIFEST,
                )
            )

    if source_value is not None:
        if not source_extension_is_accepted(source_value, accepted_extensions):
            accepted_text = ", ".join(sorted(accepted_extensions))
            issues.append(
                _row_issue(
                    row_number=row_number,
                    row_identifier=row_identifier,
                    source_path=source_value,
                    source_image_id=source_image_id,
                    field="source_path",
                    code="DISALLOWED_SOURCE_EXTENSION",
                    message=(
                        f"Source extension '{source_extension_for_message(source_value)}' "
                        f"is not accepted by profile {profile.profile_name}; expected "
                        f"one of: {accepted_text}."
                    ),
                    severity=IssueSeverity.ERROR,
                    phase=RequirementPhase.SOURCE_FILE_PREFLIGHT,
                )
            )

        location = _source_location(source_value, manifest_path)
        if location.is_local:
            if location.path_to_check is not None and not location.path_to_check.exists():
                issues.append(
                    _row_issue(
                        row_number=row_number,
                        row_identifier=row_identifier,
                        source_path=source_value,
                        source_image_id=source_image_id,
                        field="source_path",
                        code="MISSING_LOCAL_SOURCE",
                        message=f"Local source path does not exist: {source_value}.",
                        severity=IssueSeverity.ERROR,
                        phase=RequirementPhase.SOURCE_FILE_PREFLIGHT,
                    )
                )
        else:
            issues.append(
                _row_issue(
                    row_number=row_number,
                    row_identifier=row_identifier,
                    source_path=source_value,
                    source_image_id=source_image_id,
                    field="source_path",
                    code="DEFERRED_NON_LOCAL_SOURCE_EXISTENCE",
                    message=(
                        "Source path is not a local filesystem path or file:// URI; "
                        "existence checking is deferred."
                    ),
                    severity=IssueSeverity.DEFERRED,
                    phase=RequirementPhase.SOURCE_FILE_PREFLIGHT,
                )
            )

    if not any(issue.code == "MISSING_REQUIRED_FIELD" for issue in issues):
        data = {
            column: _row_value(row, column_lookup, column)
            for column in SUPPORTED_MANIFEST_COLUMNS
            if column in column_lookup
        }
        try:
            SourceSlide.model_validate(data)
        except ValidationError as exc:
            for error in exc.errors():
                field = ".".join(str(part) for part in error.get("loc", ())) or None
                issues.append(
                    _row_issue(
                        row_number=row_number,
                        row_identifier=row_identifier,
                        source_path=source_value,
                        source_image_id=source_image_id,
                        field=field,
                        code="INVALID_SOURCE_SLIDE_RECORD",
                        message=f"Invalid source slide record: {error.get('msg', str(exc))}.",
                        severity=IssueSeverity.ERROR,
                        phase=RequirementPhase.PREFLIGHT_MANIFEST,
                    )
                )

    issues.extend(
        _row_requirement_issues(
            row,
            row_number=row_number,
            column_lookup=column_lookup,
            profile=profile,
            row_identifier=row_identifier,
            source_path=source_value,
            source_image_id=source_image_id,
        )
    )

    valid = not any(issue.severity is IssueSeverity.ERROR for issue in issues)
    return PreflightRowResult(
        row_number=row_number,
        row_identifier=row_identifier,
        source_path=source_value,
        source_image_id=source_image_id,
        valid=valid,
        preflight_status=_status_for_issues(issues),
        issues=issues,
    )


def _row_requirement_issues(
    row: Mapping[str, str | None],
    *,
    row_number: int,
    column_lookup: Mapping[str, str],
    profile: DatabaseProfile,
    row_identifier: str | None,
    source_path: str | None,
    source_image_id: str | None,
) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    for spec in ROW_REQUIREMENTS:
        if not spec.enabled(profile):
            continue
        phase = _requirement_phase(profile, spec)
        missing_fields = [
            field for field in spec.manifest_fields if _row_value(row, column_lookup, field) is None
        ]

        if phase is RequirementPhase.PREFLIGHT_MANIFEST:
            if missing_fields or not spec.manifest_fields:
                issues.append(
                    _row_issue(
                        row_number=row_number,
                        row_identifier=row_identifier,
                        source_path=source_path,
                        source_image_id=source_image_id,
                        field=", ".join(missing_fields) or spec.field,
                        code=spec.missing_code,
                        message=(
                            f"Profile marks '{spec.key}' as a preflight manifest "
                            "requirement, but the required manifest value is missing."
                        ),
                        severity=IssueSeverity.ERROR,
                        phase=phase,
                    )
                )
        elif missing_fields or not spec.manifest_fields:
            issues.append(
                _row_issue(
                    row_number=row_number,
                    row_identifier=row_identifier,
                    source_path=source_path,
                    source_image_id=source_image_id,
                    field=", ".join(missing_fields) or spec.field,
                    code=spec.deferred_code,
                    message=spec.deferred_message,
                    severity=IssueSeverity.DEFERRED,
                    phase=phase,
                )
            )
    return issues


def _batch_requirement_issues(profile: DatabaseProfile) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    for spec in BATCH_REQUIREMENTS:
        if not spec.enabled(profile):
            continue
        phase = _requirement_phase(profile, spec)
        if phase is RequirementPhase.PREFLIGHT_MANIFEST:
            issues.append(
                PreflightIssue(
                    field=spec.field,
                    code=spec.missing_code,
                    message=(
                        f"Profile marks '{spec.key}' as a preflight manifest requirement, "
                        "but it cannot be satisfied before later workflow artifacts exist."
                    ),
                    severity=IssueSeverity.ERROR,
                    requirement_phase=phase,
                )
            )
        else:
            issues.append(
                PreflightIssue(
                    field=spec.field,
                    code=spec.deferred_code,
                    message=spec.deferred_message,
                    severity=IssueSeverity.DEFERRED,
                    requirement_phase=phase,
                )
            )
    return issues


def _build_report(
    *,
    profile: DatabaseProfile,
    profile_path: Path,
    manifest_path: Path,
    generated_at: datetime,
    strict: bool,
    batch_issues: list[PreflightIssue],
    row_results: list[PreflightRowResult],
) -> PreflightReport:
    all_issues = list(batch_issues)
    for row in row_results:
        all_issues.extend(row.issues)

    counts = Counter(issue.severity for issue in all_issues)
    error_count = counts[IssueSeverity.ERROR]
    warning_count = counts[IssueSeverity.WARNING]
    deferred_count = counts[IssueSeverity.DEFERRED]
    batch_status = _status_from_counts(
        error_count=error_count,
        warning_count=warning_count,
        deferred_count=deferred_count,
    )
    return PreflightReport(
        generated_at=generated_at,
        profile_path=profile_path,
        manifest_path=manifest_path,
        profile_identifier=_profile_identifier(profile),
        profile_name=profile.profile_name,
        profile_version=profile.profile_version,
        total_row_count=len(row_results),
        valid_row_count=sum(1 for row in row_results if row.valid),
        rows_with_warnings=sum(
            1
            for row in row_results
            if any(issue.severity is IssueSeverity.WARNING for issue in row.issues)
        ),
        rows_with_deferred_requirements=sum(
            1
            for row in row_results
            if any(issue.severity is IssueSeverity.DEFERRED for issue in row.issues)
        ),
        rows_with_errors=sum(
            1
            for row in row_results
            if any(issue.severity is IssueSeverity.ERROR for issue in row.issues)
        ),
        warning_count=warning_count,
        deferred_count=deferred_count,
        error_count=error_count,
        batch_status=batch_status,
        ready_for_next_stage=error_count == 0,
        strict=strict,
        batch_issues=batch_issues,
        row_results=row_results,
    )


def _build_state(report: PreflightReport, json_report_path: Path | None) -> PreflightState:
    return PreflightState(
        batch_id=_batch_id(report.profile_identifier, report.manifest_path),
        generated_at=report.generated_at,
        profile_identifier=report.profile_identifier,
        manifest_path=report.manifest_path,
        preflight_status=report.batch_status,
        row_statuses=[
            PreflightStateRow(
                row_number=row.row_number,
                row_identifier=row.row_identifier,
                source_path=row.source_path,
                source_image_id=row.source_image_id,
                preflight_status=row.preflight_status,
                valid=row.valid,
                issue_codes=[issue.code for issue in row.issues],
                issues=[
                    PreflightStateIssue(
                        code=issue.code,
                        severity=issue.severity,
                        message=issue.message,
                        field=issue.field,
                        requirement_phase=issue.requirement_phase,
                    )
                    for issue in row.issues
                ],
            )
            for row in report.row_results
        ],
        json_report_path=json_report_path,
    )


def build_preflight_state(
    report: PreflightReport,
    json_report_path: str | Path | None = None,
) -> PreflightState:
    """Build an in-memory preflight state from a report without writing files."""
    json_path = Path(json_report_path) if json_report_path is not None else None
    return _build_state(report, json_path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _status_for_issues(issues: list[PreflightIssue]) -> BatchStatus:
    counts = Counter(issue.severity for issue in issues)
    return _status_from_counts(
        error_count=counts[IssueSeverity.ERROR],
        warning_count=counts[IssueSeverity.WARNING],
        deferred_count=counts[IssueSeverity.DEFERRED],
    )


def _status_from_counts(
    *,
    error_count: int,
    warning_count: int,
    deferred_count: int,
) -> BatchStatus:
    if error_count:
        return BatchStatus.PREFLIGHT_FAILED
    if deferred_count:
        return BatchStatus.PREFLIGHT_PASSED_WITH_DEFERRED_REQUIREMENTS
    if warning_count:
        return BatchStatus.PREFLIGHT_PASSED_WITH_WARNINGS
    return BatchStatus.PREFLIGHT_PASSED


def _row_value(
    row: Mapping[str, str | None],
    column_lookup: Mapping[str, str],
    column: str,
) -> str | None:
    original_name = column_lookup.get(column)
    if original_name is None:
        return None
    return _blank_to_none(row.get(original_name))


def _row_identifier(specimen_id: str | None, slide_id: str | None) -> str | None:
    if specimen_id is None or slide_id is None:
        return None
    return f"{specimen_id}/{slide_id}"


def _row_issue(
    *,
    row_number: int,
    row_identifier: str | None,
    source_path: str | None,
    source_image_id: str | None,
    field: str | None,
    code: str,
    message: str,
    severity: IssueSeverity,
    phase: RequirementPhase,
) -> PreflightIssue:
    return PreflightIssue(
        row_number=row_number,
        row_identifier=row_identifier,
        source_path=source_path,
        source_image_id=source_image_id,
        field=field,
        code=code,
        message=message,
        severity=severity,
        requirement_phase=phase,
    )


def _requirement_phase(profile: DatabaseProfile, spec: _RequirementSpec) -> RequirementPhase:
    for key in (spec.key, *spec.aliases):
        phase = profile.requirement_phases.get(key)
        if phase is not None:
            return phase
    return spec.default_phase


def _source_location(source_value: str, manifest_path: Path) -> _SourceLocation:
    parsed = urlparse(source_value)
    if parsed.scheme and parsed.scheme != "file":
        return _SourceLocation(is_local=False)

    if parsed.scheme == "file":
        path_text = unquote(parsed.path)
        if parsed.netloc and parsed.netloc != "localhost":
            path_text = f"//{parsed.netloc}{path_text}"
        return _SourceLocation(is_local=True, path_to_check=Path(path_text))

    source_path = Path(source_value)
    path_to_check = source_path if source_path.is_absolute() else manifest_path.parent / source_path
    return _SourceLocation(is_local=True, path_to_check=path_to_check)


def _profile_identifier(profile: DatabaseProfile) -> str:
    return f"{profile.profile_name}@{profile.profile_version}"


def _batch_id(profile_identifier: str, manifest_path: Path) -> str:
    digest = hashlib.sha256(f"{profile_identifier}|{manifest_path}".encode()).hexdigest()
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", profile_identifier).strip("-").lower()
    return f"preflight-{slug}-{digest[:8]}"


__all__ = [
    "IssueSeverity",
    "PREFLIGHT_REPORT_VERSION",
    "PREFLIGHT_STATE_VERSION",
    "PreflightIssue",
    "PreflightReport",
    "PreflightRunResult",
    "PreflightState",
    "PreflightStateIssue",
    "PreflightStateRow",
    "PreflightRowResult",
    "build_preflight_state",
    "run_preflight",
]
