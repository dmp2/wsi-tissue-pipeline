"""Structural-only validation for existing OME-TIFF submission batches."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field

from .models import SubmissionBaseModel
from .preflight import IssueSeverity
from .profiles import RequirementPhase, load_database_profile
from .setup import (
    OMETIFF_SUFFIXES,
    TIFF_SUFFIXES,
    SetupIssue,
    SetupRowSummary,
    WorkflowMode,
    run_setup,
)
from .source_paths import source_location, source_target_for_extension

OMETIFF_VALIDATION_REPORT_VERSION = "1.0"
OMETIFF_VALIDATION_SCOPE = "filesystem_and_manifest_only"


class OmetiffValidationIssue(SubmissionBaseModel):
    """One structural-only existing OME-TIFF batch finding."""

    row_number: int | None = None
    row_identifier: str | None = None
    source_path: str | None = None
    source_image_id: str | None = None
    field: str | None = None
    code: str
    message: str
    severity: IssueSeverity
    requirement_phase: RequirementPhase | None = None
    source: str = "validate_ometiff"


class OmetiffValidationRowSummary(SubmissionBaseModel):
    """Row-level summary for a structural-only OME-TIFF filesystem check."""

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
    issues: list[OmetiffValidationIssue] = Field(default_factory=list)


class OmetiffValidationReport(SubmissionBaseModel):
    """Machine-readable structural-only report for existing OME-TIFF batches."""

    validation_report_version: str = OMETIFF_VALIDATION_REPORT_VERSION
    validation_scope: str = OMETIFF_VALIDATION_SCOPE
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
    ready_for_next_action: bool
    next_action: str
    recommended_next_action: str
    strict: bool
    row_summaries: list[OmetiffValidationRowSummary] = Field(default_factory=list)
    issues: list[OmetiffValidationIssue] = Field(default_factory=list)


@dataclass(frozen=True)
class OmetiffValidationRunResult:
    """Return object for an existing OME-TIFF structural check."""

    report: OmetiffValidationReport
    validation_report_path: Path | None = None

    def exit_code(self, *, strict: bool | None = None) -> int:
        """Return the CLI exit code for this structural check."""
        strict_mode = self.report.strict if strict is None else strict
        if self.report.error_count:
            return 1
        if strict_mode and (self.report.warning_count or self.report.deferred_count):
            return 1
        return 0


def run_validate_ometiff(
    profile_path: str | Path,
    manifest_path: str | Path,
    *,
    validation_report_path: str | Path | None = None,
    strict: bool = False,
    generated_at: datetime | None = None,
) -> OmetiffValidationRunResult:
    """Run structural-only checks for existing OME-TIFF-like submission files."""
    profile_path = Path(profile_path)
    manifest_path = Path(manifest_path)
    profile = load_database_profile(profile_path)
    accepted_extensions = profile.accepted_extensions_for_workflow_mode(
        WorkflowMode.EXISTING_OMETIFF_UPLOAD.value
    )
    setup = run_setup(
        profile_path,
        manifest_path,
        WorkflowMode.EXISTING_OMETIFF_UPLOAD,
        strict=strict,
        generated_at=generated_at,
    )

    row_summaries: list[OmetiffValidationRowSummary] = []
    issues: list[OmetiffValidationIssue] = [
        _validation_issue_from_setup(issue)
        for issue in setup.report.issues
        if issue.row_number is None
    ]
    known_input_bytes = 0

    for setup_row in setup.report.row_summaries:
        row_issues = [_validation_issue_from_setup(issue) for issue in setup_row.issues]
        known_row_bytes, structural_issues = _structural_file_checks(
            setup_row,
            manifest_path=manifest_path,
            accepted_extensions=accepted_extensions,
            existing_codes={issue.code for issue in row_issues},
        )
        row_issues.extend(structural_issues)
        if known_row_bytes is not None:
            known_input_bytes += known_row_bytes

        counts = Counter(issue.severity for issue in row_issues)
        row_summaries.append(
            OmetiffValidationRowSummary(
                row_number=setup_row.row_number,
                row_identifier=setup_row.row_identifier,
                source_image_id=setup_row.source_image_id,
                source_path=setup_row.source_path,
                valid=counts[IssueSeverity.ERROR] == 0,
                blocked=counts[IssueSeverity.ERROR] > 0,
                deferred=counts[IssueSeverity.DEFERRED] > 0,
                warning=counts[IssueSeverity.WARNING] > 0,
                known_input_bytes=known_row_bytes,
                input_size_status=_input_size_status(setup_row, known_row_bytes, row_issues),
                mode_compatible=not any(
                    issue.code == "OMETIFF_WRONG_MODE_EXTENSION" for issue in row_issues
                ),
                issues=row_issues,
            )
        )
        issues.extend(row_issues)

    issue_counts = Counter(issue.severity for issue in issues)
    error_count = issue_counts[IssueSeverity.ERROR]
    warning_count = issue_counts[IssueSeverity.WARNING]
    deferred_count = issue_counts[IssueSeverity.DEFERRED]
    ready_for_next_action = error_count == 0 and not (strict and (warning_count or deferred_count))
    next_action, recommended_next_action = _next_action_and_recommendation(
        issues=issues,
        error_count=error_count,
        warning_count=warning_count,
        deferred_count=deferred_count,
        strict=strict,
    )

    report = OmetiffValidationReport(
        generated_at=generated_at or datetime.now(timezone.utc),
        workflow_mode=WorkflowMode.EXISTING_OMETIFF_UPLOAD,
        profile_identifier=setup.report.profile_identifier,
        profile_name=setup.report.profile_name,
        profile_version=setup.report.profile_version,
        profile_path=profile_path,
        manifest_path=manifest_path,
        row_count=setup.report.row_count,
        valid_row_count=sum(1 for row in row_summaries if row.valid),
        blocked_row_count=sum(1 for row in row_summaries if row.blocked),
        deferred_row_count=sum(1 for row in row_summaries if row.deferred),
        warning_count=warning_count,
        deferred_count=deferred_count,
        error_count=error_count,
        known_input_bytes=known_input_bytes,
        ready_for_next_action=ready_for_next_action,
        next_action=next_action,
        recommended_next_action=recommended_next_action,
        strict=strict,
        row_summaries=row_summaries,
        issues=issues,
    )

    output_path = Path(validation_report_path) if validation_report_path is not None else None
    if output_path is not None:
        _write_json(output_path, report.to_dict())

    return OmetiffValidationRunResult(report=report, validation_report_path=output_path)


def _validation_issue_from_setup(issue: SetupIssue) -> OmetiffValidationIssue:
    code = _ISSUE_CODE_MAP.get(issue.code, issue.code)
    return OmetiffValidationIssue(
        row_number=issue.row_number,
        row_identifier=issue.row_identifier,
        source_path=issue.source_path,
        source_image_id=issue.source_image_id,
        field=issue.field,
        code=code,
        message=_ISSUE_MESSAGES.get(code, issue.message),
        severity=issue.severity,
        requirement_phase=issue.requirement_phase,
        source=issue.source,
    )


def _structural_file_checks(
    row: SetupRowSummary,
    *,
    manifest_path: Path,
    accepted_extensions: list[str],
    existing_codes: set[str],
) -> tuple[int | None, list[OmetiffValidationIssue]]:
    if not row.source_path:
        return None, []

    issues: list[OmetiffValidationIssue] = []
    if (
        "OMETIFF_WRONG_MODE_EXTENSION" not in existing_codes
        and not _existing_ometiff_extension_is_allowed(row.source_path, accepted_extensions)
    ):
        issues.append(
            _row_issue(
                row,
                code="OMETIFF_WRONG_MODE_EXTENSION",
                message=(
                    "Existing OME-TIFF upload mode expects an OME-TIFF-like suffix, "
                    "or a generic TIFF suffix explicitly allowed by the profile."
                ),
                severity=IssueSeverity.ERROR,
                phase=RequirementPhase.SOURCE_FILE_PREFLIGHT,
            )
        )

    location = source_location(row.source_path, manifest_path)
    if not location.is_local:
        if "OMETIFF_NONLOCAL_DEFERRED" not in existing_codes:
            issues.append(
                _row_issue(
                    row,
                    code="OMETIFF_NONLOCAL_DEFERRED",
                    message=(
                        "Source path is not local; filesystem structure checks are deferred "
                        "for this row."
                    ),
                    severity=IssueSeverity.DEFERRED,
                    phase=RequirementPhase.SOURCE_FILE_PREFLIGHT,
                )
            )
        return None, issues

    path_to_check = location.path_to_check
    if path_to_check is None or not path_to_check.exists():
        if "OMETIFF_MISSING_LOCAL_FILE" not in existing_codes:
            issues.append(
                _row_issue(
                    row,
                    code="OMETIFF_MISSING_LOCAL_FILE",
                    message=f"Local OME-TIFF-like source path does not exist: {row.source_path}.",
                    severity=IssueSeverity.ERROR,
                    phase=RequirementPhase.SOURCE_FILE_PREFLIGHT,
                )
            )
        return None, issues

    if not path_to_check.is_file():
        issues.append(
            _row_issue(
                row,
                code="OMETIFF_NOT_REGULAR_FILE",
                message=f"Local OME-TIFF-like source path is not a regular file: {row.source_path}.",
                severity=IssueSeverity.ERROR,
                phase=RequirementPhase.SOURCE_FILE_PREFLIGHT,
            )
        )
        return None, issues

    size = path_to_check.stat().st_size
    if size == 0:
        issues.append(
            _row_issue(
                row,
                code="OMETIFF_EMPTY_FILE",
                message=f"Local OME-TIFF-like source file is empty: {row.source_path}.",
                severity=IssueSeverity.ERROR,
                phase=RequirementPhase.SOURCE_FILE_PREFLIGHT,
            )
        )
    return size, issues


def _existing_ometiff_extension_is_allowed(
    source_path: str, accepted_extensions: list[str]
) -> bool:
    target = source_target_for_extension(source_path)
    lower_name = Path(target).name.lower()
    suffix = Path(target).suffix.lower()
    accepted = {extension.lower() for extension in accepted_extensions}
    if lower_name.endswith(OMETIFF_SUFFIXES):
        return any(lower_name.endswith(extension) for extension in accepted)
    if suffix in TIFF_SUFFIXES:
        return any(lower_name.endswith(extension) for extension in accepted)
    return False


def _input_size_status(
    row: SetupRowSummary,
    known_row_bytes: int | None,
    row_issues: list[OmetiffValidationIssue],
) -> str:
    if known_row_bytes is not None:
        return "known"
    if any(issue.code == "OMETIFF_NONLOCAL_DEFERRED" for issue in row_issues):
        return "unknown_non_local_source"
    if any(issue.code == "OMETIFF_NOT_REGULAR_FILE" for issue in row_issues):
        return "unknown_not_regular_file"
    if any(issue.code == "OMETIFF_MISSING_LOCAL_FILE" for issue in row_issues):
        return "unknown_missing_local_source"
    return row.input_size_status


def _row_issue(
    row: SetupRowSummary,
    *,
    code: str,
    message: str,
    severity: IssueSeverity,
    phase: RequirementPhase,
) -> OmetiffValidationIssue:
    return OmetiffValidationIssue(
        row_number=row.row_number,
        row_identifier=row.row_identifier,
        source_path=row.source_path,
        source_image_id=row.source_image_id,
        field="source_path",
        code=code,
        message=message,
        severity=severity,
        requirement_phase=phase,
    )


def _next_action_and_recommendation(
    *,
    issues: list[OmetiffValidationIssue],
    error_count: int,
    warning_count: int,
    deferred_count: int,
    strict: bool,
) -> tuple[str, str]:
    if error_count:
        return "fix_structural_issues", "Fix blocking filesystem/manifest issues before continuing."
    if any(issue.code == "OMETIFF_METADATA_VALIDATION_DEFERRED" for issue in issues):
        return (
            "ome_tiff_metadata_validation",
            "Run future OME-TIFF metadata validation before packaging/upload.",
        )
    if strict and (warning_count or deferred_count):
        return (
            "resolve_warnings_or_deferred_checks",
            "Resolve warnings or deferred checks, or rerun without strict mode.",
        )
    if deferred_count:
        return "resolve_deferred_checks", "Resolve deferred checks before packaging/upload."
    if warning_count:
        return "review_warnings", "Review warnings before future package planning."
    return (
        "package_planning",
        "Ready for future package planning after required metadata validation policy is satisfied.",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


_ISSUE_CODE_MAP = {
    "DISALLOWED_SOURCE_EXTENSION": "OMETIFF_WRONG_MODE_EXTENSION",
    "MODE_INPUT_EXTENSION_MISMATCH": "OMETIFF_WRONG_MODE_EXTENSION",
    "WRONG_MODE_FOR_OMETIFF_UPLOAD": "OMETIFF_WRONG_MODE_EXTENSION",
    "MISSING_LOCAL_SOURCE": "OMETIFF_MISSING_LOCAL_FILE",
    "DEFERRED_NON_LOCAL_SOURCE_EXISTENCE": "OMETIFF_NONLOCAL_DEFERRED",
    "DEFERRED_OME_METADATA_VALIDATION": "OMETIFF_METADATA_VALIDATION_DEFERRED",
}

_ISSUE_MESSAGES = {
    "OMETIFF_WRONG_MODE_EXTENSION": (
        "Existing OME-TIFF upload mode expects an OME-TIFF-like suffix, or a generic "
        "TIFF suffix explicitly allowed by the profile."
    ),
    "OMETIFF_MISSING_LOCAL_FILE": "Local OME-TIFF-like source path does not exist.",
    "OMETIFF_NONLOCAL_DEFERRED": (
        "Source path is not local; filesystem structure checks are deferred for this row."
    ),
    "OMETIFF_METADATA_VALIDATION_DEFERRED": (
        "OME-TIFF metadata validation is required by the profile, but this structural "
        "check does not open TIFF files or parse OME-XML."
    ),
}

__all__ = [
    "OMETIFF_VALIDATION_REPORT_VERSION",
    "OMETIFF_VALIDATION_SCOPE",
    "OmetiffValidationIssue",
    "OmetiffValidationReport",
    "OmetiffValidationRowSummary",
    "OmetiffValidationRunResult",
    "run_validate_ometiff",
]
