"""Dry-run tissue-detection planning for preflighted submission rows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import Field, ValidationError

from .models import SubmissionBaseModel
from .preflight import IssueSeverity, PreflightState, PreflightStateIssue
from .statuses import TissuePlanRowCategory, TissuePlanStatus

TISSUE_PLAN_VERSION = "1.0"
PLANNED_TISSUE_STAGE = "tissue_detection"
PLANNED_JOB_STATUS = "PLANNED"


class TissuePlanError(ValueError):
    """Raised when a tissue-detection plan cannot be created from state."""


class TissuePlanJobInputs(SubmissionBaseModel):
    """Inputs a future tissue-detection job would consume."""

    source_path: str
    profile_identifier: str


class TissuePlanJobOutputs(SubmissionBaseModel):
    """Named future outputs without committing to output schemas."""

    tissue_table: str | None = None
    qc_contact_sheet: str | None = None
    review_state: str | None = None


class TissuePlanJob(SubmissionBaseModel):
    """Dry-run record for one future tissue-detection job."""

    job_id: str
    row_number: int
    row_identifier: str | None = None
    source_image_id: str | None = None
    source_path: str
    planned_stage: str = PLANNED_TISSUE_STAGE
    status: str = PLANNED_JOB_STATUS
    category: TissuePlanRowCategory = TissuePlanRowCategory.ELIGIBLE_FOR_TISSUE_DETECTION
    inputs: TissuePlanJobInputs
    outputs_planned: TissuePlanJobOutputs = Field(default_factory=TissuePlanJobOutputs)
    deferred_requirements: list[PreflightStateIssue] = Field(default_factory=list)


class TissuePlanRowSummary(SubmissionBaseModel):
    """Summary for a row that does not become a local planned job."""

    row_number: int
    row_identifier: str | None = None
    source_image_id: str | None = None
    source_path: str | None = None
    category: TissuePlanRowCategory
    reason_code: str
    message: str
    issues: list[PreflightStateIssue] = Field(default_factory=list)


class TissueDetectionPlan(SubmissionBaseModel):
    """Machine-readable dry-run plan for future tissue detection."""

    plan_version: str = TISSUE_PLAN_VERSION
    generated_at: datetime
    source_state_path: Path
    batch_id: str
    profile_identifier: str
    plan_status: TissuePlanStatus
    total_rows: int
    eligible_job_count: int
    blocked_row_count: int
    deferred_row_count: int
    skipped_row_count: int
    jobs: list[TissuePlanJob] = Field(default_factory=list)
    blocked_rows: list[TissuePlanRowSummary] = Field(default_factory=list)
    skipped_rows: list[TissuePlanRowSummary] = Field(default_factory=list)


@dataclass(frozen=True)
class TissuePlanRunResult:
    """Return object for a tissue-planning run."""

    plan: TissueDetectionPlan
    plan_out_path: Path | None = None


def plan_tissues_from_state(
    state_path: str | Path,
    *,
    plan_out_path: str | Path | None = None,
    generated_at: datetime | None = None,
) -> TissuePlanRunResult:
    """Create a deterministic dry-run tissue-detection plan from preflight state."""
    state_path = Path(state_path)
    state = load_preflight_state(state_path)
    plan = build_tissue_detection_plan(
        state,
        source_state_path=state_path,
        generated_at=generated_at,
    )

    output_path = Path(plan_out_path) if plan_out_path is not None else None
    if output_path is not None:
        _write_json(output_path, plan.to_dict())

    return TissuePlanRunResult(plan=plan, plan_out_path=output_path)


def load_preflight_state(state_path: str | Path) -> PreflightState:
    """Load a preflight state JSON file."""
    path = Path(state_path)
    if not path.exists():
        raise FileNotFoundError(f"Preflight state not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TissuePlanError(f"Malformed preflight state JSON {path}: {exc.msg}") from exc

    try:
        return PreflightState.from_dict(payload)
    except ValidationError as exc:
        raise TissuePlanError(f"Invalid preflight state JSON {path}: {exc}") from exc


def build_tissue_detection_plan(
    state: PreflightState,
    *,
    source_state_path: str | Path,
    generated_at: datetime | None = None,
) -> TissueDetectionPlan:
    """Build a dry-run tissue plan from an already-loaded state object."""
    jobs: list[TissuePlanJob] = []
    blocked_rows: list[TissuePlanRowSummary] = []
    skipped_rows: list[TissuePlanRowSummary] = []

    for row in state.row_statuses:
        error_issues = _issues_with_severity(row.issues, IssueSeverity.ERROR)
        if error_issues or _legacy_row_has_error(row.valid, row.preflight_status.value):
            blocked_rows.append(_blocked_row_summary(row, error_issues))
            continue

        if not row.source_path:
            skipped_rows.append(
                _skipped_row_summary(
                    row_number=row.row_number,
                    row_identifier=row.row_identifier,
                    source_image_id=row.source_image_id,
                    source_path=row.source_path,
                    category=TissuePlanRowCategory.SKIPPED_UNSUPPORTED_OR_UNKNOWN,
                    reason_code="MISSING_SOURCE_PATH_IN_STATE",
                    message="Preflight state row does not include a source path to plan.",
                    issues=row.issues,
                )
            )
            continue

        if _is_nonlocal_source(row.source_path):
            skipped_rows.append(_nonlocal_row_summary(row))
            continue

        deferred_requirements = _issues_with_severity(row.issues, IssueSeverity.DEFERRED)
        category = (
            TissuePlanRowCategory.DEFERRED_SOURCE_METADATA
            if deferred_requirements
            else TissuePlanRowCategory.ELIGIBLE_FOR_TISSUE_DETECTION
        )
        jobs.append(
            TissuePlanJob(
                job_id=_job_id(state.batch_id, row.row_number, row.row_identifier, row.source_path),
                row_number=row.row_number,
                row_identifier=row.row_identifier,
                source_image_id=row.source_image_id or row.row_identifier,
                source_path=row.source_path,
                category=category,
                inputs=TissuePlanJobInputs(
                    source_path=row.source_path,
                    profile_identifier=state.profile_identifier,
                ),
                deferred_requirements=deferred_requirements,
            )
        )

    deferred_row_count = sum(1 for job in jobs if job.deferred_requirements) + sum(
        1 for row in skipped_rows if row.category is TissuePlanRowCategory.SKIPPED_NONLOCAL_SOURCE
    )
    plan_status = _plan_status(
        total_rows=len(state.row_statuses),
        jobs=jobs,
        blocked_rows=blocked_rows,
        skipped_rows=skipped_rows,
        deferred_row_count=deferred_row_count,
    )

    return TissueDetectionPlan(
        generated_at=generated_at or datetime.now(timezone.utc),
        source_state_path=Path(source_state_path),
        batch_id=state.batch_id,
        profile_identifier=state.profile_identifier,
        plan_status=plan_status,
        total_rows=len(state.row_statuses),
        eligible_job_count=len(jobs),
        blocked_row_count=len(blocked_rows),
        deferred_row_count=deferred_row_count,
        skipped_row_count=len(skipped_rows),
        jobs=jobs,
        blocked_rows=blocked_rows,
        skipped_rows=skipped_rows,
    )


def _issues_with_severity(
    issues: list[PreflightStateIssue], severity: IssueSeverity
) -> list[PreflightStateIssue]:
    return [issue for issue in issues if issue.severity is severity]


def _legacy_row_has_error(valid: bool, preflight_status: str) -> bool:
    return not valid or preflight_status == "PREFLIGHT_FAILED"


def _blocked_row_summary(row: Any, error_issues: list[PreflightStateIssue]) -> TissuePlanRowSummary:
    issue = error_issues[0] if error_issues else None
    reason_code = issue.code if issue is not None else "PREFLIGHT_ROW_INVALID"
    message = (
        issue.message
        if issue is not None
        else "Preflight state marks this row invalid or failed without issue details."
    )
    return TissuePlanRowSummary(
        row_number=row.row_number,
        row_identifier=row.row_identifier,
        source_image_id=row.source_image_id,
        source_path=row.source_path,
        category=TissuePlanRowCategory.BLOCKED_BY_PREFLIGHT_ERROR,
        reason_code=reason_code,
        message=message,
        issues=row.issues,
    )


def _nonlocal_row_summary(row: Any) -> TissuePlanRowSummary:
    deferred_issues = _issues_with_severity(row.issues, IssueSeverity.DEFERRED)
    nonlocal_issue = next(
        (issue for issue in deferred_issues if issue.code == "DEFERRED_NON_LOCAL_SOURCE_EXISTENCE"),
        None,
    )
    return _skipped_row_summary(
        row_number=row.row_number,
        row_identifier=row.row_identifier,
        source_image_id=row.source_image_id,
        source_path=row.source_path,
        category=TissuePlanRowCategory.SKIPPED_NONLOCAL_SOURCE,
        reason_code=nonlocal_issue.code if nonlocal_issue is not None else "NON_LOCAL_SOURCE",
        message=(
            nonlocal_issue.message
            if nonlocal_issue is not None
            else "Non-local source URIs are skipped by this local tissue-planning dry run."
        ),
        issues=row.issues,
    )


def _skipped_row_summary(
    *,
    row_number: int,
    row_identifier: str | None,
    source_image_id: str | None,
    source_path: str | None,
    category: TissuePlanRowCategory,
    reason_code: str,
    message: str,
    issues: list[PreflightStateIssue],
) -> TissuePlanRowSummary:
    return TissuePlanRowSummary(
        row_number=row_number,
        row_identifier=row_identifier,
        source_image_id=source_image_id,
        source_path=source_path,
        category=category,
        reason_code=reason_code,
        message=message,
        issues=issues,
    )


def _is_nonlocal_source(source_path: str) -> bool:
    parsed = urlparse(source_path)
    return bool(parsed.scheme and parsed.scheme != "file")


def _job_id(
    batch_id: str,
    row_number: int,
    row_identifier: str | None,
    source_path: str | None,
) -> str:
    fingerprint = "|".join(
        [
            batch_id,
            str(row_number),
            row_identifier or "",
            source_path or "",
            PLANNED_TISSUE_STAGE,
        ]
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    return f"tissue-detection-row-{row_number:05d}-{digest[:12]}"


def _plan_status(
    *,
    total_rows: int,
    jobs: list[TissuePlanJob],
    blocked_rows: list[TissuePlanRowSummary],
    skipped_rows: list[TissuePlanRowSummary],
    deferred_row_count: int,
) -> TissuePlanStatus:
    if total_rows == 0:
        return TissuePlanStatus.TISSUE_PLAN_EMPTY
    if jobs:
        if not blocked_rows and not skipped_rows and deferred_row_count == 0:
            return TissuePlanStatus.TISSUE_PLAN_READY
        return TissuePlanStatus.TISSUE_PLAN_READY_WITH_DEFERRED_REQUIREMENTS
    if blocked_rows and len(blocked_rows) == total_rows:
        return TissuePlanStatus.TISSUE_PLAN_BLOCKED
    return TissuePlanStatus.TISSUE_PLAN_EMPTY


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "PLANNED_TISSUE_STAGE",
    "TISSUE_PLAN_VERSION",
    "TissueDetectionPlan",
    "TissuePlanError",
    "TissuePlanJob",
    "TissuePlanJobInputs",
    "TissuePlanJobOutputs",
    "TissuePlanRowSummary",
    "TissuePlanRunResult",
    "build_tissue_detection_plan",
    "load_preflight_state",
    "plan_tissues_from_state",
]
