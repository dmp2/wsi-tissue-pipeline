"""Dry-run package planning for existing OME-TIFF submission batches."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Formatter
from typing import Any

from pydantic import Field

from .models import SubmissionBaseModel
from .ometiff_validation import OMETIFF_VALIDATION_SCOPE, OmetiffValidationIssue
from .preflight import IssueSeverity
from .profiles import DatabaseProfile, RequirementPhase, load_database_profile
from .setup import OMETIFF_SUFFIXES
from .source_paths import source_location, source_target_for_extension
from .validation import ManifestValidationError

PACKAGE_PLAN_VERSION = "1.0"
PACKAGE_PLAN_FILENAME = "package_plan.json"
PACKAGE_MANIFEST_FILENAME = "package_manifest.csv"
PACKAGE_SUMMARY_FILENAME = "package_summary.txt"

PACKAGE_STATUS_READY = "ready"
PACKAGE_STATUS_READY_WITH_WARNINGS = "ready_with_warnings"
PACKAGE_STATUS_BLOCKED = "blocked"

PACKAGE_NAME_TEMPLATE_UNRESOLVED_PLACEHOLDER = "PACKAGE_NAME_TEMPLATE_UNRESOLVED_PLACEHOLDER"
PACKAGE_NAME_COLLISION = "PACKAGE_NAME_COLLISION"

_PACKAGE_IDENTIFIER_FIELDS = (
    "specimen_id",
    "sample_id",
    "slide_id",
    "source_image_id",
    "tissue_id",
    "section_id",
    "row_id",
    "block_id",
    "section_number",
    "stain",
)
_TISSUE_IDENTIFIER_PLACEHOLDERS = {"tissue_id", "section_id", "row_id"}
_DEFERRED_CAPABILITIES = [
    "copy/link/upload package execution is not implemented",
    "actual image payload packaging is not implemented",
    "database upload integration is not implemented",
]
_OUT_OF_SCOPE = [
    "copying, symlinking, hardlinking, linking, or modifying image files",
    "reading image pixels or parsing TIFF headers",
    "parsing OME-XML or validating OME-TIFF metadata",
    "computing checksums over image file bytes",
    "uploading to a database or launching GUI/viewer tools",
]


class OmetiffPackagePlanIssue(SubmissionBaseModel):
    """One package-planning finding."""

    row_number: int | None = None
    row_identifier: str | None = None
    source_path: str | None = None
    source_image_id: str | None = None
    field: str | None = None
    code: str
    message: str
    severity: IssueSeverity
    requirement_phase: RequirementPhase | None = None
    source: str = "package_ometiff"


class OmetiffPackagePlanRow(SubmissionBaseModel):
    """Dry-run package plan for one manifest row."""

    row_number: int
    row_identifier: str | None = None
    source_image_id: str | None = None
    source_path: str | None = None
    source_exists: bool | None = None
    source_is_local: bool
    file_size_bytes: int | None = None
    planned_package_name: str | None = None
    identifiers: dict[str, str | None] = Field(default_factory=dict)
    package_status: str
    blockers: list[OmetiffPackagePlanIssue] = Field(default_factory=list)
    warnings: list[OmetiffPackagePlanIssue] = Field(default_factory=list)
    deferred_checks: list[OmetiffPackagePlanIssue] = Field(default_factory=list)


class OmetiffPackagePlanReport(SubmissionBaseModel):
    """Machine-readable dry-run package plan for existing OME-TIFF batches."""

    package_plan_version: str = PACKAGE_PLAN_VERSION
    generated_at: datetime
    profile_identifier: str
    profile_name: str
    profile_version: str
    profile_path: Path
    manifest_path: Path
    output_dir: Path
    row_count: int
    ready_count: int
    ready_with_warnings_count: int
    blocked_count: int
    deferred_check_count: int
    total_known_input_bytes: int
    validation_scope: str = OMETIFF_VALIDATION_SCOPE
    ready_for_next_action: bool
    next_action: str
    recommended_next_action: str
    strict: bool
    deferred_capabilities: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    package_plan_path: Path
    package_manifest_path: Path
    package_summary_path: Path
    rows: list[OmetiffPackagePlanRow] = Field(default_factory=list)
    issues: list[OmetiffPackagePlanIssue] = Field(default_factory=list)


@dataclass(frozen=True)
class OmetiffPackagePlanRunResult:
    """Return object for an OME-TIFF package dry run."""

    report: OmetiffPackagePlanReport
    package_plan_path: Path
    package_manifest_path: Path
    package_summary_path: Path

    def exit_code(self, *, strict: bool | None = None) -> int:
        """Return the CLI exit code for this package dry run."""
        strict_mode = self.report.strict if strict is None else strict
        if self.report.blocked_count:
            return 1
        if strict_mode and any(row.warnings or row.deferred_checks for row in self.report.rows):
            return 1
        return 0


def run_package_ometiff_dry_run(
    profile_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    strict: bool = False,
    generated_at: datetime | None = None,
) -> OmetiffPackagePlanRunResult:
    """Create dry-run package planning artifacts for an existing OME-TIFF batch."""
    from .ometiff_validation import run_validate_ometiff

    profile_path = Path(profile_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)

    profile = load_database_profile(profile_path)
    validation = run_validate_ometiff(
        profile_path,
        manifest_path,
        strict=strict,
        generated_at=generated_at,
    )
    manifest_rows = _read_manifest_rows_by_number(manifest_path)

    rows = [
        _package_row_from_validation(
            validation_row,
            profile=profile,
            manifest_path=manifest_path,
            manifest_row=manifest_rows.get(validation_row.row_number, {}),
        )
        for validation_row in validation.report.row_summaries
    ]
    _apply_package_name_collisions(rows)

    package_plan_path = output_dir / PACKAGE_PLAN_FILENAME
    package_manifest_path = output_dir / PACKAGE_MANIFEST_FILENAME
    package_summary_path = output_dir / PACKAGE_SUMMARY_FILENAME
    report = _build_report(
        profile=profile,
        profile_path=profile_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        generated_at=generated_at or datetime.now(timezone.utc),
        strict=strict,
        rows=rows,
        package_plan_path=package_plan_path,
        package_manifest_path=package_manifest_path,
        package_summary_path=package_summary_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(package_plan_path, report.to_dict())
    _write_package_manifest(package_manifest_path, report.rows)
    _write_summary(package_summary_path, report)

    return OmetiffPackagePlanRunResult(
        report=report,
        package_plan_path=package_plan_path,
        package_manifest_path=package_manifest_path,
        package_summary_path=package_summary_path,
    )


def _package_row_from_validation(
    validation_row: Any,
    *,
    profile: DatabaseProfile,
    manifest_path: Path,
    manifest_row: Mapping[str, str | None],
) -> OmetiffPackagePlanRow:
    source_is_local, source_exists = _source_flags(validation_row.source_path, manifest_path)
    identifiers = _identifiers(
        manifest_row,
        row_identifier=validation_row.row_identifier,
        source_image_id=validation_row.source_image_id,
    )
    blockers: list[OmetiffPackagePlanIssue] = []
    warnings: list[OmetiffPackagePlanIssue] = []
    deferred_checks: list[OmetiffPackagePlanIssue] = []

    for issue in validation_row.issues:
        package_issue = _issue_from_validation(issue)
        if package_issue.severity is IssueSeverity.ERROR:
            blockers.append(package_issue)
        elif package_issue.severity is IssueSeverity.WARNING:
            warnings.append(package_issue)
        elif package_issue.severity is IssueSeverity.DEFERRED:
            deferred_checks.append(package_issue)

    planned_name, naming_issues = _planned_package_name(
        profile,
        validation_row,
        manifest_row,
    )
    blockers.extend(naming_issues)
    deferred_checks.extend(
        _row_specific_deferred_checks(
            profile,
            validation_row,
            existing_codes={issue.code for issue in deferred_checks},
        )
    )

    return OmetiffPackagePlanRow(
        row_number=validation_row.row_number,
        row_identifier=validation_row.row_identifier,
        source_image_id=validation_row.source_image_id,
        source_path=validation_row.source_path,
        source_exists=source_exists,
        source_is_local=source_is_local,
        file_size_bytes=validation_row.known_input_bytes,
        planned_package_name=planned_name,
        identifiers=identifiers,
        package_status=_package_status(blockers, warnings, deferred_checks),
        blockers=blockers,
        warnings=warnings,
        deferred_checks=deferred_checks,
    )


def _issue_from_validation(issue: OmetiffValidationIssue) -> OmetiffPackagePlanIssue:
    return OmetiffPackagePlanIssue(
        row_number=issue.row_number,
        row_identifier=issue.row_identifier,
        source_path=issue.source_path,
        source_image_id=issue.source_image_id,
        field=issue.field,
        code=issue.code,
        message=issue.message,
        severity=issue.severity,
        requirement_phase=issue.requirement_phase,
        source=issue.source,
    )


def _row_specific_deferred_checks(
    profile: DatabaseProfile,
    row: Any,
    *,
    existing_codes: set[str],
) -> list[OmetiffPackagePlanIssue]:
    deferred: list[OmetiffPackagePlanIssue] = []
    if _profile_requires_ome_metadata_validation(profile):
        if "OMETIFF_METADATA_VALIDATION_DEFERRED" not in existing_codes:
            deferred.append(
                _package_issue(
                    row,
                    code="OMETIFF_METADATA_VALIDATION_DEFERRED",
                    message=(
                        "OME-TIFF metadata validation is deferred; this dry run does not "
                        "open TIFF files or validate scientific OME-TIFF metadata."
                    ),
                    field="ome_metadata_validation",
                    severity=IssueSeverity.DEFERRED,
                    phase=RequirementPhase.UPLOAD_VALIDATION,
                )
            )
        if "OME_XML_PARSING_DEFERRED" not in existing_codes:
            deferred.append(
                _package_issue(
                    row,
                    code="OME_XML_PARSING_DEFERRED",
                    message="OME-XML parsing is deferred; this dry run does not parse OME-XML.",
                    field="ome_xml",
                    severity=IssueSeverity.DEFERRED,
                    phase=RequirementPhase.UPLOAD_VALIDATION,
                )
            )

    if _profile_requires_checksum_computation(profile) and (
        "CHECKSUM_COMPUTATION_DEFERRED" not in existing_codes
    ):
        deferred.append(
            _package_issue(
                row,
                code="CHECKSUM_COMPUTATION_DEFERRED",
                message=(
                    "Checksum computation is deferred; this dry run does not read image "
                    "file bytes to compute checksums."
                ),
                field="checksum",
                severity=IssueSeverity.DEFERRED,
                phase=RequirementPhase.UPLOAD_VALIDATION,
            )
        )
    return deferred


def _profile_requires_ome_metadata_validation(profile: DatabaseProfile) -> bool:
    return profile.ometiff.require_ome_xml or profile.validation.validate_ometiff


def _profile_requires_checksum_computation(profile: DatabaseProfile) -> bool:
    return profile.output.emit_checksums or profile.validation.validate_checksums


def _planned_package_name(
    profile: DatabaseProfile,
    row: Any,
    manifest_row: Mapping[str, str | None],
) -> tuple[str | None, list[OmetiffPackagePlanIssue]]:
    placeholders = _template_placeholders(profile.naming.template)
    values = _template_values(
        row,
        manifest_row,
        placeholders=placeholders,
    )
    unresolved = [placeholder for placeholder in placeholders if placeholder not in values]
    if unresolved:
        return None, [
            _package_issue(
                row,
                code=PACKAGE_NAME_TEMPLATE_UNRESOLVED_PLACEHOLDER,
                message=(
                    "Profile naming template cannot be resolved for this row; missing "
                    f"placeholder(s): {', '.join(sorted(unresolved))}."
                ),
                field="naming.template",
                severity=IssueSeverity.ERROR,
                phase=RequirementPhase.PREFLIGHT_MANIFEST,
            )
        ]

    try:
        rendered = profile.naming.template.format(**values)
    except (IndexError, KeyError, ValueError) as exc:
        return None, [
            _package_issue(
                row,
                code=PACKAGE_NAME_TEMPLATE_UNRESOLVED_PLACEHOLDER,
                message=f"Profile naming template cannot be rendered for this row: {exc}.",
                field="naming.template",
                severity=IssueSeverity.ERROR,
                phase=RequirementPhase.PREFLIGHT_MANIFEST,
            )
        ]

    return (
        _normalize_package_filename(
            rendered,
            source_path=row.source_path,
            default_extension=profile.output.extension,
        ),
        [],
    )


def _template_values(
    row: Any,
    manifest_row: Mapping[str, str | None],
    *,
    placeholders: set[str],
) -> dict[str, str | int]:
    values: dict[str, str | int] = {"row_number": row.row_number}
    for key, value in manifest_row.items():
        if value is not None:
            values[key] = _sanitize_component(value)

    if row.source_image_id:
        values.setdefault("source_image_id", _sanitize_component(row.source_image_id))
    if row.row_identifier:
        values.setdefault("row_identifier", _sanitize_component(row.row_identifier))
    source_stem = _source_stem(row.source_path, row.row_number)
    values.setdefault("source_stem", source_stem)

    tissue_identifier = _fallback_tissue_identifier(row, manifest_row, source_stem)
    for placeholder in _TISSUE_IDENTIFIER_PLACEHOLDERS & placeholders:
        values.setdefault(placeholder, tissue_identifier)

    return values


def _fallback_tissue_identifier(
    row: Any,
    manifest_row: Mapping[str, str | None],
    source_stem: str,
) -> str:
    for key in ("tissue_id", "section_id", "row_id"):
        value = manifest_row.get(key)
        if value:
            return _sanitize_component(value)
    return f"{source_stem}_row{row.row_number:04d}"


def _normalize_package_filename(
    rendered: str,
    *,
    source_path: str | None,
    default_extension: str,
) -> str:
    filename = _sanitize_filename(rendered)
    desired_suffix = _source_ometiff_suffix(source_path) or default_extension
    lower_filename = filename.lower()
    if lower_filename.endswith(OMETIFF_SUFFIXES):
        for suffix in sorted(OMETIFF_SUFFIXES, key=len, reverse=True):
            if lower_filename.endswith(suffix):
                return filename[: -len(suffix)] + desired_suffix
    if Path(filename).suffix:
        return filename
    return filename + desired_suffix


def _template_placeholders(template: str) -> set[str]:
    placeholders: set[str] = set()
    for _literal_text, field_name, _format_spec, _conversion in Formatter().parse(template):
        if field_name:
            placeholders.add(field_name.split(".", 1)[0].split("[", 1)[0])
    return placeholders


def _apply_package_name_collisions(rows: list[OmetiffPackagePlanRow]) -> None:
    counts = Counter(row.planned_package_name for row in rows if row.planned_package_name)
    duplicate_names = {name for name, count in counts.items() if count > 1}
    for row in rows:
        if row.planned_package_name not in duplicate_names:
            continue
        row.blockers.append(
            _package_issue(
                row,
                code=PACKAGE_NAME_COLLISION,
                message=(
                    "Planned package name collides with another manifest row: "
                    f"{row.planned_package_name}."
                ),
                field="planned_package_name",
                severity=IssueSeverity.ERROR,
                phase=RequirementPhase.PREFLIGHT_MANIFEST,
            )
        )
        row.package_status = _package_status(row.blockers, row.warnings, row.deferred_checks)


def _package_status(
    blockers: list[OmetiffPackagePlanIssue],
    warnings: list[OmetiffPackagePlanIssue],
    deferred_checks: list[OmetiffPackagePlanIssue],
) -> str:
    if blockers:
        return PACKAGE_STATUS_BLOCKED
    if warnings or deferred_checks:
        return PACKAGE_STATUS_READY_WITH_WARNINGS
    return PACKAGE_STATUS_READY


def _source_flags(
    source_path: str | None,
    manifest_path: Path,
) -> tuple[bool, bool | None]:
    if not source_path:
        return True, None
    location = source_location(source_path, manifest_path)
    if not location.is_local:
        return False, None
    if location.path_to_check is None:
        return True, None
    return True, location.path_to_check.exists()


def _source_stem(source_path: str | None, row_number: int) -> str:
    if not source_path:
        return f"row{row_number:04d}"
    target = source_target_for_extension(source_path)
    name = Path(target).name
    lower_name = name.lower()
    for suffix in sorted(OMETIFF_SUFFIXES, key=len, reverse=True):
        if lower_name.endswith(suffix):
            return _sanitize_component(name[: -len(suffix)])
    stem = Path(name).stem or f"row{row_number:04d}"
    return _sanitize_component(stem)


def _source_ometiff_suffix(source_path: str | None) -> str | None:
    if not source_path:
        return None
    target = source_target_for_extension(source_path)
    lower_name = Path(target).name.lower()
    for suffix in sorted(OMETIFF_SUFFIXES, key=len, reverse=True):
        if lower_name.endswith(suffix):
            return suffix
    return None


def _identifiers(
    manifest_row: Mapping[str, str | None],
    *,
    row_identifier: str | None,
    source_image_id: str | None,
) -> dict[str, str | None]:
    identifiers = {key: manifest_row.get(key) for key in _PACKAGE_IDENTIFIER_FIELDS}
    identifiers["row_identifier"] = row_identifier
    identifiers["source_image_id"] = source_image_id or manifest_row.get("source_image_id")
    return identifiers


def _package_issue(
    row: Any,
    *,
    code: str,
    message: str,
    field: str,
    severity: IssueSeverity,
    phase: RequirementPhase,
) -> OmetiffPackagePlanIssue:
    return OmetiffPackagePlanIssue(
        row_number=row.row_number,
        row_identifier=row.row_identifier,
        source_path=row.source_path,
        source_image_id=row.source_image_id,
        field=field,
        code=code,
        message=message,
        severity=severity,
        requirement_phase=phase,
    )


def _build_report(
    *,
    profile: DatabaseProfile,
    profile_path: Path,
    manifest_path: Path,
    output_dir: Path,
    generated_at: datetime,
    strict: bool,
    rows: list[OmetiffPackagePlanRow],
    package_plan_path: Path,
    package_manifest_path: Path,
    package_summary_path: Path,
) -> OmetiffPackagePlanReport:
    ready_count = sum(1 for row in rows if row.package_status == PACKAGE_STATUS_READY)
    ready_with_warnings_count = sum(
        1 for row in rows if row.package_status == PACKAGE_STATUS_READY_WITH_WARNINGS
    )
    blocked_count = sum(1 for row in rows if row.package_status == PACKAGE_STATUS_BLOCKED)
    deferred_check_count = sum(len(row.deferred_checks) for row in rows)
    warning_count = sum(len(row.warnings) for row in rows)
    ready_for_next_action = blocked_count == 0 and not (
        strict and (warning_count or deferred_check_count)
    )
    issues = [
        issue for row in rows for issue in [*row.blockers, *row.warnings, *row.deferred_checks]
    ]
    return OmetiffPackagePlanReport(
        generated_at=generated_at,
        profile_identifier=f"{profile.profile_name}@{profile.profile_version}",
        profile_name=profile.profile_name,
        profile_version=profile.profile_version,
        profile_path=profile_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        row_count=len(rows),
        ready_count=ready_count,
        ready_with_warnings_count=ready_with_warnings_count,
        blocked_count=blocked_count,
        deferred_check_count=deferred_check_count,
        total_known_input_bytes=sum(
            row.file_size_bytes for row in rows if row.file_size_bytes is not None
        ),
        ready_for_next_action=ready_for_next_action,
        next_action=_next_action(
            blocked_count=blocked_count,
            ready_for_next_action=ready_for_next_action,
            strict=strict,
            warning_count=warning_count,
            deferred_check_count=deferred_check_count,
        ),
        recommended_next_action=_recommended_next_action(
            blocked_count=blocked_count,
            strict=strict,
            warning_count=warning_count,
            deferred_check_count=deferred_check_count,
        ),
        strict=strict,
        deferred_capabilities=list(_DEFERRED_CAPABILITIES),
        out_of_scope=list(_OUT_OF_SCOPE),
        package_plan_path=package_plan_path,
        package_manifest_path=package_manifest_path,
        package_summary_path=package_summary_path,
        rows=rows,
        issues=issues,
    )


def _next_action(
    *,
    blocked_count: int,
    ready_for_next_action: bool,
    strict: bool,
    warning_count: int,
    deferred_check_count: int,
) -> str:
    if blocked_count:
        return "fix_blocked_rows_before_package_review"
    if strict and (warning_count or deferred_check_count) and not ready_for_next_action:
        return "resolve_warnings_or_deferred_checks_before_package_review"
    return "review_dry_run_package_plan"


def _recommended_next_action(
    *,
    blocked_count: int,
    strict: bool,
    warning_count: int,
    deferred_check_count: int,
) -> str:
    if blocked_count:
        return "Fix blocked rows before package execution."
    if strict and (warning_count or deferred_check_count):
        return "Resolve warnings or deferred checks, or rerun without strict mode."
    if deferred_check_count:
        return (
            "Review package_plan.json and package_manifest.csv, then run future OME-TIFF "
            "metadata validation before database upload."
        )
    if warning_count:
        return "Review package_plan.json and package_manifest.csv before enabling package behavior."
    return "Review package_plan.json and package_manifest.csv before enabling copy/link/upload behavior."


def _read_manifest_rows_by_number(manifest_path: Path) -> dict[int, dict[str, str | None]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Submission manifest not found: {manifest_path}")

    rows: dict[int, dict[str, str | None]] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManifestValidationError(f"Submission manifest {manifest_path} is empty")
        for row in reader:
            rows[reader.line_num] = {
                (key.strip() if key else ""): _blank_to_none(value)
                for key, value in row.items()
                if key and key.strip()
            }
    return rows


def _write_package_manifest(path: Path, rows: list[OmetiffPackagePlanRow]) -> None:
    fieldnames = [
        "row_number",
        "row_identifier",
        "source_image_id",
        "specimen_id",
        "sample_id",
        "slide_id",
        "tissue_id",
        "section_id",
        "row_id",
        "source_path",
        "source_exists",
        "source_is_local",
        "file_size_bytes",
        "planned_package_name",
        "package_status",
        "blocker_codes",
        "warning_codes",
        "deferred_check_codes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "row_number": row.row_number,
                    "row_identifier": row.row_identifier,
                    "source_image_id": row.source_image_id,
                    "specimen_id": row.identifiers.get("specimen_id"),
                    "sample_id": row.identifiers.get("sample_id"),
                    "slide_id": row.identifiers.get("slide_id"),
                    "tissue_id": row.identifiers.get("tissue_id"),
                    "section_id": row.identifiers.get("section_id"),
                    "row_id": row.identifiers.get("row_id"),
                    "source_path": row.source_path,
                    "source_exists": row.source_exists,
                    "source_is_local": row.source_is_local,
                    "file_size_bytes": row.file_size_bytes,
                    "planned_package_name": row.planned_package_name,
                    "package_status": row.package_status,
                    "blocker_codes": _issue_codes(row.blockers),
                    "warning_codes": _issue_codes(row.warnings),
                    "deferred_check_codes": _issue_codes(row.deferred_checks),
                }
            )


def _write_summary(path: Path, report: OmetiffPackagePlanReport) -> None:
    lines = [
        "OME-TIFF package dry-run summary",
        f"Generated at: {report.generated_at.isoformat()}",
        f"Validation scope: {report.validation_scope}",
        f"Profile: {report.profile_identifier}",
        f"Manifest: {report.manifest_path}",
        f"Output directory: {report.output_dir}",
        f"Rows: {report.row_count}",
        f"Ready: {report.ready_count}",
        f"Ready with warnings: {report.ready_with_warnings_count}",
        f"Blocked: {report.blocked_count}",
        f"Deferred checks: {report.deferred_check_count}",
        f"Total known input bytes: {report.total_known_input_bytes}",
        f"Ready for package review: {'yes' if report.ready_for_next_action else 'no'}",
        f"Recommended next action: {report.recommended_next_action}",
        "",
        "Deferred capabilities:",
        *[f"- {capability}" for capability in report.deferred_capabilities],
        "",
        "Out of scope:",
        *[f"- {item}" for item in report.out_of_scope],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _issue_codes(issues: list[OmetiffPackagePlanIssue]) -> str:
    return json.dumps([issue.code for issue in issues])


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _sanitize_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._=-]+", "-", value.strip()).strip("-")
    return sanitized or "unknown"


def _sanitize_filename(value: str) -> str:
    filename = value.replace("/", "-").replace("\\", "-").strip()
    filename = re.sub(r"[^A-Za-z0-9._=-]+", "-", filename).strip("-")
    return filename or "package_item"


__all__ = [
    "PACKAGE_MANIFEST_FILENAME",
    "PACKAGE_NAME_COLLISION",
    "PACKAGE_NAME_TEMPLATE_UNRESOLVED_PLACEHOLDER",
    "PACKAGE_PLAN_FILENAME",
    "PACKAGE_PLAN_VERSION",
    "PACKAGE_STATUS_BLOCKED",
    "PACKAGE_STATUS_READY",
    "PACKAGE_STATUS_READY_WITH_WARNINGS",
    "PACKAGE_SUMMARY_FILENAME",
    "OmetiffPackagePlanIssue",
    "OmetiffPackagePlanReport",
    "OmetiffPackagePlanRow",
    "OmetiffPackagePlanRunResult",
    "run_package_ometiff_dry_run",
]
