"""Submission manifest CSV loading and structural validation."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from urllib.parse import unquote, urlparse

from pydantic import Field, ValidationError

from .models import SourceSlide, SubmissionBaseModel
from .profiles import DatabaseProfile
from .validation import ManifestValidationError

REQUIRED_MANIFEST_COLUMNS = {"specimen_id", "slide_id", "source_path"}
OPTIONAL_MANIFEST_COLUMNS = {
    "stain",
    "block_id",
    "section_number",
    "notes",
    "ets_path",
    "checksum",
    "width_px",
    "height_px",
    "physical_pixel_size_x",
    "physical_pixel_size_y",
    "physical_pixel_size_unit",
    "metadata_status",
}
SUPPORTED_MANIFEST_COLUMNS = REQUIRED_MANIFEST_COLUMNS | OPTIONAL_MANIFEST_COLUMNS


class SubmissionManifest(SubmissionBaseModel):
    """Loaded source-slide manifest."""

    manifest_path: Path
    source_slides: list[SourceSlide] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def load_submission_manifest(
    path: str | Path,
    *,
    profile: DatabaseProfile | None = None,
    check_paths: bool = False,
) -> SubmissionManifest:
    """Load a submission CSV and return source-slide records.

    CSV row numbers in errors are human row numbers: the header is row 1, so the
    first data row is row 2.
    """
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Submission manifest not found: {manifest_path}")

    errors: list[str] = []
    source_slides: list[SourceSlide] = []

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManifestValidationError(f"Submission manifest {manifest_path} is empty")

        column_lookup = _column_lookup(reader.fieldnames)
        missing_columns = sorted(REQUIRED_MANIFEST_COLUMNS - set(column_lookup))
        if missing_columns:
            raise ManifestValidationError(
                f"Submission manifest {manifest_path} is missing required columns: "
                + ", ".join(missing_columns)
            )

        for row in reader:
            row_number = reader.line_num
            row_errors = _validate_row(
                row,
                row_number=row_number,
                column_lookup=column_lookup,
                manifest_path=manifest_path,
                profile=profile,
                check_paths=check_paths,
            )
            if row_errors:
                errors.extend(row_errors)
                continue

            data = {
                column: _blank_to_none(row.get(original_name))
                for column, original_name in column_lookup.items()
                if column in SUPPORTED_MANIFEST_COLUMNS
            }
            try:
                source_slides.append(SourceSlide.model_validate(data))
            except ValidationError as exc:
                errors.append(f"row {row_number}: invalid source slide record: {exc}")

    if errors:
        raise ManifestValidationError(
            f"Invalid submission manifest {manifest_path}: " + "; ".join(errors)
        )

    return SubmissionManifest(manifest_path=manifest_path, source_slides=source_slides)


def _column_lookup(fieldnames: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name in fieldnames:
        stripped = name.strip() if name else ""
        if stripped:
            lookup[stripped] = name
    return lookup


def _validate_row(
    row: Mapping[str, str | None],
    *,
    row_number: int,
    column_lookup: Mapping[str, str],
    manifest_path: Path,
    profile: DatabaseProfile | None,
    check_paths: bool,
) -> list[str]:
    errors: list[str] = []
    for column in sorted(REQUIRED_MANIFEST_COLUMNS):
        value = _blank_to_none(row.get(column_lookup[column]))
        if value is None:
            errors.append(f"row {row_number}: required field '{column}' is blank")

    source_value = _blank_to_none(row.get(column_lookup["source_path"]))
    if source_value is None:
        return errors

    source_path = Path(source_value)
    if profile is not None:
        if not source_extension_is_accepted(source_value, profile.input.accepted_extensions):
            accepted_text = ", ".join(sorted(profile.input.accepted_extensions))
            suffix = source_extension_for_message(source_value)
            errors.append(
                f"row {row_number}: source_path extension '{suffix or '<none>'}' "
                f"is not accepted by profile {profile.profile_name}; expected one of: "
                f"{accepted_text}"
            )

    if check_paths:
        path_to_check = (
            source_path if source_path.is_absolute() else manifest_path.parent / source_path
        )
        if not path_to_check.exists():
            errors.append(f"row {row_number}: source_path does not exist: {source_path}")

    return errors


def source_extension_is_accepted(
    source_value: str,
    accepted_extensions: Iterable[str],
) -> bool:
    """Return whether a local path or URI ends with an accepted extension."""
    source_target = _source_target_for_extension(source_value).lower()
    return any(source_target.endswith(extension.lower()) for extension in accepted_extensions)


def source_extension_for_message(source_value: str) -> str:
    """Return the best suffix to include in validation messages."""
    suffix = Path(_source_target_for_extension(source_value)).suffix.lower()
    return suffix or "<none>"


def _source_target_for_extension(source_value: str) -> str:
    parsed = urlparse(source_value)
    if parsed.scheme and parsed.path:
        return unquote(parsed.path)
    return source_value


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


__all__ = [
    "OPTIONAL_MANIFEST_COLUMNS",
    "REQUIRED_MANIFEST_COLUMNS",
    "SUPPORTED_MANIFEST_COLUMNS",
    "SubmissionManifest",
    "load_submission_manifest",
    "source_extension_for_message",
    "source_extension_is_accepted",
]
