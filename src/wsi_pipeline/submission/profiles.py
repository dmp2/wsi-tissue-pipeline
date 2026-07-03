"""Database profile loading and structural validation."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from string import Formatter
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .validation import ProfileValidationError

REQUIRED_TOP_LEVEL_KEYS = {
    "profile_name",
    "profile_version",
    "input",
    "output",
    "ometiff",
    "metadata",
    "qc",
    "naming",
    "validation",
}
REQUIRED_NAMING_PLACEHOLDERS = {"specimen_id", "slide_id", "tissue_id"}


class RequirementPhase(str, Enum):
    """Workflow phase where a database requirement can be validated."""

    PREFLIGHT_MANIFEST = "preflight_manifest"
    SOURCE_FILE_PREFLIGHT = "source_file_preflight"
    SOURCE_METADATA_VALIDATION = "source_metadata_validation"
    DERIVATIVE_EXPORT = "derivative_export"
    UPLOAD_VALIDATION = "upload_validation"


class ProfileSection(BaseModel):
    """Base class for database profile sections."""

    model_config = ConfigDict(extra="allow")


class InputProfile(ProfileSection):
    accepted_extensions: list[str] = Field(min_length=1)
    require_ets_companion: bool
    raw_files_read_only: bool

    @field_validator("accepted_extensions")
    @classmethod
    def validate_accepted_extensions(cls, value: list[str]) -> list[str]:
        for extension in value:
            _validate_extension(extension, "input.accepted_extensions")
        return value


class OutputProfile(ProfileSection):
    mode: str = Field(min_length=1)
    extension: str = Field(min_length=1)
    emit_sidecar_json: bool
    emit_qc_png: bool
    emit_per_tissue_provenance: bool
    emit_batch_manifest: bool
    emit_checksums: bool

    @field_validator("extension")
    @classmethod
    def validate_output_extension(cls, value: str) -> str:
        _validate_extension(value, "output.extension")
        return value


class OmeTiffProfile(ProfileSection):
    tiled: bool
    pyramidal: bool
    bigtiff: bool
    preserve_source_resolution: bool
    compression: str = Field(min_length=1)
    tile_size: str | int
    require_ome_xml: bool


class MetadataProfile(ProfileSection):
    require_physical_pixel_size: bool
    require_units: bool
    require_channel_metadata: bool
    require_parent_source_checksum: bool
    require_crop_bounds_parent_pixels: bool
    require_child_array_to_physical_transform: bool


class QCProfile(ProfileSection):
    require_tissue_detection_qc: bool
    require_operator_approval: bool
    require_expert_review_for_warnings: bool
    block_on_failed_segmentation: bool
    block_on_missing_parent_mapping: bool


class NamingProfile(ProfileSection):
    template: str = Field(min_length=1)
    require_unique_output_names: bool

    @field_validator("template")
    @classmethod
    def validate_template_placeholders(cls, value: str) -> str:
        placeholders = _template_placeholders(value)
        missing = sorted(REQUIRED_NAMING_PLACEHOLDERS - placeholders)
        if missing:
            joined = ", ".join(f"{{{name}}}" for name in missing)
            raise ValueError(f"naming.template must include placeholders: {joined}")
        return value


class ValidationProfile(ProfileSection):
    validate_ometiff: bool
    validate_sidecar_consistency: bool
    validate_checksums: bool
    fail_batch_if_any_required_output_fails: bool


class DatabaseProfile(BaseModel):
    """Typed representation of a database submission profile."""

    model_config = ConfigDict(extra="allow")

    profile_name: str = Field(min_length=1)
    profile_version: str = Field(min_length=1)
    description: str | None = None
    input: InputProfile
    output: OutputProfile
    ometiff: OmeTiffProfile
    metadata: MetadataProfile
    qc: QCProfile
    naming: NamingProfile
    validation: ValidationProfile
    requirement_phases: dict[str, RequirementPhase] = Field(default_factory=dict)

    @field_validator("requirement_phases", mode="before")
    @classmethod
    def validate_requirement_phases(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("requirement_phases must be a mapping of requirement name to phase")

        allowed = {phase.value for phase in RequirementPhase}
        normalized: dict[str, str] = {}
        for requirement, phase in value.items():
            if not isinstance(requirement, str) or not requirement.strip():
                raise ValueError("requirement_phases keys must be non-empty strings")
            phase_value = phase.value if isinstance(phase, RequirementPhase) else str(phase).strip()
            if phase_value not in allowed:
                expected = ", ".join(sorted(allowed))
                raise ValueError(
                    f"requirement_phases.{requirement} has unknown phase "
                    f"{phase_value!r}; expected one of: {expected}"
                )
            normalized[requirement.strip()] = phase_value
        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatabaseProfile:
        """Create a profile from a dictionary produced by ``to_dict``."""
        return cls.model_validate(data)


def load_database_profile(path: str | Path) -> DatabaseProfile:
    """Load and structurally validate a database profile YAML file."""
    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Database profile not found: {profile_path}")

    with profile_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, Mapping):
        raise ProfileValidationError(
            f"Database profile {profile_path} must be a YAML mapping at the top level"
        )

    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(raw.keys()))
    if missing:
        raise ProfileValidationError(
            f"Database profile {profile_path} is missing required top-level keys: "
            + ", ".join(missing)
        )

    try:
        return DatabaseProfile.model_validate(raw)
    except ValidationError as exc:
        raise ProfileValidationError(f"Invalid database profile {profile_path}: {exc}") from exc


def _validate_extension(extension: str, field_name: str) -> None:
    if not isinstance(extension, str):
        raise ValueError(f"{field_name} entries must be strings")
    if not extension.startswith("."):
        raise ValueError(f"{field_name} must start with '.'")
    if extension != extension.lower():
        raise ValueError(f"{field_name} must be lowercase: {extension}")


def _template_placeholders(template: str) -> set[str]:
    placeholders: set[str] = set()
    for _literal_text, field_name, _format_spec, _conversion in Formatter().parse(template):
        if field_name:
            placeholders.add(field_name.split(".", 1)[0].split("[", 1)[0])
    return placeholders


__all__ = [
    "DatabaseProfile",
    "InputProfile",
    "MetadataProfile",
    "NamingProfile",
    "OmeTiffProfile",
    "OutputProfile",
    "QCProfile",
    "REQUIRED_NAMING_PLACEHOLDERS",
    "REQUIRED_TOP_LEVEL_KEYS",
    "RequirementPhase",
    "ValidationProfile",
    "load_database_profile",
]
