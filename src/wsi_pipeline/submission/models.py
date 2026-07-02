"""Lightweight submission factory data contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .statuses import BatchStatus, ReviewDecision, TissueStatus

ModelT = TypeVar("ModelT", bound="SubmissionBaseModel")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SubmissionBaseModel(BaseModel):
    """Base model with JSON-safe serialization helpers."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls: type[ModelT], data: Mapping[str, Any]) -> ModelT:
        """Create a model from a dictionary produced by ``to_dict``."""
        return cls.model_validate(data)


class PixelBoundingBox(SubmissionBaseModel):
    """Bounding box in parent-slide pixel coordinates."""

    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    x_max: int = Field(ge=0)
    y_max: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_extent(self) -> PixelBoundingBox:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min")
        return self


class PhysicalBoundingBox(SubmissionBaseModel):
    """Bounding box in physical coordinates when metadata are available."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    unit: str | None = None

    @model_validator(mode="after")
    def validate_extent(self) -> PhysicalBoundingBox:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min")
        return self


class SourceSlide(SubmissionBaseModel):
    """Manifest and preflight record for one parent source slide."""

    specimen_id: str = Field(min_length=1)
    slide_id: str = Field(min_length=1)
    source_path: Path
    ets_path: Path | None = None
    stain: str | None = None
    block_id: str | None = None
    section_number: int | None = None
    notes: str | None = None
    checksum: str | None = None
    width_px: int | None = Field(default=None, ge=1)
    height_px: int | None = Field(default=None, ge=1)
    physical_pixel_size_x: float | None = Field(default=None, gt=0)
    physical_pixel_size_y: float | None = Field(default=None, gt=0)
    physical_pixel_size_unit: str | None = None
    metadata_status: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class DetectedTissueSection(SubmissionBaseModel):
    """Contract record for one tissue section detected on a parent slide."""

    specimen_id: str = Field(min_length=1)
    slide_id: str = Field(min_length=1)
    tissue_id: str = Field(min_length=1)
    bbox_parent_px: PixelBoundingBox
    bbox_parent_physical: PhysicalBoundingBox | None = None
    area_px: int | None = Field(default=None, ge=0)
    area_physical: float | None = Field(default=None, ge=0)
    mask_path: Path | None = None
    qc_overlay_path: Path | None = None
    status: TissueStatus = TissueStatus.NOT_DETECTED
    review_decision: ReviewDecision | None = None
    reviewer: str | None = None
    review_notes: str | None = None
    output_filename: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class OutputDerivative(SubmissionBaseModel):
    """Provenance record for one future single-tissue OME-TIFF derivative."""

    specimen_id: str = Field(min_length=1)
    slide_id: str = Field(min_length=1)
    tissue_id: str = Field(min_length=1)
    output_path: Path
    sidecar_path: Path | None = None
    qc_path: Path | None = None
    parent_source_path: Path
    parent_source_id: str | None = None
    parent_checksum: str | None = None
    crop_bounds_parent_px: PixelBoundingBox
    crop_bounds_parent_physical: PhysicalBoundingBox | None = None
    array_to_physical_transform: list[list[float]] | None = None
    resampling_applied: bool = False
    flip_applied: bool = False
    rotation_applied: bool = False
    is_derivative: bool = True
    derivative_reason: str | None = "split_from_multi_tissue_source_wsi"
    conversion_profile: str = Field(min_length=1)
    conversion_profile_version: str = Field(min_length=1)
    conversion_config_hash: str | None = None
    output_checksum: str | None = None
    validation_status: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class SubmissionBatch(SubmissionBaseModel):
    """Submission-level contract for a resumable future batch workflow."""

    batch_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=_utc_now)
    profile_name: str = Field(min_length=1)
    profile_version: str = Field(min_length=1)
    input_root: Path | None = None
    output_root: Path
    manifest_path: Path | None = None
    source_slides: list[SourceSlide] = Field(default_factory=list)
    tissue_sections: list[DetectedTissueSection] = Field(default_factory=list)
    derivatives: list[OutputDerivative] = Field(default_factory=list)
    status: BatchStatus = BatchStatus.DRAFT
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


__all__ = [
    "DetectedTissueSection",
    "OutputDerivative",
    "PhysicalBoundingBox",
    "PixelBoundingBox",
    "SourceSlide",
    "SubmissionBaseModel",
    "SubmissionBatch",
]
