"""Submission factory contracts for database-ready OME-TIFF batches."""

from .manifests import SubmissionManifest, load_submission_manifest
from .models import (
    DetectedTissueSection,
    OutputDerivative,
    PhysicalBoundingBox,
    PixelBoundingBox,
    SourceSlide,
    SubmissionBatch,
)
from .profiles import DatabaseProfile, load_database_profile
from .statuses import BatchStatus, ReviewDecision, TissueStatus
from .validation import ManifestValidationError, ProfileValidationError, SubmissionValidationError

__all__ = [
    "BatchStatus",
    "DatabaseProfile",
    "DetectedTissueSection",
    "ManifestValidationError",
    "OutputDerivative",
    "PhysicalBoundingBox",
    "PixelBoundingBox",
    "ProfileValidationError",
    "ReviewDecision",
    "SourceSlide",
    "SubmissionBatch",
    "SubmissionManifest",
    "SubmissionValidationError",
    "TissueStatus",
    "load_database_profile",
    "load_submission_manifest",
]
