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
from .preflight import (
    IssueSeverity,
    PreflightIssue,
    PreflightReport,
    PreflightRowResult,
    PreflightRunResult,
    PreflightState,
    PreflightStateRow,
    run_preflight,
)
from .profiles import DatabaseProfile, RequirementPhase, load_database_profile
from .statuses import BatchStatus, ReviewDecision, TissueStatus
from .validation import ManifestValidationError, ProfileValidationError, SubmissionValidationError

__all__ = [
    "BatchStatus",
    "DatabaseProfile",
    "DetectedTissueSection",
    "IssueSeverity",
    "ManifestValidationError",
    "OutputDerivative",
    "PhysicalBoundingBox",
    "PixelBoundingBox",
    "PreflightIssue",
    "PreflightReport",
    "PreflightRunResult",
    "PreflightState",
    "PreflightStateRow",
    "PreflightRowResult",
    "ProfileValidationError",
    "RequirementPhase",
    "ReviewDecision",
    "SourceSlide",
    "SubmissionBatch",
    "SubmissionManifest",
    "SubmissionValidationError",
    "TissueStatus",
    "load_database_profile",
    "load_submission_manifest",
    "run_preflight",
]
