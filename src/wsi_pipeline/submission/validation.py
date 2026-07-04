"""Validation errors for submission factory loaders."""

from __future__ import annotations


class SubmissionValidationError(ValueError):
    """Base error for submission scaffold validation failures."""


class ProfileValidationError(SubmissionValidationError):
    """Raised when a database profile is structurally invalid."""


class ManifestValidationError(SubmissionValidationError):
    """Raised when a submission manifest is structurally invalid."""


__all__ = [
    "ManifestValidationError",
    "ProfileValidationError",
    "SubmissionValidationError",
]
