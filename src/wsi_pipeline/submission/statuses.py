"""Submission factory status enums."""

from __future__ import annotations

from enum import Enum


class BatchStatus(str, Enum):
    """Lifecycle states for a submission batch."""

    DRAFT = "DRAFT"
    PREFLIGHT_READY = "PREFLIGHT_READY"
    PREFLIGHT_WARNING = "PREFLIGHT_WARNING"
    PREFLIGHT_BLOCKED = "PREFLIGHT_BLOCKED"
    TISSUE_DETECTION_READY = "TISSUE_DETECTION_READY"
    AWAITING_REVIEW = "AWAITING_REVIEW"
    READY_FOR_CONVERSION = "READY_FOR_CONVERSION"
    CONVERSION_IN_PROGRESS = "CONVERSION_IN_PROGRESS"
    CONVERSION_COMPLETE = "CONVERSION_COMPLETE"
    VALIDATION_COMPLETE = "VALIDATION_COMPLETE"
    UPLOAD_READY = "UPLOAD_READY"
    FAILED = "FAILED"


class TissueStatus(str, Enum):
    """Lifecycle and review states for a detected tissue section."""

    NOT_DETECTED = "NOT_DETECTED"
    PENDING_REVIEW = "PENDING_REVIEW"
    PASS = "PASS"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_EXPERT_REVIEW = "NEEDS_EXPERT_REVIEW"
    CONVERTED = "CONVERTED"
    VALIDATED = "VALIDATED"
    FAILED = "FAILED"


class ReviewDecision(str, Enum):
    """Operator or expert review decisions for tissue sections."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    NEEDS_EXPERT_REVIEW = "NEEDS_EXPERT_REVIEW"
    DEFER = "DEFER"


__all__ = ["BatchStatus", "TissueStatus", "ReviewDecision"]
