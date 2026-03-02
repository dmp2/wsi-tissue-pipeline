"""Orientation validation helpers for atlas-registration initialization."""

from __future__ import annotations

from itertools import permutations, product
from typing import Any

import numpy as np

from .config import OrientationResolution

_AXIS_GROUPS = (("R", "L"), ("A", "P"), ("S", "I"))
_EXAMPLE_CODES = ("RAS", "LPI", "PIR", "RIP")
_VALID_CODES = tuple(
    sorted(
        {
            "".join(perm)
            for choice in product(*_AXIS_GROUPS)
            for perm in permutations(choice)
        }
    )
)


def orientation_validation_rule() -> str:
    """Return the canonical backend orientation rule in plain language."""

    return (
        "Orientation codes must be exactly 3 letters and use one axis from each pair "
        "{R/L}, {A/P}, and {S/I}, with no duplicate anatomical axes."
    )


def list_valid_orientation_codes() -> list[str]:
    """Return all valid orientation codes in deterministic order."""

    return list(_VALID_CODES)


def validate_orientation_code(code: str | None) -> str:
    """Validate and normalize a backend orientation code."""

    normalized = (code or "").strip().upper()
    if len(normalized) != 3:
        raise ValueError(
            f"Invalid orientation code {code!r}. {orientation_validation_rule()} "
            f"Examples: {', '.join(_EXAMPLE_CODES)}."
        )
    if normalized not in _VALID_CODES:
        raise ValueError(
            f"Invalid orientation code {code!r}. {orientation_validation_rule()} "
            f"Examples: {', '.join(_EXAMPLE_CODES)}."
        )
    return normalized


def resolve_orientation_init(
    backend: Any,
    orientation_from: str,
    orientation_to: str,
) -> tuple[np.ndarray, OrientationResolution]:
    """Resolve a backend-derived initial affine and structured orientation metadata."""

    normalized_from = validate_orientation_code(orientation_from)
    normalized_to = validate_orientation_code(orientation_to)
    rotation = np.asarray(
        backend.orientation_to_orientation(normalized_from, normalized_to),
        dtype=np.float32,
    )
    affine = np.eye(4, dtype=np.float32)
    affine[:3, :3] = rotation
    resolution = OrientationResolution(
        mode="orientation",
        orientation_from=normalized_from,
        orientation_to=normalized_to,
        is_valid=True,
        validation_rule=orientation_validation_rule(),
        resolved_matrix=rotation.tolist(),
        source="backend.orientation_to_orientation",
    )
    return affine, resolution


def matrix_orientation_resolution() -> OrientationResolution:
    """Return metadata for file-derived affine initialization."""

    return OrientationResolution(
        mode="matrix",
        is_valid=True,
        validation_rule=orientation_validation_rule(),
        source="init_affine_path",
    )


def none_orientation_resolution() -> OrientationResolution:
    """Return metadata for runs that do not derive an orientation affine."""

    return OrientationResolution(
        mode="none",
        is_valid=True,
        validation_rule=orientation_validation_rule(),
        source="not-used",
    )
