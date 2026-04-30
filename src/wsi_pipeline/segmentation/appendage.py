"""
H&E-aware refinement for weakly stained appendages on tissue components.

This module operates at the same thumbnail scale as tissue segmentation.  It is
intentionally conservative: it only trims low-H&E candidate regions that are
peripheral, small relative to their parent component, and geometrically
appendage-like.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from skimage import measure, morphology
from skimage.morphology import disk

from .stain import he_features, he_stain_mask

AppendageRefinementMode = Literal["trim", "annotate"]
AppendageRefinementProfile = Literal["he_sections"]


@dataclass(frozen=True)
class AppendageRecord:
    component_label: int
    appendage_label: int
    area_px: int
    area_fraction: float
    bbox_y0: int
    bbox_x0: int
    bbox_y1: int
    bbox_x1: int
    aspect_ratio: float
    extent: float
    mean_he_signal: float
    he_threshold: float
    boundary_contact_fraction: float
    trimmed: bool
    appendage_reason: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


def _aspect_ratio(region: measure._regionprops.RegionProperties) -> float:
    y0, x0, y1, x1 = region.bbox
    height = max(1, y1 - y0)
    width = max(1, x1 - x0)
    return max(height, width) / max(1, min(height, width))


def _touches_component_boundary(candidate: np.ndarray, component: np.ndarray) -> tuple[bool, float]:
    eroded = morphology.erosion(component, footprint=disk(2))
    boundary = component & ~eroded
    contact = int((candidate & boundary).sum())
    area = max(1, int(candidate.sum()))
    return contact > 0, contact / area


def refine_appendages(
    image: np.ndarray,
    mask_bool: np.ndarray,
    *,
    mode: AppendageRefinementMode = "trim",
    profile: AppendageRefinementProfile = "he_sections",
    min_area_px: int = 3000,
    he_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Trim or annotate weakly stained peripheral appendages.

    Parameters
    ----------
    image : np.ndarray
        RGB thumbnail image.
    mask_bool : np.ndarray
        Boolean tissue mask at thumbnail scale.
    mode : str
        ``"trim"`` removes likely appendages. ``"annotate"`` records them but
        leaves the mask unchanged.
    profile : str
        Currently only ``"he_sections"`` is supported.
    min_area_px : int
        Minimum tissue component area at thumbnail scale, used for cleanup.
    he_mask : np.ndarray, optional
        Precomputed H&E stain-confidence mask.
    """
    if profile != "he_sections":
        raise ValueError("appendage refinement profile must be 'he_sections'")
    if mode not in {"trim", "annotate"}:
        raise ValueError("appendage refinement mode must be 'trim' or 'annotate'")

    mask = np.asarray(mask_bool, dtype=bool)
    if not mask.any():
        return mask, {
            "enabled": True,
            "mode": mode,
            "profile": profile,
            "n_appendages_flagged": 0,
            "flagged_area_px": 0,
            "n_appendages_trimmed": 0,
            "trimmed_area_px": 0,
            "trimmed_fraction": 0.0,
            "appendage_reason": "",
            "records": [],
        }

    if he_mask is None:
        he_mask, stain_info = he_stain_mask(
            image,
            mode="adaptive-he",
            min_he_signal=0.0,
            return_info=True,
        )
    else:
        stain_info = {"he_threshold": 0.0}
    he_mask = np.asarray(he_mask, dtype=bool)
    features = he_features(image)
    he_signal = features.he_signal
    he_threshold = float(stain_info.get("he_threshold") or 0.0)

    refined = mask.copy()
    trim_mask = np.zeros_like(mask, dtype=bool)
    records: list[AppendageRecord] = []

    labeled = measure.label(mask, connectivity=2)
    for component_region in measure.regionprops(labeled):
        component = labeled == component_region.label
        component_area = max(1, int(component_region.area))

        low_he = component & ~he_mask
        if not low_he.any():
            continue

        low_labeled = measure.label(low_he, connectivity=2)
        for appendage_region in measure.regionprops(low_labeled):
            candidate = low_labeled == appendage_region.label
            area = int(appendage_region.area)
            area_fraction = area / component_area
            touches_boundary, boundary_contact_fraction = _touches_component_boundary(
                candidate,
                component,
            )
            aspect_ratio = _aspect_ratio(appendage_region)
            mean_he = float(he_signal[candidate].mean()) if area else 0.0
            extent = float(appendage_region.extent)

            reasons: list[str] = []
            if touches_boundary and area_fraction <= 0.18:
                if aspect_ratio >= 4.0 and boundary_contact_fraction >= 0.03:
                    reasons.append("peripheral_low_he_strip")
                elif extent <= 0.35 and boundary_contact_fraction >= 0.08:
                    reasons.append("peripheral_low_he_appendage")

            # Guardrail: the candidate must be genuinely weak in deconvolved H&E.
            low_he_signal = mean_he <= max(he_threshold * 1.25, 0.04)
            should_trim = bool(reasons) and low_he_signal

            if should_trim and mode == "trim":
                trim_mask |= candidate

            y0, x0, y1, x1 = (int(v) for v in appendage_region.bbox)
            records.append(
                AppendageRecord(
                    component_label=int(component_region.label),
                    appendage_label=int(appendage_region.label),
                    area_px=area,
                    area_fraction=float(area_fraction),
                    bbox_y0=y0,
                    bbox_x0=x0,
                    bbox_y1=y1,
                    bbox_x1=x1,
                    aspect_ratio=float(aspect_ratio),
                    extent=extent,
                    mean_he_signal=mean_he,
                    he_threshold=he_threshold,
                    boundary_contact_fraction=float(boundary_contact_fraction),
                    trimmed=bool(should_trim and mode == "trim"),
                    appendage_reason=";".join(reasons),
                )
            )

    if mode == "trim" and trim_mask.any():
        refined &= ~trim_mask
        refined = morphology.remove_small_objects(
            refined,
            min_size=max(64, int(min_area_px // 2)),
        )

    trimmed_area = int(trim_mask.sum()) if mode == "trim" else 0
    total_area = max(1, int(mask.sum()))
    flagged_records = [record for record in records if record.appendage_reason]
    trimmed_records = [record for record in records if record.trimmed]
    return refined, {
        "enabled": True,
        "mode": mode,
        "profile": profile,
        "n_appendages_flagged": len(flagged_records),
        "flagged_area_px": int(sum(record.area_px for record in flagged_records)),
        "n_appendages_trimmed": len(trimmed_records),
        "trimmed_area_px": trimmed_area,
        "trimmed_fraction": float(trimmed_area / total_area),
        "appendage_reason": ";".join(
            sorted({record.appendage_reason for record in flagged_records if record.appendage_reason})
        ),
        "records": [record.to_dict() for record in records],
    }


__all__ = [
    "AppendageRecord",
    "AppendageRefinementMode",
    "AppendageRefinementProfile",
    "refine_appendages",
]
