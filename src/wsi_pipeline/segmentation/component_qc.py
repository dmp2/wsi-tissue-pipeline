"""
Component-level QC for segmented tissue candidates.

This module scores connected components after tissue segmentation.  The goal is
not to replace segmentation, but to annotate likely artifacts such as thin,
low-stain strips so manual QC can be faster and safer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage import measure

from .stain import he_stain_mask

ComponentQCMode = Literal["annotate", "drop_artifacts"]
ComponentQCProfile = Literal["he_sections"]


@dataclass(frozen=True)
class ComponentQCRecord:
    component_label: int
    tile_index_on_source: int
    area_rank: int
    component_area_px: int
    area_fraction: float
    bbox_y0: int
    bbox_x0: int
    bbox_y1: int
    bbox_x1: int
    bbox_height: int
    bbox_width: int
    aspect_ratio: float
    thinness: float
    extent: float
    solidity: float
    edge_contact_fraction: float
    mean_intensity: float
    mean_od: float
    stain_fraction: float
    artifact_likely: bool
    artifact_reason: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


def label_components_left_to_right(mask_bool: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """
    Label a boolean mask and return labels sorted by centroid x-coordinate.

    This mirrors the ordering used by tile generation so component QC records
    can be matched to saved tile indices.
    """
    raw_lbl, n_labels = measure.label(mask_bool.astype(bool), connectivity=2, return_num=True)
    if n_labels == 0:
        return raw_lbl.astype(np.int32), []

    filled_lbl = np.zeros_like(raw_lbl, dtype=np.int32)
    for label in range(1, n_labels + 1):
        comp = raw_lbl == label
        if comp.any():
            filled_lbl[binary_fill_holes(comp)] = label

    props = measure.regionprops(filled_lbl)
    labels = [p.label for p in sorted(props, key=lambda p: p.centroid[1])]
    return filled_lbl, labels


def filter_mask_by_labels(mask_bool: np.ndarray, labels_to_keep: set[int]) -> np.ndarray:
    """Return a boolean mask containing only the selected component labels."""
    labeled, _labels = label_components_left_to_right(mask_bool)
    if not labels_to_keep:
        return np.zeros_like(mask_bool, dtype=bool)
    return np.isin(labeled, list(labels_to_keep))


def _as_rgb_float(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    arr = arr[..., :3].astype(np.float32, copy=False)
    if np.issubdtype(np.asarray(image).dtype, np.integer):
        arr = arr / float(np.iinfo(np.asarray(image).dtype).max)
    elif arr.max(initial=0) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _component_reasons(
    *,
    area_fraction: float,
    area_rank: int,
    aspect_ratio: float,
    thinness: float,
    extent: float,
    edge_contact_fraction: float,
    mean_od: float,
    stain_fraction: float,
) -> list[str]:
    reasons: list[str] = []

    if area_fraction < 0.035 and stain_fraction < 0.35 and mean_od < 0.55:
        reasons.append("tiny_low_od_fragment")

    if area_rank > 4 and area_fraction < 0.12 and stain_fraction < 0.45:
        reasons.append("low_rank_low_stain_component")

    if aspect_ratio >= 7.0 and thinness <= 0.14 and stain_fraction < 0.55:
        reasons.append("thin_low_stain_component")

    if edge_contact_fraction > 0.18 and aspect_ratio >= 5.0 and extent < 0.45:
        reasons.append("edge_strip")

    return reasons


def score_components(
    image: np.ndarray,
    mask_bool: np.ndarray,
    *,
    stain_mask: np.ndarray | None = None,
    profile: ComponentQCProfile = "he_sections",
) -> list[ComponentQCRecord]:
    """
    Score connected components from a tissue mask.

    Parameters
    ----------
    image : np.ndarray
        Source RGB image.
    mask_bool : np.ndarray
        Boolean tissue mask at the same resolution as ``image``.
    stain_mask : np.ndarray, optional
        H&E stain-confidence mask. If omitted, an adaptive OD mask is computed.
    profile : str
        Component scoring profile. Currently only ``"he_sections"`` is defined.
    """
    if profile != "he_sections":
        raise ValueError("component QC profile must be 'he_sections'")

    rgb = _as_rgb_float(image)
    if rgb.shape[:2] != mask_bool.shape:
        raise ValueError("image and mask must have matching Y/X dimensions")

    if stain_mask is None:
        stain_mask = he_stain_mask(
            image,
            mode="adaptive-he",
            min_he_signal=0.0,
            od_bg_percentile=0.80,
            od_mad_multiplier=4.0,
        )
    stain_mask = np.asarray(stain_mask, dtype=bool)

    labeled, sorted_labels = label_components_left_to_right(mask_bool)
    if not sorted_labels:
        return []

    props_by_label = {p.label: p for p in measure.regionprops(labeled)}
    areas = {label: int(props_by_label[label].area) for label in sorted_labels}
    total_area = max(1, sum(areas.values()))
    area_order = {
        label: rank
        for rank, label in enumerate(
            sorted(sorted_labels, key=lambda item: areas[item], reverse=True),
            start=1,
        )
    }

    gray = rgb.mean(axis=-1)
    od = -np.log(np.clip(rgb, 1.0 / 255.0, 1.0)).sum(axis=-1)
    H, W = mask_bool.shape
    records: list[ComponentQCRecord] = []

    for tile_index, label in enumerate(sorted_labels):
        region = props_by_label[label]
        component = labeled == label
        y0, x0, y1, x1 = (int(v) for v in region.bbox)
        bbox_h = y1 - y0
        bbox_w = x1 - x0
        long_side = max(1, max(bbox_h, bbox_w))
        short_side = max(1, min(bbox_h, bbox_w))
        aspect_ratio = long_side / short_side
        thinness = short_side / long_side

        edge_pixels = 0
        if y0 == 0:
            edge_pixels += int(component[0, :].sum())
        if y1 == H:
            edge_pixels += int(component[H - 1, :].sum())
        if x0 == 0:
            edge_pixels += int(component[:, 0].sum())
        if x1 == W:
            edge_pixels += int(component[:, W - 1].sum())

        area = areas[label]
        area_fraction = area / total_area
        stain_fraction = float(stain_mask[component].mean()) if area else 0.0
        mean_intensity = float(gray[component].mean()) if area else 0.0
        mean_od = float(od[component].mean()) if area else 0.0
        edge_contact_fraction = edge_pixels / max(1, area)

        reasons = _component_reasons(
            area_fraction=area_fraction,
            area_rank=area_order[label],
            aspect_ratio=aspect_ratio,
            thinness=thinness,
            extent=float(region.extent),
            edge_contact_fraction=edge_contact_fraction,
            mean_od=mean_od,
            stain_fraction=stain_fraction,
        )

        records.append(
            ComponentQCRecord(
                component_label=int(label),
                tile_index_on_source=tile_index,
                area_rank=area_order[label],
                component_area_px=area,
                area_fraction=float(area_fraction),
                bbox_y0=y0,
                bbox_x0=x0,
                bbox_y1=y1,
                bbox_x1=x1,
                bbox_height=bbox_h,
                bbox_width=bbox_w,
                aspect_ratio=float(aspect_ratio),
                thinness=float(thinness),
                extent=float(region.extent),
                solidity=float(region.solidity),
                edge_contact_fraction=float(edge_contact_fraction),
                mean_intensity=mean_intensity,
                mean_od=mean_od,
                stain_fraction=stain_fraction,
                artifact_likely=bool(reasons),
                artifact_reason=";".join(reasons),
            )
        )

    return records


__all__ = [
    "ComponentQCMode",
    "ComponentQCProfile",
    "ComponentQCRecord",
    "filter_mask_by_labels",
    "label_components_left_to_right",
    "score_components",
]
