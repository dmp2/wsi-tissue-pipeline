from __future__ import annotations

from pathlib import Path

from wsi_pipeline.submission import (
    BatchStatus,
    DetectedTissueSection,
    OutputDerivative,
    PixelBoundingBox,
    ReviewDecision,
    SourceSlide,
    SubmissionBatch,
    TissueStatus,
)


def test_status_enum_values_serialize_as_expected():
    assert BatchStatus.DRAFT.value == "DRAFT"
    assert TissueStatus.NEEDS_EXPERT_REVIEW.value == "NEEDS_EXPERT_REVIEW"
    assert ReviewDecision.DEFER.value == "DEFER"

    batch = SubmissionBatch(
        batch_id="batch-001",
        profile_name="national_database_ometiff",
        profile_version="0.1.0",
        output_root=Path("outputs"),
        status=BatchStatus.PREFLIGHT_READY,
    )

    assert batch.to_dict()["status"] == "PREFLIGHT_READY"


def test_source_slide_can_be_created_with_required_fields():
    slide = SourceSlide(
        specimen_id="E241",
        slide_id="Image_01",
        source_path=Path("inputs/Image_01.vsi"),
    )

    assert slide.specimen_id == "E241"
    assert slide.slide_id == "Image_01"
    assert slide.source_path == Path("inputs/Image_01.vsi")
    assert slide.warnings == []
    assert slide.errors == []


def test_detected_tissue_section_can_be_created_with_parent_pixel_bbox():
    tissue = DetectedTissueSection(
        specimen_id="E241",
        slide_id="Image_01",
        tissue_id="tissue00",
        bbox_parent_px=PixelBoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        status=TissueStatus.PENDING_REVIEW,
    )

    assert tissue.bbox_parent_px.x_min == 10
    assert tissue.status is TissueStatus.PENDING_REVIEW


def test_output_derivative_can_be_created_with_parent_provenance():
    derivative = OutputDerivative(
        specimen_id="E241",
        slide_id="Image_01",
        tissue_id="tissue00",
        output_path=Path("outputs/sub-E241_slide-Image_01_tissue-tissue00.ome.tif"),
        parent_source_path=Path("inputs/Image_01.vsi"),
        parent_checksum="sha256:parent",
        crop_bounds_parent_px=PixelBoundingBox(x_min=1, y_min=2, x_max=101, y_max=202),
        array_to_physical_transform=[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
        resampling_applied=False,
        flip_applied=False,
        rotation_applied=True,
        conversion_profile="national_database_ometiff",
        conversion_profile_version="0.1.0",
        output_checksum="sha256:output",
    )

    payload = derivative.to_dict()
    assert payload["is_derivative"] is True
    assert payload["derivative_reason"] == "split_from_multi_tissue_source_wsi"
    assert payload["parent_source_path"] == "inputs/Image_01.vsi"
    assert payload["parent_checksum"] == "sha256:parent"
    assert payload["crop_bounds_parent_px"] == {
        "x_min": 1,
        "y_min": 2,
        "x_max": 101,
        "y_max": 202,
    }
    assert payload["array_to_physical_transform"] == [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]
    assert payload["rotation_applied"] is True
    assert payload["output_checksum"] == "sha256:output"


def test_submission_batch_round_trips_through_dict_serialization():
    slide = SourceSlide(
        specimen_id="E241",
        slide_id="Image_01",
        source_path=Path("inputs/Image_01.vsi"),
    )
    tissue = DetectedTissueSection(
        specimen_id="E241",
        slide_id="Image_01",
        tissue_id="tissue00",
        bbox_parent_px=PixelBoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        status=TissueStatus.APPROVED,
        review_decision=ReviewDecision.APPROVE,
    )
    derivative = OutputDerivative(
        specimen_id="E241",
        slide_id="Image_01",
        tissue_id="tissue00",
        output_path=Path("outputs/sub-E241_slide-Image_01_tissue-tissue00.ome.tif"),
        parent_source_path=Path("inputs/Image_01.vsi"),
        crop_bounds_parent_px=PixelBoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        conversion_profile="national_database_ometiff",
        conversion_profile_version="0.1.0",
    )
    batch = SubmissionBatch(
        batch_id="batch-001",
        profile_name="national_database_ometiff",
        profile_version="0.1.0",
        input_root=Path("inputs"),
        output_root=Path("outputs"),
        manifest_path=Path("manifest.csv"),
        source_slides=[slide],
        tissue_sections=[tissue],
        derivatives=[derivative],
        status=BatchStatus.READY_FOR_CONVERSION,
    )

    payload = batch.to_dict()
    restored = SubmissionBatch.from_dict(payload)

    assert restored.to_dict() == payload
    assert payload["created_at"].endswith("Z") or "+00:00" in payload["created_at"]
