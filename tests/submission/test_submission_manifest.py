from __future__ import annotations

from pathlib import Path

import pytest

from wsi_pipeline.submission import (
    ManifestValidationError,
    load_database_profile,
    load_submission_manifest,
)

EXAMPLE_MANIFEST = Path("examples/submission_factory/example_submission_manifest.csv")
PROFILE_PATH = Path("configs/database_profiles/national_database_ometiff.yaml")


def test_manifest_loader_accepts_example_manifest():
    manifest = load_submission_manifest(EXAMPLE_MANIFEST)

    assert len(manifest.source_slides) == 1
    slide = manifest.source_slides[0]
    assert slide.specimen_id == "E241"
    assert slide.slide_id == "Image_01"
    assert slide.source_path == Path("/path/to/E241/Image_01.vsi")
    assert slide.section_number == 1


def test_manifest_loader_fails_when_required_columns_are_missing(tmp_path):
    manifest_path = tmp_path / "missing_columns.csv"
    manifest_path.write_text(
        """specimen_id,source_path
E241,/path/to/E241/Image_01.vsi
""",
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="missing required columns") as excinfo:
        load_submission_manifest(manifest_path)

    assert "slide_id" in str(excinfo.value)


def test_manifest_loader_reports_human_row_numbers_for_blank_required_fields(tmp_path):
    manifest_path = tmp_path / "blank_required.csv"
    manifest_path.write_text(
        """specimen_id,slide_id,source_path
,Image_01,/path/to/E241/Image_01.vsi
""",
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="row 2") as excinfo:
        load_submission_manifest(manifest_path)

    assert "specimen_id" in str(excinfo.value)


def test_manifest_loader_does_not_require_source_paths_to_exist_by_default(tmp_path):
    manifest_path = tmp_path / "planning_manifest.csv"
    manifest_path.write_text(
        """specimen_id,slide_id,source_path
E241,Image_01,missing.vsi
""",
        encoding="utf-8",
    )

    manifest = load_submission_manifest(manifest_path)

    assert len(manifest.source_slides) == 1
    assert manifest.source_slides[0].source_path == Path("missing.vsi")


def test_manifest_loader_can_optionally_check_source_paths(tmp_path):
    manifest_path = tmp_path / "path_check_manifest.csv"
    manifest_path.write_text(
        """specimen_id,slide_id,source_path
E241,Image_01,missing.vsi
""",
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="row 2") as excinfo:
        load_submission_manifest(manifest_path, check_paths=True)

    assert "does not exist" in str(excinfo.value)


def test_manifest_loader_accepts_vsi_extension_when_profile_is_provided(tmp_path):
    profile = load_database_profile(PROFILE_PATH)
    manifest_path = tmp_path / "profile_manifest.csv"
    manifest_path.write_text(
        """specimen_id,slide_id,source_path
E241,Image_01,/path/to/E241/Image_01.vsi
""",
        encoding="utf-8",
    )

    manifest = load_submission_manifest(manifest_path, profile=profile)

    assert len(manifest.source_slides) == 1
    assert manifest.source_slides[0].source_path.suffix == ".vsi"


def test_manifest_loader_rejects_unsupported_extension_when_profile_is_provided(tmp_path):
    profile = load_database_profile(PROFILE_PATH)
    manifest_path = tmp_path / "profile_manifest.csv"
    manifest_path.write_text(
        """specimen_id,slide_id,source_path
E241,Image_01,/path/to/E241/Image_01.tif
""",
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="row 2") as excinfo:
        load_submission_manifest(manifest_path, profile=profile)

    assert "not accepted" in str(excinfo.value)
    assert ".vsi" in str(excinfo.value)
