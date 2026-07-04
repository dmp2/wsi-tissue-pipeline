from __future__ import annotations

import ast
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from wsi_pipeline import cli
from wsi_pipeline.submission import load_database_profile, run_validate_ometiff

FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
BUNDLED_PROFILE = Path("configs/database_profiles/national_database_ometiff.yaml")


def _write_profile(
    tmp_path: Path,
    *,
    existing_extensions: tuple[str, ...] = (".ome.tif", ".ome.tiff"),
    validation_enabled: bool = False,
    require_ome_xml: bool = False,
) -> Path:
    def flag(value: bool) -> str:
        return "true" if value else "false"

    existing_text = "\n".join(f"      - {extension}" for extension in existing_extensions)
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        f"""
profile_name: test_profile
profile_version: 0.1.0
description: test profile
input:
  accepted_extensions:
    - .vsi
  workflow_mode_extensions:
    existing-ometiff-upload:
{existing_text}
    convert-single-tissue:
      - .vsi
      - .ets
    extract-convert-upload:
      - .vsi
      - .ets
  require_ets_companion: false
  raw_files_read_only: true
output:
  mode: single_tissue_section_ome_tiff
  extension: .ome.tif
  emit_sidecar_json: false
  emit_qc_png: false
  emit_per_tissue_provenance: false
  emit_batch_manifest: false
  emit_checksums: false
ometiff:
  tiled: true
  pyramidal: true
  bigtiff: true
  preserve_source_resolution: true
  compression: default
  tile_size: default
  require_ome_xml: {flag(require_ome_xml)}
metadata:
  require_physical_pixel_size: false
  require_units: false
  require_channel_metadata: false
  require_parent_source_checksum: false
  require_crop_bounds_parent_pixels: false
  require_child_array_to_physical_transform: false
qc:
  require_tissue_detection_qc: false
  require_operator_approval: false
  require_expert_review_for_warnings: false
  block_on_failed_segmentation: false
  block_on_missing_parent_mapping: false
naming:
  template: "sub-{{specimen_id}}_slide-{{slide_id}}_tissue-{{tissue_id}}.ome.tif"
  require_unique_output_names: true
validation:
  validate_ometiff: {flag(validation_enabled)}
  validate_sidecar_consistency: false
  validate_checksums: false
  fail_batch_if_any_required_output_fails: false
""".lstrip(),
        encoding="utf-8",
    )
    return profile_path


def _write_manifest(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["specimen_id", "slide_id", "source_path"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _write_ometiff_like_filesystem_fixture(tmp_path: Path, name: str, size: int = 16) -> Path:
    source_dir = tmp_path / "inputs"
    source_dir.mkdir(exist_ok=True)
    source_path = source_dir / name
    source_path.write_bytes(b"x" * size)
    return source_path


def _codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_cli_validate_ometiff_help_works():
    result = CliRunner().invoke(cli.main, ["submit", "validate-ometiff", "--help"])

    assert result.exit_code == 0
    assert "--profile" in result.output
    assert "--manifest" in result.output
    assert "--validation-report" in result.output
    assert "filesystem/manifest-only" in result.output


@pytest.mark.parametrize("filename", ["slide.ome.tif", "slide.ome.tiff"])
def test_ometiff_like_filesystem_fixture_passes_structural_check(tmp_path, filename):
    profile_path = _write_profile(tmp_path)
    source_path = _write_ometiff_like_filesystem_fixture(tmp_path, filename, size=25)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    report = run_validate_ometiff(profile_path, manifest_path, generated_at=FIXED_TIME).report

    assert report.validation_scope == "filesystem_and_manifest_only"
    assert report.workflow_mode.value == "existing-ometiff-upload"
    assert report.error_count == 0
    assert report.valid_row_count == 1
    assert report.blocked_row_count == 0
    assert report.known_input_bytes == 25
    assert report.ready_for_next_action is True


def test_generic_tif_fails_unless_profile_explicitly_allows_generic_tiff(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_path = _write_ometiff_like_filesystem_fixture(tmp_path, "slide.tif", size=25)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    blocked = run_validate_ometiff(profile_path, manifest_path).report

    assert blocked.ready_for_next_action is False
    assert "OMETIFF_WRONG_MODE_EXTENSION" in _codes(blocked)

    generic_profile_path = _write_profile(
        tmp_path, existing_extensions=(".ome.tif", ".ome.tiff", ".tif")
    )
    allowed = run_validate_ometiff(generic_profile_path, manifest_path).report

    assert allowed.error_count == 0
    assert allowed.valid_row_count == 1


@pytest.mark.parametrize("filename", ["slide.vsi", "slide.ets"])
def test_source_microscopy_extensions_fail_as_wrong_mode(tmp_path, filename):
    profile_path = _write_profile(tmp_path)
    source_path = _write_ometiff_like_filesystem_fixture(tmp_path, filename, size=25)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    report = run_validate_ometiff(profile_path, manifest_path).report

    assert report.ready_for_next_action is False
    assert report.blocked_row_count == 1
    assert "OMETIFF_WRONG_MODE_EXTENSION" in _codes(report)


def test_missing_local_file_blocks_structural_check(tmp_path):
    profile_path = _write_profile(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Missing", "source_path": "missing.ome.tiff"}],
    )

    report = run_validate_ometiff(profile_path, manifest_path).report

    assert report.ready_for_next_action is False
    assert report.blocked_row_count == 1
    assert report.known_input_bytes == 0
    assert "OMETIFF_MISSING_LOCAL_FILE" in _codes(report)


def test_zero_byte_file_blocks_structural_check(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_path = _write_ometiff_like_filesystem_fixture(tmp_path, "empty.ome.tiff", size=0)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Empty", "source_path": str(source_path)}],
    )

    report = run_validate_ometiff(profile_path, manifest_path).report

    assert report.ready_for_next_action is False
    assert report.blocked_row_count == 1
    assert report.known_input_bytes == 0
    assert "OMETIFF_EMPTY_FILE" in _codes(report)


def test_nonlocal_uri_is_deferred_not_crashed(tmp_path):
    profile_path = _write_profile(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Remote", "source_path": "s3://bucket/slide.ome.tiff"}],
    )

    result = run_validate_ometiff(profile_path, manifest_path)

    assert result.exit_code() == 0
    assert result.report.error_count == 0
    assert result.report.deferred_count == 1
    assert result.report.ready_for_next_action is True
    assert "OMETIFF_NONLOCAL_DEFERRED" in _codes(result.report)


def test_validation_json_is_written(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_path = _write_ometiff_like_filesystem_fixture(tmp_path, "slide.ome.tiff", size=12)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )
    report_path = tmp_path / "reports" / "ometiff_structural_report.json"

    result = run_validate_ometiff(
        profile_path,
        manifest_path,
        validation_report_path=report_path,
        generated_at=FIXED_TIME,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.validation_report_path == report_path
    assert payload["validation_report_version"] == "1.0"
    assert payload["validation_scope"] == "filesystem_and_manifest_only"
    assert payload["workflow_mode"] == "existing-ometiff-upload"
    assert payload["known_input_bytes"] == 12
    assert payload["row_summaries"][0]["row_identifier"] == "S1/Slide1"


def test_strict_mode_returns_nonzero_for_deferred_findings(tmp_path):
    profile_path = _write_profile(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Remote", "source_path": "s3://bucket/slide.ome.tiff"}],
    )

    result = run_validate_ometiff(profile_path, manifest_path, strict=True)

    assert result.report.error_count == 0
    assert result.report.ready_for_next_action is False
    assert result.exit_code() != 0


def test_ometiff_metadata_validation_is_deferred_to_future_step(tmp_path):
    profile_path = _write_profile(tmp_path, validation_enabled=True, require_ome_xml=True)
    source_path = _write_ometiff_like_filesystem_fixture(tmp_path, "slide.ome.tiff", size=12)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    report = run_validate_ometiff(profile_path, manifest_path).report

    assert report.error_count == 0
    assert report.deferred_count == 1
    assert report.next_action == "ome_tiff_metadata_validation"
    assert (
        report.recommended_next_action
        == "Run future OME-TIFF metadata validation before packaging/upload."
    )
    assert "OMETIFF_METADATA_VALIDATION_DEFERRED" in _codes(report)


def test_report_is_deterministic_with_fixed_timestamp(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_path = _write_ometiff_like_filesystem_fixture(tmp_path, "slide.ome.tiff", size=12)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    run_validate_ometiff(
        profile_path, manifest_path, validation_report_path=first_path, generated_at=FIXED_TIME
    )
    run_validate_ometiff(
        profile_path, manifest_path, validation_report_path=second_path, generated_at=FIXED_TIME
    )

    assert json.loads(first_path.read_text(encoding="utf-8")) == json.loads(
        second_path.read_text(encoding="utf-8")
    )


def test_bundled_profile_supports_existing_ometiff_workflow_mode(tmp_path):
    profile = load_database_profile(BUNDLED_PROFILE)
    assert profile.accepted_extensions_for_workflow_mode("existing-ometiff-upload") == [
        ".ome.tif",
        ".ome.tiff",
    ]
    source_path = _write_ometiff_like_filesystem_fixture(tmp_path, "slide.ome.tiff", size=12)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    report = run_validate_ometiff(BUNDLED_PROFILE, manifest_path, generated_at=FIXED_TIME).report

    assert report.error_count == 0
    assert report.valid_row_count == 1
    assert report.next_action == "ome_tiff_metadata_validation"


def test_validate_ometiff_has_no_processing_side_effects(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_path = _write_ometiff_like_filesystem_fixture(tmp_path, "slide.ome.tiff", size=12)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    report = run_validate_ometiff(profile_path, manifest_path, generated_at=FIXED_TIME).report
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    assert report.error_count == 0
    assert after == before


def test_validate_ometiff_module_does_not_import_processing_or_viewer_integrations():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "src" / "wsi_pipeline" / "submission" / "ometiff_validation.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    forbidden = {
        "bioformats",
        "boto3",
        "cv2",
        "hashlib",
        "napari",
        "neuroglancer",
        "notebook",
        "PIL",
        "pyjnius",
        "qupath",
        "requests",
        "skimage",
        "tifffile",
        "wsi_pipeline.bioformats_runtime",
        "wsi_pipeline.neuroglancer",
        "wsi_pipeline.omezarr",
        "wsi_pipeline.pipeline",
        "wsi_pipeline.qc_grid",
        "wsi_pipeline.segmentation",
        "wsi_pipeline.vsi_converter",
    }
    assert imported_modules.isdisjoint(forbidden)
