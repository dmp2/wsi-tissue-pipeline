from __future__ import annotations

import ast
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from wsi_pipeline import cli
from wsi_pipeline.submission import WorkflowMode, run_setup

FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _write_profile(
    tmp_path: Path,
    *,
    accepted_extensions: tuple[str, ...] = (".vsi",),
    metadata_flags: dict[str, bool] | None = None,
) -> Path:
    metadata = {
        "require_physical_pixel_size": False,
        "require_units": False,
        "require_channel_metadata": False,
        "require_parent_source_checksum": False,
        "require_crop_bounds_parent_pixels": False,
        "require_child_array_to_physical_transform": False,
    }
    if metadata_flags:
        metadata.update(metadata_flags)

    def flag(value: bool) -> str:
        return "true" if value else "false"

    accepted_text = "\n".join(f"    - {extension}" for extension in accepted_extensions)
    metadata_text = "\n".join(f"  {key}: {flag(value)}" for key, value in metadata.items())
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        f"""
profile_name: test_profile
profile_version: 0.1.0
description: test profile
input:
  accepted_extensions:
{accepted_text}
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
  require_ome_xml: false
metadata:
{metadata_text}
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
  validate_ometiff: false
  validate_sidecar_consistency: false
  validate_checksums: false
  fail_batch_if_any_required_output_fails: false
""".lstrip(),
        encoding="utf-8",
    )
    return profile_path


def _write_manifest(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *,
    columns: list[str] | None = None,
) -> Path:
    manifest_path = tmp_path / "manifest.csv"
    if columns is None:
        columns = ["specimen_id", "slide_id", "source_path"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _write_source(tmp_path: Path, name: str, size: int) -> Path:
    source_dir = tmp_path / "inputs"
    source_dir.mkdir(exist_ok=True)
    source_path = source_dir / name
    source_path.write_bytes(b"x" * size)
    return source_path


def _codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_cli_setup_help_works():
    result = CliRunner().invoke(cli.main, ["submit", "setup", "--help"])

    assert result.exit_code == 0
    assert "--profile" in result.output
    assert "--manifest" in result.output
    assert "--mode" in result.output
    assert "existing-ometiff-upload" in result.output
    assert "--upload-mbps" in result.output


def test_existing_ometiff_upload_setup_report_success(tmp_path):
    profile_path = _write_profile(tmp_path, accepted_extensions=(".ome.tif",))
    source_path = _write_source(tmp_path, "slide.ome.tif", 1000)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    report = run_setup(
        profile_path,
        manifest_path,
        WorkflowMode.EXISTING_OMETIFF_UPLOAD,
        upload_mbps=100.0,
        upload_overhead=1.25,
        generated_at=FIXED_TIME,
    ).report

    assert report.ready_for_next_action is True
    assert report.workflow_mode is WorkflowMode.EXISTING_OMETIFF_UPLOAD
    assert report.row_count == 1
    assert report.valid_row_count == 1
    assert report.blocked_row_count == 0
    assert report.known_input_bytes == 1000
    assert report.unknown_size_row_count == 0
    assert report.estimated_output_bytes_low == 1000
    assert report.estimated_output_bytes_high == 1000
    assert report.estimated_processing_seconds_low == 0
    assert report.estimated_upload_seconds_low == pytest.approx(0.0001)
    assert "validation/package/upload" in report.recommended_next_action


def test_convert_single_tissue_setup_report_success(tmp_path):
    profile_path = _write_profile(tmp_path, accepted_extensions=(".vsi",))
    source_path = _write_source(tmp_path, "slide.vsi", 1_000_000)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    report = run_setup(
        profile_path,
        manifest_path,
        "convert-single-tissue",
        generated_at=FIXED_TIME,
    ).report

    assert report.ready_for_next_action is True
    assert report.estimated_output_bytes_low == 750_000
    assert report.estimated_output_bytes_high == 1_500_000
    assert report.estimated_processing_seconds_low == pytest.approx(0.3)
    assert report.estimated_processing_seconds_high == pytest.approx(1.8)
    assert report.next_action == "Run future single-tissue conversion stage."


def test_extract_convert_upload_setup_reuses_tissue_plan_row_eligibility(tmp_path):
    profile_path = _write_profile(tmp_path, accepted_extensions=(".vsi",))
    ready_source = _write_source(tmp_path, "ready.vsi", 10)
    manifest_path = _write_manifest(
        tmp_path,
        [
            {"specimen_id": "S1", "slide_id": "Ready", "source_path": str(ready_source)},
            {"specimen_id": "S2", "slide_id": "Blocked", "source_path": "missing.vsi"},
            {"specimen_id": "S3", "slide_id": "Remote", "source_path": "s3://bucket/remote.vsi"},
        ],
    )

    report = run_setup(
        profile_path,
        manifest_path,
        WorkflowMode.EXTRACT_CONVERT_UPLOAD,
        generated_at=FIXED_TIME,
    ).report
    by_identifier = {row.row_identifier: row for row in report.row_summaries}

    assert report.blocked_row_count == 1
    assert report.valid_row_count == 2
    assert report.deferred_row_count == 1
    assert report.known_input_bytes == 10
    assert report.unknown_size_row_count == 2
    assert report.estimated_output_bytes_low is None
    assert by_identifier["S1/Ready"].tissue_plan_category == "ELIGIBLE_FOR_TISSUE_DETECTION"
    assert by_identifier["S2/Blocked"].tissue_plan_category == "BLOCKED_BY_PREFLIGHT_ERROR"
    assert by_identifier["S2/Blocked"].tissue_plan_reason_code == "MISSING_LOCAL_SOURCE"
    assert by_identifier["S3/Remote"].tissue_plan_category == "SKIPPED_NONLOCAL_SOURCE"


def test_local_file_sizes_are_summed(tmp_path):
    profile_path = _write_profile(tmp_path, accepted_extensions=(".vsi",))
    source_a = _write_source(tmp_path, "a.vsi", 10)
    source_b = _write_source(tmp_path, "b.vsi", 20)
    manifest_path = _write_manifest(
        tmp_path,
        [
            {"specimen_id": "S1", "slide_id": "A", "source_path": str(source_a)},
            {"specimen_id": "S2", "slide_id": "B", "source_path": str(source_b)},
        ],
    )

    report = run_setup(profile_path, manifest_path, "convert-single-tissue").report

    assert report.known_input_bytes == 30
    assert report.unknown_size_row_count == 0


@pytest.mark.parametrize(
    ("mode", "extension", "expected_low", "expected_high"),
    [
        ("existing-ometiff-upload", ".ome.tif", 1000, 1000),
        ("convert-single-tissue", ".vsi", 750, 1500),
        ("extract-convert-upload", ".vsi", 250, 1500),
    ],
)
def test_output_size_range_is_deterministic_by_mode(
    tmp_path, mode, extension, expected_low, expected_high
):
    accepted_extensions = (extension,) if extension == ".vsi" else (".ome.tif",)
    profile_path = _write_profile(tmp_path, accepted_extensions=accepted_extensions)
    source_path = _write_source(tmp_path, f"slide{extension}", 1000)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    report = run_setup(profile_path, manifest_path, mode).report

    assert report.estimated_output_bytes_low == expected_low
    assert report.estimated_output_bytes_high == expected_high


def test_upload_time_calculation_is_deterministic(tmp_path):
    profile_path = _write_profile(tmp_path, accepted_extensions=(".ome.tif",))
    source_path = _write_source(tmp_path, "slide.ome.tif", 1_000_000)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    report = run_setup(
        profile_path,
        manifest_path,
        "existing-ometiff-upload",
        upload_mbps=10.0,
        upload_overhead=2.0,
    ).report

    assert report.estimated_upload_seconds_low == pytest.approx(1.6)
    assert report.estimated_upload_seconds_high == pytest.approx(1.6)
    assert report.estimated_total_seconds_low == pytest.approx(1.6)


def test_missing_local_source_file_blocks_setup(tmp_path):
    profile_path = _write_profile(tmp_path, accepted_extensions=(".vsi",))
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Missing", "source_path": "missing.vsi"}],
    )

    report = run_setup(profile_path, manifest_path, "convert-single-tissue").report

    assert report.ready_for_next_action is False
    assert report.blocked_row_count == 1
    assert report.unknown_size_row_count == 1
    assert report.estimated_output_bytes_low is None
    assert "MISSING_LOCAL_SOURCE" in _codes(report)


def test_nonlocal_uri_marks_full_batch_estimates_unavailable(tmp_path):
    profile_path = _write_profile(tmp_path, accepted_extensions=(".vsi",))
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Remote", "source_path": "s3://bucket/slide.vsi"}],
    )

    report = run_setup(profile_path, manifest_path, "convert-single-tissue").report

    assert report.error_count == 0
    assert report.ready_for_next_action is True
    assert report.deferred_row_count == 1
    assert report.unknown_size_row_count == 1
    assert report.estimated_output_bytes_low is None
    assert report.estimated_processing_seconds_low is None
    assert report.estimated_upload_seconds_low is None
    assert "DEFERRED_NON_LOCAL_SOURCE_EXISTENCE" in _codes(report)


def test_setup_json_is_written(tmp_path):
    profile_path = _write_profile(tmp_path, accepted_extensions=(".vsi",))
    source_path = _write_source(tmp_path, "slide.vsi", 123)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )
    report_path = tmp_path / "setup_report.json"

    result = run_setup(
        profile_path,
        manifest_path,
        "convert-single-tissue",
        setup_report_path=report_path,
        generated_at=FIXED_TIME,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.setup_report_path == report_path
    assert payload["setup_report_version"] == "1.0"
    assert payload["generated_at"].startswith("2026-01-02T03:04:05")
    assert payload["workflow_mode"] == "convert-single-tissue"
    assert payload["known_input_bytes"] == 123
    assert payload["row_summaries"][0]["row_identifier"] == "S1/Slide1"


def test_strict_mode_returns_nonzero_for_deferred_findings(tmp_path):
    profile_path = _write_profile(tmp_path, accepted_extensions=(".vsi",))
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Remote", "source_path": "s3://bucket/slide.vsi"}],
    )

    result = run_setup(profile_path, manifest_path, "convert-single-tissue", strict=True)

    assert result.report.error_count == 0
    assert result.report.ready_for_next_action is False
    assert result.exit_code() != 0


@pytest.mark.parametrize(
    ("mode", "source_name", "accepted_extensions", "expected_code"),
    [
        ("existing-ometiff-upload", "slide.vsi", (".vsi",), "WRONG_MODE_FOR_OMETIFF_UPLOAD"),
        (
            "convert-single-tissue",
            "slide.ome.tif",
            (".ome.tif",),
            "WRONG_MODE_FOR_SOURCE_CONVERSION",
        ),
        (
            "extract-convert-upload",
            "slide.ome.tif",
            (".ome.tif",),
            "WRONG_MODE_FOR_TISSUE_EXTRACTION",
        ),
    ],
)
def test_mode_input_compatibility_errors_are_reported(
    tmp_path, mode, source_name, accepted_extensions, expected_code
):
    profile_path = _write_profile(tmp_path, accepted_extensions=accepted_extensions)
    source_path = _write_source(tmp_path, source_name, 100)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    report = run_setup(profile_path, manifest_path, mode).report

    assert report.ready_for_next_action is False
    assert expected_code in _codes(report)


def test_cli_rejects_nonpositive_upload_parameters(tmp_path):
    profile_path = _write_profile(tmp_path, accepted_extensions=(".vsi",))
    source_path = _write_source(tmp_path, "slide.vsi", 1)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    result = CliRunner().invoke(
        cli.main,
        [
            "submit",
            "setup",
            "--profile",
            str(profile_path),
            "--manifest",
            str(manifest_path),
            "--mode",
            "convert-single-tissue",
            "--upload-mbps",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "must be greater than 0" in result.output
    assert "Traceback" not in result.output


def test_setup_module_does_not_import_processing_or_viewer_integrations():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "src" / "wsi_pipeline" / "submission" / "setup.py"
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
        "napari",
        "neuroglancer",
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
