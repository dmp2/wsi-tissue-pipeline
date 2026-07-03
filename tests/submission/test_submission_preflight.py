from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from wsi_pipeline import cli
from wsi_pipeline.submission import (
    BatchStatus,
    IssueSeverity,
    ProfileValidationError,
    load_database_profile,
    run_preflight,
)


def _write_profile(
    tmp_path: Path,
    *,
    accepted_extensions: tuple[str, ...] = (".vsi",),
    metadata_flags: dict[str, bool] | None = None,
    requirement_phases: dict[str, str] | None = None,
    validation_enabled: bool = False,
    qc_enabled: bool = False,
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
    requirement_text = ""
    if requirement_phases:
        requirement_lines = ["requirement_phases:"]
        requirement_lines.extend(f"  {key}: {value}" for key, value in requirement_phases.items())
        requirement_text = "\n" + "\n".join(requirement_lines) + "\n"

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
{requirement_text}qc:
  require_tissue_detection_qc: {flag(qc_enabled)}
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


def _touch_source(tmp_path: Path, name: str = "slide.vsi") -> Path:
    source_dir = tmp_path / "inputs"
    source_dir.mkdir(exist_ok=True)
    source_path = source_dir / name
    source_path.touch()
    return source_path


def _codes(report) -> set[str]:
    codes = {issue.code for issue in report.batch_issues}
    for row in report.row_results:
        codes.update(issue.code for issue in row.issues)
    return codes


def test_run_preflight_successful_minimal_manifest(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_path = _touch_source(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    result = run_preflight(profile_path, manifest_path)

    assert result.report.batch_status is BatchStatus.PREFLIGHT_PASSED
    assert result.report.total_row_count == 1
    assert result.report.valid_row_count == 1
    assert result.report.error_count == 0
    assert result.exit_code() == 0


def test_cli_missing_profile_path_returns_nonzero(tmp_path):
    source_path = _touch_source(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    result = CliRunner().invoke(
        cli.main,
        [
            "submit",
            "preflight",
            "--profile",
            str(tmp_path / "missing.yaml"),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert result.exit_code != 0
    assert "Database profile not found" in result.output
    assert "Traceback" not in result.output


def test_cli_missing_manifest_path_returns_nonzero(tmp_path):
    profile_path = _write_profile(tmp_path)

    result = CliRunner().invoke(
        cli.main,
        [
            "submit",
            "preflight",
            "--profile",
            str(profile_path),
            "--manifest",
            str(tmp_path / "missing.csv"),
        ],
    )

    assert result.exit_code != 0
    assert "Submission manifest not found" in result.output
    assert "Traceback" not in result.output


def test_invalid_manifest_row_continues_after_row_failure(tmp_path):
    profile_path = _write_profile(tmp_path)
    bad_source = _touch_source(tmp_path, "bad.vsi")
    good_source = _touch_source(tmp_path, "good.vsi")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {"specimen_id": "", "slide_id": "Bad", "source_path": str(bad_source)},
            {"specimen_id": "S1", "slide_id": "Good", "source_path": str(good_source)},
        ],
    )

    result = run_preflight(profile_path, manifest_path)

    assert result.report.total_row_count == 2
    assert result.report.valid_row_count == 1
    assert result.report.rows_with_errors == 1
    assert "MISSING_REQUIRED_FIELD" in _codes(result.report)
    assert result.exit_code() != 0


def test_disallowed_source_extension_is_error(tmp_path):
    profile_path = _write_profile(tmp_path, accepted_extensions=(".vsi",))
    source_path = _touch_source(tmp_path, "slide.tif")
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    result = run_preflight(profile_path, manifest_path)

    assert result.report.batch_status is BatchStatus.PREFLIGHT_FAILED
    assert "DISALLOWED_SOURCE_EXTENSION" in _codes(result.report)
    assert result.report.valid_row_count == 0


def test_missing_local_source_file_is_error(tmp_path):
    profile_path = _write_profile(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": "missing.vsi"}],
    )

    result = run_preflight(profile_path, manifest_path)

    assert result.report.batch_status is BatchStatus.PREFLIGHT_FAILED
    assert "MISSING_LOCAL_SOURCE" in _codes(result.report)
    assert result.exit_code() != 0


def test_non_local_source_existence_check_is_deferred(tmp_path):
    profile_path = _write_profile(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": "s3://bucket/slide.vsi"}],
    )

    result = run_preflight(profile_path, manifest_path)

    assert result.report.error_count == 0
    assert result.report.valid_row_count == 1
    assert result.report.batch_status is BatchStatus.PREFLIGHT_PASSED_WITH_DEFERRED_REQUIREMENTS
    assert "DEFERRED_NON_LOCAL_SOURCE_EXISTENCE" in _codes(result.report)
    assert result.exit_code() == 0


def test_json_report_is_written(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_path = _touch_source(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )
    report_path = tmp_path / "report.json"

    result = run_preflight(profile_path, manifest_path, json_report_path=report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.json_report_path == report_path
    assert payload["preflight_report_version"] == "1.0"
    assert payload["total_row_count"] == 1
    assert payload["valid_row_count"] == 1
    assert payload["batch_status"] == "PREFLIGHT_PASSED"


def test_state_output_is_written(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_path = _touch_source(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )
    report_path = tmp_path / "report.json"
    state_path = tmp_path / "state.json"

    result = run_preflight(
        profile_path,
        manifest_path,
        json_report_path=report_path,
        state_out_path=state_path,
    )
    payload = json.loads(state_path.read_text(encoding="utf-8"))

    assert result.state_out_path == state_path
    assert payload["preflight_state_version"] == "1.0"
    assert payload["preflight_status"] == "PREFLIGHT_PASSED"
    assert payload["json_report_path"] == str(report_path)
    assert payload["row_statuses"][0]["valid"] is True


def test_strict_mode_returns_nonzero_without_relabeling_deferred(tmp_path):
    profile_path = _write_profile(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": "s3://bucket/slide.vsi"}],
    )

    result = run_preflight(profile_path, manifest_path, strict=True)

    assert result.report.error_count == 0
    assert result.exit_code() != 0
    deferred_issues = [
        issue
        for row in result.report.row_results
        for issue in row.issues
        if issue.code == "DEFERRED_NON_LOCAL_SOURCE_EXISTENCE"
    ]
    assert deferred_issues[0].severity is IssueSeverity.DEFERRED


def test_missing_source_metadata_is_deferred_not_error(tmp_path):
    profile_path = _write_profile(
        tmp_path,
        metadata_flags={
            "require_physical_pixel_size": True,
            "require_parent_source_checksum": True,
        },
    )
    source_path = _touch_source(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    result = run_preflight(profile_path, manifest_path)

    assert result.report.error_count == 0
    assert result.report.valid_row_count == 1
    assert result.report.batch_status is BatchStatus.PREFLIGHT_PASSED_WITH_DEFERRED_REQUIREMENTS
    assert "DEFERRED_SOURCE_PIXEL_SIZE" in _codes(result.report)
    assert "DEFERRED_SOURCE_CHECKSUM" in _codes(result.report)
    assert result.exit_code() == 0


def test_preflight_manifest_metadata_requirement_fails_when_missing(tmp_path):
    profile_path = _write_profile(
        tmp_path,
        metadata_flags={
            "require_physical_pixel_size": True,
            "require_parent_source_checksum": True,
        },
        requirement_phases={
            "physical_pixel_size": "preflight_manifest",
            "parent_source_checksum": "preflight_manifest",
        },
    )
    source_path = _touch_source(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    result = run_preflight(profile_path, manifest_path)

    assert result.report.batch_status is BatchStatus.PREFLIGHT_FAILED
    assert "MISSING_SOURCE_PIXEL_SIZE" in _codes(result.report)
    assert "MISSING_SOURCE_CHECKSUM" in _codes(result.report)
    assert result.report.valid_row_count == 0


def test_duplicate_specimen_slide_pair_is_warning(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_a = _touch_source(tmp_path, "a.vsi")
    source_b = _touch_source(tmp_path, "b.vsi")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_a)},
            {"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_b)},
        ],
    )

    result = run_preflight(profile_path, manifest_path)

    assert result.report.error_count == 0
    assert result.report.warning_count == 1
    assert result.report.valid_row_count == 2
    assert result.report.batch_status is BatchStatus.PREFLIGHT_PASSED_WITH_WARNINGS
    assert "DUPLICATE_SOURCE_SLIDE_IDENTIFIER" in _codes(result.report)


def test_unknown_requirement_phase_fails_profile_validation(tmp_path):
    profile_path = _write_profile(
        tmp_path,
        requirement_phases={"physical_pixel_size": "not_a_phase"},
    )

    with pytest.raises(ProfileValidationError, match="unknown phase"):
        load_database_profile(profile_path)


def test_preflight_has_no_processing_side_effects(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_path = _touch_source(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    result = run_preflight(profile_path, manifest_path)
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    assert result.report.batch_status is BatchStatus.PREFLIGHT_PASSED
    assert after == before
