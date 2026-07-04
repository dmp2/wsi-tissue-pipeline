from __future__ import annotations

import ast
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from click.testing import CliRunner

from wsi_pipeline import cli
from wsi_pipeline.submission import run_package_ometiff_dry_run

FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _write_profile(
    tmp_path: Path,
    *,
    existing_extensions: tuple[str, ...] = (".ome.tif", ".ome.tiff"),
    naming_template: str = "sub-{specimen_id}_slide-{slide_id}_tissue-{tissue_id}.ome.tif",
    validation_enabled: bool = False,
    require_ome_xml: bool = False,
    emit_checksums: bool = False,
    validate_checksums: bool = False,
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
  emit_checksums: {flag(emit_checksums)}
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
  template: "{naming_template}"
  require_unique_output_names: true
validation:
  validate_ometiff: {flag(validation_enabled)}
  validate_sidecar_consistency: false
  validate_checksums: {flag(validate_checksums)}
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


def _write_ometiff_like_file(tmp_path: Path, name: str, size: int = 16) -> Path:
    source_dir = tmp_path / "inputs"
    source_dir.mkdir(exist_ok=True)
    source_path = source_dir / name
    source_path.write_bytes(b"x" * size)
    return source_path


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_cli_package_ometiff_help_works():
    result = CliRunner().invoke(cli.main, ["submit", "package-ometiff", "--help"])

    assert result.exit_code == 0
    assert "--profile" in result.output
    assert "--manifest" in result.output
    assert "--output-dir" in result.output
    assert "--dry-run" in result.output


def test_cli_package_ometiff_requires_dry_run(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_path = _write_ometiff_like_file(tmp_path, "slide.ome.tiff")
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    result = CliRunner().invoke(
        cli.main,
        [
            "submit",
            "package-ometiff",
            "--profile",
            str(profile_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "package"),
        ],
    )

    assert result.exit_code != 0
    assert "--dry-run is required" in result.output
    assert "Traceback" not in result.output


def test_successful_dry_run_writes_all_planning_artifacts(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_path = _write_ometiff_like_file(tmp_path, "slide.ome.tiff", size=25)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )
    output_dir = tmp_path / "package"

    result = run_package_ometiff_dry_run(
        profile_path,
        manifest_path,
        output_dir,
        generated_at=FIXED_TIME,
    )
    payload = json.loads((output_dir / "package_plan.json").read_text(encoding="utf-8"))
    csv_rows = _load_csv(output_dir / "package_manifest.csv")
    summary = (output_dir / "package_summary.txt").read_text(encoding="utf-8")

    assert result.exit_code() == 0
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "package_manifest.csv",
        "package_plan.json",
        "package_summary.txt",
    ]
    assert payload["package_plan_version"] == "1.0"
    assert payload["validation_scope"] == "filesystem_and_manifest_only"
    assert payload["row_count"] == 1
    assert payload["ready_count"] == 1
    assert payload["blocked_count"] == 0
    assert payload["ready_for_next_action"] is True
    assert payload["rows"][0]["file_size_bytes"] == 25
    assert payload["rows"][0]["source_exists"] is True
    assert payload["rows"][0]["source_is_local"] is True
    assert (
        payload["rows"][0]["planned_package_name"]
        == "sub-S1_slide-Slide1_tissue-slide_row0002.ome.tiff"
    )
    assert csv_rows[0]["package_status"] == "ready"
    assert csv_rows[0]["planned_package_name"] == payload["rows"][0]["planned_package_name"]
    assert "Ready for package review: yes" in summary
    assert (
        "copy/link/upload package execution is not implemented" in payload["deferred_capabilities"]
    )


def test_missing_source_file_produces_blocked_item(tmp_path):
    profile_path = _write_profile(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Missing", "source_path": "missing.ome.tiff"}],
    )

    report = run_package_ometiff_dry_run(
        profile_path,
        manifest_path,
        tmp_path / "package",
        generated_at=FIXED_TIME,
    ).report

    row = report.rows[0]
    assert report.ready_for_next_action is False
    assert row.package_status == "blocked"
    assert row.source_exists is False
    assert "OMETIFF_MISSING_LOCAL_FILE" in {issue.code for issue in row.blockers}


def test_zero_byte_source_file_produces_blocked_item(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_path = _write_ometiff_like_file(tmp_path, "empty.ome.tiff", size=0)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Empty", "source_path": str(source_path)}],
    )

    report = run_package_ometiff_dry_run(
        profile_path,
        manifest_path,
        tmp_path / "package",
        generated_at=FIXED_TIME,
    ).report

    row = report.rows[0]
    assert row.package_status == "blocked"
    assert row.file_size_bytes == 0
    assert "OMETIFF_EMPTY_FILE" in {issue.code for issue in row.blockers}


def test_planned_output_names_are_deterministic_and_use_identifier_fallback_order(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_a = _write_ometiff_like_file(tmp_path, "folder_a_image.ome.tiff", size=12)
    source_b = _write_ometiff_like_file(tmp_path, "folder_b_image.ome.tiff", size=12)
    source_c = _write_ometiff_like_file(tmp_path, "folder_c_image.ome.tiff", size=12)
    columns = ["specimen_id", "slide_id", "source_path", "section_id", "row_id"]
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "specimen_id": "S1",
                "slide_id": "A",
                "source_path": str(source_a),
                "section_id": "SEC-1",
            },
            {"specimen_id": "S1", "slide_id": "B", "source_path": str(source_b), "row_id": "ROW-2"},
            {"specimen_id": "S1", "slide_id": "C", "source_path": str(source_c)},
        ],
        columns=columns,
    )

    first = run_package_ometiff_dry_run(
        profile_path,
        manifest_path,
        tmp_path / "first",
        generated_at=FIXED_TIME,
    ).report
    second = run_package_ometiff_dry_run(
        profile_path,
        manifest_path,
        tmp_path / "second",
        generated_at=datetime(2027, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    ).report

    names = [row.planned_package_name for row in first.rows]
    assert names == [row.planned_package_name for row in second.rows]
    assert names == [
        "sub-S1_slide-A_tissue-SEC-1.ome.tiff",
        "sub-S1_slide-B_tissue-ROW-2.ome.tiff",
        "sub-S1_slide-C_tissue-folder_c_image_row0004.ome.tiff",
    ]


def test_duplicate_planned_package_names_are_blocked(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_a = _write_ometiff_like_file(tmp_path, "a.ome.tiff", size=12)
    source_b = _write_ometiff_like_file(tmp_path, "b.ome.tiff", size=12)
    columns = ["specimen_id", "slide_id", "source_path", "tissue_id"]
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "specimen_id": "S1",
                "slide_id": "Slide",
                "source_path": str(source_a),
                "tissue_id": "T1",
            },
            {
                "specimen_id": "S1",
                "slide_id": "Slide",
                "source_path": str(source_b),
                "tissue_id": "T1",
            },
        ],
        columns=columns,
    )

    report = run_package_ometiff_dry_run(
        profile_path,
        manifest_path,
        tmp_path / "package",
        generated_at=FIXED_TIME,
    ).report

    assert report.blocked_count == 2
    assert all(row.package_status == "blocked" for row in report.rows)
    assert all(
        "PACKAGE_NAME_COLLISION" in {issue.code for issue in row.blockers} for row in report.rows
    )


def test_unresolved_naming_template_placeholder_blocks_row(tmp_path):
    profile_path = _write_profile(
        tmp_path,
        naming_template="sub-{specimen_id}_slide-{slide_id}_donor-{donor_id}_tissue-{tissue_id}.ome.tif",
    )
    source_path = _write_ometiff_like_file(tmp_path, "slide.ome.tiff", size=12)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    report = run_package_ometiff_dry_run(
        profile_path,
        manifest_path,
        tmp_path / "package",
        generated_at=FIXED_TIME,
    ).report

    row = report.rows[0]
    assert row.package_status == "blocked"
    assert row.planned_package_name is None
    assert "PACKAGE_NAME_TEMPLATE_UNRESOLVED_PLACEHOLDER" in {issue.code for issue in row.blockers}


def test_json_and_csv_reports_contain_expected_rows(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_path = _write_ometiff_like_file(tmp_path, "slide.ome.tiff", size=12)
    columns = ["specimen_id", "slide_id", "source_path", "tissue_id", "sample_id"]
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "specimen_id": "S1",
                "slide_id": "Slide1",
                "source_path": str(source_path),
                "tissue_id": "T1",
                "sample_id": "SampleA",
            }
        ],
        columns=columns,
    )
    output_dir = tmp_path / "package"

    run_package_ometiff_dry_run(profile_path, manifest_path, output_dir, generated_at=FIXED_TIME)
    payload = json.loads((output_dir / "package_plan.json").read_text(encoding="utf-8"))
    csv_rows = _load_csv(output_dir / "package_manifest.csv")

    assert payload["rows"][0]["identifiers"]["sample_id"] == "SampleA"
    assert payload["rows"][0]["identifiers"]["tissue_id"] == "T1"
    assert payload["rows"][0]["planned_package_name"] == "sub-S1_slide-Slide1_tissue-T1.ome.tiff"
    assert csv_rows[0]["sample_id"] == "SampleA"
    assert csv_rows[0]["tissue_id"] == "T1"
    assert json.loads(csv_rows[0]["blocker_codes"]) == []


def test_deferred_ome_metadata_ome_xml_and_checksums_are_reported_honestly(tmp_path):
    profile_path = _write_profile(
        tmp_path,
        validation_enabled=True,
        require_ome_xml=True,
        emit_checksums=True,
        validate_checksums=True,
    )
    source_path = _write_ometiff_like_file(tmp_path, "slide.ome.tiff", size=12)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    report = run_package_ometiff_dry_run(
        profile_path,
        manifest_path,
        tmp_path / "package",
        generated_at=FIXED_TIME,
    ).report
    row = report.rows[0]
    codes = {issue.code for issue in row.deferred_checks}

    assert row.package_status == "ready_with_warnings"
    assert "OMETIFF_METADATA_VALIDATION_DEFERRED" in codes
    assert "OME_XML_PARSING_DEFERRED" in codes
    assert "CHECKSUM_COMPUTATION_DEFERRED" in codes
    assert "package/upload" not in " ".join(issue.message for issue in row.deferred_checks).lower()
    assert report.ready_for_next_action is True


def test_strict_mode_returns_nonzero_for_deferred_findings(tmp_path):
    profile_path = _write_profile(tmp_path, validation_enabled=True, require_ome_xml=True)
    source_path = _write_ometiff_like_file(tmp_path, "slide.ome.tiff", size=12)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    result = run_package_ometiff_dry_run(
        profile_path,
        manifest_path,
        tmp_path / "package",
        strict=True,
        generated_at=FIXED_TIME,
    )

    assert result.report.blocked_count == 0
    assert result.report.ready_for_next_action is False
    assert result.exit_code() != 0


def test_dry_run_does_not_create_payload_outputs_or_modify_sources(tmp_path):
    profile_path = _write_profile(tmp_path)
    source_path = _write_ometiff_like_file(tmp_path, "slide.ome.tiff", size=12)
    manifest_path = _write_manifest(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )
    source_stat = source_path.stat()
    output_dir = tmp_path / "package"
    sentinel = output_dir / "keep.txt"
    output_dir.mkdir()
    sentinel.write_text("do not delete", encoding="utf-8")

    run_package_ometiff_dry_run(profile_path, manifest_path, output_dir, generated_at=FIXED_TIME)

    assert source_path.read_bytes() == b"x" * 12
    after_stat = source_path.stat()
    assert after_stat.st_size == source_stat.st_size
    assert after_stat.st_nlink == source_stat.st_nlink
    assert sentinel.read_text(encoding="utf-8") == "do not delete"
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "keep.txt",
        "package_manifest.csv",
        "package_plan.json",
        "package_summary.txt",
    ]
    assert not any(path.suffix in {".tif", ".tiff"} for path in output_dir.iterdir())
    assert not any(path.is_symlink() for path in output_dir.iterdir())
    assert not (output_dir / "upload").exists()


def test_package_plan_module_does_not_import_processing_or_side_effect_integrations():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "src" / "wsi_pipeline" / "submission" / "ometiff_package_plan.py"
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
        "shutil",
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
