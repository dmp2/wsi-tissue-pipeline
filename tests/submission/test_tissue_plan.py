from __future__ import annotations

import ast
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from click.testing import CliRunner

from wsi_pipeline import cli
from wsi_pipeline.submission import (
    BatchStatus,
    TissuePlanRowCategory,
    TissuePlanStatus,
    plan_tissues_from_state,
    run_preflight,
)

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


def _touch_source(tmp_path: Path, name: str) -> Path:
    source_dir = tmp_path / "inputs"
    source_dir.mkdir(exist_ok=True)
    source_path = source_dir / name
    source_path.touch()
    return source_path


def _preflight_state(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *,
    profile_path: Path | None = None,
    columns: list[str] | None = None,
) -> Path:
    profile_path = profile_path or _write_profile(tmp_path)
    manifest_path = _write_manifest(tmp_path, rows, columns=columns)
    state_path = tmp_path / "preflight_state.json"
    run_preflight(profile_path, manifest_path, state_out_path=state_path)
    return state_path


def test_planning_succeeds_for_eligible_local_source_and_writes_plan_json(tmp_path):
    source_path = _touch_source(tmp_path, "slide.vsi")
    state_path = _preflight_state(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )
    plan_path = tmp_path / "tissue_detection_plan.json"

    result = plan_tissues_from_state(
        state_path,
        plan_out_path=plan_path,
        generated_at=FIXED_TIME,
    )
    payload = json.loads(plan_path.read_text(encoding="utf-8"))

    assert result.plan.plan_status is TissuePlanStatus.TISSUE_PLAN_READY
    assert result.plan.eligible_job_count == 1
    assert result.plan.blocked_row_count == 0
    assert result.plan.skipped_row_count == 0
    assert payload["plan_version"] == "1.0"
    assert payload["generated_at"].startswith("2026-01-02T03:04:05")
    assert payload["jobs"][0]["row_identifier"] == "S1/Slide1"
    assert payload["jobs"][0]["source_image_id"] == "S1/Slide1"
    assert payload["jobs"][0]["planned_stage"] == "tissue_detection"
    assert payload["jobs"][0]["status"] == "PLANNED"
    assert payload["jobs"][0]["outputs_planned"] == {
        "qc_contact_sheet": None,
        "review_state": None,
        "tissue_table": None,
    }
    assert plan_path.is_relative_to(tmp_path)


def test_planned_job_ids_are_deterministic(tmp_path):
    source_path = _touch_source(tmp_path, "slide.vsi")
    state_path = _preflight_state(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )

    first = plan_tissues_from_state(state_path, generated_at=FIXED_TIME).plan
    second = plan_tissues_from_state(
        state_path,
        generated_at=datetime(2027, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    ).plan

    assert first.jobs[0].job_id == second.jobs[0].job_id
    assert first.jobs[0].row_identifier == second.jobs[0].row_identifier == "S1/Slide1"


def test_rows_with_preflight_errors_are_blocked_not_planned(tmp_path):
    state_path = _preflight_state(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": "missing.vsi"}],
    )

    plan = plan_tissues_from_state(state_path, generated_at=FIXED_TIME).plan

    assert plan.plan_status is TissuePlanStatus.TISSUE_PLAN_BLOCKED
    assert plan.eligible_job_count == 0
    assert plan.blocked_row_count == 1
    assert plan.blocked_rows[0].category is TissuePlanRowCategory.BLOCKED_BY_PREFLIGHT_ERROR
    assert plan.blocked_rows[0].reason_code == "MISSING_LOCAL_SOURCE"


def test_deferred_source_metadata_is_preserved_on_planned_job(tmp_path):
    profile_path = _write_profile(
        tmp_path,
        metadata_flags={
            "require_physical_pixel_size": True,
            "require_parent_source_checksum": True,
        },
    )
    source_path = _touch_source(tmp_path, "slide.vsi")
    state_path = _preflight_state(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
        profile_path=profile_path,
    )

    plan = plan_tissues_from_state(state_path, generated_at=FIXED_TIME).plan
    deferred_codes = {issue.code for issue in plan.jobs[0].deferred_requirements}

    assert plan.plan_status is TissuePlanStatus.TISSUE_PLAN_READY_WITH_DEFERRED_REQUIREMENTS
    assert plan.eligible_job_count == 1
    assert plan.deferred_row_count == 1
    assert plan.jobs[0].category is TissuePlanRowCategory.DEFERRED_SOURCE_METADATA
    assert {"DEFERRED_SOURCE_PIXEL_SIZE", "DEFERRED_SOURCE_CHECKSUM"} <= deferred_codes
    assert all(issue.severity.value == "deferred" for issue in plan.jobs[0].deferred_requirements)


def test_nonlocal_uri_rows_are_skipped_not_missing_local_errors(tmp_path):
    state_path = _preflight_state(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Remote", "source_path": "s3://bucket/slide.vsi"}],
    )

    plan = plan_tissues_from_state(state_path, generated_at=FIXED_TIME).plan

    assert plan.eligible_job_count == 0
    assert plan.blocked_row_count == 0
    assert plan.skipped_row_count == 1
    assert plan.deferred_row_count == 1
    assert plan.skipped_rows[0].category is TissuePlanRowCategory.SKIPPED_NONLOCAL_SOURCE
    assert plan.skipped_rows[0].reason_code == "DEFERRED_NON_LOCAL_SOURCE_EXISTENCE"
    assert "MISSING_LOCAL_SOURCE" not in {issue.code for issue in plan.skipped_rows[0].issues}


def test_mixed_row_state_plans_eligible_rows_despite_failed_batch_preflight(tmp_path):
    profile_path = _write_profile(
        tmp_path,
        metadata_flags={
            "require_physical_pixel_size": True,
            "require_parent_source_checksum": True,
        },
    )
    ready_source = _touch_source(tmp_path, "ready.vsi")
    deferred_source = _touch_source(tmp_path, "deferred.vsi")
    columns = [
        "specimen_id",
        "slide_id",
        "source_path",
        "physical_pixel_size_x",
        "physical_pixel_size_y",
        "checksum",
    ]
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "specimen_id": "S1",
                "slide_id": "Ready",
                "source_path": str(ready_source),
                "physical_pixel_size_x": "0.5",
                "physical_pixel_size_y": "0.5",
                "checksum": "sha256:ready",
            },
            {"specimen_id": "S2", "slide_id": "Blocked", "source_path": "missing.vsi"},
            {"specimen_id": "S3", "slide_id": "Deferred", "source_path": str(deferred_source)},
            {"specimen_id": "S4", "slide_id": "Remote", "source_path": "s3://bucket/remote.vsi"},
        ],
        columns=columns,
    )
    state_path = tmp_path / "preflight_state.json"
    preflight = run_preflight(profile_path, manifest_path, state_out_path=state_path)

    plan = plan_tissues_from_state(state_path, generated_at=FIXED_TIME).plan

    assert preflight.report.batch_status is BatchStatus.PREFLIGHT_FAILED
    assert plan.plan_status is TissuePlanStatus.TISSUE_PLAN_READY_WITH_DEFERRED_REQUIREMENTS
    assert [job.row_identifier for job in plan.jobs] == ["S1/Ready", "S3/Deferred"]
    assert plan.blocked_rows[0].row_identifier == "S2/Blocked"
    assert plan.blocked_rows[0].reason_code == "MISSING_LOCAL_SOURCE"
    assert plan.skipped_rows[0].row_identifier == "S4/Remote"
    assert plan.skipped_rows[0].category is TissuePlanRowCategory.SKIPPED_NONLOCAL_SOURCE
    assert plan.jobs[0].deferred_requirements == []
    assert {issue.code for issue in plan.jobs[1].deferred_requirements} == {
        "DEFERRED_SOURCE_PIXEL_SIZE",
        "DEFERRED_SOURCE_CHECKSUM",
    }


def test_empty_state_produces_empty_plan_status(tmp_path):
    manifest_path = _write_manifest(tmp_path, [])
    state_path = tmp_path / "preflight_state.json"
    run_preflight(_write_profile(tmp_path), manifest_path, state_out_path=state_path)

    plan = plan_tissues_from_state(state_path, generated_at=FIXED_TIME).plan

    assert plan.plan_status is TissuePlanStatus.TISSUE_PLAN_EMPTY
    assert plan.total_rows == 0
    assert plan.eligible_job_count == 0


def test_cli_missing_state_file_returns_nonzero(tmp_path):
    result = CliRunner().invoke(
        cli.main,
        [
            "submit",
            "plan-tissues",
            "--state",
            str(tmp_path / "missing_state.json"),
            "--plan-out",
            str(tmp_path / "plan.json"),
        ],
    )

    assert result.exit_code != 0
    assert "Preflight state not found" in result.output
    assert "Traceback" not in result.output


def test_cli_malformed_state_json_returns_nonzero(tmp_path):
    state_path = tmp_path / "bad_state.json"
    state_path.write_text("{not-json", encoding="utf-8")

    result = CliRunner().invoke(
        cli.main,
        [
            "submit",
            "plan-tissues",
            "--state",
            str(state_path),
            "--plan-out",
            str(tmp_path / "plan.json"),
        ],
    )

    assert result.exit_code != 0
    assert "Malformed preflight state JSON" in result.output
    assert "Traceback" not in result.output


def test_cli_plan_tissues_help_works():
    result = CliRunner().invoke(cli.main, ["submit", "plan-tissues", "--help"])

    assert result.exit_code == 0
    assert "--state" in result.output
    assert "--plan-out" in result.output


def test_tissue_plan_has_no_processing_side_effects(tmp_path):
    source_path = _touch_source(tmp_path, "slide.vsi")
    state_path = _preflight_state(
        tmp_path,
        [{"specimen_id": "S1", "slide_id": "Slide1", "source_path": str(source_path)}],
    )
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    plan = plan_tissues_from_state(state_path, generated_at=FIXED_TIME).plan
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    assert plan.eligible_job_count == 1
    assert after == before


def test_tissue_plan_module_does_not_import_processing_or_viewer_integrations():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "src" / "wsi_pipeline" / "submission" / "tissue_plan.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    forbidden = {
        "bioformats",
        "cv2",
        "napari",
        "neuroglancer",
        "PIL",
        "pyjnius",
        "qupath",
        "skimage",
        "tifffile",
        "wsi_pipeline.neuroglancer",
        "wsi_pipeline.segmentation",
        "wsi_pipeline.vsi_converter",
    }
    assert imported_modules.isdisjoint(forbidden)
