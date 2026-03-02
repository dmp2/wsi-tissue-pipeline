from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import wsi_pipeline.registration as registration_module


def _load_run_pipeline_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "run_pipeline.py"
    module_name = "test_run_pipeline_module"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_step5_cli_prefers_dataset_root_and_passes_new_flags(monkeypatch, tmp_path):
    run_pipeline = _load_run_pipeline_module()
    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            mode="atlas_free",
            plan_path=tmp_path / "emlddmm" / "resolved_run_plan.json",
            registration_output=tmp_path / "emlddmm",
            summary_path=tmp_path / "emlddmm" / "registration_summary.json",
        )

    monkeypatch.setattr(registration_module, "run_emlddmm_workflow", fake_runner)

    parser = run_pipeline.build_parser()
    args = parser.parse_args(
        [
            "step5",
            "--dataset-root",
            str(tmp_path),
            "--dry-run",
            "--skip-self-alignment",
            "--write-notebook-bundle",
        ]
    )
    rc = run_pipeline.step5_reconstruct(args, config=None)

    assert rc == 0
    assert captured["dataset_root"] == tmp_path
    assert captured["target_source"] == tmp_path
    assert captured["dry_run"] is True
    assert captured["skip_self_alignment"] is True
    assert captured["write_notebook_bundle"] is True
    assert captured["write_qc_report"] is False
    assert captured["used_legacy_output_alias"] is False
    assert captured["verbose"] is False


def test_step5_cli_legacy_output_alias_still_works(monkeypatch, tmp_path):
    run_pipeline = _load_run_pipeline_module()
    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            mode="atlas_free",
            plan_path=tmp_path / "emlddmm" / "resolved_run_plan.json",
            registration_output=tmp_path / "emlddmm",
            summary_path=tmp_path / "emlddmm" / "registration_summary.json",
        )

    monkeypatch.setattr(registration_module, "run_emlddmm_workflow", fake_runner)

    parser = run_pipeline.build_parser()
    args = parser.parse_args(["step5", "-o", str(tmp_path)])
    rc = run_pipeline.step5_reconstruct(args, config=None)

    assert rc == 0
    assert captured["dataset_root"] == tmp_path
    assert captured["output_dir"] == tmp_path
    assert captured["target_source"] == tmp_path
    assert captured["used_legacy_output_alias"] is True


def test_step5_cli_passes_atlas_precomputed_and_transformation_graph(monkeypatch, tmp_path):
    run_pipeline = _load_run_pipeline_module()
    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            mode="atlas_to_target",
            plan_path=tmp_path / "emlddmm" / "resolved_run_plan.json",
            registration_output=tmp_path / "emlddmm",
            summary_path=tmp_path / "emlddmm" / "registration_summary.json",
        )

    monkeypatch.setattr(registration_module, "run_emlddmm_workflow", fake_runner)

    parser = run_pipeline.build_parser()
    args = parser.parse_args(
        [
            "step5",
            "--dataset-root",
            str(tmp_path),
            "--atlas",
            str(tmp_path / "atlas.vtk"),
            "--label",
            str(tmp_path / "labels.vtk"),
            "--target-source",
            str(tmp_path / "precomputed"),
            "--target-source-format",
            "precomputed",
            "--precomputed-manifest",
            str(tmp_path / "manifest.json"),
            "--orientation-from",
            "PIR",
            "--orientation-to",
            "RIP",
            "--upsample-between-slices",
            "--upsample-mode",
            "img",
            "--run-transformation-graph",
            "--transformation-graph-script",
            str(tmp_path / "external" / "transformation_graph_v01.py"),
            "--write-qc-report",
        ]
    )
    rc = run_pipeline.step5_reconstruct(args, config=None)

    assert rc == 0
    assert captured["atlas"] == tmp_path / "atlas.vtk"
    assert captured["label"] == tmp_path / "labels.vtk"
    assert captured["target_source"] == tmp_path / "precomputed"
    assert captured["target_source_format"] == "precomputed"
    assert captured["precomputed_manifest"] == tmp_path / "manifest.json"
    assert captured["orientation_from"] == "PIR"
    assert captured["orientation_to"] == "RIP"
    assert captured["upsample_between_slices"] is True
    assert captured["upsample_mode"] == "img"
    assert captured["run_transformation_graph"] is True
    assert captured["transformation_graph_script"] == tmp_path / "external" / "transformation_graph_v01.py"
    assert captured["write_qc_report"] is True


def test_step5_cli_lists_orientations_without_running(monkeypatch, capsys):
    run_pipeline = _load_run_pipeline_module()

    def fail_runner(**kwargs):
        raise AssertionError("run_emlddmm_workflow should not be called for --list-orientations")

    monkeypatch.setattr(registration_module, "run_emlddmm_workflow", fail_runner)

    parser = run_pipeline.build_parser()
    args = parser.parse_args(["step5", "--list-orientations"])
    rc = run_pipeline.step5_reconstruct(args, config=None)

    output = capsys.readouterr().out
    assert rc == 0
    assert "Orientation codes must be exactly 3 letters" in output
    assert "RAS" in output
    assert "LPI" in output
