from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import wsi_pipeline.registration.workflow as workflow_module


def _expected_target_volume() -> np.ndarray:
    volume = np.zeros((1, 3, 2, 2), dtype=np.float32)
    volume[:, 0] = 3.0
    volume[:, 2] = 6.0
    return volume


def _axes_with_spacing(spacing: list[float], *, length: int = 4) -> list[np.ndarray]:
    return [
        np.arange(length, dtype=np.float32) * np.float32(axis_spacing)
        for axis_spacing in spacing
    ]


class FakeBackend:
    name = "fake-backend"
    origin_type = "vendored"
    package_version = None
    read_matrix_data = None

    def __init__(self) -> None:
        self.last_multiscale_kwargs = None
        self.module = SimpleNamespace(__file__=__file__)

    def read_data(self, path):
        path = Path(path)
        x = [
            np.array([0.0, 5.0, 10.0], dtype=np.float32),
            np.array([-1.0, 1.0], dtype=np.float32),
            np.array([-1.0, 1.0], dtype=np.float32),
        ]
        if path.name == "atlas.vtk":
            return x, np.ones((1, 3, 2, 2), dtype=np.float32), "atlas", ["atlas"]
        if path.name == "labels.vtk":
            return x, np.ones((1, 3, 2, 2), dtype=np.float32), "labels", ["labels"]

        J = _expected_target_volume()
        W0 = np.zeros((3, 2, 2), dtype=np.float32)
        W0[0] = 1.0
        W0[2] = 1.0
        images = np.concatenate([J, W0[None]], axis=0)
        return x, images, "slice_dataset", ["red", "mask"]

    def atlas_free_reconstruction(self, *, xJ, J, W, **kwargs):
        A2d = np.repeat(np.eye(3, dtype=np.float32)[None], J.shape[1], axis=0)
        return {"A2d": A2d, "I": np.asarray(J), "Jr": np.asarray(J)}

    def emlddmm_multiscale(self, **kwargs):
        self.last_multiscale_kwargs = kwargs
        return {
            "A": np.eye(4, dtype=np.float32),
            "A2d": kwargs["A2d"],
            "xv": [],
            "v": 0.0,
        }

    def orientation_to_orientation(self, src, dst):
        return np.eye(3, dtype=np.float32)

    def downsample_image_domain(self, xI, I, down, W=None):
        if W is None:
            return xI, I
        return xI, I, W

    def write_transform_outputs(self, output_dir, registration, atlas_image, target_image):
        transforms_dir = Path(output_dir) / "transforms"
        transforms_dir.mkdir(parents=True, exist_ok=True)
        (transforms_dir / "affine.txt").write_text("affine", encoding="utf-8")

    def write_qc_outputs(self, output_dir, registration, atlas_image, target_image, xS=None, S=None):
        qc_dir = Path(output_dir) / "qc"
        images_dir = Path(output_dir) / "images"
        nested_qc_dir = qc_dir / "atlas" / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        nested_qc_dir.mkdir(parents=True, exist_ok=True)
        (qc_dir / "overview.png").write_text("png", encoding="utf-8")
        (nested_qc_dir / "slice_0001.jpg").write_text("jpg", encoding="utf-8")
        (images_dir / "overlay.png").write_text("png", encoding="utf-8")

    def write_vtk_data(self, path, x, data, title):
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(title, encoding="utf-8")

    def write_matrix_data(self, path, matrix):
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_path, np.asarray(matrix, dtype=np.float32))


def _write_override(tmp_path: Path, payload: dict) -> Path:
    override_path = tmp_path / "override.json"
    override_path.write_text(json.dumps(payload), encoding="utf-8")
    return override_path


def test_plan_workflow_resolves_atlas_free_prepared_dir(tmp_path):
    backend = FakeBackend()

    plan = workflow_module.plan_emlddmm_workflow(
        dataset_root=tmp_path,
        target_source=tmp_path,
        target_source_format="prepared-dir",
        registration_output=tmp_path / "emlddmm",
        backend=backend,
        dry_run=True,
    )

    assert plan.mode == "atlas_free"
    assert plan.enabled_stages == ["self_alignment"]
    assert plan.skipped_stages["atlas_registration"] == "no atlas provided"
    assert plan.target_source_format == "prepared-dir"
    assert "self_alignment" in plan.expected_outputs
    assert plan.workflow_config["normalization"]["atlas_registration_input"] == "none"
    assert plan.workflow_config["resampling"]["policy"] == "sectioned-stack"
    assert plan.pre_resampling_plan.policy == "sectioned-stack"
    assert plan.pre_resampling_plan.target_locked_axes == [0]


def test_sectioned_stack_target_pre_resampling_uses_in_plane_resolution_only():
    axes = _axes_with_spacing([16.0, 16.0, 16.0])
    resolution_um = workflow_module._sectioned_stack_target_resolution_um(axes, 200.0)

    down = workflow_module._compute_target_downsampling(
        axes,
        200.0,
        per_axis_resolution_um=resolution_um,
        locked_axes=frozenset({0}),
    )

    assert down == [1, 12, 12]


def test_sectioned_stack_target_pre_resampling_keeps_slice_axis_locked():
    axes = _axes_with_spacing([10.0, 20.0, 20.0])
    resolution_um = workflow_module._sectioned_stack_target_resolution_um(axes, 200.0)

    down = workflow_module._compute_target_downsampling(
        axes,
        200.0,
        per_axis_resolution_um=resolution_um,
        locked_axes=frozenset({0}),
    )

    assert down == [1, 10, 10]


def test_sectioned_stack_atlas_pre_resampling_downsamples_finer_in_plane_axes():
    target_working_axes = _axes_with_spacing([16.0, 192.0, 192.0])
    atlas_axes = _axes_with_spacing([50.0, 50.0, 50.0])

    atlas_down = workflow_module._compute_atlas_downsampling(
        atlas_axes,
        target_working_axes,
        locked_axes=frozenset({0}),
    )

    assert atlas_down == [1, 4, 4]


def test_sectioned_stack_atlas_pre_resampling_leaves_matching_grid_unchanged():
    target_working_axes = _axes_with_spacing([16.0, 192.0, 192.0])
    atlas_axes = _axes_with_spacing([200.0, 200.0, 200.0])

    atlas_down = workflow_module._compute_atlas_downsampling(
        atlas_axes,
        target_working_axes,
        locked_axes=frozenset({0}),
    )

    assert atlas_down == [1, 1, 1]


def test_sectioned_stack_plan_notes_when_atlas_is_already_coarser():
    target_axes = _axes_with_spacing([16.0, 16.0, 16.0])
    target_working_axes = _axes_with_spacing([16.0, 192.0, 192.0])
    atlas_axes = _axes_with_spacing([400.0, 400.0, 400.0])
    atlas_down = workflow_module._compute_atlas_downsampling(
        atlas_axes,
        target_working_axes,
        locked_axes=frozenset({0}),
    )

    plan = workflow_module._plan_pre_resampling(
        policy="sectioned-stack",
        target_axes=target_axes,
        target_downsampling=[1, 12, 12],
        target_working_axes=target_working_axes,
        atlas_axes=atlas_axes,
        atlas_downsampling=atlas_down,
    )

    assert atlas_down == [1, 1, 1]
    assert any("coarser than or equal to the target working grid" in note for note in plan.notes)


def test_legacy_target_first_target_pre_resampling_preserves_current_all_axis_behavior():
    axes = _axes_with_spacing([10.0, 20.0, 20.0])

    down = workflow_module._compute_target_downsampling(axes, 200.0)

    assert down == [20, 10, 10]


def test_workflow_can_use_legacy_target_first_policy(tmp_path):
    override = _write_override(tmp_path, {"resampling": {"policy": "legacy-target-first"}})

    plan = workflow_module.plan_emlddmm_workflow(
        dataset_root=tmp_path,
        target_source=tmp_path,
        target_source_format="prepared-dir",
        registration_output=tmp_path / "emlddmm",
        emlddmm_config=override,
        backend=FakeBackend(),
        dry_run=True,
    )

    assert plan.pre_resampling_plan.policy == "legacy-target-first"
    assert plan.pre_resampling_plan.target_locked_axes == []
    assert plan.target_downsampling == [40, 100, 100]


def test_workflow_dry_run_writes_plan_and_summary_only(tmp_path):
    backend = FakeBackend()

    result = workflow_module.run_emlddmm_workflow(
        dataset_root=tmp_path,
        target_source=tmp_path,
        target_source_format="prepared-dir",
        registration_output=tmp_path / "emlddmm",
        backend=backend,
        dry_run=True,
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    plan = json.loads(result.plan_path.read_text(encoding="utf-8"))
    assert result.plan_path.exists()
    assert result.summary_path.exists()
    assert result.log_path.exists()
    assert result.provenance_path.exists()
    assert result.reproduce_command_path.exists()
    assert not (tmp_path / "emlddmm" / "self_alignment").exists()
    assert summary["completed_stages"] == []
    assert summary["enabled_stages"] == ["self_alignment"]
    assert summary["backend_name"] == "fake-backend"
    assert summary["schema_version"] == "emlddmm-step5/v1"
    assert summary["log_path"].endswith("registration.log")
    assert summary["provenance_path"].endswith("run_provenance.json")
    assert summary["reproduce_command_path"].endswith("reproduce_step5_command.txt")
    assert "warnings" in summary
    assert summary["pre_resampling_plan"]["policy"] == "sectioned-stack"
    assert summary["pre_resampling_plan"]["target_locked_axes"] == [0]
    assert plan["backend_name"] == "fake-backend"
    assert plan["schema_version"] == "emlddmm-step5/v1"
    assert plan["log_path"].endswith("registration.log")
    assert plan["provenance_path"].endswith("run_provenance.json")
    assert plan["pre_resampling_plan"]["policy"] == "sectioned-stack"
    assert "target_downsampling" in plan
    assert "atlas_downsampling" in plan


def test_workflow_rejects_label_without_atlas(tmp_path):
    with pytest.raises(ValueError, match="--label requires --atlas"):
        workflow_module.run_emlddmm_workflow(
            dataset_root=tmp_path,
            target_source=tmp_path,
            target_source_format="prepared-dir",
            label=tmp_path / "labels.vtk",
            backend=FakeBackend(),
        )


def test_workflow_requires_init_for_atlas_registration(tmp_path):
    with pytest.raises(
        ValueError,
        match="Atlas registration requires --init-affine or both --orientation-from and --orientation-to",
    ):
        workflow_module.run_emlddmm_workflow(
            dataset_root=tmp_path,
            target_source=tmp_path,
            target_source_format="prepared-dir",
            atlas=tmp_path / "atlas.vtk",
            backend=FakeBackend(),
        )


def test_workflow_can_skip_atlas_registration_via_config(tmp_path):
    override = _write_override(
        tmp_path,
        {
            "atlas_registration": {"enabled": False},
            "stage_controls": {"atlas_registration_enabled": False},
        },
    )

    result = workflow_module.run_emlddmm_workflow(
        dataset_root=tmp_path,
        target_source=tmp_path,
        target_source_format="prepared-dir",
        registration_output=tmp_path / "emlddmm",
        atlas=tmp_path / "atlas.vtk",
        emlddmm_config=override,
        backend=FakeBackend(),
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.mode == "atlas_to_target"
    assert result.stage_results["atlas_registration"]["status"] == "skipped"
    assert summary["atlas_registration_enabled"] is False
    assert not (tmp_path / "emlddmm" / "atlas_registration").exists()


def test_workflow_uses_raw_inputs_for_default_atlas_registration(tmp_path):
    backend = FakeBackend()

    workflow_module.run_emlddmm_workflow(
        dataset_root=tmp_path,
        target_source=tmp_path,
        target_source_format="prepared-dir",
        registration_output=tmp_path / "emlddmm",
        atlas=tmp_path / "atlas.vtk",
        orientation_from="PIR",
        orientation_to="RIP",
        backend=backend,
    )

    expected = _expected_target_volume()
    assert np.array_equal(backend.last_multiscale_kwargs["J"], expected)
    assert np.array_equal(
        backend.last_multiscale_kwargs["I"],
        np.ones((1, 3, 2, 2), dtype=np.float32),
    )
    expected_mu_b = float(np.min(expected))
    expected_mu_a = float(np.quantile(expected, 0.999))
    assert backend.last_multiscale_kwargs["muB"] == [[expected_mu_b]]
    assert backend.last_multiscale_kwargs["muA"] == [[expected_mu_a]]


def test_workflow_records_orientation_resolution_and_artifact_manifest(tmp_path):
    result = workflow_module.run_emlddmm_workflow(
        dataset_root=tmp_path,
        target_source=tmp_path,
        target_source_format="prepared-dir",
        registration_output=tmp_path / "emlddmm",
        atlas=tmp_path / "atlas.vtk",
        orientation_from="pir",
        orientation_to="rip",
        backend=FakeBackend(),
    )

    plan = json.loads(result.plan_path.read_text(encoding="utf-8"))
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    artifacts_manifest = json.loads(
        (tmp_path / "emlddmm" / "atlas_registration" / "artifacts.json").read_text(encoding="utf-8")
    )

    assert plan["orientation_resolution"]["mode"] == "orientation"
    assert plan["orientation_resolution"]["orientation_from"] == "PIR"
    assert summary["orientation_resolution"]["orientation_to"] == "RIP"
    assert plan["pre_resampling_plan"]["policy"] == "sectioned-stack"
    assert any(entry["artifact_kind"] == "config" for entry in artifacts_manifest["entries"])


def test_workflow_runs_atlas_registration_and_upsampling(tmp_path, monkeypatch):
    backend = FakeBackend()

    def fake_upsample_between_slices(*args, **kwargs):
        J = np.asarray(args[1], dtype=np.float32)
        return {
            "J_filled": J,
            "J_nearest_bad": J,
            "pairs": [(0, 2)],
            "slices_with_data": [0, 2],
        }

    monkeypatch.setattr(workflow_module, "_upsample_between_slices_impl", fake_upsample_between_slices)

    result = workflow_module.run_emlddmm_workflow(
        dataset_root=tmp_path,
        target_source=tmp_path,
        target_source_format="prepared-dir",
        registration_output=tmp_path / "emlddmm",
        atlas=tmp_path / "atlas.vtk",
        label=tmp_path / "labels.vtk",
        orientation_from="PIR",
        orientation_to="RIP",
        upsample_between_slices=True,
        upsample_mode="img",
        backend=backend,
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.mode == "atlas_to_target"
    assert result.stage_results["atlas_registration"]["status"] == "completed"
    assert result.stage_results["upsampling"]["status"] == "completed"
    assert (tmp_path / "emlddmm" / "atlas_registration" / "transformation_graph_config.json").exists()
    assert (
        tmp_path / "emlddmm" / "atlas_registration" / "transformation_graph_execution_config.json"
    ).exists()
    assert (tmp_path / "emlddmm" / "atlas_registration" / "registration_data.npy").exists()
    assert (tmp_path / "emlddmm" / "upsampling" / "filled_volume.vtk").exists()
    assert (tmp_path / "emlddmm" / "upsampling" / "filled_volume_overview.png").exists()
    assert (
        tmp_path / "emlddmm" / "upsampling" / "nearest_slice_reference_overview.png"
    ).exists()
    assert (tmp_path / "emlddmm" / "run_provenance.json").exists()
    assert (tmp_path / "emlddmm" / "reproduce_step5_command.txt").exists()
    assert summary["upsampling_enabled"] is True
    assert summary["atlas_unit_scale"] == 1000.0
    assert summary["target_unit_scale"] == 1.0
    assert summary["pre_resampling_plan"]["policy"] == "sectioned-stack"
    assert len(summary["stage_timeline"]) == 3

    effective_config = json.loads(
        (tmp_path / "emlddmm" / "atlas_registration" / "effective_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert effective_config["resampling_policy"] == "sectioned-stack"
    assert effective_config["pre_resampling_plan"]["target_locked_axes"] == [0]


def test_workflow_rejects_transformation_graph_without_atlas(tmp_path):
    with pytest.raises(ValueError, match="--run-transformation-graph requires atlas registration"):
        workflow_module.run_emlddmm_workflow(
            dataset_root=tmp_path,
            target_source=tmp_path,
            target_source_format="prepared-dir",
            backend=FakeBackend(),
            run_transformation_graph=True,
        )


def test_workflow_requires_external_transformation_graph_script_when_requested(tmp_path):
    with pytest.raises(FileNotFoundError, match="external emlddmm package"):
        workflow_module.plan_emlddmm_workflow(
            dataset_root=tmp_path,
            target_source=tmp_path,
            target_source_format="prepared-dir",
            registration_output=tmp_path / "emlddmm",
            atlas=tmp_path / "atlas.vtk",
            orientation_from="PIR",
            orientation_to="RIP",
            backend=FakeBackend(),
            run_transformation_graph=True,
            dry_run=True,
        )


def test_workflow_records_explicit_transformation_graph_script(tmp_path):
    script_path = tmp_path / "transformation_graph_v01.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")

    plan = workflow_module.plan_emlddmm_workflow(
        dataset_root=tmp_path,
        target_source=tmp_path,
        target_source_format="prepared-dir",
        registration_output=tmp_path / "emlddmm",
        atlas=tmp_path / "atlas.vtk",
        orientation_from="PIR",
        orientation_to="RIP",
        backend=FakeBackend(),
        transformation_graph_script=script_path,
        dry_run=True,
    )

    assert plan.transformation_graph_script == script_path.resolve()


def test_workflow_records_legacy_output_alias_usage(tmp_path):
    result = workflow_module.run_emlddmm_workflow(
        output_dir=tmp_path,
        target_source=tmp_path,
        target_source_format="prepared-dir",
        registration_output=tmp_path / "emlddmm",
        backend=FakeBackend(),
        dry_run=True,
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    plan = json.loads(result.plan_path.read_text(encoding="utf-8"))
    assert summary["used_legacy_output_alias"] is True
    assert plan["used_legacy_output_alias"] is True


def test_workflow_writes_qc_report_and_relative_image_manifest(tmp_path, monkeypatch):
    backend = FakeBackend()

    def fake_upsample_between_slices(*args, **kwargs):
        J = np.asarray(args[1], dtype=np.float32)
        return {
            "J_filled": J,
            "J_nearest_bad": J,
            "pairs": [(0, 2)],
            "slices_with_data": [0, 2],
        }

    monkeypatch.setattr(workflow_module, "_upsample_between_slices_impl", fake_upsample_between_slices)

    result = workflow_module.run_emlddmm_workflow(
        dataset_root=tmp_path,
        target_source=tmp_path,
        target_source_format="prepared-dir",
        registration_output=tmp_path / "emlddmm",
        atlas=tmp_path / "atlas.vtk",
        orientation_from="PIR",
        orientation_to="RIP",
        upsample_between_slices=True,
        write_qc_report=True,
        backend=backend,
    )

    report_manifest = json.loads((tmp_path / "emlddmm" / "registration_report.json").read_text(encoding="utf-8"))
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    atlas_stage = next(stage for stage in report_manifest["stages"] if stage["name"] == "atlas_registration")
    upsampling_stage = next(stage for stage in report_manifest["stages"] if stage["name"] == "upsampling")

    assert result.report_manifest_path == tmp_path / "emlddmm" / "registration_report.json"
    assert result.report_path == tmp_path / "emlddmm" / "registration_report.html"
    assert result.report_manifest_path.exists()
    assert result.report_path.exists()
    assert result.provenance_path.exists()
    assert report_manifest["log_path"] == "registration.log"
    assert report_manifest["schema_version"] == "emlddmm-step5/v1"
    assert report_manifest["provenance_path"] == "run_provenance.json"
    assert report_manifest["reproduce_command_path"] == "reproduce_step5_command.txt"
    assert atlas_stage["gallery"]["selected_count"] >= 1
    assert any(image["path"].startswith("atlas_registration/") for image in atlas_stage["gallery"]["images"])
    assert upsampling_stage["gallery"]["selected_count"] == 2
    assert summary["report_manifest_path"].endswith("registration_report.json")
    assert summary["report_path"].endswith("registration_report.html")


def test_workflow_registration_log_records_stage_progress(tmp_path):
    result = workflow_module.run_emlddmm_workflow(
        dataset_root=tmp_path,
        target_source=tmp_path,
        target_source_format="prepared-dir",
        registration_output=tmp_path / "emlddmm",
        backend=FakeBackend(),
    )

    log_text = result.log_path.read_text(encoding="utf-8")
    assert "Starting EM-LDDMM workflow planning" in log_text
    assert "Pre-resampling policy=sectioned-stack" in log_text
    assert "Starting stage: self_alignment" in log_text
    assert "Completed stage: self_alignment" in log_text


def test_workflow_writes_provenance_and_replay_metadata(tmp_path):
    result = workflow_module.run_emlddmm_workflow(
        dataset_root=tmp_path,
        target_source=tmp_path,
        target_source_format="prepared-dir",
        registration_output=tmp_path / "emlddmm",
        backend=FakeBackend(),
        dry_run=True,
    )

    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    replay = result.reproduce_command_path.read_text(encoding="utf-8")

    assert provenance["schema_version"] == "emlddmm-step5/v1"
    assert provenance["pipeline"]["version"]
    assert provenance["backend"]["backend_name"] == "fake-backend"
    assert provenance["resolved_cli"]["normalized_command"]
    assert "--dataset-root" in replay
