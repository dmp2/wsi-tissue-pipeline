from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from wsi_pipeline.registration.sweep import (
    DEFAULT_ATLAS_FREE_EXTRA_KWARGS,
    build_ea2d_override,
    diagnose_self_alignment_run,
    format_ea2d_label,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, delimiter=",")


def test_format_ea2d_label():
    assert format_ea2d_label(5e3) == "eA2d_5e3"
    assert format_ea2d_label(1e4) == "eA2d_1e4"
    assert format_ea2d_label(2e4) == "eA2d_2e4"
    assert format_ea2d_label(5e4) == "eA2d_5e4"
    assert format_ea2d_label(1e5) == "eA2d_1e5"
    assert format_ea2d_label(2e5) == "eA2d_2e5"


def test_build_ea2d_override_keeps_atlas_free_defaults():
    override = build_ea2d_override(5e4, n_steps=10)

    assert override["units"]["desired_resolution_um"] == 200.0
    assert override["self_alignment"]["n_steps"] == 10
    assert override["self_alignment"]["eA2d"] == 5e4
    assert override["self_alignment"]["extra_kwargs"] == DEFAULT_ATLAS_FREE_EXTRA_KWARGS


def test_diagnose_self_alignment_run_transform_metrics(tmp_path):
    manifest_path = _write_json(
        tmp_path / "emlddmm_dataset_manifest.json",
        {
            "entries": [
                {"grid_index": 0, "sample_id": "s0", "status": "present"},
                {"grid_index": 1, "sample_id": "missing_1", "status": "missing"},
                {"grid_index": 2, "sample_id": "s2", "status": "present"},
            ]
        },
    )
    run_dir = tmp_path / "emlddmm" / "sweeps" / "ea2d" / "eA2d_5e4"
    _write_json(
        run_dir / "resolved_run_plan.json",
        {
            "manifest_path": str(manifest_path),
            "pre_resampling_plan": {"target_working_spacing_um": [16.0, 200.0, 100.0]},
            "workflow_config": {
                "units": {"desired_resolution_um": 200.0},
                "self_alignment": {"n_steps": 10, "eA2d": 5e4},
            },
        },
    )

    identity = np.eye(3, dtype=np.float64)
    moved = np.eye(3, dtype=np.float64)
    moved[:2, 2] = [200.0, 100.0]
    angle_rad = math.radians(5.0)
    moved[:2, :2] = [
        [math.cos(angle_rad), -math.sin(angle_rad)],
        [math.sin(angle_rad), math.cos(angle_rad)],
    ]
    _write_matrix(run_dir / "self_alignment" / "transforms" / "A2d_0000.txt", identity)
    _write_matrix(run_dir / "self_alignment" / "transforms" / "A2d_0001.txt", identity)
    _write_matrix(run_dir / "self_alignment" / "transforms" / "A2d_0002.txt", moved)

    diagnostics = diagnose_self_alignment_run(run_dir, top_n=1)

    assert diagnostics["present_count"] == 2
    assert diagnostics["missing_count"] == 1
    assert diagnostics["missing_nonidentity_count"] == 0
    assert diagnostics["present_nonidentity_count"] == 1
    assert diagnostics["translation_um"]["max"] == pytest.approx(math.sqrt(200.0**2 + 100.0**2))
    assert diagnostics["translation_working_pixels"]["max"] == pytest.approx(math.sqrt(2.0))
    assert diagnostics["rotation_abs_deg"]["max"] == pytest.approx(5.0)
    assert diagnostics["top_moved_slices"][0]["transform_index"] == 2
    assert diagnostics["top_moved_slices"][0]["sample_id"] == "s2"
    assert Path(diagnostics["diagnostics_path"]).exists()
    assert (run_dir / "self_alignment" / "diagnostics" / "top_moved_slices.csv").exists()
