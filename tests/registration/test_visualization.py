from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from wsi_pipeline.registration.visualization import (
    open_registration_neuroglancer_view,
    prepare_registration_neuroglancer_bundle,
    read_vtk_structured_points,
    resolve_registration_visualization_artifacts,
)


def _write_vtk(path: Path, *, value_offset: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    z_size, y_size, x_size = 2, 3, 4
    base = np.arange(z_size * y_size * x_size, dtype=np.float32).reshape(
        z_size,
        y_size,
        x_size,
    )
    scalars = [
        ("data_000(b)", base + value_offset + 3.0),
        ("data_001(g)", base + value_offset + 2.0),
        ("data_002(r)", base + value_offset + 1.0),
    ]
    header = (
        "# vtk DataFile Version 3.0\n"
        "synthetic\n"
        "BINARY\n"
        "DATASET STRUCTURED_POINTS\n"
        f"DIMENSIONS {x_size} {y_size} {z_size}\n"
        "ORIGIN 10.0 20.0 30.0\n"
        "SPACING 2.0 3.0 4.0\n"
        f"POINT_DATA {x_size * y_size * z_size}\n"
    ).encode()
    payload = bytearray(header)
    for name, array in scalars:
        payload.extend(f"SCALARS {name} float\nLOOKUP_TABLE default\n".encode())
        payload.extend(np.asarray(array, dtype=">f4").tobytes(order="C"))
        payload.extend(b"\n")
    path.write_bytes(bytes(payload))
    return path


def _write_manifest(path: Path) -> Path:
    payload = {
        "version": 1,
        "space": "right-inferior-posterior",
        "dv_um": [2.0, 3.0, 4.0],
        "z_axis_um": [30.0, 34.0],
        "full_grid_count": 2,
        "dense_present_count": 1,
        "target_ext": ".tif",
        "subject_dir": str(path.parent),
        "entries": [
            {"status": "present", "grid_index": 0, "sample_id": "slice_0001.tif"},
            {"status": "missing", "grid_index": 1, "sample_id": "slice_0002.tif"},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_plan(registration_output: Path, manifest_path: Path) -> None:
    (registration_output / "resolved_run_plan.json").write_text(
        json.dumps({"manifest_path": str(manifest_path)}),
        encoding="utf-8",
    )


def test_read_vtk_structured_points_reads_emlddmm_channel_arrays(tmp_path):
    vtk_path = _write_vtk(tmp_path / "target_registered.vtk")

    volume = read_vtk_structured_points(vtk_path)

    assert volume.dimensions_xyz == (4, 3, 2)
    assert volume.origin_xyz == (10.0, 20.0, 30.0)
    assert volume.spacing_xyz == (2.0, 3.0, 4.0)
    assert volume.scalar_names == ("data_000(b)", "data_001(g)", "data_002(r)")
    assert volume.data_czyx.shape == (3, 2, 3, 4)


def test_resolve_registration_visualization_artifacts_prefers_upsampled_volume(tmp_path):
    registration_output = tmp_path / "emlddmm"
    registered = _write_vtk(
        registration_output / "self_alignment" / "images" / "target_registered.vtk"
    )
    filled = _write_vtk(registration_output / "upsampling" / "filled_volume.vtk")

    artifacts = resolve_registration_visualization_artifacts(registration_output)

    assert artifacts.aligned_vtk == filled.resolve()
    assert artifacts.aligned_kind == "upsampled_filled_volume"
    assert artifacts.registered_vtk == registered.resolve()


def test_prepare_registration_neuroglancer_bundle_writes_precomputed_layers(tmp_path):
    registration_output = tmp_path / "emlddmm"
    _write_vtk(registration_output / "self_alignment" / "images" / "input_target.vtk")
    _write_vtk(registration_output / "self_alignment" / "images" / "target_registered.vtk")
    _write_vtk(registration_output / "self_alignment" / "images" / "atlas_free_template.vtk")
    manifest_path = _write_manifest(tmp_path / "emlddmm_dataset_manifest.json")
    _write_plan(registration_output, manifest_path)

    bundle = prepare_registration_neuroglancer_bundle(registration_output)

    assert bundle.aligned_precomputed is not None
    assert bundle.aligned_precomputed.exists()
    assert bundle.original_precomputed is not None
    assert bundle.original_precomputed.exists()
    assert bundle.registered_precomputed is not None
    assert bundle.registered_precomputed.exists()
    assert bundle.tissue_mask_precomputed is not None
    assert bundle.tissue_mask_precomputed.exists()
    info = json.loads((bundle.aligned_precomputed / "info").read_text())
    assert info["num_channels"] == 3
    assert info["scales"][0]["size"] == [4, 3, 2]
    assert info["scales"][0]["resolution"] == [2000, 3000, 4000]

    state = json.loads(bundle.state_path.read_text())
    assert [layer["name"] for layer in state["layers"]] == [
        "aligned_volume",
        "original_slices",
        "registered_slices",
        "tissue_mask",
    ]
    assert state["layers"][-1]["type"] == "segmentation"

    chunk_path = bundle.registered_precomputed / "0" / "0-4_0-3_0-2"
    chunk = np.frombuffer(chunk_path.read_bytes(), dtype=np.uint8).reshape(
        (4, 3, 2, 3),
        order="F",
    )
    assert np.any(chunk[:, :, 0, :])
    assert not np.any(chunk[:, :, 1, :])
    assert tuple(chunk[0, 0, 0, :]) == (1, 2, 3)

    mask_info = json.loads((bundle.tissue_mask_precomputed / "info").read_text())
    assert mask_info["type"] == "segmentation"
    assert mask_info["data_type"] == "uint32"


def test_open_registration_neuroglancer_view_reports_missing_dependency(tmp_path, monkeypatch):
    registration_output = tmp_path / "emlddmm"
    _write_vtk(registration_output / "self_alignment" / "images" / "target_registered.vtk")
    bundle = prepare_registration_neuroglancer_bundle(registration_output)
    monkeypatch.setitem(sys.modules, "neuroglancer", None)

    with pytest.raises(ImportError, match="Neuroglancer is not installed"):
        open_registration_neuroglancer_view(bundle)

