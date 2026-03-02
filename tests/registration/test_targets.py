from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

import wsi_pipeline.registration.targets as targets_module


def _manifest_payload(subject_dir: str) -> dict:
    return {
        "version": 1,
        "space": "right-inferior-posterior",
        "dv_um": [1.0, 2.0, 5.0],
        "z_axis_um": [0.0, 5.0, 10.0],
        "full_grid_count": 3,
        "dense_present_count": 2,
        "target_ext": ".tif",
        "subject_dir": subject_dir,
        "entries": [
            {
                "sample_id": "slice_0001.tif",
                "status": "present",
                "grid_index": 0,
                "present_rank": 0,
                "overall_index": 1,
                "image_filename": "slice_0001.tif",
                "sidecar_filename": "slice_0001.json",
                "z_position_um": 0.0,
                "shape_yx": [2, 2],
                "space_origin_um": [0.0, 0.0, 0.0],
                "space_directions_um": [
                    [1.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 5.0],
                ],
            },
            {
                "sample_id": "slice_0002.tif",
                "status": "missing",
                "grid_index": 1,
                "present_rank": None,
                "overall_index": 2,
                "image_filename": "slice_0002.tif",
                "sidecar_filename": None,
                "z_position_um": 5.0,
                "shape_yx": None,
                "space_origin_um": None,
                "space_directions_um": None,
            },
            {
                "sample_id": "slice_0003.tif",
                "status": "present",
                "grid_index": 2,
                "present_rank": 1,
                "overall_index": 3,
                "image_filename": "slice_0003.tif",
                "sidecar_filename": "slice_0003.json",
                "z_position_um": 10.0,
                "shape_yx": [2, 2],
                "space_origin_um": [0.0, 0.0, 10.0],
                "space_directions_um": [
                    [1.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 5.0],
                ],
            },
        ],
    }


def test_load_precomputed_target_reconstructs_missing_planes(tmp_path, monkeypatch):
    precomputed_dir = tmp_path / "precomputed"
    precomputed_dir.mkdir()
    (precomputed_dir / "info").write_text(json.dumps({"num_channels": 1}), encoding="utf-8")
    manifest = _manifest_payload(str(tmp_path))
    manifest_path = tmp_path / "emlddmm_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dense = np.asarray(
        [
            [
                [[7.0, 7.0], [7.0, 7.0]],
                [[9.0, 9.0], [9.0, 9.0]],
            ]
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(targets_module, "_load_dense_precomputed_scale0", lambda _: dense)

    target = targets_module.load_precomputed_target(precomputed_dir, manifest_path=manifest_path)

    assert target.source_format == "precomputed"
    assert target.J.shape == (1, 3, 2, 2)
    assert np.all(target.J[:, 0] == 7.0)
    assert np.all(target.J[:, 1] == 0.0)
    assert np.all(target.J[:, 2] == 9.0)
    assert np.all(target.W0[1] == 0.0)
    assert np.array_equal(target.present_mask, np.array([True, False, True]))
    assert np.array_equal(target.xJ[0], np.array([0.0, 5.0, 10.0], dtype=np.float32))


def test_prepared_and_precomputed_targets_are_parity_compatible(tmp_path, monkeypatch):
    subject_dir = tmp_path / "prepared"
    subject_dir.mkdir()
    (subject_dir / "samples.tsv").write_text(
        "sample_id\tstatus\toverall_index\n"
        "slice_0001.tif\tpresent\t1\n"
        "slice_0002.tif\tmissing\t2\n"
        "slice_0003.tif\tpresent\t3\n",
        encoding="utf-8",
    )
    manifest = _manifest_payload(str(subject_dir))
    manifest_path = subject_dir / "emlddmm_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dense = np.asarray(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(targets_module, "_load_dense_precomputed_scale0", lambda _: dense)

    xJ = [
        np.array([0.0, 5.0, 10.0], dtype=np.float32),
        np.array([-1.0, 1.0], dtype=np.float32),
        np.array([-0.5, 0.5], dtype=np.float32),
    ]
    prepared_J = np.zeros((1, 3, 2, 2), dtype=np.float32)
    prepared_J[:, 0] = dense[:, 0]
    prepared_J[:, 2] = dense[:, 1]
    prepared_W0 = np.zeros((3, 2, 2), dtype=np.float32)
    prepared_W0[0] = 1.0
    prepared_W0[2] = 1.0
    images = np.concatenate([prepared_J, prepared_W0[None]], axis=0)
    backend = SimpleNamespace(
        read_data=lambda path: (xJ, images, "slice_dataset", ["red", "mask"])
    )

    prepared = targets_module.load_prepared_target(
        subject_dir,
        backend=backend,
        manifest_path=manifest_path,
    )
    precomputed = targets_module.load_precomputed_target(
        tmp_path / "precomputed",
        manifest_path=manifest_path,
    )

    assert np.array_equal(prepared.J, precomputed.J)
    assert np.array_equal(prepared.W0, precomputed.W0)
    assert np.array_equal(prepared.xJ[0], precomputed.xJ[0])
