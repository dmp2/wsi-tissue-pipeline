from __future__ import annotations

import json

from wsi_pipeline.emlddmm_prep import write_emlddmm_dataset_manifest


def _write_sidecar(path, *, data_file, x=20, y=10, dx=2.0, dy=3.0, dz=5.0, z=-10.0):
    payload = {
        "DataFile": data_file,
        "Type": "uint8",
        "Dimension": 4,
        "Sizes": [3, x, y, 1],
        "Endian": "big",
        "Space": "right-inferior-posterior",
        "SpaceDimension": 3,
        "SpaceUnits": ["um", "um", "um"],
        "SpaceDirections": [
            "none",
            [dx, 0.0, 0.0],
            [0.0, dy, 0.0],
            [0.0, 0.0, dz],
        ],
        "SpaceOrigin": [100.0, 200.0, z],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manifest_writer_infers_full_grid_and_missing_z_positions(tmp_path):
    samples_tsv = tmp_path / "samples.tsv"
    samples_tsv.write_text(
        "\n".join(
            [
                "sample_id\tparticipant_id\tspecies\tstatus",
                "slice_0001.tif\tP001\tMacaque\tpresent",
                "slice_0002.tif\tP001\tMacaque\tmissing",
                "slice_0003.tif\tP001\tMacaque\tpresent",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_sidecar(tmp_path / "slice_0001.json", data_file="slice_0001.tif", z=-10.0)
    _write_sidecar(tmp_path / "slice_0003.json", data_file="slice_0003.tif", z=0.0)

    manifest_path = write_emlddmm_dataset_manifest(
        tmp_path,
        ext=".tif",
        dv_um=[2.0, 3.0, 5.0],
        space="right-inferior-posterior",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert manifest["full_grid_count"] == 3
    assert manifest["dense_present_count"] == 2
    assert manifest["dv_um"] == [2.0, 3.0, 5.0]
    assert manifest["z_axis_um"] == [-10.0, -5.0, 0.0]

    entries = manifest["entries"]
    assert [entry["grid_index"] for entry in entries] == [0, 1, 2]
    assert [entry["present_rank"] for entry in entries] == [0, None, 1]
    assert entries[1]["z_position_um"] == -5.0
    assert entries[0]["shape_yx"] == [10, 20]
    assert entries[0]["space_directions_um"] == [
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 5.0],
    ]
