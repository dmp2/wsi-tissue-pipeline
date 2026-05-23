from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


def _write_tile(path: Path, *, width: int, height: int, value: int) -> None:
    arr = np.full((height, width, 3), value, dtype=np.uint8)
    arr[:, :, 1] = np.clip(value + 20, 0, 255)
    arr[:, :, 2] = np.clip(value + 40, 0, 255)
    Image.fromarray(arr).save(path)


def _write_manifest(input_dir: Path, records: list[dict[str, object]]) -> Path:
    manifest_path = input_dir / "tile_manifest.json"
    payload = {
        "version": 1,
        "input_dir": str(input_dir.resolve()),
        "generated_by": "tests.test_qc_grid",
        "records": records,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _write_tissue_ome_zarr(path: Path, *, value: int, tissue_index: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "tissue_manifest.json").write_text(
        json.dumps(
            {
                "role": "derivative",
                "derivative_type": "tissue_crop_ome_zarr",
                "source_vsi": "/data/source.vsi",
                "source_ets": "/data/source.ets",
                "source_ome_zarr": "/data/source.ome.zarr",
                "source_level": 0,
                "segmentation_level": 7,
                "tissue_index": tissue_index,
                "crop_bounds_source_level": [0, 0, 20, 16],
                "crop_bounds_segmentation_level": [0, 0, 10, 8],
                "physical_pixel_size": {"x": 0.25, "y": 0.5, "unit": "micrometer"},
                "operations": [
                    "read_ets_pyramid",
                    "segment_lowres",
                    "extract_tissue",
                    "write_ome_zarr",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_process_wsi_metadata_includes_tile_records(sample_image: Path, temp_dir: Path):
    from wsi_pipeline.config import PipelineConfig
    from wsi_pipeline.wsi_processing import process_wsi

    config = PipelineConfig()
    config.output.format = "tiff"
    config.output.generate_qc = False
    config.mlflow.enabled = False

    output_dir = temp_dir / "output"
    result = process_wsi(sample_image, output_dir, config=config)

    assert "tile_records" in result
    assert len(result["tile_records"]) == len(result["output_paths"])
    assert len(result["tile_records"]) >= 1

    first_record = result["tile_records"][0]
    assert first_record["source_image"] == sample_image.name
    assert first_record["tile_index_on_source"] == 0
    assert Path(first_record["path"]).exists()
    assert first_record["width"] > 0
    assert first_record["height"] > 0
    assert "component_qc" in first_record
    assert "component_qc" in result
    assert result["component_qc"]["mode"] == "annotate"

    metadata_path = output_dir / f"{sample_image.stem}_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["tile_records"] == result["tile_records"]


def test_rename_outputs_writes_manifest_and_excludes_deleted(temp_dir: Path):
    from wsi_pipeline.tiles.naming import rename_outputs_by_overall_index

    output_dir = temp_dir / "tiles"
    output_dir.mkdir()

    tile_a = output_dir / "specimen_A_11_01.tif"
    tile_b = output_dir / "specimen_A_12_02.tif"
    deleted_tile = output_dir / "specimen_A_13_03.tif"
    _write_tile(tile_a, width=24, height=18, value=80)
    _write_tile(tile_b, width=30, height=22, value=120)

    metadata = {
        "input_path": "/data/input/specimen_A.jpg",
        "output_dir": str(output_dir),
        "n_tiles": 3,
        "output_paths": [str(tile_a), str(tile_b), str(deleted_tile)],
        "tile_records": [
            {
                "source_image": "specimen_A.jpg",
                "tile_index_on_source": 0,
                "path": str(tile_a),
                "width": 24,
                "height": 18,
            },
            {
                "source_image": "specimen_A.jpg",
                "tile_index_on_source": 1,
                "path": str(tile_b),
                "width": 30,
                "height": 22,
                "component_qc": {
                    "component_area_px": 120,
                    "artifact_likely": True,
                    "artifact_reason": "thin_low_stain_component",
                },
            },
            {
                "source_image": "specimen_A.jpg",
                "tile_index_on_source": 2,
                "path": str(deleted_tile),
                "width": 12,
                "height": 10,
            },
        ],
    }
    (output_dir / "specimen_A_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    renames = rename_outputs_by_overall_index(
        output_dir,
        pattern="*.jpg",
        spacing=9,
        pad=4,
        start=1,
        dry_run=False,
    )

    assert [new.name for _old, new in renames] == [
        "specimen_A_11_01_0001.tif",
        "specimen_A_12_02_0011.tif",
    ]

    manifest = json.loads((output_dir / "tile_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generated_by"] == "wsi_pipeline.tiles.naming.rename_outputs_by_overall_index"
    assert [record["filename"] for record in manifest["records"]] == [
        "specimen_A_11_01_0001.tif",
        "specimen_A_12_02_0011.tif",
    ]
    assert [record["overall_label"] for record in manifest["records"]] == ["0001", "0011"]
    assert [record["tile_index_on_source"] for record in manifest["records"]] == [0, 1]
    assert [record["source_image"] for record in manifest["records"]] == [
        "specimen_A.jpg",
        "specimen_A.jpg",
    ]
    assert manifest["records"][0]["component_qc"] is None
    assert manifest["records"][1]["component_qc"]["artifact_likely"] is True


def test_run_qc_workflow_consumes_manifest_and_writes_artifacts(temp_dir: Path):
    from wsi_pipeline.qc_grid import run_qc_workflow

    input_dir = temp_dir / "input_tiles"
    input_dir.mkdir()
    output_dir = input_dir / "_qc_grids"

    tile_a = input_dir / "alpha_piece.tif"
    tile_b = input_dir / "beta_piece.tif"
    tile_c = input_dir / "gamma_piece.tif"
    _write_tile(tile_a, width=18, height=12, value=60)
    _write_tile(tile_b, width=20, height=15, value=110)
    _write_tile(tile_c, width=22, height=16, value=160)

    manifest_path = _write_manifest(
        input_dir,
        [
            {
                "relative_path": tile_a.name,
                "filename": tile_a.name,
                "source_image": "slide_b",
                "tile_index_on_source": 1,
                "overall_index": 11,
                "overall_label": "0011",
                "width": 18,
                "height": 12,
            },
            {
                "relative_path": tile_b.name,
                "filename": tile_b.name,
                "source_image": "slide_a",
                "tile_index_on_source": 0,
                "overall_index": 1,
                "overall_label": "0001",
                "width": 20,
                "height": 15,
                "component_qc": {
                    "component_area_px": 42,
                    "aspect_ratio": 12.0,
                    "artifact_likely": True,
                    "artifact_reason": "thin_low_stain_component",
                },
            },
            {
                "relative_path": tile_c.name,
                "filename": tile_c.name,
                "source_image": "slide_b",
                "tile_index_on_source": 2,
                "overall_index": 21,
                "overall_label": "0021",
                "width": 22,
                "height": 16,
            },
        ],
    )

    result = run_qc_workflow(input_dir, output_dir)

    assert len(result.records) == 3
    assert result.artifacts.records_manifest == manifest_path
    assert result.artifacts.master_contact_sheet is not None
    assert result.artifacts.master_contact_sheet.exists()
    assert [path.name for path in result.artifacts.per_slide_grids] == [
        "slide_01_grid.png",
        "slide_02_grid.png",
    ]
    assert result.artifacts.stats_csv is not None
    assert result.artifacts.stats_csv.exists()
    assert result.records[1].artifact_likely
    assert result.records[1].artifact_reason == "thin_low_stain_component"

    stats_df = pd.read_csv(result.artifacts.stats_csv)
    assert list(stats_df["source_image"]) == ["slide_b", "slide_a", "slide_b"]
    assert set(stats_df["filename"]) == {tile_a.name, tile_b.name, tile_c.name}
    beta_stats = stats_df.loc[stats_df["filename"] == tile_b.name].iloc[0]
    assert bool(beta_stats["artifact_likely"])
    assert beta_stats["artifact_reason"] == "thin_low_stain_component"
    assert beta_stats["component_area_px"] == 42


def test_sorted_groups_follow_overall_index_instead_of_lexicographic_source_name():
    from wsi_pipeline.qc_grid import QCRecord, _sorted_groups

    records = [
        QCRecord(
            relative_path="slide_10_piece.tif",
            filename="slide_10_piece.tif",
            source_image="slide_10",
            tile_index_on_source=0,
            overall_index=11,
            overall_label="0011",
            width=20,
            height=20,
        ),
        QCRecord(
            relative_path="slide_2_piece.tif",
            filename="slide_2_piece.tif",
            source_image="slide_2",
            tile_index_on_source=0,
            overall_index=1,
            overall_label="0001",
            width=20,
            height=20,
        ),
    ]

    groups = _sorted_groups(records)

    assert [source_image for _ordinal, source_image, _records in groups] == [
        "slide_2",
        "slide_10",
    ]


def test_run_qc_workflow_falls_back_to_legacy_filename_parsing(temp_dir: Path):
    from wsi_pipeline.qc_grid import run_qc_workflow

    input_dir = temp_dir / "legacy_tiles"
    input_dir.mkdir()
    output_dir = input_dir / "_qc_grids"

    _write_tile(input_dir / "legacy_11_01_0001.tif", width=20, height=14, value=75)
    _write_tile(input_dir / "legacy_12_02_0011.tif", width=24, height=18, value=125)

    result = run_qc_workflow(input_dir, output_dir)

    assert len(result.records) == 2
    assert result.artifacts.records_manifest is None
    assert result.artifacts.master_contact_sheet is not None
    assert result.artifacts.master_contact_sheet.exists()
    assert len(result.artifacts.per_slide_grids) == 2
    assert result.artifacts.stats_csv is not None
    assert result.artifacts.stats_csv.exists()


def test_run_qc_workflow_falls_back_to_processing_metadata(temp_dir: Path):
    from wsi_pipeline.qc_grid import run_qc_workflow

    input_dir = temp_dir / "process_outputs"
    input_dir.mkdir()
    output_dir = input_dir / "_qc_grids"

    tile_a = input_dir / "level_7_Image_00_00.tif"
    tile_b = input_dir / "level_7_Image_00_01.tif"
    tile_c = input_dir / "level_7_Image_01_00.tif"
    _write_tile(tile_a, width=18, height=12, value=60)
    _write_tile(tile_b, width=20, height=15, value=110)
    _write_tile(tile_c, width=22, height=16, value=160)

    (input_dir / "level_7_Image_00_metadata.json").write_text(
        json.dumps(
            {
                "input_path": "/data/input/level_7_Image_00.png",
                "tile_records": [
                    {
                        "source_image": "level_7_Image_00.png",
                        "tile_index_on_source": 0,
                        "path": str(tile_a),
                        "width": 18,
                        "height": 12,
                    },
                    {
                        "source_image": "level_7_Image_00.png",
                        "tile_index_on_source": 1,
                        "path": str(tile_b),
                        "width": 20,
                        "height": 15,
                        "component_qc": {
                            "component_area_px": 56,
                            "artifact_likely": True,
                            "artifact_reason": "edge_strip",
                        },
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (input_dir / "level_7_Image_01_metadata.json").write_text(
        json.dumps(
            {
                "input_path": "/data/input/level_7_Image_01.png",
                "tile_records": [
                    {
                        "source_image": "level_7_Image_01.png",
                        "tile_index_on_source": 0,
                        "path": str(tile_c),
                        "width": 22,
                        "height": 16,
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = run_qc_workflow(input_dir, output_dir)

    assert len(result.records) == 3
    assert result.artifacts.records_manifest is None
    assert [record.filename for record in result.records] == [
        "level_7_Image_00_00.tif",
        "level_7_Image_00_01.tif",
        "level_7_Image_01_00.tif",
    ]
    assert [record.source_image for record in result.records] == [
        "level_7_Image_00.png",
        "level_7_Image_00.png",
        "level_7_Image_01.png",
    ]
    assert result.records[1].artifact_likely
    assert result.records[1].artifact_reason == "edge_strip"
    assert result.artifacts.master_contact_sheet is not None
    assert result.artifacts.master_contact_sheet.exists()
    assert len(result.artifacts.per_slide_grids) == 2


def test_run_qc_workflow_consumes_tissue_ome_zarr_manifests(monkeypatch, temp_dir: Path):
    import wsi_pipeline.qc_grid as qc_grid

    input_dir = temp_dir / "per_tissue_ngff"
    input_dir.mkdir()
    output_dir = input_dir / "_qc_grids"

    _write_tissue_ome_zarr(input_dir / "source_tissue_00.ome.zarr", value=80, tissue_index=0)
    _write_tissue_ome_zarr(input_dir / "source_tissue_01.ome.zarr", value=130, tissue_index=1)

    class _FakeRoot:
        attrs = {
            "multiscales": [
                {
                    "datasets": [
                        {"path": "s0", "coordinateTransformations": []},
                        {"path": "s1", "coordinateTransformations": []},
                    ]
                }
            ]
        }

    class _FakeArray:
        def __init__(self, path: str):
            value = 80 if "source_tissue_00" in path else 130
            self.data = np.full((3, 16, 20), value, dtype=np.uint8)
            if path.endswith("s1"):
                self.data = self.data[:, ::2, ::2]

        @property
        def shape(self):
            return self.data.shape

        def __getitem__(self, item):
            return self.data[item]

    monkeypatch.setattr(qc_grid.zarr, "open_group", lambda *args, **kwargs: _FakeRoot())
    monkeypatch.setattr(qc_grid.zarr, "open_array", lambda path, **kwargs: _FakeArray(str(path)))

    result = qc_grid.run_qc_workflow(input_dir, output_dir)

    assert len(result.records) == 2
    assert result.artifacts.records_manifest is None
    assert [record.filename for record in result.records] == [
        "source_tissue_00.ome.zarr",
        "source_tissue_01.ome.zarr",
    ]
    assert [record.tile_index_on_source for record in result.records] == [0, 1]
    assert [record.width for record in result.records] == [20, 20]
    assert [record.height for record in result.records] == [16, 16]
    assert result.artifacts.master_contact_sheet is not None
    assert result.artifacts.master_contact_sheet.exists()
    assert result.artifacts.stats_csv is not None
    stats_df = pd.read_csv(result.artifacts.stats_csv)
    assert set(stats_df["filename"]) == {
        "source_tissue_00.ome.zarr",
        "source_tissue_01.ome.zarr",
    }


def test_ome_zarr_qc_uses_axes_and_s0_for_small_stats(monkeypatch, temp_dir: Path):
    import wsi_pipeline.qc_grid as qc_grid

    input_dir = temp_dir / "per_tissue_ngff"
    input_dir.mkdir()
    output_dir = input_dir / "_qc_grids"

    _write_tissue_ome_zarr(input_dir / "source_tissue_00.ome.zarr", value=0, tissue_index=0)

    class _FakeRoot:
        attrs = {
            "multiscales": [
                {
                    "axes": [
                        {"name": "c", "type": "channel"},
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": [
                        {"path": "s0", "coordinateTransformations": []},
                        {"path": "s1", "coordinateTransformations": []},
                    ],
                }
            ]
        }

    class _FakeArray:
        def __init__(self, path: str):
            if path.endswith("s0"):
                self.data = np.zeros((3, 8, 10), dtype=np.uint8)
                self.data[0, :, :] = 10
                self.data[1, :, :] = 20
                self.data[2, :, :] = 30
            else:
                self.data = np.full((3, 4, 5), 99, dtype=np.uint8)

        @property
        def shape(self):
            return self.data.shape

        def __getitem__(self, item):
            return self.data[item]

    monkeypatch.setattr(qc_grid.zarr, "open_group", lambda *args, **kwargs: _FakeRoot())
    monkeypatch.setattr(qc_grid.zarr, "open_array", lambda path, **kwargs: _FakeArray(str(path)))

    result = qc_grid.run_qc_workflow(input_dir, output_dir)

    assert result.artifacts.stats_csv is not None
    stats_df = pd.read_csv(result.artifacts.stats_csv)
    row = stats_df.iloc[0]
    assert row["ngff_dataset_path"] == "s0"
    assert row["ngff_axes"] == "c,y,x"
    assert row["ngff_raw_shape"] == "3x8x10"
    assert row["image_mean_intensity"] == 20.0
    assert row["image_mean_red"] == 10.0
    assert row["image_mean_green"] == 20.0
    assert row["image_mean_blue"] == 30.0
    assert not bool(row["channels_nearly_identical"])


def test_build_qc_grids_uses_pil_default_even_when_torch_available(
    monkeypatch,
    temp_dir: Path,
):
    import wsi_pipeline.qc_grid as qc_grid

    input_dir = temp_dir / "manifest_tiles"
    input_dir.mkdir()
    output_dir = input_dir / "_qc_grids"

    tile_a = input_dir / "one.tif"
    tile_b = input_dir / "two.tif"
    _write_tile(tile_a, width=18, height=12, value=90)
    _write_tile(tile_b, width=18, height=12, value=130)
    _write_manifest(
        input_dir,
        [
            {
                "relative_path": tile_a.name,
                "filename": tile_a.name,
                "source_image": "slide_a",
                "tile_index_on_source": 0,
                "overall_index": 1,
                "overall_label": "0001",
                "width": 18,
                "height": 12,
            },
            {
                "relative_path": tile_b.name,
                "filename": tile_b.name,
                "source_image": "slide_a",
                "tile_index_on_source": 1,
                "overall_index": 11,
                "overall_label": "0011",
                "width": 18,
                "height": 12,
            },
        ],
    )

    monkeypatch.setattr(qc_grid, "TORCH_AVAILABLE", True)

    def _unexpected_torch(*args, **kwargs):
        raise AssertionError("torch backend should not be used by default")

    monkeypatch.setattr(qc_grid, "create_grid_torch", _unexpected_torch)

    paths = qc_grid.build_qc_grids(input_dir, output_dir)

    assert [path.name for path in paths] == [
        "slide_01_grid.png",
        "master_contact_sheet.png",
    ]


def test_ome_zarr_qc_masked_rgb_composes_from_unmasked_primary(temp_dir: Path):
    import wsi_pipeline.qc_grid as qc_grid

    tissue_dir = temp_dir / "tissue_00.ome.zarr"
    tissue_dir.mkdir()
    raw = np.full((4, 4, 3), 120, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:, :2] = True

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        qc_grid,
        "_load_ngff_level",
        lambda *args, **kwargs: qc_grid.NGFFLevelSelection(
            dataset_path="s0",
            axes=("c", "y", "x"),
            raw_shape=(3, 4, 4),
            array_yxc=raw,
        ),
    )
    monkeypatch.setattr(qc_grid, "_ngff_primary_rgb_mode", lambda path: "unmasked_rgb")
    monkeypatch.setattr(qc_grid, "_load_ngff_mask_for_dataset", lambda path, dataset_path: mask)

    try:
        img = qc_grid.load_thumbnail(tissue_dir, size=8, qc_display_mode="masked_rgb")
        arr = np.asarray(img)
    finally:
        monkeypatch.undo()

    assert arr[:, :2].max() == 120
    assert arr[:, 2:].max() == 0


def test_ome_zarr_qc_raw_rgb_fails_for_masked_primary(temp_dir: Path):
    import wsi_pipeline.qc_grid as qc_grid

    tissue_dir = temp_dir / "masked.ome.zarr"
    tissue_dir.mkdir()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        qc_grid,
        "_load_ngff_level",
        lambda *args, **kwargs: qc_grid.NGFFLevelSelection(
            dataset_path="s0",
            axes=("c", "y", "x"),
            raw_shape=(3, 2, 2),
            array_yxc=np.zeros((2, 2, 3), dtype=np.uint8),
        ),
    )
    monkeypatch.setattr(qc_grid, "_ngff_primary_rgb_mode", lambda path: "masked_rgb")
    monkeypatch.setattr(qc_grid, "_load_ngff_mask_for_dataset", lambda path, dataset_path: None)

    try:
        with pytest.raises(ValueError, match="Raw RGB is not stored"):
            qc_grid.load_thumbnail(tissue_dir, size=8, qc_display_mode="raw_rgb")
    finally:
        monkeypatch.undo()
