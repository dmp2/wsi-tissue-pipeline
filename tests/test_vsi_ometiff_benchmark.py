from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from click.testing import CliRunner


class _WriterSpy:
    instances: list[_WriterSpy] = []

    def __init__(self, path: str, *, bigtiff: bool):
        self.path = Path(path)
        self.bigtiff = bool(bigtiff)
        self.calls: list[dict[str, Any]] = []
        _WriterSpy.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def write(self, data, **kwargs):
        tiles = []
        for tile in data:
            tiles.append(np.asarray(tile))
        self.calls.append(
            {
                "tile_count": len(tiles),
                "tile_shapes": [tuple(tile.shape) for tile in tiles],
                "nonzero_tiles": sum(1 for tile in tiles if np.any(tile)),
                "kwargs": dict(kwargs),
            }
        )


def test_native_ometiff_writer_skips_empty_mask_decode_but_writes_zero_tiles(
    monkeypatch,
    tmp_path,
):
    from wsi_pipeline.pipeline import vsi_ometiff

    decode_calls: list[tuple[int, int, int]] = []

    class FakeETS:
        nlevels = 1
        tile_ysize = 4
        tile_xsize = 4

        def __init__(self, path):
            self.path = Path(path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def level_shape(self, level):
            assert int(level) == 0
            return (8, 8)

        def read_level(self, level):  # pragma: no cover - this must not be used by writer
            raise AssertionError("per-tissue OME-TIFF writer must not read full ETS levels")

        def get_tile_decoded(self, level, col, row):
            decode_calls.append((int(level), int(col), int(row)))
            tile = np.full((4, 4, 3), 200, dtype=np.uint8)
            tile[..., 1] = 100
            tile[..., 2] = 50
            return tile

    monkeypatch.setattr(vsi_ometiff, "ETSFile", FakeETS)
    monkeypatch.setattr(vsi_ometiff, "tifffile", SimpleNamespace(TiffWriter=_WriterSpy))
    _WriterSpy.instances.clear()

    record = vsi_ometiff.TissueTileRecord(
        tile=None,
        tissue_index=0,
        label_id=1,
        crop_bounds_source_level=(0, 0, 8, 8),
        crop_bounds_segmentation_level=(0, 0, 4, 4),
        tile_dim=8,
        frame_debug={
            "logical_canvas_source_yx": {"y0": 0, "x0": 0, "y1": 8, "x1": 8},
            "label_crop_seg_yx": {"y0": 0, "x0": 0, "y1": 4, "x1": 4},
        },
    )
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[:2, :2] = 1

    stats = vsi_ometiff.write_native_ets_tissue_pyramid_ometiff(
        ets_path=tmp_path / "fake.ets",
        rgb_path=tmp_path / "tissue_rgb.ome.tif",
        mask_path=tmp_path / "tissue_mask.ome.tif",
        record=record,
        lr_labels=labels,
        source_level=0,
        source_shape_yx=(8, 8),
        source_phys_xy_um=(0.25, 0.5),
        tile_size=4,
        name="tissue",
        compression="deflate",
        segmentation_level=0,
        native_mip_stop_policy="segmentation_level",
        native_mip_stop_level="segmentation_level",
        source_tile_aligned_canvas=False,
    )

    assert stats["rgb_tiles_skipped_before_decode"] == 3
    assert stats["zero_rgb_tiles_written"] == 3
    assert stats["positive_rgb_tiles_written"] == 1
    assert stats["mask_empty_tiles"] == 3
    assert stats["mask_positive_tiles"] == 1
    assert stats["source_tile_decode_calls"] == 1
    assert len(decode_calls) == 1
    assert len(_WriterSpy.instances) == 2
    rgb_writer = _WriterSpy.instances[0]
    mask_writer = _WriterSpy.instances[1]
    assert rgb_writer.bigtiff is True
    assert mask_writer.bigtiff is True
    assert rgb_writer.calls[0]["tile_count"] == 4
    assert rgb_writer.calls[0]["nonzero_tiles"] == 1
    assert rgb_writer.calls[0]["kwargs"]["shape"] == (8, 8, 3)
    assert rgb_writer.calls[0]["kwargs"]["compression"] == "deflate"
    assert mask_writer.calls[0]["kwargs"]["shape"] == (8, 8)


def test_benchmark_vsi_ometiff_cli_parses_sample_options(monkeypatch, tmp_path):
    import wsi_pipeline.pipeline as pipeline_mod
    from wsi_pipeline import cli

    vsi_path = tmp_path / "Image_01.vsi"
    vsi_path.touch()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
segmentation:
  backend: local-entropy
tiles:
  chunk_size: 64
  pad_multiple: 64
output:
  primary_rgb_mode: masked_rgb
  masked_rgb_fill_value: 0
  source_tile_aligned_canvas: true
  native_mip_stop_policy: segmentation_level
  native_mip_stop_level: segmentation_level
""",
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def _fake_benchmark(vsi_arg, output_dir_arg, **kwargs):
        calls.append({"vsi": Path(vsi_arg), "output_dir": Path(output_dir_arg), **kwargs})
        return {
            "summary": {
                "ets_path": str(tmp_path / "_Image_01_" / "stack10002" / "frame_t.ets")
            },
            "totals": {
                "positive_rgb_tiles_written": 1,
                "zero_rgb_tiles_written": 3,
                "rgb_tiles_skipped_before_decode": 3,
            },
            "decision_rules": {
                "interchange_export": "acceptable",
                "production_database_staging": "not_competitive",
            },
        }

    monkeypatch.setattr(pipeline_mod, "run_vsi_ometiff_benchmark", _fake_benchmark)

    result = CliRunner().invoke(
        cli.main,
        [
            "benchmark-vsi-ometiff",
            "--vsi",
            str(vsi_path),
            "--output-dir",
            str(tmp_path / "bench"),
            "--config",
            str(config_path),
            "--source-level",
            "3",
            "--segmentation-level",
            "7",
            "--max-tissues",
            "1",
            "--tissue-index",
            "0",
            "--tissue-selection",
            "heaviest-tiles",
            "--no-synthetic-benchmarks",
            "--qc-masked-background",
            "black",
            "--max-blocks",
            "64",
            "--tile-sampling",
            "stratified",
            "--tile-random-seed",
            "5",
            "--compression",
            "deflate",
            "--estimate-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls[0]["vsi"] == vsi_path
    assert calls[0]["source_level"] == "3"
    assert calls[0]["segmentation_level"] == "7"
    assert calls[0]["max_tissues"] == 1
    assert calls[0]["tissue_index"] == 0
    assert calls[0]["tissue_selection"] == "heaviest-tiles"
    assert calls[0]["no_synthetic_benchmarks"] is True
    assert calls[0]["qc_masked_background"] == "black"
    assert calls[0]["command_used"]
    assert calls[0]["git_commit"]
    assert calls[0]["max_tiles"] == 64
    assert calls[0]["tile_sampling"] == "stratified"
    assert calls[0]["tile_random_seed"] == 5
    assert calls[0]["compression"] == "deflate"
    assert calls[0]["estimate_only"] is True



def test_tissue_selection_helpers_choose_heaviest_and_explicit_index():
    from types import SimpleNamespace

    from wsi_pipeline.pipeline import vsi_ometiff

    tissues = [
        SimpleNamespace(record=SimpleNamespace(tissue_index=0, tile_shape_yx=(10, 10))),
        SimpleNamespace(record=SimpleNamespace(tissue_index=1, tile_shape_yx=(20, 20))),
        SimpleNamespace(record=SimpleNamespace(tissue_index=2, tile_shape_yx=(30, 30))),
    ]
    candidates = [
        {"tissue_index": 0, "estimated_tiff_tile_payload_count": 7, "area_px": 100},
        {"tissue_index": 1, "estimated_tiff_tile_payload_count": 12, "area_px": 400},
        {"tissue_index": 2, "estimated_tiff_tile_payload_count": 9, "area_px": 900},
    ]

    selected, reason = vsi_ometiff._select_tissues(
        tissues,
        candidates,
        tissue_index=None,
        tissue_selection="heaviest-tiles",
        max_tissues=1,
    )
    assert [item.record.tissue_index for item in selected] == [1]
    assert "heaviest-tiles" in reason

    selected, reason = vsi_ometiff._select_tissues(
        tissues,
        candidates,
        tissue_index=0,
        tissue_selection="heaviest-tiles",
        max_tissues=1,
    )
    assert [item.record.tissue_index for item in selected] == [0]
    assert "explicit tissue-index" in reason


def test_tissue_selection_explicit_missing_index_fails():
    from types import SimpleNamespace

    import pytest

    from wsi_pipeline.pipeline import vsi_ometiff

    tissues = [SimpleNamespace(record=SimpleNamespace(tissue_index=0, tile_shape_yx=(10, 10)))]
    with pytest.raises(ValueError, match="Requested tissue_index=2"):
        vsi_ometiff._select_tissues(
            tissues,
            [{"tissue_index": 0, "estimated_tiff_tile_payload_count": 1, "area_px": 100}],
            tissue_index=2,
            tissue_selection="first",
            max_tissues=1,
        )
