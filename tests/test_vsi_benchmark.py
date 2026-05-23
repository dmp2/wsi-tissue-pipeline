from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner


class _FakeETS:
    nlevels = 2
    tile_xsize = 32
    tile_ysize = 32

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def level_shape(self, level):
        return [(128, 128), (32, 32)][int(level)]

    def read_level(self, level):
        shape = self.level_shape(level)
        arr = np.zeros((*shape, 3), dtype=np.uint8)
        arr[..., 0] = 120
        arr[..., 1] = 80
        arr[..., 2] = 40
        return arr

    def get_tile_decoded(self, level, col, row):
        tile = np.zeros((self.tile_ysize, self.tile_xsize, 3), dtype=np.uint8)
        tile[..., 0] = int(col) + 1
        tile[..., 1] = int(row) + 1
        tile[..., 2] = int(level) + 1
        return tile


def _patch_fake_geometry(monkeypatch, tmp_path):
    from wsi_pipeline.pipeline import vsi_benchmark

    vsi_path = tmp_path / "Image_01.vsi"
    vsi_path.touch()
    ets_path = tmp_path / "_Image_01_" / "stack10002" / "frame_t.ets"
    ets_path.parent.mkdir(parents=True)
    ets_path.touch()

    def _fake_segment(image, **kwargs):  # noqa: ARG001
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        return mask, {"component_source": "fake"}

    monkeypatch.setattr(vsi_benchmark, "find_ets_file", lambda path: ets_path)
    monkeypatch.setattr(
        vsi_benchmark,
        "get_vsi_metadata",
        lambda *args, **kwargs: {"physical_pixel_size_um": {"x": 0.25, "y": 0.5}},
    )
    monkeypatch.setattr(vsi_benchmark, "ETSFile", _FakeETS)
    monkeypatch.setattr(vsi_benchmark, "_segment_for_plating", _fake_segment)

    class _FakeArray:
        def __init__(self, *, shape, dtype):
            self.shape = tuple(shape)
            self.data = np.zeros(shape, dtype=dtype)
            self.attrs: dict[str, object] = {}

        def __setitem__(self, key, value):
            self.data[key] = value

    class _FakeGroup:
        def __init__(self):
            self.attrs: dict[str, object] = {}
            self.children: dict[str, _FakeGroup] = {}
            self.arrays: dict[str, _FakeArray] = {}

        def create_group(self, name, overwrite=True):  # noqa: ARG002
            child = _FakeGroup()
            self.children[name] = child
            return child

    def _fake_open_group(store, *, mode="w"):  # noqa: ARG001
        return _FakeGroup()

    def _fake_create_group_array(group, name, **kwargs):
        arr = _FakeArray(shape=kwargs["shape"], dtype=kwargs["dtype"])
        group.arrays[name] = arr
        return arr

    monkeypatch.setattr(vsi_benchmark, "open_group_v2", _fake_open_group)
    monkeypatch.setattr(vsi_benchmark, "create_group_array", _fake_create_group_array)
    return vsi_path


def test_benchmark_codec_mapping():
    from wsi_pipeline.pipeline.vsi_benchmark import _compressor_for_benchmark_codec

    assert _compressor_for_benchmark_codec("none").compressor is None
    zstd = _compressor_for_benchmark_codec("zstd-1-byte-shuffle")
    assert zstd.descriptor["cname"] == "zstd"
    assert zstd.descriptor["clevel"] == 1
    assert zstd.descriptor["shuffle"] == "byte"
    lz4 = _compressor_for_benchmark_codec("lz4-byte-shuffle")
    assert lz4.descriptor["cname"] == "lz4"


def test_benchmark_cli_parses_options(monkeypatch, tmp_path):
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
""",
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def _fake_benchmark(vsi_arg, benchmark_dir_arg, **kwargs):
        calls.append({"vsi": Path(vsi_arg), "benchmark_dir": Path(benchmark_dir_arg), **kwargs})
        Path(benchmark_dir_arg).mkdir(parents=True, exist_ok=True)
        return {
            "decision_rules": {"top_bottleneck_category": "inconclusive"},
            "artifacts": {},
            "rows": [],
        }

    monkeypatch.setattr(pipeline_mod, "run_vsi_transcode_benchmark", _fake_benchmark)

    result = CliRunner().invoke(
        cli.main,
        [
            "benchmark-vsi-transcode",
            "--vsi",
            str(vsi_path),
            "--benchmark-dir",
            str(tmp_path / "bench"),
            "--source-level",
            "3",
            "--segmentation-level",
            "1",
            "--mode",
            "ets-read-only",
            "--codec",
            "none",
            "--max-tissues",
            "1",
            "--max-blocks",
            "2",
            "--warm-cache",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls[0]["vsi"] == vsi_path
    assert calls[0]["source_level"] == "3"
    assert calls[0]["segmentation_level"] == "1"
    assert calls[0]["modes"] == ("ets-read-only",)
    assert calls[0]["codecs"] == ("none",)
    assert calls[0]["max_tissues"] == 1
    assert calls[0]["max_blocks"] == 2
    assert calls[0]["warm_cache"] is True


def test_benchmark_fake_ets_writes_artifacts_and_source_accounting(monkeypatch, tmp_path):
    from wsi_pipeline.config import PipelineConfig, TileConfig
    from wsi_pipeline.pipeline.vsi_benchmark import run_vsi_transcode_benchmark

    vsi_path = _patch_fake_geometry(monkeypatch, tmp_path)
    bench_dir = tmp_path / "bench"
    config = PipelineConfig(tiles=TileConfig(chunk_size=64, pad_multiple=64, extra_margin_px=0))

    result = run_vsi_transcode_benchmark(
        vsi_path,
        bench_dir,
        source_level=0,
        segmentation_level=1,
        output_profile="validation",
        crop_shape_policy="compact_rectangle",
        pipeline_config=config,
        metadata_backend="ets_only",
        modes=[
            "ets-read-only",
            "synthetic-zero-write",
            "materialized-source-crop-write",
        ],
        codecs=["none"],
        max_tissues=1,
        max_blocks=1,
        materialized_read_max_gib=0.0,
    )

    for name in (
        "resolved_config.json",
        "geometry.json",
        "environment.json",
        "benchmark.json",
        "benchmark.csv",
    ):
        assert (bench_dir / name).is_file()

    rows = result["rows"]
    assert [row["mode"] for row in rows] == [
        "ets-read-only",
        "synthetic-zero-write",
        "materialized-source-crop-write",
    ]
    read_row = rows[0]
    assert read_row["source_tile_accounting"]["total_ets_tile_decode_calls"] > 0
    assert read_row["source_tile_accounting"]["unique_ets_source_tiles_touched"] > 0
    assert "ets_tile_decode_s" in read_row["stage_timers"]
    synthetic_row = rows[1]
    assert synthetic_row["artifact_deleted_after_run"] is True
    materialized_row = rows[2]
    assert materialized_row["skipped"] is True
    assert "exceeds cap" in materialized_row["skip_reason"]

    benchmark_json = json.loads((bench_dir / "benchmark.json").read_text(encoding="utf-8"))
    assert benchmark_json["decision_rules"]["top_bottleneck_category"]
    assert benchmark_json["summary"]["max_blocks"] == 1
    assert benchmark_json["artifacts"]["benchmark_csv"] == str(bench_dir / "benchmark.csv")


def test_benchmark_decision_rules_flag_decode_and_alignment():
    from wsi_pipeline.pipeline.vsi_benchmark import _derive_bottleneck

    rows = [
        {
            "mode": "direct-ets-rgb-plus-mask-mips",
            "elapsed_s": 100.0,
            "blocks_per_sec": 1.0,
            "codec": "zstd-5-bitshuffle",
            "source_tile_accounting": {
                "estimated_repeated_decode_factor": 3.0,
                "potential_alignment_win": 2.5,
            },
        },
        {
            "mode": "materialized-source-crop-write",
            "elapsed_s": 40.0,
            "blocks_per_sec": 3.0,
            "codec": "zstd-5-bitshuffle",
            "source_tile_accounting": {
                "estimated_repeated_decode_factor": 0.0,
                "potential_alignment_win": 1.0,
            },
        },
        {
            "mode": "direct-ets-rgb-plus-mask-no-mips",
            "elapsed_s": 95.0,
            "blocks_per_sec": 1.1,
            "codec": "zstd-5-bitshuffle",
            "source_tile_accounting": {
                "estimated_repeated_decode_factor": 3.0,
                "potential_alignment_win": 2.5,
            },
        },
    ]

    decision = _derive_bottleneck(rows)

    assert decision["rules"]["materialized_much_faster_than_direct"] is True
    assert decision["rules"]["repeated_decode_factor_high"] is True
    assert decision["rules"]["potential_alignment_win_high"] is True
    assert "ETS decode/access/redecode" in decision["bottleneck_candidates"]
