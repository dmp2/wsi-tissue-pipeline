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
            "--block-sampling",
            "stratified",
            "--block-random-seed",
            "17",
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
    assert calls[0]["block_sampling"] == "stratified"
    assert calls[0]["block_random_seed"] == 17
    assert calls[0]["warm_cache"] is True
    assert calls[0]["config_source"] == str(config_path)


def _benchmark_block(idx: int, mask_fraction: float, *, requires_source: bool = True):
    from wsi_pipeline.pipeline.vsi_benchmark import (
        BenchmarkBlock,
        _block_strata_for_mask_fraction,
    )

    coarse, detailed = _block_strata_for_mask_fraction(
        requires_source=requires_source,
        mask_fraction=mask_fraction,
    )
    return BenchmarkBlock(
        tissue_index=0,
        block_index=idx,
        y0=idx * 64,
        y1=idx * 64 + 64,
        x0=0,
        x1=64,
        source_y0=idx * 64,
        source_y1=idx * 64 + 64,
        source_x0=0,
        source_x1=64,
        valid_y0=idx * 64 if requires_source else 0,
        valid_y1=idx * 64 + 64 if requires_source else 0,
        valid_x0=0,
        valid_x1=64 if requires_source else 0,
        source_tiles=((0, idx, 0),) if requires_source else (),
        requires_source=requires_source,
        skipped_before_read=not requires_source,
        ideal_source_tile_count=1,
        mask_fraction=mask_fraction,
        coarse_stratum=coarse,
        detailed_stratum=detailed,
    )


def test_block_sampling_stratifies_and_reports_positive_bands():
    from wsi_pipeline.pipeline.vsi_benchmark import _select_blocks_for_sampling

    blocks = [
        _benchmark_block(0, 0.0, requires_source=False),
        _benchmark_block(1, 0.0),
        _benchmark_block(2, 0.01),
        _benchmark_block(3, 0.20),
        _benchmark_block(4, 0.80),
    ]

    selected, summary = _select_blocks_for_sampling(
        blocks,
        max_blocks=4,
        block_sampling="stratified",
        block_random_seed=11,
    )

    assert len(selected) == 4
    assert summary["requested_coarse_counts"] == {
        "padding": 1,
        "background": 1,
        "mixed": 1,
        "tissue": 1,
    }
    assert summary["candidate_counts"]["detailed"]["positive_any"] == 3
    assert summary["candidate_counts"]["detailed"]["low_positive"] == 1
    assert summary["candidate_counts"]["detailed"]["moderate_positive"] == 1
    assert summary["candidate_counts"]["detailed"]["high_positive"] == 1


def test_tissue_sampling_prioritizes_positive_blocks():
    from wsi_pipeline.pipeline.vsi_benchmark import _select_blocks_for_sampling

    blocks = [
        _benchmark_block(0, 0.0, requires_source=False),
        _benchmark_block(1, 0.0),
        _benchmark_block(2, 0.01),
        _benchmark_block(3, 0.20),
        _benchmark_block(4, 0.80),
    ]

    selected, summary = _select_blocks_for_sampling(
        blocks,
        max_blocks=2,
        block_sampling="tissue",
        block_random_seed=0,
    )

    assert [block.block_index for block in selected] == [3, 4]
    assert summary["selected_counts"]["detailed"]["positive_any"] == 2


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
        block_sampling="stratified",
        block_random_seed=5,
        config_source="unit-test-config",
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
    assert benchmark_json["summary"]["block_sampling"] == "stratified"
    assert benchmark_json["summary"]["effective_extra_margin_px"] == 0
    assert benchmark_json["summary"]["config_source"] == "unit-test-config"
    assert benchmark_json["artifacts"]["benchmark_csv"] == str(bench_dir / "benchmark.csv")
    assert "sampling_summary" in benchmark_json["rows"][0]
    assert "s0_rgb_write_compress_s" in benchmark_json["rows"][1]["stage_timers"]


def test_benchmark_native_mode_records_alignment_and_write_metrics(monkeypatch, tmp_path):
    from wsi_pipeline.config import PipelineConfig, TileConfig
    from wsi_pipeline.pipeline import vsi_benchmark
    from wsi_pipeline.pipeline.vsi_benchmark import run_vsi_transcode_benchmark

    vsi_path = _patch_fake_geometry(monkeypatch, tmp_path)
    calls: list[dict[str, object]] = []

    def _fake_native_writer(**kwargs):
        calls.append(kwargs)
        Path(kwargs["out_dir"]).mkdir(parents=True, exist_ok=True)
        return {
            "rgb_chunks_expected": 2,
            "rgb_chunks_written": 2,
            "rgb_chunks_skipped": 0,
            "mask_chunks_expected": 2,
            "mask_chunks_written": 2,
            "mask_chunks_skipped": 0,
            "source_tile_decode_calls": 2,
            "unique_source_tiles_touched": 2,
            "rgb_chunk_write_calls": 2,
            "unique_rgb_chunks_written": 2,
            "mask_chunk_write_calls": 2,
            "unique_mask_chunks_written": 2,
            "rgb_write_amplification": 1.0,
            "mask_write_amplification": 1.0,
            "mask_empty_chunks": 1,
            "mask_positive_chunks": 1,
            "rgb_chunks_skipped_before_decode": 1,
            "native_pyramid_levels": [{"output_shape_yx": [64, 64]}],
        }

    monkeypatch.setattr(vsi_benchmark, "write_native_ets_tissue_pyramid_ome", _fake_native_writer)
    config = PipelineConfig(tiles=TileConfig(chunk_size=64, pad_multiple=64, extra_margin_px=0))

    result = run_vsi_transcode_benchmark(
        vsi_path,
        tmp_path / "bench_native",
        source_level=0,
        segmentation_level=1,
        output_profile="validation",
        crop_shape_policy="compact_rectangle",
        pipeline_config=config,
        metadata_backend="ets_only",
        modes=["native-source-pyramid-rgb-plus-mask-mips-aligned"],
        codecs=["none"],
        max_tissues=1,
        max_blocks=0,
    )

    row = result["rows"][0]
    assert calls[0]["source_tile_aligned_canvas"] is True
    assert calls[0]["max_chunks_per_level"] == 0
    assert row["source_tile_aligned_canvas"] is True
    assert row["native_writer_metrics"]["rgb_write_amplification"] == 1.0
    assert row["native_writer_metrics"]["mask_write_amplification"] == 1.0
    assert row["rgb_write_amplification"] == 1.0
    assert row["mask_write_amplification"] == 1.0
    assert row["primary_rgb_mode"] == "masked_rgb"
    assert row["mask_empty_chunks"] == 1
    assert row["mask_positive_chunks"] == 1
    assert row["rgb_chunks_skipped_before_decode"] == 1
    assert row["source_tile_accounting"]["output_chunks_skipped_before_read"] == 1


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


def test_benchmark_decision_rules_label_pyramid_write_path():
    from wsi_pipeline.pipeline.vsi_benchmark import _derive_bottleneck

    rows = [
        {
            "mode": "direct-ets-rgb-plus-mask-mips",
            "elapsed_s": 10.0,
            "blocks_per_sec": 5.0,
            "codec": "zstd-5-bitshuffle",
            "stage_timers": {
                "rgb_mip_downsample_s": 0.01,
                "mask_mip_downsample_s": 0.02,
                "rgb_mip_write_compress_s": 4.0,
                "mask_mip_write_compress_s": 0.5,
            },
            "source_tile_accounting": {
                "estimated_repeated_decode_factor": 1.0,
                "potential_alignment_win": 1.0,
            },
        },
        {
            "mode": "direct-ets-rgb-plus-mask-no-mips",
            "elapsed_s": 2.0,
            "blocks_per_sec": 25.0,
            "codec": "zstd-5-bitshuffle",
            "stage_timers": {},
            "source_tile_accounting": {
                "estimated_repeated_decode_factor": 1.0,
                "potential_alignment_win": 1.0,
            },
        },
    ]

    decision = _derive_bottleneck(rows)

    assert decision["top_bottleneck_category"] == "pyramid write path"
    assert decision["rules"]["pyramid_write_path_dominates_downsample"] is True
    assert "mip generation" not in decision["bottleneck_candidates"]
