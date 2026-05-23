from __future__ import annotations

import inspect
import json
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from click.testing import CliRunner


def test_vsi_to_source_ome_zarr_uses_direct_writer_by_default(monkeypatch, tmp_path):
    from wsi_pipeline.pipeline import vsi_ets

    vsi_path = tmp_path / "sample.vsi"
    vsi_path.touch()
    ets_path = tmp_path / "_sample_" / "stack10002" / "frame_t.ets"
    ets_path.parent.mkdir(parents=True)
    ets_path.touch()

    metadata = {
        "channel_labels": ["r", "g", "b"],
        "physical_pixel_size_um": {"x": 0.25, "y": 0.5, "z": None},
    }
    writer_call: dict[str, object] = {}

    def _fake_direct_writer(ets, out_dir, **kwargs):
        writer_call.update({"ets": Path(ets), "out_dir": Path(out_dir), **kwargs})
        Path(out_dir).mkdir(parents=True)

    monkeypatch.setattr(vsi_ets, "find_ets_file", lambda path: ets_path)
    monkeypatch.setattr(vsi_ets, "get_vsi_metadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(vsi_ets, "write_ets_pyramid_to_ngff_zarr", _fake_direct_writer)
    monkeypatch.setattr(
        vsi_ets,
        "ETSFile",
        lambda *args, **kwargs: pytest.fail("default source writer should not use ETSFile.to_dask"),
    )

    out_path, resolved_ets, resolved_metadata = vsi_ets.vsi_to_source_ome_zarr(
        vsi_path,
        tmp_path / "source.ome.zarr",
        metadata_backend="auto",
        metadata_schema="v0.4",
        chunks_xy=256,
        overwrite=True,
    )

    assert out_path == tmp_path / "source.ome.zarr"
    assert out_path.exists()
    assert resolved_ets == ets_path
    assert resolved_metadata == metadata
    assert writer_call["ets"] == ets_path
    assert writer_call["out_dir"].name == ".source.ome.zarr.incomplete"
    assert writer_call["phys_xy_um"] == (0.25, 0.5)
    assert writer_call["chunks_xy"] == 256
    assert writer_call["ngff_metadata"] == metadata
    assert writer_call["channel_labels"] == ["red", "green", "blue"]
    assert writer_call["channel_colors"] == ["FF0000", "00FF00", "0000FF"]


def test_vsi_to_source_ome_zarr_ngff_zarr_escape_hatch_uses_ets_dask_levels(monkeypatch, tmp_path):
    from wsi_pipeline.pipeline import vsi_ets

    vsi_path = tmp_path / "sample.vsi"
    vsi_path.touch()
    ets_path = tmp_path / "_sample_" / "stack10002" / "frame_t.ets"
    ets_path.parent.mkdir(parents=True)
    ets_path.touch()

    metadata = {
        "channel_labels": ["r", "g", "b"],
        "physical_pixel_size_um": {"x": 0.25, "y": 0.5, "z": None},
    }
    to_dask_calls: list[int] = []

    class _DummyETS:
        nlevels = 3

        def __init__(self, path):
            assert Path(path) == ets_path
            self.closed = False

        def to_dask(self, level):
            to_dask_calls.append(level)
            return np.zeros((8 >> level, 10 >> level, 3), dtype=np.uint8)

        def close(self):
            self.closed = True

    writer_call: dict[str, object] = {}

    def _fake_writer(**kwargs):
        writer_call.update(kwargs)
        Path(kwargs["out_dir"]).mkdir(parents=True)

    monkeypatch.setattr(vsi_ets, "find_ets_file", lambda path: ets_path)
    monkeypatch.setattr(vsi_ets, "get_vsi_metadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(vsi_ets, "ETSFile", _DummyETS)
    monkeypatch.setattr(vsi_ets, "write_ngff_from_mips_ngffzarr", _fake_writer)

    out_path, resolved_ets, resolved_metadata = vsi_ets.vsi_to_source_ome_zarr(
        vsi_path,
        tmp_path / "source.ome.zarr",
        metadata_backend="auto",
        metadata_schema="v0.4",
        chunks_xy=256,
        overwrite=True,
        source_writer="ngff-zarr",
    )

    assert out_path == tmp_path / "source.ome.zarr"
    assert resolved_ets == ets_path
    assert resolved_metadata == metadata
    assert to_dask_calls == [0, 1, 2]
    assert len(writer_call["mips_yxc"]) == 3
    assert writer_call["phys_xy_um"] == (0.25, 0.5)
    assert writer_call["chunks_xy"] == 256
    assert writer_call["ngff_metadata"] == metadata
    assert writer_call["channel_labels"] == ["red", "green", "blue"]
    assert writer_call["channel_colors"] == ["FF0000", "00FF00", "0000FF"]


def test_vsi_to_source_ome_zarr_rejects_incomplete_cached_source(monkeypatch, tmp_path):
    from wsi_pipeline.pipeline import vsi_ets

    vsi_path = tmp_path / "sample.vsi"
    vsi_path.touch()
    ets_path = tmp_path / "_sample_" / "stack10002" / "frame_t.ets"
    ets_path.parent.mkdir(parents=True)
    ets_path.touch()
    output_path = tmp_path / "source.ome.zarr"
    (output_path / "s0").mkdir(parents=True)
    (output_path / "s0" / ".zarray").write_text("{}", encoding="utf-8")

    metadata = {
        "num_levels": 3,
        "physical_pixel_size_um": {"x": 0.25, "y": 0.5, "z": None},
    }

    monkeypatch.setattr(vsi_ets, "find_ets_file", lambda path: ets_path)
    monkeypatch.setattr(vsi_ets, "get_vsi_metadata", lambda *args, **kwargs: metadata)

    with pytest.raises(RuntimeError, match="appears incomplete"):
        vsi_ets.vsi_to_source_ome_zarr(vsi_path, output_path, overwrite=False)


def test_vsi_to_source_ome_zarr_rejects_stale_cached_source_shape(monkeypatch, tmp_path):
    from wsi_pipeline.pipeline import vsi_ets

    vsi_path = tmp_path / "sample.vsi"
    vsi_path.touch()
    ets_path = tmp_path / "_sample_" / "stack10002" / "frame_t.ets"
    ets_path.parent.mkdir(parents=True)
    ets_path.touch()
    output_path = tmp_path / "source.ome.zarr"
    for level, shape in enumerate(([3, 10, 10], [3, 5, 5])):
        array_path = output_path / f"s{level}"
        array_path.mkdir(parents=True)
        (array_path / ".zarray").write_text(f'{{"shape": {shape}}}', encoding="utf-8")

    metadata = {
        "num_levels": 2,
        "channel_count": 3,
        "physical_pixel_size_um": {"x": 0.25, "y": 0.5, "z": None},
    }

    class _DummyETS:
        def __init__(self, path):
            assert Path(path) == ets_path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def level_shape(self, level):
            return [(10, 10), (5, 6)][level]

    monkeypatch.setattr(vsi_ets, "find_ets_file", lambda path: ets_path)
    monkeypatch.setattr(vsi_ets, "get_vsi_metadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(vsi_ets, "ETSFile", _DummyETS)

    with pytest.raises(RuntimeError, match="does not match the ETS pyramid"):
        vsi_ets.vsi_to_source_ome_zarr(vsi_path, output_path, overwrite=False)


def test_estimate_vsi_direct_plating_reports_chunks_and_bytes(monkeypatch, tmp_path):
    from wsi_pipeline.config import TileConfig
    from wsi_pipeline.pipeline import vsi_ets

    vsi_path = tmp_path / "sample.vsi"
    vsi_path.touch()
    ets_path = tmp_path / "_sample_" / "stack10002" / "frame_t.ets"
    ets_path.parent.mkdir(parents=True)
    ets_path.touch()

    class _DummyETS:
        nlevels = 2

        def __init__(self, path):
            assert Path(path) == ets_path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def level_shape(self, level):
            return [(128, 128), (64, 64)][level]

        def read_level(self, level):
            shape = self.level_shape(level)
            return np.zeros((*shape, 3), dtype=np.uint8)

    def _fake_segment(image, **kwargs):  # noqa: ARG001
        mask = np.zeros((64, 64), dtype=bool)
        mask[16:48, 16:48] = True
        return mask, {"component_source": "fake"}

    monkeypatch.setattr(vsi_ets, "find_ets_file", lambda path: ets_path)
    monkeypatch.setattr(vsi_ets, "get_vsi_metadata", lambda *args, **kwargs: {
        "physical_pixel_size_um": {"x": 0.25, "y": 0.5, "z": None}
    })
    monkeypatch.setattr(vsi_ets, "ETSFile", _DummyETS)
    monkeypatch.setattr(vsi_ets, "_segment_for_plating", _fake_segment)

    estimate = vsi_ets.estimate_vsi_direct_plating(
        vsi_path,
        source_level=0,
        segmentation_level=1,
        tile_frame_level="segmentation",
        tile_config=TileConfig(chunk_size=64, pad_multiple=64, extra_margin_px=0),
        min_side_for_mips=64,
    )

    assert estimate["tissue_count"] == 1
    tissue = estimate["tissues"][0]
    assert tissue["source_tile_dim"] == 128
    assert tissue["segmentation_tile_dim"] == 64
    assert tissue["s0_shape_yxc"] == [128, 128, 3]
    assert tissue["s0_chunks"] == 4
    assert tissue["num_mips"] == 2
    assert tissue["mip_shapes_yxc"] == [[128, 128, 3], [64, 64, 3]]
    assert tissue["all_mip_chunks"] == 5
    assert tissue["combined_logical_chunks"] == 5
    assert tissue["rgb_uncompressed_bytes_all_mips"] == 61440
    assert tissue["mask_uncompressed_bytes_all_mips"] == 0
    assert tissue["total_uncompressed_bytes_rgb_plus_mask"] == 61440
    assert tissue["uncompressed_bytes_all_mips"] == 61440
    assert tissue["uncompressed_bytes_estimate"] == 61440
    assert estimate["totals"]["rgb_uncompressed_bytes_all_mips"] == 61440
    assert estimate["totals"]["mask_uncompressed_bytes_all_mips"] == 0
    assert estimate["totals"]["total_uncompressed_bytes_rgb_plus_mask"] == 61440
    assert estimate["totals"]["combined_logical_chunks"] == 5
    assert estimate["totals"]["uncompressed_bytes_all_mips"] == 61440


def test_estimate_vsi_direct_plating_production_compact_rectangle(monkeypatch, tmp_path):
    from wsi_pipeline.config import TileConfig
    from wsi_pipeline.pipeline import vsi_ets

    vsi_path = tmp_path / "sample.vsi"
    vsi_path.touch()
    ets_path = tmp_path / "_sample_" / "stack10002" / "frame_t.ets"
    ets_path.parent.mkdir(parents=True)
    ets_path.touch()

    class _DummyETS:
        nlevels = 2

        def __init__(self, path):
            assert Path(path) == ets_path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def level_shape(self, level):
            return [(256, 512), (64, 128)][level]

        def read_level(self, level):
            return np.zeros((*self.level_shape(level), 3), dtype=np.uint8)

    def _fake_segment(image, **kwargs):  # noqa: ARG001
        mask = np.zeros((64, 128), dtype=bool)
        mask[10:20, 30:55] = True
        return mask, {}

    monkeypatch.setattr(vsi_ets, "find_ets_file", lambda path: ets_path)
    monkeypatch.setattr(vsi_ets, "get_vsi_metadata", lambda *args, **kwargs: {
        "physical_pixel_size_um": {"x": 0.25, "y": 0.5, "z": None}
    })
    monkeypatch.setattr(vsi_ets, "ETSFile", _DummyETS)
    monkeypatch.setattr(vsi_ets, "_segment_for_plating", _fake_segment)

    estimate = vsi_ets.estimate_vsi_direct_plating(
        vsi_path,
        source_level=0,
        segmentation_level=1,
        output_profile="production",
        tile_config=TileConfig(chunk_size=64, pad_multiple=64, extra_margin_px=0),
        min_side_for_mips=64,
    )

    tissue = estimate["tissues"][0]
    assert estimate["output_profile"] == "production"
    assert estimate["crop_shape_policy"] == "compact_rectangle"
    assert estimate["store_tissue_mask"] is True
    assert estimate["sparse_zero_chunks"] is True
    assert tissue["tile_shape_yx"][0] % 64 == 0
    assert tissue["tile_shape_yx"][1] % 64 == 0
    assert tissue["s0_shape_yxc"][:2] == tissue["tile_shape_yx"]
    assert tissue["mask_s0_shape_yx"] == tissue["tile_shape_yx"]
    assert tissue["mask_all_mip_chunks"] > 0
    assert tissue["combined_logical_chunks"] == tissue["all_mip_chunks"] + tissue["mask_all_mip_chunks"]
    assert tissue["rgb_uncompressed_bytes_all_mips"] == tissue["uncompressed_bytes_all_mips"]
    assert tissue["total_uncompressed_bytes_rgb_plus_mask"] == (
        tissue["rgb_uncompressed_bytes_all_mips"] + tissue["mask_uncompressed_bytes_all_mips"]
    )
    assert tissue["uncompressed_bytes_estimate"] > tissue["uncompressed_bytes_all_mips"]
    totals = estimate["totals"]
    assert totals["mask_all_mip_chunks"] > 0
    assert totals["combined_logical_chunks"] == totals["all_mip_chunks"] + totals["mask_all_mip_chunks"]
    assert totals["rgb_uncompressed_bytes_all_mips"] == totals["uncompressed_bytes_all_mips"]
    assert totals["total_uncompressed_bytes_rgb_plus_mask"] == totals["uncompressed_bytes_estimate"]
    assert totals["total_uncompressed_bytes_rgb_plus_mask"] == (
        totals["rgb_uncompressed_bytes_all_mips"] + totals["mask_uncompressed_bytes_all_mips"]
    )
    assert totals["compressed_bytes_sample_estimate"] is None
    assert estimate["compression"]["mode"] == "lossless"


def test_estimate_vsi_plating_cli_reports_rgb_mask_and_total(monkeypatch, tmp_path):
    import wsi_pipeline.pipeline as pipeline_mod
    from wsi_pipeline import cli

    vsi_path = tmp_path / "sample.vsi"
    vsi_path.touch()

    def _fake_estimate(*args, **kwargs):  # noqa: ARG001
        return {
            "tissue_count": 3,
            "store_tissue_mask": True,
            "totals": {
                "all_mip_chunks": 10,
                "mask_all_mip_chunks": 4,
                "combined_logical_chunks": 14,
                "rgb_uncompressed_size_all_mips": "1.00 GiB",
                "mask_uncompressed_size_all_mips": "256.00 MiB",
                "total_uncompressed_size_rgb_plus_mask": "1.25 GiB",
                "warnings": [],
            },
        }

    monkeypatch.setattr(pipeline_mod, "estimate_vsi_direct_plating", _fake_estimate)

    result = CliRunner().invoke(
        cli.main,
        [
            "estimate-vsi-plating",
            "--vsi",
            str(vsi_path),
            "--output-profile",
            "production",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Estimated tissues" in result.output
    assert "RGB pyramid bytes" in result.output
    assert "1.00 GiB" in result.output
    assert "Mask pyramid bytes" in result.output
    assert "256.00 MiB" in result.output
    assert "Total RGB+mask bytes" in result.output
    assert "14 logical chunks" in result.output


def test_process_vsi_directory_with_plating_reuses_source_and_returns_mapping(monkeypatch, tmp_path):
    from wsi_pipeline.pipeline import vsi_ets

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    vsi_a = input_dir / "a.vsi"
    vsi_b = input_dir / "b.vsi"
    vsi_a.touch()
    vsi_b.touch()

    source_calls: list[Path] = []
    plating_calls: list[dict[str, object]] = []

    def _fake_vsi_to_source_ome_zarr(vsi_path, output_path, **kwargs):
        source_calls.append(Path(vsi_path))
        Path(output_path).mkdir(parents=True, exist_ok=True)
        return Path(output_path), Path(f"/ets/{Path(vsi_path).stem}.ets"), {
            "physical_pixel_size_um": {"x": 0.25, "y": 0.5, "z": None}
        }

    def _fake_process_slide_with_plating(source_ome_zarr, per_tissue_dir, **kwargs):
        plating_calls.append(
            {
                "source_ome_zarr": Path(source_ome_zarr),
                "per_tissue_dir": Path(per_tissue_dir),
                **kwargs,
            }
        )
        stem = Path(source_ome_zarr).name.removesuffix(".ome.zarr")
        return [Path(per_tissue_dir) / f"{stem}_tissue_{idx:02d}.ome.zarr" for idx in range(2)]

    monkeypatch.setattr(vsi_ets, "vsi_to_source_ome_zarr", _fake_vsi_to_source_ome_zarr)
    monkeypatch.setattr(vsi_ets, "process_slide_with_plating", _fake_process_slide_with_plating)

    results = vsi_ets.process_vsi_directory_with_plating(
        input_dir,
        tmp_path / "out",
        source_level=0,
        segmentation_level=7,
        materialize_source=True,
    )

    assert "expected_tissues" not in inspect.signature(
        vsi_ets.process_vsi_directory_with_plating
    ).parameters
    assert source_calls == [vsi_a, vsi_b]
    assert set(results) == {str(vsi_a), str(vsi_b)}
    assert all(len(paths) == 2 for paths in results.values())
    assert [call["source_level"] for call in plating_calls] == [0, 0]
    assert [call["segmentation_level"] for call in plating_calls] == [7, 7]
    assert [call["tile_frame_level"] for call in plating_calls] == ["segmentation", "segmentation"]
    assert plating_calls[0]["source_context"]["source_vsi"] == str(vsi_a)
    assert plating_calls[0]["source_context"]["source_ets"] == "/ets/a.ets"


def test_process_vsi_directory_with_plating_uses_direct_ets_by_default(monkeypatch, tmp_path):
    from wsi_pipeline.pipeline import vsi_ets

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    vsi_path = input_dir / "a.vsi"
    vsi_path.touch()

    direct_calls: list[dict[str, object]] = []

    def _fake_direct(vsi_path_arg, per_tissue_dir, **kwargs):
        direct_calls.append(
            {
                "vsi_path": Path(vsi_path_arg),
                "per_tissue_dir": Path(per_tissue_dir),
                **kwargs,
            }
        )
        return [Path(per_tissue_dir) / "a_tissue_00.ome.zarr"]

    monkeypatch.setattr(vsi_ets, "process_vsi_with_direct_plating", _fake_direct)
    monkeypatch.setattr(
        vsi_ets,
        "vsi_to_source_ome_zarr",
        lambda *args, **kwargs: pytest.fail("default directory path should not materialize source OME-Zarr"),
    )

    results = vsi_ets.process_vsi_directory_with_plating(
        input_dir,
        tmp_path / "out",
        source_level=0,
        segmentation_level=7,
    )

    assert results == {str(vsi_path): [tmp_path / "out" / "per_tissue_ngff" / "a_tissue_00.ome.zarr"]}
    assert direct_calls[0]["vsi_path"] == vsi_path
    assert direct_calls[0]["source_level"] == 0
    assert direct_calls[0]["segmentation_level"] == 7
    assert direct_calls[0]["tile_frame_level"] == "segmentation"


def test_direct_vsi_plating_forwards_uncompressed_streaming_progress(monkeypatch, tmp_path):
    from wsi_pipeline.pipeline import vsi_ets

    vsi_path = tmp_path / "a.vsi"
    vsi_path.touch()
    ets_path = tmp_path / "_a_" / "stack10002" / "frame_t.ets"
    ets_path.parent.mkdir(parents=True)
    ets_path.touch()

    class _DummyETS:
        nlevels = 2

        def __init__(self, path):
            assert Path(path) == ets_path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def level_shape(self, level):
            return [(16, 16), (8, 8)][level]

        def read_level(self, level):
            return np.zeros((*self.level_shape(level), 3), dtype=np.uint8)

    record = vsi_ets.TissueTileRecord(
        tile=da.from_array(np.zeros((16, 16, 3), dtype=np.uint8), chunks=(8, 8, 3)),
        tissue_index=0,
        label_id=1,
        crop_bounds_source_level=(0, 0, 16, 16),
        crop_bounds_segmentation_level=(0, 0, 8, 8),
        tile_dim=16,
    )
    writer_calls: list[dict[str, object]] = []

    def _fake_writer(**kwargs):
        writer_calls.append(kwargs)
        Path(kwargs["out_dir"]).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(vsi_ets, "find_ets_file", lambda path: ets_path)
    monkeypatch.setattr(vsi_ets, "get_vsi_metadata", lambda *args, **kwargs: {
        "physical_pixel_size_um": {"x": 0.25, "y": 0.5, "z": None}
    })
    monkeypatch.setattr(vsi_ets, "ETSFile", _DummyETS)
    monkeypatch.setattr(vsi_ets, "_segment_for_plating", lambda *args, **kwargs: (np.ones((8, 8), dtype=bool), {}))
    monkeypatch.setattr(vsi_ets, "_direct_ets_tissue_tile_records", lambda **kwargs: ([record], 16))
    monkeypatch.setattr(vsi_ets, "_is_big_tile", lambda *args, **kwargs: True)
    monkeypatch.setattr(vsi_ets, "write_ngff_from_tile_streaming_ome", _fake_writer)

    paths = vsi_ets.process_vsi_with_direct_plating(
        vsi_path,
        tmp_path / "out",
        source_level=0,
        segmentation_level=1,
        compression="none",
        progress_mode="both",
        progress_interval_s=2.0,
    )

    assert len(paths) == 1
    assert len(writer_calls) == 1
    assert writer_calls[0]["compressor"] is None
    assert writer_calls[0]["progress_mode"] == "both"
    assert writer_calls[0]["progress_interval_s"] == 2.0


def test_native_level_specs_align_canvas_and_preserve_translation():
    from wsi_pipeline.pipeline import vsi_ets

    record = vsi_ets.TissueTileRecord(
        tile=da.from_array(np.zeros((1024, 1024, 3), dtype=np.uint8), chunks=(512, 512, 3)),
        tissue_index=0,
        label_id=1,
        crop_bounds_source_level=(0, 170, 511, 682),
        crop_bounds_segmentation_level=(0, 0, 16, 16),
        tile_dim=1024,
        frame_debug={
            "logical_canvas_source_yx": {"y0": 170, "x0": -1, "y1": 1194, "x1": 1023},
            "label_crop_seg_yx": {"y0": 0, "x0": 0, "y1": 16, "x1": 16},
        },
    )

    specs = vsi_ets._native_pyramid_level_specs(
        record=record,
        source_level=0,
        source_shape_yx=(2048, 2048),
        ets_level_shapes_yx=[(2048, 2048), (1024, 1024)],
        source_phys_xy_um=(0.25, 0.5),
        block_xy=512,
        min_side_for_mips=256,
        source_tile_aligned_canvas=True,
        source_tile_size_yx=(512, 512),
    )

    assert len(specs) == 2
    assert specs[0].canvas_source_yx.as_yx() == (0, -512, 1536, 1024)
    assert specs[0].translation_yx_um == (0.0, -128.0)
    assert specs[1].source_level == 1
    assert specs[1].canonical_canvas_source_yx.as_yx() == (0, -512, 1536, 1024)
    assert specs[1].canvas_source_yx.as_yx() == (0, -256, 768, 512)
    assert specs[1].output_shape_yx == (768, 768)
    assert specs[1].source_read_envelope_yx.as_yx() == (0, -512, 1024, 512)
    dataset = {
        "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, specs[1].phys_xy_um[1], specs[1].phys_xy_um[0]]},
            {"type": "translation", "translation": [0.0, *specs[1].translation_yx_um]},
        ]
    }
    recovered = vsi_ets._parent_bounds_from_ngff_transform_source_yx(
        dataset,
        shape_yx=specs[1].output_shape_yx,
        source_phys_xy_um=(0.25, 0.5),
    )
    assert recovered == {"y0": 0.0, "x0": -512.0, "y1": 1536.0, "x1": 1024.0}
    capped = vsi_ets._native_pyramid_level_specs(
        record=record,
        source_level=0,
        source_shape_yx=(2048, 2048),
        ets_level_shapes_yx=[(2048, 2048), (1024, 1024)],
        source_phys_xy_um=(0.25, 0.5),
        block_xy=512,
        min_side_for_mips=256,
        requested_mips=1,
        source_tile_aligned_canvas=False,
        source_tile_size_yx=(512, 512),
    )
    assert len(capped) == 1


def test_native_level_specs_do_not_inflate_coarse_scale_fov():
    from wsi_pipeline.pipeline import vsi_ets

    record = vsi_ets.TissueTileRecord(
        tile=da.from_array(np.zeros((13824, 10240, 3), dtype=np.uint8), chunks=(512, 512, 3)),
        tissue_index=0,
        label_id=1,
        crop_bounds_source_level=(0, 0, 10240, 13824),
        crop_bounds_segmentation_level=(0, 0, 640, 864),
        tile_dim=13824,
        frame_debug={
            "logical_canvas_source_yx": {"y0": 0, "x0": 0, "y1": 13824, "x1": 10240},
            "label_crop_seg_yx": {"y0": 0, "x0": 0, "y1": 864, "x1": 640},
        },
    )

    specs = vsi_ets._native_pyramid_level_specs(
        record=record,
        source_level=0,
        source_shape_yx=(27648, 20480),
        ets_level_shapes_yx=[
            (27648, 20480),
            (13824, 10240),
            (6912, 5120),
            (3456, 2560),
            (1728, 1280),
        ],
        source_phys_xy_um=(0.25, 0.5),
        block_xy=512,
        min_side_for_mips=512,
        requested_mips=5,
        source_tile_aligned_canvas=True,
        source_tile_size_yx=(512, 512),
    )

    assert specs[0].output_shape_yx == (13824, 10240)
    assert specs[4].output_shape_yx == (864, 640)
    assert specs[4].output_shape_yx != (1536, 1536)
    assert specs[4].source_read_envelope_yx.as_yx() == (0, 0, 1024, 1024)


def test_write_native_ets_tissue_pyramid_writes_native_levels_and_metrics(monkeypatch, tmp_path):
    from wsi_pipeline.pipeline import vsi_ets

    class FakeETS:
        nlevels = 2
        tile_ysize = 4
        tile_xsize = 4

        def __init__(self, path):
            self.path = Path(path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def level_shape(self, level):
            return [(8, 8), (4, 4)][int(level)]

        def get_tile_decoded(self, level, col, row):
            value = int(level) * 100 + int(row) * 10 + int(col)
            tile = np.zeros((4, 4, 3), dtype=np.uint8)
            tile[..., 0] = value
            tile[..., 1] = value + 1
            tile[..., 2] = value + 2
            return tile

    class FakeArray:
        def __init__(self, *, shape, chunks, dtype):
            self.shape = tuple(shape)
            self.chunks = tuple(chunks)
            self.data = np.zeros(shape, dtype=dtype)
            self.attrs: dict[str, object] = {}

        def __setitem__(self, key, value):
            self.data[key] = value

    class FakeGroup:
        def __init__(self):
            self.attrs: dict[str, object] = {}
            self.arrays: dict[str, FakeArray] = {}
            self.children: dict[str, FakeGroup] = {}

        def create_group(self, name, overwrite=True):  # noqa: ARG002
            child = FakeGroup()
            self.children[name] = child
            return child

    root = FakeGroup()

    def fake_create_group_array(group, name, **kwargs):
        arr = FakeArray(
            shape=kwargs["shape"],
            chunks=kwargs["chunks"],
            dtype=kwargs["dtype"],
        )
        group.arrays[name] = arr
        return arr

    monkeypatch.setattr(vsi_ets, "ETSFile", FakeETS)
    monkeypatch.setattr(vsi_ets, "open_group_v2", lambda *args, **kwargs: root)
    monkeypatch.setattr(vsi_ets, "create_group_array", fake_create_group_array)

    record = vsi_ets.TissueTileRecord(
        tile=da.from_array(np.zeros((8, 8, 3), dtype=np.uint8), chunks=(4, 4, 3)),
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
    labels = np.ones((4, 4), dtype=np.int32)

    stats = vsi_ets.write_native_ets_tissue_pyramid_ome(
        ets_path=tmp_path / "fake.ets",
        out_dir=tmp_path / "native.ome.zarr",
        record=record,
        lr_labels=labels,
        source_level=0,
        source_shape_yx=(8, 8),
        source_phys_xy_um=(0.25, 0.5),
        block_xy=4,
        name="native",
        compressor=None,
        sparse_zero_chunks=False,
        store_tissue_mask=True,
        min_side_for_mips=2,
    )

    assert root.arrays["s0"].shape == (3, 8, 8)
    assert root.arrays["s1"].shape == (3, 4, 4)
    assert root.arrays["s0"].data[:, 0, 0].tolist() == [0, 1, 2]
    assert root.arrays["s1"].data[:, 0, 0].tolist() == [100, 101, 102]
    assert root.attrs["native_source_pyramid"]["output_scale_to_source_level"] == {
        "s0": 0,
        "s1": 1,
    }
    assert root.attrs["native_source_pyramid"][
        "canonical_canvas_in_source_level_coordinates"
    ] == {"y0": 0, "x0": 0, "y1": 8, "x1": 8, "h": 8, "w": 8}
    assert root.attrs["native_source_pyramid"]["levels"][1]["source_level"] == 1
    assert root.attrs["native_source_pyramid"]["levels"][1]["output_canvas_source_yx"] == {
        "y0": 0,
        "x0": 0,
        "y1": 4,
        "x1": 4,
        "h": 4,
        "w": 4,
    }
    assert root.attrs["native_source_pyramid"]["levels"][1]["source_read_envelope_yx"] == {
        "y0": 0,
        "x0": 0,
        "y1": 4,
        "x1": 4,
        "h": 4,
        "w": 4,
    }
    mask_group = root.children["labels"].children["tissue_mask"]
    assert mask_group.arrays["s0"].shape == (8, 8)
    assert set(np.unique(mask_group.arrays["s0"].data).tolist()).issubset({0, 1})
    assert np.count_nonzero(mask_group.arrays["s0"].data) > 0
    assert stats["rgb_write_amplification"] == 1.0
    assert stats["mask_write_amplification"] == 1.0
    assert stats["rgb_pyramid_semantics"] == "native_scanner_pyramid"


def test_vsi_direct_plating_native_policy_routes_to_native_writer(monkeypatch, tmp_path):
    from wsi_pipeline.pipeline import vsi_ets

    vsi_path = tmp_path / "Image_01.vsi"
    vsi_path.touch()
    ets_path = tmp_path / "_Image_01_" / "stack10002" / "frame_t.ets"
    ets_path.parent.mkdir(parents=True)
    ets_path.touch()

    class DummyETS:
        nlevels = 2

        def __init__(self, path):
            self.path = Path(path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def level_shape(self, level):
            return [(16, 16), (8, 8)][level]

        def read_level(self, level):
            return np.zeros((*self.level_shape(level), 3), dtype=np.uint8)

    record = vsi_ets.TissueTileRecord(
        tile=da.from_array(np.zeros((16, 16, 3), dtype=np.uint8), chunks=(8, 8, 3)),
        tissue_index=0,
        label_id=1,
        crop_bounds_source_level=(0, 0, 16, 16),
        crop_bounds_segmentation_level=(0, 0, 8, 8),
        tile_dim=16,
        frame_debug={
            "logical_canvas_source_yx": {"y0": 0, "x0": 0, "y1": 16, "x1": 16},
            "label_crop_seg_yx": {"y0": 0, "x0": 0, "y1": 8, "x1": 8},
        },
    )
    native_calls: list[dict[str, object]] = []

    def fake_native_writer(**kwargs):
        native_calls.append(kwargs)
        Path(kwargs["out_dir"]).mkdir(parents=True, exist_ok=True)
        return {
            "rgb_s0_chunks_expected": 1,
            "mask_s0_chunks_expected": 1,
            "rgb_chunks_written": 1,
            "mask_chunks_written": 1,
            "rgb_write_amplification": 1.0,
            "mask_write_amplification": 1.0,
            "pyramid_generation_policy": "native_source_pyramid_crop",
            "source_tile_aligned_canvas": True,
            "canonical_canvas_in_source_level_coordinates": {
                "y0": 0,
                "x0": 0,
                "y1": 16,
                "x1": 16,
                "h": 16,
                "w": 16,
            },
            "output_scale_to_source_level": {"s0": 0},
            "native_pyramid_levels": [
                {
                    "path": "s0",
                    "source_level": 0,
                    "output_canvas_source_yx": {
                        "y0": 0,
                        "x0": 0,
                        "y1": 16,
                        "x1": 16,
                        "h": 16,
                        "w": 16,
                    },
                    "source_read_envelope_yx": {
                        "y0": 0,
                        "x0": 0,
                        "y1": 16,
                        "x1": 16,
                        "h": 16,
                        "w": 16,
                    },
                    "output_shape_yx": [16, 16],
                    "translation_yx_um": [0.0, 0.0],
                }
            ],
            "mask_generation_policy": "project_segmentation_per_scale",
            "mask_pyramid_semantics": "label_safe_nearest",
        }

    monkeypatch.setattr(vsi_ets, "find_ets_file", lambda path: ets_path)
    monkeypatch.setattr(
        vsi_ets,
        "get_vsi_metadata",
        lambda *args, **kwargs: {"physical_pixel_size_um": {"x": 0.25, "y": 0.5, "z": None}},
    )
    monkeypatch.setattr(vsi_ets, "ETSFile", DummyETS)
    monkeypatch.setattr(
        vsi_ets,
        "_segment_for_plating",
        lambda *args, **kwargs: (np.ones((8, 8), dtype=bool), {}),
    )
    monkeypatch.setattr(vsi_ets, "_direct_ets_tissue_tile_records", lambda **kwargs: ([record], 16))
    monkeypatch.setattr(
        vsi_ets,
        "write_ngff_from_tile_streaming_ome",
        lambda **kwargs: pytest.fail("native policy should not use streamed-s0 writer"),
    )
    monkeypatch.setattr(vsi_ets, "write_native_ets_tissue_pyramid_ome", fake_native_writer)

    paths = vsi_ets.process_vsi_with_direct_plating(
        vsi_path,
        tmp_path / "out",
        source_level=0,
        segmentation_level=1,
        compression="none",
        pyramid_generation_policy="native_source_pyramid_crop",
        source_tile_aligned_canvas=True,
    )

    assert len(paths) == 1
    assert native_calls[0]["source_tile_aligned_canvas"] is True
    assert native_calls[0]["source_level"] == 0
    manifest = json.loads((paths[0] / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["pyramid_generation_policy"] == "native_source_pyramid_crop"
    assert manifest["source_tile_aligned_canvas"] is True
    tissue_manifest = json.loads((paths[0] / "tissue_manifest.json").read_text(encoding="utf-8"))
    assert tissue_manifest["pyramid_generation_policy"] == "native_source_pyramid_crop"
    assert tissue_manifest["source_tile_aligned_canvas"] is True
    assert tissue_manifest["output_scale_to_source_level"] == {"s0": 0}
    assert tissue_manifest["canonical_canvas_in_source_level_coordinates"]["h"] == 16
    assert tissue_manifest["native_pyramid_levels"][0]["output_shape_yx"] == [16, 16]


def test_direct_ets_tissue_tile_records_read_only_tissue_blocks(monkeypatch, tmp_path):
    from wsi_pipeline.pipeline import vsi_ets

    def _fake_read_region(_ets_path, *, level, x0, y0, x1, y1):
        yy, xx = np.mgrid[y0:y1, x0:x1]
        out = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        out[..., 0] = xx + 1
        out[..., 1] = yy + 1
        out[..., 2] = level
        return out

    monkeypatch.setattr(vsi_ets, "_read_ets_region_yxc", _fake_read_region)
    low_res_mask = np.zeros((2, 4), dtype=bool)
    low_res_mask[:, 0] = True
    low_res_mask[:, 3] = True

    records, tile_dim = vsi_ets._direct_ets_tissue_tile_records(
        ets_path=tmp_path / "fake.ets",
        source_level=1,
        source_shape_yx=(4, 8),
        low_res_filled=low_res_mask,
        chunk=2,
        pad_multiple=2,
        extra_margin_px=0,
        tile_frame_level="source",
    )

    assert tile_dim == 4
    assert [record.tissue_index for record in records] == [0, 1]
    first = records[0].tile.compute()
    second = records[1].tile.compute()
    assert first.shape == (4, 4, 3)
    assert second.shape == (4, 4, 3)
    assert np.count_nonzero(first[..., 0]) > 0
    assert np.count_nonzero(second[..., 0]) > 0
    assert np.count_nonzero(first[:, -1, 0]) == 0
    assert np.count_nonzero(second[..., 0]) < second[..., 0].size


def _assert_direct_ets_records_match_shared_generator(
    monkeypatch,
    tmp_path,
    *,
    source_shape_yx: tuple[int, int],
    mask_shape_yx: tuple[int, int],
) -> None:
    from wsi_pipeline.pipeline import vsi_ets
    from wsi_pipeline.tiles.generator import generate_tissue_tile_records

    source_h, source_w = source_shape_yx
    yy, xx = np.mgrid[:source_h, :source_w]
    source_yxc = np.zeros((source_h, source_w, 3), dtype=np.uint8)
    source_yxc[..., 0] = xx + 1
    source_yxc[..., 1] = yy + 1
    source_yxc[..., 2] = (xx + yy) % 251

    def _fake_read_region(_ets_path, *, level, x0, y0, x1, y1):  # noqa: ARG001
        return source_yxc[y0:y1, x0:x1, :]

    monkeypatch.setattr(vsi_ets, "_read_ets_region_yxc", _fake_read_region)
    mask = np.zeros(mask_shape_yx, dtype=bool)
    mask[:, : max(1, mask_shape_yx[1] // 4)] = True
    mask[:, -max(1, mask_shape_yx[1] // 4) :] = True

    shared_records, shared_dim = generate_tissue_tile_records(
        s0_cyx=da.from_array(np.moveaxis(source_yxc, -1, 0), chunks=(3, 4, 4)),
        low_res_filled=mask,
        tile_frame_level="source",
        chunk=4,
        pad_multiple=4,
        extra_margin_px=0,
    )
    direct_records, direct_dim = vsi_ets._direct_ets_tissue_tile_records(
        ets_path=tmp_path / "fake.ets",
        source_level=2,
        source_shape_yx=source_shape_yx,
        low_res_filled=mask,
        chunk=4,
        pad_multiple=4,
        extra_margin_px=0,
        tile_frame_level="source",
    )

    assert direct_dim == shared_dim
    assert len(direct_records) == len(shared_records) == 2
    for direct, shared in zip(direct_records, shared_records, strict=True):
        assert direct.tissue_index == shared.tissue_index
        assert direct.crop_bounds_source_level == shared.crop_bounds_source_level
        assert direct.crop_bounds_segmentation_level == shared.crop_bounds_segmentation_level
        np.testing.assert_array_equal(direct.tile.compute(), shared.tile.compute())


def test_direct_ets_tissue_tile_records_match_shared_generator_same_level(
    monkeypatch,
    tmp_path,
):
    _assert_direct_ets_records_match_shared_generator(
        monkeypatch,
        tmp_path,
        source_shape_yx=(4, 8),
        mask_shape_yx=(4, 8),
    )


def test_direct_ets_tissue_tile_records_match_shared_generator_cross_level(
    monkeypatch,
    tmp_path,
):
    _assert_direct_ets_records_match_shared_generator(
        monkeypatch,
        tmp_path,
        source_shape_yx=(8, 16),
        mask_shape_yx=(4, 8),
    )


def test_direct_ets_tissue_tile_records_can_frame_at_segmentation_level(monkeypatch, tmp_path):
    from wsi_pipeline.pipeline import vsi_ets

    source_yxc = np.zeros((80, 160, 3), dtype=np.uint8)

    def _fake_read_region(_ets_path, *, level, x0, y0, x1, y1):  # noqa: ARG001
        return source_yxc[y0:y1, x0:x1, :]

    monkeypatch.setattr(vsi_ets, "_read_ets_region_yxc", _fake_read_region)
    mask = np.zeros((40, 80), dtype=bool)
    mask[12:32, 20:60] = True

    records, tile_dim = vsi_ets._direct_ets_tissue_tile_records(
        ets_path=tmp_path / "fake.ets",
        source_level=6,
        source_shape_yx=(80, 160),
        low_res_filled=mask,
        chunk=16,
        pad_multiple=16,
        extra_margin_px=0,
        tile_frame_level="segmentation",
    )

    assert tile_dim == 96
    assert len(records) == 1
    assert records[0].segmentation_tile_dim == 48
    assert records[0].source_tile_dim == 96
    assert records[0].crop_bounds_segmentation_level == (16, 0, 64, 40)
    assert records[0].crop_bounds_source_level == (32, 0, 128, 80)
    assert records[0].tile.shape == (96, 96, 3)


def test_array_stats_flags_channel_collapsed_pixels():
    from wsi_pipeline.pipeline import vsi_ets

    gray = np.repeat(np.arange(16, dtype=np.uint8).reshape(4, 4, 1), 3, axis=2)
    stats = vsi_ets._array_stats(gray)

    assert stats["channel_means"] == [7.5, 7.5, 7.5]
    assert stats["channel_absdiff_means"] == {"rg": 0.0, "rb": 0.0, "gb": 0.0}
    assert stats["channels_nearly_identical"] is True


def test_read_ome_zarr_s0_yxc_reports_rgb_channel_stats(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from wsi_pipeline.pipeline import vsi_ets

    root = tmp_path / "tile.ome.zarr"
    data = np.zeros((3, 4, 4), dtype=np.uint8)
    data[0, :, :] = 10
    data[1, :, :] = 20
    data[2, :, :] = 30

    class _FakeArray:
        shape = data.shape

        def __getitem__(self, key):
            return data[key]

    monkeypatch_module = SimpleNamespace(open_array=lambda *args, **kwargs: _FakeArray())
    monkeypatch.setitem(sys.modules, "zarr", monkeypatch_module)

    readback, stats = vsi_ets._read_ome_zarr_s0_yxc(root, max_debug_pixels=100)

    assert readback.shape == (4, 4, 3)
    assert stats["channel_means"] == [10.0, 20.0, 30.0]
    assert stats["channels_nearly_identical"] is False


def test_diagnose_vsi_replating_writes_lightweight_outputs(monkeypatch, tmp_path):
    from wsi_pipeline.pipeline import vsi_ets

    vsi_path = tmp_path / "sample.vsi"
    vsi_path.touch()
    ets_path = tmp_path / "_sample_" / "stack10002" / "frame_t.ets"
    ets_path.parent.mkdir(parents=True)
    ets_path.touch()

    class _DummyETS:
        nlevels = 3

        def __init__(self, path):
            assert Path(path) == ets_path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def level_shape(self, level):
            return [(16, 32), (8, 16), (4, 8)][level]

        def read_level(self, level):
            shape = self.level_shape(level)
            arr = np.zeros((*shape, 3), dtype=np.uint8)
            arr[..., 0] = 120
            arr[..., 1] = 80
            arr[..., 2] = 40
            return arr

    def _fake_segment(image, **kwargs):  # noqa: ARG001
        shape = tuple(map(int, image.shape))
        h, w = shape[:2] if shape[-1] == 3 else shape[1:]
        mask = np.zeros((h, w), dtype=bool)
        mask[:, :2] = True
        mask[:, -2:] = True
        return mask, {"component_source": "fake"}

    monkeypatch.setattr(vsi_ets, "find_ets_file", lambda path: ets_path)
    monkeypatch.setattr(vsi_ets, "ETSFile", _DummyETS)
    monkeypatch.setattr(vsi_ets, "_segment_for_plating", _fake_segment)
    monkeypatch.setattr(
        vsi_ets,
        "_read_ets_region_yxc",
        lambda _ets_path, *, level, x0, y0, x1, y1: np.ones((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        * 100,
    )

    result = vsi_ets.diagnose_vsi_replating(
        vsi_path,
        tmp_path / "diag",
        source_level=1,
        segmentation_level=2,
    )

    assert (tmp_path / "diag" / "diagnostics.json").is_file()
    assert (tmp_path / "diag" / "ets_overlay.png").is_file()
    assert result["source_level"] == 1
    assert result["segmentation_level"] == 2
    assert result["tile_frame_level"] == "segmentation"
    assert result["source_tile_dim"] == 1024
    assert result["segmentation_tile_dim"] == 512
    assert result["effective_segmentation_tile_dim"] == 512.0
    assert result["notebook_equivalent_frame"] is True
    assert result["ets_segmentation_input"]["component_count"] == 2
    assert [record["tissue_index"] for record in result["tile_records"]] == [0, 1]
