from __future__ import annotations

import inspect
from pathlib import Path

import dask.array as da
import numpy as np
import pytest


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
    assert tissue["uncompressed_bytes_all_mips"] == 61440
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
    assert tissue["uncompressed_bytes_estimate"] > tissue["uncompressed_bytes_all_mips"]
    assert estimate["compression"]["mode"] == "lossless"


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
