from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest


def test_vsi_to_source_ome_zarr_uses_ets_dask_levels(monkeypatch, tmp_path):
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
    )

    assert out_path == tmp_path / "source.ome.zarr"
    assert resolved_ets == ets_path
    assert resolved_metadata == metadata
    assert to_dask_calls == [0, 1, 2]
    assert len(writer_call["mips_yxc"]) == 3
    assert writer_call["phys_xy_um"] == (0.25, 0.5)
    assert writer_call["chunks_xy"] == 256
    assert writer_call["ngff_metadata"] == metadata


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
    )

    assert "expected_tissues" not in inspect.signature(
        vsi_ets.process_vsi_directory_with_plating
    ).parameters
    assert source_calls == [vsi_a, vsi_b]
    assert set(results) == {str(vsi_a), str(vsi_b)}
    assert all(len(paths) == 2 for paths in results.values())
    assert [call["source_level"] for call in plating_calls] == [0, 0]
    assert [call["segmentation_level"] for call in plating_calls] == [7, 7]
    assert plating_calls[0]["source_context"]["source_vsi"] == str(vsi_a)
    assert plating_calls[0]["source_context"]["source_ets"] == "/ets/a.ets"
