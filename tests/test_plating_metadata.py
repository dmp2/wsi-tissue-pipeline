from __future__ import annotations

import inspect
import json
from pathlib import Path

import dask.array as da
import numpy as np

import wsi_pipeline.vsi_converter as vsi_converter
from wsi_pipeline.pipeline import plating as plating_mod


def _make_metadata_payload(*, include_stage_origin: bool = False) -> dict[str, object]:
    """Build a lightweight dual-schema metadata payload for plating tests."""
    axes = [
        {"name": "c", "type": "channel"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    coordinate_systems = [
        {
            "name": "array",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "array"},
                {"name": "x", "type": "array"},
            ],
        },
        {"name": "image-plane", "axes": axes},
    ]
    if include_stage_origin:
        coordinate_systems.append({"name": "slide", "axes": axes})

    latest_dataset = {
        "path": "s0",
        "coordinateTransformations": [
            {
                "type": "scale",
                "input": "array",
                "output": "image-plane",
                "scale": [1.0, 0.5, 0.25],
            }
        ],
    }
    if include_stage_origin:
        latest_dataset["coordinateTransformations"].append(
            {
                "type": "translation",
                "input": "image-plane",
                "output": "slide",
                "translation": [0.0, 20.0, 10.0],
            }
        )

    return {
        "channel_count": 3,
        "channel_labels": ["red", "green", "blue"],
        "physical_pixel_size_um": {"x": 0.25, "y": 0.5, "z": None},
        "canonical_metadata": {
            "stage_origin_um": {"x": 10.0, "y": 20.0} if include_stage_origin else None
        },
        "compatibility": {"selected_schema": "latest"},
        "ngff_latest": {
            "schema": "latest",
            "dimension_order": ["c", "y", "x"],
            "axes": axes,
            "coordinateSystems": coordinate_systems,
            "arrayToPhysicalTransformations": [
                {
                    "type": "scale",
                    "input": "array",
                    "output": "image-plane",
                    "scale": [1.0, 0.5, 0.25],
                }
            ],
            "multiscales": [
                {
                    "name": "source",
                    "axes": axes,
                    "coordinateSystems": coordinate_systems,
                    "datasets": [latest_dataset],
                }
            ],
        },
        "ngff_v04": {
            "multiscales": [
                {
                    "name": "source",
                    "version": "0.4",
                    "axes": axes,
                    "datasets": [
                        {
                            "path": "s0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 0.5, 0.25]}
                            ],
                        }
                    ],
                }
            ]
        },
    }


def _make_source_root(path: Path) -> None:
    """Create a minimal NGFF root used by plating metadata tests."""
    path.mkdir(parents=True, exist_ok=True)


def _fake_ngff_root(dataset_count: int = 2):
    datasets = []
    for idx in range(dataset_count):
        datasets.append(
            {
                "path": f"s{idx}",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 0.5 * (2**idx), 0.25 * (2**idx)]}
                ],
            }
        )

    class _Root:
        attrs = {"multiscales": [{"datasets": datasets}]}

    return _Root()


def _run_plating_capture(
    monkeypatch,
    tmp_path: Path,
    *,
    source_context: dict[str, object] | None,
) -> list[dict[str, object]]:
    """Run plating with patched compute/writer paths and capture writer kwargs."""
    source_root = tmp_path / "source.ome.zarr"
    _make_source_root(source_root)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(plating_mod.zarr, "open_group", lambda *args, **kwargs: _fake_ngff_root())

    def _fake_from_zarr(path, name=None):  # noqa: ARG001
        shape = (3, 16, 16) if str(path).endswith("s0") else (3, 8, 8)
        data = np.zeros(shape, dtype=np.uint8)
        return da.from_array(data, chunks=shape)

    monkeypatch.setattr(plating_mod.da, "from_zarr", _fake_from_zarr)
    monkeypatch.setattr(
        plating_mod,
        "generate_tissue_tile_records",
        lambda **kwargs: (
            [
                plating_mod.TissueTileRecord(
                    tile=da.from_array(np.ones((8, 8, 3), dtype=np.uint8), chunks=(8, 8, 3)),
                    tissue_index=0,
                    label_id=1,
                    crop_bounds_source_level=(2, 4, 10, 12),
                    crop_bounds_segmentation_level=(1, 2, 5, 6),
                    tile_dim=8,
                )
            ],
            8,
        ),
    )
    monkeypatch.setattr(plating_mod, "compute_num_mips_min_side", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        plating_mod,
        "build_mips_from_yxc",
        lambda tile, levels: [tile, tile[::2, ::2, :]],
    )
    monkeypatch.setattr(plating_mod, "_safe_close_existing_client", lambda: None)

    writer_calls: list[dict[str, object]] = []

    def _capture_writer(*args, **kwargs):
        Path(kwargs["out_dir"]).mkdir(parents=True, exist_ok=True)
        writer_calls.append(kwargs)

    def _unexpected_writer(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("Unexpected big-tile writer path in plating metadata test.")

    monkeypatch.setattr(plating_mod, "write_ngff_from_mips_ngffzarr", _capture_writer)
    monkeypatch.setattr(plating_mod, "write_ngff_from_tile_ts", _unexpected_writer)
    monkeypatch.setattr(plating_mod, "write_ngff_from_tile_streaming_ome", _unexpected_writer)

    def _segment_fn(image, **kwargs):  # noqa: ARG001
        return np.ones((4, 4), dtype=bool), None

    plating_mod.process_slide_with_plating(
        source_root,
        out_dir,
        segment_fn=_segment_fn,
        source_context=source_context,
    )
    return writer_calls


def test_plating_without_source_context_uses_phys_fallback_only(monkeypatch, tmp_path):
    calls = _run_plating_capture(monkeypatch, tmp_path, source_context=None)

    assert len(calls) == 1
    assert calls[0]["ngff_metadata"] is None
    assert calls[0]["metadata_schema"] == "v0.4"


def test_plating_uses_supplied_metadata_and_projects_tile_levels(monkeypatch, tmp_path):
    payload = _make_metadata_payload(include_stage_origin=True)
    calls = _run_plating_capture(
        monkeypatch,
        tmp_path,
        source_context={"ngff_metadata": payload, "metadata_schema": "latest"},
    )

    assert len(calls) == 1
    assert calls[0]["metadata_schema"] == "latest"

    projected = calls[0]["ngff_metadata"]
    assert isinstance(projected, dict)
    latest = projected["ngff_latest"]["multiscales"][0]
    assert len(latest["datasets"]) == 2
    assert all(
        transform.get("type") != "translation"
        for dataset in latest["datasets"]
        for transform in dataset.get("coordinateTransformations", [])
    )
    assert "absolute_origin_translation" in projected["compatibility"]["lossy_fields_for_v04"]


def test_tile_metadata_projection_uses_selected_source_level_scale():
    payload = _make_metadata_payload()

    projected = plating_mod._tile_ngff_metadata_or_none(
        payload,
        dataset_count=2,
        name="tile-at-source-level",
        phys_xy_um=(2.0, 4.0),
    )

    assert projected is not None
    datasets = projected["ngff_v04"]["multiscales"][0]["datasets"]
    assert datasets[0]["coordinateTransformations"][0]["scale"] == [1.0, 4.0, 2.0]
    assert datasets[1]["coordinateTransformations"][0]["scale"] == [1.0, 8.0, 4.0]


def test_plating_vsi_context_fetches_metadata_once(monkeypatch, tmp_path):
    payload = _make_metadata_payload()
    call_counter = {"count": 0}

    def _fake_get_vsi_metadata(*args, **kwargs):  # noqa: ARG001
        call_counter["count"] += 1
        return payload

    monkeypatch.setattr(vsi_converter, "get_vsi_metadata", _fake_get_vsi_metadata)
    calls = _run_plating_capture(
        monkeypatch,
        tmp_path,
        source_context={
            "source_kind": "vsi",
            "source_path": str(tmp_path / "sample.vsi"),
            "metadata_backend": "auto",
            "metadata_schema": "v0.4",
        },
    )

    assert call_counter["count"] == 1
    assert len(calls) == 1
    assert calls[0]["metadata_schema"] == "v0.4"
    assert calls[0]["ngff_metadata"] is not None


def test_plating_default_config_uses_notebook_segmentation(monkeypatch):
    from wsi_pipeline.config import SegmentationConfig

    calls: dict[str, object] = {}

    def _fake_segment_tissue(image, **kwargs):
        calls.update(kwargs)
        return np.ones((8, 10), dtype=bool), {"ok": True}

    monkeypatch.setattr(plating_mod, "segment_tissue", _fake_segment_tissue)
    config = SegmentationConfig(
        target_long_side=777,
        min_area_px=321,
        struct_elem_px=6,
        stain_gate=True,
        stain_gate_mode="adaptive-he",
        stain_min_od=0.12,
        split_touching=False,
        appendage_refinement_enabled=True,
        diagnostics=True,
    )

    mask, info = plating_mod._segment_for_plating(
        da.zeros((3, 8, 10), chunks=(3, 8, 10), dtype=np.uint8),
        segment_fn=None,
        segmentation_config=config,
        min_size=1,
        struct_elem_px=2,
    )

    assert mask.shape == (8, 10)
    assert info == {"ok": True}
    assert calls["target_long_side"] == 777
    assert calls["min_area_px"] == 321
    assert calls["struct_elem_px"] == 6
    assert calls["stain_gate"] is True
    assert calls["stain_gate_mode"] == "adaptive-he"
    assert calls["split_touching"] is False
    assert calls["appendage_refinement_enabled"] is True
    assert calls["diagnostics"] is True


def test_plating_writes_derivative_manifest_and_forwards_config(monkeypatch, tmp_path):
    from wsi_pipeline.config import SegmentationConfig, TileConfig

    source_root = tmp_path / "source.ome.zarr"
    _make_source_root(source_root)
    monkeypatch.setattr(plating_mod.zarr, "open_group", lambda *args, **kwargs: _fake_ngff_root(3))

    def _fake_from_zarr(path, name=None):  # noqa: ARG001
        if str(path).endswith("s1"):
            shape = (3, 12, 16)
        elif str(path).endswith("s2"):
            shape = (3, 6, 8)
        else:
            shape = (3, 24, 32)
        return da.from_array(np.zeros(shape, dtype=np.uint8), chunks=shape)

    segment_kwargs: dict[str, object] = {}

    def _segment_fn(image, **kwargs):  # noqa: ARG001
        segment_kwargs.update(kwargs)
        mask = np.zeros((6, 8), dtype=bool)
        mask[1:4, 2:6] = True
        return mask, {"ok": True}

    monkeypatch.setattr(plating_mod.da, "from_zarr", _fake_from_zarr)
    monkeypatch.setattr(
        plating_mod,
        "generate_tissue_tile_records",
        lambda **kwargs: (
            [
                plating_mod.TissueTileRecord(
                    tile=da.from_array(np.ones((8, 8, 3), dtype=np.uint8), chunks=(4, 4, 3)),
                    tissue_index=0,
                    label_id=1,
                    crop_bounds_source_level=(2, 4, 10, 12),
                    crop_bounds_segmentation_level=(1, 2, 5, 6),
                    tile_dim=8,
                )
            ],
            8,
        ),
    )
    monkeypatch.setattr(plating_mod, "compute_num_mips_min_side", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        plating_mod,
        "build_mips_from_yxc",
        lambda tile, levels: [tile, tile[::2, ::2, :]],
    )
    monkeypatch.setattr(plating_mod, "_safe_close_existing_client", lambda: None)

    writer_calls: list[dict[str, object]] = []

    def _capture_writer(*args, **kwargs):
        Path(kwargs["out_dir"]).mkdir(parents=True, exist_ok=True)
        writer_calls.append(kwargs)

    monkeypatch.setattr(plating_mod, "write_ngff_from_mips_ngffzarr", _capture_writer)

    paths = plating_mod.process_slide_with_plating(
        source_root,
        tmp_path / "out",
        segment_fn=_segment_fn,
        source_level=1,
        segmentation_level="s2",
        segmentation_config=SegmentationConfig(
            min_area_px=123,
            struct_elem_px=7,
            stain_gate=True,
            stain_gate_mode="adaptive-he",
            split_touching=False,
            keep_top_k=4,
            appendage_refinement_enabled=True,
        ),
        tile_config=TileConfig(chunk_size=256, pad_multiple=512, extra_margin_px=9),
        source_context={
            "source_vsi": "/data/source.vsi",
            "source_ets": "/data/_source_/stack10002/frame_t.ets",
            "source_ome_zarr": str(source_root),
        },
    )

    assert "expected_tissues" not in inspect.signature(
        plating_mod.process_slide_with_plating
    ).parameters
    assert segment_kwargs["min_size"] == 123
    assert segment_kwargs["struct_elem_px"] == 7
    assert segment_kwargs["stain_gate"] is True
    assert segment_kwargs["stain_gate_mode"] == "adaptive-he"
    assert segment_kwargs["split_touching"] is False
    assert segment_kwargs["keep_top_k"] == 4
    assert segment_kwargs["appendage_refinement_enabled"] is True
    assert len(writer_calls) == 1
    assert writer_calls[0]["chunks_xy"] == 256
    assert writer_calls[0]["channel_labels"] == ["red", "green", "blue"]
    assert writer_calls[0]["channel_colors"] == ["FF0000", "00FF00", "0000FF"]
    assert [path.name for path in paths] == ["source_tissue_00.ome.zarr"]

    manifest = json.loads((paths[0] / "tissue_manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "role": "derivative",
        "derivative_type": "tissue_crop_ome_zarr",
        "source_vsi": "/data/source.vsi",
        "source_ets": "/data/_source_/stack10002/frame_t.ets",
        "source_ome_zarr": str(source_root),
        "source_level": 1,
        "segmentation_level": 2,
        "tissue_index": 0,
        "crop_bounds_source_level": [2, 4, 10, 12],
        "crop_bounds_segmentation_level": [1, 2, 5, 6],
        "physical_pixel_size": {"x": 0.5, "y": 1.0, "unit": "micrometer"},
        "operations": [
            "read_ets_pyramid",
            "segment_lowres",
            "extract_tissue",
            "write_ome_zarr",
        ],
    }
