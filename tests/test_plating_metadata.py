from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import zarr

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
    root = zarr.open_group(str(path), mode="w")
    root.attrs.put(
        {
            "multiscales": [
                {
                    "datasets": [
                        {
                            "path": "s0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 0.5, 0.25]}
                            ],
                        },
                        {
                            "path": "s1",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 1.0, 0.5]}
                            ],
                        },
                    ]
                }
            ]
        }
    )


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

    def _fake_from_zarr(path, name=None):  # noqa: ARG001
        shape = (3, 16, 16) if str(path).endswith("s0") else (3, 8, 8)
        data = np.zeros(shape, dtype=np.uint8)
        return da.from_array(data, chunks=shape)

    monkeypatch.setattr(plating_mod.da, "from_zarr", _fake_from_zarr)
    monkeypatch.setattr(
        plating_mod,
        "generate_tissue_tiles",
        lambda **kwargs: ([da.from_array(np.ones((8, 8, 3), dtype=np.uint8), chunks=(8, 8, 3))], 8),
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
