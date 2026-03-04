from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import zarr

from wsi_pipeline.omezarr import (
    materialize_ngff_root_attrs,
    write_ngff_from_mips,
    write_ngff_from_mips_ngffzarr,
    write_ngff_from_tile_streaming_ome,
    write_ngff_from_tile_ts,
)
from wsi_pipeline.omezarr.metadata import _phys_xy_um


def _make_mips(levels: int = 3, channels: int = 3) -> list[np.ndarray]:
    """Build a small synthetic RGB pyramid in YXC layout."""
    height, width = 16, 12
    base = np.arange(height * width * channels, dtype=np.uint8).reshape(height, width, channels)
    mips: list[np.ndarray] = []
    current = base
    for _ in range(levels):
        mips.append(current.copy())
        current = current[::2, ::2, :]
    return mips


def _make_metadata_payload(
    *,
    dataset_count: int,
    channel_count: int = 3,
    name: str = "sample",
    base_phys_xy_um: tuple[float, float] = (0.25, 0.5),
    include_scales: bool = True,
) -> dict[str, object]:
    """Create a minimal rich NGFF payload that mirrors get_vsi_metadata()."""
    px_um, py_um = base_phys_xy_um
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
    latest_datasets = []
    v04_datasets = []
    for level in range(dataset_count):
        scale = [1.0, py_um * (2**level), px_um * (2**level)]
        latest_datasets.append(
            {
                "path": f"s{level}",
                "coordinateTransformations": (
                    [
                        {
                            "type": "scale",
                            "input": "array",
                            "output": "image-plane",
                            "scale": scale,
                        }
                    ]
                    if include_scales
                    else []
                ),
            }
        )
        v04_datasets.append(
            {
                "path": f"s{level}",
                "coordinateTransformations": (
                    [{"type": "scale", "scale": scale}] if include_scales else []
                ),
            }
        )

    return {
        "channel_count": channel_count,
        "channel_labels": [f"label_{idx}" for idx in range(channel_count)],
        "compatibility": {"selected_schema": "latest"},
        "ngff_latest": {
            "schema": "latest",
            "dimension_order": ["c", "y", "x"],
            "axes": axes,
            "coordinateSystems": coordinate_systems,
            "arrayToPhysicalTransformations": (
                [
                    {
                        "type": "scale",
                        "input": "array",
                        "output": "image-plane",
                        "scale": [1.0, py_um, px_um],
                    }
                ]
                if include_scales
                else []
            ),
            "multiscales": [
                {
                    "name": name,
                    "axes": axes,
                    "coordinateSystems": coordinate_systems,
                    "datasets": latest_datasets,
                }
            ],
        },
        "ngff_v04": {
            "multiscales": [
                {
                    "name": name,
                    "version": "0.4",
                    "axes": axes,
                    "datasets": v04_datasets,
                }
            ]
        },
    }


def _root_attrs(path) -> dict:
    """Read root attrs from a written OME-Zarr group."""
    return dict(zarr.open_group(str(path), mode="r").attrs)


def test_write_ngff_from_mips_preserves_default_v04_root_attrs(tmp_path):
    mips = _make_mips(levels=2)
    out_dir = tmp_path / "default.ome.zarr"

    write_ngff_from_mips(mips, out_dir, phys_xy_um=(0.25, 0.5))

    attrs = _root_attrs(out_dir)
    assert attrs["multiscales"][0]["version"] == "0.4"
    assert attrs["multiscales"][0]["axes"][0]["name"] == "c^"
    assert attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"] == [
        1.0,
        0.5,
        0.25,
    ]
    assert attrs["omero"]["channels"][0]["label"] == "ch0"


def test_write_ngff_from_mips_uses_materialized_v04_root_attrs(tmp_path):
    mips = _make_mips(levels=3)
    payload = _make_metadata_payload(dataset_count=3, name="payload_v04")
    out_dir = tmp_path / "payload-v04.ome.zarr"

    write_ngff_from_mips(
        mips,
        out_dir,
        phys_xy_um=(1.0, 1.0),
        ngff_metadata=payload,
        metadata_schema="v0.4",
    )

    attrs = _root_attrs(out_dir)
    expected = materialize_ngff_root_attrs(payload, "v0.4")
    assert attrs["multiscales"] == expected["multiscales"]
    assert attrs["omero"]["name"] == "payload_v04"
    assert attrs["omero"]["channels"][1]["label"] == "label_1"


def test_write_ngff_from_mips_uses_materialized_latest_root_attrs(tmp_path):
    mips = _make_mips(levels=2)
    payload = _make_metadata_payload(dataset_count=2, name="payload_latest")
    out_dir = tmp_path / "payload-latest.ome.zarr"

    write_ngff_from_mips(
        mips,
        out_dir,
        ngff_metadata=payload,
        metadata_schema="latest",
    )

    attrs = _root_attrs(out_dir)
    expected = materialize_ngff_root_attrs(payload, "latest")
    for key, value in expected.items():
        assert attrs[key] == value
    assert attrs["omero"]["version"] == "latest"
    assert _phys_xy_um(zarr.open_group(str(out_dir), mode="r")) == (0.25, 0.5)


def test_ngff_metadata_channel_mismatch_raises(tmp_path):
    mips = _make_mips(levels=2)
    payload = _make_metadata_payload(dataset_count=2, channel_count=1)

    with pytest.raises(ValueError, match="channels"):
        write_ngff_from_mips(
            mips,
            tmp_path / "channel-mismatch.ome.zarr",
            ngff_metadata=payload,
            metadata_schema="v0.4",
        )


def test_ngff_metadata_level_mismatch_raises(tmp_path):
    mips = _make_mips(levels=3)
    payload = _make_metadata_payload(dataset_count=2)

    with pytest.raises(ValueError, match="levels"):
        write_ngff_from_mips(
            mips,
            tmp_path / "level-mismatch.ome.zarr",
            ngff_metadata=payload,
            metadata_schema="v0.4",
        )


def test_missing_metadata_scales_falls_back_to_explicit_phys_xy(tmp_path):
    mips = _make_mips(levels=2)
    payload = _make_metadata_payload(dataset_count=2, include_scales=False)
    out_dir = tmp_path / "fallback-scales.ome.zarr"

    write_ngff_from_mips(
        mips,
        out_dir,
        phys_xy_um=(0.3, 0.6),
        ngff_metadata=payload,
        metadata_schema="v0.4",
    )

    attrs = _root_attrs(out_dir)
    assert attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"] == [
        1.0,
        0.6,
        0.3,
    ]


def test_metadata_scales_win_over_explicit_phys_xy(tmp_path):
    mips = _make_mips(levels=2)
    payload = _make_metadata_payload(dataset_count=2, base_phys_xy_um=(1.25, 2.5))
    out_dir = tmp_path / "metadata-wins.ome.zarr"

    write_ngff_from_mips(
        mips,
        out_dir,
        phys_xy_um=(0.3, 0.6),
        ngff_metadata=payload,
        metadata_schema="v0.4",
    )

    attrs = _root_attrs(out_dir)
    assert attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"] == [
        1.0,
        2.5,
        1.25,
    ]


def test_write_ngff_from_mips_ngffzarr_replaces_root_attrs(tmp_path):
    mips = _make_mips(levels=3)
    payload = _make_metadata_payload(dataset_count=3, name="ngffzarr")
    out_dir = tmp_path / "ngffzarr.ome.zarr"

    write_ngff_from_mips_ngffzarr(
        mips,
        out_dir,
        ngff_metadata=payload,
        metadata_schema="latest",
    )

    attrs = _root_attrs(out_dir)
    expected = materialize_ngff_root_attrs(payload, "latest")
    for key, value in expected.items():
        assert attrs[key] == value
    assert attrs["omero"]["name"] == "ngffzarr"


def test_write_ngff_from_tile_ts_replaces_root_attrs(tmp_path):
    tile = _make_mips(levels=1)[0]
    payload = _make_metadata_payload(dataset_count=3, name="tile-ts")
    out_dir = tmp_path / "tile-ts.ome.zarr"

    write_ngff_from_tile_ts(
        tile,
        out_dir,
        base_px_um_xy=(0.5, 1.0),
        num_mips=3,
        ngff_metadata=payload,
        metadata_schema="v0.4",
    )

    attrs = _root_attrs(out_dir)
    expected = materialize_ngff_root_attrs(payload, "v0.4")
    assert attrs["multiscales"] == expected["multiscales"]
    assert attrs["omero"]["channels"][2]["label"] == "label_2"


def test_write_ngff_from_tile_streaming_ome_uses_shared_metadata_path(tmp_path):
    tile = da.from_array(_make_mips(levels=1)[0], chunks=(8, 8, 3))
    payload = _make_metadata_payload(dataset_count=3, name="tile-stream")
    out_dir = tmp_path / "tile-stream.ome.zarr"

    write_ngff_from_tile_streaming_ome(
        tile,
        out_dir,
        phys_xy_um=(1.0, 1.0),
        num_mips=3,
        ngff_metadata=payload,
        metadata_schema="latest",
    )

    attrs = _root_attrs(out_dir)
    expected = materialize_ngff_root_attrs(payload, "latest")
    for key, value in expected.items():
        assert attrs[key] == value
    assert zarr.open_group(str(out_dir), mode="r")["s0"].shape == (3, 16, 12)


def test_phys_xy_reader_handles_latest_writer_output(tmp_path):
    mips = _make_mips(levels=2)
    payload = _make_metadata_payload(dataset_count=2)
    out_dir = tmp_path / "latest-reader.ome.zarr"

    write_ngff_from_mips(
        mips,
        out_dir,
        ngff_metadata=payload,
        metadata_schema="latest",
    )

    root = zarr.open_group(str(out_dir), mode="r")
    assert _phys_xy_um(root) == (0.25, 0.5)
