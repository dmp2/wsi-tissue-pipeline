from __future__ import annotations

import json

import numpy as np

from wsi_pipeline.neuroglancer import (
    emit_ng_state_for_ngff_plate,
    find_zarr_children,
    get_zarr_physical_scale,
)
from wsi_pipeline.omezarr import write_ngff_from_mips
from wsi_pipeline.omezarr.metadata import _is_ngff_image_group


def _make_mips(levels: int = 2, channels: int = 3) -> list[np.ndarray]:
    """Build a small synthetic YXC pyramid."""
    base = np.arange(16 * 12 * channels, dtype=np.uint8).reshape(16, 12, channels)
    mips: list[np.ndarray] = []
    current = base
    for _ in range(levels):
        mips.append(current.copy())
        current = current[::2, ::2, :]
    return mips


def test_is_ngff_image_group_detects_writer_output(tmp_path):
    out_dir = tmp_path / "sample.ome.zarr"
    write_ngff_from_mips(_make_mips(levels=2), out_dir, phys_xy_um=(0.25, 0.5))

    assert _is_ngff_image_group(out_dir)
    assert not _is_ngff_image_group(tmp_path / "does-not-exist.ome.zarr")


def test_neuroglancer_helpers_use_ngff_group_detection(tmp_path):
    plate_root = tmp_path / "plate"
    plate_root.mkdir(parents=True)

    valid = plate_root / "good.ome.zarr"
    write_ngff_from_mips(_make_mips(levels=2), valid, phys_xy_um=(0.25, 0.5))

    # Looks zarr-ish, but not a valid NGFF image group.
    invalid = plate_root / "bad.ome.zarr"
    invalid.mkdir()
    (invalid / ".zattrs").write_text("{}")

    children = find_zarr_children(plate_root)
    assert children == [valid]
    assert get_zarr_physical_scale(valid) == (0.25, 0.5)

    state_path = emit_ng_state_for_ngff_plate(
        plate_root=plate_root,
        base_http_url="http://localhost:8000",
        out_state_path=tmp_path / "state.json",
    )
    state = json.loads(state_path.read_text())
    assert len(state["layers"]) == 1
    assert state["layers"][0]["source"].endswith(valid.name)
