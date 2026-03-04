"""Tests for VSI/ETS to OME-TIFF conversion paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest


class _DummyETSForOmeTiff:
    """Minimal ETSFile stand-in for OME-TIFF writer tests."""

    nlevels = 3

    def __init__(self, fname: str | Path):
        self.fname = Path(fname)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def read_level(self, level: int) -> np.ndarray:
        height = max(1, 32 // (2**level))
        width = max(1, 48 // (2**level))
        return np.full((height, width, 3), fill_value=level, dtype=np.uint8)


class _WriterSpy:
    """Capture tifffile.TiffWriter writes without touching disk format internals."""

    instances: list[_WriterSpy] = []

    def __init__(self, path: str, *, bigtiff: bool):
        self.path = Path(path)
        self.bigtiff = bigtiff
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        _WriterSpy.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.closed = True
        return False

    def write(self, image: np.ndarray, **kwargs):
        self.calls.append({"shape": tuple(image.shape), "kwargs": dict(kwargs)})


def _build_vsi_tree(temp_dir: Path) -> tuple[Path, Path]:
    """Create a minimal VSI + ETS directory structure."""
    vsi_file = temp_dir / "sample.vsi"
    vsi_file.touch()

    ets_dir = temp_dir / "_sample_" / "stack10002"
    ets_dir.mkdir(parents=True)
    ets_file = ets_dir / "frame_t.ets"
    ets_file.touch()
    return vsi_file, ets_file


def _mock_vsi_metadata(*, channel_count: int = 3) -> dict[str, Any]:
    """Return deterministic metadata payload compatible with get_vsi_metadata()."""
    return {
        "channel_count": channel_count,
        "channel_labels": ["red", "green", "blue"][:channel_count],
        "physical_pixel_size_um": {"x": 0.25, "y": 0.5, "z": 2.0},
    }


def test_vsi_to_ome_tiff_returns_none_when_ets_missing(temp_dir: Path):
    """Missing ETS should short-circuit conversion."""
    from wsi_pipeline import vsi_converter

    vsi_file = temp_dir / "missing_stack.vsi"
    vsi_file.touch()

    result = vsi_converter.vsi_to_ome_tiff(vsi_file, temp_dir / "out.tif")
    assert result is None


def test_ets_to_ome_tiff_writes_pyramidal_levels(monkeypatch: pytest.MonkeyPatch, temp_dir: Path):
    """Writer should emit level 0 with SubIFDs and lower levels as reduced pages."""
    from wsi_pipeline import vsi_converter

    _WriterSpy.instances.clear()
    ets_file = temp_dir / "sample.ets"
    ets_file.touch()

    monkeypatch.setattr(vsi_converter, "ETSFile", _DummyETSForOmeTiff)
    monkeypatch.setattr(vsi_converter.tifffile, "TiffWriter", _WriterSpy)

    output_path = vsi_converter.ets_to_ome_tiff(
        ets_file,
        temp_dir / "result.tif",
        metadata_backend="ets_only",
    )

    assert output_path is not None
    assert output_path.name.endswith(".ome.tif")
    assert len(_WriterSpy.instances) == 1

    writer = _WriterSpy.instances[0]
    assert writer.path == output_path
    assert writer.bigtiff is True
    assert writer.closed is True
    assert len(writer.calls) == 3

    first_call = writer.calls[0]["kwargs"]
    assert first_call["subifds"] == 2
    assert first_call["metadata"]["axes"] == "YXS"
    assert "subfiletype" not in first_call

    for reduced in writer.calls[1:]:
        assert reduced["kwargs"]["subfiletype"] == 1


def test_ets_to_ome_tiff_uses_vsi_metadata(
    monkeypatch: pytest.MonkeyPatch,
    temp_dir: Path,
):
    """Metadata payload should be translated into OME writer metadata fields."""
    from wsi_pipeline import vsi_converter

    _WriterSpy.instances.clear()
    vsi_file, ets_file = _build_vsi_tree(temp_dir)

    monkeypatch.setattr(vsi_converter, "ETSFile", _DummyETSForOmeTiff)
    monkeypatch.setattr(vsi_converter.tifffile, "TiffWriter", _WriterSpy)
    monkeypatch.setattr(
        vsi_converter,
        "get_vsi_metadata",
        lambda *args, **kwargs: _mock_vsi_metadata(),
    )

    result = vsi_converter.ets_to_ome_tiff(
        ets_file,
        temp_dir / "metadata_target.ome.tif",
        vsi_fname=vsi_file,
        metadata_backend="auto",
    )

    assert result is not None
    first_call = _WriterSpy.instances[0].calls[0]["kwargs"]
    metadata = first_call["metadata"]
    assert metadata["PhysicalSizeX"] == 0.25
    assert metadata["PhysicalSizeY"] == 0.5
    assert metadata["Channel"]["Name"] == ["red", "green", "blue"]


def test_ets_to_ome_tiff_auto_falls_back_to_minimal_metadata(
    monkeypatch: pytest.MonkeyPatch,
    temp_dir: Path,
    caplog: pytest.LogCaptureFixture,
):
    """Auto backend should continue writing when VSI metadata lookup fails."""
    from wsi_pipeline import vsi_converter

    _WriterSpy.instances.clear()
    vsi_file, ets_file = _build_vsi_tree(temp_dir)

    monkeypatch.setattr(vsi_converter, "ETSFile", _DummyETSForOmeTiff)
    monkeypatch.setattr(vsi_converter.tifffile, "TiffWriter", _WriterSpy)
    monkeypatch.setattr(
        vsi_converter,
        "get_vsi_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Bio-Formats unavailable")),
    )

    with caplog.at_level("WARNING"):
        result = vsi_converter.ets_to_ome_tiff(
            ets_file,
            temp_dir / "fallback.tif",
            vsi_fname=vsi_file,
            metadata_backend="auto",
        )

    assert result is not None
    assert any("writing OME-TIFF with minimal metadata" in msg for msg in caplog.messages)
    assert _WriterSpy.instances[0].calls[0]["kwargs"]["metadata"] == {"axes": "YXS"}


def test_ets_to_ome_tiff_rejects_unsupported_channel_count(
    monkeypatch: pytest.MonkeyPatch,
    temp_dir: Path,
):
    """Unsupported non-RGB channel metadata should raise immediately."""
    from wsi_pipeline import vsi_converter

    vsi_file, ets_file = _build_vsi_tree(temp_dir)
    monkeypatch.setattr(vsi_converter, "ETSFile", _DummyETSForOmeTiff)
    monkeypatch.setattr(vsi_converter.tifffile, "TiffWriter", _WriterSpy)
    monkeypatch.setattr(
        vsi_converter,
        "get_vsi_metadata",
        lambda *args, **kwargs: _mock_vsi_metadata(channel_count=4),
    )

    with pytest.raises(ValueError, match="Unsupported VSI channel count 4"):
        vsi_converter.ets_to_ome_tiff(
            ets_file,
            temp_dir / "bad_channels.tif",
            vsi_fname=vsi_file,
            metadata_backend="auto",
        )


def test_vsi_to_ome_tiff_normalizes_output_suffix(
    monkeypatch: pytest.MonkeyPatch,
    temp_dir: Path,
):
    """VSI entrypoint should enforce the `.ome.tif` suffix."""
    from wsi_pipeline import vsi_converter

    vsi_file, _ = _build_vsi_tree(temp_dir)
    observed: list[Path] = []

    def _fake_ets_to_ome_tiff(*args, **kwargs):
        output_path = Path(args[1])
        observed.append(output_path)
        return output_path

    monkeypatch.setattr(vsi_converter, "ets_to_ome_tiff", _fake_ets_to_ome_tiff)

    result = vsi_converter.vsi_to_ome_tiff(vsi_file, temp_dir / "normalized.tiff")
    assert result is not None
    assert result.name == "normalized.ome.tif"
    assert observed[0].name == "normalized.ome.tif"


def test_batch_convert_vsi_routes_ome_tiff_and_emits_ome_suffix(
    monkeypatch: pytest.MonkeyPatch,
    temp_dir: Path,
):
    """Batch conversion should call the OME-TIFF entrypoint and produce `.ome.tif` outputs."""
    from wsi_pipeline import vsi_converter

    for name in ("one.vsi", "two.vsi"):
        (temp_dir / name).touch()

    observed_outputs: list[Path] = []

    def _fake_vsi_to_ome_tiff(vsi_fname, output_path, **kwargs):  # noqa: ARG001
        output = Path(output_path)
        observed_outputs.append(output)
        return output

    monkeypatch.setattr(vsi_converter, "vsi_to_ome_tiff", _fake_vsi_to_ome_tiff)

    results = vsi_converter.batch_convert_vsi(
        str(temp_dir / "*.vsi"),
        temp_dir / "out",
        format="ome-tiff",
    )

    assert len(results) == 2
    assert all(path.name.endswith(".ome.tif") for path in observed_outputs)
    assert all(path.name.endswith(".ome.tif") for path in results)


def test_output_config_accepts_ome_tiff_alias():
    """`ome-tiff` should be a valid output-format literal."""
    from wsi_pipeline.config import OutputConfig

    cfg = OutputConfig(format="ome-tiff")
    assert cfg.format == "ome-tiff"
