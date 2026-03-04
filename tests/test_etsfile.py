"""
Tests for ETS File Reader

Tests the Olympus ETS file format reader functionality.
Note: Most tests use mock data since real ETS files are proprietary.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path


class DummyETSFile:
    """Minimal ETSFile stand-in for VSI metadata tests."""

    npix_x = 4096
    npix_y = 2048
    nlevels = 3
    ntiles = 21
    tile_xsize = 512
    tile_ysize = 512
    compression_str = "JPEG"
    is_bgr = False
    fsize = 123456

    def __init__(self, fname: str | Path):
        self.fname = Path(fname)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def level_shape(self, level: int) -> tuple[int, int]:
        return (self.npix_y // (2**level), self.npix_x // (2**level))

    def level_ntiles(self, level: int) -> tuple[int, int]:
        return (max(1, 8 // (2**level)), max(1, 4 // (2**level)))


def _build_mock_vsi_tree(temp_dir: Path) -> tuple[Path, Path]:
    """Create a minimal VSI + ETS directory structure."""
    vsi_file = temp_dir / "slide.vsi"
    vsi_file.touch()

    ets_dir = temp_dir / "_slide_" / "stack10002"
    ets_dir.mkdir(parents=True)
    ets_file = ets_dir / "frame_t.ets"
    ets_file.touch()
    return vsi_file, ets_file


def _mock_vendor_metadata(
    *,
    size_c: int = 3,
    include_stage_origin: bool = True,
    include_physical_sizes: bool = True,
) -> dict:
    """Return deterministic mock Bio-Formats metadata for VSI tests."""
    return {
        "name": "slide.vsi",
        "sizeX": 4096,
        "sizeY": 2048,
        "sizeZ": 1,
        "sizeC": size_c,
        "sizeT": 1,
        "physical_size_x": 0.25 if include_physical_sizes else None,
        "physical_size_y": 0.5 if include_physical_sizes else None,
        "physical_size_z": 2.0 if include_physical_sizes else None,
        "channel_labels": ["red", "green", "blue"][:size_c],
        "stage_origin_um": (
            {"x": 10.0, "y": 20.0, "z": 30.0} if include_stage_origin else None
        ),
    }


class TestETSFileBasics:
    """Test basic ETS file operations."""
    
    def test_import(self):
        """Test that the module can be imported."""
        from wsi_pipeline import etsfile
        assert hasattr(etsfile, "ETSFile")
    
    def test_etsfile_init_missing_file(self, temp_dir: Path):
        """Test that missing files raise appropriate errors."""
        from wsi_pipeline.etsfile import ETSFile
        
        missing_path = temp_dir / "nonexistent.ets"
        
        with pytest.raises(FileNotFoundError):
            ETSFile(missing_path)
    
    def test_etsfile_init_invalid_file(self, mock_ets_file: Path):
        """Test handling of invalid ETS files."""
        from wsi_pipeline.etsfile import ETSFile
        
        # Our mock file doesn't have valid ETS structure
        # Real implementation should raise an error
        with pytest.raises(Exception):  # Could be ValueError or custom exception
            ETSFile(mock_ets_file)


class TestETSFileMetadata:
    """Test ETS file metadata extraction."""
    
    @pytest.mark.skip(reason="Requires real ETS file")
    def test_dimensions(self, real_ets_file: Path):
        """Test that dimensions are correctly parsed."""
        from wsi_pipeline.etsfile import ETSFile
        
        ets = ETSFile(real_ets_file)
        
        assert ets.width > 0
        assert ets.height > 0
        assert ets.num_levels >= 1
    
    @pytest.mark.skip(reason="Requires real ETS file")
    def test_pixel_size(self, real_ets_file: Path):
        """Test that pixel size is correctly parsed."""
        from wsi_pipeline.etsfile import ETSFile
        
        ets = ETSFile(real_ets_file)
        
        # Pixel size should be in micrometers
        assert ets.pixel_size_um > 0
        assert ets.pixel_size_um < 10  # Typical range


class TestETSFileTileAccess:
    """Test tile access functionality."""
    
    @pytest.mark.skip(reason="Requires real ETS file")
    def test_get_tile(self, real_ets_file: Path):
        """Test reading a single tile."""
        from wsi_pipeline.etsfile import ETSFile
        
        ets = ETSFile(real_ets_file)
        
        tile = ets.get_tile(level=0, row=0, col=0)
        
        assert tile is not None
        assert isinstance(tile, np.ndarray)
        assert tile.ndim == 3  # H, W, C
        assert tile.dtype == np.uint8 or tile.dtype == np.uint16
    
    @pytest.mark.skip(reason="Requires real ETS file")
    def test_get_region(self, real_ets_file: Path):
        """Test reading a region."""
        from wsi_pipeline.etsfile import ETSFile
        
        ets = ETSFile(real_ets_file)
        
        region = ets.get_region(
            level=0,
            x=0, y=0,
            width=512, height=512,
        )
        
        assert region is not None
        assert region.shape[:2] == (512, 512)


class TestETSFileDaskIntegration:
    """Test Dask array integration."""
    
    @pytest.mark.skip(reason="Requires real ETS file")
    def test_to_dask_array(self, real_ets_file: Path):
        """Test conversion to Dask array."""
        from wsi_pipeline.etsfile import ETSFile
        import dask.array as da
        
        ets = ETSFile(real_ets_file)
        
        dask_arr = ets.to_dask_array(level=0)
        
        assert isinstance(dask_arr, da.Array)
        assert dask_arr.ndim == 3
    
    @pytest.mark.skip(reason="Requires real ETS file")
    def test_dask_array_compute(self, real_ets_file: Path):
        """Test computing a region from Dask array."""
        from wsi_pipeline.etsfile import ETSFile
        
        ets = ETSFile(real_ets_file)
        dask_arr = ets.to_dask_array(level=0)
        
        # Compute a small region
        region = dask_arr[:512, :512, :].compute()
        
        assert isinstance(region, np.ndarray)
        assert region.shape == (512, 512, 3) or region.shape == (512, 512, 4)


class TestETSFilePyramid:
    """Test pyramid level access."""
    
    @pytest.mark.skip(reason="Requires real ETS file")
    def test_level_dimensions(self, real_ets_file: Path):
        """Test that level dimensions follow pyramid structure."""
        from wsi_pipeline.etsfile import ETSFile
        
        ets = ETSFile(real_ets_file)
        
        for level in range(ets.num_levels):
            dims = ets.get_level_dimensions(level)
            assert dims[0] > 0 and dims[1] > 0
            
            if level > 0:
                prev_dims = ets.get_level_dimensions(level - 1)
                # Each level should be roughly half the previous
                assert dims[0] <= prev_dims[0]
                assert dims[1] <= prev_dims[1]
    
    @pytest.mark.skip(reason="Requires real ETS file")
    def test_get_thumbnail(self, real_ets_file: Path):
        """Test thumbnail generation."""
        from wsi_pipeline.etsfile import ETSFile
        
        ets = ETSFile(real_ets_file)
        
        thumb = ets.get_thumbnail(max_size=1000)
        
        assert thumb is not None
        assert max(thumb.shape[:2]) <= 1000


class TestETSFileHelpers:
    """Test helper functions."""

    def test_find_ets_file(self, temp_dir: Path):
        """Test finding ETS file associated with a VSI file."""
        from wsi_pipeline.vsi_converter import find_ets_file

        # Create mock VSI + ETS directory structure
        vsi_file = temp_dir / "slide.vsi"
        vsi_file.touch()

        # ETS files live in _<vsi_stem>_/stackNNNNN/ directories
        ets_dir = temp_dir / "_slide_" / "stack10001"
        ets_dir.mkdir(parents=True)
        (ets_dir / "frame_t.ets").touch()

        result = find_ets_file(vsi_file)
        assert result is not None
        assert result.suffix == ".ets"

    def test_find_ets_file_missing(self, temp_dir: Path):
        """Test that missing VSI returns None."""
        from wsi_pipeline.vsi_converter import find_ets_file

        missing = temp_dir / "nonexistent.vsi"
        result = find_ets_file(missing)
        assert result is None

    def test_find_all_ets_files(self, temp_dir: Path):
        """Test finding all ETS files associated with a VSI file."""
        from wsi_pipeline.vsi_converter import find_all_ets_files

        vsi_file = temp_dir / "slide.vsi"
        vsi_file.touch()

        # Create multiple stack folders with ETS files
        for stack in ["stack10001", "stack10002"]:
            d = temp_dir / "_slide_" / stack
            d.mkdir(parents=True)
            (d / "frame_t.ets").touch()

        results = find_all_ets_files(vsi_file)
        assert len(results) == 2

    def test_find_ets_file_prefers_highest_numbered_stack(self, temp_dir: Path):
        """Test that the full-resolution ETS path is selected when multiple stacks exist."""
        from wsi_pipeline.vsi_converter import find_ets_file

        vsi_file = temp_dir / "slide.vsi"
        vsi_file.touch()

        for stack in ["stack10001", "stack10002", "stack10000"]:
            d = temp_dir / "_slide_" / stack
            d.mkdir(parents=True)
            (d / "frame_t.ets").touch()

        result = find_ets_file(vsi_file)
        assert result is not None
        assert result.parent.name == "stack10002"


class TestVSIMetadata:
    """Test VSI metadata projection and schema compatibility."""

    def test_get_vsi_metadata_ets_only_is_incomplete(
        self,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """ETS-only metadata should preserve structure but report incomplete NGFF transforms."""
        from wsi_pipeline import vsi_converter

        vsi_file, _ = _build_mock_vsi_tree(temp_dir)
        monkeypatch.setattr(vsi_converter, "ETSFile", DummyETSFile)

        metadata = vsi_converter.get_vsi_metadata(vsi_file, metadata_backend="ets_only")

        assert metadata["metadata_sources"]["physical"] == "ets_only"
        assert metadata["width"] == 4096
        assert metadata["source_array_order"] == ["y", "x", "c"]
        assert metadata["ngff_axis_order"] == ["c", "y", "x"]
        assert metadata["ngff"] is metadata["ngff_latest"]
        assert metadata["is_ngff_minimum_complete"] is False
        assert metadata["ngff_latest"]["coordinateSystems"][1]["name"] == "image-plane"
        assert (
            metadata["ngff_latest"]["multiscales"][0]["datasets"][0]["coordinateTransformations"]
            == []
        )
        assert "Physical pixel sizes for X and Y were not available" in metadata["warnings"][0]

    def test_get_vsi_metadata_auto_falls_back_to_ets_only(
        self,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Auto mode should fall back to ETS-only metadata when Bio-Formats is unavailable."""
        from wsi_pipeline import vsi_converter

        vsi_file, _ = _build_mock_vsi_tree(temp_dir)
        monkeypatch.setattr(vsi_converter, "ETSFile", DummyETSFile)
        monkeypatch.setattr(
            vsi_converter,
            "_extract_vsi_physical_metadata",
            lambda _path: (_ for _ in ()).throw(RuntimeError("Bio-Formats unavailable.")),
        )

        metadata = vsi_converter.get_vsi_metadata(vsi_file, metadata_backend="auto")

        assert metadata["metadata_sources"]["physical"] == "ets_only"
        assert any("Falling back to ETS-only metadata." in warning for warning in metadata["warnings"])

    def test_get_vsi_metadata_bioformats_backend_is_strict(
        self,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The explicit Bio-Formats backend should raise when metadata extraction fails."""
        from wsi_pipeline import vsi_converter

        vsi_file, _ = _build_mock_vsi_tree(temp_dir)
        monkeypatch.setattr(vsi_converter, "ETSFile", DummyETSFile)
        monkeypatch.setattr(
            vsi_converter,
            "_extract_vsi_physical_metadata",
            lambda _path: (_ for _ in ()).throw(RuntimeError("Bio-Formats unavailable.")),
        )

        with pytest.raises(RuntimeError, match="Bio-Formats unavailable"):
            vsi_converter.get_vsi_metadata(vsi_file, metadata_backend="bioformats")

    def test_get_vsi_metadata_emits_latest_and_v04_projections(
        self,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Successful VSI metadata extraction should emit both schema projections."""
        from wsi_pipeline import vsi_converter

        vsi_file, _ = _build_mock_vsi_tree(temp_dir)
        monkeypatch.setattr(vsi_converter, "ETSFile", DummyETSFile)
        monkeypatch.setattr(
            vsi_converter,
            "_extract_vsi_physical_metadata",
            lambda _path: _mock_vendor_metadata(),
        )

        metadata = vsi_converter.get_vsi_metadata(vsi_file, metadata_backend="auto")

        assert metadata["metadata_sources"]["physical"] == "bioformats"
        assert metadata["physical_pixel_size_um"] == {"x": 0.25, "y": 0.5, "z": 2.0}
        assert metadata["channel_labels"] == ["red", "green", "blue"]
        assert metadata["is_ngff_minimum_complete"] is True
        assert metadata["ngff_latest"]["arrayToPhysicalTransformations"][0]["scale"] == [
            1.0,
            0.5,
            0.25,
        ]
        assert metadata["ngff_latest"]["arrayToPhysicalTransformations"][1]["translation"] == [
            0.0,
            20.0,
            10.0,
        ]
        assert metadata["ngff_v04"]["multiscales"][0]["version"] == "0.4"
        assert metadata["ngff_v04"]["multiscales"][0]["axes"][0]["name"] == "c"
        assert (
            metadata["ngff_v04"]["multiscales"][0]["datasets"][1]["coordinateTransformations"][0][
                "scale"
            ]
            == [1.0, 1.0, 0.5]
        )
        assert metadata["compatibility"]["lossy_fields_for_v04"] == [
            "named_coordinate_systems",
            "absolute_origin_translation",
        ]

    def test_get_vsi_metadata_target_schema_aliases_projection(
        self,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The `ngff` alias should follow the requested target schema."""
        from wsi_pipeline import vsi_converter

        vsi_file, _ = _build_mock_vsi_tree(temp_dir)
        monkeypatch.setattr(vsi_converter, "ETSFile", DummyETSFile)
        monkeypatch.setattr(
            vsi_converter,
            "_extract_vsi_physical_metadata",
            lambda _path: _mock_vendor_metadata(include_stage_origin=False),
        )

        metadata = vsi_converter.get_vsi_metadata(
            vsi_file,
            metadata_backend="auto",
            target_schema="0.4",
        )

        assert metadata["compatibility"]["selected_schema"] == "v0.4"
        assert metadata["ngff"] is metadata["ngff_v04"]
        assert metadata["ngff_latest"] is not metadata["ngff_v04"]

    def test_get_vsi_metadata_rejects_unsupported_channel_count(
        self,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Unsupported channel layouts should fail fast instead of being guessed."""
        from wsi_pipeline import vsi_converter

        vsi_file, _ = _build_mock_vsi_tree(temp_dir)
        monkeypatch.setattr(vsi_converter, "ETSFile", DummyETSFile)
        monkeypatch.setattr(
            vsi_converter,
            "_extract_vsi_physical_metadata",
            lambda _path: _mock_vendor_metadata(size_c=4),
        )

        with pytest.raises(ValueError, match="Unsupported VSI channel count 4"):
            vsi_converter.get_vsi_metadata(vsi_file, metadata_backend="auto")

    def test_materialize_ngff_root_attrs_supports_full_payload_and_projection(
        self,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The OME-Zarr adapter should accept either the full payload or a direct projection."""
        from wsi_pipeline import vsi_converter
        from wsi_pipeline.omezarr import materialize_ngff_root_attrs

        vsi_file, _ = _build_mock_vsi_tree(temp_dir)
        monkeypatch.setattr(vsi_converter, "ETSFile", DummyETSFile)
        monkeypatch.setattr(
            vsi_converter,
            "_extract_vsi_physical_metadata",
            lambda _path: _mock_vendor_metadata(),
        )

        metadata = vsi_converter.get_vsi_metadata(vsi_file, metadata_backend="auto")
        v04_attrs = materialize_ngff_root_attrs(metadata, schema="v0.4")
        latest_attrs = materialize_ngff_root_attrs(metadata["ngff_latest"])

        assert v04_attrs["multiscales"][0]["version"] == "0.4"
        assert v04_attrs is not metadata["ngff_v04"]
        assert latest_attrs["schema"] == "latest"
        assert latest_attrs["multiscales"][0]["datasets"][0]["path"] == "s0"


# Integration tests (require real data)
@pytest.mark.integration
class TestETSFileIntegration:
    """Integration tests with real ETS files."""
    
    @pytest.fixture
    def real_ets_file(self):
        """Fixture to provide a real ETS file path."""
        # Look for test data in common locations
        test_paths = [
            Path("/data/test/sample.ets"),
            Path("./test_data/sample.ets"),
            Path.home() / "wsi_test_data" / "sample.ets",
        ]
        
        for path in test_paths:
            if path.exists():
                return path
        
        pytest.skip("No real ETS file available for integration testing")
    
    def test_full_pipeline_with_real_file(self, real_ets_file: Path, temp_dir: Path):
        """Test full pipeline with real ETS file."""
        from wsi_pipeline.etsfile import ETSFile
        
        ets = ETSFile(real_ets_file)
        
        # Get thumbnail
        thumb = ets.get_thumbnail(max_size=2000)
        assert thumb is not None
        
        # Get a tile
        tile = ets.get_tile(level=0, row=0, col=0)
        assert tile is not None
        
        # Convert to Dask and compute region
        dask_arr = ets.to_dask_array(level=0)
        region = dask_arr[:256, :256, :].compute()
        assert region is not None
