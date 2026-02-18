"""
Tests for ETS File Reader

Tests the Olympus ETS file format reader functionality.
Note: Most tests use mock data since real ETS files are proprietary.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path


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
