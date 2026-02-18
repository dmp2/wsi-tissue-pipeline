"""
Pytest Configuration and Fixtures

Provides common fixtures for testing the WSI Tissue Pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """Create a sample test image."""
    # Create a simple image with tissue-like regions
    width, height = 1024, 768
    
    # White background
    img = np.ones((height, width, 3), dtype=np.uint8) * 245
    
    # Add a tissue-like region (pink ellipse)
    y, x = np.ogrid[:height, :width]
    cx, cy = width // 2, height // 2
    rx, ry = 200, 150
    
    mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
    
    # Pink tissue color
    img[mask] = [200, 160, 180]
    
    # Save as JPEG
    output_path = temp_dir / "test_image.jpg"
    Image.fromarray(img).save(output_path, quality=95)
    
    return output_path


@pytest.fixture
def sample_image_multichannel(temp_dir: Path) -> Path:
    """Create a multi-tissue test image."""
    width, height = 2048, 1536
    
    # White background
    img = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Add multiple tissue regions
    tissues = [
        {"cx": 400, "cy": 400, "rx": 150, "ry": 120, "color": [200, 150, 170]},
        {"cx": 800, "cy": 600, "rx": 180, "ry": 140, "color": [210, 160, 180]},
        {"cx": 1400, "cy": 400, "rx": 200, "ry": 160, "color": [195, 155, 175]},
        {"cx": 1200, "cy": 1000, "rx": 170, "ry": 130, "color": [205, 165, 185]},
    ]
    
    y, x = np.ogrid[:height, :width]
    
    for t in tissues:
        mask = ((x - t["cx"]) / t["rx"]) ** 2 + ((y - t["cy"]) / t["ry"]) ** 2 <= 1
        img[mask] = t["color"]
    
    output_path = temp_dir / "test_image_multi.jpg"
    Image.fromarray(img).save(output_path, quality=95)
    
    return output_path


@pytest.fixture
def sample_config() -> dict:
    """Return a sample configuration dictionary."""
    return {
        "segmentation": {
            "backend": "local-entropy",
            "target_long_side": 1000,
            "min_area_px": 100,
            "struct_elem_px": 3,
            "split_touching": False,
            "r_split": 3.0,
        },
        "tiles": {
            "chunk_size": 512,
            "pad_multiple": 64,
            "extra_margin_px": 10,
        },
        "output": {
            "format": "tiff",
            "compression": None,
            "convert_to_uint8": True,
            "generate_qc": False,
        },
        "mlflow": {
            "enabled": False,
            "tracking_uri": "sqlite:///test_mlflow.db",
            "experiment_name": "test-experiment",
        },
    }


@pytest.fixture
def sample_zarr_dir(temp_dir: Path) -> Path:
    """Create a sample OME-Zarr directory structure."""
    import zarr
    
    zarr_path = temp_dir / "test.ome.zarr"
    
    # Create a simple zarr array
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)
    
    # Create a small test array
    data = np.random.randint(0, 255, (3, 256, 256), dtype=np.uint8)
    root.create_dataset("0", data=data, chunks=(3, 64, 64))
    
    # Add minimal OME metadata
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [
            {"name": "c", "type": "channel"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ],
        "datasets": [{"path": "0"}],
    }]
    
    return zarr_path


@pytest.fixture
def mock_ets_file(temp_dir: Path) -> Path:
    """Create a mock ETS file structure for testing."""
    # ETS files are proprietary, so we create a mock structure
    ets_path = temp_dir / "test.ets"
    
    # Create a simple file with header-like content
    # Real ETS files have a specific binary format
    with open(ets_path, "wb") as f:
        # Write a mock header
        f.write(b"ETS\x00")  # Magic bytes
        f.write(b"\x00" * 1024)  # Placeholder header
    
    return ets_path


class MockWSIReader:
    """Mock WSI reader for testing without real slide files."""
    
    def __init__(self, width: int = 4096, height: int = 3072, levels: int = 3):
        self.width = width
        self.height = height
        self.levels = levels
        self._level_dimensions = [
            (width // (2 ** i), height // (2 ** i))
            for i in range(levels)
        ]
    
    @property
    def dimensions(self) -> tuple[int, int]:
        return (self.width, self.height)
    
    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        return self._level_dimensions
    
    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        """Return a synthetic region."""
        w, h = size
        # Create a pink-ish region
        region = np.ones((h, w, 3), dtype=np.uint8) * 200
        region[:, :, 0] = 210  # R
        region[:, :, 1] = 170  # G
        region[:, :, 2] = 190  # B
        return region
    
    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        """Return a thumbnail with tissue regions."""
        w, h = size
        thumb = np.ones((h, w, 3), dtype=np.uint8) * 240
        
        # Add a central tissue region
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        mask = ((x - cx) / (w // 4)) ** 2 + ((y - cy) / (h // 4)) ** 2 <= 1
        thumb[mask] = [200, 160, 180]
        
        return thumb


@pytest.fixture
def mock_wsi_reader() -> MockWSIReader:
    """Return a mock WSI reader."""
    return MockWSIReader()
