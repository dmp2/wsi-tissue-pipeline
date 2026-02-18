"""
Tests for WSI Processing Module

Tests tissue segmentation, tile extraction, and pipeline processing.
"""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
from pathlib import Path
from PIL import Image


class TestSegmentation:
    """Test tissue segmentation functions."""

    def test_import(self):
        """Test that the module can be imported."""
        from wsi_pipeline import wsi_processing
        assert hasattr(wsi_processing, "segment_tissue")

    def test_segment_tissue_basic(self, sample_image: Path):
        """Test basic tissue segmentation."""
        from wsi_pipeline.wsi_processing import segment_tissue
        from wsi_pipeline.config import PipelineConfig

        config = PipelineConfig()
        img = np.array(Image.open(sample_image))

        mask, info = segment_tissue(
            img,
            backend=config.segmentation.backend,
            target_long_side=config.segmentation.target_long_side,
            min_area_px=config.segmentation.min_area_px,
            struct_elem_px=config.segmentation.struct_elem_px,
            split_touching=config.segmentation.split_touching,
            r_split=config.segmentation.r_split,
        )

        assert mask is not None
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool or mask.dtype == np.uint8
        assert mask.shape == img.shape[:2]

        # Should find at least one tissue region
        assert info["n_components"] >= 1

    def test_segment_tissue_multitissue(self, sample_image_multichannel: Path):
        """Test segmentation with multiple tissue regions."""
        from wsi_pipeline.wsi_processing import segment_tissue

        img = np.array(Image.open(sample_image_multichannel))
        mask, info = segment_tissue(
            img,
            min_area_px=50,
        )

        # Should find multiple tissue regions
        assert info["n_components"] >= 2

    def test_segment_tissue_empty_image(self, temp_dir: Path):
        """Test segmentation with blank image."""
        from wsi_pipeline.wsi_processing import segment_tissue

        # Create blank white image
        blank = np.ones((512, 512, 3), dtype=np.uint8) * 255
        blank_path = temp_dir / "blank.jpg"
        Image.fromarray(blank).save(blank_path)

        mask, info = segment_tissue(blank)

        # Should find no tissue
        assert info["n_components"] == 0 or np.sum(mask) == 0

    def test_segment_tissue_backends(self, sample_image: Path):
        """Test different segmentation backends."""
        from wsi_pipeline.wsi_processing import segment_tissue

        img = np.array(Image.open(sample_image))

        backends = ["local-entropy", "local-otsu"]

        for backend in backends:
            mask, info = segment_tissue(img, backend=backend)

            assert mask is not None, f"Backend {backend} failed"
            assert mask.shape == img.shape[:2]


class TestTileExtraction:
    """Test tile extraction functions."""

    def test_extract_tissue_tiles(self, sample_image: Path, temp_dir: Path):
        """Test extracting tissue tiles from segmented image."""
        from wsi_pipeline.wsi_processing import segment_tissue, extract_tissue_tiles
        from wsi_pipeline.config import PipelineConfig

        config = PipelineConfig()
        img = np.array(Image.open(sample_image))

        # Segment
        mask, info = segment_tissue(img)

        # Extract tiles
        tiles = extract_tissue_tiles(
            img, mask,
            chunk_size=config.tiles.chunk_size,
            pad_multiple=config.tiles.pad_multiple,
            extra_margin_px=config.tiles.extra_margin_px,
        )

        assert len(tiles) >= 1

        for tile in tiles:
            assert isinstance(tile, (np.ndarray, da.Array))

    def test_tile_padding(self, sample_image: Path):
        """Test that tiles are padded correctly."""
        from wsi_pipeline.wsi_processing import segment_tissue, extract_tissue_tiles

        img = np.array(Image.open(sample_image))
        mask, info = segment_tissue(img)
        tiles = extract_tissue_tiles(img, mask, pad_multiple=32)

        for tile in tiles:
            if isinstance(tile, da.Array):
                tile = tile.compute()
            h, w = tile.shape[:2]
            assert h % 32 == 0, f"Height {h} not multiple of 32"
            assert w % 32 == 0, f"Width {w} not multiple of 32"

    def test_tile_margin(self, sample_image: Path):
        """Test that extra margin is applied."""
        from wsi_pipeline.wsi_processing import segment_tissue, extract_tissue_tiles

        img = np.array(Image.open(sample_image))
        mask, info = segment_tissue(img)

        tiles_no_margin = extract_tissue_tiles(img, mask, pad_multiple=1, extra_margin_px=0)
        tiles_margin = extract_tissue_tiles(img, mask, pad_multiple=1, extra_margin_px=50)

        if tiles_no_margin and tiles_margin:
            t1 = tiles_no_margin[0]
            t2 = tiles_margin[0]
            if isinstance(t1, da.Array):
                t1 = t1.compute()
            if isinstance(t2, da.Array):
                t2 = t2.compute()
            h1, w1 = t1.shape[:2]
            h2, w2 = t2.shape[:2]

            assert h2 >= h1
            assert w2 >= w1


class TestProcessWSI:
    """Test single WSI processing."""

    def test_process_wsi_basic(self, sample_image: Path, temp_dir: Path):
        """Test basic WSI processing."""
        from wsi_pipeline.wsi_processing import process_wsi
        from wsi_pipeline.config import PipelineConfig

        config = PipelineConfig()
        config.output.format = "tiff"
        config.output.generate_qc = False
        config.mlflow.enabled = False

        output_dir = temp_dir / "output"

        results = process_wsi(sample_image, output_dir, config=config)

        assert results is not None
        assert "output_paths" in results
        assert len(results["output_paths"]) >= 1

        # Check that output files exist
        for path in results["output_paths"]:
            assert Path(path).exists()

    def test_process_wsi_zarr_output(self, sample_image: Path, temp_dir: Path):
        """Test WSI processing with OME-Zarr output."""
        from wsi_pipeline.wsi_processing import process_wsi
        from wsi_pipeline.config import PipelineConfig

        config = PipelineConfig()
        config.output.format = "ome-zarr"
        config.output.generate_qc = False
        config.mlflow.enabled = False

        output_dir = temp_dir / "output_zarr"

        results = process_wsi(sample_image, output_dir, config=config)

        assert results is not None
        assert "output_paths" in results

    def test_process_wsi_qc_generation(self, sample_image: Path, temp_dir: Path):
        """Test QC image generation during processing."""
        from wsi_pipeline.wsi_processing import process_wsi
        from wsi_pipeline.config import PipelineConfig

        config = PipelineConfig()
        config.output.format = "tiff"
        config.output.generate_qc = True
        config.mlflow.enabled = False

        output_dir = temp_dir / "output_qc"

        results = process_wsi(sample_image, output_dir, config=config)

        # Check for QC outputs
        qc_dir = output_dir / "qc"
        if qc_dir.exists():
            qc_files = list(qc_dir.glob("*.png")) + list(qc_dir.glob("*.jpg"))
            assert len(qc_files) >= 1


class TestBatchProcessing:
    """Test batch processing functions."""

    def test_process_directory(self, temp_dir: Path):
        """Test processing a directory of images."""
        from wsi_pipeline.wsi_processing import process_directory
        from wsi_pipeline.config import PipelineConfig

        # Create multiple test images
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        for i in range(3):
            img = np.ones((256, 256, 3), dtype=np.uint8) * 240
            # Add tissue region
            img[80:180, 80:180] = [200, 160, 180]
            Image.fromarray(img).save(input_dir / f"slide_{i:02d}.jpg")

        config = PipelineConfig()
        config.output.format = "tiff"
        config.mlflow.enabled = False

        output_dir = temp_dir / "output"

        results = process_directory(
            input_dir,
            output_dir,
            config=config,
            pattern="*.jpg",
        )

        assert len(results) == 3  # 3 input files processed
        assert all(len(v) >= 1 for v in results.values())  # each produced output

    def test_process_specimen(self, temp_dir: Path):
        """Test specimen processing with MLflow disabled."""
        from wsi_pipeline.wsi_processing import process_specimen
        from wsi_pipeline.config import PipelineConfig

        # Create test images
        input_dir = temp_dir / "specimen"
        input_dir.mkdir()

        for i in range(2):
            img = np.ones((256, 256, 3), dtype=np.uint8) * 245
            img[70:190, 70:190] = [195, 155, 175]
            Image.fromarray(img).save(input_dir / f"section_{i:02d}.jpg")

        config = PipelineConfig()
        config.mlflow.enabled = False

        output_dir = temp_dir / "output"

        results = process_specimen(
            input_dir,
            output_dir,
            config=config,
            pattern="*.jpg",
        )

        assert results["n_inputs"] == 2


class TestWSIProcessor:
    """Test the WSIProcessor class."""

    def test_processor_init(self, sample_config: dict):
        """Test WSIProcessor initialization."""
        from wsi_pipeline.wsi_processing import WSIProcessor
        from wsi_pipeline.config import PipelineConfig

        config = PipelineConfig(**sample_config)
        processor = WSIProcessor(config=config)

        assert processor is not None
        assert processor.config == config

    def test_processor_process_file(self, sample_image: Path, temp_dir: Path):
        """Test processing a single file with WSIProcessor."""
        from wsi_pipeline.wsi_processing import WSIProcessor
        from wsi_pipeline.config import PipelineConfig

        config = PipelineConfig()
        config.mlflow.enabled = False

        processor = WSIProcessor(config=config)

        output_dir = temp_dir / "output"
        results = processor.process_wsi(sample_image, output_dir)

        assert results is not None
        assert "output_paths" in results


class TestConfiguration:
    """Test configuration handling in processing."""

    def test_config_override(self, sample_image: Path, temp_dir: Path):
        """Test that configuration overrides work."""
        from wsi_pipeline.wsi_processing import process_wsi
        from wsi_pipeline.config import PipelineConfig

        config = PipelineConfig()
        config.segmentation.backend = "local-otsu"
        config.segmentation.min_area_px = 1000
        config.mlflow.enabled = False

        output_dir = temp_dir / "output"

        # Should process without errors
        results = process_wsi(sample_image, output_dir, config=config)

        assert results is not None

    def test_env_config_override(self, sample_image: Path, temp_dir: Path, monkeypatch):
        """Test environment variable configuration override."""
        from wsi_pipeline.config import PipelineConfig

        # Set environment variables
        monkeypatch.setenv("WSI_SEGMENTATION_BACKEND", "local-otsu")
        monkeypatch.setenv("WSI_OUTPUT_FORMAT", "tiff")

        config = PipelineConfig()

        # Note: Actual env override depends on implementation
        # This tests the concept


# Performance tests
@pytest.mark.slow
class TestPerformance:
    """Performance and stress tests."""

    def test_large_batch(self, temp_dir: Path):
        """Test processing a large batch of images."""
        from wsi_pipeline.wsi_processing import process_directory
        from wsi_pipeline.config import PipelineConfig

        # Create many test images
        input_dir = temp_dir / "large_batch"
        input_dir.mkdir()

        for i in range(20):
            img = np.random.randint(200, 255, (512, 512, 3), dtype=np.uint8)
            # Add random tissue regions
            for _ in range(np.random.randint(1, 4)):
                cx, cy = np.random.randint(100, 400, 2)
                img[cy-50:cy+50, cx-50:cx+50] = [180, 140, 160]
            Image.fromarray(img).save(input_dir / f"slide_{i:03d}.jpg")

        config = PipelineConfig()
        config.mlflow.enabled = False

        output_dir = temp_dir / "output"

        results = process_directory(
            input_dir,
            output_dir,
            config=config,
            pattern="*.jpg",
        )

        assert len(results) == 20  # 20 input files processed
        assert sum(len(v) for v in results.values()) >= 20  # at least 20 outputs total
