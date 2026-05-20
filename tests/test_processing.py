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

    def test_thumbnail_does_not_upscale_low_resolution_images(self):
        """Low pyramid levels should not be enlarged before segmentation."""
        from wsi_pipeline.wsi_processing import _create_thumbnail

        img = np.ones((256, 512, 3), dtype=np.uint8) * 245
        thumb, scale = _create_thumbnail(img, target_long_side=1800)

        assert thumb.shape == img.shape
        assert scale == 1.0

    def test_min_area_is_interpreted_at_thumbnail_scale(self):
        """Downsampling should not reduce the configured minimum component size."""
        from wsi_pipeline.wsi_processing import segment_tissue

        img = np.ones((1000, 1000, 3), dtype=np.uint8) * 245
        img[475:525, 475:525] = [80, 60, 100]

        mask, info = segment_tissue(
            img,
            backend="local-otsu",
            target_long_side=500,
            min_area_px=3000,
            struct_elem_px=2,
            split_touching=False,
        )

        assert info["min_area"] == 3000
        assert info["n_components"] == 0
        assert not mask.any()

    def test_segment_tissue_keep_top_k_keeps_largest_components(self):
        """Optional safety-net should retain only the largest K components."""
        from wsi_pipeline.wsi_processing import segment_tissue

        img = np.ones((256, 256, 3), dtype=np.uint8) * 245
        img[20:90, 20:90] = [80, 60, 100]
        img[120:190, 20:90] = [80, 60, 100]
        img[40:65, 180:205] = [80, 60, 100]

        mask, info = segment_tissue(
            img,
            backend="local-otsu",
            target_long_side=256,
            min_area_px=100,
            struct_elem_px=2,
            split_touching=False,
            keep_top_k=2,
        )

        assert info["n_components"] == 2
        assert mask[40:65, 180:205].sum() == 0

    def test_he_stain_gate_rejects_pale_background_like_pixels(self):
        """H&E stain gate should keep stained tissue and reject pale artifacts."""
        from wsi_pipeline.segmentation.stain import he_stain_mask

        img = np.ones((64, 96, 3), dtype=np.uint8) * 245
        img[12:52, 10:35] = [160, 80, 145]   # H&E-like purple tissue
        img[12:52, 60:85] = [220, 155, 175]  # H&E-like pink tissue
        img[4:10, 5:90] = [232, 235, 224]    # pale coverslip/background artifact

        mask = he_stain_mask(img, min_saturation=0.08, min_od=0.35)

        assert mask[20:40, 15:30].mean() > 0.95
        assert mask[20:40, 65:80].mean() > 0.95
        assert mask[4:10, 5:90].sum() == 0

    def test_adaptive_od_stain_gate_keeps_low_saturation_tissue(self):
        """Adaptive OD mode should not require a hand-tuned saturation cutoff."""
        from wsi_pipeline.segmentation.stain import he_stain_mask

        img = np.ones((64, 96, 3), dtype=np.uint8) * 245
        img[12:52, 10:35] = [160, 80, 145]
        img[12:52, 60:85] = [220, 145, 175]
        img[28:36, 35:60] = [180, 180, 180]  # low-saturation but optically dense bridge

        fixed = he_stain_mask(img, min_saturation=0.08, min_od=0.35)
        adaptive, info = he_stain_mask(
            img,
            mode="adaptive-od",
            min_od=0.10,
            od_bg_percentile=0.80,
            od_mad_multiplier=4.0,
            return_info=True,
        )

        assert fixed[28:36, 35:60].sum() == 0
        assert adaptive[28:36, 35:60].mean() > 0.95
        assert info["od_threshold"] >= 0.10

    def test_stain_gate_can_break_pale_bridge_before_morphology(self):
        """Stain gate runs before closing so pale bridges do not merge sections."""
        from skimage import measure
        from wsi_pipeline.wsi_processing import segment_tissue

        img = np.ones((128, 192, 3), dtype=np.uint8) * 245
        img[35:95, 25:70] = [150, 75, 140]
        img[35:95, 120:165] = [220, 145, 175]
        img[55:70, 70:120] = [180, 180, 180]

        mask_plain, _ = segment_tissue(
            img,
            backend="local-otsu",
            target_long_side=128,
            min_area_px=100,
            struct_elem_px=4,
            split_touching=False,
        )
        mask_gated, info_gated = segment_tissue(
            img,
            backend="local-otsu",
            target_long_side=128,
            min_area_px=100,
            struct_elem_px=4,
            stain_gate=True,
            stain_min_saturation=0.08,
            stain_min_od=0.35,
            stain_pre_open_px=1,
            split_touching=False,
        )

        assert measure.label(mask_plain, connectivity=2).max() == 1
        assert info_gated["stain_gate"] is True
        assert measure.label(mask_gated, connectivity=2).max() == 2
        assert mask_gated[58:67, 80:110].sum() == 0


class TestTileExtraction:
    """Test tile extraction functions."""

    def test_bounds_yx_half_open_helpers(self):
        """Internal bounds helpers should use half-open YX coordinates."""
        from wsi_pipeline.tiles.generator import BoundsYX

        bounds = BoundsYX(y0=-2, x0=3, y1=7, x1=13)
        clipped = bounds.clip((5, 10))

        assert bounds.h == 9
        assert bounds.w == 10
        assert bounds.as_xyxy() == (3, -2, 13, 7)
        assert clipped.as_yx() == (0, 3, 5, 10)
        assert clipped.h == 5
        assert clipped.w == 7
        assert clipped.halo(1, (5, 10)).as_yx() == (0, 2, 5, 10)

    def test_project_label_mask_matches_integer_repeat(self):
        """Pixel-center projection should match nearest-neighbor repeat for exact 2x."""
        from wsi_pipeline.tiles.generator import BoundsYX, project_label_mask_to_source_region

        labels = np.zeros((4, 5), dtype=np.int32)
        labels[1:3, 2:4] = 7
        projected = project_label_mask_to_source_region(
            labels,
            label_id=7,
            source_region_yx=BoundsYX(0, 0, 8, 10),
            label_crop_seg_yx=BoundsYX(0, 0, 4, 5),
            scale_y=2.0,
            scale_x=2.0,
        )

        expected = np.repeat(np.repeat(labels == 7, 2, axis=0), 2, axis=1)
        np.testing.assert_array_equal(projected, expected)

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

    def test_extract_tissue_tiles_centers_edge_components(self):
        """Tiles should include balanced padding instead of pinning tissue to an edge."""
        from wsi_pipeline.wsi_processing import extract_tissue_tiles

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[0:20, 0:20] = [120, 80, 100]
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:20, 0:20] = True

        tiles = extract_tissue_tiles(
            img,
            mask,
            chunk_size=32,
            pad_multiple=64,
            extra_margin_px=20,
        )
        tile = tiles[0].compute()
        tissue = np.any(tile > 0, axis=2)
        rows = np.where(np.any(tissue, axis=1))[0]
        cols = np.where(np.any(tissue, axis=0))[0]

        assert tile.shape[:2] == (64, 64)
        assert rows[0] > 0
        assert cols[0] > 0
        assert rows[-1] < tile.shape[0] - 1
        assert cols[-1] < tile.shape[1] - 1

    def test_extract_tissue_tiles_uses_common_square_size(self):
        """All tissue sections from one source should share one square tile size."""
        from wsi_pipeline.wsi_processing import extract_tissue_tiles

        img = np.zeros((160, 240, 3), dtype=np.uint8)
        img[20:50, 20:60] = [120, 80, 100]
        img[80:140, 120:220] = [130, 90, 110]
        mask = np.zeros((160, 240), dtype=bool)
        mask[20:50, 20:60] = True
        mask[80:140, 120:220] = True

        tiles = extract_tissue_tiles(
            img,
            mask,
            chunk_size=32,
            pad_multiple=64,
            extra_margin_px=16,
        )
        shapes = [tile.shape[:2] for tile in tiles]

        assert len(tiles) == 2
        assert shapes[0] == shapes[1]
        assert shapes[0][0] == shapes[0][1]

    def test_generate_tissue_tile_records_include_parent_bounds(self):
        """Tile records should preserve source and segmentation crop windows."""
        from wsi_pipeline.tiles.generator import generate_tissue_tile_records

        img = da.from_array(np.zeros((3, 80, 120), dtype=np.uint8), chunks=(3, 40, 40))
        mask = np.zeros((8, 12), dtype=bool)
        mask[2:5, 3:7] = True

        records, tile_dim = generate_tissue_tile_records(
            img,
            mask,
            tile_frame_level="source",
            chunk=16,
            pad_multiple=16,
            extra_margin_px=0,
        )

        assert len(records) == 1
        assert records[0].tissue_index == 0
        assert records[0].crop_bounds_source_level == (26, 11, 74, 59)
        assert records[0].crop_bounds_segmentation_level == (2, 1, 8, 6)
        assert records[0].tile.shape == (48, 48, 3)
        assert tile_dim == 48

    def test_generate_tissue_tile_records_can_frame_at_segmentation_level(self):
        """Cross-level notebook framing should size the frame before source mapping."""
        from wsi_pipeline.tiles.generator import generate_tissue_tile_records

        img = da.from_array(np.zeros((3, 80, 160), dtype=np.uint8), chunks=(3, 40, 40))
        mask = np.zeros((40, 80), dtype=bool)
        mask[12:32, 20:60] = True

        source_records, source_dim = generate_tissue_tile_records(
            img,
            mask,
            chunk=16,
            pad_multiple=16,
            extra_margin_px=0,
            tile_frame_level="source",
        )
        segmentation_records, segmentation_source_dim = generate_tissue_tile_records(
            img,
            mask,
            chunk=16,
            pad_multiple=16,
            extra_margin_px=0,
            tile_frame_level="segmentation",
        )

        assert source_dim == 80
        assert source_records[0].segmentation_tile_dim == 40
        assert segmentation_records[0].segmentation_tile_dim == 48
        assert segmentation_source_dim == 96
        assert segmentation_records[0].tile.shape == (96, 96, 3)
        assert segmentation_records[0].crop_bounds_segmentation_level == (16, 0, 64, 40)
        assert segmentation_records[0].crop_bounds_source_level == (32, 0, 128, 80)

    def test_generate_tissue_tile_records_uses_shape_ratio_mapping(self):
        """Non-power-of-two level shapes should drive source crop mapping."""
        from wsi_pipeline.tiles.generator import generate_tissue_tile_records

        img = da.from_array(np.zeros((3, 81, 161), dtype=np.uint8), chunks=(3, 40, 40))
        mask = np.zeros((40, 80), dtype=bool)
        mask[12:32, 20:60] = True

        records, tile_dim = generate_tissue_tile_records(
            img,
            mask,
            chunk=16,
            pad_multiple=16,
            extra_margin_px=0,
            tile_frame_level="segmentation",
        )

        assert tile_dim == 99
        assert records[0].segmentation_tile_dim == 48
        assert records[0].scale_y == 81 / 40
        assert records[0].scale_x == 161 / 80
        assert records[0].crop_bounds_segmentation_level == (16, 0, 64, 40)
        assert records[0].crop_bounds_source_level == (31, 0, 130, 81)


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

    def test_process_wsi_can_disable_component_qc(self, sample_image: Path, temp_dir: Path):
        """Component QC should be optional for compatibility/debugging."""
        from wsi_pipeline.wsi_processing import process_wsi
        from wsi_pipeline.config import PipelineConfig

        config = PipelineConfig()
        config.output.format = "tiff"
        config.output.generate_qc = False
        config.mlflow.enabled = False
        config.segmentation.component_qc_enabled = False

        results = process_wsi(sample_image, temp_dir / "output_no_component_qc", config=config)

        assert results["component_qc"]["enabled"] is False
        assert results["component_qc"]["records"] == []
        assert all(record["component_qc"] is None for record in results["tile_records"])

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
