"""
WSI Tissue Pipeline
====================

A reproducible whole-slide image (WSI) processing pipeline for
tissue section extraction and downstream analysis.

Main Components:
- ETSFile: Reader for Olympus ETS/VSI format files
- WSIProcessor: Main processing pipeline for tissue segmentation
- QCGrid: Quality control visualization tools

Example Usage:
-------------
>>> from wsi_pipeline import process_specimen
>>> results = process_specimen(
...     input_dir="/path/to/wsi",
...     output_dir="/path/to/output",
...     config="configs/default.yaml"
... )

For Colab:
>>> from notebooks.colab_setup import setup_colab
>>> setup_colab()

Submodule imports (not re-exported at top level):
- OME-Zarr writers: ``from wsi_pipeline.omezarr import write_ngff_from_mips, build_mips_from_yxc, ...``
- Precomputed: ``from wsi_pipeline.precomputed import PlatePrecomputedWriter, ...``
- Neuroglancer: ``from wsi_pipeline.neuroglancer import open_neuroglancer_plate_view, NeuroglancerViewer, ...``
- MLflow: ``from wsi_pipeline.mlflow_utils import init_mlflow, ...``
- EM-LDDMM: ``from wsi_pipeline.emlddmm_prep import remove_json_sidecars, ...``
"""

import logging

__version__ = "0.1.0"
__author__ = "Dominic M. Padova"
__email__ = "dpadova95@gmail.com"

# Silence "No handler found" warnings from library users who haven't configured logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Core configuration
from wsi_pipeline.config import PipelineConfig, load_config

# ETS/VSI file handling
from wsi_pipeline.etsfile import ETSFile, ETSFileError

# Quality control
from wsi_pipeline.qc_grid import QCGridBuilder, build_qc_grids

# Segmentation (high-level)
from wsi_pipeline.segmentation import WSISegmenter, segment_mask
from wsi_pipeline.vsi_converter import find_ets_file, vsi_to_flat_image

# Main processing (top-level interface)
from wsi_pipeline.wsi_processing import (
    WSIProcessor,
    process_directory,
    process_specimen,
    process_wsi,
)

__all__ = [
    # Version info
    "__version__",
    # Configuration
    "PipelineConfig",
    "load_config",
    # ETS/VSI file handling
    "ETSFile",
    "ETSFileError",
    "find_ets_file",
    "vsi_to_flat_image",
    # Main processing
    "WSIProcessor",
    "process_wsi",
    "process_directory",
    "process_specimen",
    # Segmentation
    "WSISegmenter",
    "segment_mask",
    # Quality control
    "QCGridBuilder",
    "build_qc_grids",
]
