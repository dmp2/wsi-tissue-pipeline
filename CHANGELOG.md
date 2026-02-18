# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2025

### Added

- ETS/VSI file reading and flat image conversion
- Tissue segmentation with multiple backends (local-entropy, Otsu)
- Tile extraction with configurable chunk size and padding
- OME-Zarr multi-resolution pyramid building (NGFF)
- Neuroglancer precomputed format writing (TensorStore and CloudVolume backends)
- Neuroglancer visualization with plate and precomputed viewers
- Quality control contact sheet generation
- EM-LDDMM histology preparation utilities
- CLI interface (`wsi-pipeline` command) with process, batch, qc, visualize subcommands
- MLflow experiment tracking integration
- Pydantic-based configuration with YAML and environment variable support
- Google Colab support with setup helper and optimized config
- SciServer deployment support
- Docker and Docker Compose configurations (CPU and GPU)
