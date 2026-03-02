# WSI Tissue Pipeline

[![CI](https://github.com/dmp2/wsi-tissue-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/dmp2/wsi-tissue-pipeline/actions/workflows/ci.yml)

A reproducible whole-slide image (WSI) processing pipeline for converting VSI/ETS format images to OME-Zarr format and extracting individual tissue sections for downstream analysis including 3D visualization and registration.

## Overview

This pipeline processes large whole-slide images (potentially 0.5TB+ per specimen) through the following stages:

```
VSI/ETS Files → OME-Zarr Pyramids → Tissue Section Extraction → Downstream Tasks
                                                                   ├── Neuroglancer Visualization
                                                                   ├── Quality Control Grids
                                                                   └── 3D Registration (EM-LDDMM)
```

## Quick Start

> **System Requirements:** Processing a single specimen (~0.5 TB of VSI/ETS files) requires:
> - **Storage:** ≥1 TB free disk space (raw input + OME-Zarr outputs)
> - **RAM:** ≥16 GB recommended (32 GB+ for very large slides)
> - **CPU:** Multi-core recommended; GPU optional (improves segmentation with some backends)
>
> For low-resource environments, use Google Colab (free GPU/CPU, Google Drive for storage).

**Which option should I use?**

| I want to… | Use |
|---|---|
| Try the pipeline with no local setup | Google Colab |
| Reproduce results in a consistent environment | Docker |
| Customize the pipeline or integrate into my code | Local installation |
| Run on JHU SciServer infrastructure | SciServer (see `docs/sciserver_guide.md`) |

### Option 1: Google Colab (Recommended for getting started)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmp2/wsi-tissue-pipeline/blob/main/notebooks/01_wsi_to_tissue_sections.ipynb)

1. Click the Colab badge above or open `notebooks/01_wsi_to_tissue_sections.ipynb`
2. Connect to a GPU runtime (recommended for large images)
3. Run the setup cell to install dependencies
4. Configure your data paths and run the pipeline

### Option 2: Local Development with Docker

```bash
# Clone the repository
git clone https://github.com/dmp2/wsi-tissue-pipeline.git
cd wsi-tissue-pipeline

# Build and run with Docker Compose
docker-compose -f docker/docker-compose.yml up --build

# Access services:
# - Jupyter Lab: http://localhost:8888
# - MLflow UI: http://localhost:5000
```

### Option 3: Local Installation

```bash
# Clone and setup
git clone https://github.com/dmp2/wsi-tissue-pipeline.git
cd wsi-tissue-pipeline

# Create conda environment
conda env create -f environment.yml
conda activate wsi-pipeline

# Install package in development mode
pip install -e .

# Start MLflow tracking server (optional)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

## Project Structure

```
wsi-tissue-pipeline/
├── README.md                    # This file
├── pyproject.toml              # Package configuration
├── environment.yml             # Conda environment specification
├── requirements.txt            # Pip requirements
├── .gitignore                  # Git ignore patterns
├── .env.example                # Environment variables template
│
├── docker/
│   ├── Dockerfile              # Main Docker image
│   ├── Dockerfile.gpu          # GPU-enabled Docker image
│   └── docker-compose.yml      # Multi-container orchestration
│
├── src/
│   └── wsi_pipeline/
│       ├── __init__.py         # Main package exports
│       ├── config.py           # Configuration management (Pydantic)
│       ├── etsfile.py          # ETS file reader
│       ├── vsi_converter.py    # VSI to flat file conversion
│       ├── wsi_processing.py   # Main WSI processing module
│       ├── qc_grid.py          # Quality control image grids
│       ├── neuroglancer.py     # Neuroglancer state, server, and viewer
│       ├── emlddmm_prep.py     # EM-LDDMM registration preparation
│       ├── mlflow_utils.py     # MLflow experiment tracking helpers
│       ├── cli.py              # Command-line interface
│       │
│       ├── segmentation/       # Tissue segmentation module
│       │   ├── __init__.py
│       │   ├── core.py         # Core utilities (grayscale, thumb, upsample)
│       │   ├── entropy.py      # Entropy-based masking
│       │   ├── otsu.py         # Otsu threshold masking
│       │   ├── morphology.py   # Morphological operations
│       │   └── segmenter.py    # WSISegmenter class
│       │
│       ├── tiles/              # Tile generation module
│       │   ├── __init__.py
│       │   ├── generator.py    # generate_tissue_tiles()
│       │   ├── io.py           # Tile I/O utilities
│       │   └── naming.py       # File naming conventions
│       │
│       ├── omezarr/            # OME-Zarr/NGFF writing module
│       │   ├── __init__.py
│       │   ├── metadata.py     # NGFF metadata utilities
│       │   ├── pyramid.py      # MIP pyramid building
│       │   ├── streaming.py    # Memory-efficient streaming writers
│       │   └── writers.py      # Standard NGFF writers
│       │
│       ├── precomputed/        # Neuroglancer precomputed format
│       │   ├── __init__.py
│       │   ├── cloudvolume.py  # CloudVolume backend
│       │   ├── tensorstore.py  # TensorStore backend
│       │   └── plate_writer.py # PlatePrecomputedWriter class
│       │
│       ├── pipeline/           # High-level pipeline orchestration
│       │   ├── __init__.py
│       │   └── plating.py      # process_slide_with_plating()
│       │
│       └── sciserver/          # SciServer deployment utilities (optional)
│           ├── __init__.py
│           ├── sciserver_integration.py  # SciServerPipeline class
│           ├── sciserver_mlflow.py       # MLflow config for SciServer
│           ├── sciserver_lineage.py      # Data lineage tracking
│           └── environment.py            # Environment detection
│
├── notebooks/
│   ├── 01_wsi_to_tissue_sections.ipynb    # Example pipeline / MWE — step-by-step tutorial
│   ├── 02_quality_control.ipynb           # QC and validation
│   ├── 03_neuroglancer_visualization.ipynb # 3D visualization
│   └── colab_setup.py                      # Colab initialization helper
│
├── configs/
│   ├── default.yaml            # Default pipeline configuration
│   ├── colab.yaml              # Google Colab optimized config
│   └── sciserver.yaml          # SciServer deployment config
│
├── scripts/
│   ├── run_pipeline.py         # Production runner — batch processing without notebook overhead
│   ├── setup_mlflow.sh         # MLflow server setup script
│   └── download_sample_data.py # Fallback data source when real specimens cannot be shared
│
├── tests/
│   ├── __init__.py
│   ├── test_etsfile.py
│   ├── test_processing.py
│   └── conftest.py             # Pytest fixtures
│
└── docs/
    ├── installation.md         # Detailed installation guide
    ├── configuration.md        # Configuration options
    ├── colab_guide.md          # Google Colab usage guide
    └── sciserver_guide.md      # SciServer deployment guide
```

## Module Architecture

The pipeline is organized into focused submodules:

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `segmentation` | Tissue mask generation | `WSISegmenter`, `segment_mask`, `make_lowres_mask` |
| `tiles` | Tile extraction from masks | `generate_tissue_tiles`, `save_tile`, `to_uint8` |
| `omezarr` | OME-Zarr pyramid building | `build_mips_from_yxc`, `write_ngff_from_mips`, `write_ngff_from_mips_ngffzarr` |
| `precomputed` | Neuroglancer Precomputed format writing | `PlatePrecomputedWriter` |
| `pipeline` | End-to-end orchestration | `process_slide_with_plating` |
| `neuroglancer` | Neuroglancer state, server, and viewer | `NeuroglancerViewer`, `emit_ng_state_for_ngff_plate`, `open_neuroglancer_plate_view` |
| `sciserver` | SciServer deployment (optional) | `SciServerPipeline`, `setup_sciserver_tracking` |

**Canonical API:** Prefer standalone functions (`process_wsi`, `process_specimen`, `build_qc_grids`) for scripting. Use `WSIProcessor` and `QCGridBuilder` classes when you want to configure once and call multiple times.

### Import Examples

```python
# High-level API (backward compatible)
from wsi_pipeline import process_specimen, PipelineConfig

# Specific modules for custom workflows
from wsi_pipeline.segmentation import WSISegmenter
from wsi_pipeline.tiles import generate_tissue_tiles
from wsi_pipeline.omezarr import build_mips_from_yxc, write_ngff_from_mips
from wsi_pipeline.precomputed import PlatePrecomputedWriter
from wsi_pipeline.pipeline import process_slide_with_plating
from wsi_pipeline.neuroglancer import NeuroglancerViewer, emit_ng_state_for_ngff_plate
```

## Glossary

| Term | Meaning |
|------|---------|
| **WSI** | Whole-Slide Image — a high-resolution digital scan of an entire microscopy slide |
| **VSI / ETS** | Olympus file formats for whole-slide images; `.vsi` is the container, `.ets` files hold the pyramid tile data |
| **OME-Zarr / OME-NGFF** | Open Microscopy Environment — Next Generation File Format. A chunked, cloud-friendly image format built on Zarr. The `omezarr/` module writes this format. |
| **Neuroglancer Precomputed** | A chunked format for large volumetric data, natively read by the Neuroglancer browser viewer. The `precomputed/` module writes this format. |
| **Plating** | Histology term for laying out individual tissue sections in a grid arrangement for downstream processing and registration. Used in `pipeline/plating.py`. |
| **EM-LDDMM** | Expectation-Maximization Large Deformation Diffeomorphic Metric Mapping — a 3D image registration algorithm. `emlddmm_prep.py` prepares metadata for the external `emlddmm` package. |
| **MIP** | Multi-resolution Image Pyramid — the downsampled resolution levels written into OME-Zarr for efficient viewing at any zoom level. |
| **QC** | Quality Control — visual validation grids (contact sheets) showing all extracted tissue tiles. |

## MLflow Experiment Tracking

All pipeline runs are automatically tracked with MLflow:

```python
import mlflow
from wsi_pipeline import process_specimen

# Start a tracked run
with mlflow.start_run(run_name="specimen_E241"):
    results = process_specimen(
        input_dir="/path/to/wsi/files",
        output_dir="/path/to/output",
        config="configs/default.yaml"
    )
    # Metrics and artifacts are automatically logged
```

View results in the MLflow UI at `http://localhost:5000`

## Configuration

Pipeline behavior is controlled via YAML configuration files:

```yaml
# configs/default.yaml
segmentation:
  backend: "local-entropy"
  target_long_side: 1800
  min_area_px: 3000
  struct_elem_px: 4
  split_touching: true
  r_split: 2

output:
  format: "ome-zarr"  # or "tiff"
  chunk_size: 512
  compression: "zstd"

mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "wsi-tissue-pipeline"
```

## Command Line Interface

```bash
# Process a single WSI file
wsi-pipeline process --input /path/to/file.vsi --output /path/to/output

# Process a directory of WSI files
wsi-pipeline batch --input-dir /path/to/wsi --output-dir /path/to/output --pattern "*.vsi"

# Generate QC grids
wsi-pipeline qc --input-dir /path/to/tissues --output-dir /path/to/qc

# Start Neuroglancer visualization server
wsi-pipeline visualize --zarr-dir /path/to/zarr --port 9999
```

## Staged Runner

The staged runner in [`scripts/run_pipeline.py`](/C:/Users/dpado/Documents/git/temporal_bone_mapping/wsi-tissue-pipeline/wsi-tissue-pipeline/scripts/run_pipeline.py) exposes the notebook-aligned EM-LDDMM workflow as `step5` / `reconstruct`.

Use the dedicated step-5 docs for the full workflow:
- [`docs/emlddmm_registration.md`](/C:/Users/dpado/Documents/git/temporal_bone_mapping/wsi-tissue-pipeline/wsi-tissue-pipeline/docs/emlddmm_registration.md): onboarding, defaults, units, target modes, logging, QC report, and transformation-graph execution.
- [`docs/emlddmm_notebook_parity.md`](/C:/Users/dpado/Documents/git/temporal_bone_mapping/wsi-tissue-pipeline/wsi-tissue-pipeline/docs/emlddmm_notebook_parity.md): high-level mapping from `legacy_scripts/tb_macaque_emlddmm.ipynb` to the staged pipeline.

Step 4 writes `samples.tsv`, per-slice JSON sidecars, and `emlddmm_dataset_manifest.json`. Step 5 then loads the target, rescales axes into micrometers, downsamples to the working grid, runs self-alignment, optionally runs atlas registration, and optionally runs between-slice filling.

Key step-5 behavior:
- Use `--dataset-root` for the prepared dataset root. `-o/--output` remains a backward-compatible but deprecated alias.
- Atlas registration requires `--atlas` plus either `--init-affine` or `--orientation-from/--orientation-to`.
- `--list-orientations` prints the valid backend orientation codes without running a registration.
- Precomputed targets require `--target-source-format precomputed` plus `--precomputed-manifest`.
- `--dry-run` resolves the full plan without executing stages.
- `--write-qc-report` writes `registration_report.html` and `registration_report.json`.
- Every run writes `run_provenance.json` and `reproduce_step5_command.txt`.
- `--run-transformation-graph` resolves `transformation_graph_v01.py` from the external `emlddmm` package unless `--transformation-graph-script` is supplied.

Example atlas-free run:

```bash
python scripts/run_pipeline.py step5 \
  --dataset-root /data/tiles
```

Example atlas-registration run from prepared slices:

```bash
python scripts/run_pipeline.py step5 \
  --dataset-root /data/tiles \
  --atlas /data/atlas.vtk \
  --label /data/atlas_labels.vtk \
  --orientation-from PIR \
  --orientation-to RIP
```

Example atlas-free run from Neuroglancer precomputed data:

```bash
python scripts/run_pipeline.py step5 \
  --dataset-root /data/tiles \
  --target-source /data/precomputed_plate \
  --target-source-format precomputed \
  --precomputed-manifest /data/tiles/emlddmm_dataset_manifest.json
```

Example dry-run plan resolution:

```bash
python scripts/run_pipeline.py step5 \
  --dataset-root /data/tiles \
  --atlas /data/atlas.vtk \
  --orientation-from PIR \
  --orientation-to RIP \
  --write-qc-report \
  --dry-run
```

## Deployment Options

### Google Colab

Best for initial exploration and sharing:
- GPU acceleration for segmentation
- High RAM options for large images
- Easy sharing of notebooks; use `download_sample_data.py` if raw data cannot be shared

### SciServer

Best for production processing at JHU:
- Persistent storage and scheduled processing
- See `docs/sciserver_guide.md` and `src/wsi_pipeline/sciserver/` for integration utilities

### Local / Docker

Best for development and reproducible batch runs:
- Full control over environment and dependencies
- Use `scripts/run_pipeline.py` for non-interactive batch processing
- Use `notebooks/` for step-by-step exploration and experimentation

## Outputs

The pipeline produces:

1. **OME-Zarr files**: Multi-resolution pyramids for each tissue section
2. **Metadata**: JSON files with segmentation parameters and provenance
3. **QC Grids**: Contact sheets for visual quality control
4. **MLflow artifacts**: Experiment logs, metrics, and model artifacts

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Johns Hopkins University for the ETS file format documentation
- The OME-Zarr community for standardized image storage
- Neuroglancer for 3D visualization capabilities

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{wsi_tissue_pipeline,
  title = {WSI Tissue Pipeline},
  author = {Dominic Padova},
  year = {2025},
  url = {https://github.com/dmp2/wsi-tissue-pipeline}
}
```
