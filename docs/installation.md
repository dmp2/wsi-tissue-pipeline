# Installation Guide

> **System Requirements:** Processing a single specimen (~0.5 TB of VSI/ETS files) requires:
> - **Storage:** ≥1 TB free disk space (raw input + OME-Zarr outputs)
> - **RAM:** ≥16 GB recommended (32 GB+ for very large slides)
> - **CPU:** Multi-core recommended; GPU optional (improves segmentation with some backends)
>
> For low-resource environments, use Google Colab (free GPU/CPU, Google Drive for storage).

This guide covers all methods for installing the WSI Tissue Pipeline.

## Quick Start

### Option 1: pip install (recommended for most users)

```bash
# Clone the repository
git clone https://github.com/dmp2/wsi-tissue-pipeline.git
cd wsi-tissue-pipeline

# Install in development mode
pip install -e .

# Install with Bio-Formats-backed VSI metadata support
pip install -e ".[bioformats]"

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Option 2: Conda environment

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate wsi-pipeline

# Install the package
pip install -e .
```

### Option 3: Docker

```bash
# Build and run with Docker Compose
cd docker
docker-compose up --build
```

## Detailed Installation

### System Requirements

- **Python**: 3.10, 3.11, or 3.12
- **Memory**: 8GB minimum, 16GB+ recommended for large slides
- **Storage**: SSD recommended for processing large files
- **GPU** (optional): NVIDIA GPU with CUDA support for acceleration

### Core Dependencies

The pipeline requires these core packages:

```
numpy>=1.21.0
scipy>=1.7.0
scikit-image>=0.19.0
Pillow>=9.0.0
imageio>=2.9.0
dask>=2022.1.0
zarr>=2.10.0
tifffile>=2022.2.0
pydantic>=2.0.0
PyYAML>=6.0
click>=8.0.0
rich>=12.0.0
```

### Optional Dependencies

#### Visualization (`pip install -e ".[visualization]"`)
- `neuroglancer`: 3D visualization server
- `matplotlib`: Plotting and QC images
- `napari`: Interactive viewer

#### GPU Acceleration (`pip install -e ".[torch]"`)
- `torch`: PyTorch for GPU processing
- `torchvision`: Image transforms

#### Pathology Tools (`pip install -e ".[pathology]"`)
- `openslide-python`: SVS/MRXS file support
- `tiatoolbox`: Pathology analysis tools
- `pathml`: PathML integration

#### Development (`pip install -e ".[dev]"`)
- `pytest`: Testing framework
- `pytest-cov`: Coverage reporting
- `black`: Code formatting
- `ruff`: Fast linting
- `mypy`: Type checking
- `pre-commit`: Git hooks

#### Bio-Formats VSI Metadata (`pip install -e ".[bioformats]"`)
- `pyjnius`: Python-to-Java bridge used to call Bio-Formats
- `platformdirs`: Managed cache location for the pinned Bio-Formats jar
- `filelock`: First-use download locking for concurrent processes

This extra enables `get_vsi_metadata(..., metadata_backend="auto")` to pull physical pixel sizes and related VSI metadata through Bio-Formats.

### Step-5 EM-LDDMM Dependencies

`step5` uses a hybrid backend resolver for registration:
- First choice: the installed external `emlddmm` package.
- Fallback: the vendored legacy compatibility code in this repository.

Important step-5 extras:
- `tensorstore` is required when `--target-source-format precomputed` is used.
- `transformation_graph_v01.py` is treated as part of the external `emlddmm` package, not this repository.
- `--run-transformation-graph` will fail early unless that external script can be resolved or `--transformation-graph-script` points to it explicitly.

Recommended install when you plan to use step 5 with precomputed targets and transformation-graph execution:

```bash
pip install tensorstore emlddmm
```

### Platform-Specific Instructions

#### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    libvips-dev \
    libopenslide-dev \
    libjpeg-dev \
    libpng-dev

# Install the package
pip install -e ".[all]"
```

#### macOS

```bash
# Install dependencies with Homebrew
brew install python@3.11 vips openslide

# Install the package
pip install -e ".[all]"
```

#### Windows

```powershell
# Using conda (recommended for Windows)
conda create -n wsi-pipeline python=3.11
conda activate wsi-pipeline

# Install dependencies
conda install -c conda-forge openslide-python numpy scipy scikit-image

# Install the package
pip install -e .
```

### VSI Metadata / Bio-Formats

Use the Bio-Formats extra when you want physical VSI metadata populated automatically:

```bash
pip install -e ".[bioformats]"
```

#### What happens on first use

- The pipeline validates that Java is available.
- If you did not provide a custom jar, it auto-downloads the pinned official `bioformats_package.jar`.
- The jar is cached at `platformdirs.user_cache_dir("wsi-pipeline")/bioformats/8.4.0/bioformats_package.jar` by default.
- The cache download is guarded by a file lock so concurrent first-use calls do not clobber the jar.

#### Supported setup paths

##### Pip

- `pip install -e ".[bioformats]"` installs the Python-side dependencies.
- You still need a working JVM on the machine. `JAVA_HOME` is honored if set.
- The managed jar is downloaded automatically unless you disable downloads.

##### Conda

- `environment.yml` includes `openjdk` and `pyjnius`.
- After `conda env create -f environment.yml`, VSI metadata works without separate Java setup.

##### Docker

- `docker/Dockerfile` installs `openjdk-17-jre-headless`.
- The image also installs `.[all]`, which now includes the Bio-Formats extra.
- This is the recommended fully managed local runtime when you want the pipeline environment curated end to end.

#### Environment overrides

- `WSI_PIPELINE_BIOFORMATS_JAR`: use an explicit local jar instead of the managed cache.
- `WSI_PIPELINE_BIOFORMATS_CACHE_DIR`: override the cache root for the managed jar.
- `WSI_PIPELINE_BIOFORMATS_DOWNLOAD=0`: disable automatic jar download for offline or locked-down environments.
- `JAVA_HOME`: point Pyjnius at a specific Java runtime.

#### Example

```python
from wsi_pipeline.omezarr import build_mips_from_yxc, write_ngff_from_mips
from wsi_pipeline.vsi_converter import get_vsi_metadata, vsi_to_flat_image

slide = "data/specimen.vsi"
image = vsi_to_flat_image(slide, level=2)
mips = build_mips_from_yxc(image, num_mips=4)
metadata = get_vsi_metadata(slide, metadata_backend="auto")

write_ngff_from_mips(
    mips,
    "output/specimen.ome.zarr",
    ngff_metadata=metadata,
    metadata_schema="v0.4",
)
```

### GPU Support

#### NVIDIA CUDA

```bash
# Install CUDA toolkit (see NVIDIA docs)
# Then install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"
```

#### Docker with GPU

```bash
# Ensure nvidia-docker is installed
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Verification

After installation, verify everything works:

```bash
# Check CLI is available
wsi-pipeline info

# Run tests
pytest tests/ -v

# Process a sample image
python scripts/download_sample_data.py
wsi-pipeline process -i sample_data/synthetic/synthetic_wsi_001.png -o output/
```

## Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'wsi_pipeline'"

Ensure you've installed the package in development mode:
```bash
pip install -e .
```

#### OpenSlide not found

On Linux:
```bash
sudo apt-get install libopenslide-dev
```

On macOS:
```bash
brew install openslide
```

#### CUDA out of memory

Reduce chunk size in configuration:
```yaml
tile:
  chunk_size: 2048  # Reduce from default 4096
```

#### Permission denied errors in Docker

```bash
# Fix permissions
sudo chown -R $USER:$USER ./data ./output
```

#### Java runtime not found

Install Java or set `JAVA_HOME`. The curated Conda environment and Docker image already include Java.

#### Bio-Formats auto-download blocked

If your environment blocks network access, either:

- set `WSI_PIPELINE_BIOFORMATS_JAR` to a local `bioformats_package.jar`, or
- pre-populate the managed cache directory and set `WSI_PIPELINE_BIOFORMATS_DOWNLOAD=0`.

#### `jnius` imported before classpath setup

Do not import `jnius` yourself before calling `get_vsi_metadata()` or the runtime bootstrap helper. Pyjnius requires the classpath to be configured before the JVM bridge is imported.

#### Custom Bio-Formats jar

Set `WSI_PIPELINE_BIOFORMATS_JAR=/path/to/bioformats_package.jar` when you need to pin a local artifact explicitly.

### Getting Help

- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share ideas
- Documentation: Full API reference at `/docs`

## Updating

```bash
# Pull latest changes
git pull origin main

# Reinstall
pip install -e .

# Or rebuild Docker
docker-compose build --no-cache
```

## Uninstalling

```bash
# Uninstall package
pip uninstall wsi-tissue-pipeline

# Remove conda environment
conda env remove -n wsi-pipeline

# Remove Docker images
docker-compose down --rmi all
```
