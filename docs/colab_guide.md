# Google Colab Guide

This guide covers running the WSI Tissue Pipeline on Google Colab, including setup, GPU acceleration, and Google Drive integration.

## Quick Start

1. Open a new Colab notebook
2. Run the setup cell (see below)
3. Process your images!

### Setup Cell

Copy and run this cell at the start of your notebook:

```python
# Install WSI Tissue Pipeline in Colab
!pip install -q git+https://github.com/dmp2/wsi-tissue-pipeline.git

# Or clone and install in development mode
!git clone https://github.com/dmp2/wsi-tissue-pipeline.git
%cd wsi-tissue-pipeline
!pip install -q -e .

# Import and verify
import wsi_pipeline
print(f"WSI Pipeline version: {wsi_pipeline.__version__}")
```

## Mounting Google Drive

Your data is likely stored in Google Drive. Mount it with:

```python
from google.colab import drive
drive.mount('/content/drive')

# Your data is now at /content/drive/MyDrive/
```

Or use the built-in helper:

```python
from wsi_pipeline.config import PipelineConfig

config = PipelineConfig()
config.colab.mount_drive = True  # Automatically mounts Drive

# The colab_setup module handles this automatically
from notebooks.colab_setup import setup_colab
setup_colab()
```

## GPU Acceleration

### Enabling GPU

1. Go to **Runtime > Change runtime type**
2. Select **GPU** under Hardware accelerator
3. Click **Save**

### Verify GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# For the pipeline
from wsi_pipeline.config import PipelineConfig
config = PipelineConfig()
config.colab.use_gpu = True
```

### Colab GPU Tiers

| Tier | GPU | RAM | Best For |
|------|-----|-----|----------|
| Free | T4 (15GB) | 12GB | Testing, small batches |
| Pro | T4/V100 (16GB) | 25GB | Regular processing |
| Pro+ | A100 (40GB) | 52GB | Large-scale production |

## Sample Workflow

### 1. Setup Environment

```python
# Mount Drive and setup
from google.colab import drive
drive.mount('/content/drive')

# Install pipeline
!pip install -q git+https://github.com/dmp2/wsi-tissue-pipeline.git

# Configure paths
DATA_DIR = '/content/drive/MyDrive/wsi-data'
OUTPUT_DIR = '/content/drive/MyDrive/wsi-output'
```

### 2. Process Images

```python
from pathlib import Path
from wsi_pipeline.wsi_processing import process_specimen
from wsi_pipeline.config import load_config

# Load Colab-optimized configuration
config = load_config('/content/wsi-tissue-pipeline/configs/colab.yaml')

# Process all slides
results = process_specimen(
    input_dir=Path(DATA_DIR) / 'specimen_001',
    output_dir=Path(OUTPUT_DIR) / 'specimen_001',
    config=config,
    pattern='*.jpg'
)

print(f"Processed {results['n_inputs']} slides")
print(f"Created {results['n_outputs']} tissue tiles")
```

### 3. Generate QC Grids

```python
from wsi_pipeline.qc_grid import build_qc_grids

qc_paths = build_qc_grids(
    input_dir=Path(OUTPUT_DIR) / 'specimen_001',
    output_dir=Path(OUTPUT_DIR) / 'specimen_001' / 'qc',
    thumb_size=256,
    create_master=True
)

# Display in notebook
from IPython.display import Image
Image(filename=str(qc_paths[0]))
```

### 4. MLflow Tracking

```python
# Initialize MLflow with Drive storage
from wsi_pipeline.mlflow_utils import init_mlflow

init_mlflow(
    tracking_uri='sqlite:////content/drive/MyDrive/mlflow/mlflow.db',
    experiment_name='colab-wsi-pipeline'
)

# View runs in MLflow UI
!mlflow ui --port 5000 &
from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(5000)"))
```

## Working with Large Files

### Handling Memory

```python
# Check available memory
!free -h

# Clear GPU cache between operations
import torch
torch.cuda.empty_cache()

# Use smaller chunk sizes for limited memory
config.tile.chunk_size = 2048  # Default is 4096
```

### Processing in Batches

```python
from pathlib import Path
import gc

input_files = list(Path(DATA_DIR).glob('*.jpg'))

# Process in batches of 10
batch_size = 10
for i in range(0, len(input_files), batch_size):
    batch = input_files[i:i+batch_size]
    
    for f in batch:
        process_wsi(f, OUTPUT_DIR, config=config)
    
    # Clear memory between batches
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Completed batch {i//batch_size + 1}")
```

### Using Local SSD for Speed

Colab's local storage is faster than Google Drive:

```python
# Copy data to local storage for processing
!cp -r /content/drive/MyDrive/wsi-data/specimen_001 /content/temp_data/

# Process from local storage
results = process_specimen(
    input_dir='/content/temp_data/specimen_001',
    output_dir='/content/temp_output',
    config=config
)

# Copy results back to Drive
!cp -r /content/temp_output/* /content/drive/MyDrive/wsi-output/
```

## Notebooks

The repository includes ready-to-use notebooks:

### 01_wsi_to_tissue_sections.ipynb
Complete workflow for segmenting whole-slide images into tissue sections.

```python
# Open in Colab
# Click the "Open in Colab" badge in the notebook or:
# https://colab.research.google.com/github/dmp2/wsi-tissue-pipeline/blob/main/notebooks/01_wsi_to_tissue_sections.ipynb
```

### 02_quality_control.ipynb
Generate and review QC images for processed datasets.

### 03_neuroglancer_visualization.ipynb
Launch Neuroglancer for 3D visualization of tissue stacks.

## Troubleshooting

### Session Disconnects

Colab sessions timeout after ~12 hours (90 min idle). Strategies:

```python
# Keep session alive with periodic activity
import time
while True:
    print('.', end='', flush=True)
    time.sleep(60 * 5)  # Every 5 minutes
```

### Out of Memory (OOM)

```python
# Reduce batch size
config.tile.chunk_size = 1024

# Process fewer files at once
for f in input_files:
    process_wsi(f, output_dir, config=config)
    torch.cuda.empty_cache()
    gc.collect()
```

### Drive Quota Issues

```python
# Check Drive usage
!df -h /content/drive

# Clean up intermediate files
!rm -rf /content/temp_output
```

### Package Import Errors

```python
# Restart runtime after installation
import os
os.kill(os.getpid(), 9)  # Restart runtime

# Then re-run imports
```

## Best Practices

1. **Save frequently** - Colab sessions can disconnect unexpectedly
2. **Use checkpointing** - Enable in config for long-running jobs
3. **Monitor GPU memory** - Use `!nvidia-smi` to check usage
4. **Process to local first** - Then copy to Drive (faster)
5. **Keep data organized** - Use consistent directory structure

## Resource Links

- [Google Colab Documentation](https://colab.research.google.com/notebooks/)
- [Colab Pro Features](https://colab.research.google.com/signup)
- [Managing Colab Resources](https://research.google.com/colaboratory/faq.html)
