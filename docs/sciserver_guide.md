# SciServer Guide

This guide covers deploying and running the WSI Tissue Pipeline on Johns Hopkins SciServer infrastructure.

## Overview

[SciServer](https://www.sciserver.org/) provides free cloud-based computing resources for scientific research. It offers:

- Persistent storage volumes
- Jupyter notebook environment
- Compute containers with GPU options
- Integration with SkyServer and CasJobs

## Getting Started

### 1. Create a SciServer Account

1. Go to [sciserver.org](https://www.sciserver.org/)
2. Click "Sign Up" and create an account
3. Verify your email

### 2. Create a Storage Volume

1. Navigate to **Files** > **Data Volumes**
2. Click **Create New Volume**
3. Name it `wsi-pipeline-data`
4. Select appropriate size (start with 100GB)

### 3. Launch a Compute Container

1. Go to **Compute** > **Create Container**
2. Select image: **Python (latest)** or **JupyterLab**
3. Add your storage volume
4. Select resources:
   - For testing: 2 cores, 8GB RAM
   - For production: 8+ cores, 32GB+ RAM, GPU if available

## Installation

### In a SciServer Notebook

```python
# Clone the repository
!git clone https://github.com/dmp2/wsi-tissue-pipeline.git
%cd wsi-tissue-pipeline

# Install the package
!pip install --user -e .

# Verify installation
import wsi_pipeline
print(f"Version: {wsi_pipeline.__version__}")
```

### Using Terminal

```bash
# Open a terminal in SciServer
cd /home/idies/workspace/Storage/<username>/persistent

# Clone repository
git clone https://github.com/dmp2/wsi-tissue-pipeline.git
cd wsi-tissue-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install
pip install -e .
```

## Storage Layout

SciServer has a specific directory structure:

```
/home/idies/workspace/
├── Storage/
│   ├── <username>/
│   │   └── persistent/          # Your persistent storage
│   │       ├── wsi-pipeline/    # Pipeline code
│   │       └── wsi-data/        # Your data
│   │           ├── input/       # Raw images
│   │           └── output/      # Processed results
│   └── shared-data/             # Shared datasets (read-only)
└── Temporary/
    └── <username>/
        └── scratch/             # Fast temporary storage
```

### Recommended Setup

```python
from pathlib import Path

# Define paths
WORKSPACE = Path("/home/idies/workspace")
PERSISTENT = WORKSPACE / "Storage" / "username" / "persistent"
SCRATCH = WORKSPACE / "Temporary" / "username" / "scratch"

# Data directories
DATA_DIR = PERSISTENT / "wsi-data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
MLFLOW_DIR = DATA_DIR / "mlruns"

# Create directories
for d in [INPUT_DIR, OUTPUT_DIR, MLFLOW_DIR]:
    d.mkdir(parents=True, exist_ok=True)
```

## Configuration

Use the SciServer-optimized configuration:

```python
from wsi_pipeline.config import load_config

# Load SciServer config
config = load_config('configs/sciserver.yaml')

# Or customize
config.mlflow.tracking_uri = f"sqlite:///{MLFLOW_DIR}/mlflow.db"
config.mlflow.experiment_name = "sciserver-processing"
```

### Custom sciserver.yaml

```yaml
segmentation:
  backend: "local-entropy"
  target_long_side: 2000
  min_area_px: 500

tile:
  chunk_size: 8192  # Larger for server hardware
  pad_multiple: 32

output:
  format: "ome-zarr"
  compression: "zlib"

mlflow:
  enabled: true
  tracking_uri: "sqlite:////home/idies/workspace/Storage/username/persistent/mlruns/mlflow.db"
  experiment_name: "sciserver-wsi-pipeline"

performance:
  num_workers: 4
  use_dask_distributed: false
  enable_checkpointing: true
```

## Processing Workflows

### Single Specimen Processing

```python
from wsi_pipeline.wsi_processing import process_specimen
from wsi_pipeline.config import load_config

config = load_config('configs/sciserver.yaml')

# Process specimen
results = process_specimen(
    input_dir=INPUT_DIR / 'specimen_001',
    output_dir=OUTPUT_DIR / 'specimen_001',
    config=config,
    pattern='*.jpg'
)

print(f"Processed {results['n_inputs']} slides")
```

### Batch Processing Multiple Specimens

```python
from pathlib import Path
import json

# Find all specimens
specimens = [d for d in INPUT_DIR.iterdir() if d.is_dir()]

# Process each
all_results = []
for specimen in specimens:
    print(f"Processing: {specimen.name}")
    
    results = process_specimen(
        input_dir=specimen,
        output_dir=OUTPUT_DIR / specimen.name,
        config=config,
        pattern='*.jpg'
    )
    
    all_results.append({
        'specimen': specimen.name,
        'n_inputs': results['n_inputs'],
        'n_outputs': results['n_outputs']
    })

# Save summary
with open(OUTPUT_DIR / 'batch_summary.json', 'w') as f:
    json.dump(all_results, f, indent=2)
```

### Using Scratch for Speed

Process on fast scratch storage, then copy results:

```python
import shutil

# Copy to scratch for processing
scratch_input = SCRATCH / 'input'
shutil.copytree(INPUT_DIR / 'specimen_001', scratch_input)

scratch_output = SCRATCH / 'output'
scratch_output.mkdir(exist_ok=True)

# Process on scratch (faster I/O)
results = process_specimen(
    input_dir=scratch_input,
    output_dir=scratch_output,
    config=config
)

# Copy results to persistent storage
final_output = OUTPUT_DIR / 'specimen_001'
shutil.copytree(scratch_output, final_output)

# Clean up scratch
shutil.rmtree(SCRATCH / 'input')
shutil.rmtree(SCRATCH / 'output')
```

## Job Scheduling

### Using SciServer Jobs

SciServer allows scheduling long-running jobs:

1. Create a script:

```python
#!/usr/bin/env python3
# process_job.py

import sys
sys.path.insert(0, '/home/idies/workspace/Storage/username/persistent/wsi-pipeline')

from wsi_pipeline.wsi_processing import process_specimen
from wsi_pipeline.config import load_config
from pathlib import Path

# Configuration
INPUT_DIR = Path(sys.argv[1])
OUTPUT_DIR = Path(sys.argv[2])

config = load_config('/path/to/sciserver.yaml')

# Process
results = process_specimen(INPUT_DIR, OUTPUT_DIR, config=config)

# Log completion
print(f"Completed: {results}")
```

2. Submit job via SciServer UI or API

### Checkpointing for Long Jobs

```python
from wsi_pipeline.config import PipelineConfig

config = PipelineConfig()
config.batch.enable_checkpointing = True
config.batch.checkpoint_interval = 10  # Save every 10 files
config.batch.resume_on_failure = True

# If job fails and restarts, it continues from last checkpoint
```

## CasJobs Integration

Store metadata in SciServer's database:

```python
from SciServer import CasJobs

# Create metadata table
create_table = """
CREATE TABLE wsi_processing_metadata (
    id INT IDENTITY(1,1) PRIMARY KEY,
    specimen_name VARCHAR(255),
    slide_name VARCHAR(255),
    n_tiles INT,
    processing_time FLOAT,
    created_at DATETIME DEFAULT GETDATE()
)
"""
CasJobs.executeQuery(create_table, context="MyDB")

# Insert processing results
for result in all_results:
    insert = f"""
    INSERT INTO wsi_processing_metadata 
    (specimen_name, slide_name, n_tiles, processing_time)
    VALUES ('{result['specimen']}', '{result['slide']}', 
            {result['n_tiles']}, {result['time']})
    """
    CasJobs.executeQuery(insert, context="MyDB")
```

## Monitoring and Logging

### View MLflow Runs

```python
import mlflow

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")

# List experiments
for exp in mlflow.search_experiments():
    print(f"{exp.name}: {exp.experiment_id}")

# List runs
runs = mlflow.search_runs(experiment_names=["sciserver-wsi-pipeline"])
print(runs[['run_id', 'start_time', 'metrics.n_outputs']])
```

### Resource Monitoring

```python
import psutil

# Check memory
mem = psutil.virtual_memory()
print(f"Memory: {mem.percent}% used ({mem.used/1e9:.1f}GB / {mem.total/1e9:.1f}GB)")

# Check disk
disk = psutil.disk_usage(str(PERSISTENT))
print(f"Disk: {disk.percent}% used ({disk.used/1e12:.2f}TB / {disk.total/1e12:.2f}TB)")
```

## Sharing Results

### With Collaborators

1. Add collaborators to your storage volume
2. Set appropriate permissions:

```python
# Create a shared output directory
shared_output = PERSISTENT / 'shared_results'
shared_output.mkdir(exist_ok=True)

# Copy key results
import shutil
shutil.copytree(
    OUTPUT_DIR / 'specimen_001' / 'qc',
    shared_output / 'specimen_001_qc'
)
```

### Publishing to SkyServer

Contact SciServer support to publish datasets to SkyServer for public access.

## Troubleshooting

### Container Won't Start

- Check resource allocation
- Verify storage volume is attached
- Try a smaller container first

### Out of Memory

```python
# Use smaller chunks
config.tile.chunk_size = 2048

# Process files individually
for f in input_files:
    process_wsi(f, output_dir, config=config)
    import gc
    gc.collect()
```

### Slow I/O

- Use scratch storage for processing
- Increase chunk size if memory allows
- Use compression wisely (blosc is faster than zlib)

### Package Not Found

```python
# Add to Python path
import sys
sys.path.insert(0, '/home/idies/workspace/Storage/username/persistent/wsi-pipeline')
```

## Best Practices

1. **Use persistent storage** - Keep important data in Storage volumes
2. **Use scratch for processing** - Faster I/O than persistent storage
3. **Enable checkpointing** - Protect against container timeouts
4. **Monitor resources** - Check memory and disk usage regularly
5. **Document your workflow** - Keep notes in your notebooks
6. **Version control** - Keep pipeline code in git

## Support

- [SciServer Help](https://www.sciserver.org/support/)
- [SciServer Forums](https://www.sciserver.org/support/forums/)
- [GitHub Issues](https://github.com/dmp2/wsi-tissue-pipeline/issues)
