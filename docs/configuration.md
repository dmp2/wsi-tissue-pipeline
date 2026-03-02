# Configuration Guide

The WSI Tissue Pipeline uses a hierarchical configuration system that supports YAML files, environment variables, and programmatic overrides.

## Configuration Files

Configuration files are located in the `configs/` directory:

- `default.yaml` - Standard settings for local processing
- `colab.yaml` - Optimized for Google Colab
- `sciserver.yaml` - Optimized for SciServer environment

### Loading Configuration

```python
from wsi_pipeline.config import PipelineConfig, load_config

# Load from file
config = load_config("configs/colab.yaml")

# Or create default configuration
config = PipelineConfig()
```

## Configuration Sections

### Segmentation Settings

Control how tissue regions are detected in whole-slide images.

```yaml
segmentation:
  # Segmentation algorithm
  # Options: "local-entropy", "local-otsu", "tiatoolbox", "pathml"
  backend: "local-entropy"
  
  # Target size for the thumbnail used in segmentation (longest side)
  # Larger = more accurate but slower
  target_long_side: 2000
  
  # Minimum area (in pixels at thumbnail scale) for a valid tissue region
  # Smaller values detect more regions including artifacts
  min_area_px: 500
  
  # Structuring element size for morphological operations
  # Larger = smoother boundaries but may merge close sections
  struct_elem_px: 5
  
  # Whether to attempt splitting touching tissue sections
  split_touching: true
  
  # Aspect ratio threshold for splitting elongated components
  # Higher = only split very elongated shapes
  r_split: 3.0
```

#### Backend Details

| Backend | Description | Speed | Quality |
|---------|-------------|-------|---------|
| `local-entropy` | Entropy-based detection | Fast | Good |
| `local-otsu` | Otsu thresholding | Fastest | Basic |
| `tiatoolbox` | TIAToolbox algorithms | Medium | Excellent |
| `pathml` | PathML tissue detection | Medium | Excellent |

### Tile Settings

Control how tissue regions are extracted as individual tiles.

```yaml
tile:
  # Chunk size for Dask processing
  # Larger = faster but more memory
  chunk_size: 4096
  
  # Pad output to multiple of this value
  # Use 16 or 32 for GPU compatibility
  pad_multiple: 16
  
  # Extra margin around detected tissue (in pixels at full resolution)
  extra_margin_px: 100
```

### Output Settings

Control the format and quality of output files.

```yaml
output:
  # Output format
  # Options: "ome-zarr", "tiff", "both"
  format: "ome-zarr"
  
  # Compression codec
  # Options: "zlib", "blosc", "lz4", null (no compression)
  compression: "zlib"
  
  # Convert 16-bit images to 8-bit to save space
  convert_to_uint8: true
  
  # Generate QC images during processing
  generate_qc: true
```

#### Format Comparison

| Format | Pros | Cons | Best For |
|--------|------|------|----------|
| `ome-zarr` | Cloud-friendly, pyramidal | Requires zarr tools | Large-scale analysis |
| `tiff` | Universal compatibility | Large files | Single-image workflows |
| `both` | Maximum flexibility | 2x storage | Development/debugging |

### MLflow Settings

Configure experiment tracking with MLflow.

```yaml
mlflow:
  # Enable/disable tracking
  enabled: true
  
  # Tracking server URI
  # Local SQLite: "sqlite:///mlflow.db"
  # Remote server: "http://mlflow-server:5000"
  tracking_uri: "sqlite:///mlflow.db"
  
  # Experiment name for organizing runs
  experiment_name: "wsi-tissue-pipeline"
  
  # Template for run names
  # Supports: {specimen}, {timestamp}, {config_hash}
  run_name_template: "{specimen}_{timestamp}"
  
  # Log output files as artifacts (increases storage)
  log_artifacts: false
```

### Colab Settings

Settings specific to Google Colab environment.

```yaml
colab:
  # Automatically mount Google Drive
  mount_drive: true
  
  # Use GPU if available
  use_gpu: true
  
  # Install dependencies when running in Colab
  install_dependencies: true
```

## Environment Variables

All configuration values can be overridden with environment variables using the `WSI_` prefix:

```bash
# Override segmentation backend
export WSI_SEGMENTATION_BACKEND="local-otsu"

# Override output format
export WSI_OUTPUT_FORMAT="tiff"

# Disable MLflow
export WSI_MLFLOW_ENABLED="false"
```

### Nested Values

Use double underscores for nested configuration:

```bash
# Set segmentation.min_area_px
export WSI_SEGMENTATION__MIN_AREA_PX=1000

# Set tile.chunk_size
export WSI_TILE__CHUNK_SIZE=2048
```

## Programmatic Configuration

### Direct Modification

```python
from wsi_pipeline.config import PipelineConfig

config = PipelineConfig()

# Modify settings
config.segmentation.backend = "local-otsu"
config.segmentation.min_area_px = 300
config.output.format = "tiff"
config.mlflow.enabled = False

# Use in processing
from wsi_pipeline.wsi_processing import process_wsi
results = process_wsi(input_path, output_dir, config=config)
```

### Configuration Dictionary

```python
config = PipelineConfig(
    segmentation={
        "backend": "local-entropy",
        "target_long_side": 3000,
    },
    output={
        "format": "ome-zarr",
        "compression": "blosc",
    },
    mlflow={
        "enabled": True,
        "experiment_name": "my-experiment",
    }
)
```

### Merging Configurations

```python
from wsi_pipeline.config import load_config, merge_configs

# Load base config
base_config = load_config("configs/default.yaml")

# Override specific values
overrides = {"segmentation": {"backend": "tiatoolbox"}}
final_config = merge_configs(base_config, overrides)
```

## Step-5 EM-LDDMM Overrides

`step5` also accepts a JSON workflow override via `--emlddmm-config`. This is separate from the YAML pipeline config used for the earlier stages.

Minimal example:

```json
{
  "units": {
    "atlas_unit_scale": 1000.0,
    "target_unit_scale": 1.0,
    "desired_resolution_um": 200.0
  },
  "stage_controls": {
    "self_alignment_enabled": true,
    "atlas_registration_enabled": true,
    "upsampling_enabled": false
  },
  "atlas_registration": {
    "enabled": true
  },
  "upsampling": {
    "enabled": false,
    "mode": "seg"
  },
  "transformation_graph": {
    "write_config": true,
    "execute": false,
    "script_path": "C:/path/to/transformation_graph_v01.py"
  },
  "outputs": {
    "write_qc_report": true
  }
}
```

Important fields:
- `units.atlas_unit_scale`: scales atlas axes into micrometers. The notebook-aligned preset assumes the atlas is stored in millimeters, so the default is `1000.0`.
- `units.target_unit_scale`: scales target axes into micrometers. The notebook-aligned preset assumes the prepared target already uses micrometers, so the default is `1.0`.
- `units.desired_resolution_um`: working-grid resolution. The default notebook-aligned value is `200.0`.
- `orientation_from` and `orientation_to`: backend orientation codes used to derive the initial affine when `init_affine_path` is not supplied. These are validated before registration starts and must use one axis from each pair `{R/L}`, `{A/P}`, and `{S/I}`.
- `transformation_graph.script_path`: explicit path to the external `emlddmm` package's `transformation_graph_v01.py`. If omitted, step 5 tries to resolve it automatically from the installed package and only falls back to workspace-local development copies.
- `outputs.write_qc_report`: writes `registration_report.html` and `registration_report.json` in the step-5 output directory.

Step-5 output metadata now also includes:
- `resolved_run_plan.json.schema_version`
- `registration_summary.json.schema_version`
- `registration_summary.json.orientation_resolution`
- `registration_summary.json.stage_timeline`
- always-on `run_provenance.json`
- always-on `reproduce_step5_command.txt`

## Creating Custom Configurations

### Project-Specific Config

Create a new YAML file for your project:

```yaml
# configs/my_project.yaml

# Inherit from default
_base_: default.yaml

# Override specific settings
segmentation:
  backend: tiatoolbox
  min_area_px: 1000

output:
  format: both
  generate_qc: true

mlflow:
  experiment_name: "my-project"
  log_artifacts: true

# Custom paths
paths:
  input_root: "/data/my_project/raw"
  output_root: "/data/my_project/processed"
```

### Specimen-Specific Config

```yaml
# configs/specimen_001.yaml

segmentation:
  # This specimen has unusually small tissue sections
  min_area_px: 200
  split_touching: false

tile:
  # Need more margin for registration
  extra_margin_px: 200
```

## Configuration Validation

The pipeline validates all configuration values:

```python
from wsi_pipeline.config import PipelineConfig
from pydantic import ValidationError

try:
    config = PipelineConfig(
        segmentation={"backend": "invalid_backend"}
    )
except ValidationError as e:
    print(f"Invalid configuration: {e}")
```

## Best Practices

1. **Start with defaults** - Use `default.yaml` and override only what's needed
2. **Version control configs** - Keep configuration files in your repository
3. **Use environment variables** - For secrets and deployment-specific values
4. **Document custom settings** - Add comments explaining why settings were changed
5. **Test configurations** - Validate on sample data before full processing

## Configuration Reference

### Complete Example

```yaml
# Full configuration with all options
segmentation:
  backend: "local-entropy"
  target_long_side: 2000
  min_area_px: 500
  struct_elem_px: 5
  split_touching: true
  r_split: 3.0

tile:
  chunk_size: 4096
  pad_multiple: 16
  extra_margin_px: 100

output:
  format: "ome-zarr"
  compression: "zlib"
  convert_to_uint8: true
  generate_qc: true

mlflow:
  enabled: true
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "wsi-tissue-pipeline"
  run_name_template: "{specimen}_{timestamp}"
  log_artifacts: false

colab:
  mount_drive: true
  use_gpu: true
  install_dependencies: true
```
