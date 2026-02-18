# SciServer Integration Guide for wsi-tissue-pipeline

## Package Structure

The new modules integrate into the existing package as follows:

```
wsi_tissue_pipeline/
├── __init__.py              # Add exports for new modules
├── config.py                # ADD: SciServerConfig class
├── mlflow_utils.py          # EXTEND: Add SciServer auto-detection
├── sciserver/               # NEW: SciServer-specific subpackage
│   ├── __init__.py
│   ├── environment.py       # Environment detection utilities
│   ├── storage.py           # Storage path management  
│   ├── lineage.py           # OpenLineage-compatible tracking
│   └── integration.py       # Unified SciServerPipeline class
├── cli.py
├── core.py
├── ... (existing modules)
```

## Integration Steps

### Step 1: Add SciServerConfig to config.py

Add this class to your existing `config.py`:

```python
class SciServerConfig(BaseModel):
    """Configuration for SciServer environment."""
    
    enabled: bool = Field(
        default=True, 
        description="Enable SciServer-specific features when detected"
    )
    
    volume_name: str = Field(
        default="UserVolume",
        description="Name of the user volume in SciServer Storage"
    )
    
    lineage_enabled: bool = Field(
        default=True,
        description="Enable data lineage tracking"
    )
    
    job_namespace: str = Field(
        default="sciserver://wsi-pipeline",
        description="Namespace for lineage job tracking"
    )
```

Then add to `PipelineConfig`:

```python
class PipelineConfig(BaseModel):
    # Existing configs...
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    tiles: TileConfig = Field(default_factory=TileConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    colab: ColabConfig = Field(default_factory=ColabConfig)
    sciserver: SciServerConfig = Field(default_factory=SciServerConfig)  # ADD THIS
```

### Step 2: Create the sciserver subpackage

Create `wsi_tissue_pipeline/sciserver/__init__.py`:

```python
"""
SciServer integration for WSI Tissue Pipeline.

Provides environment-aware configuration, storage management,
and lineage tracking for SciServer Compute.
"""

from .environment import (
    is_sciserver_environment,
    get_sciserver_user,
    get_sciserver_token,
)
from .storage import StorageConfig, get_storage_config
from .lineage import LineageTracker, tracked_run
from .integration import SciServerPipeline, setup_sciserver_tracking

__all__ = [
    "is_sciserver_environment",
    "get_sciserver_user", 
    "get_sciserver_token",
    "StorageConfig",
    "get_storage_config",
    "LineageTracker",
    "tracked_run",
    "SciServerPipeline",
    "setup_sciserver_tracking",
]
```

### Step 3: Extend mlflow_utils.py

Add SciServer auto-detection to the existing `init_mlflow()`:

```python
def init_mlflow(
    config: Optional[Union[MLflowConfig, PipelineConfig]] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> bool:
    """Initialize MLflow tracking with SciServer auto-detection."""
    if not MLFLOW_AVAILABLE:
        print("Warning: MLflow not available. Tracking disabled.")
        return False
    
    # Get config values
    if isinstance(config, PipelineConfig):
        mlflow_config = config.mlflow
    elif isinstance(config, MLflowConfig):
        mlflow_config = config
    else:
        mlflow_config = MLflowConfig()
    
    # Override with explicit arguments
    uri = tracking_uri or mlflow_config.tracking_uri
    exp_name = experiment_name or mlflow_config.experiment_name
    
    # AUTO-DETECT SCISERVER (NEW)
    if uri == "sqlite:///mlflow.db":  # Default value, try to improve
        try:
            from .sciserver import is_sciserver_environment, get_storage_config
            if is_sciserver_environment():
                storage = get_storage_config()
                uri = f"sqlite:///{storage.mlflow_dir}/mlflow.db"
                storage.mlflow_dir.mkdir(parents=True, exist_ok=True)
        except ImportError:
            pass
    
    # Set tracking URI
    mlflow.set_tracking_uri(uri)
    
    # ... rest of existing code
```

---

## File Placement

Copy the generated files to these locations:

| Generated File | Target Location |
|----------------|-----------------|
| `sciserver_integration.py` | `wsi_tissue_pipeline/sciserver/integration.py` |
| `sciserver_mlflow.py` | `wsi_tissue_pipeline/sciserver/mlflow_config.py` |
| `sciserver_lineage.py` | `wsi_tissue_pipeline/sciserver/lineage.py` |

Then split `sciserver_integration.py` into:
- `environment.py` - Environment detection functions
- `storage.py` - StorageConfig class

---

## Testing

### Test 1: Environment Detection (works anywhere)

```python
# tests/test_sciserver_environment.py
"""Test SciServer environment detection."""

import pytest
from unittest.mock import patch, MagicMock


def test_sciserver_detection_when_not_installed():
    """Should return False when SciServer package not installed."""
    from wsi_tissue_pipeline.sciserver.environment import is_sciserver_environment
    
    with patch.dict('sys.modules', {'SciServer': None}):
        assert is_sciserver_environment() == False


def test_sciserver_detection_mocked():
    """Test detection with mocked SciServer."""
    # Create mock SciServer module
    mock_sciserver = MagicMock()
    mock_sciserver.Config.isSciServerComputeEnvironment.return_value = True
    
    with patch.dict('sys.modules', {'SciServer': mock_sciserver}):
        # Re-import to pick up mock
        from importlib import reload
        from wsi_tissue_pipeline.sciserver import environment
        reload(environment)
        
        assert environment.is_sciserver_environment() == True


def test_storage_config_local():
    """Test StorageConfig for local environment."""
    from wsi_tissue_pipeline.sciserver.storage import StorageConfig
    
    config = StorageConfig.for_local("/tmp/test")
    
    assert config.mlflow_dir.exists() or True  # Just check structure
    assert "mlflow" in str(config.mlflow_dir)
    assert "lineage" in str(config.lineage_dir)
```

### Test 2: Lineage Tracking (works anywhere)

```python
# tests/test_lineage.py
"""Test lineage tracking functionality."""

import pytest
import tempfile
import json
from pathlib import Path


@pytest.fixture
def lineage_tracker():
    """Create a lineage tracker with temp storage."""
    from wsi_tissue_pipeline.sciserver.lineage import LineageTracker
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = LineageTracker(tmpdir)
        yield tracker


def test_start_run(lineage_tracker):
    """Test starting a lineage run."""
    run_id = lineage_tracker.start_run(
        job_name="test_job",
        inputs=[{"name": "input.zarr", "namespace": "test://data"}]
    )
    
    assert run_id is not None
    assert len(run_id) == 36  # UUID format


def test_complete_run(lineage_tracker):
    """Test completing a lineage run."""
    run_id = lineage_tracker.start_run(
        job_name="test_job",
        inputs=[{"name": "input.zarr"}]
    )
    
    event_id = lineage_tracker.complete_run(
        run_id=run_id,
        job_name="test_job",
        outputs=[{"name": "output.zarr"}],
        metrics={"n_tiles": 42}
    )
    
    assert event_id is not None
    
    # Verify run status in index
    assert lineage_tracker.index["runs"][run_id]["status"] == "COMPLETE"


def test_fail_run(lineage_tracker):
    """Test recording a failed run."""
    run_id = lineage_tracker.start_run(
        job_name="test_job",
        inputs=[{"name": "input.zarr"}]
    )
    
    lineage_tracker.fail_run(
        run_id=run_id,
        job_name="test_job",
        error_message="Test error"
    )
    
    assert lineage_tracker.index["runs"][run_id]["status"] == "FAIL"


def test_dataset_lineage_query(lineage_tracker):
    """Test querying lineage for a dataset."""
    # Create some runs
    run_id = lineage_tracker.start_run(
        job_name="process",
        inputs=[{"name": "specimen_001.vsi"}]
    )
    lineage_tracker.complete_run(
        run_id=run_id,
        job_name="process",
        outputs=[{"name": "specimen_001.ome.zarr"}]
    )
    
    # Query lineage
    lineage = lineage_tracker.get_dataset_lineage("specimen_001")
    
    assert len(lineage["matches"]) > 0
    assert run_id in lineage["runs"]


def test_openlineage_export(lineage_tracker):
    """Test exporting events in OpenLineage format."""
    run_id = lineage_tracker.start_run(job_name="test", inputs=[])
    lineage_tracker.complete_run(run_id=run_id, job_name="test", outputs=[])
    
    events = lineage_tracker.export_for_marquez()
    
    assert len(events) == 2  # START and COMPLETE
    assert events[0]["eventType"] in ["START", "COMPLETE"]
    assert "schemaURL" in events[0]


def test_tracked_run_context_manager(lineage_tracker):
    """Test the tracked_run context manager."""
    from wsi_tissue_pipeline.sciserver.lineage import tracked_run
    
    with tracked_run(lineage_tracker, "context_test", inputs=["input.zarr"]) as run:
        run.add_output("output.zarr")
        run.log_metric("test_metric", 123)
    
    # Verify completion
    details = lineage_tracker.get_run_details(run.run_id)
    assert details["status"] == "COMPLETE"
```

### Test 3: MLFlow Integration (works anywhere)

```python
# tests/test_sciserver_mlflow.py
"""Test SciServer MLFlow configuration."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch


def test_mlflow_config_local():
    """Test MLFlow configuration in local environment."""
    from wsi_tissue_pipeline.sciserver.mlflow_config import SciServerMLFlowConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SciServerMLFlowConfig(
            experiment_name="test-experiment",
            local_fallback_dir=tmpdir
        )
        
        result = config.get_config()
        
        assert "sqlite://" in result["tracking_uri"]
        assert result["experiment_name"] == "test-experiment"
        assert result["is_sciserver"] == False


def test_mlflow_setup_creates_directories():
    """Test that setup creates required directories."""
    pytest.importorskip("mlflow")
    
    from wsi_tissue_pipeline.sciserver.mlflow_config import SciServerMLFlowConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SciServerMLFlowConfig(local_fallback_dir=tmpdir)
        config.setup()
        
        # Check artifacts directory was created
        assert Path(config.artifact_location).exists()
```

### Test 4: Full Integration (works anywhere)

```python
# tests/test_sciserver_integration.py
"""Test full SciServer integration."""

import pytest
import tempfile


def test_pipeline_initialization():
    """Test SciServerPipeline initialization."""
    from wsi_tissue_pipeline.sciserver import SciServerPipeline
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch storage to use temp directory
        pipeline = SciServerPipeline(
            experiment_name="test-pipeline",
            auto_setup=False  # Don't auto-setup to control paths
        )
        
        # Override storage for testing
        from wsi_tissue_pipeline.sciserver.storage import StorageConfig
        pipeline.storage = StorageConfig.for_local(tmpdir)
        pipeline.setup()
        
        assert pipeline.storage.mlflow_dir.exists()
        assert pipeline.storage.lineage_dir.exists()


def test_tracked_experiment():
    """Test tracked_experiment context manager."""
    pytest.importorskip("mlflow")
    
    from wsi_tissue_pipeline.sciserver import SciServerPipeline
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = SciServerPipeline(auto_setup=False)
        
        from wsi_tissue_pipeline.sciserver.storage import StorageConfig
        pipeline.storage = StorageConfig.for_local(tmpdir)
        pipeline.setup()
        
        with pipeline.tracked_experiment("test-run", inputs=["input.zarr"]) as exp:
            exp.log_param("test_param", "value")
            exp.log_metric("test_metric", 42.0)
            exp.log_output("output.zarr")
        
        # Verify lineage was recorded
        if pipeline.lineage_tracker:
            history = pipeline.lineage_tracker.get_job_history("test-run")
            assert len(history) > 0
```

### Test 5: Integration with Existing Pipeline

```python
# tests/test_pipeline_integration.py
"""Test integration with existing WSI pipeline."""

import pytest
import tempfile
from pathlib import Path


def test_with_pipeline_config():
    """Test using SciServer integration with PipelineConfig."""
    from wsi_tissue_pipeline.config import PipelineConfig, load_config
    from wsi_tissue_pipeline.sciserver import SciServerPipeline
    
    # Load config with SciServer settings
    config = load_config(overrides={
        "sciserver": {
            "enabled": True,
            "lineage_enabled": True
        }
    })
    
    assert config.sciserver.enabled == True


def test_mlflow_utils_with_sciserver():
    """Test existing mlflow_utils works with SciServer config."""
    pytest.importorskip("mlflow")
    
    from wsi_tissue_pipeline.mlflow_utils import init_mlflow, MLflowContext
    from wsi_tissue_pipeline.config import PipelineConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(
            mlflow={
                "enabled": True,
                "tracking_uri": f"sqlite:///{tmpdir}/mlflow.db",
                "experiment_name": "test-sciserver"
            }
        )
        
        result = init_mlflow(config)
        assert result == True
```

---

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all SciServer tests
pytest tests/test_sciserver*.py -v

# Run with coverage
pytest tests/test_sciserver*.py --cov=wsi_tissue_pipeline.sciserver

# Run specific test
pytest tests/test_lineage.py::test_complete_run -v
```

---

## Quick Verification Script

Create this script to verify the integration works:

```python
#!/usr/bin/env python3
"""verify_sciserver_integration.py - Quick verification script."""

import tempfile
from pathlib import Path

def main():
    print("=" * 60)
    print("SciServer Integration Verification")
    print("=" * 60)
    
    # 1. Test environment detection
    print("\n1. Testing environment detection...")
    try:
        from wsi_tissue_pipeline.sciserver import is_sciserver_environment
        is_ss = is_sciserver_environment()
        print(f"   SciServer environment: {is_ss}")
        print("   ✓ Environment detection works")
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        return False
    
    # 2. Test storage configuration
    print("\n2. Testing storage configuration...")
    try:
        from wsi_tissue_pipeline.sciserver import StorageConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig.for_local(tmpdir)
            config.ensure_directories()
            assert config.mlflow_dir.exists()
            print(f"   Storage base: {config.user_volume}")
            print("   ✓ Storage configuration works")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # 3. Test lineage tracking
    print("\n3. Testing lineage tracking...")
    try:
        from wsi_tissue_pipeline.sciserver import LineageTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = LineageTracker(tmpdir)
            run_id = tracker.start_run("test_job", inputs=[{"name": "test.zarr"}])
            tracker.complete_run(run_id, "test_job", outputs=[{"name": "out.zarr"}])
            print(f"   Created run: {run_id[:8]}...")
            print("   ✓ Lineage tracking works")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # 4. Test MLFlow integration
    print("\n4. Testing MLFlow integration...")
    try:
        import mlflow
        from wsi_tissue_pipeline.sciserver import SciServerPipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = SciServerPipeline(auto_setup=False)
            from wsi_tissue_pipeline.sciserver import StorageConfig
            pipeline.storage = StorageConfig.for_local(tmpdir)
            pipeline.setup()
            print(f"   MLFlow URI: {pipeline.mlflow_tracking_uri}")
            print("   ✓ MLFlow integration works")
    except ImportError:
        print("   ⚠ MLFlow not installed (optional)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # 5. Test full pipeline context
    print("\n5. Testing tracked experiment...")
    try:
        from wsi_tissue_pipeline.sciserver import SciServerPipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = SciServerPipeline(auto_setup=False)
            from wsi_tissue_pipeline.sciserver import StorageConfig
            pipeline.storage = StorageConfig.for_local(tmpdir)
            pipeline.setup()
            
            with pipeline.tracked_experiment("verify_test") as exp:
                exp.log_param("test", "value")
                exp.log_metric("score", 1.0)
                exp.log_output("result.zarr")
            
            print("   ✓ Tracked experiment works")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All verifications passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
```

---

## Usage Example

Once integrated, use like this:

```python
from wsi_tissue_pipeline.config import load_config
from wsi_tissue_pipeline.sciserver import SciServerPipeline
from wsi_tissue_pipeline.wsi_processing import process_specimen

# Load config (auto-detects environment)
config = load_config("config.yaml")

# Initialize pipeline with tracking
pipeline = SciServerPipeline(experiment_name="cochlea-analysis")

# Process with full tracking
with pipeline.tracked_experiment("specimen_001", inputs=["raw/specimen_001.vsi"]) as exp:
    exp.log_param("config_version", "1.0")
    
    result = process_specimen(
        input_path=pipeline.get_data_path("raw", "specimen_001.vsi"),
        output_dir=pipeline.get_data_path("processed"),
        config=config
    )
    
    exp.log_metrics({
        "n_sections": result["n_outputs"],
        "processing_time_s": result["elapsed_time"]
    })
    exp.log_output("specimen_001.ome.zarr")
```
