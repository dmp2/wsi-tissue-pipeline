# SciServer MLOps Architecture Compatibility Analysis

## Executive Summary

This document analyzes the compatibility of the proposed MLOps stack with SciServer's infrastructure, identifying required adaptations and recommended deployment patterns for the WSI tissue pipeline.

---

## Proposed Stack vs. SciServer Capabilities

### 1. Workflow Orchestration: Prefect/Dagster

**SciServer Context:**
- Compute environment runs Jupyter notebooks in server-side Docker containers
- No native support for external orchestrators (Prefect Server, Dagster Daemon)
- Batch job submission via `Jobs.submitShellCommandJob()` or notebook execution

**Compatibility Assessment: ⚠️ PARTIAL**

| Feature | Prefect/Dagster | SciServer Alternative |
|---------|-----------------|----------------------|
| DAG Definition | Python decorators | Jupyter notebooks with cell dependencies |
| Scheduling | Native scheduler | SciServer Job Queue (manual/programmatic) |
| Distributed Execution | Workers/Agents | Compute containers (no horizontal scaling) |
| State Management | Backend DB | Must use CasJobs/SciQuery or file-based |
| UI Dashboard | Web UI | None native; custom logging required |

**Recommended Adaptation:**
```python
# Option A: Prefect in "local" mode within Compute container
from prefect import flow, task

@task
def process_slide(zarr_path: str) -> dict:
    # Processing logic
    pass

@flow
def wsi_pipeline(input_dir: str):
    # Flow runs entirely within single Compute job
    pass

# Option B: Native SciServer job chaining
from SciServer import Jobs

def submit_pipeline_stage(stage: str, params: dict):
    script = f"python run_stage.py --stage {stage} --params '{json.dumps(params)}'"
    job_id = Jobs.submitShellCommandJob(
        shellCommand=script,
        dockerComputeDomain="Science Pipelines",
        dockerImageName="wsi-pipeline:latest",
        userVolumes=[{"name": "Storage", "rootVolumeName": "Storage"}]
    )
    return job_id
```

**Recommendation:** Use Prefect in **local execution mode** within Compute containers for complex DAGs, or implement lightweight job chaining via SciServer Jobs API for simpler workflows.

---

### 2. Experiment Tracking: MLFlow

**SciServer Context:**
- No managed MLFlow server
- Persistent storage available at `/home/idies/workspace/Storage/<username>/<volume>/`
- CasJobs provides SQL database access

**Compatibility Assessment: ✅ COMPATIBLE (with configuration)**

**Deployment Options:**

| Option | Tracking URI | Artifact Store | Pros | Cons |
|--------|-------------|----------------|------|------|
| File-based | `file:///home/idies/workspace/Storage/.../mlruns` | Same path | Simple, no setup | No concurrent access UI |
| SQLite + Files | `sqlite:///...mlflow.db` | File path | Queryable | Single-writer limitation |
| CasJobs Backend | Custom adapter | SciDrive | Shared access | Requires custom implementation |

**Recommended Configuration:**
```python
# mlflow_sciserver.py
import os
from pathlib import Path

def get_mlflow_config():
    """Configure MLFlow for SciServer environment."""
    from SciServer import Config, Authentication
    
    if Config.isSciServerComputeEnvironment():
        # Use persistent storage
        base_path = Path("/home/idies/workspace/Storage")
        username = Authentication.getKeystoneUserWithToken().userName
        
        mlflow_dir = base_path / username / "UserVolume" / "mlflow"
        mlflow_dir.mkdir(parents=True, exist_ok=True)
        
        tracking_uri = f"sqlite:///{mlflow_dir}/mlflow.db"
        artifact_location = str(mlflow_dir / "artifacts")
        
        return {
            "tracking_uri": tracking_uri,
            "artifact_location": artifact_location,
            "experiment_name": "wsi-tissue-pipeline"
        }
    else:
        # Local development fallback
        return {
            "tracking_uri": "sqlite:///mlflow.db",
            "artifact_location": "./mlruns",
            "experiment_name": "wsi-tissue-pipeline-dev"
        }

# Usage in pipeline
import mlflow
config = get_mlflow_config()
mlflow.set_tracking_uri(config["tracking_uri"])
mlflow.set_experiment(config["experiment_name"])
```

---

### 3. Data Versioning: DVC + Zarr

**SciServer Context:**
- Large file storage via mounted volumes
- SciDrive for cloud-like file storage with public URLs
- No native Git LFS or object storage (S3/GCS)

**Compatibility Assessment: ⚠️ PARTIAL**

**DVC Remote Options:**

| Remote Type | SciServer Support | Notes |
|-------------|-------------------|-------|
| Local filesystem | ✅ Full | `/home/idies/workspace/Storage/` |
| SSH | ❌ None | No SSH access between containers |
| S3/GCS | ❌ None | No cloud object storage |
| HTTP | ⚠️ Limited | SciDrive public URLs (read-only) |
| Custom | ✅ Possible | Implement SciDrive adapter |

**Recommended Adaptation:**
```yaml
# .dvc/config for SciServer
[core]
    remote = sciserver

[remote "sciserver"]
    url = /home/idies/workspace/Storage/{username}/wsi-data/dvc-cache
```

```python
# Zarr versioning without DVC (alternative approach)
# Use directory-based versioning with metadata tracking

from pathlib import Path
import json
from datetime import datetime
import hashlib

class ZarrVersionManager:
    """Simple version control for Zarr datasets on SciServer."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.versions_file = self.base_path / ".versions.json"
        self._load_versions()
    
    def _load_versions(self):
        if self.versions_file.exists():
            self.versions = json.loads(self.versions_file.read_text())
        else:
            self.versions = {"datasets": {}, "current": {}}
    
    def register_dataset(self, name: str, zarr_path: str, metadata: dict = None):
        """Register a new version of a dataset."""
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        entry = {
            "version_id": version_id,
            "path": str(zarr_path),
            "created": datetime.now().isoformat(),
            "metadata": metadata or {},
            "checksum": self._compute_checksum(zarr_path)
        }
        
        if name not in self.versions["datasets"]:
            self.versions["datasets"][name] = []
        
        self.versions["datasets"][name].append(entry)
        self.versions["current"][name] = version_id
        self._save_versions()
        
        return version_id
    
    def _compute_checksum(self, zarr_path: str) -> str:
        """Compute checksum of zarr .zattrs for quick validation."""
        zattrs = Path(zarr_path) / ".zattrs"
        if zattrs.exists():
            return hashlib.md5(zattrs.read_bytes()).hexdigest()
        return "unknown"
    
    def _save_versions(self):
        self.versions_file.write_text(json.dumps(self.versions, indent=2))
```

**Recommendation:** Use **local DVC remote** pointing to persistent storage, or implement lightweight Zarr version tracking via JSON manifests stored alongside datasets.

---

### 4. Data Contracts: Pydantic

**SciServer Context:**
- Python environment supports Pydantic
- No constraints on validation libraries

**Compatibility Assessment: ✅ FULLY COMPATIBLE**

Pydantic works unchanged. The existing `config.py` implementation is already well-suited.

---

### 5. Lineage Tracking: OpenLineage → Marquez

**SciServer Context:**
- No managed Marquez server
- Cannot run persistent services
- CasJobs available for structured storage

**Compatibility Assessment: ⚠️ REQUIRES ADAPTATION**

**Options:**

| Approach | Implementation | Trade-offs |
|----------|---------------|------------|
| File-based lineage | JSON files in storage | Simple, no server needed |
| CasJobs storage | Lineage events in SQL tables | Queryable, shared access |
| External Marquez | Connect to external server | Requires network access |

**Recommended Adaptation: Lightweight Lineage Tracker**
```python
# lineage_tracker.py
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import json
import uuid

@dataclass
class DatasetRef:
    namespace: str
    name: str
    version: Optional[str] = None

@dataclass
class LineageEvent:
    event_id: str
    event_type: str  # "START", "COMPLETE", "FAIL"
    job_name: str
    job_namespace: str
    inputs: List[DatasetRef]
    outputs: List[DatasetRef]
    timestamp: str
    metadata: dict

class SciServerLineageTracker:
    """OpenLineage-compatible event tracking for SciServer."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.events_dir = self.storage_path / "lineage_events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
    
    def emit_event(
        self,
        event_type: str,
        job_name: str,
        inputs: List[dict],
        outputs: List[dict],
        metadata: dict = None
    ) -> str:
        """Emit a lineage event."""
        event = LineageEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            job_name=job_name,
            job_namespace="sciserver/wsi-pipeline",
            inputs=[DatasetRef(**i) for i in inputs],
            outputs=[DatasetRef(**o) for o in outputs],
            timestamp=datetime.utcnow().isoformat() + "Z",
            metadata=metadata or {}
        )
        
        # Store event
        event_file = self.events_dir / f"{event.event_id}.json"
        event_file.write_text(json.dumps(asdict(event), indent=2))
        
        return event.event_id
    
    def start_run(self, job_name: str, inputs: List[dict]) -> str:
        """Record job start."""
        return self.emit_event("START", job_name, inputs, [])
    
    def complete_run(self, job_name: str, inputs: List[dict], outputs: List[dict], metrics: dict = None):
        """Record successful job completion."""
        return self.emit_event("COMPLETE", job_name, inputs, outputs, {"metrics": metrics or {}})
    
    def fail_run(self, job_name: str, inputs: List[dict], error: str):
        """Record job failure."""
        return self.emit_event("FAIL", job_name, inputs, [], {"error": error})
    
    def query_lineage(self, dataset_name: str) -> List[dict]:
        """Find all events involving a dataset."""
        results = []
        for event_file in self.events_dir.glob("*.json"):
            event = json.loads(event_file.read_text())
            
            # Check inputs and outputs
            for ds in event.get("inputs", []) + event.get("outputs", []):
                if ds.get("name") == dataset_name:
                    results.append(event)
                    break
        
        return sorted(results, key=lambda e: e["timestamp"])
```

---

### 6. Model Serving: FastAPI + BentoML

**SciServer Context:**
- Compute containers are not publicly accessible
- No ingress/load balancer for custom services
- Jobs are batch-oriented, not long-running services

**Compatibility Assessment: ❌ NOT DIRECTLY COMPATIBLE**

**Alternative Approaches:**

| Approach | Description | Use Case |
|----------|-------------|----------|
| Batch inference | Run inference as scheduled jobs | Periodic processing |
| SciDrive + polling | Upload inputs, poll for outputs | Async external access |
| External deployment | Deploy serving outside SciServer | Production serving |
| Notebook endpoints | Interactive Jupyter-based inference | Development/demo |

**Recommended Pattern for SciServer:**
```python
# batch_inference.py - Batch processing pattern
from pathlib import Path
import json

class BatchInferenceProcessor:
    """Process inference requests from a queue directory."""
    
    def __init__(self, queue_dir: str, results_dir: str):
        self.queue_dir = Path(queue_dir)
        self.results_dir = Path(results_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def submit_request(self, request_id: str, input_data: dict):
        """Submit an inference request."""
        request_file = self.queue_dir / f"{request_id}.json"
        request_file.write_text(json.dumps({
            "request_id": request_id,
            "status": "pending",
            "input": input_data,
            "submitted_at": datetime.utcnow().isoformat()
        }))
    
    def process_queue(self, model, max_requests: int = 100):
        """Process pending requests (run as batch job)."""
        pending = list(self.queue_dir.glob("*.json"))[:max_requests]
        
        for request_file in pending:
            request = json.loads(request_file.read_text())
            
            try:
                # Run inference
                result = model.predict(request["input"])
                
                # Write result
                result_file = self.results_dir / f"{request['request_id']}.json"
                result_file.write_text(json.dumps({
                    "request_id": request["request_id"],
                    "status": "completed",
                    "result": result,
                    "completed_at": datetime.utcnow().isoformat()
                }))
                
                # Remove from queue
                request_file.unlink()
                
            except Exception as e:
                # Mark as failed
                result_file = self.results_dir / f"{request['request_id']}.json"
                result_file.write_text(json.dumps({
                    "request_id": request["request_id"],
                    "status": "failed",
                    "error": str(e)
                }))
```

**Recommendation:** For SciServer, use **batch inference patterns** with file-based queuing. For production serving requiring real-time inference, deploy FastAPI/BentoML externally and use SciServer for training/preprocessing only.

---

### 7. Containerization: Docker

**SciServer Context:**
- Compute runs in pre-built Docker containers
- Custom containers supported via "Compute Domain" configuration
- No Docker-in-Docker capability

**Compatibility Assessment: ✅ COMPATIBLE (with registration)**

**SciServer Container Requirements:**
1. Base image must be compatible with SciServer Compute
2. Container must be registered with the SciServer team
3. User volumes are mounted at runtime

**Recommended Dockerfile:**
```dockerfile
# Dockerfile.sciserver
FROM sciserver/jupyter:latest

# Install pipeline dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install pipeline package
COPY wsi_pipeline/ /opt/wsi_pipeline/
RUN pip install -e /opt/wsi_pipeline/

# Set up MLFlow
ENV MLFLOW_TRACKING_URI=/home/idies/workspace/Storage/${USER}/mlflow/mlflow.db

# Entry point for batch jobs
COPY scripts/entrypoint.sh /opt/
RUN chmod +x /opt/entrypoint.sh
```

---

## 8. Storage Strategy

**Available Storage Options:**

| Location | Type | Size Limit | Persistence | Access Pattern |
|----------|------|------------|-------------|----------------|
| `/home/idies/workspace/Storage/<user>/<vol>/` | Persistent | Per-allocation | Permanent | Read/Write |
| `/home/idies/workspace/Temporary/<user>/<vol>/` | Temporary | Per-allocation | Session-scoped | Read/Write |
| `/home/idies/workspace/Scratch/` | Shared scratch | Large | Periodic cleanup | Read/Write |
| CasJobs MyDB | Database | 500 GB | Permanent | SQL queries |
| SciDrive | Cloud storage | Per-allocation | Permanent | HTTP/API |

**Recommended Storage Layout:**
```
/home/idies/workspace/Storage/<username>/wsi-pipeline/
├── config/
│   ├── pipeline_config.yaml
│   └── .env
├── data/
│   ├── raw/                    # Raw VSI/ETS files (or symlinks)
│   ├── processed/              # OME-Zarr outputs
│   │   ├── specimen_001.ome.zarr/
│   │   └── specimen_002.ome.zarr/
│   └── versions.json           # Dataset version tracking
├── mlflow/
│   ├── mlflow.db              # SQLite tracking
│   └── artifacts/             # MLFlow artifacts
├── lineage/
│   └── events/                # Lineage event JSONs
├── models/
│   ├── segmentation_v1/
│   └── registry.json
└── outputs/
    ├── qc_grids/
    └── reports/
```

---

## 9. Authentication & Automation

**SciServer Authentication:**
- Keystone-based SSO
- Automatic within Compute environment
- Token accessible via `Authentication.getToken()`

**For Automated Workflows:**
```python
# auth_utils.py
from SciServer import Authentication, Config
import os

def ensure_authenticated():
    """Ensure valid authentication for pipeline execution."""
    if Config.isSciServerComputeEnvironment():
        # Automatic authentication in Compute
        token = Authentication.getToken()
        if not token:
            raise RuntimeError("No authentication token available in Compute environment")
        return token
    else:
        # External execution - require credentials
        username = os.environ.get("SCISERVER_USERNAME")
        password = os.environ.get("SCISERVER_PASSWORD")
        
        if not username or not password:
            raise RuntimeError(
                "Set SCISERVER_USERNAME and SCISERVER_PASSWORD environment variables"
            )
        
        return Authentication.login(username, password)

def get_user_storage_path() -> str:
    """Get the user's persistent storage path."""
    ensure_authenticated()
    user = Authentication.getKeystoneUserWithToken()
    return f"/home/idies/workspace/Storage/{user.userName}/UserVolume"
```

---

## Implementation Recommendations

### Phase 1: Core Pipeline (Week 1-2)
1. ✅ Pydantic configs (already done)
2. ✅ Basic processing pipeline (already done)
3. 🔄 Add MLFlow tracking with SciServer config
4. 🔄 Implement file-based lineage tracking

### Phase 2: Data Management (Week 3-4)
1. 🔄 Set up DVC with local remote
2. 🔄 Implement Zarr version manager
3. 🔄 Create storage layout script

### Phase 3: Workflow Orchestration (Week 5-6)
1. 🔄 Implement Prefect flows (local mode)
2. 🔄 Create job chaining utilities
3. 🔄 Add retry/checkpoint logic

### Phase 4: Deployment (Week 7-8)
1. 🔄 Build SciServer-compatible Docker image
2. 🔄 Register with SciServer team
3. 🔄 Create batch inference patterns
4. 🔄 Documentation and training

---

## Summary Matrix

| Component | Original Choice | SciServer Compatibility | Recommended Approach |
|-----------|-----------------|------------------------|---------------------|
| Orchestration | Prefect/Dagster | ⚠️ Partial | Prefect local mode + Jobs API |
| Tracking | MLFlow | ✅ Compatible | SQLite + file artifacts |
| Versioning | DVC | ⚠️ Partial | Local remote + Zarr manifests |
| Contracts | Pydantic | ✅ Full | No changes needed |
| Lineage | OpenLineage/Marquez | ⚠️ Partial | Custom file-based tracker |
| Serving | FastAPI/BentoML | ❌ Limited | Batch inference patterns |
| Containers | Docker | ✅ Compatible | Register custom image |

---

## Next Steps

1. **Immediate:** Implement MLFlow SciServer configuration
2. **Short-term:** Create lineage tracker module
3. **Medium-term:** Build job orchestration utilities
4. **Long-term:** Register custom Docker image with SciServer

This analysis provides a roadmap for adapting the proposed MLOps stack to work effectively within SciServer's constraints while maintaining best practices for reproducibility and experiment tracking.
