"""
SciServer Lineage Tracking

Provides OpenLineage-compatible event tracking for data lineage
within SciServer Compute environment.

This module implements a lightweight, file-based lineage tracker
that follows OpenLineage event schemas but stores events locally
rather than requiring a Marquez server.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from enum import Enum


class EventType(str, Enum):
    """OpenLineage event types."""
    START = "START"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"
    ABORT = "ABORT"


@dataclass
class DatasetFacets:
    """Facets describing a dataset."""
    schema_fields: List[Dict[str, str]] = field(default_factory=list)
    data_source: Optional[str] = None
    storage_format: Optional[str] = None
    size_bytes: Optional[int] = None
    row_count: Optional[int] = None
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    """
    OpenLineage Dataset representation.
    
    Attributes
    ----------
    namespace : str
        Dataset namespace (e.g., "sciserver://storage/username").
    name : str
        Dataset name (e.g., "specimen_001.ome.zarr").
    facets : DatasetFacets, optional
        Additional dataset metadata.
    """
    namespace: str
    name: str
    facets: Optional[DatasetFacets] = None
    
    def to_dict(self) -> Dict:
        d = {"namespace": self.namespace, "name": self.name}
        if self.facets:
            d["facets"] = asdict(self.facets)
        return d
    
    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        namespace_prefix: str = "sciserver://storage"
    ) -> "Dataset":
        """Create dataset reference from file path."""
        path = Path(path)
        return cls(
            namespace=namespace_prefix,
            name=str(path.name)
        )


@dataclass 
class JobFacets:
    """Facets describing a job."""
    source_code_location: Optional[str] = None
    documentation: Optional[str] = None
    sql: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    """
    OpenLineage Job representation.
    
    Attributes
    ----------
    namespace : str
        Job namespace (e.g., "sciserver://wsi-pipeline").
    name : str
        Job name (e.g., "process_specimen").
    facets : JobFacets, optional
        Additional job metadata.
    """
    namespace: str
    name: str
    facets: Optional[JobFacets] = None
    
    def to_dict(self) -> Dict:
        d = {"namespace": self.namespace, "name": self.name}
        if self.facets:
            d["facets"] = asdict(self.facets)
        return d


@dataclass
class RunFacets:
    """Facets describing a run."""
    nominal_time: Optional[str] = None
    parent_run: Optional[Dict[str, str]] = None
    error_message: Optional[str] = None
    processing_engine: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Run:
    """
    OpenLineage Run representation.
    
    Attributes
    ----------
    run_id : str
        Unique run identifier (UUID).
    facets : RunFacets, optional
        Additional run metadata.
    """
    run_id: str
    facets: Optional[RunFacets] = None
    
    def to_dict(self) -> Dict:
        d = {"runId": self.run_id}
        if self.facets:
            d["facets"] = asdict(self.facets)
        return d


@dataclass
class LineageEvent:
    """
    OpenLineage Event representation.
    
    This follows the OpenLineage spec for interoperability
    with tools like Marquez, while storing events locally.
    """
    event_type: EventType
    event_time: str
    run: Run
    job: Job
    inputs: List[Dataset]
    outputs: List[Dataset]
    producer: str = "sciserver-wsi-pipeline"
    schema_url: str = "https://openlineage.io/spec/1-0-5/OpenLineage.json"
    
    def to_dict(self) -> Dict:
        return {
            "eventType": self.event_type.value,
            "eventTime": self.event_time,
            "run": self.run.to_dict(),
            "job": self.job.to_dict(),
            "inputs": [ds.to_dict() for ds in self.inputs],
            "outputs": [ds.to_dict() for ds in self.outputs],
            "producer": self.producer,
            "schemaURL": self.schema_url
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class LineageTracker:
    """
    File-based lineage tracker for SciServer.
    
    Tracks data lineage by emitting OpenLineage-compatible events
    to JSON files. Events can later be:
    - Queried locally for lineage visualization
    - Exported to Marquez or other OpenLineage backends
    - Analyzed for data provenance
    
    Parameters
    ----------
    storage_path : str or Path
        Base path for lineage event storage.
    job_namespace : str
        Default namespace for jobs.
    dataset_namespace : str
        Default namespace for datasets.
    
    Examples
    --------
    >>> tracker = LineageTracker("/data/lineage")
    >>> 
    >>> # Start a processing run
    >>> run_id = tracker.start_run(
    ...     job_name="process_specimen",
    ...     inputs=[{"name": "raw_001.vsi", "namespace": "sciserver://raw"}]
    ... )
    >>> 
    >>> # ... do processing ...
    >>> 
    >>> # Complete the run
    >>> tracker.complete_run(
    ...     run_id=run_id,
    ...     job_name="process_specimen",
    ...     outputs=[{"name": "specimen_001.ome.zarr", "namespace": "sciserver://processed"}],
    ...     metrics={"n_tiles": 42, "processing_time_s": 120.5}
    ... )
    """
    
    def __init__(
        self,
        storage_path: Union[str, Path],
        job_namespace: str = "sciserver://wsi-pipeline",
        dataset_namespace: str = "sciserver://storage"
    ):
        self.storage_path = Path(storage_path)
        self.events_dir = self.storage_path / "events"
        self.runs_dir = self.storage_path / "runs"
        self.job_namespace = job_namespace
        self.dataset_namespace = dataset_namespace
        
        # Ensure directories exist
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Index file for fast lookups
        self.index_file = self.storage_path / "index.json"
        self._load_index()
    
    def _load_index(self):
        """Load or create the event index."""
        if self.index_file.exists():
            self.index = json.loads(self.index_file.read_text())
        else:
            self.index = {
                "runs": {},  # run_id -> {job, status, events}
                "datasets": {},  # dataset_name -> [run_ids]
                "jobs": {}  # job_name -> [run_ids]
            }
    
    def _save_index(self):
        """Persist the event index."""
        self.index_file.write_text(json.dumps(self.index, indent=2))
    
    def _now_iso(self) -> str:
        """Get current time in ISO format."""
        return datetime.utcnow().isoformat() + "Z"
    
    def _emit_event(self, event: LineageEvent) -> str:
        """Write event to storage and update index."""
        # Save event file
        event_id = f"{event.run.run_id}_{event.event_type.value}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        event_file = self.events_dir / f"{event_id}.json"
        event_file.write_text(event.to_json())
        
        # Update index
        run_id = event.run.run_id
        job_name = event.job.name
        
        # Update runs index
        if run_id not in self.index["runs"]:
            self.index["runs"][run_id] = {
                "job": job_name,
                "status": event.event_type.value,
                "events": [],
                "inputs": [],
                "outputs": []
            }
        
        self.index["runs"][run_id]["events"].append(event_id)
        self.index["runs"][run_id]["status"] = event.event_type.value
        
        # Track datasets
        for ds in event.inputs:
            ds_key = f"{ds.namespace}:{ds.name}"
            if ds_key not in self.index["datasets"]:
                self.index["datasets"][ds_key] = []
            if run_id not in self.index["datasets"][ds_key]:
                self.index["datasets"][ds_key].append(run_id)
            if ds.name not in self.index["runs"][run_id]["inputs"]:
                self.index["runs"][run_id]["inputs"].append(ds.name)
        
        for ds in event.outputs:
            ds_key = f"{ds.namespace}:{ds.name}"
            if ds_key not in self.index["datasets"]:
                self.index["datasets"][ds_key] = []
            if run_id not in self.index["datasets"][ds_key]:
                self.index["datasets"][ds_key].append(run_id)
            if ds.name not in self.index["runs"][run_id]["outputs"]:
                self.index["runs"][run_id]["outputs"].append(ds.name)
        
        # Track jobs
        if job_name not in self.index["jobs"]:
            self.index["jobs"][job_name] = []
        if run_id not in self.index["jobs"][job_name]:
            self.index["jobs"][job_name].append(run_id)
        
        self._save_index()
        
        return event_id
    
    def _parse_datasets(self, datasets: List[Union[Dict, Dataset, str]]) -> List[Dataset]:
        """Convert various dataset specifications to Dataset objects."""
        result = []
        for ds in datasets:
            if isinstance(ds, Dataset):
                result.append(ds)
            elif isinstance(ds, dict):
                result.append(Dataset(
                    namespace=ds.get("namespace", self.dataset_namespace),
                    name=ds["name"],
                    facets=DatasetFacets(**ds.get("facets", {})) if "facets" in ds else None
                ))
            elif isinstance(ds, str):
                # Assume it's a path or name
                result.append(Dataset(
                    namespace=self.dataset_namespace,
                    name=Path(ds).name
                ))
            else:
                raise ValueError(f"Unknown dataset type: {type(ds)}")
        return result
    
    def start_run(
        self,
        job_name: str,
        inputs: List[Union[Dict, Dataset, str]] = None,
        run_id: str = None,
        parent_run_id: str = None,
        job_facets: Dict = None,
    ) -> str:
        """
        Record the start of a processing run.
        
        Parameters
        ----------
        job_name : str
            Name of the job being executed.
        inputs : list
            Input datasets (as Dataset objects, dicts, or paths).
        run_id : str, optional
            Explicit run ID. Generated if not provided.
        parent_run_id : str, optional
            Parent run ID for hierarchical runs.
        job_facets : dict, optional
            Additional job metadata.
        
        Returns
        -------
        str
            The run ID for this execution.
        """
        run_id = run_id or str(uuid.uuid4())
        inputs = self._parse_datasets(inputs or [])
        
        run_facets = RunFacets(
            nominal_time=self._now_iso(),
            parent_run={"runId": parent_run_id} if parent_run_id else None
        )
        
        event = LineageEvent(
            event_type=EventType.START,
            event_time=self._now_iso(),
            run=Run(run_id=run_id, facets=run_facets),
            job=Job(
                namespace=self.job_namespace,
                name=job_name,
                facets=JobFacets(**job_facets) if job_facets else None
            ),
            inputs=inputs,
            outputs=[]
        )
        
        self._emit_event(event)
        
        # Save run state for later
        run_state = {
            "run_id": run_id,
            "job_name": job_name,
            "inputs": [ds.to_dict() for ds in inputs],
            "started_at": self._now_iso()
        }
        run_file = self.runs_dir / f"{run_id}.json"
        run_file.write_text(json.dumps(run_state, indent=2))
        
        return run_id
    
    def complete_run(
        self,
        run_id: str,
        job_name: str = None,
        inputs: List[Union[Dict, Dataset, str]] = None,
        outputs: List[Union[Dict, Dataset, str]] = None,
        metrics: Dict[str, Any] = None,
    ) -> str:
        """
        Record successful completion of a run.
        
        Parameters
        ----------
        run_id : str
            Run ID from start_run.
        job_name : str, optional
            Job name (loaded from saved state if not provided).
        inputs : list, optional
            Input datasets (loaded from saved state if not provided).
        outputs : list
            Output datasets produced by this run.
        metrics : dict, optional
            Metrics to attach to the run.
        
        Returns
        -------
        str
            The event ID.
        """
        # Load run state
        run_file = self.runs_dir / f"{run_id}.json"
        if run_file.exists():
            run_state = json.loads(run_file.read_text())
            job_name = job_name or run_state.get("job_name")
            if inputs is None:
                inputs = [Dataset(**ds) for ds in run_state.get("inputs", [])]
        
        inputs = self._parse_datasets(inputs or [])
        outputs = self._parse_datasets(outputs or [])
        
        run_facets = RunFacets(
            custom={"metrics": metrics} if metrics else {}
        )
        
        event = LineageEvent(
            event_type=EventType.COMPLETE,
            event_time=self._now_iso(),
            run=Run(run_id=run_id, facets=run_facets),
            job=Job(namespace=self.job_namespace, name=job_name),
            inputs=inputs,
            outputs=outputs
        )
        
        return self._emit_event(event)
    
    def fail_run(
        self,
        run_id: str,
        job_name: str = None,
        inputs: List[Union[Dict, Dataset, str]] = None,
        error_message: str = None,
    ) -> str:
        """
        Record failure of a run.
        
        Parameters
        ----------
        run_id : str
            Run ID from start_run.
        job_name : str, optional
            Job name.
        inputs : list, optional
            Input datasets.
        error_message : str, optional
            Error message describing the failure.
        
        Returns
        -------
        str
            The event ID.
        """
        # Load run state
        run_file = self.runs_dir / f"{run_id}.json"
        if run_file.exists():
            run_state = json.loads(run_file.read_text())
            job_name = job_name or run_state.get("job_name")
            if inputs is None:
                inputs = [Dataset(**ds) for ds in run_state.get("inputs", [])]
        
        inputs = self._parse_datasets(inputs or [])
        
        run_facets = RunFacets(
            error_message=error_message
        )
        
        event = LineageEvent(
            event_type=EventType.FAIL,
            event_time=self._now_iso(),
            run=Run(run_id=run_id, facets=run_facets),
            job=Job(namespace=self.job_namespace, name=job_name),
            inputs=inputs,
            outputs=[]
        )
        
        return self._emit_event(event)
    
    # Query methods
    
    def get_dataset_lineage(self, dataset_name: str) -> Dict:
        """
        Get lineage information for a dataset.
        
        Returns all runs that produced or consumed this dataset.
        """
        # Find matching datasets
        matches = []
        for ds_key, run_ids in self.index["datasets"].items():
            if dataset_name in ds_key:
                matches.append({
                    "dataset": ds_key,
                    "run_ids": run_ids
                })
        
        result = {
            "dataset_name": dataset_name,
            "matches": matches,
            "runs": {}
        }
        
        # Load run details
        seen_runs = set()
        for match in matches:
            for run_id in match["run_ids"]:
                if run_id in seen_runs:
                    continue
                seen_runs.add(run_id)
                
                if run_id in self.index["runs"]:
                    result["runs"][run_id] = self.index["runs"][run_id]
        
        return result
    
    def get_run_details(self, run_id: str) -> Optional[Dict]:
        """Get details for a specific run."""
        if run_id not in self.index["runs"]:
            return None
        
        run_info = self.index["runs"][run_id].copy()
        
        # Load events
        run_info["event_details"] = []
        for event_id in run_info.get("events", []):
            event_file = self.events_dir / f"{event_id}.json"
            if event_file.exists():
                run_info["event_details"].append(
                    json.loads(event_file.read_text())
                )
        
        return run_info
    
    def get_job_history(self, job_name: str) -> List[Dict]:
        """Get history of all runs for a job."""
        run_ids = self.index["jobs"].get(job_name, [])
        return [self.get_run_details(rid) for rid in run_ids if rid in self.index["runs"]]
    
    def export_for_marquez(self, output_file: Union[str, Path] = None) -> List[Dict]:
        """
        Export all events in OpenLineage format for Marquez import.
        
        Parameters
        ----------
        output_file : str or Path, optional
            If provided, write events to this file.
        
        Returns
        -------
        list
            List of OpenLineage event dictionaries.
        """
        events = []
        for event_file in sorted(self.events_dir.glob("*.json")):
            events.append(json.loads(event_file.read_text()))
        
        if output_file:
            Path(output_file).write_text(json.dumps(events, indent=2))
        
        return events


# Context manager for easy run tracking
from contextlib import contextmanager

@contextmanager
def tracked_run(
    tracker: LineageTracker,
    job_name: str,
    inputs: List = None,
    **kwargs
):
    """
    Context manager for automatic lineage tracking.
    
    Examples
    --------
    >>> tracker = LineageTracker("/data/lineage")
    >>> with tracked_run(tracker, "process_slide", inputs=["input.zarr"]) as run:
    ...     # Do processing
    ...     run.set_outputs(["output.zarr"])
    ...     run.log_metric("n_tiles", 42)
    """
    
    class RunContext:
        def __init__(self, run_id: str):
            self.run_id = run_id
            self.outputs = []
            self.metrics = {}
        
        def set_outputs(self, outputs: List):
            self.outputs = outputs
        
        def add_output(self, output):
            self.outputs.append(output)
        
        def log_metric(self, key: str, value: Any):
            self.metrics[key] = value
    
    run_id = tracker.start_run(job_name, inputs=inputs, **kwargs)
    ctx = RunContext(run_id)
    
    try:
        yield ctx
        tracker.complete_run(
            run_id=run_id,
            job_name=job_name,
            outputs=ctx.outputs,
            metrics=ctx.metrics
        )
    except Exception as e:
        tracker.fail_run(
            run_id=run_id,
            job_name=job_name,
            error_message=str(e)
        )
        raise


# Convenience function for SciServer setup
def get_default_tracker() -> LineageTracker:
    """Get a lineage tracker configured for SciServer."""
    try:
        from SciServer import Config, Authentication
        if Config.isSciServerComputeEnvironment():
            user = Authentication.getKeystoneUserWithToken()
            storage_path = f"/home/idies/workspace/Storage/{user.userName}/UserVolume/lineage"
        else:
            storage_path = "./lineage"
    except ImportError:
        storage_path = "./lineage"
    
    return LineageTracker(storage_path)


__all__ = [
    "LineageTracker",
    "LineageEvent",
    "Dataset",
    "Job",
    "Run",
    "EventType",
    "DatasetFacets",
    "JobFacets",
    "RunFacets",
    "tracked_run",
    "get_default_tracker",
]
