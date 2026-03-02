"""
SciServer Integration Module for WSI Pipeline

Provides unified access to SciServer-specific utilities including:
- MLFlow experiment tracking configuration
- Data lineage tracking
- Storage path management
- Authentication handling
- Job submission utilities

Usage
-----
>>> from sciserver_integration import SciServerPipeline
>>>
>>> # Initialize with automatic environment detection
>>> pipeline = SciServerPipeline()
>>>
>>> # Run with tracking
>>> with pipeline.tracked_experiment("process_specimen_001") as exp:
...     exp.log_param("input_path", "/data/raw/specimen_001")
...     # ... processing ...
...     exp.log_metric("n_tiles", 42)
...     exp.log_output("specimen_001.ome.zarr")
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Environment detection
def is_sciserver_environment() -> bool:
    """Check if running in SciServer Compute environment."""
    try:
        from SciServer import Config
        return Config.isSciServerComputeEnvironment()
    except ImportError:
        return False


def get_sciserver_user() -> str | None:
    """Get current SciServer username."""
    if not is_sciserver_environment():
        return None
    try:
        from SciServer import Authentication
        user = Authentication.getKeystoneUserWithToken()
        return user.userName if user else None
    except Exception:
        return None


@dataclass
class StorageConfig:
    """SciServer storage configuration."""
    persistent_base: Path
    temporary_base: Path
    scratch_base: Path
    user_volume: Path
    mlflow_dir: Path
    lineage_dir: Path
    data_dir: Path
    outputs_dir: Path

    @classmethod
    def for_sciserver(cls, username: str, volume_name: str = "UserVolume") -> StorageConfig:
        """Create storage config for SciServer environment."""
        base = Path("/home/idies/workspace")
        user_volume = base / "Storage" / username / volume_name

        return cls(
            persistent_base=base / "Storage" / username,
            temporary_base=base / "Temporary" / username,
            scratch_base=base / "Scratch",
            user_volume=user_volume,
            mlflow_dir=user_volume / "mlflow",
            lineage_dir=user_volume / "lineage",
            data_dir=user_volume / "data",
            outputs_dir=user_volume / "outputs",
        )

    @classmethod
    def for_local(cls, base_dir: str = ".") -> StorageConfig:
        """Create storage config for local development."""
        base = Path(base_dir).absolute()

        return cls(
            persistent_base=base,
            temporary_base=base / "temp",
            scratch_base=base / "scratch",
            user_volume=base,
            mlflow_dir=base / "mlflow",
            lineage_dir=base / "lineage",
            data_dir=base / "data",
            outputs_dir=base / "outputs",
        )

    def ensure_directories(self):
        """Create all required directories."""
        for path in [
            self.mlflow_dir,
            self.lineage_dir,
            self.data_dir,
            self.outputs_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
        ]:
            path.mkdir(parents=True, exist_ok=True)


class SciServerPipeline:
    """
    Unified interface for WSI pipeline execution on SciServer.

    Handles:
    - Environment detection (SciServer vs local)
    - Storage path configuration
    - MLFlow experiment tracking
    - Data lineage tracking
    - Automatic logging and metrics

    Parameters
    ----------
    experiment_name : str
        Name for MLFlow experiment.
    project_name : str
        Project name for lineage tracking.
    auto_setup : bool
        Automatically configure MLFlow and create directories.

    Examples
    --------
    >>> pipeline = SciServerPipeline(experiment_name="wsi-analysis")
    >>>
    >>> # Check environment
    >>> print(f"SciServer: {pipeline.is_sciserver}")
    >>> print(f"Storage: {pipeline.storage.data_dir}")
    >>>
    >>> # Run with full tracking
    >>> with pipeline.tracked_experiment("analyze_specimen_001") as exp:
    ...     exp.log_input("specimen_001.vsi")
    ...     result = process_specimen(...)
    ...     exp.log_output("specimen_001.ome.zarr")
    ...     exp.log_metrics(result.metrics)
    """

    def __init__(
        self,
        experiment_name: str = "wsi-tissue-pipeline",
        project_name: str = "wsi-pipeline",
        auto_setup: bool = True,
    ):
        self.experiment_name = experiment_name
        self.project_name = project_name

        # Detect environment
        self.is_sciserver = is_sciserver_environment()
        self.username = get_sciserver_user()

        # Configure storage
        if self.is_sciserver and self.username:
            self.storage = StorageConfig.for_sciserver(self.username)
        else:
            self.storage = StorageConfig.for_local()

        # Lazy-loaded components
        self._mlflow_config = None
        self._lineage_tracker = None

        if auto_setup:
            self.setup()

    def setup(self):
        """Initialize all components."""
        # Create directories
        self.storage.ensure_directories()

        # Setup MLFlow
        self._setup_mlflow()

        # Setup lineage tracker
        self._setup_lineage()

    def _setup_mlflow(self):
        """Configure MLFlow for the environment."""
        try:
            from sciserver_mlflow import SciServerMLFlowConfig
            self._mlflow_config = SciServerMLFlowConfig(
                experiment_name=self.experiment_name,
                mlflow_subdir=str(self.storage.mlflow_dir.relative_to(self.storage.user_volume))
                if self.is_sciserver else "mlflow"
            )
            self._mlflow_config.setup()
        except ImportError:
            # Fall back to direct MLFlow setup
            try:
                import mlflow
                tracking_uri = f"sqlite:///{self.storage.mlflow_dir}/mlflow.db"
                mlflow.set_tracking_uri(tracking_uri)

                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    mlflow.create_experiment(
                        self.experiment_name,
                        artifact_location=str(self.storage.mlflow_dir / "artifacts")
                    )
                mlflow.set_experiment(self.experiment_name)

                self._mlflow_config = {
                    "tracking_uri": tracking_uri,
                    "artifact_location": str(self.storage.mlflow_dir / "artifacts")
                }
            except ImportError:
                self._mlflow_config = None

    def _setup_lineage(self):
        """Configure lineage tracker."""
        try:
            from sciserver_lineage import LineageTracker
            self._lineage_tracker = LineageTracker(
                storage_path=str(self.storage.lineage_dir),
                job_namespace=f"sciserver://{self.project_name}",
                dataset_namespace=f"sciserver://storage/{self.username or 'local'}"
            )
        except ImportError:
            self._lineage_tracker = None

    @property
    def mlflow_tracking_uri(self) -> str | None:
        """Get MLFlow tracking URI."""
        if self._mlflow_config:
            if isinstance(self._mlflow_config, dict):
                return self._mlflow_config.get("tracking_uri")
            return self._mlflow_config.tracking_uri
        return None

    @property
    def lineage_tracker(self):
        """Get lineage tracker instance."""
        return self._lineage_tracker

    @contextmanager
    def tracked_experiment(
        self,
        run_name: str,
        inputs: list[str] = None,
        tags: dict[str, str] = None,
    ):
        """
        Context manager for experiment tracking.

        Combines MLFlow run tracking with lineage tracking.

        Parameters
        ----------
        run_name : str
            Name for the experiment run.
        inputs : list, optional
            Input dataset names/paths.
        tags : dict, optional
            Additional tags for the run.

        Yields
        ------
        ExperimentContext
            Context object for logging params, metrics, and outputs.
        """

        class ExperimentContext:
            def __init__(ctx, parent, mlflow_run, lineage_run_id):
                ctx.parent = parent
                ctx.mlflow_run = mlflow_run
                ctx.lineage_run_id = lineage_run_id
                ctx._inputs = list(inputs or [])
                ctx._outputs = []
                ctx._metrics = {}

            def log_param(ctx, key: str, value: Any):
                if ctx.mlflow_run:
                    import mlflow
                    mlflow.log_param(key, value)

            def log_params(ctx, params: dict[str, Any]):
                if ctx.mlflow_run:
                    import mlflow
                    mlflow.log_params(params)

            def log_metric(ctx, key: str, value: float, step: int = None):
                ctx._metrics[key] = value
                if ctx.mlflow_run:
                    import mlflow
                    mlflow.log_metric(key, value, step=step)

            def log_metrics(ctx, metrics: dict[str, float], step: int = None):
                ctx._metrics.update(metrics)
                if ctx.mlflow_run:
                    import mlflow
                    mlflow.log_metrics(metrics, step=step)

            def log_input(ctx, input_name: str):
                ctx._inputs.append(input_name)

            def log_output(ctx, output_name: str):
                ctx._outputs.append(output_name)

            def log_artifact(ctx, path: str, artifact_path: str = None):
                if ctx.mlflow_run:
                    import mlflow
                    mlflow.log_artifact(path, artifact_path)

            def set_tag(ctx, key: str, value: str):
                if ctx.mlflow_run:
                    import mlflow
                    mlflow.set_tag(key, value)

        mlflow_run = None
        lineage_run_id = None

        # Start MLFlow run
        try:
            import mlflow
            all_tags = {
                "environment": "sciserver" if self.is_sciserver else "local",
                "username": self.username or "local",
                **(tags or {})
            }
            mlflow_run = mlflow.start_run(run_name=run_name, tags=all_tags)
        except Exception:
            pass

        # Start lineage tracking
        if self._lineage_tracker:
            lineage_run_id = self._lineage_tracker.start_run(
                job_name=run_name,
                inputs=[{"name": i, "namespace": f"sciserver://storage/{self.username or 'local'}"}
                       for i in (inputs or [])]
            )

        ctx = ExperimentContext(self, mlflow_run, lineage_run_id)

        try:
            yield ctx

            # Complete lineage tracking
            if self._lineage_tracker and lineage_run_id:
                self._lineage_tracker.complete_run(
                    run_id=lineage_run_id,
                    job_name=run_name,
                    outputs=[{"name": o, "namespace": f"sciserver://storage/{self.username or 'local'}"}
                            for o in ctx._outputs],
                    metrics=ctx._metrics
                )

            # End MLFlow run
            if mlflow_run:
                import mlflow
                mlflow.end_run()

        except Exception as e:
            # Record failure
            if self._lineage_tracker and lineage_run_id:
                self._lineage_tracker.fail_run(
                    run_id=lineage_run_id,
                    job_name=run_name,
                    error_message=str(e)
                )

            if mlflow_run:
                import mlflow
                mlflow.end_run(status="FAILED")

            raise

    def get_data_path(self, *parts: str, data_type: str = "processed") -> Path:
        """
        Get a path within the data directory.

        Parameters
        ----------
        *parts : str
            Path components.
        data_type : str
            "raw", "processed", or custom subdirectory.

        Returns
        -------
        Path
            Full path within data directory.
        """
        return self.storage.data_dir / data_type / Path(*parts)

    def get_output_path(self, *parts: str) -> Path:
        """Get a path within the outputs directory."""
        return self.storage.outputs_dir / Path(*parts)

    def log_config(self, config: Any, name: str = "pipeline_config"):
        """Log a configuration object to the current MLFlow run."""
        try:
            import mlflow

            # Convert to dict if possible
            if hasattr(config, 'to_dict'):
                config_dict = config.to_dict()
            elif hasattr(config, '__dict__'):
                config_dict = vars(config)
            else:
                config_dict = {"value": str(config)}

            # Log as artifact
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_dict, f, indent=2, default=str)
                f.flush()
                mlflow.log_artifact(f.name, "config")

            # Also log key params
            def flatten(d, parent_key=''):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    if isinstance(v, dict) and len(str(v)) < 200:
                        items.extend(flatten(v, new_key).items())
                    else:
                        items.append((new_key, v))
                return dict(items)

            for key, value in flatten(config_dict).items():
                try:
                    str_val = str(value)[:250]
                    mlflow.log_param(key, str_val)
                except Exception:
                    pass

        except ImportError:
            pass

    def submit_batch_job(
        self,
        script: str,
        job_name: str = None,
        docker_image: str = None,
        volumes: list[str] = None,
    ) -> str | None:
        """
        Submit a batch job to SciServer.

        Parameters
        ----------
        script : str
            Shell command or script to execute.
        job_name : str, optional
            Name for the job.
        docker_image : str, optional
            Docker image to use.
        volumes : list, optional
            Volume names to mount.

        Returns
        -------
        str or None
            Job ID if submitted successfully, None if not on SciServer.
        """
        if not self.is_sciserver:
            logger.warning("Not in SciServer environment - cannot submit batch job")
            return None

        try:
            from SciServer import Jobs

            job_id = Jobs.submitShellCommandJob(
                shellCommand=script,
                dockerComputeDomain="Science Pipelines",
                dockerImageName=docker_image or "sciserver/jupyter:latest",
                userVolumes=volumes,
                jobAlias=job_name or f"wsi-pipeline-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )

            return str(job_id)

        except Exception as e:
            logger.error("Failed to submit job: %s", e)
            return None

    def get_job_status(self, job_id: str) -> dict | None:
        """Get status of a submitted job."""
        if not self.is_sciserver:
            return None

        try:
            from SciServer import Jobs
            return Jobs.getJobStatus(int(job_id))
        except Exception:
            return None


# Convenience function for quick setup
def setup_sciserver_tracking(
    experiment_name: str = "wsi-tissue-pipeline",
) -> SciServerPipeline:
    """
    Quick setup for SciServer experiment tracking.

    Returns a configured SciServerPipeline instance.

    Examples
    --------
    >>> pipeline = setup_sciserver_tracking("my-experiment")
    >>> with pipeline.tracked_experiment("run-001") as exp:
    ...     exp.log_param("param", "value")
    """
    return SciServerPipeline(experiment_name=experiment_name)


__all__ = [
    "SciServerPipeline",
    "StorageConfig",
    "setup_sciserver_tracking",
    "is_sciserver_environment",
    "get_sciserver_user",
]
