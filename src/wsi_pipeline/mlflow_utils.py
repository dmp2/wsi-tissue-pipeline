"""
MLflow Integration for WSI Tissue Pipeline

Provides experiment tracking, artifact logging, and reproducibility features.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

from .config import MLflowConfig, PipelineConfig

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


def check_mlflow_available():
    """Check if MLflow is available."""
    if not MLFLOW_AVAILABLE:
        raise ImportError(
            "MLflow is not installed. Install with: pip install mlflow"
        )


def init_mlflow(
    config: MLflowConfig | PipelineConfig | None = None,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> bool:
    """
    Initialize MLflow tracking.

    Parameters
    ----------
    config : MLflowConfig or PipelineConfig, optional
        Configuration object.
    tracking_uri : str, optional
        MLflow tracking server URI. Overrides config.
    experiment_name : str, optional
        Experiment name. Overrides config.

    Returns
    -------
    bool
        True if MLflow was successfully initialized.

    Examples
    --------
    >>> init_mlflow(tracking_uri="sqlite:///mlflow.db", experiment_name="my-experiment")
    >>> # Or use config
    >>> config = PipelineConfig()
    >>> init_mlflow(config)
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available. Tracking disabled.")
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

    # Set tracking URI
    mlflow.set_tracking_uri(uri)

    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment is None:
            mlflow.create_experiment(exp_name)
        mlflow.set_experiment(exp_name)
        return True
    except Exception as e:
        logger.warning("Failed to initialize MLflow: %s", e)
        return False


def get_run_name(
    template: str = "{specimen}_{timestamp}",
    specimen: str | None = None,
    **kwargs,
) -> str:
    """
    Generate a run name from template.

    Parameters
    ----------
    template : str
        Template string with placeholders.
    specimen : str, optional
        Specimen name.
    **kwargs
        Additional template variables.

    Returns
    -------
    str
        Formatted run name.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    variables = {
        "timestamp": timestamp,
        "specimen": specimen or "unknown",
        "date": datetime.now().strftime("%Y-%m-%d"),
        **kwargs,
    }
    return template.format(**variables)


@contextmanager
def MLflowContext(
    run_name: str | None = None,
    config: PipelineConfig | None = None,
    tags: dict[str, str] | None = None,
    log_config: bool = True,
):
    """
    Context manager for MLflow tracking.

    Parameters
    ----------
    run_name : str, optional
        Name for the run.
    config : PipelineConfig, optional
        Pipeline configuration to log.
    tags : dict, optional
        Additional tags for the run.
    log_config : bool
        Whether to log configuration as parameters.

    Yields
    ------
    mlflow.ActiveRun or None
        The active MLflow run, or None if MLflow is not available.

    Examples
    --------
    >>> with MLflowContext(run_name="experiment_1", config=config) as run:
    ...     # Do processing
    ...     mlflow.log_metric("n_tiles", 42)
    ...     mlflow.log_artifact("output/qc_grid.png")
    """
    if not MLFLOW_AVAILABLE:
        yield None
        return

    # Initialize MLflow if needed
    if config and config.mlflow.enabled:
        init_mlflow(config)

    # Default run name
    if run_name is None:
        run_name = get_run_name()

    # Merge tags
    all_tags = {
        "pipeline_version": "0.1.0",
        "python_version": sys.version.split()[0],
    }
    if tags:
        all_tags.update(tags)

    with mlflow.start_run(run_name=run_name, tags=all_tags) as run:
        # Log configuration
        if log_config and config:
            log_config_params(config)

        yield run


def log_config_params(config: PipelineConfig):
    """
    Log pipeline configuration as MLflow parameters.

    Parameters
    ----------
    config : PipelineConfig
        Configuration to log.
    """
    if not MLFLOW_AVAILABLE:
        return

    # Flatten config to parameters
    params = _flatten_dict(config.to_dict(), separator=".")

    # MLflow has limits on parameter values - truncate if needed
    for key, value in params.items():
        str_value = str(value)
        if len(str_value) > 250:
            str_value = str_value[:247] + "..."
        try:
            mlflow.log_param(key, str_value)
        except Exception:
            pass  # Skip parameters that fail


def _flatten_dict(d: dict, parent_key: str = "", separator: str = ".") -> dict:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, separator).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_processing_run(
    results: dict[str, Any],
    config: PipelineConfig | None = None,
    qc_images: list[str | Path] | None = None,
    output_dir: str | Path | None = None,
):
    """
    Log processing results to MLflow.

    Parameters
    ----------
    results : dict
        Processing results from process_specimen or similar.
    config : PipelineConfig, optional
        Configuration used for processing.
    qc_images : list, optional
        Paths to QC images to log as artifacts.
    output_dir : str or Path, optional
        Output directory to log as artifact directory.
    """
    if not MLFLOW_AVAILABLE:
        return

    # Log metrics
    if "n_inputs" in results:
        mlflow.log_metric("n_inputs", results["n_inputs"])
    if "n_outputs" in results:
        mlflow.log_metric("n_outputs", results["n_outputs"])
    if "n_tiles" in results:
        mlflow.log_metric("n_tiles", results["n_tiles"])

    # Log parameters
    if "specimen_name" in results:
        mlflow.log_param("specimen_name", results["specimen_name"])
    if "input_dir" in results:
        mlflow.log_param("input_dir", results["input_dir"])

    # Log results as JSON artifact
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(results, f, indent=2, default=str)
        f.flush()
        mlflow.log_artifact(f.name, "results")

    # Log QC images
    if qc_images:
        for img_path in qc_images:
            if Path(img_path).exists():
                mlflow.log_artifact(str(img_path), "qc_images")

    # Log output directory
    if output_dir and Path(output_dir).exists():
        try:
            # Log metadata files only (not large image files)
            for p in Path(output_dir).glob("*_metadata.json"):
                mlflow.log_artifact(str(p), "metadata")
        except Exception:
            pass


def track_processing(
    run_name: str | None = None,
    log_artifacts: bool = True,
):
    """
    Decorator for tracking function execution with MLflow.

    Parameters
    ----------
    run_name : str, optional
        Run name template. Use {func_name} for function name.
    log_artifacts : bool
        Whether to log returned artifacts.

    Returns
    -------
    decorator
        Function decorator.

    Examples
    --------
    >>> @track_processing(run_name="process_{func_name}")
    ... def my_processing_function(input_dir, output_dir):
    ...     # Processing code
    ...     return {"n_tiles": 42, "output_paths": [...]}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not MLFLOW_AVAILABLE:
                return func(*args, **kwargs)

            # Generate run name
            name = run_name or "{func_name}_{timestamp}"
            name = name.format(
                func_name=func.__name__,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            )

            with mlflow.start_run(run_name=name):
                # Log function arguments
                for i, arg in enumerate(args):
                    if isinstance(arg, (str, int, float, bool)):
                        mlflow.log_param(f"arg_{i}", arg)
                for key, value in kwargs.items():
                    if isinstance(value, (str, int, float, bool)):
                        mlflow.log_param(key, value)

                # Execute function
                result = func(*args, **kwargs)

                # Log results if dict
                if isinstance(result, dict) and log_artifacts:
                    log_processing_run(result)

                return result

        return wrapper
    return decorator


class MLflowLogger:
    """
    Helper class for MLflow logging during processing.

    Examples
    --------
    >>> logger = MLflowLogger()
    >>> logger.start_run("my_experiment")
    >>> logger.log_param("batch_size", 32)
    >>> logger.log_metric("accuracy", 0.95, step=100)
    >>> logger.end_run()
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._run = None
        self._initialized = False

    def initialize(self):
        """Initialize MLflow connection."""
        if self._initialized:
            return
        if not MLFLOW_AVAILABLE:
            return
        init_mlflow(
            tracking_uri=self.tracking_uri,
            experiment_name=self.experiment_name,
        )
        self._initialized = True

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        """Start a new MLflow run."""
        if not MLFLOW_AVAILABLE:
            return
        self.initialize()
        self._run = mlflow.start_run(run_name=run_name, tags=tags)

    def end_run(self):
        """End the current run."""
        if not MLFLOW_AVAILABLE or self._run is None:
            return
        mlflow.end_run()
        self._run = None

    def log_param(self, key: str, value: Any):
        """Log a parameter."""
        if not MLFLOW_AVAILABLE:
            return
        mlflow.log_param(key, value)

    def log_params(self, params: dict[str, Any]):
        """Log multiple parameters."""
        if not MLFLOW_AVAILABLE:
            return
        mlflow.log_params(params)

    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None = None,
    ):
        """Log a metric."""
        if not MLFLOW_AVAILABLE:
            return
        mlflow.log_metric(key, value, step=step)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ):
        """Log multiple metrics."""
        if not MLFLOW_AVAILABLE:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None):
        """Log an artifact file."""
        if not MLFLOW_AVAILABLE:
            return
        mlflow.log_artifact(str(path), artifact_path)

    def log_artifacts(self, dir_path: str | Path, artifact_path: str | None = None):
        """Log all files in a directory."""
        if not MLFLOW_AVAILABLE:
            return
        mlflow.log_artifacts(str(dir_path), artifact_path)

    def set_tag(self, key: str, value: str):
        """Set a tag on the current run."""
        if not MLFLOW_AVAILABLE:
            return
        mlflow.set_tag(key, value)

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False
