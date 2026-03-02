"""
SciServer MLFlow Integration

Configures MLFlow for use within SciServer Compute environment.
Handles automatic path configuration and authentication.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..config import PipelineConfig

# Lazy imports to avoid hard dependency
_mlflow = None
_sciserver_available = False

def _ensure_mlflow():
    global _mlflow
    if _mlflow is None:
        try:
            import mlflow
            _mlflow = mlflow
        except ImportError as err:
            raise ImportError("MLFlow not installed. Install with: pip install mlflow") from err
    return _mlflow


def _check_sciserver():
    """Check if running in SciServer environment."""
    global _sciserver_available
    try:
        from SciServer import Config
        _sciserver_available = Config.isSciServerComputeEnvironment()
    except ImportError:
        _sciserver_available = False
    return _sciserver_available


def get_sciserver_username() -> str | None:
    """Get current SciServer username."""
    if not _check_sciserver():
        return None
    try:
        from SciServer import Authentication
        user = Authentication.getKeystoneUserWithToken()
        return user.userName if user else None
    except Exception:
        return None


def get_sciserver_storage_path() -> Path | None:
    """Get user's persistent storage path in SciServer."""
    username = get_sciserver_username()
    if username:
        base = Path("/home/idies/workspace/Storage")
        # Try common volume names
        for vol_name in ["UserVolume", username, "persistent"]:
            vol_path = base / username / vol_name
            if vol_path.exists():
                return vol_path
        # Fallback to first available
        user_dir = base / username
        if user_dir.exists():
            subdirs = [d for d in user_dir.iterdir() if d.is_dir()]
            if subdirs:
                return subdirs[0]
    return None


class SciServerMLFlowConfig:
    """
    MLFlow configuration manager for SciServer.

    Automatically configures MLFlow based on execution environment:
    - SciServer Compute: Uses persistent storage paths
    - Local development: Uses local paths

    Examples
    --------
    >>> config = SciServerMLFlowConfig()
    >>> config.setup()
    >>>
    >>> # Now use MLFlow normally
    >>> import mlflow
    >>> with mlflow.start_run():
    ...     mlflow.log_param("param", "value")
    """

    def __init__(
        self,
        experiment_name: str = "wsi-tissue-pipeline",
        mlflow_subdir: str = "mlflow",
        local_fallback_dir: str = "./mlruns",
    ):
        self.experiment_name = experiment_name
        self.mlflow_subdir = mlflow_subdir
        self.local_fallback_dir = local_fallback_dir

        self._tracking_uri: str | None = None
        self._artifact_location: str | None = None
        self._is_sciserver: bool = False
        self._configured: bool = False

    def _determine_paths(self):
        """Determine appropriate paths based on environment."""
        if _check_sciserver():
            self._is_sciserver = True
            storage_path = get_sciserver_storage_path()

            if storage_path:
                mlflow_dir = storage_path / self.mlflow_subdir
                mlflow_dir.mkdir(parents=True, exist_ok=True)

                self._tracking_uri = f"sqlite:///{mlflow_dir}/mlflow.db"
                self._artifact_location = str(mlflow_dir / "artifacts")

                # Ensure artifacts dir exists
                Path(self._artifact_location).mkdir(parents=True, exist_ok=True)
            else:
                # Fallback to temporary storage
                temp_dir = Path("/home/idies/workspace/Temporary")
                username = get_sciserver_username() or "unknown"
                mlflow_dir = temp_dir / username / self.mlflow_subdir
                mlflow_dir.mkdir(parents=True, exist_ok=True)

                self._tracking_uri = f"sqlite:///{mlflow_dir}/mlflow.db"
                self._artifact_location = str(mlflow_dir / "artifacts")
                Path(self._artifact_location).mkdir(parents=True, exist_ok=True)
        else:
            self._is_sciserver = False
            local_dir = Path(self.local_fallback_dir)
            local_dir.mkdir(parents=True, exist_ok=True)

            self._tracking_uri = f"sqlite:///{local_dir}/mlflow.db"
            self._artifact_location = str(local_dir / "artifacts")
            Path(self._artifact_location).mkdir(parents=True, exist_ok=True)

    def setup(self) -> dict[str, Any]:
        """
        Configure MLFlow for the current environment.

        Returns
        -------
        dict
            Configuration details including tracking_uri, artifact_location.
        """
        mlflow = _ensure_mlflow()

        if not self._configured:
            self._determine_paths()

        # Set MLFlow configuration
        mlflow.set_tracking_uri(self._tracking_uri)

        # Create or set experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                self.experiment_name,
                artifact_location=self._artifact_location
            )
        mlflow.set_experiment(self.experiment_name)

        self._configured = True

        return self.get_config()

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        if not self._tracking_uri:
            self._determine_paths()

        return {
            "tracking_uri": self._tracking_uri,
            "artifact_location": self._artifact_location,
            "experiment_name": self.experiment_name,
            "is_sciserver": self._is_sciserver,
            "username": get_sciserver_username() if self._is_sciserver else None,
        }

    @property
    def tracking_uri(self) -> str:
        if not self._tracking_uri:
            self._determine_paths()
        return self._tracking_uri

    @property
    def artifact_location(self) -> str:
        if not self._artifact_location:
            self._determine_paths()
        return self._artifact_location


# Singleton instance for easy access
_default_config: SciServerMLFlowConfig | None = None


def setup_mlflow(
    experiment_name: str = "wsi-tissue-pipeline",
    **kwargs
) -> dict[str, Any]:
    """
    Initialize MLFlow for SciServer or local environment.

    This is the main entry point for setting up MLFlow tracking.
    Call this once at the start of your pipeline.

    Parameters
    ----------
    experiment_name : str
        Name for the MLFlow experiment.
    **kwargs
        Additional arguments passed to SciServerMLFlowConfig.

    Returns
    -------
    dict
        Configuration details.

    Examples
    --------
    >>> config = setup_mlflow("my-experiment")
    >>> print(f"Tracking URI: {config['tracking_uri']}")
    """
    global _default_config
    _default_config = SciServerMLFlowConfig(experiment_name=experiment_name, **kwargs)
    return _default_config.setup()


def get_mlflow_config() -> dict[str, Any]:
    """Get current MLFlow configuration."""
    global _default_config
    if _default_config is None:
        _default_config = SciServerMLFlowConfig()
    return _default_config.get_config()


@contextmanager
def mlflow_run(
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    log_system_info: bool = True
):
    """
    Context manager for MLFlow runs with SciServer metadata.

    Parameters
    ----------
    run_name : str, optional
        Name for the run.
    tags : dict, optional
        Additional tags to add.
    log_system_info : bool
        Whether to log SciServer system information.

    Yields
    ------
    mlflow.ActiveRun
        The active MLFlow run.

    Examples
    --------
    >>> with mlflow_run("process_specimen_001") as run:
    ...     mlflow.log_param("input_path", "/data/specimen_001")
    ...     # ... processing ...
    ...     mlflow.log_metric("n_tiles", 42)
    """
    mlflow = _ensure_mlflow()

    # Ensure MLFlow is configured
    if _default_config is None:
        setup_mlflow()

    # Build tags
    all_tags = tags.copy() if tags else {}

    if log_system_info:
        all_tags["environment"] = "sciserver" if _check_sciserver() else "local"
        username = get_sciserver_username()
        if username:
            all_tags["sciserver_user"] = username

    with mlflow.start_run(run_name=run_name, tags=all_tags) as run:
        if log_system_info:
            # Log environment info
            mlflow.log_param("env.is_sciserver", _check_sciserver())
            mlflow.log_param("env.username", get_sciserver_username() or "local")

            # Log Python environment
            import sys
            mlflow.log_param("env.python_version", sys.version.split()[0])

        yield run


def log_zarr_artifact(
    zarr_path: str,
    artifact_name: str | None = None,
    log_metadata_only: bool = True
):
    """
    Log a Zarr dataset as an MLFlow artifact.

    For large Zarr datasets, logs only metadata by default
    to avoid duplicating large data.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset.
    artifact_name : str, optional
        Name for the artifact. Defaults to directory name.
    log_metadata_only : bool
        If True, only log .zattrs and .zarray files.
    """
    mlflow = _ensure_mlflow()
    zarr_path = Path(zarr_path)

    if not zarr_path.exists():
        raise ValueError(f"Zarr path does not exist: {zarr_path}")

    artifact_name = artifact_name or zarr_path.name

    if log_metadata_only:
        # Log only metadata files
        metadata_files = list(zarr_path.rglob(".zattrs")) + list(zarr_path.rglob(".zarray"))

        # Create temp directory with metadata structure
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_zarr = Path(tmpdir) / artifact_name

            for meta_file in metadata_files:
                rel_path = meta_file.relative_to(zarr_path)
                dst = tmp_zarr / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(meta_file, dst)

            # Also log the path reference
            ref_file = tmp_zarr / "_source_path.txt"
            ref_file.write_text(str(zarr_path.absolute()))

            mlflow.log_artifacts(str(tmp_zarr), artifact_name)
    else:
        # Log entire Zarr directory (use cautiously for large datasets)
        mlflow.log_artifacts(str(zarr_path), artifact_name)


def log_pipeline_config(config: PipelineConfig):
    """
    Log pipeline configuration to MLFlow.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration object.
    """
    mlflow = _ensure_mlflow()

    # Log as parameters (flattened)
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)

    def flatten_dict(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_config = flatten_dict(config_dict)

    for key, value in flat_config.items():
        # MLFlow has limits on parameter value length
        str_value = str(value)
        if len(str_value) > 250:
            str_value = str_value[:247] + "..."
        try:
            mlflow.log_param(key, str_value)
        except Exception:
            pass  # Skip parameters that fail

    # Also log full config as artifact
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_dict, f, indent=2, default=str)
        f.flush()
        mlflow.log_artifact(f.name, "config")


# Convenience exports
__all__ = [
    "SciServerMLFlowConfig",
    "setup_mlflow",
    "get_mlflow_config",
    "mlflow_run",
    "log_zarr_artifact",
    "log_pipeline_config",
    "get_sciserver_username",
    "get_sciserver_storage_path",
]
