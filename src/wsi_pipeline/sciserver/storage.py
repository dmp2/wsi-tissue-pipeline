"""
SciServer Storage Configuration

Manages storage paths for SciServer Compute environment
and local development.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .environment import get_sciserver_user, is_sciserver_environment


@dataclass
class StorageConfig:
    """
    SciServer storage configuration.

    Manages paths for:
    - Persistent storage
    - Temporary storage
    - MLFlow tracking
    - Lineage events
    - Data (raw/processed)
    - Outputs
    """
    persistent_base: Path
    temporary_base: Path
    scratch_base: Path
    user_volume: Path
    mlflow_dir: Path
    lineage_dir: Path
    data_dir: Path
    outputs_dir: Path

    @classmethod
    def for_sciserver(
        cls,
        username: str | None = None,
        volume_name: str = "UserVolume"
    ) -> StorageConfig:
        """
        Create storage config for SciServer environment.

        Parameters
        ----------
        username : str, optional
            SciServer username. Auto-detected if not provided.
        volume_name : str
            Name of the user volume.
        """
        if username is None:
            username = get_sciserver_user()
        if username is None:
            raise ValueError("Could not determine SciServer username")

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
        """
        Create storage config for local development.

        Parameters
        ----------
        base_dir : str
            Base directory for all storage.
        """
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

    @classmethod
    def auto(cls, local_fallback: str = ".") -> StorageConfig:
        """
        Auto-detect environment and create appropriate config.

        Parameters
        ----------
        local_fallback : str
            Base directory to use if not in SciServer.
        """
        if is_sciserver_environment():
            return cls.for_sciserver()
        return cls.for_local(local_fallback)

    def ensure_directories(self):
        """Create all required directories."""
        for path in [
            self.mlflow_dir,
            self.mlflow_dir / "artifacts",
            self.lineage_dir,
            self.lineage_dir / "events",
            self.data_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.outputs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def get_data_path(self, *parts: str, data_type: str = "processed") -> Path:
        """Get path within data directory."""
        return self.data_dir / data_type / Path(*parts)

    def get_output_path(self, *parts: str) -> Path:
        """Get path within outputs directory."""
        return self.outputs_dir / Path(*parts)


def get_storage_config(local_fallback: str = ".") -> StorageConfig:
    """
    Get storage configuration for current environment.

    Convenience function that auto-detects environment.
    """
    return StorageConfig.auto(local_fallback)


__all__ = [
    "StorageConfig",
    "get_storage_config",
]
