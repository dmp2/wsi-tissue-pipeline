"""
Configuration management for WSI Tissue Pipeline.

Uses Pydantic for type-safe configuration with validation.
Supports YAML config files and environment variable overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SegmentationConfig(BaseModel):
    """Configuration for tissue segmentation."""

    backend: Literal[
        "local-entropy", "local-otsu", "tiatoolbox-otsu", "tiatoolbox-morph", "pathml-he"
    ] = Field(default="local-entropy", description="Segmentation algorithm backend")

    target_long_side: int = Field(
        default=1800, ge=100, le=10000,
        description=(
            "Thumbnail long-axis size (px) used for segmentation. "
            "Larger = more accurate but slower. Default 1800 works well for most slides; "
            "reduce to 900 for speed."
        ),
    )

    min_area_px: int = Field(
        default=3000, ge=100,
        description=(
            "Minimum tissue region area (pixels, at thumbnail scale). "
            "Regions smaller than this are discarded as artifacts. "
            "Default 3000 is roughly a 55x55 px patch at thumbnail scale."
        ),
    )

    struct_elem_px: int = Field(
        default=4, ge=1, le=50,
        description=(
            "Radius (px) of the morphological structuring element used to smooth and close "
            "holes in the tissue mask. Increase for noisier slides."
        ),
    )

    split_touching: bool = Field(
        default=True, description="Whether to split touching tissue sections"
    )

    r_split: int = Field(
        default=2, ge=1, le=10,
        description=(
            "Radius (px, at thumbnail scale) used by watershed to split touching tissue sections. "
            "Increase if sections that should be separate are merged; "
            "decrease if single sections are split incorrectly."
        ),
    )

    diagnostics: bool = Field(
        default=False, description="Enable detailed diagnostic output"
    )


class TileConfig(BaseModel):
    """Configuration for tile extraction."""

    chunk_size: int = Field(
        default=512, ge=64, le=4096,
        description=(
            "Tile chunk size (px) for OME-Zarr output. "
            "512 is optimal for most use cases; "
            "256 improves random-access performance at the cost of more files."
        ),
    )

    pad_multiple: int = Field(default=512, ge=64, description="Padding multiple for tiles")

    extra_margin_px: int = Field(
        default=0, ge=0,
        description=(
            "Extra padding (px, at full resolution) added around each tissue bounding box "
            "before writing. Useful to avoid clipping edge tissue."
        ),
    )


class OutputConfig(BaseModel):
    """Configuration for output format and storage."""

    format: Literal["ome-zarr", "tiff", "both"] = Field(
        default="ome-zarr", description="Output format for tissue sections"
    )

    compression: str | None = Field(
        default="zstd", description="Compression algorithm (zstd, lz4, gzip, or None)"
    )

    compression_level: int = Field(
        default=5, ge=1, le=9, description="Compression level"
    )

    convert_to_uint8: bool = Field(
        default=True, description="Convert output to uint8"
    )

    generate_qc: bool = Field(
        default=True, description="Generate QC contact sheets"
    )


class MLflowConfig(BaseModel):
    """Configuration for MLflow experiment tracking."""

    enabled: bool = Field(default=False, description="Enable MLflow tracking (requires: pip install wsi-tissue-pipeline[mlflow])")

    tracking_uri: str = Field(
        default="sqlite:///mlflow.db", description="MLflow tracking server URI"
    )

    experiment_name: str = Field(
        default="wsi-tissue-pipeline", description="MLflow experiment name"
    )

    run_name_template: str = Field(
        default="{specimen}_{timestamp}", description="Template for run names"
    )

    log_artifacts: bool = Field(
        default=True, description="Log output artifacts to MLflow"
    )

    log_qc_images: bool = Field(
        default=True, description="Log QC images as artifacts"
    )


class ColabConfig(BaseModel):
    """Configuration specific to Google Colab environment."""

    mount_drive: bool = Field(default=True, description="Mount Google Drive")

    drive_mount_path: str = Field(
        default="/content/drive", description="Google Drive mount path"
    )

    use_gpu: bool = Field(default=True, description="Use GPU if available")

    install_dependencies: bool = Field(
        default=True, description="Install missing dependencies"
    )


class PipelineConfig(BaseModel):
    """Main configuration for the WSI Tissue Pipeline."""

    model_config = ConfigDict(extra="ignore")

    # Sub-configurations
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    tiles: TileConfig = Field(default_factory=TileConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    colab: ColabConfig = Field(default_factory=ColabConfig)

    # Global settings
    specimen_spacing: int = Field(
        default=1,
        ge=1,
        description=(
            "Section collection interval -- e.g., 10 means every 10th section was collected. "
            "Used to compute the physical Z-spacing between tiles for 3D reconstruction "
            "(dv = specimen_spacing x section_thickness_um)."
        ),
    )

    num_workers: int = Field(
        default=-1, ge=-1, description="Number of parallel workers (-1 for auto)"
    )

    verbose: bool = Field(default=True, description="Enable verbose output")

    random_seed: int | None = Field(
        default=42, description="Random seed for reproducibility"
    )

    @model_validator(mode="after")
    def validate_config(self) -> PipelineConfig:
        """Validate cross-field dependencies."""
        # Add any cross-field validation here
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class EnvironmentSettings(BaseSettings):
    """
    Environment-based settings.

    These can be set via environment variables with WSI_ prefix.
    Example: WSI_MLFLOW_TRACKING_URI=http://mlflow:5000
    """

    model_config = SettingsConfigDict(
        env_prefix="WSI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Field(default=Path("/data"))
    output_dir: Path = Field(default=Path("/output"))
    config_path: Path | None = Field(default=None)

    # MLflow
    mlflow_tracking_uri: str = Field(default="sqlite:///mlflow.db")
    mlflow_experiment_name: str = Field(default="wsi-tissue-pipeline")

    # Processing
    num_workers: int = Field(default=-1)
    use_gpu: bool = Field(default=True)

    # Debug
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    """
    Load pipeline configuration from file with optional overrides.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to YAML configuration file. If None, uses default config.
    overrides : dict, optional
        Dictionary of values to override in the configuration.

    Returns
    -------
    PipelineConfig
        Loaded and validated configuration.

    Examples
    --------
    >>> config = load_config("configs/default.yaml")
    >>> config = load_config(overrides={"segmentation": {"backend": "local-otsu"}})
    """
    if config_path is not None:
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig()

    if overrides:
        # Merge overrides
        config_dict = config.to_dict()
        _deep_update(config_dict, overrides)
        config = PipelineConfig(**config_dict)

    return config


def _deep_update(base: dict, update: dict) -> dict:
    """Recursively update nested dictionary."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def get_default_config_path() -> Path:
    """Get path to default configuration file."""
    # Check common locations
    locations = [
        Path("configs/default.yaml"),
        Path("config.yaml"),
        Path.home() / ".config" / "wsi-pipeline" / "config.yaml",
    ]

    for loc in locations:
        if loc.exists():
            return loc

    # Return first location (will be created if needed)
    return locations[0]


def create_default_config(output_path: Path | None = None) -> Path:
    """
    Create a default configuration file.

    Parameters
    ----------
    output_path : Path, optional
        Path to write the config file. If None, uses default location.

    Returns
    -------
    Path
        Path to the created configuration file.
    """
    if output_path is None:
        output_path = get_default_config_path()

    config = PipelineConfig()
    config.save_yaml(output_path)
    return output_path
