"""VSI/ETS entry points for source and tissue OME-Zarr generation."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..config import SegmentationConfig, TileConfig
from ..etsfile import ETSFile
from ..omezarr.ets_writer import write_ets_pyramid_to_ngff_zarr
from ..omezarr.writers import write_ngff_from_mips_ngffzarr
from ..vsi_converter import find_ets_file, get_vsi_metadata
from .plating import process_slide_with_plating

logger = logging.getLogger(__name__)


def _expected_dataset_paths_from_metadata(metadata: dict[str, Any]) -> list[str]:
    canonical = metadata.get("canonical_metadata")
    if isinstance(canonical, dict):
        dataset_paths = canonical.get("dataset_paths")
        if isinstance(dataset_paths, list) and dataset_paths:
            return [str(path) for path in dataset_paths]

    for key in ("ngff", "ngff_v04", "ngff_latest"):
        ngff = metadata.get(key)
        if not isinstance(ngff, dict):
            continue
        multiscales = ngff.get("multiscales")
        if not isinstance(multiscales, list) or not multiscales:
            continue
        datasets = multiscales[0].get("datasets")
        if isinstance(datasets, list) and datasets:
            paths = [dataset.get("path") for dataset in datasets if isinstance(dataset, dict)]
            if all(path is not None for path in paths):
                return [str(path) for path in paths]

    num_levels = metadata.get("num_levels")
    if num_levels is not None:
        return [f"s{level}" for level in range(int(num_levels))]

    return []


def _missing_source_ome_zarr_arrays(output_path: Path, metadata: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for dataset_path in _expected_dataset_paths_from_metadata(metadata):
        array_path = output_path / dataset_path
        if not ((array_path / ".zarray").is_file() or (array_path / "zarr.json").is_file()):
            missing.append(dataset_path)
    return missing


def _source_ome_zarr_shape_errors(
    output_path: Path,
    metadata: dict[str, Any],
    ets_path: Path,
) -> list[str]:
    """Return dataset paths whose cached array shapes do not match the ETS pyramid."""
    dataset_paths = _expected_dataset_paths_from_metadata(metadata)
    if not dataset_paths:
        return []

    channel_count = 3
    canonical = metadata.get("canonical_metadata")
    if isinstance(canonical, dict) and canonical.get("channel_count") is not None:
        channel_count = int(canonical["channel_count"])
    elif metadata.get("channel_count") is not None:
        channel_count = int(metadata["channel_count"])

    errors: list[str] = []
    with ETSFile(ets_path) as ets:
        for level, dataset_path in enumerate(dataset_paths):
            zarray_path = output_path / dataset_path / ".zarray"
            if not zarray_path.is_file():
                continue
            try:
                zarray = json.loads(zarray_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                errors.append(f"{dataset_path} (unreadable .zarray)")
                continue

            height, width = ets.level_shape(level)
            expected_shape = [channel_count, int(height), int(width)]
            actual_shape = zarray.get("shape")
            if list(actual_shape) != expected_shape:
                errors.append(f"{dataset_path} shape={actual_shape!r}, expected={expected_shape!r}")
    return errors


def _temporary_source_path(output_path: Path) -> Path:
    """Return the hidden incomplete output path used for source conversion."""
    return output_path.with_name(f".{output_path.name}.incomplete")


def _promote_completed_source(temp_path: Path, output_path: Path) -> None:
    """Replace a source OME-Zarr only after a complete temp tree exists."""
    if output_path.exists():
        shutil.rmtree(output_path)
    temp_path.rename(output_path)


def _physical_xy_from_metadata(metadata: dict[str, Any]) -> tuple[float, float] | None:
    physical = metadata.get("physical_pixel_size_um")
    if isinstance(physical, dict):
        x_um = physical.get("x")
        y_um = physical.get("y")
        if x_um is not None and y_um is not None:
            return float(x_um), float(y_um)
    canonical = metadata.get("canonical_metadata")
    if isinstance(canonical, dict):
        canonical_physical = canonical.get("physical_pixel_size_um")
        if isinstance(canonical_physical, dict):
            x_um = canonical_physical.get("x")
            y_um = canonical_physical.get("y")
            if x_um is not None and y_um is not None:
                return float(x_um), float(y_um)
    return None


def vsi_to_source_ome_zarr(
    vsi_path: str | Path,
    output_path: str | Path,
    *,
    metadata_backend: str = "auto",
    metadata_schema: str = "v0.4",
    chunks_xy: int = 512,
    overwrite: bool = False,
    source_writer: str = "direct",
) -> tuple[Path, Path, dict[str, Any]]:
    """
    Convert a VSI/ETS pyramid to a source OME-Zarr pyramid without flat files.

    Returns ``(source_ome_zarr_path, ets_path, vsi_metadata)``.
    """
    vsi_path = Path(vsi_path)
    output_path = Path(output_path)
    ets_path = find_ets_file(vsi_path)
    if ets_path is None:
        raise FileNotFoundError(f"No ETS file found for VSI {vsi_path}")

    metadata = get_vsi_metadata(
        vsi_path,
        metadata_backend=metadata_backend,
        target_schema="latest",
    )
    if not metadata:
        raise RuntimeError(f"Unable to extract structural metadata for VSI {vsi_path}")

    if output_path.exists() and not overwrite:
        missing = _missing_source_ome_zarr_arrays(output_path, metadata)
        if missing:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview += f", ... ({len(missing)} missing total)"
            raise RuntimeError(
                f"Existing source OME-Zarr at {output_path} appears incomplete; "
                f"missing dataset array(s): {preview}. "
                "Rerun with overwrite_source=True or choose a fresh output directory."
            )
        shape_errors = _source_ome_zarr_shape_errors(output_path, metadata, Path(ets_path))
        if shape_errors:
            preview = "; ".join(shape_errors[:3])
            if len(shape_errors) > 3:
                preview += f"; ... ({len(shape_errors)} shape mismatches total)"
            raise RuntimeError(
                f"Existing source OME-Zarr at {output_path} does not match the ETS pyramid; "
                f"{preview}. Rerun with overwrite_source=True or choose a fresh output directory."
            )
        return output_path, Path(ets_path), metadata

    channel_labels = metadata.get("channel_labels")
    phys_xy_um = _physical_xy_from_metadata(metadata)
    if phys_xy_um is None:
        logger.warning(
            "Physical pixel size unavailable for %s; source OME-Zarr will use 1.0 um fallback scales.",
            vsi_path,
        )
        phys_xy_um = (1.0, 1.0)

    writer_name = source_writer.strip().lower().replace("_", "-")
    if writer_name not in {"direct", "ngff-zarr"}:
        raise ValueError("source_writer must be one of ['direct', 'ngff-zarr'].")

    temp_output_path = _temporary_source_path(output_path)
    if temp_output_path.exists():
        shutil.rmtree(temp_output_path)

    if writer_name == "direct":
        write_ets_pyramid_to_ngff_zarr(
            ets_path,
            temp_output_path,
            phys_xy_um=phys_xy_um,
            name=vsi_path.stem,
            chunks_xy=chunks_xy,
            overwrite=True,
            channel_labels=channel_labels if isinstance(channel_labels, list) else None,
            channel_colors=["FFFFFF"] * 3,
            add_omero=True,
            ngff_metadata=metadata,
            metadata_schema=metadata_schema,
        )
    else:
        ets = ETSFile(ets_path)
        try:
            levels_yxc = [ets.to_dask(level) for level in range(ets.nlevels)]
            write_ngff_from_mips_ngffzarr(
                mips_yxc=levels_yxc,
                out_dir=temp_output_path,
                phys_xy_um=phys_xy_um,
                name=vsi_path.stem,
                chunks_xy=chunks_xy,
                version="0.4",
                overwrite=True,
                channel_labels=channel_labels if isinstance(channel_labels, list) else None,
                channel_colors=["FFFFFF"] * 3,
                add_omero=True,
                ngff_metadata=metadata,
                metadata_schema=metadata_schema,
            )
        finally:
            ets.close()

    _promote_completed_source(temp_output_path, output_path)

    return output_path, Path(ets_path), metadata


def process_vsi_directory_with_plating(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "*.vsi",
    source_ome_zarr_dir: str | Path | None = None,
    per_tissue_dir: str | Path | None = None,
    source_level: int | str = 0,
    segmentation_level: int | str | None = 7,
    segmentation_config: SegmentationConfig | None = None,
    tile_config: TileConfig | None = None,
    metadata_backend: str = "auto",
    metadata_schema: str = "v0.4",
    overwrite_source: bool = False,
    source_writer: str = "direct",
    parallel: bool = False,
    min_side_for_mips: int | None = None,
    dtype: np.dtype | None = "uint8",
) -> dict[str, list[Path]]:
    """
    Process all matching VSI files into per-tissue OME-Zarr derivatives.

    The returned mapping mirrors ``process_directory()`` style:
    ``{input_vsi_path: [tissue_ome_zarr_paths...]}``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    source_ome_zarr_dir = (
        Path(source_ome_zarr_dir) if source_ome_zarr_dir else output_dir / "source_ome_zarr"
    )
    per_tissue_dir = Path(per_tissue_dir) if per_tissue_dir else output_dir / "per_tissue_ngff"
    source_ome_zarr_dir.mkdir(parents=True, exist_ok=True)
    per_tissue_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[Path]] = {}
    vsi_paths = sorted(input_dir.glob(pattern))
    if not vsi_paths:
        logger.warning("No VSI files matched %s in %s", pattern, input_dir)
        return results

    chunk_xy = tile_config.chunk_size if tile_config is not None else 512
    for vsi_path in vsi_paths:
        source_ome_zarr = source_ome_zarr_dir / f"{vsi_path.stem}.ome.zarr"
        source_ome_zarr, ets_path, metadata = vsi_to_source_ome_zarr(
            vsi_path,
            source_ome_zarr,
            metadata_backend=metadata_backend,
            metadata_schema=metadata_schema,
            chunks_xy=chunk_xy,
            overwrite=overwrite_source,
            source_writer=source_writer,
        )
        tissue_paths = process_slide_with_plating(
            source_ome_zarr,
            per_tissue_dir,
            source_level=source_level,
            segmentation_level=segmentation_level,
            segmentation_config=segmentation_config,
            tile_config=tile_config,
            parallel=parallel,
            min_side_for_mips=min_side_for_mips,
            dtype=dtype,
            source_context={
                "source_kind": "vsi",
                "source_path": str(vsi_path),
                "source_vsi": str(vsi_path),
                "source_ets": str(ets_path),
                "source_ome_zarr": str(source_ome_zarr),
                "ngff_metadata": metadata,
                "metadata_backend": metadata_backend,
                "metadata_schema": metadata_schema,
            },
        )
        results[str(vsi_path)] = tissue_paths

    return results
