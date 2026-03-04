"""
VSI/ETS File Conversion Utilities

Provides functions for finding ETS files within VSI directory structures
and converting them to flat file formats.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

from .bioformats_runtime import ensure_bioformats_jnius
from .etsfile import ETSFile

logger = logging.getLogger(__name__)

SOURCE_ARRAY_ORDER = ["y", "x", "c"]
NGFF_AXIS_ORDER = ["c", "y", "x"]
SUPPORTED_METADATA_BACKENDS = {"auto", "bioformats", "ets_only"}
SUPPORTED_TARGET_SCHEMAS = {"latest", "v0.4"}


def find_ets_file(vsi_fname: str | Path) -> Path | None:
    """
    Find the corresponding ETS file for a VSI file.

    VSI files have associated ETS files in a subfolder with the same name.
    This function finds the main (largest/highest-numbered) ETS file.

    Parameters
    ----------
    vsi_fname : str or Path
        Path to the VSI file.

    Returns
    -------
    Path or None
        Path to the main ETS file, or None if not found.

    Examples
    --------
    >>> ets_path = find_ets_file("data/specimen.vsi")
    >>> if ets_path:
    ...     ets = ETSFile(ets_path)

    Notes
    -----
    VSI file structure typically looks like:

    data/
    |-- specimen.vsi
    `-- _specimen_/
        |-- stack10001/
        |   `-- frame_t.ets  (thumbnail)
        `-- stack10002/
            `-- frame_t.ets  (full resolution)

    This function returns the ETS file in the highest-numbered stack folder.
    """
    p = Path(vsi_fname)
    if not p.exists():
        return None

    # Search in subfolder with pattern _<vsi_stem>_
    ets_folder = p.parent / f"_{p.stem}_"
    if not ets_folder.exists():
        return None

    # Find all ETS files
    longest = ""
    ets_full = None

    for ets in ets_folder.rglob("*.ets"):
        folder = ets.parent.name
        # Take the highest-numbered folder (typically 'stack10002' > 'stack10001')
        if folder > longest:
            longest = folder
            ets_full = ets

    return ets_full


def find_all_ets_files(vsi_fname: str | Path) -> list[Path]:
    """
    Find all ETS files associated with a VSI file.

    Parameters
    ----------
    vsi_fname : str or Path
        Path to the VSI file.

    Returns
    -------
    list of Path
        All ETS files found, sorted by folder name.
    """
    p = Path(vsi_fname)
    if not p.exists():
        return []

    ets_folder = p.parent / f"_{p.stem}_"
    if not ets_folder.exists():
        return []

    return sorted(ets_folder.rglob("*.ets"), key=lambda x: x.parent.name)


def _normalize_ome_tiff_output_path(output_path: str | Path) -> Path:
    """Normalize output paths to the `.ome.tif` extension."""
    path = Path(output_path)
    if path.name.lower().endswith(".ome.tif"):
        return path

    base_name = path.name
    for suffix in path.suffixes:
        base_name = base_name[: -len(suffix)]

    return path.with_name(f"{base_name}.ome.tif")


def _infer_vsi_path_from_ets(ets_fname: str | Path) -> Path | None:
    """Infer a sibling VSI path from a standard ETS stack layout."""
    ets_path = Path(ets_fname)
    stack_dir = ets_path.parent
    vsi_container = stack_dir.parent

    if (
        stack_dir.name.lower().startswith("stack")
        and vsi_container.name.startswith("_")
        and vsi_container.name.endswith("_")
    ):
        vsi_stem = vsi_container.name[1:-1]
        candidate = vsi_container.parent / f"{vsi_stem}.vsi"
        if candidate.exists():
            return candidate

    return None


def _ome_metadata_from_vsi_payload(vsi_metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Build a minimal OME metadata dict from VSI metadata when available."""
    ome_metadata: dict[str, Any] = {"axes": "YXS"}
    if not vsi_metadata:
        return ome_metadata

    channel_count = _safe_int(vsi_metadata.get("channel_count"))
    if channel_count is not None and channel_count != 3:
        raise ValueError(
            f"Unsupported VSI channel count {channel_count}; the ETS reader currently expects RGB data."
        )

    channel_labels = vsi_metadata.get("channel_labels")
    if isinstance(channel_labels, list) and channel_labels:
        ome_metadata["Channel"] = {"Name": [str(label) for label in channel_labels[:3]]}

    physical_pixel_size_um = vsi_metadata.get("physical_pixel_size_um")
    if isinstance(physical_pixel_size_um, dict):
        size_x = _safe_float(physical_pixel_size_um.get("x"))
        size_y = _safe_float(physical_pixel_size_um.get("y"))
        if size_x is not None:
            ome_metadata["PhysicalSizeX"] = size_x
            ome_metadata["PhysicalSizeXUnit"] = "um"
        if size_y is not None:
            ome_metadata["PhysicalSizeY"] = size_y
            ome_metadata["PhysicalSizeYUnit"] = "um"

    return ome_metadata


def vsi_to_flat_image(
    vsi_fname: str | Path,
    level: int = 0,
    output_path: str | Path | None = None,
    format: str = "jpg",
    jpeg_quality: int = 95,
) -> np.ndarray | None:
    """
    Convert a VSI file to a flat image file.

    Parameters
    ----------
    vsi_fname : str or Path
        Path to the VSI file.
    level : int
        Pyramid level to extract (0 = full resolution).
    output_path : str or Path, optional
        Path to save the output image. If None, returns array only.
    format : str
        Output format: 'jpg', 'png', or 'tiff'.
    jpeg_quality : int
        JPEG quality (1-100) if format is 'jpg'.

    Returns
    -------
    np.ndarray or None
        The extracted image as RGB array, or None if extraction fails.

    Examples
    --------
    >>> # Get array only
    >>> img = vsi_to_flat_image("specimen.vsi", level=2)
    >>>
    >>> # Save to file
    >>> img = vsi_to_flat_image("specimen.vsi", level=2,
    ...                          output_path="output/specimen_level2.jpg")
    """
    ets_path = find_ets_file(vsi_fname)
    if ets_path is None:
        return None

    return ets_to_flat_image(
        ets_path,
        level=level,
        output_path=output_path,
        format=format,
        jpeg_quality=jpeg_quality,
    )


def ets_to_flat_image(
    ets_fname: str | Path,
    level: int = 0,
    output_path: str | Path | None = None,
    format: str = "jpg",
    jpeg_quality: int = 95,
) -> np.ndarray | None:
    """
    Convert an ETS file to a flat image.

    Parameters
    ----------
    ets_fname : str or Path
        Path to the ETS file.
    level : int
        Pyramid level to extract.
    output_path : str or Path, optional
        Path to save the output image.
    format : str
        Output format.
    jpeg_quality : int
        JPEG quality if applicable.

    Returns
    -------
    np.ndarray or None
        The extracted image as RGB array.
    """
    try:
        with ETSFile(ets_fname) as ets:
            if level < 0 or level >= ets.nlevels:
                raise ValueError(
                    f"Level {level} out of range [0, {ets.nlevels - 1}]"
                )

            img = ets.read_level(level)

            if output_path is not None:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # OpenCV expects BGR
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if format.lower() in ("jpg", "jpeg"):
                    cv2.imwrite(
                        str(output_path),
                        img_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                    )
                elif format.lower() == "png":
                    cv2.imwrite(str(output_path), img_bgr)
                elif format.lower() in ("tif", "tiff"):
                    cv2.imwrite(str(output_path), img_bgr)
                else:
                    raise ValueError(f"Unsupported format: {format}")

            return img

    except Exception as e:
        logger.error("Error processing %s: %s", ets_fname, e)
        return None


def ets_to_ome_tiff(
    ets_fname: str | Path,
    output_path: str | Path,
    *,
    vsi_fname: str | Path | None = None,
    metadata_backend: str = "auto",
    tile_size: int = 512,
    compression: str = "jpeg",
) -> Path | None:
    """
    Convert an ETS file to pyramidal OME-TIFF.

    Parameters
    ----------
    ets_fname : str or Path
        Path to the ETS file.
    output_path : str or Path
        Destination path. Normalized to ``.ome.tif``.
    vsi_fname : str or Path, optional
        Associated VSI path used for physical metadata extraction.
    metadata_backend : {"auto", "bioformats", "ets_only"}
        Metadata backend used when VSI metadata is available.
    tile_size : int
        OME-TIFF tile edge size in pixels.
    compression : str
        TIFF compression mode passed to ``tifffile`` (for example ``"jpeg"``).

    Returns
    -------
    Path or None
        Output path on success, otherwise ``None``.
    """
    backend = _normalize_metadata_backend(metadata_backend)
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")

    normalized_output_path = _normalize_ome_tiff_output_path(output_path)
    normalized_output_path.parent.mkdir(parents=True, exist_ok=True)

    ets_path = Path(ets_fname)
    vsi_path = Path(vsi_fname) if vsi_fname is not None else _infer_vsi_path_from_ets(ets_path)
    vsi_metadata: dict[str, Any] | None = None

    if backend == "bioformats" and vsi_path is None:
        raise RuntimeError(
            "Bio-Formats metadata backend requires a VSI path; provide `vsi_fname`."
        )

    if vsi_path is not None:
        try:
            metadata_payload = get_vsi_metadata(
                vsi_path,
                metadata_backend=backend,
                target_schema="latest",
            )
            if metadata_payload:
                vsi_metadata = metadata_payload
            elif backend == "bioformats":
                raise RuntimeError(
                    f"Bio-Formats metadata backend returned empty metadata for {vsi_path}."
                )
            elif backend == "auto":
                logger.warning(
                    "VSI metadata extraction returned empty metadata for %s; "
                    "writing OME-TIFF with minimal metadata.",
                    vsi_path,
                )
        except ValueError:
            raise
        except Exception as exc:
            if backend == "bioformats":
                raise
            logger.warning(
                "Unable to resolve VSI metadata for %s (%s); writing OME-TIFF with minimal metadata.",
                vsi_path,
                exc,
            )
    elif backend == "auto":
        logger.warning(
            "No VSI path available for %s; writing OME-TIFF with minimal metadata.",
            ets_path,
        )

    ome_metadata = _ome_metadata_from_vsi_payload(vsi_metadata)

    try:
        with ETSFile(ets_path) as ets:
            if ets.nlevels <= 0:
                raise ValueError("ETS file has no pyramid levels.")

            write_options = {
                "tile": (tile_size, tile_size),
                "compression": compression,
                "photometric": "rgb",
            }

            with tifffile.TiffWriter(str(normalized_output_path), bigtiff=True) as ome_writer:
                for level in range(ets.nlevels):
                    image = ets.read_level(level)
                    channel_count = image.shape[-1] if image.ndim == 3 else 1
                    if channel_count != 3:
                        raise ValueError(
                            "Unsupported VSI channel count "
                            f"{channel_count}; the ETS reader currently expects RGB data."
                        )

                    if level == 0:
                        level0_options = dict(write_options)
                        if ets.nlevels > 1:
                            level0_options["subifds"] = ets.nlevels - 1
                        level0_options["metadata"] = ome_metadata
                        ome_writer.write(image, **level0_options)
                    else:
                        ome_writer.write(image, subfiletype=1, **write_options)

        return normalized_output_path
    except ValueError:
        raise
    except Exception as exc:
        logger.error("Error writing OME-TIFF %s from %s: %s", normalized_output_path, ets_path, exc)
        return None


def vsi_to_ome_tiff(
    vsi_fname: str | Path,
    output_path: str | Path,
    *,
    metadata_backend: str = "auto",
    tile_size: int = 512,
    compression: str = "jpeg",
) -> Path | None:
    """
    Convert a VSI file to pyramidal OME-TIFF.

    Parameters
    ----------
    vsi_fname : str or Path
        Path to the VSI file.
    output_path : str or Path
        Destination path. Normalized to ``.ome.tif``.
    metadata_backend : {"auto", "bioformats", "ets_only"}
        Metadata backend selection.
    tile_size : int
        OME-TIFF tile edge size in pixels.
    compression : str
        TIFF compression mode passed to ``tifffile``.
    """
    ets_path = find_ets_file(vsi_fname)
    if ets_path is None:
        return None

    normalized_output_path = _normalize_ome_tiff_output_path(output_path)
    return ets_to_ome_tiff(
        ets_path,
        normalized_output_path,
        vsi_fname=vsi_fname,
        metadata_backend=metadata_backend,
        tile_size=tile_size,
        compression=compression,
    )


def batch_convert_vsi(
    input_pattern: str,
    output_dir: str | Path,
    level: int = 0,
    format: str = "jpg",
    jpeg_quality: int = 95,
    verbose: bool = True,
) -> list[Path]:
    """
    Batch convert VSI files to flat images.

    Parameters
    ----------
    input_pattern : str
        Glob pattern for VSI files (e.g., "data/*.vsi").
    output_dir : str or Path
        Directory to save output images.
    level : int
        Pyramid level to extract.
    format : str
        Output format.
    jpeg_quality : int
        JPEG quality if applicable.
    verbose : bool
        Print progress information.

    Returns
    -------
    list of Path
        Paths to successfully created output files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vsi_files = glob.glob(input_pattern)
    if not vsi_files:
        logger.warning("No files match pattern: %s", input_pattern)
        return []

    logger.info("Processing %d files...", len(vsi_files))

    format_normalized = format.strip().lower()
    output_files = []
    for vsi_path in vsi_files:
        vsi_path = Path(vsi_path)
        output_name = (
            f"level_{level}_{vsi_path.stem}.ome.tif"
            if format_normalized == "ome-tiff"
            else f"level_{level}_{vsi_path.stem}.{format}"
        )
        output_path = output_dir / output_name

        logger.debug("  %s -> %s", vsi_path.name, output_name)

        if format_normalized == "ome-tiff":
            ome_tiff_path = vsi_to_ome_tiff(vsi_path, output_path)
            if ome_tiff_path is not None:
                output_files.append(ome_tiff_path)
        else:
            img = vsi_to_flat_image(
                vsi_path,
                level=level,
                output_path=output_path,
                format=format,
                jpeg_quality=jpeg_quality,
            )

            if img is not None:
                output_files.append(output_path)

    logger.info("Completed: %d/%d files", len(output_files), len(vsi_files))
    return output_files


def _normalize_metadata_backend(metadata_backend: str) -> str:
    """Normalize supported metadata backend names."""
    backend = metadata_backend.strip().lower()
    if backend not in SUPPORTED_METADATA_BACKENDS:
        raise ValueError(
            f"Unsupported metadata_backend '{metadata_backend}'. "
            f"Expected one of {sorted(SUPPORTED_METADATA_BACKENDS)}."
        )
    return backend


def _normalize_target_schema(target_schema: str) -> str:
    """Normalize schema aliases to the supported NGFF projections."""
    schema = target_schema.strip().lower()
    if schema == "0.4":
        schema = "v0.4"
    if schema not in SUPPORTED_TARGET_SCHEMAS:
        raise ValueError(
            f"Unsupported target_schema '{target_schema}'. "
            f"Expected one of ['latest', 'v0.4', '0.4']."
        )
    return schema


def _safe_float(value: Any) -> float | None:
    """Convert Java or Python numeric metadata values to float."""
    if value is None:
        return None
    if hasattr(value, "value") and callable(value.value):
        try:
            value = value.value()
        except Exception:
            return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Convert metadata values to int when possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_call(metadata_store: Any, method_name: str, *args: Any) -> Any:
    """Call a Bio-Formats metadata accessor if it exists."""
    method = getattr(metadata_store, method_name, None)
    if method is None:
        return None
    try:
        return method(*args)
    except Exception:
        return None


def _extract_channel_labels(
    metadata_store: Any,
    image_index: int,
    channel_count: int,
) -> list[str]:
    """Extract human-readable channel labels when present."""
    labels: list[str] = []
    for channel_index in range(channel_count):
        label = _metadata_call(metadata_store, "getChannelName", image_index, channel_index)
        if label is None:
            label = _metadata_call(metadata_store, "getChannelID", image_index, channel_index)
        labels.append(str(label) if label is not None else f"ch{channel_index}")
    return labels


def _extract_stage_origin_um(metadata_store: Any, image_index: int) -> dict[str, float] | None:
    """
    Extract stage or plane origin in micrometers when available.

    The exact fields depend on the Bio-Formats reader, so this performs a
    best-effort lookup across common OME metadata accessors.
    """
    origin: dict[str, float] = {}
    candidates = {
        "x": [
            ("getPlanePositionX", (image_index, 0)),
            ("getStageLabelX", (image_index,)),
        ],
        "y": [
            ("getPlanePositionY", (image_index, 0)),
            ("getStageLabelY", (image_index,)),
        ],
        "z": [
            ("getPlanePositionZ", (image_index, 0)),
            ("getStageLabelZ", (image_index,)),
        ],
    }

    for axis_name, methods in candidates.items():
        for method_name, args in methods:
            value = _safe_float(_metadata_call(metadata_store, method_name, *args))
            if value is not None:
                origin[axis_name] = value
                break

    return origin or None


def _extract_vsi_physical_metadata(vsi_fname: str | Path) -> dict[str, Any]:
    """
    Extract VSI physical-space metadata using Bio-Formats via Pyjnius.

    Raises
    ------
    RuntimeError
        If Bio-Formats or Pyjnius is unavailable or metadata extraction fails.
    ValueError
        If the VSI metadata describes an unsupported channel layout.
    """
    jnius = ensure_bioformats_jnius()

    autoclass = getattr(jnius, "autoclass", None)
    if autoclass is None:
        raise RuntimeError("Bio-Formats metadata backend unavailable: 'jnius.autoclass' not found.")

    try:
        image_reader_cls = autoclass("loci.formats.ImageReader")
        metadata_tools_cls = autoclass("loci.formats.meta.MetadataTools")
    except Exception as exc:
        raise RuntimeError("Bio-Formats classes could not be loaded via Pyjnius.") from exc

    reader = image_reader_cls()
    metadata_store = metadata_tools_cls.createOMEXMLMetadata()
    reader.setMetadataStore(metadata_store)

    try:
        reader.setId(str(vsi_fname))

        size_x = _safe_int(reader.getSizeX())
        size_y = _safe_int(reader.getSizeY())
        size_z = _safe_int(reader.getSizeZ())
        size_c = _safe_int(reader.getSizeC()) or 3
        size_t = _safe_int(reader.getSizeT())

        if size_c != 3:
            raise ValueError(
                f"Unsupported VSI channel count {size_c}; the ETS reader currently expects RGB data."
            )

        physical_size_x = _safe_float(_metadata_call(metadata_store, "getPixelsPhysicalSizeX", 0))
        physical_size_y = _safe_float(_metadata_call(metadata_store, "getPixelsPhysicalSizeY", 0))
        physical_size_z = _safe_float(_metadata_call(metadata_store, "getPixelsPhysicalSizeZ", 0))
        stage_origin_um = _extract_stage_origin_um(metadata_store, 0)
        channel_labels = _extract_channel_labels(metadata_store, 0, size_c)

        return {
            "name": Path(vsi_fname).name,
            "sizeX": size_x,
            "sizeY": size_y,
            "sizeZ": size_z,
            "sizeC": size_c,
            "sizeT": size_t,
            "physical_size_x": physical_size_x,
            "physical_size_y": physical_size_y,
            "physical_size_z": physical_size_z,
            "channel_labels": channel_labels,
            "stage_origin_um": stage_origin_um,
        }
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to extract VSI metadata from {vsi_fname}.") from exc
    finally:
        try:
            reader.close()
        except Exception:
            pass


def _ngff_axes(channel_name: str = "c") -> list[dict[str, str]]:
    """Return NGFF-style axis definitions for a 2D RGB image."""
    return [
        {"name": channel_name, "type": "channel"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]


def _array_coordinate_system() -> dict[str, Any]:
    """Coordinate system for the source array axes."""
    return {
        "name": "array",
        "axes": [
            {"name": "c", "type": "channel"},
            {"name": "y", "type": "array"},
            {"name": "x", "type": "array"},
        ],
    }


def _physical_coordinate_system(name: str) -> dict[str, Any]:
    """Coordinate system for physical image-space axes."""
    return {"name": name, "axes": _ngff_axes()}


def _build_canonical_vsi_metadata(
    vsi_path: Path,
    ets: ETSFile,
    vendor_metadata: dict[str, Any] | None,
    warnings: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Build a schema-neutral metadata model from ETS and VSI metadata."""
    vendor_metadata = dict(vendor_metadata or {})
    channel_count = int(vendor_metadata.get("sizeC", 3) or 3)
    if channel_count != 3:
        warnings.append(
            f"Unsupported VSI channel count {channel_count}; the ETS reader currently expects RGB data."
        )
        raise ValueError(warnings[-1])

    channel_labels = list(vendor_metadata.get("channel_labels") or [f"ch{idx}" for idx in range(3)])
    physical_pixel_size_um = {
        "x": _safe_float(vendor_metadata.get("physical_size_x")),
        "y": _safe_float(vendor_metadata.get("physical_size_y")),
        "z": _safe_float(vendor_metadata.get("physical_size_z")),
    }

    if vendor_metadata.get("sizeX") not in (None, ets.npix_x):
        warnings.append(
            "Bio-Formats VSI width does not match ETS width; using ETS dimensions as the structural source."
        )
    if vendor_metadata.get("sizeY") not in (None, ets.npix_y):
        warnings.append(
            "Bio-Formats VSI height does not match ETS height; using ETS dimensions as the structural source."
        )

    stage_origin_raw = vendor_metadata.get("stage_origin_um") or {}
    stage_origin_um = {
        axis_name: value
        for axis_name, value in {
            "x": _safe_float(stage_origin_raw.get("x")),
            "y": _safe_float(stage_origin_raw.get("y")),
            "z": _safe_float(stage_origin_raw.get("z")),
        }.items()
        if value is not None
    }
    if not stage_origin_um:
        stage_origin_um = None

    dataset_paths = [f"s{level}" for level in range(ets.nlevels)]
    level_scales = {
        level: (
            [1.0, physical_pixel_size_um["y"] * (2**level), physical_pixel_size_um["x"] * (2**level)]
            if physical_pixel_size_um["x"] is not None and physical_pixel_size_um["y"] is not None
            else None
        )
        for level in range(ets.nlevels)
    }

    if physical_pixel_size_um["x"] is None or physical_pixel_size_um["y"] is None:
        warnings.append(
            "Physical pixel sizes for X and Y were not available; NGFF physical transforms are incomplete."
        )

    return {
        "name": vsi_path.stem,
        "source_array_order": list(SOURCE_ARRAY_ORDER),
        "ngff_axis_order": list(NGFF_AXIS_ORDER),
        "axes": _ngff_axes(),
        "dataset_paths": dataset_paths,
        "channel_count": channel_count,
        "channel_labels": channel_labels,
        "physical_pixel_size_um": physical_pixel_size_um,
        "stage_origin_um": stage_origin_um,
        "coordinate_systems": [
            _array_coordinate_system(),
            _physical_coordinate_system("image-plane"),
            *([_physical_coordinate_system("slide")] if stage_origin_um else []),
        ],
        "level_scales": level_scales,
        "vendor_metadata": vendor_metadata,
    }, warnings


def _project_ngff_latest(canonical: dict[str, Any]) -> dict[str, Any]:
    """Project canonical metadata to a latest-NGFF-oriented root-attrs payload."""
    base_transforms: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []
    stage_origin = canonical.get("stage_origin_um")

    level_zero_scale = canonical["level_scales"].get(0)
    if level_zero_scale is not None:
        base_transforms.append(
            {
                "type": "scale",
                "input": "array",
                "output": "image-plane",
                "scale": list(level_zero_scale),
            }
        )

    if stage_origin:
        base_transforms.append(
            {
                "type": "translation",
                "input": "image-plane",
                "output": "slide",
                "translation": [0.0, stage_origin.get("y", 0.0), stage_origin.get("x", 0.0)],
            }
        )

    for level, path in enumerate(canonical["dataset_paths"]):
        dataset_transforms: list[dict[str, Any]] = []
        level_scale = canonical["level_scales"].get(level)
        if level_scale is not None:
            dataset_transforms.append(
                {
                    "type": "scale",
                    "input": "array",
                    "output": "image-plane",
                    "scale": list(level_scale),
                }
            )
        if stage_origin:
            dataset_transforms.append(
                {
                    "type": "translation",
                    "input": "image-plane",
                    "output": "slide",
                    "translation": [0.0, stage_origin.get("y", 0.0), stage_origin.get("x", 0.0)],
                }
            )

        datasets.append(
            {
                "path": path,
                "coordinateTransformations": dataset_transforms,
            }
        )

    return {
        "schema": "latest",
        "dimension_order": list(canonical["ngff_axis_order"]),
        "axes": [dict(axis) for axis in canonical["axes"]],
        "coordinateSystems": [dict(system) for system in canonical["coordinate_systems"]],
        "arrayToPhysicalTransformations": base_transforms,
        "multiscales": [
            {
                "name": canonical["name"],
                "axes": [dict(axis) for axis in canonical["axes"]],
                "coordinateSystems": [dict(system) for system in canonical["coordinate_systems"]],
                "datasets": datasets,
            }
        ],
    }


def _project_ngff_v04(canonical: dict[str, Any]) -> dict[str, Any]:
    """Project canonical metadata to a v0.4-compatible root-attrs payload."""
    datasets = []
    for level, path in enumerate(canonical["dataset_paths"]):
        transforms = []
        level_scale = canonical["level_scales"].get(level)
        if level_scale is not None:
            transforms.append({"type": "scale", "scale": list(level_scale)})
        datasets.append(
            {
                "path": path,
                "coordinateTransformations": transforms,
            }
        )

    return {
        "multiscales": [
            {
                "name": canonical["name"],
                "version": "0.4",
                "axes": [dict(axis) for axis in canonical["axes"]],
                "datasets": datasets,
            }
        ]
    }


def get_vsi_metadata(
    vsi_fname: str | Path,
    *,
    metadata_backend: str = "auto",
    target_schema: str = "latest",
) -> dict[str, Any]:
    """
    Extract metadata from a VSI file.

    Parameters
    ----------
    vsi_fname : str or Path
        Path to the VSI file.
    metadata_backend : {"auto", "bioformats", "ets_only"}
        Metadata backend selection. ``"auto"`` attempts Bio-Formats and falls
        back to ETS-only structural metadata when unavailable.
    target_schema : {"latest", "v0.4", "0.4"}
        Select the convenience alias stored at ``metadata["ngff"]`` while
        always emitting both schema projections.

    Returns
    -------
    dict
        Metadata dictionary with structural ETS properties plus canonical,
        latest-NGFF, and v0.4-compatible metadata projections.
    """
    backend = _normalize_metadata_backend(metadata_backend)
    selected_schema = _normalize_target_schema(target_schema)
    vsi_path = Path(vsi_fname)
    ets_path = find_ets_file(vsi_path)
    if ets_path is None:
        return {}

    warnings: list[str] = []
    vendor_metadata: dict[str, Any] | None = None
    physical_source = "ets_only"

    if backend != "ets_only":
        try:
            vendor_metadata = _extract_vsi_physical_metadata(vsi_path)
            physical_source = "bioformats"
        except ValueError:
            raise
        except RuntimeError as exc:
            if backend == "bioformats":
                raise
            warnings.append(f"{exc} Falling back to ETS-only metadata.")

    with ETSFile(ets_path) as ets:
        canonical_metadata, warnings = _build_canonical_vsi_metadata(
            vsi_path,
            ets,
            vendor_metadata,
            warnings,
        )
        ngff_latest = _project_ngff_latest(canonical_metadata)
        ngff_v04 = _project_ngff_v04(canonical_metadata)
        compatibility_warnings: list[str] = []
        lossy_fields_for_v04: list[str] = []

        if canonical_metadata["stage_origin_um"]:
            lossy_fields_for_v04.extend(
                ["named_coordinate_systems", "absolute_origin_translation"]
            )
            compatibility_warnings.append(
                "Stage-origin metadata is preserved only in the latest-NGFF projection."
            )

        is_ngff_minimum_complete = (
            canonical_metadata["physical_pixel_size_um"]["x"] is not None
            and canonical_metadata["physical_pixel_size_um"]["y"] is not None
        )

        metadata = {
            "vsi_path": str(vsi_fname),
            "ets_path": str(ets_path),
            "width": ets.npix_x,
            "height": ets.npix_y,
            "num_levels": ets.nlevels,
            "num_tiles": ets.ntiles,
            "tile_size": (ets.tile_xsize, ets.tile_ysize),
            "compression": ets.compression_str,
            "is_bgr": ets.is_bgr,
            "file_size_bytes": ets.fsize,
            "level_shapes": {
                lvl: ets.level_shape(lvl) for lvl in range(ets.nlevels)
            },
            "level_tile_counts": {
                lvl: ets.level_ntiles(lvl) for lvl in range(ets.nlevels)
            },
            "metadata_sources": {
                "ets": "ETSFile",
                "physical": physical_source,
            },
            "source_array_order": list(SOURCE_ARRAY_ORDER),
            "ngff_axis_order": list(NGFF_AXIS_ORDER),
            "channel_count": canonical_metadata["channel_count"],
            "channel_labels": list(canonical_metadata["channel_labels"]),
            "physical_pixel_size_um": dict(canonical_metadata["physical_pixel_size_um"]),
            "vendor_metadata": dict(canonical_metadata["vendor_metadata"]),
            "canonical_metadata": canonical_metadata,
            "ngff_latest": ngff_latest,
            "ngff_v04": ngff_v04,
            "compatibility": {
                "selected_schema": selected_schema,
                "automatic_v04_projection": True,
                "lossy_fields_for_v04": lossy_fields_for_v04,
                "warnings": compatibility_warnings,
            },
            "warnings": warnings,
            "is_ngff_minimum_complete": is_ngff_minimum_complete,
        }
        metadata["ngff"] = (
            metadata["ngff_latest"] if selected_schema == "latest" else metadata["ngff_v04"]
        )
        return metadata
