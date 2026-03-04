"""
NGFF metadata utilities for reading and writing OME-Zarr metadata.

Provides functions for parsing and generating NGFF-compliant metadata,
including multiscales, physical pixel sizes, and coordinate transformations.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import zarr


def _is_ngff_image_group(path: Path) -> bool:
    """Basic NGFF image check: requires .zgroup, .zattrs with 'multiscales'."""
    try:
        if not (path / ".zgroup").exists():
            return False
        attrs_p = path / ".zattrs"
        if not attrs_p.exists():
            return False
        attrs = json.loads(attrs_p.read_text())
        return "multiscales" in attrs
    except Exception:
        return False


def _safe_name(s: str) -> str:
    """
    Sanitize for filesystem folder name.
    """
    keep = [c if c.isalnum() or c in ("_", "-", ".") else "_" for c in s.strip()]
    name = "".join(keep).strip("._")
    return name or "untitled"


def _get_multiscales_paths(root: zarr.Group) -> list[str]:
    """
    Get the paths info from the datasets within the zarr group.
    """
    ms = root.attrs["multiscales"][0]
    return [d["path"] for d in ms["datasets"]]


def _phys_xy_um(root: zarr.Group, L: int=0) -> tuple[float,float]:
    """
    Read (phys_x_um, phys_y_um) from the child's NGFF multiscales at Lth resolution.
    L=0 is the highest resolution, used by default
    """
    # dpaths = _get_multiscales_paths(root)
    ms = root.attrs["multiscales"][0]
    scale = ms["datasets"][L]["coordinateTransformations"][0]["scale"]  # NGFF order: [c,y,x]
    phys_y = float(scale[1])
    phys_x = float(scale[2])
    return phys_x, phys_y # (px_um, py_um)


def _detect_source_ds_factor(root: zarr.Group) -> float:
    """
    Detect the source downsample schedule. Usually 2x or 4x.
    """
    ms = root.attrs["multiscales"][0]
    s = [ds["coordinateTransformations"][0]["scale"] for ds in ms["datasets"]]  # [ [1, py, px], ... ]
    ys = [float(si[1]) for si in s]  # use Y only (X should match)
    ratios = [ys[i+1] / ys[i] for i in range(len(ys)-1)]

    # Return the median rounded to 2 decimals (often ~2.0 or ~4.0)
    ratios.sort()

    return round(ratios[len(ratios)//2], 2)




def _sizes_for_mips_xy(W: int, H: int, levels: int) -> list[tuple[int,int]]:
    sizes = []
    w, h = W, H
    for _ in range(levels):
        sizes.append((w, h))
        w = max(1, w // 2)
        h = max(1, h // 2)
    return sizes

def _voxel_sizes_for_mips_xy(phys_xyz: int, levels: int, scale_factor: int = 2) -> list[tuple[int,int,int]]:
    """
    phys_xyz is (x_nm, y_nm, z_nm).
    Double XY per MIP; keep Z fixed.
    Returns [(x0,y0,z0), (x1,y1,z1), ...] as ints.
    """
    voxel_sizes = []
    x, y, z = phys_xyz
    x = int(round(x))
    y = int(round(y))
    z = int(round(z))
    for _ in range(levels):
        voxel_sizes.append((x, y, z))
        x *= scale_factor
        y *= scale_factor
        # Keep z the same since these are 2D plates
    return voxel_sizes


def _normalize_ngff_schema(schema: str | None) -> str:
    """Normalize NGFF schema aliases used by metadata adapters."""
    if schema is None:
        return "latest"
    normalized = schema.strip().lower()
    if normalized == "0.4":
        normalized = "v0.4"
    if normalized not in {"latest", "v0.4"}:
        raise ValueError(
            f"Unsupported NGFF schema '{schema}'. Expected one of ['latest', 'v0.4', '0.4']."
        )
    return normalized


def materialize_ngff_root_attrs(
    metadata: dict[str, Any],
    schema: str | None = None,
) -> dict[str, Any]:
    """
    Materialize root attributes for the requested NGFF schema.

    Parameters
    ----------
    metadata : dict
        Either the full ``get_vsi_metadata()`` payload or a direct NGFF
        projection that already contains ``multiscales``.
    schema : str, optional
        Target schema. Defaults to the selected schema recorded in the
        compatibility block, or ``"latest"`` when unspecified.

    Returns
    -------
    dict
        A deep-copied root-attributes payload suitable for Zarr writers.
    """
    if "multiscales" in metadata and "ngff_latest" not in metadata and "ngff_v04" not in metadata:
        return copy.deepcopy(metadata)

    selected_schema = schema
    if selected_schema is None:
        compatibility = metadata.get("compatibility", {})
        selected_schema = compatibility.get("selected_schema", "latest")

    normalized_schema = _normalize_ngff_schema(selected_schema)
    projection_key = "ngff_latest" if normalized_schema == "latest" else "ngff_v04"
    projection = metadata.get(projection_key)
    if not isinstance(projection, dict) or "multiscales" not in projection:
        raise ValueError(
            f"Metadata payload does not include a valid '{projection_key}' projection."
        )
    return copy.deepcopy(projection)


def _extract_scale_transform(dataset: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first scale transform from a dataset metadata block."""
    for transform in dataset.get("coordinateTransformations", []):
        if transform.get("type") == "scale" and "scale" in transform:
            return transform
    return None


def _metadata_channel_count(metadata: dict[str, Any] | None) -> int | None:
    """Best-effort channel-count extraction for writer validation."""
    if not isinstance(metadata, dict):
        return None
    for key_path in (
        ("channel_count",),
        ("canonical_metadata", "channel_count"),
        ("vendor_metadata", "sizeC"),
    ):
        current: Any = metadata
        for key in key_path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            try:
                return int(current)
            except (TypeError, ValueError):
                return None

    omero = metadata.get("omero")
    if isinstance(omero, dict):
        channels = omero.get("channels")
        if isinstance(channels, list):
            return len(channels)
    return None


def _metadata_channel_labels(metadata: dict[str, Any] | None) -> list[str] | None:
    """Best-effort channel-label extraction for OMERO metadata."""
    if not isinstance(metadata, dict):
        return None

    for key_path in (
        ("channel_labels",),
        ("canonical_metadata", "channel_labels"),
    ):
        current: Any = metadata
        for key in key_path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, list) and current:
            return [str(label) for label in current]

    omero = metadata.get("omero")
    if isinstance(omero, dict):
        channels = omero.get("channels")
        if isinstance(channels, list) and channels:
            return [str(channel.get("label", f"ch{idx}")) for idx, channel in enumerate(channels)]
    return None


def _dataset_paths(root_attrs: dict[str, Any]) -> list[str]:
    """Return multiscale dataset paths from root attrs."""
    multiscales = root_attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError("NGFF root attrs must include a non-empty 'multiscales' list.")
    datasets = multiscales[0].get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("NGFF root attrs must include a dataset list in multiscales[0].")
    return [str(dataset.get("path", "")) for dataset in datasets]


def _extract_phys_xy_from_root_attrs(root_attrs: dict[str, Any], level: int = 0) -> tuple[float, float] | None:
    """Read base physical spacing from root attrs when a scale transform exists."""
    try:
        dataset = root_attrs["multiscales"][0]["datasets"][level]
    except (KeyError, IndexError, TypeError):
        return None
    transform = _extract_scale_transform(dataset)
    if transform is None:
        return None
    scale = transform.get("scale", [])
    if len(scale) < 3:
        return None
    return float(scale[2]), float(scale[1])


def _build_default_ngff_root_attrs(
    *,
    name: str,
    dataset_count: int,
    phys_xy_um: tuple[float, float] | None,
    schema: str,
    channel_axis_name: str = "c",
) -> dict[str, Any]:
    """Build a minimal root-attrs payload when only array spacing is known."""
    normalized_schema = _normalize_ngff_schema(schema)
    datasets: list[dict[str, Any]] = []
    px_um = py_um = None
    if phys_xy_um is not None:
        px_um, py_um = map(float, phys_xy_um)

    for level in range(dataset_count):
        transforms: list[dict[str, Any]] = []
        if px_um is not None and py_um is not None:
            scale = [1.0, py_um * (2**level), px_um * (2**level)]
            if normalized_schema == "latest":
                transforms.append(
                    {
                        "type": "scale",
                        "input": "array",
                        "output": "image-plane",
                        "scale": scale,
                    }
                )
            else:
                transforms.append({"type": "scale", "scale": scale})
        datasets.append(
            {
                "path": f"s{level}",
                "coordinateTransformations": transforms,
            }
        )

    axes = [
        {"name": channel_axis_name, "type": "channel"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]

    if normalized_schema == "latest":
        array_to_physical: list[dict[str, Any]] = []
        if px_um is not None and py_um is not None:
            array_to_physical.append(
                {
                    "type": "scale",
                    "input": "array",
                    "output": "image-plane",
                    "scale": [1.0, py_um, px_um],
                }
            )
        return {
            "schema": "latest",
            "dimension_order": ["c", "y", "x"],
            "axes": axes,
            "coordinateSystems": [
                {
                    "name": "array",
                    "axes": [
                        {"name": "c", "type": "channel"},
                        {"name": "y", "type": "array"},
                        {"name": "x", "type": "array"},
                    ],
                },
                {
                    "name": "image-plane",
                    "axes": [
                        {"name": "c", "type": "channel"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ],
                },
            ],
            "arrayToPhysicalTransformations": array_to_physical,
            "multiscales": [
                {
                    "name": name,
                    "axes": axes,
                    "coordinateSystems": [
                        {
                            "name": "array",
                            "axes": [
                                {"name": "c", "type": "channel"},
                                {"name": "y", "type": "array"},
                                {"name": "x", "type": "array"},
                            ],
                        },
                        {
                            "name": "image-plane",
                            "axes": [
                                {"name": "c", "type": "channel"},
                                {"name": "y", "type": "space", "unit": "micrometer"},
                                {"name": "x", "type": "space", "unit": "micrometer"},
                            ],
                        },
                    ],
                    "datasets": datasets,
                }
            ],
        }

    return {
        "multiscales": [
            {
                "name": name,
                "version": "0.4",
                "axes": axes,
                "datasets": datasets,
            }
        ]
    }


def _inject_fallback_scales(
    root_attrs: dict[str, Any],
    *,
    schema: str,
    phys_xy_um: tuple[float, float],
) -> None:
    """Add scale transforms only when the supplied metadata omitted them."""
    px_um, py_um = map(float, phys_xy_um)
    multiscale = root_attrs["multiscales"][0]
    datasets = multiscale.get("datasets", [])
    for level, dataset in enumerate(datasets):
        transforms = list(dataset.get("coordinateTransformations", []))
        if _extract_scale_transform(dataset) is not None:
            continue
        scale = [1.0, py_um * (2**level), px_um * (2**level)]
        if schema == "latest":
            transforms.insert(
                0,
                {
                    "type": "scale",
                    "input": "array",
                    "output": "image-plane",
                    "scale": scale,
                },
            )
        else:
            transforms.insert(0, {"type": "scale", "scale": scale})
        dataset["coordinateTransformations"] = transforms

    if schema == "latest" and not root_attrs.get("arrayToPhysicalTransformations"):
        root_attrs["arrayToPhysicalTransformations"] = [
            {
                "type": "scale",
                "input": "array",
                "output": "image-plane",
                "scale": [1.0, py_um, px_um],
            }
        ]


def _prepare_ngff_writer_metadata(
    *,
    dataset_count: int,
    channel_count: int,
    name: str,
    fallback_phys_xy_um: tuple[float, float] | None,
    ngff_metadata: dict[str, Any] | None,
    metadata_schema: str | None,
    channel_labels: list[str] | None,
    channel_axis_name: str = "c",
) -> dict[str, Any]:
    """
    Resolve, validate, and normalize NGFF writer metadata.

    Returns a dict with keys ``root_attrs``, ``resolved_name``,
    ``resolved_phys_xy_um``, ``resolved_channel_labels``, and ``schema``.
    """
    normalized_schema = _normalize_ngff_schema(metadata_schema or "v0.4")

    if ngff_metadata is None:
        if fallback_phys_xy_um is None:
            raise ValueError("Physical pixel size is required when ngff_metadata is not provided.")
        root_attrs = _build_default_ngff_root_attrs(
            name=name,
            dataset_count=dataset_count,
            phys_xy_um=fallback_phys_xy_um,
            schema=normalized_schema,
            channel_axis_name=channel_axis_name,
        )
    else:
        root_attrs = materialize_ngff_root_attrs(ngff_metadata, normalized_schema)

    resolved_schema = normalized_schema
    if root_attrs.get("schema") == "latest":
        resolved_schema = "latest"
    else:
        try:
            version = root_attrs["multiscales"][0].get("version")
        except (KeyError, IndexError, TypeError):
            version = None
        if version == "0.4":
            resolved_schema = "v0.4"

    dataset_paths = _dataset_paths(root_attrs)
    if len(dataset_paths) != dataset_count:
        raise ValueError(
            f"NGFF metadata describes {len(dataset_paths)} levels, but the writer is emitting {dataset_count}."
        )

    metadata_channel_count = _metadata_channel_count(ngff_metadata)
    if metadata_channel_count is not None and metadata_channel_count != channel_count:
        raise ValueError(
            f"NGFF metadata describes {metadata_channel_count} channels, but the writer is emitting {channel_count}."
        )

    if _extract_phys_xy_from_root_attrs(root_attrs) is None and fallback_phys_xy_um is not None:
        _inject_fallback_scales(
            root_attrs,
            schema=resolved_schema,
            phys_xy_um=fallback_phys_xy_um,
        )

    resolved_phys_xy_um = _extract_phys_xy_from_root_attrs(root_attrs)
    if resolved_phys_xy_um is None:
        resolved_phys_xy_um = (
            tuple(map(float, fallback_phys_xy_um)) if fallback_phys_xy_um is not None else None
        )

    multiscale = root_attrs["multiscales"][0]
    resolved_name = str(multiscale.get("name") or name)
    multiscale["name"] = resolved_name

    resolved_channel_labels = (
        [str(label) for label in channel_labels]
        if channel_labels is not None
        else _metadata_channel_labels(ngff_metadata)
    )
    if resolved_channel_labels is None:
        resolved_channel_labels = [f"ch{idx}" for idx in range(channel_count)]
    if len(resolved_channel_labels) != channel_count:
        raise ValueError(
            f"Writer received {len(resolved_channel_labels)} channel labels for {channel_count} channels."
        )

    return {
        "root_attrs": root_attrs,
        "resolved_name": resolved_name,
        "resolved_phys_xy_um": resolved_phys_xy_um,
        "resolved_channel_labels": resolved_channel_labels,
        "schema": resolved_schema,
    }
