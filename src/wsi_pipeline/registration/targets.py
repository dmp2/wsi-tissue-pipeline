"""Target adapters for prepared-directory and precomputed EM-LDDMM inputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .backend import EmlddmmBackend


@dataclass
class EmlddmmTarget:
    """Normalized target representation for the workflow."""

    xJ: list[np.ndarray]
    J: np.ndarray
    W0: np.ndarray
    manifest: dict[str, Any] | None
    manifest_path: Path | None
    source_format: Literal["prepared-dir", "precomputed"]
    source_path: Path
    present_mask: np.ndarray


def load_manifest(path: str | Path | None) -> dict[str, Any] | None:
    """Load and minimally validate the EM-LDDMM dataset manifest."""

    if path is None:
        return None
    manifest_path = Path(path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    required_keys = {
        "version",
        "space",
        "dv_um",
        "z_axis_um",
        "full_grid_count",
        "dense_present_count",
        "target_ext",
        "subject_dir",
        "entries",
    }
    missing = sorted(required_keys.difference(manifest))
    if missing:
        raise ValueError(
            f"Manifest {manifest_path} is missing required keys: {', '.join(missing)}"
        )
    return manifest


def resolve_target_source_format(
    source_path: str | Path,
    source_format: Literal["auto", "prepared-dir", "precomputed"] = "auto",
) -> Literal["prepared-dir", "precomputed"]:
    """Infer the target source format when the CLI leaves it as auto."""

    if source_format != "auto":
        return source_format

    path = Path(source_path)
    if (path / "samples.tsv").exists():
        return "prepared-dir"
    if (path / "info").exists():
        return "precomputed"
    raise ValueError(
        f"Could not infer target source format for {path}. "
        "Expected a prepared directory with samples.tsv or a precomputed dataset with info."
    )


def _default_manifest_path(source_path: Path) -> Path | None:
    """Return a default manifest path when one exists next to the source."""

    manifest_path = source_path / "emlddmm_dataset_manifest.json"
    return manifest_path if manifest_path.exists() else None


def _load_samples_tsv_sample_ids(subject_dir: Path) -> list[str]:
    """Load sample ids from a prepared-dir samples.tsv file."""

    samples_tsv = subject_dir / "samples.tsv"
    if not samples_tsv.exists():
        raise FileNotFoundError(f"Prepared directory is missing samples.tsv: {subject_dir}")
    with open(samples_tsv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [str(row["sample_id"]) for row in reader]


def _is_present_status(value: Any) -> bool:
    """Interpret manifest or TSV slice status values."""

    return str(value).strip().lower() in {"present", "true", "1"}


def _validate_prepared_target_manifest(
    subject_dir: Path,
    target: EmlddmmTarget,
    manifest: dict[str, Any] | None,
) -> None:
    """Apply non-fatal consistency checks for prepared-directory targets."""

    if manifest is None:
        return
    if len(target.xJ[0]) != int(manifest["full_grid_count"]):
        raise ValueError(
            "Prepared target z-grid length does not match manifest full_grid_count"
        )
    expected_present = np.array(
        [_is_present_status(entry["status"]) for entry in manifest["entries"]],
        dtype=bool,
    )
    if expected_present.shape != target.present_mask.shape:
        raise ValueError("Manifest present-mask length does not match prepared target volume")
    if np.count_nonzero(expected_present) != int(manifest["dense_present_count"]):
        raise ValueError("Manifest dense_present_count does not match manifest entries")
    if not np.array_equal(expected_present, target.present_mask):
        raise ValueError("Manifest present-mask does not match prepared target volume")
    sample_ids = _load_samples_tsv_sample_ids(subject_dir)
    manifest_sample_ids = [str(entry["sample_id"]) for entry in manifest["entries"]]
    if sample_ids != manifest_sample_ids:
        raise ValueError("Manifest sample-id ordering does not match samples.tsv")


def load_prepared_target(
    source_path: str | Path,
    *,
    backend: EmlddmmBackend,
    manifest_path: str | Path | None = None,
) -> EmlddmmTarget:
    """Load a prepared image directory into the normalized target representation."""

    subject_dir = Path(source_path)
    chosen_manifest = manifest_path or _default_manifest_path(subject_dir)
    manifest = load_manifest(chosen_manifest)

    xJ, images, _title, _names = backend.read_data(str(subject_dir))
    images_np = np.asarray(images, dtype=np.float32)
    if images_np.ndim != 4 or images_np.shape[0] < 2:
        raise ValueError(
            f"Prepared EM-LDDMM target must be channel-first with a mask channel; got {images_np.shape}"
        )

    J = np.asarray(images_np[:-1], dtype=np.float32)
    W0 = np.asarray(images_np[-1], dtype=np.float32)
    present_mask = np.any(W0 > 0, axis=(1, 2))
    target = EmlddmmTarget(
        xJ=[np.asarray(axis, dtype=np.float32) for axis in xJ],
        J=J,
        W0=W0,
        manifest=manifest,
        manifest_path=Path(chosen_manifest) if chosen_manifest is not None else None,
        source_format="prepared-dir",
        source_path=subject_dir,
        present_mask=present_mask,
    )
    _validate_prepared_target_manifest(subject_dir, target, manifest)
    return target


def _normalize_precomputed_path(path: str | Path) -> Path:
    """Normalize a local precomputed path and reject remote sources."""

    path_str = str(path)
    remote_prefixes = ("precomputed://", "http://", "https://", "gs://", "s3://")
    if path_str.startswith(remote_prefixes):
        raise ValueError("Remote precomputed inputs are not supported in v1")
    if path_str.startswith("file://"):
        path_str = path_str[7:]
    return Path(path_str).resolve()


def _load_dense_precomputed_scale0(precomputed_path: Path) -> np.ndarray:
    """Read the base-resolution precomputed dataset into memory."""

    import tensorstore as ts

    spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": {"driver": "file", "path": str(precomputed_path)},
        "scale_index": 0,
        "open": True,
    }
    dataset = ts.open(spec).result()
    dense = np.asarray(dataset[...].read().result())
    info_path = precomputed_path / "info"
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    num_channels = int(info["num_channels"])

    if dense.ndim == 4:
        if dense.shape[-1] != num_channels:
            raise ValueError(
                f"Unexpected precomputed channel axis: shape {dense.shape}, num_channels={num_channels}"
            )
        return dense.transpose(3, 2, 1, 0).astype(np.float32)
    if dense.ndim == 3 and num_channels == 1:
        return dense.transpose(2, 1, 0)[None].astype(np.float32)
    raise ValueError(f"Unsupported precomputed dense shape: {dense.shape}")


def load_precomputed_target(
    source_path: str | Path,
    *,
    manifest_path: str | Path,
) -> EmlddmmTarget:
    """Load a manifest-backed precomputed plate into the normalized target representation."""

    precomputed_path = _normalize_precomputed_path(source_path)
    manifest = load_manifest(manifest_path)
    if manifest is None:
        raise ValueError("Precomputed targets require an EM-LDDMM dataset manifest")

    dense_present = _load_dense_precomputed_scale0(precomputed_path)
    dense_z = dense_present.shape[1]
    expected_dense = int(manifest["dense_present_count"])
    if dense_z != expected_dense:
        raise ValueError(
            "Precomputed z-depth does not match manifest dense_present_count "
            f"({dense_z} != {expected_dense})"
        )

    full_grid_count = int(manifest["full_grid_count"])
    entries = list(manifest["entries"])
    if len(entries) != full_grid_count:
        raise ValueError("Manifest entries length does not match manifest full_grid_count")

    present_entries = sorted(
        (entry for entry in entries if _is_present_status(entry["status"])),
        key=lambda entry: int(entry["present_rank"]),
    )
    if len(present_entries) != expected_dense:
        raise ValueError("Manifest present entries do not match dense_present_count")

    for entry in present_entries:
        if entry.get("shape_yx") is None:
            raise ValueError("Precomputed manifest entries for present slices require shape_yx")
        if entry.get("space_directions_um") is None:
            raise ValueError(
                "Precomputed manifest entries for present slices require space_directions_um"
            )

    max_y = max(int(entry["shape_yx"][0]) for entry in present_entries)
    max_x = max(int(entry["shape_yx"][1]) for entry in present_entries)
    channels = dense_present.shape[0]
    J = np.zeros((channels, full_grid_count, max_y, max_x), dtype=np.float32)
    W0 = np.zeros((full_grid_count, max_y, max_x), dtype=np.float32)

    for entry in present_entries:
        dense_idx = int(entry["present_rank"])
        grid_idx = int(entry["grid_index"])
        y, x = int(entry["shape_yx"][0]), int(entry["shape_yx"][1])
        dense_slice = dense_present[:, dense_idx]
        J[:, grid_idx, :y, :x] = dense_slice[:, :y, :x]
        W0[grid_idx, :y, :x] = np.any(dense_slice[:, :y, :x] > 0, axis=0).astype(np.float32)

    z_axis = np.asarray(manifest["z_axis_um"], dtype=np.float32)
    if len(z_axis) != full_grid_count:
        raise ValueError("Manifest z_axis_um length does not match full_grid_count")

    first_present = present_entries[0]
    directions = np.asarray(first_present["space_directions_um"], dtype=np.float32)
    dy = float(directions[1][1])
    dx = float(directions[0][0])
    y_axis = np.arange(max_y, dtype=np.float32) * dy
    y_axis -= np.mean(y_axis)
    x_axis = np.arange(max_x, dtype=np.float32) * dx
    x_axis -= np.mean(x_axis)
    present_mask = np.array(
        [_is_present_status(entry["status"]) for entry in entries],
        dtype=bool,
    )

    return EmlddmmTarget(
        xJ=[z_axis, y_axis, x_axis],
        J=J,
        W0=W0,
        manifest=manifest,
        manifest_path=Path(manifest_path),
        source_format="precomputed",
        source_path=precomputed_path,
        present_mask=present_mask,
    )
