"""
EM-LDDMM Histology Preparation
===============================

Utilities for preparing extracted tissue image stacks for the EM-LDDMM
(Expectation-Maximization Large Deformation Diffeomorphic Metric Mapping)
registration pipeline.

Provides:
- ``make_samples_tsv`` -- write a ``samples.tsv`` inventory with missing-slice detection
- ``write_emlddmm_dataset_manifest`` -- write a manifest for prepared-dir and precomputed step-5 inputs
- ``set_up_hist_for_emlddmm`` -- orchestrate downsampling, TSV generation, and JSON sidecars
- ``remove_json_sidecars`` -- delete all ``.json`` files in a directory (for re-generation)

The ``histsetup`` module from the ``emlddmm`` package is an optional dependency.
Functions that require it will raise a clear error if it is not installed.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_img_ext(fnames: list[str]) -> str:
    """Auto-detect the most common image extension from a list of filenames.

    Parameters
    ----------
    fnames : list[str]
        Filenames (basenames or full paths) to scan.

    Returns
    -------
    str
        Extension with leading dot, e.g. ``'.tif'``.

    Raises
    ------
    FileNotFoundError
        If no known image extension is found.
    """
    known = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    counts: dict[str, int] = {}
    for n in fnames:
        sfx = Path(n).suffix.lower()
        if sfx in known:
            counts[sfx] = counts.get(sfx, 0) + 1
    if not counts:
        raise FileNotFoundError(
            "Could not detect an image extension in the provided filenames."
        )
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _extract_frame_number(
    name: str, sep: str = "_", fnumidx: int = -1
) -> int:
    """Extract trailing integer from a filename token.

    Splits *name* (stem only, extension stripped) by *sep* and parses the
    trailing digits of the token at index *fnumidx*.

    Parameters
    ----------
    name : str
        Filename (may include extension).
    sep : str
        Token separator.
    fnumidx : int
        Index of the token containing the frame number.

    Returns
    -------
    int
        Parsed frame number.
    """
    stem = Path(name).stem
    tokens = stem.split(sep)
    if not tokens:
        raise ValueError(f"Cannot split {name!r} by separator {sep!r}")
    try:
        token = tokens[fnumidx]
    except IndexError:
        raise ValueError(
            f"fnumidx={fnumidx} out of range for {name!r} split by {sep!r}"
        )
    m = re.search(r"(\d+)$", token)
    if not m:
        raise ValueError(
            f"No trailing digits in token {token!r} from {name!r}"
        )
    return int(m.group(1))


def _format_missing_like(
    reference_name: str,
    new_num: int,
    sep: str,
    fnumidx: int,
    ext_with_dot: str,
) -> str:
    """Generate a placeholder filename for a missing slice.

    The new name mimics the token pattern of *reference_name*, replacing only
    the numeric portion of the token at *fnumidx*.

    Parameters
    ----------
    reference_name : str
        An existing filename to use as a formatting template.
    new_num : int
        Slice number to embed.
    sep : str
        Token separator.
    fnumidx : int
        Token index containing the number.
    ext_with_dot : str
        File extension including the leading dot.

    Returns
    -------
    str
        Formatted filename.
    """
    stem = Path(reference_name).stem
    tokens = stem.split(sep)
    token = tokens[fnumidx]

    m = re.search(r"^(.*?)(\d+)?$", token)
    prefix = m.group(1)
    ref_digits = m.group(2) or ""
    width = len(ref_digits) if ref_digits else 4

    tokens[fnumidx] = f"{prefix}{new_num:0{width}d}"
    return sep.join(tokens) + ext_with_dot


def _load_samples_rows(samples_tsv: str | Path) -> list[dict[str, str]]:
    """Load rows from a samples.tsv file."""

    with open(samples_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def _load_sidecar(path: Path) -> dict[str, Any] | None:
    """Load a JSON sidecar when it exists."""

    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sidecar_shape_yx(sidecar: dict[str, Any] | None) -> list[int] | None:
    """Extract [Y, X] image shape from a sidecar payload."""

    if sidecar is None:
        return None
    sizes = sidecar.get("Sizes")
    if not isinstance(sizes, list) or len(sizes) < 3:
        return None
    return [int(sizes[2]), int(sizes[1])]


def _sidecar_space_directions(sidecar: dict[str, Any] | None) -> list[list[float]] | None:
    """Extract the 3x3 space-directions matrix from a sidecar payload."""

    if sidecar is None:
        return None
    directions = sidecar.get("SpaceDirections")
    if not isinstance(directions, list) or len(directions) < 4:
        return None
    return [[float(value) for value in axis] for axis in directions[1:4]]


def _sidecar_origin(sidecar: dict[str, Any] | None) -> list[float] | None:
    """Extract the space origin from a sidecar payload."""

    if sidecar is None:
        return None
    origin = sidecar.get("SpaceOrigin")
    if not isinstance(origin, list) or len(origin) < 3:
        return None
    return [float(value) for value in origin[:3]]


def _infer_manifest_z_step(
    present_rows: list[dict[str, Any]],
    default_step: float,
) -> float:
    """Infer the per-grid-step z spacing from present sidecars."""

    if len(present_rows) < 2:
        return float(default_step)

    candidates: list[float] = []
    previous = None
    for row in present_rows:
        if row["z_position_um"] is None:
            continue
        if previous is not None:
            delta_index = int(row["overall_index"]) - int(previous["overall_index"])
            if delta_index > 0:
                delta_z = float(row["z_position_um"]) - float(previous["z_position_um"])
                candidates.append(delta_z / delta_index)
        previous = row

    if not candidates:
        return float(default_step)
    return float(np.median(np.asarray(candidates, dtype=np.float64)))


def write_emlddmm_dataset_manifest(
    subject_dir: str | Path,
    *,
    ext: str = "",
    dv_um: list[float] | tuple[float, float, float] | None = None,
    space: str | None = None,
    sep: str = "_",
    fnumidx: int = -1,
) -> Path:
    """Write ``emlddmm_dataset_manifest.json`` for a prepared EM-LDDMM dataset."""

    subject_dir = Path(subject_dir)
    samples_tsv = subject_dir / "samples.tsv"
    if not samples_tsv.exists():
        raise FileNotFoundError(f"Could not find samples.tsv in {subject_dir}")

    rows = _load_samples_rows(samples_tsv)
    if not rows:
        raise ValueError(f"samples.tsv in {subject_dir} does not contain any slice rows")

    ext_with_dot = ext if ext else _detect_img_ext([row["sample_id"] for row in rows])
    if not ext_with_dot.startswith("."):
        ext_with_dot = f".{ext_with_dot}"

    manifest_rows: list[dict[str, Any]] = []
    present_rows: list[dict[str, Any]] = []
    overall_indices: list[int] = []

    for row in rows:
        sample_id = row["sample_id"]
        overall_index = _extract_frame_number(sample_id, sep=sep, fnumidx=fnumidx)
        overall_indices.append(overall_index)

        sidecar_name = f"{Path(sample_id).stem}.json"
        sidecar = _load_sidecar(subject_dir / sidecar_name)
        entry_space = space
        if entry_space is None and sidecar is not None:
            entry_space = str(sidecar.get("Space", ""))

        entry = {
            "sample_id": sample_id,
            "status": row["status"],
            "grid_index": 0,
            "present_rank": None,
            "overall_index": overall_index,
            "image_filename": sample_id,
            "sidecar_filename": sidecar_name if sidecar is not None else None,
            "z_position_um": None,
            "shape_yx": _sidecar_shape_yx(sidecar),
            "space_origin_um": _sidecar_origin(sidecar),
            "space_directions_um": _sidecar_space_directions(sidecar),
        }
        if entry["space_origin_um"] is not None:
            entry["z_position_um"] = float(entry["space_origin_um"][2])
        if str(row["status"]).strip().lower() == "present":
            present_rows.append(entry)
        if space is None and entry_space:
            space = entry_space
        manifest_rows.append(entry)

    min_overall = min(overall_indices)
    max_overall = max(overall_indices)
    full_grid_count = int(max_overall - min_overall + 1)

    for entry in manifest_rows:
        entry["grid_index"] = int(entry["overall_index"] - min_overall)

    present_rows.sort(key=lambda item: int(item["overall_index"]))
    for present_rank, entry in enumerate(present_rows):
        entry["present_rank"] = int(present_rank)

    default_dv = [0.27385655, 0.27385655, 10.0] if dv_um is None else [float(v) for v in dv_um]
    first_present = present_rows[0] if present_rows else None
    directions = first_present["space_directions_um"] if first_present is not None else None
    if directions is not None:
        default_dv = [
            float(directions[0][0]),
            float(directions[1][1]),
            float(directions[2][2]),
        ]

    z_step_um = _infer_manifest_z_step(present_rows, default_dv[2])
    if first_present is not None and first_present["z_position_um"] is not None:
        z0 = float(first_present["z_position_um"]) - (
            int(first_present["grid_index"]) * float(z_step_um)
        )
    else:
        z0 = 0.0
    z_axis_um = [float(z0 + (idx * z_step_um)) for idx in range(full_grid_count)]

    for entry in manifest_rows:
        if entry["z_position_um"] is None:
            entry["z_position_um"] = float(z_axis_um[int(entry["grid_index"])])

    manifest = {
        "version": 1,
        "space": space or "unknown",
        "dv_um": [float(value) for value in default_dv],
        "z_axis_um": z_axis_um,
        "full_grid_count": full_grid_count,
        "dense_present_count": len(present_rows),
        "target_ext": ext_with_dot,
        "subject_dir": str(subject_dir.resolve()),
        "entries": manifest_rows,
    }

    out_path = subject_dir / "emlddmm_dataset_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_samples_tsv(
    subject_dir: str | Path,
    species_name: str = "Homo Sapiens",
    ext: str = "",
    slice_downfactor: int = 1,
    max_slice: Optional[int] = None,
    sep: str = "_",
    fnumidx: int = -1,
) -> Path:
    """Write a ``samples.tsv`` file listing images with missing-slice detection.

    Scans *subject_dir* for image files, detects gaps in the slice numbering
    sequence, and writes a TSV with columns
    ``sample_id | participant_id | species | status``.

    Parameters
    ----------
    subject_dir : str or Path
        Directory containing the tissue images.
    species_name : str
        Species label written into the TSV.
    ext : str
        Image extension to filter (e.g. ``'.tif'``).  Auto-detected if empty.
    slice_downfactor : int
        Step size between expected slice indices.
    max_slice : int, optional
        Highest expected slice number.  Inferred from data if *None*.
    sep : str
        Filename token separator.
    fnumidx : int
        Token index that contains the slice number.

    Returns
    -------
    Path
        Path to the written ``samples.tsv``.
    """
    subject_dir = Path(subject_dir)
    if not subject_dir.is_dir():
        raise FileNotFoundError(
            f"Subject directory does not exist: {subject_dir}"
        )

    fnames = os.listdir(subject_dir)
    if not fnames:
        raise FileNotFoundError("Subject directory is empty.")

    # Normalize / auto-detect extension
    if ext == "":
        ext = _detect_img_ext(fnames)
    else:
        ext = ext if ext.startswith(".") else f".{ext}"

    fnames = [f for f in fnames if Path(f).suffix.lower() == ext.lower()]
    if not fnames:
        raise FileNotFoundError(
            f"No files with extension {ext} found in {subject_dir}"
        )

    # Sort by parsed slice number
    fnames = sorted(
        fnames,
        key=lambda x: _extract_frame_number(x, sep=sep, fnumidx=fnumidx),
    )

    img_nums = [
        _extract_frame_number(x, sep=sep, fnumidx=fnumidx) for x in fnames
    ]

    if max_slice is None:
        max_slice = img_nums[-1] if img_nums else 0

    slice_range = np.arange(1, max_slice + 1, slice_downfactor, dtype=int)

    # Missing = symmetric difference of expected vs present
    missing_imgs = sorted(
        list(set(img_nums) - set(slice_range))
        + list(set(slice_range) - set(img_nums))
    )

    logger.debug("Missing images: %s", missing_imgs)

    # Insert placeholder entries for missing slices
    missing_ids: list[int] = []
    if missing_imgs:
        ref_fname = fnames[0]
        N = len(img_nums)
        for miss in missing_imgs:
            fname = _format_missing_like(
                ref_fname, miss, sep=sep, fnumidx=fnumidx, ext_with_dot=ext
            )
            inserted = False
            for j in range(N):
                if img_nums[j] > miss:
                    fnames.insert(j, fname)
                    img_nums.insert(j, miss)
                    missing_ids.append(j)
                    N += 1
                    inserted = True
                    break
            if not inserted:
                fnames.append(fname)
                img_nums.append(miss)
                missing_ids.append(N)
                N += 1

    # Write TSV
    out_path = subject_dir / "samples.tsv"
    with open(out_path, "w") as f:
        f.write("sample_id\tparticipant_id\tspecies\tstatus\n")
        for i, fname in enumerate(fnames):
            status = "missing" if i in missing_ids else "present"
            f.write(f"{fname}\t{fname[:5]}\t{species_name}\t{status}\n")

    return out_path


def remove_json_sidecars(
    dir_path: str | Path,
    *,
    dry_run: bool = False,
    verbose: bool = True,
) -> List[Path]:
    """Delete all ``.json`` files in a directory (non-recursive).

    Useful for cleaning up before regenerating JSON sidecar metadata.

    Parameters
    ----------
    dir_path : str or Path
        Directory containing sidecars.
    dry_run : bool
        If *True*, only report what would be deleted.
    verbose : bool
        Print progress.

    Returns
    -------
    list[Path]
        Paths that were deleted (or would be, if *dry_run*).
    """
    p = Path(dir_path).expanduser().resolve()
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {p}")

    targets = [
        e for e in p.iterdir() if e.is_file() and e.suffix.lower() == ".json"
    ]
    deleted: list[Path] = []

    for f in targets:
        if dry_run:
            logger.info("[DRY RUN] Would delete: %s", f)
            deleted.append(f)
            continue
        try:
            f.unlink()
            deleted.append(f)
            logger.info("Deleted: %s", f.name)
        except OSError as e:
            logger.error("Error deleting '%s': %s", f, e)

    msg = "would be removed" if dry_run else "removed"
    logger.info("Done. %d .json file(s) %s in %s", len(deleted), msg, p)

    return deleted


# Backward compatibility
nuke_json_sidecars = remove_json_sidecars


def set_up_hist_for_emlddmm(config: dict) -> None:
    """Orchestrate histology stack preparation for EM-LDDMM.

    Performs three steps in sequence:

    1. **Downsample** slices (if ``slice_down > 1`` or ``res_down > 1``)
    2. **Generate** ``samples.tsv`` via :func:`make_samples_tsv`
    3. **Write** per-image JSON sidecars with voxel sizes via ``histsetup.generate_sidecars``

    Parameters
    ----------
    config : dict
        Keys:

        - ``subject_dir`` -- source image directory
        - ``output_dir`` -- destination (may equal subject_dir)
        - ``species_name`` -- species label for the TSV (default ``'Homo Sapiens'``)
        - ``ext`` -- image extension (auto-detected if empty)
        - ``slice_down`` -- take every *n*-th slice (default 1)
        - ``res_down`` -- spatial downsampling factor (default 1)
        - ``max_slice`` -- highest expected slice number (*None* to infer)
        - ``dv`` -- ``[dx, dy, dz]`` voxel size in micrometers
        - ``space`` -- anatomical coordinate space string

    Raises
    ------
    ImportError
        If the ``histsetup`` module (from the ``emlddmm`` package) is not installed.
    """
    try:
        import histsetup as hs
    except ImportError:
        raise ImportError(
            "The 'histsetup' module from the 'emlddmm' package is required "
            "for set_up_hist_for_emlddmm(). Install it with:\n"
            "  pip install emlddmm\n"
            "or add it to sys.path manually."
        )

    subject_dir = Path(config.get("subject_dir", "."))
    output_dir = Path(config.get("output_dir", "."))
    species_name = config.get("species_name", "Homo Sapiens")
    ext = config.get("ext", "")
    slice_down = int(config.get("slice_down", 1))
    res_down = int(config.get("res_down", 1))
    max_slice = config.get("max_slice", None)
    dv = list(config.get("dv", [0.27385655, 0.27385655, 10.0]))
    space = config.get("space", "right-inferior-posterior")

    # Scale voxel sizes by actual downsampling
    dv = [dv[0] * res_down, dv[1] * res_down, dv[2] * max(1, slice_down)]

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "subject directory: %s | output directory: %s | slice_down: %s | "
        "res_down: %s | max_slice: %s | dv (microns): %s | space: %s",
        subject_dir, output_dir, slice_down, res_down, max_slice, dv, space,
    )

    # 1) Optional downsampling / copy
    if slice_down > 1 or res_down > 1:
        hs.downsample_slices(
            str(subject_dir),
            str(output_dir),
            ext,
            slice_downfactor=slice_down,
            image_downfactor=res_down,
        )

    # 2) Generate samples.tsv
    make_samples_tsv(output_dir, species_name=species_name, ext=ext)

    # 3) JSON sidecars with voxel sizes and coordinate space
    hs.generate_sidecars(
        str(output_dir),
        ext=ext,
        max_slice=max_slice,
        dv=dv,
        slice_downfactor=slice_down,
        space=space,
    )

    # 4) Manifest for prepared-dir and precomputed-backed step-5 inputs
    write_emlddmm_dataset_manifest(
        output_dir,
        ext=ext,
        dv_um=dv,
        space=space,
    )

    logger.info("Finished preparing histology stack.")
