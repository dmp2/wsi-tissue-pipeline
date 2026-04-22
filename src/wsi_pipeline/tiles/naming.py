"""
File naming utilities for WSI tile processing.

Provides functions for parsing and generating filenames for tissue tiles,
including support for coordinate-based naming conventions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


def normalize_ext(ext: str) -> str:
    ext = ext.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        return "jpg"
    if ext in ("tif", "tiff"):
        return "tiff"
    if ext == "png":
        return "png"
    raise ValueError(f"Unsupported output_ext: {ext!r}")


def build_xy_regex_from_pattern(pattern: str) -> re.Pattern:
    """
    Build a regex that captures (XX, YY) from filenames based on the input glob `pattern`.

    Strategy:
      - Take ONLY the basename portion of `pattern` (ignore directories).
      - Use the static prefix before the first '*' as an anchor (may be empty).
      - Then look for '_<digits>_<digits>' anywhere after that prefix.
    """
    basename = Path(pattern).name            # e.g., "E241_right_level_7_Image_*.jpg"
    prefix = re.escape(basename.split('*')[0])  # static head before first '*'
    # Anchor: prefix, then anything, then "_<digits>_<digits>"
    return re.compile(rf"{prefix}.*?_([0-9]+)_([0-9]+)", re.IGNORECASE)

def build_output_glob_from_pattern(pattern: str) -> str:
    """
    Build a glob for outputs from the input `pattern`, assuming outputs include '_XX_YY'
    and may have a different extension (e.g., .tif instead of .jpg).

    We reuse the static prefix before the first '*' and then look for '*_*.*'.
    """
    basename = Path(pattern).name            # e.g., "E241_right_level_7_Image_*.jpg"
    prefix = basename.split('*')[0]          # e.g., "E241_right_level_7_Image_"
    # Outputs like: "<prefix>*_*.<any ext>"
    return f"{prefix}*_*.*"

def parse_xx_yy_from_name(name: str, rx_xy: re.Pattern) -> tuple[int, int]:
    m = rx_xy.search(name)
    if not m:
        return (10**9, 10**9)  # push non-matching names to the end deterministically
    return (int(m.group(1)), int(m.group(2)))

def overall_label(rank_zero_based: int, *, spacing: int, pad: int, start: int) -> str:
    """Arithmetic progression per global file rank:
       label(rank) = start + rank * (spacing + 1)"""
    step = spacing + 1
    n = start + rank_zero_based * step
    return f"{n:0{pad}d}"

def add_overall_suffix(p: Path, label: str, pad: int) -> Path:
    # Idempotent: strip existing trailing _dddd (with width `pad`)
    stem = re.sub(rf"(_\d{{{pad}}})$", "", p.stem)
    return p.with_name(f"{stem}_{label}{p.suffix}")


def _strip_overall_suffix(name: str, pad: int) -> str:
    path = Path(name)
    stem = re.sub(rf"(_\d{{{pad}}})$", "", path.stem)
    return f"{stem}{path.suffix}"


def _infer_source_image_and_tile_index(path: Path, pad: int) -> tuple[str, int]:
    stem = re.sub(rf"(_\d{{{pad}}})$", "", path.stem)
    match = re.match(r"^(?P<source>.+)_(?P<tile>\d+)$", stem)
    if match:
        return match.group("source"), int(match.group("tile"))
    return stem, 0


def _load_tile_metadata(output_dir: Path, pad: int) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for metadata_path in sorted(output_dir.glob("*_metadata.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_image = Path(payload.get("input_path", "")).name or metadata_path.name.replace("_metadata.json", "")

        tile_records = payload.get("tile_records") or []
        if tile_records:
            for record in tile_records:
                path = Path(record["path"])
                entry = {
                    "source_image": record.get("source_image", source_image),
                    "tile_index_on_source": int(record.get("tile_index_on_source", 0)),
                    "width": int(record.get("width", 0)),
                    "height": int(record.get("height", 0)),
                }
                lookup[path.name] = entry
                lookup[_strip_overall_suffix(path.name, pad)] = entry
            continue

        for idx, path_str in enumerate(payload.get("output_paths", [])):
            path = Path(path_str)
            entry = {
                "source_image": source_image,
                "tile_index_on_source": idx,
                "width": 0,
                "height": 0,
            }
            lookup[path.name] = entry
            lookup[_strip_overall_suffix(path.name, pad)] = entry

    return lookup


def _build_manifest_record(
    output_dir: Path,
    path: Path,
    overall_label_text: str,
    pad: int,
    metadata_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metadata = (
        metadata_lookup.get(path.name)
        or metadata_lookup.get(_strip_overall_suffix(path.name, pad))
        or {}
    )
    source_image, tile_index_on_source = _infer_source_image_and_tile_index(path, pad)
    source_image = str(metadata.get("source_image", source_image))
    tile_index_on_source = int(metadata.get("tile_index_on_source", tile_index_on_source))

    width = int(metadata.get("width", 0))
    height = int(metadata.get("height", 0))
    if width <= 0 or height <= 0:
        with Image.open(path) as img:
            width = img.width
            height = img.height

    return {
        "relative_path": str(path.relative_to(output_dir)),
        "filename": path.name,
        "source_image": source_image,
        "tile_index_on_source": tile_index_on_source,
        "overall_index": int(overall_label_text),
        "overall_label": overall_label_text,
        "width": width,
        "height": height,
    }


def _write_tile_manifest(
    output_dir: Path,
    manifest_records: list[dict[str, Any]],
) -> Path:
    manifest_path = output_dir / "tile_manifest.json"
    payload = {
        "version": 1,
        "input_dir": str(output_dir.resolve()),
        "generated_by": "wsi_pipeline.tiles.naming.rename_outputs_by_overall_index",
        "records": manifest_records,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path



def rename_outputs_by_overall_index(
    output_dir: Path | str,
    *,
    pattern: str,          # input pattern, used to derive both regex & output glob
    spacing: int = 9,      # skip 9 -> step 10
    pad: int = 4,
    start: int = 1,        # 0001, 0011, 0021, ...
    dry_run: bool = False,
) -> list[tuple[Path, Path]]:
    """
    One global pass: rank all matching files by (XX,YY) and rename to append `_ZZZZ`,
    where ZZZZ = start + rank * (spacing+1), zero-padded to `pad`. Idempotent.

    `pattern` is the same input glob you used for `process_directory`; we derive:
      - rx_xy to parse (XX,YY)
      - output glob to select files in `output_dir` (handles different extensions)
    """
    output_dir = Path(output_dir)
    rx_xy = build_xy_regex_from_pattern(pattern)
    output_glob = build_output_glob_from_pattern(pattern)

    permissible_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    all_paths = output_dir.glob(output_glob)
    files = sorted((
        p for p in all_paths if p.suffix.lower() in permissible_exts),
        key=lambda p: parse_xx_yy_from_name(p.name, rx_xy)
    )
    metadata_lookup = _load_tile_metadata(output_dir, pad)
    renames: list[tuple[Path, Path]] = []
    manifest_records: list[dict[str, Any]] = []
    for rank, p in enumerate(files):
        label = overall_label(rank, spacing=spacing, pad=pad, start=start)
        new_p = add_overall_suffix(p, label, pad)
        if new_p.name != p.name and not dry_run:
            try:
                p.rename(new_p)
            except FileNotFoundError:
                new_p = p
        renames.append((p, new_p))

        if dry_run:
            continue

        manifest_target = new_p if new_p.exists() else p
        if manifest_target.exists():
            manifest_records.append(
                _build_manifest_record(
                    output_dir,
                    manifest_target,
                    label,
                    pad,
                    metadata_lookup,
                )
            )

    if not dry_run:
        _write_tile_manifest(output_dir, manifest_records)

    return renames
