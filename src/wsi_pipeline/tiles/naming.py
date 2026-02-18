"""
File naming utilities for WSI tile processing.

Provides functions for parsing and generating filenames for tissue tiles,
including support for coordinate-based naming conventions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple


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



def rename_outputs_by_overall_index(
    output_dir: Path | str,
    *,
    pattern: str,          # input pattern, used to derive both regex & output glob
    spacing: int = 9,      # skip 9 -> step 10
    pad: int = 4,
    start: int = 1,        # 0001, 0011, 0021, ...
    dry_run: bool = False,
) -> List[Tuple[Path, Path]]:
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
    renames: List[Tuple[Path, Path]] = []
    for rank, p in enumerate(files):
        label = overall_label(rank, spacing=spacing, pad=pad, start=start)
        new_p = add_overall_suffix(p, label, pad)
        if new_p.name != p.name and not dry_run:
            try:
                p.rename(new_p)
            except FileNotFoundError:
                new_p = p
        renames.append((p, new_p))

    return renames
