"""
Quality Control Grid Generation

Creates contact sheets and thumbnail grids for visual quality control.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

logger = logging.getLogger(__name__)

# Try to import torch for faster grid generation
try:
    import torch
    from torchvision.utils import make_grid

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Filename pattern for parsed tissue images.
# The slide and slice fields are usually zero-padded, but some local/demo
# workflows can emit wider numeric fields (for example, slide 001).
DEFAULT_PATTERN = re.compile(
    r"^(?P<prefix>.+?)_(?P<slide>\d+)_(?P<slice_on_slide>\d+)_(?P<overall>\d+)"
    r"(?:\.(?P<ext>tif|tiff|png|jpg|jpeg))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QCRecord:
    relative_path: str
    filename: str
    source_image: str
    tile_index_on_source: int
    overall_index: int
    overall_label: str
    width: int
    height: int

    def path(self, input_dir: str | Path) -> Path:
        return Path(input_dir) / self.relative_path


@dataclass(frozen=True)
class QCArtifacts:
    master_contact_sheet: Path | None
    per_slide_grids: list[Path]
    stats_csv: Path | None
    records_manifest: Path | None


@dataclass(frozen=True)
class QCWorkflowResult:
    records: list[QCRecord]
    artifacts: QCArtifacts


def find_images(
    root: Path,
    pattern: re.Pattern | None = None,
    extensions: tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
) -> list[Path]:
    """
    Find all images matching a pattern in a directory.

    Parameters
    ----------
    root : Path
        Directory to search.
    pattern : re.Pattern, optional
        Regex pattern for filename matching.
    extensions : tuple
        Valid file extensions.

    Returns
    -------
    list of Path
        Matching image files.
    """
    root = Path(root)
    if pattern is None:
        pattern = DEFAULT_PATTERN

    images = []
    for ext in extensions:
        for p in root.glob(f"*{ext}"):
            if pattern.match(p.name):
                images.append(p)

    return sorted(set(images))


def parse_filename(
    fname: str,
    pattern: re.Pattern | None = None,
) -> dict[str, str | int] | None:
    """
    Parse tissue image filename into components.

    Parameters
    ----------
    fname : str
        Filename to parse.
    pattern : re.Pattern, optional
        Regex pattern with named groups.

    Returns
    -------
    dict or None
        Parsed components or None if no match.
    """
    if pattern is None:
        pattern = DEFAULT_PATTERN

    m = pattern.match(fname)
    if not m:
        return None

    groups = m.groupdict()
    return {
        "prefix": groups.get("prefix", ""),
        "slide": int(groups["slide"]) if "slide" in groups else 0,
        "slide_raw": groups.get("slide", ""),
        "slice_on_slide": int(groups["slice_on_slide"]) if "slice_on_slide" in groups else 0,
        "slice_on_slide_raw": groups.get("slice_on_slide", ""),
        "overall": int(groups["overall"]) if "overall" in groups else 0,
        "overall_raw": groups.get("overall", ""),
        "ext": (groups.get("ext") or "").lower(),
    }


def group_by_slide(
    paths: list[Path],
    pattern: re.Pattern | None = None,
) -> dict[int, list[tuple[int, int, Path]]]:
    """
    Group images by slide number.

    Parameters
    ----------
    paths : list of Path
        Image file paths.
    pattern : re.Pattern, optional
        Filename pattern.

    Returns
    -------
    dict
        {slide_num: [(slice_idx, overall_idx, Path), ...]}
    """
    grouped: dict[int, list[tuple[int, int, Path]]] = {}

    for p in paths:
        parsed = parse_filename(p.name, pattern)
        if parsed is None:
            continue

        slide = int(parsed["slide"])
        grouped.setdefault(slide, []).append(
            (int(parsed["slice_on_slide"]), int(parsed["overall"]), p)
        )

    # Sort within each slide
    for slide in grouped:
        grouped[slide].sort(key=lambda x: (x[0], x[1]))

    return dict(sorted(grouped.items()))


def load_thumbnail(
    path: Path,
    size: int = 256,
    mode: str = "RGB",
) -> Image.Image:
    """
    Load and resize image to thumbnail.

    Parameters
    ----------
    path : Path
        Image path.
    size : int
        Maximum size for shortest edge.
    mode : str
        PIL image mode.

    Returns
    -------
    PIL.Image
        Thumbnail image.
    """
    with Image.open(path) as img:
        if img.mode != mode:
            img = img.convert(mode)
        img.thumbnail((size, size), Image.BICUBIC)
        return img.copy()


def annotate_image(
    img: Image.Image,
    text: str,
    position: str = "top-left",
    font_size: int = 12,
) -> Image.Image:
    """
    Add text annotation to image.

    Parameters
    ----------
    img : PIL.Image
        Image to annotate.
    text : str
        Annotation text.
    position : str
        Position: 'top-left', 'top-right', 'bottom-left', 'bottom-right'.
    font_size : int
        Font size.

    Returns
    -------
    PIL.Image
        Annotated image.
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Calculate position
    w, h = img.size
    pad = 2

    if "top" in position:
        y = pad
    else:
        y = h - th - pad * 2

    if "left" in position:
        x = pad
    else:
        x = w - tw - pad * 2

    # Draw background box
    draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return img


def create_grid_pil(
    images: list[Image.Image],
    columns: int,
    padding: int = 1,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Create image grid using PIL.

    Parameters
    ----------
    images : list of PIL.Image
        Images to arrange.
    columns : int
        Number of columns.
    padding : int
        Padding between images.
    background : tuple
        Background color.

    Returns
    -------
    PIL.Image
        Grid image.
    """
    if not images:
        raise ValueError("No images provided")

    # Normalize sizes to the largest thumbnail dimensions so mixed-aspect
    # thumbnails do not get cropped down to the size of the first image.
    w = max(img.size[0] for img in images)
    h = max(img.size[1] for img in images)
    images = [ImageOps.pad(img, (w, h), color=background) for img in images]

    rows = math.ceil(len(images) / columns)
    grid_w = columns * w + (columns - 1) * padding
    grid_h = rows * h + (rows - 1) * padding

    grid = Image.new("RGB", (grid_w, grid_h), background)

    for idx, img in enumerate(images):
        r = idx // columns
        c = idx % columns
        x = c * (w + padding)
        y = r * (h + padding)
        grid.paste(img, (x, y))

    return grid


def create_grid_torch(
    images: list[Image.Image],
    columns: int,
    padding: int = 1,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Create image grid using PyTorch (faster for large grids).

    Parameters
    ----------
    images : list of PIL.Image
        Images to arrange.
    columns : int
        Number of columns.
    padding : int
        Padding between images.

    Returns
    -------
    PIL.Image
        Grid image.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")

    if not images:
        raise ValueError("No images provided")

    # Normalize sizes so aspect-ratio-preserving thumbnails can still be
    # stacked into a single tensor batch, using the largest thumbnail size
    # rather than the first image's size.
    w = max(img.size[0] for img in images)
    h = max(img.size[1] for img in images)
    images = [ImageOps.pad(img, (w, h), color=background) for img in images]

    # Convert to tensors
    tensors = []
    for img in images:
        arr = np.array(img, copy=True)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        tensors.append(t)

    batch = torch.stack(tensors)
    grid = make_grid(batch, nrow=columns, padding=padding)

    # Convert back to PIL
    grid = (grid.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(grid, mode="RGB")


def _default_manifest_path(input_dir: str | Path) -> Path:
    return Path(input_dir) / "tile_manifest.json"


def _resolve_manifest_path(input_dir: str | Path, manifest_path: str | Path | None) -> Path | None:
    if manifest_path is not None:
        chosen = Path(manifest_path)
        return chosen if chosen.exists() else None

    default_manifest = _default_manifest_path(input_dir)
    return default_manifest if default_manifest.exists() else None


def _normalize_backend(backend: str) -> str:
    if backend == "auto":
        return "torch" if TORCH_AVAILABLE else "pil"
    if backend not in {"pil", "torch"}:
        raise ValueError("backend must be one of 'pil', 'torch', or 'auto'")
    return backend


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.width, img.height


def _legacy_source_image(parsed: dict[str, str | int]) -> str:
    prefix = str(parsed["prefix"])
    slide_label = str(parsed.get("slide_raw") or parsed["slide"])
    return f"{prefix}_{slide_label}"


def _load_manifest_records(input_dir: Path, manifest_path: Path) -> list[QCRecord]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for item in payload.get("records", []):
        record = QCRecord(
            relative_path=str(item["relative_path"]),
            filename=str(item["filename"]),
            source_image=str(item["source_image"]),
            tile_index_on_source=int(item["tile_index_on_source"]),
            overall_index=int(item["overall_index"]),
            overall_label=str(item["overall_label"]),
            width=int(item["width"]),
            height=int(item["height"]),
        )
        if record.path(input_dir).exists():
            records.append(record)
        else:
            logger.warning("Skipping manifest record for missing QC image: %s", record.relative_path)
    return records


def _load_processing_metadata_records(input_dir: Path) -> list[QCRecord]:
    """
    Load records from per-slide metadata emitted by ``process_wsi``.

    Notebook 01 writes files like ``level_7_Image_00_00.tif`` plus
    ``level_7_Image_00_metadata.json`` before any global-index renaming.  These
    files do not match the legacy renamed-tile pattern, so QC needs this
    metadata fallback when review happens before artifact deletion/renaming.
    """
    records: list[QCRecord] = []
    metadata_paths = sorted(input_dir.glob("*_metadata.json"))
    for metadata_path in metadata_paths:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed processing metadata %s: %s", metadata_path, exc)
            continue

        tile_records = payload.get("tile_records") or []
        if not tile_records:
            output_paths = payload.get("output_paths") or []
            tile_records = [
                {
                    "path": path,
                    "source_image": Path(payload.get("input_path", metadata_path.stem)).name,
                    "tile_index_on_source": idx,
                }
                for idx, path in enumerate(output_paths)
            ]

        for item in tile_records:
            path = Path(str(item.get("path", "")))
            if not path.is_absolute():
                path = input_dir / path
            if not path.exists() or path.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
                continue

            width = int(item.get("width", 0) or 0)
            height = int(item.get("height", 0) or 0)
            if width <= 0 or height <= 0:
                width, height = _image_size(path)

            records.append(
                QCRecord(
                    relative_path=str(path.relative_to(input_dir)),
                    filename=path.name,
                    source_image=str(item.get("source_image") or payload.get("input_path") or metadata_path.stem),
                    tile_index_on_source=int(item.get("tile_index_on_source", 0)),
                    overall_index=0,
                    overall_label="",
                    width=width,
                    height=height,
                )
            )

    records.sort(
        key=lambda record: (
            record.source_image,
            record.tile_index_on_source,
            record.filename,
        )
    )
    for idx, record in enumerate(records, start=1):
        records[idx - 1] = QCRecord(
            relative_path=record.relative_path,
            filename=record.filename,
            source_image=record.source_image,
            tile_index_on_source=record.tile_index_on_source,
            overall_index=idx,
            overall_label=f"{idx:04d}",
            width=record.width,
            height=record.height,
        )
    return records


def _load_legacy_qc_records(
    input_dir: Path,
    pattern: re.Pattern | None = None,
) -> list[QCRecord]:
    records: list[QCRecord] = []
    for path in find_images(input_dir, pattern=pattern):
        parsed = parse_filename(path.name, pattern)
        if parsed is None:
            continue
        width, height = _image_size(path)
        records.append(
            QCRecord(
                relative_path=str(path.relative_to(input_dir)),
                filename=path.name,
                source_image=_legacy_source_image(parsed),
                tile_index_on_source=int(parsed["slice_on_slide"]),
                overall_index=int(parsed["overall"]),
                overall_label=str(parsed.get("overall_raw") or parsed["overall"]),
                width=width,
                height=height,
            )
        )
    records.sort(
        key=lambda record: (
            record.source_image,
            record.tile_index_on_source,
            record.overall_index,
            record.filename,
        )
    )
    return records


def load_qc_records(
    input_dir: str | Path,
    manifest_path: str | Path | None = None,
    pattern: re.Pattern | None = None,
) -> list[QCRecord]:
    input_dir = Path(input_dir)
    chosen_manifest = _resolve_manifest_path(input_dir, manifest_path)
    if chosen_manifest is not None:
        return _load_manifest_records(input_dir, chosen_manifest)
    metadata_records = _load_processing_metadata_records(input_dir)
    if metadata_records:
        return metadata_records
    return _load_legacy_qc_records(input_dir, pattern=pattern)


def _sorted_groups(records: list[QCRecord]) -> list[tuple[int, str, list[QCRecord]]]:
    grouped: dict[str, list[QCRecord]] = {}
    for record in records:
        grouped.setdefault(record.source_image, []).append(record)

    ordered_groups: list[tuple[int, str, list[QCRecord]]] = []
    ordered_source_images = sorted(
        grouped,
        key=lambda source_image: (
            min(record.overall_index for record in grouped[source_image]),
            source_image,
        ),
    )
    for ordinal, source_image in enumerate(ordered_source_images, start=1):
        group_records = sorted(
            grouped[source_image],
            key=lambda record: (
                record.tile_index_on_source,
                record.overall_index,
                record.filename,
            ),
        )
        ordered_groups.append((ordinal, source_image, group_records))
    return ordered_groups


def compute_qc_stats(records: list[QCRecord], input_dir: str | Path) -> pd.DataFrame:
    input_dir = Path(input_dir)
    rows: list[dict[str, str | int | float]] = []
    for record in records:
        with Image.open(record.path(input_dir)) as img:
            arr = np.asarray(img)
            rows.append(
                {
                    "relative_path": record.relative_path,
                    "filename": record.filename,
                    "source_image": record.source_image,
                    "tile_index_on_source": record.tile_index_on_source,
                    "overall_index": record.overall_index,
                    "overall_label": record.overall_label,
                    "width": img.width,
                    "height": img.height,
                    "area_px": img.width * img.height,
                    "mean_intensity": float(arr.mean()),
                    "std_intensity": float(arr.std()),
                }
            )
    return pd.DataFrame(rows)


def _calculate_columns(count: int, columns: int | str) -> int:
    if columns == "auto":
        return max(1, int(math.sqrt(count) + 0.5))
    return max(1, int(columns))


def _build_label(
    record: QCRecord,
    label_mode: str,
    *,
    group_ordinal: int | None = None,
    master: bool = False,
) -> str | None:
    if label_mode == "slice":
        return f"{record.tile_index_on_source:02d}"
    if label_mode == "overall":
        return record.overall_label
    if label_mode == "both":
        if master:
            return f"s{group_ordinal:02d}:t{record.tile_index_on_source:02d}|g{record.overall_label}"
        return f"t{record.tile_index_on_source:02d}|g{record.overall_label}"
    if label_mode == "none":
        return None
    raise ValueError("label_mode must be one of 'slice', 'overall', 'both', or 'none'")


def _create_grid(
    thumbnails: list[Image.Image],
    columns: int,
    padding: int,
    backend: str,
) -> Image.Image:
    if backend == "torch":
        return create_grid_torch(thumbnails, columns, padding)
    return create_grid_pil(thumbnails, columns, padding)


def render_qc_grids(
    records: list[QCRecord],
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    thumb_size: int = 256,
    padding: int = 1,
    columns: int | str = "auto",
    label_mode: str = "slice",
    backend: str = "pil",
    write_master: bool = True,
    write_per_slide: bool = True,
) -> QCArtifacts:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_backend = _normalize_backend(backend)

    per_slide_paths: list[Path] = []
    grouped = _sorted_groups(records)

    if write_per_slide:
        for group_ordinal, _source_image, group_records in grouped:
            thumbs = []
            for record in group_records:
                thumb = load_thumbnail(record.path(input_dir), thumb_size)
                label = _build_label(record, label_mode)
                if label:
                    thumb = annotate_image(thumb, label)
                thumbs.append(thumb)

            grid = _create_grid(
                thumbs,
                _calculate_columns(len(thumbs), columns),
                padding,
                normalized_backend,
            )
            out_path = output_dir / f"slide_{group_ordinal:02d}_grid.png"
            grid.save(out_path, dpi=(300, 300))
            per_slide_paths.append(out_path)

    master_path: Path | None = None
    if write_master and grouped:
        all_thumbs = []
        for group_ordinal, _source_image, group_records in grouped:
            for record in group_records:
                thumb = load_thumbnail(record.path(input_dir), thumb_size)
                label = _build_label(record, label_mode, group_ordinal=group_ordinal, master=True)
                if label:
                    thumb = annotate_image(thumb, label)
                all_thumbs.append(thumb)

        master = _create_grid(
            all_thumbs,
            _calculate_columns(len(all_thumbs), columns),
            padding,
            normalized_backend,
        )
        master_path = output_dir / "master_contact_sheet.png"
        master.save(master_path, dpi=(300, 300))

    return QCArtifacts(
        master_contact_sheet=master_path,
        per_slide_grids=per_slide_paths,
        stats_csv=None,
        records_manifest=None,
    )


def run_qc_workflow(
    input_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    thumb_size: int = 256,
    padding: int = 1,
    columns: int | str = "auto",
    label_mode: str = "both",
    backend: str = "pil",
    write_master: bool = True,
    write_per_slide: bool = True,
    write_stats: bool = True,
) -> QCWorkflowResult:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records_manifest = _resolve_manifest_path(input_dir, manifest_path)
    records = load_qc_records(input_dir, manifest_path=manifest_path)
    if not records:
        logger.warning("No QC inputs found in %s", input_dir)
        return QCWorkflowResult(
            records=[],
            artifacts=QCArtifacts(
                master_contact_sheet=None,
                per_slide_grids=[],
                stats_csv=None,
                records_manifest=records_manifest,
            ),
        )

    rendered = render_qc_grids(
        records,
        input_dir,
        output_dir,
        thumb_size=thumb_size,
        padding=padding,
        columns=columns,
        label_mode=label_mode,
        backend=backend,
        write_master=write_master,
        write_per_slide=write_per_slide,
    )

    stats_csv: Path | None = None
    if write_stats:
        stats_df = compute_qc_stats(records, input_dir)
        stats_csv = output_dir / "image_statistics.csv"
        stats_df.to_csv(stats_csv, index=False)

    artifacts = QCArtifacts(
        master_contact_sheet=rendered.master_contact_sheet,
        per_slide_grids=rendered.per_slide_grids,
        stats_csv=stats_csv,
        records_manifest=records_manifest,
    )
    logger.info("Created %d QC records and %d per-slide grids", len(records), len(artifacts.per_slide_grids))
    return QCWorkflowResult(records=records, artifacts=artifacts)


def build_qc_grids(
    input_dir: str | Path,
    output_dir: str | Path,
    thumb_size: int = 256,
    padding: int = 1,
    columns: int | str = "auto",
    label_mode: str = "slice",
    backend: str = "pil",
    create_master: bool = True,
) -> list[Path]:
    """
    Build QC grids for all images in a directory.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing tissue images.
    output_dir : str or Path
        Directory for output grid images.
    thumb_size : int
        Thumbnail size.
    padding : int
        Padding between thumbnails.
    columns : int or 'auto'
        Number of columns ('auto' for square-ish layout).
    label_mode : str
        Label mode: 'slice', 'overall', 'both', or 'none'.
    backend : str
        Grid backend: 'pil', 'torch', or compatibility alias 'auto'.
    create_master : bool
        Create master contact sheet with all images.

    Returns
    -------
    list of Path
        Created grid image paths.
    """
    result = run_qc_workflow(
        input_dir=input_dir,
        output_dir=output_dir,
        thumb_size=thumb_size,
        padding=padding,
        columns=columns,
        label_mode=label_mode,
        backend=backend,
        write_master=create_master,
        write_per_slide=True,
        write_stats=False,
    )
    output_paths = list(result.artifacts.per_slide_grids)
    if result.artifacts.master_contact_sheet is not None:
        output_paths.append(result.artifacts.master_contact_sheet)
    logger.info("Created %d QC grids in %s", len(output_paths), output_dir)
    return output_paths


class QCGridBuilder:
    """
    Builder class for QC grids with configurable options.

    Convenience wrapper around :func:`build_qc_grids`. Stores configuration
    and delegates to the function on each :meth:`build` call. Prefer
    :func:`build_qc_grids` for one-off calls.

    Examples
    --------
    >>> builder = QCGridBuilder(thumb_size=256, padding=2)
    >>> grids = builder.build(input_dir="tissues", output_dir="qc")
    """

    def __init__(
        self,
        thumb_size: int = 256,
        padding: int = 1,
        columns: int | str = "auto",
        label_mode: str = "slice",
        backend: str = "pil",
    ):
        self.thumb_size = thumb_size
        self.padding = padding
        self.columns = columns
        self.label_mode = label_mode
        self.backend = backend

    def build(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        create_master: bool = True,
    ) -> list[Path]:
        """Build QC grids for a directory."""
        return build_qc_grids(
            input_dir,
            output_dir,
            thumb_size=self.thumb_size,
            padding=self.padding,
            columns=self.columns,
            label_mode=self.label_mode,
            backend=self.backend,
            create_master=create_master,
        )
