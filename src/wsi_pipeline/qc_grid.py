"""
Quality Control Grid Generation

Creates contact sheets and thumbnail grids for visual quality control.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

logger = logging.getLogger(__name__)

# Try to import torch for faster grid generation
try:
    import torch
    from torchvision.utils import make_grid
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Filename pattern for parsed tissue images
# Pattern: {PREFIX}_{SLIDE:02d}_{SLICE:02d}_{OVERALL:04d}.{ext}
DEFAULT_PATTERN = re.compile(
    r"^(?P<prefix>.+?)_(?P<slide>\d{2})_(?P<slice_on_slide>\d{2})_(?P<overall>\d{4})"
    r"(?:\.(?P<ext>tif|tiff|png|jpg|jpeg))?$",
    re.IGNORECASE,
)


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
        "slice_on_slide": int(groups["slice_on_slide"]) if "slice_on_slide" in groups else 0,
        "overall": int(groups["overall"]) if "overall" in groups else 0,
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

        slide = parsed["slide"]
        grouped.setdefault(slide, []).append(
            (parsed["slice_on_slide"], parsed["overall"], p)
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
    img = Image.open(path)
    if img.mode != mode:
        img = img.convert(mode)
    img.thumbnail((size, size), Image.BICUBIC)
    return img


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

    # Normalize sizes
    w, h = images[0].size
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


def build_qc_grids(
    input_dir: str | Path,
    output_dir: str | Path,
    thumb_size: int = 256,
    padding: int = 1,
    columns: int | str = "auto",
    label_mode: str = "slice",
    backend: str = "auto",
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
        Grid backend: 'auto', 'torch', or 'pil'.
    create_master : bool
        Create master contact sheet with all images.

    Returns
    -------
    list of Path
        Created grid image paths.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select backend
    if backend == "auto":
        backend = "torch" if TORCH_AVAILABLE else "pil"

    # Find and group images
    images = find_images(input_dir)
    if not images:
        logger.warning("No matching images found in %s", input_dir)
        return []

    grouped = group_by_slide(images)

    output_paths = []

    # Build per-slide grids
    for slide, items in grouped.items():
        thumbs = []

        for slice_idx, overall_idx, path in items:
            thumb = load_thumbnail(path, thumb_size)

            # Add label
            if label_mode == "slice":
                text = f"{slice_idx:02d}"
            elif label_mode == "overall":
                text = f"{overall_idx:04d}"
            elif label_mode == "both":
                text = f"s{slice_idx:02d}|g{overall_idx:04d}"
            else:
                text = None

            if text:
                thumb = annotate_image(thumb, text)

            thumbs.append(thumb)

        # Calculate columns
        if columns == "auto":
            n_cols = max(1, int(math.sqrt(len(thumbs)) + 0.5))
        else:
            n_cols = int(columns)

        # Create grid
        if backend == "torch" and TORCH_AVAILABLE:
            grid = create_grid_torch(thumbs, n_cols, padding)
        else:
            grid = create_grid_pil(thumbs, n_cols, padding)

        # Save
        out_path = output_dir / f"slide_{slide:02d}_grid.png"
        grid.save(out_path, dpi=(300, 300))
        output_paths.append(out_path)

    # Build master sheet
    if create_master and grouped:
        all_thumbs = []

        for slide, items in grouped.items():
            for slice_idx, overall_idx, path in items:
                thumb = load_thumbnail(path, thumb_size)

                if label_mode == "slice":
                    text = f"s{slide:02d}/{slice_idx:02d}"
                elif label_mode == "overall":
                    text = f"{overall_idx:04d}"
                elif label_mode == "both":
                    text = f"s{slide:02d}:{slice_idx:02d}|{overall_idx:04d}"
                else:
                    text = None

                if text:
                    thumb = annotate_image(thumb, text)

                all_thumbs.append(thumb)

        if columns == "auto":
            n_cols = max(1, int(math.sqrt(len(all_thumbs)) + 0.5))
        else:
            n_cols = int(columns)

        if backend == "torch" and TORCH_AVAILABLE:
            master = create_grid_torch(all_thumbs, n_cols, padding)
        else:
            master = create_grid_pil(all_thumbs, n_cols, padding)

        master_path = output_dir / "master_contact_sheet.png"
        master.save(master_path, dpi=(300, 300))
        output_paths.append(master_path)

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
        backend: str = "auto",
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
