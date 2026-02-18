"""
VSI/ETS File Conversion Utilities

Provides functions for finding ETS files within VSI directory structures
and converting them to flat file formats.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import List, Optional, Union

import cv2
import numpy as np

from .etsfile import ETSFile


def find_ets_file(vsi_fname: Union[str, Path]) -> Optional[Path]:
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
    ├── specimen.vsi
    └── _specimen_/
        ├── stack10001/
        │   └── frame_t.ets  (thumbnail)
        └── stack10002/
            └── frame_t.ets  (full resolution)
    
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


def find_all_ets_files(vsi_fname: Union[str, Path]) -> List[Path]:
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


def vsi_to_flat_image(
    vsi_fname: Union[str, Path],
    level: int = 0,
    output_path: Optional[Union[str, Path]] = None,
    format: str = "jpg",
    jpeg_quality: int = 95,
) -> Optional[np.ndarray]:
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
    ets_fname: Union[str, Path],
    level: int = 0,
    output_path: Optional[Union[str, Path]] = None,
    format: str = "jpg",
    jpeg_quality: int = 95,
) -> Optional[np.ndarray]:
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


def batch_convert_vsi(
    input_pattern: str,
    output_dir: Union[str, Path],
    level: int = 0,
    format: str = "jpg",
    jpeg_quality: int = 95,
    verbose: bool = True,
) -> List[Path]:
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

    output_files = []
    for vsi_path in vsi_files:
        vsi_path = Path(vsi_path)
        output_name = f"level_{level}_{vsi_path.stem}.{format}"
        output_path = output_dir / output_name

        logger.debug("  %s -> %s", vsi_path.name, output_name)

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


def get_vsi_metadata(vsi_fname: Union[str, Path]) -> dict:
    """
    Extract metadata from a VSI file.
    
    Parameters
    ----------
    vsi_fname : str or Path
        Path to the VSI file.
    
    Returns
    -------
    dict
        Metadata dictionary with image properties.
    """
    ets_path = find_ets_file(vsi_fname)
    if ets_path is None:
        return {}

    with ETSFile(ets_path) as ets:
        return {
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
        }
