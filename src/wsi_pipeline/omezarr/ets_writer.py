"""Direct ETS-to-OME-Zarr source pyramid writer."""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numcodecs import Blosc

from ..etsfile import ETSFile
from .metadata import _prepare_ngff_writer_metadata, default_channel_colors
from .writers import _build_omero_block, _omero_version
from .zarr_compat import create_group_array, open_group_v2

logger = logging.getLogger(__name__)


def _ets_context(ets: ETSFile | str | Path):
    """Return a context manager for an ETS object or path."""
    if isinstance(ets, (str, Path)):
        return ETSFile(ets)
    return nullcontext(ets)


def _validate_tile(
    tile: np.ndarray, *, level: int, col: int, row: int, channels: int
) -> np.ndarray:
    """Validate a decoded ETS tile and return a channel-last array."""
    if tile.ndim != 3:
        raise ValueError(
            f"Decoded ETS tile at level={level}, col={col}, row={row} has shape {tile.shape}; "
            "expected (Y, X, C)."
        )
    if tile.shape[-1] != channels:
        raise ValueError(
            f"Decoded ETS tile at level={level}, col={col}, row={row} has {tile.shape[-1]} "
            f"channels; expected {channels}."
        )
    return tile


def _open_zarr_v2_group(out_dir: Path):
    """Open a filesystem Zarr group using the OME-NGFF v0.4-compatible v2 layout."""
    return open_group_v2(str(out_dir), mode="w")


def write_ets_pyramid_to_ngff_zarr(
    ets: ETSFile | str | Path,
    out_dir: str | Path,
    phys_xy_um: tuple[float, float] | None = None,
    *,
    name: str = "image",
    chunks_xy: int = 512,
    overwrite: bool = True,
    compressor: Blosc | None = None,
    dtype: np.dtype | str = "uint8",
    channel_labels: list[str] | None = None,
    channel_colors: list[str] | None = None,
    add_omero: bool = True,
    ngff_metadata: dict[str, Any] | None = None,
    metadata_schema: Literal["latest", "v0.4", "0.4"] | None = "v0.4",
    progress_interval_s: float = 30.0,
) -> None:
    """
    Stream an existing ETS pyramid directly into an OME-Zarr source pyramid.

    This writer deliberately avoids Dask, TensorStore, and ``ngff-zarr`` for
    source conversion. The ETS file already contains a tiled pyramid, so each
    decoded ETS tile is copied directly into the matching Zarr level.
    """
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)

    if chunks_xy <= 0:
        raise ValueError("chunks_xy must be > 0.")

    if compressor is None:
        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)

    with _ets_context(ets) as ets_obj:
        if ets_obj.nlevels <= 0:
            raise ValueError("ETS file has no pyramid levels.")

        channel_count = 3
        prepared = _prepare_ngff_writer_metadata(
            dataset_count=ets_obj.nlevels,
            channel_count=channel_count,
            name=name,
            fallback_phys_xy_um=phys_xy_um,
            ngff_metadata=ngff_metadata,
            metadata_schema=metadata_schema,
            channel_labels=channel_labels,
        )
        root_attrs = prepared["root_attrs"]
        resolved_name = prepared["resolved_name"]
        resolved_labels = prepared["resolved_channel_labels"]
        omero_version = _omero_version(root_attrs, prepared["schema"])

        root = _open_zarr_v2_group(out_dir)

        for level in range(ets_obj.nlevels):
            height, width = map(int, ets_obj.level_shape(level))
            n_cols, n_rows = ets_obj.level_ntiles(level)
            total_tiles = int(n_cols) * int(n_rows)
            arr = create_group_array(
                root,
                f"s{level}",
                shape=(channel_count, height, width),
                chunks=(
                    channel_count,
                    min(chunks_xy, height),
                    min(chunks_xy, width),
                ),
                dtype=dtype,
                compressor=compressor,
                overwrite=True,
                zarr_format=2,
            )
            arr.attrs["_ARRAY_DIMENSIONS"] = ["c", "y", "x"]

            logger.info(
                "Writing ETS level %d to %s/s%d: shape=(%d, %d), tiles=%d",
                level,
                out_dir,
                level,
                height,
                width,
                total_tiles,
            )
            level_start = time.perf_counter()
            last_log = level_start
            written = 0

            for tile_row in range(int(n_rows)):
                y0 = tile_row * int(ets_obj.tile_ysize)
                if y0 >= height:
                    continue
                for tile_col in range(int(n_cols)):
                    x0 = tile_col * int(ets_obj.tile_xsize)
                    if x0 >= width:
                        continue

                    tile = _validate_tile(
                        ets_obj.get_tile_decoded(level, tile_col, tile_row),
                        level=level,
                        col=tile_col,
                        row=tile_row,
                        channels=channel_count,
                    )
                    y1 = min(y0 + tile.shape[0], height)
                    x1 = min(x0 + tile.shape[1], width)
                    crop = tile[: y1 - y0, : x1 - x0, :]
                    if crop.dtype != np.dtype(dtype):
                        crop = crop.astype(dtype, copy=False)
                    arr[:, y0:y1, x0:x1] = np.moveaxis(crop, -1, 0)

                    written += 1
                    now = time.perf_counter()
                    if progress_interval_s > 0 and now - last_log >= progress_interval_s:
                        elapsed = max(now - level_start, 1e-9)
                        logger.info(
                            "ETS level %d progress: %d/%d tiles (%.1f tiles/sec)",
                            level,
                            written,
                            total_tiles,
                            written / elapsed,
                        )
                        last_log = now

            elapsed = max(time.perf_counter() - level_start, 1e-9)
            logger.info(
                "Finished ETS level %d: %d/%d tiles in %.1fs (%.1f tiles/sec)",
                level,
                written,
                total_tiles,
                elapsed,
                written / elapsed,
            )

        attrs = dict(root_attrs)
        if add_omero:
            attrs["omero"] = _build_omero_block(
                name=resolved_name,
                version=omero_version,
                channel_labels=resolved_labels,
                channel_colors=channel_colors or default_channel_colors(channel_count),
            )
        root.attrs.put(attrs)
