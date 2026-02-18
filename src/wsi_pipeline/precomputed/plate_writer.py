# ---------------------------
# Plate writer facade
# ---------------------------
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

import dask.array as da

from ..omezarr.pyramid import compute_num_mips_min_side, build_mips_from_yxc
from .cloudvolume import create_precomputed_cloudvolume, write_slice_cloudvolume
from .tensorstore import (
    create_precomputed_tensorstore,
    write_slice_tensorstore,
    write_slice_tensorstore_streaming,
)


class PlatePrecomputedWriter:
    """
    Preallocate a single precomputed layer and append tissues along Z.
    Backends: 'cloudvolume' or 'tensorstore'
    """
    def __init__(self,
                 precomp_path: str,
                 width: int, height: int, z_slices: int,
                 voxel_size_um=(1.0,1.0,1.0),
                 chunk_xy: int = 512,
                 min_side_for_mips: Optional[int] = None,
                 backend: str = "tensorstore", # "cloudvolume",
                 dtype: str = "uint8",
                 encoding: str = "raw",
                 stream_min_side: int = 8192, # 2^13
                 stream_bytes_threshold: int = 1_500_000_000,
                 **kwargs): # **kwargs for cloudvolume-specific arguments
        self.precomp_path = precomp_path
        self.W, self.H, self.Z = width, height, z_slices
        self.voxel_size_um = voxel_size_um
        self.chunk_xy = chunk_xy
        self.backend = backend
        self.dtype = dtype
        self.encoding = encoding
        self.stream_min_side = stream_min_side
        self.stream_bytes_threshold = stream_bytes_threshold
        

        if min_side_for_mips is None:
            min_side_for_mips = chunk_xy
        self.num_mips = compute_num_mips_min_side(width, height, min_side_for_mips)

        logger.info("Creating %s writer: %dx%dx%d, %d mips", backend, width, height, z_slices, self.num_mips)

        if backend == "cloudvolume":
            self._writers = create_precomputed_cloudvolume(
                precomp_path, width, height, z_slices, voxel_size_um,
                num_mips=self.num_mips, chunk_xy=chunk_xy,
                dtype=dtype, encoding=encoding, **kwargs
            )
        elif backend == "tensorstore":
            self._writers = create_precomputed_tensorstore(
                precomp_path, width, height, z_slices, voxel_size_um,
                num_mips=self.num_mips, chunk_xy=chunk_xy,
                dtype=dtype, encoding=encoding
            )
        else:
            raise ValueError("backend must be 'cloudvolume' or 'tensorstore'")


    def should_stream(self, tile_shape_yxc: Tuple[int,int,int], bytes_per_px: int = 1) -> bool:
        y, x, c = tile_shape_yxc
        # Large side or large byte footprint? stream.
        if max(y, x) >= self.stream_min_side:
            return True
        # 1.5 GB default safety cap (tune for your RAM; includes temp copies)
        est_bytes = y * x * c * max(1, bytes_per_px)
        return est_bytes >= self.stream_bytes_threshold
    
    # Streaming option writes base tile to all mips
    def write_slice_streaming(self, z_index: int, tile_yxc) -> None:
        if self.backend != "tensorstore":
            raise NotImplementedError("Streaming is implemented for TensorStore backend.")
        write_slice_tensorstore_streaming(self._writers, z_index, tile_yxc, block_xy=self.chunk_xy)

    

    def write_slice(self, z_index: int, mips_or_tile) -> None:
        """
        Minimal-change dual behavior:
        - If you pass a list of mip arrays (Y,X,C), writes a single Z slices across all mips.
        - If you pass a single base tile (Y,X,C), it auto-streams when large.
        """
        # Log first two shapes if it's a mip list (preserve your debug output)
        if isinstance(mips_or_tile, list):
            logger.debug("Writing z=%d, shapes: %s", z_index, [m.shape for m in mips_or_tile[:2]])
            if self.backend == "tensorstore":
                # If already built mips, don't second-guess; write as-is.
                # (You can opt-in to streaming by calling write_slice_streaming directly.)
                write_slice_tensorstore(self._writers, z_index, mips_or_tile)
            else:
                write_slice_cloudvolume(self._writers, z_index, mips_or_tile)
            return

        # Otherwise, assume (Y,X,C) base tile; choose streaming if big.
        tile = mips_or_tile
        if self.should_stream(getattr(tile, "shape", None)):
            logger.debug("Writing z=%d via streaming (tile %s)", z_index, tile.shape)
            return self.write_slice_streaming(z_index, tile)

        # Small tile: keep old behavior by building mips once here
        if isinstance(tile, da.Array):
            tile = tile.compute()
        ms = compute_num_mips_min_side(tile.shape[1], tile.shape[0], self.chunk_xy)
        mips = build_mips_from_yxc(tile, ms)
        logger.debug("Writing z=%d, shapes: %s", z_index, [m.shape for m in mips[:2]])
        if self.backend == "tensorstore":
            write_slice_tensorstore(self._writers, z_index, mips)
        else:
            write_slice_cloudvolume(self._writers, z_index, mips)


    def close(self):
        """Ensure all data is written when done."""
        if self.backend == "cloudvolume":
            for writer in self._writers:
                # Force any cached data to be written
                if hasattr(writer, 'cache'):
                    writer.cache.flush()
            # Clear references to ensure cleanup
            self._writers = None
        elif self.backend == "tensorstore":
            logger.debug("TensorStore writes completed (no flush needed)")
            # Just clear references
            self._writers = None
    

    def verify_write(self):
        """Verify data was written correctly."""
        if self.precomp_path.startswith("file://"):
            path = Path(self.precomp_path[7:])
        else:
            path = Path(self.precomp_path)
        
        # Check info file
        info_path = path / "info"
        if info_path.exists():
            info = json.loads(info_path.read_text())
            logger.info("Info file exists: %d channels, %d scales",
                        info['num_channels'], len(info['scales']))
        
        # Check for actual data files
        for scale_idx in range(min(2, self.num_mips)):  # Check first 2 scales
            scale_dir = path / str(scale_idx)
            if scale_dir.exists():
                chunks = list(scale_dir.glob("*"))
                if chunks:
                    # Read first chunk to verify it has data
                    first_chunk = chunks[0]
                    data = first_chunk.read_bytes()[:100]
                    has_data = any(b != 0 for b in data)
                    logger.info("Scale %d: %d chunks, has_data=%s", scale_idx, len(chunks), has_data)
                else:
                    logger.info("Scale %d: no chunks found", scale_idx)


    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

