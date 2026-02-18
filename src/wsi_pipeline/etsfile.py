"""
ETS File Reader

Reader for Olympus "CellSens" ETS image files.
Based on format analysis by Dale Roberts, Johns Hopkins University.

Coordinate Convention:
    COL == X == Horizontal
    ROW == Y == Vertical
"""

from __future__ import annotations

import os
from collections import namedtuple
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple

import numpy as np


class ETSFileError(Exception):
    """Exception raised for ETS file format errors."""
    pass


class ETSFile:
    """
    Reader for Olympus CellSens ETS image files.
    
    ETS files contain tiled, pyramidal image data with JPEG compression.
    This reader provides efficient random access to tiles at any pyramid level.
    
    Parameters
    ----------
    fname : str or Path
        Path to the ETS file.
    
    Attributes
    ----------
    npix_x : int
        Full resolution width in pixels.
    npix_y : int
        Full resolution height in pixels.
    nlevels : int
        Number of pyramid levels (0 = full resolution).
    ntiles : int
        Total number of tiles in the file.
    tile_xsize : int
        Tile width in pixels (typically 512).
    tile_ysize : int
        Tile height in pixels (typically 512).
    compression : int
        Compression type code.
    compression_str : str
        Human-readable compression type.
    
    Examples
    --------
    >>> ets = ETSFile("path/to/image.ets")
    >>> print(f"Image size: {ets.npix_x} x {ets.npix_y}")
    >>> print(f"Pyramid levels: {ets.nlevels}")
    >>> 
    >>> # Get a tile
    >>> tile_bytes = ets.get_tile(level=0, col=0, row=0)
    >>> 
    >>> # Get tile as numpy array (JPEG compressed bytes)
    >>> tile_np = ets.get_tile_np(level=0, col=0, row=0)
    """
    
    # Named tuple for tile index entries
    TileIdx = namedtuple("TileIdx", "col row level nbytes seqnum file_offset")
    TileLoc = namedtuple("TileLoc", "fpos nbytes")
    
    # Compression type codes
    COMPRESSION_TYPES = {
        0: "RAW",
        2: "JPEG",
        3: "JPEG_2000",
        5: "JPEG_LOSSLESS",
        8: "PNG",
        9: "BMP",
    }

    def __init__(self, fname: str | Path):
        fname = Path(fname)
        if not fname.is_file():
            raise FileNotFoundError(f"No such file: {fname}")

        # Disable buffering for non-sequential tile reads
        self._fh = open(fname, "rb", buffering=0)
        self._fh.seek(0, os.SEEK_END)
        self.fsize = self._fh.tell()
        self._fname = fname
        
        # Initialize attributes
        self.ntiles: int = 0
        self.nlevels: int = 0
        self.npix_x: int = 0
        self.npix_y: int = 0
        self.tile_xsize: int = 512
        self.tile_ysize: int = 512
        self.compression: int = 0
        self.compression_quality: int = 0
        self.is_bgr: bool = False
        self.use_pyramid: bool = False
        
        # Read headers
        self._read_headers()
        self._totalbytes = 0

    def __getstate__(self):
        """Support pickling by excluding file handle."""
        state = self.__dict__.copy()
        del state["_fh"]
        return state

    def __setstate__(self, state):
        """Restore state and reopen file handle."""
        self.__dict__.update(state)
        self._fh = open(self._fname, "rb", buffering=0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Close the file handle."""
        if hasattr(self, "_fh") and self._fh:
            self._fh.close()
            self._fh = None

    @property
    def compression_str(self) -> str:
        """Human-readable compression type string."""
        return self.COMPRESSION_TYPES.get(self.compression, "unknown")

    @property
    def shape(self) -> Tuple[int, int]:
        """Full resolution image shape (height, width)."""
        return (self.npix_y, self.npix_x)

    def _read_headers(self):
        """Read and parse the SIS and ETS headers."""
        fh = self._fh
        
        # =====================================================
        # Read SIS header (16 uint32 values)
        # =====================================================
        fh.seek(0)
        sis_data = np.fromfile(fh, np.uint32, 16).tolist()
        
        # Validate magic number
        sis_magic = int.from_bytes(b"SIS\x00", "little")
        if sis_data[0] != sis_magic:
            raise ETSFileError("SIS header not found - invalid file format")

        # Tile index location (64-bit offset)
        self._offset_tiles = sis_data[8] | (sis_data[9] << 32)
        self._offset_endtiles = sis_data[12] | (sis_data[13] << 32)
        self.ntiles = sis_data[10]

        # Validate tile index size (each entry is 36 bytes)
        tileidx_size = self._offset_endtiles - self._offset_tiles
        if tileidx_size != self.ntiles * 36:
            raise ETSFileError(
                f"Tile count mismatch: ntiles={self.ntiles}, "
                f"index size={tileidx_size / 36}"
            )

        if self._offset_endtiles > self.fsize:
            raise ETSFileError(
                f"Tile index offset {self._offset_endtiles} exceeds file size {self.fsize}"
            )

        # =====================================================
        # Read ETS header (57 uint32 values)
        # =====================================================
        ets_offset = sis_data[4]
        if ets_offset != fh.tell():
            raise ETSFileError("Expected ETS header immediately after SIS header")

        ets_data = np.fromfile(fh, np.uint32, 57).tolist()
        
        # Validate magic number
        ets_magic = int.from_bytes(b"ETS\x00", "little")
        if ets_data[0] != ets_magic:
            raise ETSFileError("ETS header not found")

        # Parse header fields
        self._pixtype = ets_data[2]
        self.compression = ets_data[5]
        self.compression_quality = ets_data[6]
        self.tile_xsize = ets_data[7]
        self.tile_ysize = ets_data[8]
        self.is_bgr = ets_data[37] == 1
        self.use_pyramid = bool(ets_data[38])
        self._ndim = ets_data[46]
        self.npix_x = ets_data[47]
        self.npix_y = ets_data[48]

        # Validate constraints
        if self._pixtype != 2:
            raise ETSFileError(f"Unsupported pixel type: {self._pixtype}")
        if self._ndim != 2:
            raise ETSFileError(f"Only 2D images supported, got {self._ndim}D")

        # =====================================================
        # Read tile index
        # =====================================================
        fh.seek(self._offset_tiles)
        tile_data = np.fromfile(fh, np.uint32, self.ntiles * 9)
        tile_data = tile_data.reshape(-1, 9)

        # Build tile index
        self._tiles = [
            self.TileIdx(
                col=int(row[1]),
                row=int(row[2]),
                level=int(row[4]),
                nbytes=int(row[7]),
                seqnum=int(row[8]),
                file_offset=int(row[5]) | (int(row[6]) << 32),
            )
            for row in tile_data
        ]

        # Validate sequence numbers
        seqnums = sorted(t.seqnum for t in self._tiles)
        if seqnums != list(range(self.ntiles)):
            raise ETSFileError("Invalid tile sequence numbers")

        # Build lookup dictionary
        self._tile_loc = {
            (t.level, t.col, t.row): self.TileLoc(t.file_offset, t.nbytes)
            for t in self._tiles
        }

        # Compute level info
        self.nlevels = max(t.level for t in self._tiles) + 1
        self._level_ntiles = [self._num_tiles_at_level(lvl) for lvl in range(self.nlevels)]

    def _num_tiles_at_level(self, level: int) -> Tuple[int, int]:
        """Get (num_cols, num_rows) for a pyramid level."""
        tiles_at_level = [t for t in self._tiles if t.level == level]
        if not tiles_at_level:
            return (0, 0)
        ncol = max(t.col for t in tiles_at_level) + 1
        nrow = max(t.row for t in tiles_at_level) + 1
        return (ncol, nrow)

    def level_shape(self, level: int) -> Tuple[int, int]:
        """
        Get image shape at a pyramid level.
        
        Parameters
        ----------
        level : int
            Pyramid level (0 = full resolution).
        
        Returns
        -------
        tuple
            (height, width) at the specified level.
        """
        scale = 2 ** level
        return (self.npix_y // scale, self.npix_x // scale)

    def level_ntiles(self, level: int) -> Tuple[int, int]:
        """
        Get tile grid dimensions at a pyramid level.
        
        Parameters
        ----------
        level : int
            Pyramid level.
        
        Returns
        -------
        tuple
            (num_cols, num_rows) of tiles at the level.
        """
        if 0 <= level < self.nlevels:
            return self._level_ntiles[level]
        raise ValueError(f"Invalid level {level}, must be 0-{self.nlevels - 1}")

    def get_tile(self, level: int, col: int, row: int) -> bytes:
        """
        Read a single tile as raw bytes.
        
        Parameters
        ----------
        level : int
            Pyramid level.
        col : int
            Column index.
        row : int
            Row index.
        
        Returns
        -------
        bytes
            Raw (compressed) tile data.
        """
        loc = self._tile_loc.get((level, col, row))
        if loc is None:
            raise KeyError(f"Tile not found: level={level}, col={col}, row={row}")
        
        self._fh.seek(loc.fpos)
        self._totalbytes += loc.nbytes
        return self._fh.read(loc.nbytes)

    def get_tile_np(self, level: int, col: int, row: int) -> np.ndarray:
        """
        Read a single tile as a numpy array of bytes.
        
        Parameters
        ----------
        level : int
            Pyramid level.
        col : int
            Column index.
        row : int
            Row index.
        
        Returns
        -------
        np.ndarray
            Tile data as uint8 array (still JPEG compressed).
        """
        tile_bytes = self.get_tile(level, col, row)
        return np.frombuffer(tile_bytes, dtype=np.uint8)

    def get_tile_decoded(self, level: int, col: int, row: int) -> np.ndarray:
        """
        Read and decode a tile to RGB image.
        
        Parameters
        ----------
        level : int
            Pyramid level.
        col : int
            Column index.
        row : int
            Row index.
        
        Returns
        -------
        np.ndarray
            Decoded tile as (H, W, C) uint8 array.
        """
        import cv2
        
        tile_bytes = self.get_tile_np(level, col, row)
        img = cv2.imdecode(tile_bytes, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ETSFileError(f"Failed to decode tile at level={level}, col={col}, row={row}")
        
        # Convert BGR to RGB if needed
        if self.is_bgr and img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img

    def iter_tiles(
        self,
        level: int,
        row_callback: Optional[Callable] = None,
        tile_callback: Optional[Callable] = None,
    ) -> Generator[bytes, None, None]:
        """
        Iterate over all tiles at a pyramid level.
        
        Yields tiles in row-major order (across columns, then down rows).
        
        Parameters
        ----------
        level : int
            Pyramid level.
        row_callback : callable, optional
            Called at the start of each row: callback(ets, level, row)
        tile_callback : callable, optional
            Called for each tile: callback(ets, level, col, row)
        
        Yields
        ------
        bytes
            Raw tile data.
        """
        ncol, nrow = self.level_ntiles(level)
        
        for row in range(nrow):
            if row_callback:
                row_callback(self, level, row)
            
            for col in range(ncol):
                if tile_callback:
                    tile_callback(self, level, col, row)
                yield self.get_tile(level, col, row)

    def read_level(self, level: int) -> np.ndarray:
        """
        Read entire image at a pyramid level.
        
        Parameters
        ----------
        level : int
            Pyramid level.
        
        Returns
        -------
        np.ndarray
            Complete image at the specified level as (H, W, C) array.
        
        Warnings
        --------
        This can use significant memory for large images.
        Consider using tiles or Dask arrays for large data.
        """
        import cv2
        
        if level < 0 or level >= self.nlevels:
            raise ValueError(f"Level {level} out of range [0, {self.nlevels - 1}]")
        
        ncol, nrow = self.level_ntiles(level)
        tx, ty = self.tile_xsize, self.tile_ysize
        
        # Create output array
        img = np.zeros((nrow * ty, ncol * tx, 3), dtype=np.uint8)
        
        for tile_col in range(ncol):
            for tile_row in range(nrow):
                tile_np = self.get_tile_np(level, tile_col, tile_row)
                tile_img = cv2.imdecode(tile_np, cv2.IMREAD_UNCHANGED)
                
                if tile_img is not None:
                    y0 = tile_row * ty
                    x0 = tile_col * tx
                    img[y0:y0 + ty, x0:x0 + tx, :] = tile_img
        
        # Crop to actual image size
        h, w = self.level_shape(level)
        img = img[:h, :w, :]
        
        # Convert BGR to RGB if needed
        if self.is_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img

    def to_dask(self, level: int = 0, chunks: Optional[Tuple[int, int]] = None):
        """
        Create a Dask array for lazy loading.
        
        Parameters
        ----------
        level : int
            Pyramid level.
        chunks : tuple, optional
            Chunk size (height, width). Defaults to tile size.
        
        Returns
        -------
        dask.array.Array
            Lazy-loaded image array.
        """
        import dask.array as da
        from dask import delayed
        
        h, w = self.level_shape(level)
        ncol, nrow = self.level_ntiles(level)
        tx, ty = self.tile_xsize, self.tile_ysize
        
        if chunks is None:
            chunks = (ty, tx, 3)
        
        # Create delayed tile readers
        @delayed
        def read_tile(level, col, row):
            return self.get_tile_decoded(level, col, row)
        
        # Build list of delayed arrays
        rows = []
        for tile_row in range(nrow):
            row_tiles = []
            for tile_col in range(ncol):
                tile = da.from_delayed(
                    read_tile(level, tile_col, tile_row),
                    shape=(ty, tx, 3),
                    dtype=np.uint8,
                )
                row_tiles.append(tile)
            rows.append(da.concatenate(row_tiles, axis=1))
        
        img = da.concatenate(rows, axis=0)
        
        # Crop to actual size
        return img[:h, :w, :]

    def __repr__(self) -> str:
        return (
            f"ETSFile('{self._fname.name}', "
            f"shape=({self.npix_y}, {self.npix_x}), "
            f"levels={self.nlevels}, "
            f"tiles={self.ntiles}, "
            f"compression='{self.compression_str}')"
        )
