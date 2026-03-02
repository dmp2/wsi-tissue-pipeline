"""
Entropy-based tissue segmentation.

Provides local-entropy tissue mask generation that works with both NumPy and Dask arrays.
"""


import dask.array as da
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk


def entropy_mask(gray: np.ndarray | da.Array, struct_elem_px: int, min_area: int) -> np.ndarray:
    """
    Local-entropy tissue mask that works with either NumPy (thumbnail) or Dask arrays.

    Steps:
      1) Convert grayscale to uint8 using *global* min/max (avoid per-chunk rescale).
      2) Compute rank entropy with disk(struct_elem_px) using map_overlap for Dask, direct for NumPy.
      3) One global Otsu threshold on the full entropy image.
      4) closing -> hole fill -> remove small objects/holes.

    Parameters
    ----------
    gray : np.ndarray or dask.array.Array
        Grayscale image.
    struct_elem_px : int
        Structuring element radius.
    min_area : int
        Minimum object area in pixels.

    Returns
    -------
    mask : np.ndarray
        Boolean mask (H, W).
    """
    fp = disk(max(1, int(struct_elem_px)))

    def _u8_from_gray(arr: np.ndarray, gmin: float, gscale: float) -> np.ndarray:
        # scale with shared global stats to keep chunk boundaries seamless
        out = (arr.astype(np.float32) - gmin) * gscale
        return np.clip(out, 0, 255).astype(np.uint8)

    if isinstance(gray, da.Array):
        # --- GLOBAL stats (cheap reductions) ---
        gmin = float(gray.min().compute())
        gmax = float(gray.max().compute())
        gscale = 255.0 / (gmax - gmin + 1e-6)

        # --- per-chunk entropy with overlap depth=struct_elem_px ---
        def _entropy_chunk_u8(chunk, gmin, gscale, struct_elem_px):
            u8 = _u8_from_gray(chunk, gmin, gscale)
            return rank_entropy(u8, disk(struct_elem_px))

        ent = gray.map_overlap(
            _entropy_chunk_u8,
            depth=(struct_elem_px, struct_elem_px),
            boundary="reflect",
            dtype=np.uint8,
            gmin=gmin, gscale=gscale, struct_elem_px=int(struct_elem_px)
        )

        # global threshold on the *full* entropy image (avoids blockwise biases)
        ent_np = ent.astype(np.float32).compute()
    else:
        # NumPy path (e.g., your fixed thumbnail)
        g = gray.astype(np.float32, copy=False)
        g -= g.min()
        g /= (g.max() - g.min() + 1e-6)
        u8 = (g * 255.0 + 0.5).astype(np.uint8)
        ent_np = rank_entropy(u8, fp).astype(np.float32)

    thr = filters.threshold_otsu(ent_np)
    bw = ent_np > thr

    # morphology to get solid sections
    bw = morphology.binary_closing(bw, footprint=fp)
    bw = ndi.binary_fill_holes(bw)
    bw = morphology.remove_small_objects(bw, min_size=int(min_area))
    bw = morphology.remove_small_holes(bw, area_threshold=int(min_area))
    return bw


# Backward compatibility
_entropy_mask = entropy_mask
