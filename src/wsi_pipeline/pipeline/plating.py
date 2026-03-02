# ---------------------------
# Per slide processing with per-tissue dual write
# ---------------------------
from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

import dask.array as da
import dask.config
import numpy as np
import zarr

from ..omezarr.metadata import _get_multiscales_paths, _phys_xy_um
from ..omezarr.pyramid import build_mips_from_yxc, compute_num_mips_min_side
from ..omezarr.streaming import write_ngff_from_tile_streaming_ome, write_ngff_from_tile_ts
from ..omezarr.writers import write_ngff_from_mips_ngffzarr
from ..precomputed.plate_writer import PlatePrecomputedWriter
from ..tiles.generator import generate_tissue_tiles

logger = logging.getLogger(__name__)


def _is_big_tile(tile_da, bytes_per_px=1, min_side=8192, max_bytes=1_500_000_000):
    y, x, c = map(int, tile_da.shape)
    return (max(y, x) >= min_side) or (y * x * c * bytes_per_px >= max_bytes)


def _safe_close_existing_client():
    # If a client already exists (e.g., from a previous run), close it to avoid port 8787 conflicts.
    from dask.distributed import get_client
    try:
        c = get_client()
        try:
            c.shutdown()  # politely stop scheduler + workers if we own them
        except Exception:
            c.close()     # at least drop our client
    except ValueError:
        pass  # no existing client

def process_slide_with_plating(
    zarr_root_path: os.PathLike,
    out_ngff_dir: os.PathLike,                      # where to write tissue_region_*.zarr
    *,
    # Coarse segmentation function preprocess() options
    segment_fn,                                     # callable(dask_arr)->(filled_mask_bool, _)
    struct_elem_px: int = 9,                         # structuring element radius in pixels
    min_size: int = 2000,
    # Plate options
    precomputed_plate_path: str | None = None,   # "file:///â€¦/plate_precomp"
    plate_backend: str = "tensorstore",             # or "cloudvolume"
    plate_chunk_xy: int = 512,
    parallel: bool = False,
    fill_missing: bool = False,
    # mips options
    min_side_for_mips: int | None = None,        # default to chunk size in writers
    downscale: int = 2,                             # default downsampling rate for mips (not used yet)
    # dtype policy
    dtype: np.dtype | None = "uint8"            # cast ROI to uint8 before mip (recommended for imagery)
    ) -> list[Path]:
    """
    Pipeline for ONE slide root (OME-Zarr):
        1) Run segmentation on coarsest level (sL) -> boolean mask with N labels after filling.
        2) Build **Dask** tiles at s0 (one per tissue) with ROI upsampling & HR masking.
        3) For each tile: compute -> build tinybrain mips (once) -> write per-tissue NGFF.
        4) Optionally append each tissue as a Z-slice into a single Precomputed plate.
    Returns: list of per-tissue NGFF output directories.

    Processes tissue regions at the highest resolution, generates pyramids, and writes OME-Zarr files.
    Upsample `low_res_mask` => highest-res, cut out 3 square regions, build a pyramid for each and
    (optionally) write them as separate OME-Zarr NGFF datasets.

    1) If low_res_mask is None, run your preprocess() on the coarsest level to get it.
    2) Build a **list** of high-res tissue tiles (Y,X,C) via generate_tissue_tiles().
    3) For each tile, build a multiscale pyramid and write one OME-Zarr group.

    Parameters:
    - zarr_path: Path to the highest-resolution OME-Zarr image.
    - low_res_mask: Binary mask of tissue regions, assumed to be the lowest resolution. If None, generate the mask.
    - target_dim: Target square dimensions for each tissue region at the highest resolution.

    Returns
    -------
    region_paths : list of paths. No large arrays are held in memory.
    """
    # Ensure the zarr_root_path is a Path object
    zarr_root_path = Path(zarr_root_path)

    # Make the output directory if it doesn't already exist
    out_ngff_dir = Path(out_ngff_dir)
    out_ngff_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load the NGFF root, find image and metadata for the levels
    logger.info("Loading the NGFF root.")

    root = zarr.open_group(str(zarr_root_path), mode = 'r')
    ds_paths = _get_multiscales_paths(root)        # e.g. ["s0","s1","s2","s3"]
    # Fix the level of the highest resolution you want to use: default should be 0 for the highest resolution
    L_idx = 0 # s0 is the highest resolution
    # L_idx = -2 # testing the pipeline with a lower resolution image
    L_path = ds_paths[L_idx]
    sL_path = ds_paths[-1]

    # Must read physical pixel size from that level, not hard coded
    L_idx = ds_paths.index(L_path)            # robust in case ordering changes

    # Load arrays which are stored (C,Y,X) with parallelization; give unique and stable names to prevent collisions in the graph
    RUN_UID = f"{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    dataset_id = zarr_root_path.name.replace(".", "_")
    base_name  = f"ngff-{dataset_id}-{L_path}-{RUN_UID}"
    coarse_name= f"ngff-{dataset_id}-{sL_path}-{RUN_UID}"
    base_cyx = da.from_zarr(str(zarr_root_path / L_path), name=base_name)  # (C, H, W)
    sL = da.from_zarr(str(zarr_root_path / sL_path), name=coarse_name)

    # # If the source is of modest size, persist once so that all dowstream tiles share one graph; otherwise, avoid entirely, and slice lazily.
    # base_cyx = base_cyx.persist()
    # wait(base_cyx)
    # sL = sL.persist()
    # wait(sL)

    # Should I rechunk to force the arrays to align with the tile size?
    # base_cyx = base_cyx.rechunk((3, plate_chunk_xy, plate_chunk_xy))
    # sL = sL.rechunk((3, plate_chunk_xy, plate_chunk_xy))

    # Double check high res image shape info; channel information is either grayscale or RGB
    if base_cyx.ndim != 3 or base_cyx.shape[0] not in (1,3):
        raise ValueError("Expected (C,Y,X) at base")

    # Precompute shapes and physical pixel sizes at the highest resolution
    C, H0, W0 = base_cyx.shape
    px_um, py_um = _phys_xy_um(root, L_idx) # this is the base s0 scale now

    # Step 2: Segment at the coarsest level using the segment_fn to get a tissue region mask
    logger.info("Generating tissue masks at the coarsest resolution.")

    # Ensure channel-first for the grayscale() function within segment_fn
    if sL.ndim == 3 and sL.shape[-1] in (1,3) and sL.shape[0] not in (1,3):
        sL = da.moveaxis(sL, -1, 0) # (C, Y, X)

    # Get the tissue masks by labeling connected components then filling holes per region
    filled_lr, _ = segment_fn(
        sL, # coarsest image
        dynamic_threshold=True, # adaptive thresholding using Otsu's method
        fixed_threshold=0.7, # only used if dynamic_threshold=False
        min_size=min_size, # minimum size of objects to retain (depends on pyramid level)
        struct_elem_px=struct_elem_px, # radius of the closing disk
        additional_smooth=False, # apply an additional smoothing operation for smoother edges
        output_images=False # output tissue images
    )


    # Step 2: upsample the low-resolution mask to the high-resolution image
    # Build LIST of HR tiles (Y,X,C), ordered left->right
    tiles_yxc, tile_dim = generate_tissue_tiles(
        s0_cyx=base_cyx,
        low_res_filled=filled_lr.astype(bool),
        chunk=plate_chunk_xy,
        pad_multiple=plate_chunk_xy,  # or another multiple, if we prefer
        extra_margin_px=0
    )
    # Break out early if there are no tissue sections
    if not tiles_yxc:
        logger.warning("[%s] no tissue regions found.", zarr_root_path.name)
        return []

    plate = None
    if precomputed_plate_path:
        # Z is the number of tiles; voxel_size_z is arbitrary (1.0 um) for 2D plates
        plate = PlatePrecomputedWriter(
            precomp_path=precomputed_plate_path,
            width=tile_dim,
            height=tile_dim,
            z_slices=len(tiles_yxc),
            voxel_size_um=(px_um, py_um, 1.0),
            chunk_xy=plate_chunk_xy,
            min_side_for_mips=min_side_for_mips,
            backend=plate_backend,
            dtype=dtype if dtype else str(base_cyx.dtype),
            encoding="raw",
            parallel=parallel,
            fill_missing=fill_missing
        )

    # 3) iterate tiles
    out_paths: list[Path] = []

    big_tile_threshold = 8192 # 2^13=8192; tune threshold
    item_size_threshold = 1_500_000_000
    bytes_per_px = np.dtype(dtype if dtype else np.uint8).itemsize
    # nbytes_est = np.prod(np.squeeze(tiles_yxc[0].shape)) * bytes_per_px
    any_big = any(_is_big_tile(tile_da=t, bytes_per_px=bytes_per_px, min_side=big_tile_threshold, max_bytes=item_size_threshold) for t in tiles_yxc)
    n_tiles_threshold = 16
    n_tiles = len(tiles_yxc)
    # If any tile is huge, avoid distributed 'compute-then-write' path
    use_distributed = bool(parallel) and (n_tiles >= n_tiles_threshold) and not any_big

    if use_distributed:
        from dask.distributed import Client, LocalCluster, as_completed

        _safe_close_existing_client()

        # Create a local cluster so that we can stream compute tiles. This makes us parallelized across multiple CPUs but not memory-exploding
        # Upper Limit: Each image pyramid should be ~2.5GB so we should be good. Note that agregate_memory = n_workers * memory_limit (e.g. 10 workers, "8GB" -> 80GB)
        # Use context managers so we always shut down cleanly
        n_workers = min(10, os.cpu_count())
        with LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            # processes=False,        # False w/threads may stabilize WSL2 runs
            processes=True,        # separate processes (nanny restarts workers)
            memory_limit="auto",   # let Dask size to process, safer in WSL2
            scheduler_port=0,      # random free port, avoids conflicts
            dashboard_address=None # disable dashboard to avoid port issues
        ) as cluster, Client(cluster, set_as_default=True) as client:

            # Set distributed configuration
            dask.config.set({
            "array.slicing.split_large_chunks": True, # useful when there are oversized chunks
            "distributed.worker.memory.target": 0.6,
            "distributed.worker.memory.spill": 0.7,
            "distributed.worker.memory.pause": 0.85,     # helps prevent out of memory
            "distributed.worker.memory.terminate": 0.95,
            "distributed.comm.timeouts.connect": "20s",
            "distributed.comm.timeouts.tcp": "120s"
            })

            client.wait_for_workers(1) # don't block waiting for all workers

            # Submit in batches so we donâ€™t flood scheduler/RAM
            batch_size = 8
            for start in range(0, n_tiles, batch_size):
                batch = tiles_yxc[start:start+batch_size]
                fmap = {client.compute(t): (start+i) for i, t in enumerate(batch)}

                for fut in as_completed(fmap):
                    z_idx = fmap.pop(fut)
                    tile = fut.result()                     # stores (Y,X,C) numpy array as soon as one tile finishes
                    # dtype policy (optional)
                    if dtype and tile.dtype != np.uint8:
                        logger.debug("Enforcing dtype policy.")
                        if np.issubdtype(tile.dtype, np.integer):
                            maxv = np.iinfo(tile.dtype).max
                            tile = (tile.astype(np.float32) / max(1, maxv) * 255.0).clip(0,255).astype(np.uint8)
                        else:
                            tile = (tile * 255.0).clip(0,255).astype(np.uint16)

                    # Write per-tissue NGFF
                    name = f"{zarr_root_path.stem}_tissue_{z_idx+1:02d}"
                    ngff_dir = out_ngff_dir / f"{name}.ome.zarr"
                    if any_big:
                        # STREAM the NGFF pyramid from base tile via TensorStore
                        write_ngff_from_tile_ts(
                            tile, ngff_dir, (px_um, py_um),
                            chunks_xy=plate_chunk_xy,
                            num_mips=compute_num_mips_min_side(tile.shape[1], tile.shape[0],
                                                            min_side_for_mips or plate_chunk_xy),
                            name=name, version="0.4",
                            channel_labels=[f"ch{i}" for i in range(tile.shape[2])],
                            channel_colors=["FFFFFF"] * tile.shape[2],
                        )
                        # Plate: let your writer auto-stream based on its own gate
                        if plate is not None:
                            plate.write_slice(z_idx, tile)
                    else:
                        # Small/medium: keep your fast path (build mips once in RAM)
                        ms = compute_num_mips_min_side(tile.shape[1], tile.shape[0],
                                                    min_side_for_mips or plate_chunk_xy)
                        mips = build_mips_from_yxc(tile, ms)
                        write_ngff_from_mips_ngffzarr(
                            mips_yxc=mips,
                            out_dir=ngff_dir,
                            phys_xy_um=(px_um, py_um),
                            name=name,
                            chunks_xy=plate_chunk_xy,
                            version="0.4",          # keep 0.4 unless you want sharded v0.5
                            overwrite=True,
                            channel_labels=[f"ch{i}" for i in range(mips[0].shape[2])],
                            channel_colors=["FFFFFF"] * mips[0].shape[2],
                            add_omero=True
                        )
                        # # This function uses zarr and manually specifies the metadata
                        # write_ngff_from_mips(mips,
                        # ngff_dir,
                        # (px_um, py_um),
                        # name=name,
                        # chunks_xy=plate_chunk_xy,
                        # dtype=mips[0].dtype)
                        out_paths.append(ngff_dir)
                        if plate is not None:
                            plate.write_slice(z_idx, mips)

                    del tile

                #     # Release worker memory early
                #     fut.release()

                # client.run(gc.collect)
    else:
        # Set distributed configuration
        dask.config.set({
        "array.slicing.split_large_chunks": True # useful when there are oversized chunks
        })

        # For small jobs, threads are faster and simpler (no scheduler ports, no heartbeats)
        _safe_close_existing_client()

        with dask.config.set(scheduler="threads"):
            for z_idx, tile_dask in enumerate(tiles_yxc, start=0):
                # Write per-tissue NGFF
                name = f"{zarr_root_path.stem}_tissue_{z_idx+1:02d}"
                ngff_dir = out_ngff_dir / f"{name}.ome.zarr"
                logger.debug("tile %d: %s, big=%s", z_idx, tuple(map(int, tile_dask.shape)), _is_big_tile(tile_dask, bytes_per_px))

                # For big tiles, don't build mips_yxc in memory. Instead, write directly from the base tile with the streaming writer
                if any_big:
                    # DO NOT compute() â€“ keep it lazy and (optionally) cast lazily
                    # tlazy = tile_dask.astype(np.uint8) if dtype and tile_dask.dtype != np.uint8 else tile_dask

                    # write_ngff_from_tile_ts(
                    #     tlazy,
                    #     ngff_dir,
                    #     (px_um, py_um),
                    #     chunks_xy=plate_chunk_xy,
                    #     num_mips=compute_num_mips_min_side(tlazy.shape[1], tlazy.shape[0],
                    #                                     min_side_for_mips or plate_chunk_xy),
                    #     name=name, version="0.4",
                    #     channel_labels=[f"ch{i}" for i in range(tlazy.shape[2])],
                    #     channel_colors=["FFFFFF"] * tlazy.shape[2],
                    # )
                    write_ngff_from_tile_streaming_ome(
                        tile_yxc_da=tile_dask.astype(np.uint8) if tile_dask.dtype != np.uint8 else tile_dask,
                        out_dir=ngff_dir,
                        phys_xy_um=(px_um, py_um),
                        block_xy=plate_chunk_xy,
                        num_mips=compute_num_mips_min_side(tile_dask.shape[1], tile_dask.shape[0],
                                                        min_side_for_mips or plate_chunk_xy),
                        name=name,
                        compressor=None,  # or zarr.Blosc(cname="zstd", clevel=5, shuffle=2)
                        channel_labels=[f"ch{i}" for i in range(int(tile_dask.shape[2]))],
                        channel_colors=["FFFFFF"] * int(tile_dask.shape[2]),
                    )

                    # Append to precomputed plate (optional)
                    if plate is not None:
                        # plate.write_slice(z_idx, tlazy)   # accepts dask arrays, streams blocks
                        plate.write_slice(z_idx, tile_dask.astype(np.uint8) if tile_dask.dtype != np.uint8 else tile_dask)   # accepts dask arrays, streams blocks

                else:
                    tile = tile_dask.compute()                  # numpy (Y,X,C); ok for small/medium arrays
                    # dtype policy (optional)
                    if dtype and tile.dtype != np.uint8:
                        logger.debug("Enforcing dtype policy.")
                        if np.issubdtype(tile.dtype, np.integer):
                            maxv = np.iinfo(tile.dtype).max
                            tile = (tile.astype(np.float32) / max(1, maxv) * 255.0).clip(0,255).astype(np.uint8)
                        else:
                            tile = (tile * 255.0).clip(0,255).astype(np.uint16)

                    # Compute mips once
                    ms = compute_num_mips_min_side(tile.shape[1],
                                                tile.shape[0],
                                                min_side_for_mips or plate_chunk_xy)
                    mips = build_mips_from_yxc(tile, ms)
                    # print(f"mips[0].dtype: {mips[0].dtype}")

                    write_ngff_from_mips_ngffzarr(
                        mips_yxc=mips,
                        out_dir=ngff_dir,
                        phys_xy_um=(px_um, py_um),
                        name=name,
                        chunks_xy=plate_chunk_xy,
                        version="0.4",          # keep 0.4 unless you want sharded v0.5
                        overwrite=True,
                        channel_labels=[f"ch{i}" for i in range(mips[0].shape[2])],
                        channel_colors=["FFFFFF"] * mips[0].shape[2],
                        add_omero=True
                    )
                    # This function uses zarr and manually specifies the metadata
                    # write_ngff_from_mips(mips,
                    #                     ngff_dir,
                    #                     (px_um, py_um), # base scale physical pixel size
                    #                     name=name,
                    #                     chunks_xy=plate_chunk_xy,
                    #                     dtype=mips[0].dtype)
                    out_paths.append(ngff_dir)

                    # Append to precomputed plate (optional)
                    if plate is not None:
                        plate.write_slice(z_idx, mips)

                del tile

    # # Optional: extra cleanup in notebooks so reruns donâ€™t collide
    # gc.collect()

    logger.info("Wrote %d tissue OME-Zarrs to %s", len(out_paths), out_ngff_dir)

    try:
        # Shut down the cluster
        client.close()
        client.shutdown()
    except Exception:
        pass

    return out_paths
