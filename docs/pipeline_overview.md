# Pipeline Overview

The pipeline processes large whole-slide images (potentially 0.5 TB+ per specimen)
through staged preprocessing and reconstruction:

1. **Convert** VSI/ETS data into flat image assets and metadata.
2. **Segment** tissue, extract sections, and plate them for downstream analysis.
3. **Write** OME-Zarr pyramids and optional Neuroglancer-precomputed volumes.
4. **Prepare** the EM-LDDMM dataset root and manifest (`step4`).
5. **Reconstruct** with the reproducible registration workflow (`step5`).

`step5` supports atlas-free self-alignment, optional atlas registration, optional
between-slice upsampling/filling, plus QC reports, logs, provenance, and replay
artifacts.

## Module architecture

The pipeline is organized into focused submodules (see the
[API Reference](api/index.md) for full signatures):

| Module | Purpose | Key exports |
|--------|---------|-------------|
| `segmentation` | Tissue mask generation | `WSISegmenter`, `segment_mask`, `make_lowres_mask` |
| `tiles` | Tile extraction from masks | `generate_tissue_tiles`, `save_tile`, `to_uint8` |
| `omezarr` | OME-Zarr pyramid building | `build_mips_from_yxc`, `write_ngff_from_mips`, `write_ngff_from_mips_ngffzarr` |
| `precomputed` | Neuroglancer Precomputed writing | `PlatePrecomputedWriter` |
| `pipeline` | End-to-end orchestration | `process_slide_with_plating` |
| `registration` | Staged EM-LDDMM planning, execution, reporting, upsampling | `plan_emlddmm_workflow`, `run_emlddmm_workflow`, `emlddmm_multiscale_symmetric_N`, `upsample_between_slices` |
| `neuroglancer` | Neuroglancer state, server, and viewer | `NeuroglancerViewer`, `emit_ng_state_for_ngff_plate`, `open_neuroglancer_plate_view` |
| `sciserver` | SciServer deployment (optional) | `SciServerPipeline`, `setup_sciserver_tracking` |

**Canonical API:** prefer standalone functions (`process_wsi`, `process_specimen`,
`run_qc_workflow`, `build_qc_grids`) for scripting. Use `WSIProcessor` and
`QCGridBuilder` classes when you want to configure once and call multiple times.

## Staged EM-LDDMM runner

The staged runner in `scripts/run_pipeline.py` operationalizes the
notebook-aligned EM-LDDMM workflow as `step4` / `emlddmm-prep` and
`step5` / `reconstruct`. It is the supported non-interactive path for reproducible
registration runs.

- `step4` writes `samples.tsv`, per-slice JSON sidecars, and
  `emlddmm_dataset_manifest.json` into the dataset root.
- `step5` loads that prepared target, rescales axes into micrometers, downsamples
  to the working grid, runs atlas-free self-alignment, optionally runs atlas
  registration, optionally fills between slices, and writes stable run artifacts.

For full details see:

- [EM-LDDMM registration](emlddmm_registration.md) — workflow overview, defaults,
  target modes, logging, QC reports, and optional transformation-graph execution.
- [EM-LDDMM notebook parity](emlddmm_notebook_parity.md) — mapping from the legacy
  notebook to the staged pipeline.
- [Configuration](configuration.md) — YAML pipeline config and `--emlddmm-config`
  overrides.
