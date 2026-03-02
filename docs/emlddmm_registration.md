# EM-LDDMM Registration Guide

This page is the step-4 to step-5 handoff for the staged EM-LDDMM workflow.

## What Step 4 Produces

After `step4`, the prepared dataset root should contain:
- `samples.tsv`
- per-slice JSON sidecars
- `emlddmm_dataset_manifest.json`

Use that directory as `--dataset-root` for `step5`.
`-o/--output` still works as a deprecated compatibility alias, but the canonical name is `--dataset-root`.

## Step-5 Modes

Atlas-free mode:
- Omit `--atlas`.
- The workflow loads the target, rescales coordinates, downsamples to the working grid, and runs self-alignment.

Atlas-registration mode:
- Supply `--atlas`.
- Also supply either `--init-affine` or both `--orientation-from` and `--orientation-to`.
- Optional `--label` improves atlas QC overlays.

## Target Input Modes

Prepared directory:
- Default mode for step-4 outputs.
- Either omit `--target-source` or point it at the prepared directory.

Precomputed:
- Use `--target-source-format precomputed`.
- Supply `--target-source` with the local precomputed volume root.
- Supply `--precomputed-manifest` from step 4.
- Requires `tensorstore`.

## Notebook-Aligned Defaults

The `macaque-notebook` preset is the default and currently assumes:
- `atlas_unit_scale = 1000.0`
- `target_unit_scale = 1.0`
- `desired_resolution_um = 200.0`
- `resampling.policy = "sectioned-stack"`

Interpretation:
- Atlas axes are assumed to be stored in millimeters and are converted to micrometers.
- Target axes are assumed to already be in micrometers.
- The default pre-resampling policy preserves the target section axis and applies `desired_resolution_um` to in-plane target axes.

## Pre-Resampling Policy

`step5` has an outer pre-resampling layer before EM-LDDMM's internal multiscale `downI/downJ` schedule.

Default policy:
- `sectioned-stack`
- Intended for current serial-section targets.
- Preserves target axis `0`.
- Uses `desired_resolution_um` for in-plane target preprocessing.
- Computes atlas preprocessing relative to the already chosen target working grid.

Compatibility policy:
- `legacy-target-first`
- Preserves the older target-first all-axis preprocessing behavior.
- Use this only when you need backward-compatible behavior from earlier step-5 runs.

## Atlas Initialization

Matrix initialization:
- Supply `--init-affine` with a 4x4 transform.

Orientation-derived initialization:
- Supply `--orientation-from` and `--orientation-to`.
- These values are backend orientation codes passed to `orientation_to_orientation`.

## Orientation Codes

- Use `python scripts/run_pipeline.py step5 --list-orientations` to print the valid codes without running a registration.
- Codes are validated before backend execution.
- Valid codes are exactly three letters and must use one axis from each pair: `{R/L}`, `{A/P}`, and `{S/I}`.
- Common examples: `RAS`, `LPI`, `PIR`, `RIP`.

## Transformation Graph

`--run-transformation-graph` is optional and only applies to atlas-registration runs.

Resolution order for `transformation_graph_v01.py`:
1. `--transformation-graph-script`
2. installed external `emlddmm` package
3. workspace-local development fallback

Important:
- `transformation_graph_v01.py` is treated as part of the external `emlddmm` package, not this repository.
- `--dry-run` fails early if execution is requested and the script cannot be resolved.

## Logging and Reports

Every step-5 run writes:
- `registration.log`
- `resolved_run_plan.json`
- `registration_summary.json`
- `run_provenance.json`
- `reproduce_step5_command.txt`

Optional outputs:
- `--write-qc-report` writes `registration_report.json` and `registration_report.html`
- `--write-notebook-bundle` writes a debug bundle of plan and stage payload summaries

The QC report links to images already produced by the registration stages, including atlas QC outputs and upsampling overview PNGs.
The plan and summary files also include structured `pre_resampling_plan` metadata describing the chosen policy, native spacings, locked axes, and target/atlas preprocessing factors.

## Reproducibility

Step 5 now writes provenance on every run.

- `run_provenance.json` records pipeline version, git metadata when available, Python/runtime details, backend origin, transformation-graph script resolution, control-input metadata, and the fully resolved workflow config.
- `reproduce_step5_command.txt` records a canonical replay command that uses `--dataset-root` even if the original invocation used the deprecated `--output` alias.
- Small control files such as JSON configs, manifests, init-affine files, and transformation-graph scripts are hashed with SHA-256.
- Large datasets and image volumes are not recursively hashed by default; they are tracked by resolved path, type, size, and modification time instead.

## Common Commands

Atlas-free:

```bash
python scripts/run_pipeline.py step5 \
  --dataset-root /data/tiles
```

Atlas registration from prepared slices:

```bash
python scripts/run_pipeline.py step5 \
  --dataset-root /data/tiles \
  --atlas /data/atlas.vtk \
  --label /data/atlas_labels.vtk \
  --orientation-from PIR \
  --orientation-to RIP
```

Precomputed target input:

```bash
python scripts/run_pipeline.py step5 \
  --dataset-root /data/tiles \
  --target-source /data/precomputed_plate \
  --target-source-format precomputed \
  --precomputed-manifest /data/tiles/emlddmm_dataset_manifest.json
```

Dry-run with QC report planning:

```bash
python scripts/run_pipeline.py step5 \
  --dataset-root /data/tiles \
  --atlas /data/atlas.vtk \
  --orientation-from PIR \
  --orientation-to RIP \
  --write-qc-report \
  --dry-run
```

Transformation-graph execution with explicit script path:

```bash
python scripts/run_pipeline.py step5 \
  --dataset-root /data/tiles \
  --atlas /data/atlas.vtk \
  --orientation-from PIR \
  --orientation-to RIP \
  --run-transformation-graph \
  --transformation-graph-script /opt/emlddmm/transformation_graph_v01.py
```

## Expected Stage Layout

Typical outputs under `<dataset-root>/emlddmm`:
- `self_alignment/`
- `atlas_registration/`
- `upsampling/`
- `registration.log`
- `resolved_run_plan.json`
- `registration_summary.json`
- `run_provenance.json`
- `reproduce_step5_command.txt`
- optional `registration_report.json`
- optional `registration_report.html`

Atlas-registration runs also write notebook-parity artifacts:
- `atlas_registration/registration_data.npy`
- `atlas_registration/transformation_graph_execution_config.json`

## Related Docs

- [`emlddmm_notebook_parity.md`](/C:/Users/dpado/Documents/git/temporal_bone_mapping/wsi-tissue-pipeline/wsi-tissue-pipeline/docs/emlddmm_notebook_parity.md)
- [`configuration.md`](/C:/Users/dpado/Documents/git/temporal_bone_mapping/wsi-tissue-pipeline/wsi-tissue-pipeline/docs/configuration.md)
- [`installation.md`](/C:/Users/dpado/Documents/git/temporal_bone_mapping/wsi-tissue-pipeline/wsi-tissue-pipeline/docs/installation.md)
