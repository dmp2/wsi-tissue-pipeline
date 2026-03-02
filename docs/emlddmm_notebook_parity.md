# EM-LDDMM Notebook Parity

This page maps the staged `step5` workflow to `legacy_scripts/tb_macaque_emlddmm.ipynb`.

## Matched

- Load target data from the prepared slice directory or an equivalent precomputed source.
- Convert atlas and target axes into a shared micrometer space.
- Downsample to the working registration grid.
- Run atlas-free self-alignment on the target stack.
- Optionally run atlas-to-target EM-LDDMM on the working grid.
- Optionally run between-slice filling as a separate branch.
- Write registration configs and QC artifacts for inspection.

## Matched But Reorganized

- The notebook's procedural cells are split into configuration, target loading, workflow orchestration, output writing, and report generation modules.
- The notebook's transformation-graph setup is separated into:
  - `transformation_graph_config.json` for the pipeline-facing artifact
  - `transformation_graph_execution_config.json` for the notebook-style execution payload
- The notebook's ad hoc plotting and QC inspection is surfaced as:
  - persistent stage image outputs
  - `registration.log`
  - optional `registration_report.html`
  - optional `registration_report.json`
- Reproducibility metadata is elevated into first-class artifacts:
  - `run_provenance.json`
  - `reproduce_step5_command.txt`
- The raw atlas-registration payload is preserved as `atlas_registration/registration_data.npy` instead of only living in notebook state.

## Intentionally Different

- The staged pipeline uses a fixed output directory layout instead of notebook-cell-local paths.
- Report generation replaces notebook plotting as the primary review surface.
- The HTML report is optional and artifact-based; there are no notebook UI cells.
- Transformation-graph execution is resolved from the external `emlddmm` package or an explicit override path, rather than assuming a script copied into the working directory.
- The pipeline keeps the existing lightweight transformation-graph config artifact while also writing a notebook-style execution config for fidelity.
- The staged runner writes provenance and replay metadata that the notebook did not have.

## Practical Mapping

Notebook workflow:
1. load atlas and target
2. fix units
3. choose a working grid
4. self-align slices
5. register atlas to target
6. inspect QC outputs
7. optionally build and run the transformation graph
8. optionally fill missing slices

Pipeline workflow:
1. `step4` prepares the target metadata
2. `step5` resolves a run plan and writes `resolved_run_plan.json`
3. `self_alignment/` captures the atlas-free alignment branch
4. `atlas_registration/` captures EM-LDDMM outputs, configs, QC, and raw payloads
5. `upsampling/` captures between-slice filling outputs when enabled
6. `registration_summary.json`, `registration.log`, `run_provenance.json`, and the optional report replace notebook-state inspection

## Current Fidelity Boundary

High-level fidelity is strong for the main execution spine, but the staged runner is not a literal notebook export:
- stage orchestration is cleaner and more explicit
- artifact naming is more stable
- debug and QC surfaces are file-based rather than cell-based

That is intentional. The goal is notebook-faithful workflow behavior with a pipeline-friendly interface, not a cell-by-cell clone.
