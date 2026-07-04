# Batch OME-TIFF Submission Factory

The Batch OME-TIFF Submission Factory is the planned workflow layer for turning
large Olympus VSI/ETS whole-slide image batches into database-ready,
single-tissue-section OME-TIFF derivatives.

This scaffold makes the submission batch the central object. A batch records the
input manifest, database profile, source slides, detected tissue-section records,
future derivative outputs, validation state, warnings, and blocking errors. The
workflow is not centered on an opened viewer image because each source WSI may
contain multiple tissue sections, and the database requires one image file per
tissue section.

## Why Not A QuPath-First Workflow

Manual QuPath review can be useful for expert inspection, but it is not enough
as the primary workflow for database upload. The future system must repeatedly:

- inspect many large VSI/ETS datasets
- detect or segment tissue sections per parent slide
- crop one approved derivative per tissue section
- preserve parent-slide provenance and crop mappings
- validate required metadata and upload-package contents
- resume work after warnings, expert review, or failed conversion attempts

Those are batch-submission responsibilities, not single-image viewer
responsibilities.

## Intended Users

- Operator: chooses an input folder or manifest, runs preflight checks, reviews
  clear pass/warning/blocked states, and asks for expert help when needed.
- Expert reviewer: resolves ambiguous tissue masks, missing metadata, orientation
  uncertainty, and warning-state tissue sections.
- Admin: maintains database profiles, metadata requirements, sidecar policy,
  naming templates, and validation gates.

## Current Scope

The submission-factory scaffold provides:

- submission status enums and lightweight schema models
- database profile YAML and structural validation
- example CSV submission manifest and manifest validation
- `wsi-pipeline submit setup` for novice-facing batch summaries, mode checks, and rough estimates
- `wsi-pipeline submit preflight` for manifest/profile/path checks
- `wsi-pipeline submit validate-ometiff` for structural-only existing OME-TIFF batch checks
- `wsi-pipeline submit package-ometiff --dry-run` for planning existing OME-TIFF package contents without copying, linking, or uploading files
- `wsi-pipeline submit plan-tissues` for lower-level state-only tissue-detection dry runs
- documentation for the intended workflow and review roles
- tests for the scaffold, preflight, setup, and tissue-planning layers

The current submission commands do not implement VSI/ETS reading, pixel
inspection, tissue detection, thresholding, connected components, cropping,
OME-TIFF writing, batch conversion orchestration, real upload packaging, a
dashboard, notebooks, QuPath integration, napari integration, Slicer
integration, or Neuroglancer export.

## Expected Future Workflow

1. Create or select a submission batch.
2. Choose an input folder or submission manifest.
3. Run `submit setup` against a database profile, manifest, and explicit workflow mode.
4. Review setup ready, warning, deferred, and blocked states.
5. Use lower-level tissue-detection planning only when needed for future extraction work.
6. Detect tissue sections and generate QC overlays.
7. Approve, reject, defer, or escalate tissue sections.
8. Convert approved sections to single-tissue OME-TIFF derivatives.
9. Validate OME-TIFFs, sidecars, checksums, and provenance.
10. Package upload-ready outputs.

Implemented commands:

```bash
wsi-pipeline submit setup --profile configs/database_profiles/national_database_ometiff.yaml --manifest examples/submission_factory/example_submission_manifest.csv --mode extract-convert-upload --setup-report setup_report.json
wsi-pipeline submit preflight --profile configs/database_profiles/national_database_ometiff.yaml --manifest examples/submission_factory/example_submission_manifest.csv
wsi-pipeline submit validate-ometiff --profile configs/database_profiles/national_database_ometiff.yaml --manifest existing_ometiff_manifest.csv --validation-report ometiff_structural_report.json
wsi-pipeline submit package-ometiff --profile configs/database_profiles/national_database_ometiff.yaml --manifest existing_ometiff_manifest.csv --output-dir package_dry_run --dry-run
wsi-pipeline submit plan-tissues --state preflight_state.json --plan-out tissue_detection_plan.json
```

`submit setup` is the usual operator entry point. It runs preflight in
memory, checks whether the selected mode matches input extensions, sums
known local file sizes, and estimates output size, processing time, and
upload time from simple workflow constants. Supported setup modes are
`existing-ometiff-upload`, `convert-single-tissue`, and
`extract-convert-upload`. Non-local or missing sizes keep known local bytes
visible but make full-batch estimates unavailable. `validate-ometiff` is a
filesystem/manifest-only structural check for existing OME-TIFF-like files; it
does not open TIFF files, parse OME-XML, inspect pixels, compute checksums,
package uploads, or upload to a database. `package-ometiff --dry-run` consumes
that structural readiness and writes `package_plan.json`, `package_manifest.csv`,
and `package_summary.txt` showing planned package names, blockers, warnings, and
row-specific deferred checks. It does not copy, symlink, hardlink, upload, read
image pixels, parse TIFF headers, parse OME-XML, or compute checksums. Actual
copy/link/upload behavior remains planned for a later PR. `plan-tissues` remains
a lower-level dry run for future tissue detection internals.

Later commands remain planned:

```bash
wsi-pipeline submit detect-tissues
wsi-pipeline submit review
wsi-pipeline submit convert
wsi-pipeline submit validate
wsi-pipeline submit package
```

## Design Principles

- Raw VSI/ETS files are read-only.
- Single-tissue OME-TIFFs split from a multi-tissue WSI are derivatives.
- Every derivative preserves parent-slide provenance.
- Missing physical pixel size blocks conversion unless explicitly repaired by an
  expert in a future workflow.
- Database profiles define required metadata, sidecar policy, naming templates,
  QC gates, and validation gates.
- Operators should make review decisions from profile-driven states, not from
  low-level conversion settings.
