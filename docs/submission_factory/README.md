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
- `wsi-pipeline submit preflight` for manifest/profile/path checks
- `wsi-pipeline submit plan-tissues` for state-only tissue-detection dry runs
- documentation for the intended workflow and review roles
- tests for the scaffold, preflight layer, and tissue-planning layer

The current submission commands do not implement VSI/ETS reading, pixel
inspection, tissue detection, OME-TIFF writing, batch conversion orchestration,
upload packaging, a dashboard, QuPath integration, napari integration, Slicer
integration, or Neuroglancer export.

## Expected Future Workflow

1. Create or select a submission batch.
2. Choose an input folder or submission manifest.
3. Run preflight checks against a database profile.
4. Review slide-level ready, warning, and blocked states.
5. Plan local tissue-detection jobs from preflight state.
6. Detect tissue sections and generate QC overlays.
7. Approve, reject, defer, or escalate tissue sections.
8. Convert approved sections to single-tissue OME-TIFF derivatives.
9. Validate OME-TIFFs, sidecars, checksums, and provenance.
10. Package upload-ready outputs.

Implemented commands:

```bash
wsi-pipeline submit preflight --profile configs/database_profiles/national_database_ometiff.yaml --manifest examples/submission_factory/example_submission_manifest.csv
wsi-pipeline submit plan-tissues --state preflight_state.json --plan-out tissue_detection_plan.json
```

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
