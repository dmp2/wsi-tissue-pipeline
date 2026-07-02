# Operator Guide

This guide describes the planned operator workflow for the Batch OME-TIFF
Submission Factory. PR 1 provides contracts and validation helpers only; the
commands below are planned for a later PR and are not implemented yet.

## Starting Files

Operators start with read-only source data:

- one or more `.vsi` files
- matching Olympus ETS companion data when required by the database profile
- optional notes about specimen, slide, stain, block, or section number

The source files are never edited. Any single-tissue OME-TIFF produced later is
a derivative of its parent WSI.

## Submission Manifest

A manifest is a CSV that lists source slides for a batch. PR 1 supports these
columns:

- `specimen_id`
- `slide_id`
- `source_path`
- `stain`
- `block_id`
- `section_number`
- `notes`

Only `specimen_id`, `slide_id`, and `source_path` are required. Paths do not
need to exist during planning or tests unless path checking is explicitly
enabled.

## Preflight

Preflight is the planned first check before tissue detection or conversion. It
will compare the manifest and source metadata against the selected database
profile.

Planned command:

```bash
wsi-pipeline submit preflight
```

Preflight should eventually identify:

- ready slides that can proceed
- warning slides that need attention but may be repairable
- blocked slides that cannot proceed without expert or admin action

## Status Meanings

- `PASS`: the record satisfies the current gate.
- `WARNING`: the record may be usable but needs review or repair.
- `BLOCKED`: the record cannot proceed until the blocking issue is fixed.
- `NEEDS_EXPERT_REVIEW`: an expert must resolve the issue before conversion.
- `FAILED`: the attempted step did not complete successfully.

## Review Boundaries

Operators may eventually approve obvious pass cases with clear tissue masks and
complete metadata. Operators should ask an expert reviewer when:

- tissue masks are ambiguous
- tissue is split incorrectly or merged incorrectly
- physical pixel size or units are missing
- orientation is uncertain
- any required provenance field is missing
- a tissue section is in a warning or expert-review state

## Planned Commands

The submission CLI is planned for a later PR:

```bash
wsi-pipeline submit preflight
wsi-pipeline submit detect-tissues
wsi-pipeline submit review
wsi-pipeline submit convert
wsi-pipeline submit validate
wsi-pipeline submit package
```

These commands are listed here to document the intended workflow. They are not
available in PR 1.
