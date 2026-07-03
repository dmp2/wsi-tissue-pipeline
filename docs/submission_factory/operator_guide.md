# Operator Guide

This guide describes the operator workflow for the Batch OME-TIFF Submission
Factory. The first executable command is `wsi-pipeline submit preflight`, which
checks a database profile and submission manifest before image processing.

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

Only `specimen_id`, `slide_id`, and `source_path` are required. Preflight also
recognizes optional source metadata columns such as `checksum`,
`physical_pixel_size_x`, `physical_pixel_size_y`, and
`physical_pixel_size_unit` when profiles explicitly require them at manifest
time.

## Preflight

Preflight is the first check before tissue detection or conversion. It answers:

> Is this manifest/profile structurally valid enough to proceed to later
> inspection stages?

Example command:

```bash
wsi-pipeline submit preflight \
  --profile configs/database_profiles/national_database_ometiff.yaml \
  --manifest examples/submission_factory/example_submission_manifest.csv \
  --json-report preflight_report.json \
  --state-out preflight_state.json
```

Preflight checks profile loading, manifest loading, required manifest fields,
row model validation, source extension policy, and local source path existence.
Non-local source paths are not existence-checked in this PR; they are reported
as deferred source-file checks. Missing source metadata such as pixel size,
units, OME metadata, or source checksum may be reported as deferred unless the
profile explicitly marks that requirement as `preflight_manifest`.

Preflight does not answer whether images are scientifically valid, whether
tissue crops are correct, or whether OME-TIFF derivatives are ready for upload.
It does not read image pixels, parse OME-XML from image files, compute image
checksums, run tissue detection, crop sections, convert images, or create upload
packages.

Exit code is zero when there are no error-severity findings. With `--strict`,
warnings or deferred requirements also return nonzero, but their severity is
preserved in the JSON report. The JSON report is the detailed machine-readable
record; the state file is a draft preflight state for later workflow stages and
does not imply conversion or upload readiness.

## Tissue-Detection Planning

`wsi-pipeline submit plan-tissues` consumes the preflight state JSON and writes a
deterministic dry-run plan for future local tissue detection jobs. It classifies
rows independently, so an eligible local row can still be planned even when a
different row has a preflight error.

Example command:

```bash
wsi-pipeline submit plan-tissues \
  --state preflight_state.json \
  --plan-out tissue_detection_plan.json
```

The plan lists future `tissue_detection` jobs for local or `file://` source rows
with no error-severity preflight issues. Rows blocked by preflight errors appear
in `blocked_rows`. Non-local URI rows are skipped for this local planning pass,
not treated as missing local files. Deferred metadata requirements remain
attached to otherwise eligible planned jobs.

This command does not read image pixels, inspect masks, count tissue sections,
crop images, convert OME-TIFFs, validate OME-XML from image files, compute image
checksums, or prepare upload packages. The expected next stage is a future tissue
detection command that consumes planned jobs and creates real detection outputs.

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

Available now:

```bash
wsi-pipeline submit preflight
wsi-pipeline submit plan-tissues
```

Planned for later workflow stages:

```bash
wsi-pipeline submit detect-tissues
wsi-pipeline submit review
wsi-pipeline submit convert
wsi-pipeline submit validate
wsi-pipeline submit package
```
