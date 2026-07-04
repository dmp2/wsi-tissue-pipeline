# Operator Guide

This guide describes the operator workflow for the Batch OME-TIFF Submission
Factory. The usual first operator command is `wsi-pipeline submit setup`,
which checks a database profile, submission manifest, selected workflow mode,
and rough size/runtime/upload estimates before image processing.

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

## Setup Summary

`wsi-pipeline submit setup` is the novice-facing batch check. It answers:

> Did I point the tool at the right batch, what workflow mode is this, what is
> blocked or deferred, how large is it, and roughly how long will processing or
> upload take?

Example command:

```bash
wsi-pipeline submit setup \
  --profile configs/database_profiles/national_database_ometiff.yaml \
  --manifest examples/submission_factory/example_submission_manifest.csv \
  --mode extract-convert-upload \
  --setup-report setup_report.json \
  --upload-mbps 100
```

Supported modes are:

- `existing-ometiff-upload`: existing OME-TIFF inputs, with generic TIFF allowed
  only when the profile explicitly accepts it.
- `convert-single-tissue`: source microscopy files intended for conversion.
- `extract-convert-upload`: parent source WSI files intended for future tissue
  detection and extraction.

Setup reuses preflight checks, adds mode-specific input-extension compatibility
checks, sums local `Path.stat().st_size` values, and computes rough output-size,
processing-time, upload-time, and total-time ranges from fixed workflow
constants. If any row has a non-local or unknown size, the report still shows
known local bytes, but full-batch estimates are `null`. Extract-mode estimates
are coarse until tissue detection exists.

Setup writes only the optional setup report. It does not write a preflight state,
read image pixels, parse VSI/ETS, inspect OME-XML from image files, compute
checksums, detect tissue, threshold, find connected components, crop, convert,
upload, or launch GUI/viewer tools.

## Existing OME-TIFF Structural Check

`wsi-pipeline submit validate-ometiff` is for batches that already contain one
OME-TIFF-like file per tissue section and do not need tissue detection or
conversion first. It answers:

> Are the existing OME-TIFF-like files structurally ready for the next workflow
> action?

Example command:

```bash
wsi-pipeline submit validate-ometiff \
  --profile configs/database_profiles/national_database_ometiff.yaml \
  --manifest existing_ometiff_manifest.csv \
  --validation-report ometiff_structural_report.json
```

This is a filesystem and manifest structural check only. Its JSON report records
`validation_scope: "filesystem_and_manifest_only"`. It reuses setup/preflight
for profile loading, manifest loading, source extension policy, local path
existence, and deferred requirement reporting. It then checks only whether local
paths are regular files, whether file sizes are nonzero, and whether suffixes
match existing OME-TIFF upload mode. Generic `.tif` or `.tiff` inputs are
accepted only when the profile explicitly allows generic TIFF for
`existing-ometiff-upload`.

This command does not open TIFF files, parse TIFF headers, parse OME-XML, inspect
image pixels, validate OME metadata, compute checksums, create upload packages,
or upload to a database. When a profile requires OME-TIFF metadata validation,
that requirement is reported as deferred in this PR; the recommended next action
is future OME-TIFF metadata validation before packaging or upload.

Tests and smoke examples may use tiny nonzero fake `.ome.tiff` files as
OME-TIFF-like filesystem fixtures. Those fixtures validate command behavior only
and are not scientifically valid OME-TIFF images.

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

`wsi-pipeline submit plan-tissues` is a lower-level/internal dry-run command.
It consumes the preflight state JSON and writes a deterministic dry-run plan
for future local tissue detection jobs. It classifies
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
wsi-pipeline submit setup
wsi-pipeline submit preflight
wsi-pipeline submit validate-ometiff
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
