# Submission Factory Workflow Contract

This contract defines the submission-batch states, required records, provenance
fields, and validation gates for the future Batch OME-TIFF Submission Factory.
PR 1 implements schemas and structural loaders only; it does not execute the
workflow.

## High-Level Workflow

1. Create a `SubmissionBatch` from an input folder or manifest.
2. Load a database profile that defines input, output, metadata, QC, naming, and
   validation requirements.
3. Run setup checks on the manifest, profile, selected mode, and local file sizes.
4. Run lower-level local tissue-detection planning when future extraction work needs it.
5. Detect tissue sections and write tissue-section records.
6. Generate QC overlays for detected tissue sections.
7. Collect operator or expert review decisions.
8. Convert approved tissue sections to OME-TIFF derivatives.
9. Validate OME-TIFFs, sidecars, checksums, and provenance records.
10. Mark the batch upload-ready only when all required gates pass.

Steps 3 and 4 now have executable planning commands. Steps 5 through 10 remain
planned future work.

## Setup Report Output

The `setup` command is the novice-facing summary for a batch. It runs
preflight in memory, checks whether the explicit workflow mode matches input
extensions, records blocking and deferred findings, sums known local file
sizes, and estimates output size, processing time, upload time, and total
time from fixed workflow constants. The durable artifact is the optional
setup JSON report. Setup does not write preflight state files.

Mode compatibility is extension-only in this PR: existing OME-TIFF upload
expects OME-TIFF or profile-allowed generic TIFF inputs; conversion and
extract workflows expect profile-allowed source microscopy or parent WSI
inputs. Wrong-mode findings are error-severity setup issues. Non-local or
unknown source sizes keep known local bytes visible but make full-batch
estimates unavailable.

Setup does not read image pixels, parse VSI/ETS, inspect OME-XML from image
files, compute checksums, detect tissue, threshold, find connected
components, crop, convert OME-TIFFs, package uploads, or integrate with
notebooks, QuPath, napari, Neuroglancer, or other viewers.

## Tissue-Detection Plan Output

The lower-level `plan-tissues` dry run consumes a preflight state file and emits a JSON plan
that records which rows would become future local `tissue_detection` jobs, which
rows are blocked by preflight errors, and which non-local rows are skipped for
this local planning pass. Deferred row requirements remain attached to planned
jobs, but the plan does not assert tissue locations, tissue counts, mask quality,
crop validity, OME-TIFF readiness, or upload readiness.

## Batch-Level Outputs

Future upload-ready batches should contain:

- upload manifest in CSV or JSON form
- checksum file for required outputs
- batch-level validation report
- review-decision record
- failed-case report when any records fail
- batch provenance record
- rerun or resume configuration

PR 1 models the records that make these outputs possible; it does not write an
upload package.

## Tissue-Level Outputs

Each approved tissue section should eventually produce:

- one single-tissue-section `.ome.tif`
- one metadata sidecar when required by the database profile
- one QC thumbnail or overlay
- one tissue-level provenance record

Single-tissue-section OME-TIFFs created by splitting a multi-tissue parent WSI
are derivatives and must be treated as such.

## Required Provenance

Each derivative record must preserve:

- parent source path or parent source identifier
- parent source checksum when required
- specimen, slide, and tissue identifiers
- crop bounds in parent pixel coordinates
- crop bounds in physical coordinates when available
- child array-to-physical transform
- flags for resampling, flipping, and rotation
- conversion profile name and version
- conversion configuration hash when available
- output checksum when available

Raw VSI/ETS files are read-only and must not be modified by the submission
workflow.

## Review States

Batch records use `BatchStatus`. Tissue records use `TissueStatus` and optional
`ReviewDecision` values. Operators may approve clear pass cases in a future
workflow. Warning cases, missing metadata repairs, ambiguous tissue masks, and
orientation uncertainty require expert review.

## Blocking Conditions

Future conversion must be blocked when:

- source image readability fails
- required ETS companion data are missing
- physical pixel size or units are missing
- parent-coordinate mapping is missing
- database-required metadata are missing
- tissue detection fails for a required slide
- tissue section has not been approved
- required output validation fails
- sidecar metadata conflict with OME metadata
- required checksums are missing or invalid

PR 1 only records these requirements in schemas, profile validation, and docs.

## Upload-Ready Criteria

A batch can be upload-ready only when:

- every required source slide has passed preflight
- every required tissue section has a final review decision
- every approved tissue has a validated derivative output
- all required sidecars, QC artifacts, manifests, and checksums are present
- all database-profile validation gates pass
- no blocking errors remain on the batch, slide, tissue, or derivative records
