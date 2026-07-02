# Submission Factory Workflow Contract

This contract defines the submission-batch states, required records, provenance
fields, and validation gates for the future Batch OME-TIFF Submission Factory.
PR 1 implements schemas and structural loaders only; it does not execute the
workflow.

## High-Level Workflow

1. Create a `SubmissionBatch` from an input folder or manifest.
2. Load a database profile that defines input, output, metadata, QC, naming, and
   validation requirements.
3. Run preflight checks on source slides.
4. Mark the batch as ready, warning, or blocked for tissue detection.
5. Detect tissue sections and write tissue-section records.
6. Generate QC overlays for detected tissue sections.
7. Collect operator or expert review decisions.
8. Convert approved tissue sections to OME-TIFF derivatives.
9. Validate OME-TIFFs, sidecars, checksums, and provenance records.
10. Mark the batch upload-ready only when all required gates pass.

Steps 3 through 10 are planned future work.

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
