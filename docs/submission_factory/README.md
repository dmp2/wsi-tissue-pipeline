# Batch OME-TIFF Submission Factory

This module defines a novice-facing workflow for preparing large batches of proprietary
VSI/ETS whole-slide images for national database upload as single-tissue-section OME-TIFF
derivatives.

The workflow is designed for operators who should not need to write code. It provides:

1. dataset intake
2. preflight inspection
3. metadata validation
4. tissue-section detection
5. QC review
6. provenance-preserving tissue crop export
7. tiled OME-TIFF conversion
8. output validation
9. upload package generation

The central object is a submission batch, not an individual image viewer session.

## Intended user roles

### Operator

A student, technician, or clinician-facing user who can:

- create a submission batch
- choose input and output folders
- run preflight checks
- launch tissue detection
- review obvious QC pass/fail states
- run approved conversions
- export upload-ready packages

### Expert reviewer

A trained analyst who can:

- approve ambiguous tissue masks
- approve metadata overrides
- review orientation warnings
- approve blocked or warning-state cases
- finalize upload packages

### Admin

A maintainer who controls:

- database profiles
- naming templates
- metadata requirements
- compression and tiling defaults
- validation requirements
- server/storage paths

## Submission states

Each batch and tissue section should be assigned one of:

- `READY`
- `READY_WITH_WARNINGS`
- `NEEDS_EXPERT_REVIEW`
- `BLOCKED`
- `FAILED`
- `COMPLETE`

## Non-negotiable design principles

- Raw VSI/ETS files are read-only.
- Single-tissue OME-TIFFs created by splitting a WSI are derivatives.
- Every derivative must preserve parent-slide provenance.
- Missing physical pixel size blocks conversion unless explicitly repaired by an expert.
- OME-TIFF outputs must contain valid OME metadata.
- Sidecar metadata, if emitted, must be consistent with OME metadata.
- Every output must be traceable to source file, checksum, crop bounds, and processing config.
- Novices should interact with profiles and QC decisions, not low-level conversion parameters.
