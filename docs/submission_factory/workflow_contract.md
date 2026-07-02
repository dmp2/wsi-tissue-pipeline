# Submission Factory Workflow Contract

## Goal

Convert large batches of multi-tissue-section VSI/ETS whole-slide images into
database-ready, single-tissue-section OME-TIFF derivatives with QC, validation,
and provenance.

## High-level workflow

1. Create submission batch.
2. Inspect input directory.
3. Match VSI files to ETS directories.
4. Validate image readability and metadata.
5. Estimate output size and required storage.
6. Detect tissue sections.
7. Generate tissue-section QC overlays.
8. Require operator or expert approval.
9. Export each approved tissue section as an OME-TIFF derivative.
10. Validate OME-TIFF metadata and sidecars.
11. Compute checksums.
12. Emit upload manifest.
13. Emit HTML and JSON validation reports.
14. Mark batch as upload-ready only when all required gates pass.

## Required batch-level outputs

- `upload_manifest.csv`
- `upload_manifest.json`
- `checksums.sha256`
- `validation_report.html`
- `validation_report.json`
- `qc_gallery.html`
- `review_decisions.csv`
- `failed_cases.csv`
- `rerun_config.yaml`
- `batch_provenance.json`

## Required tissue-level outputs

For each approved tissue section:

- one `.ome.tif`
- one metadata sidecar, if required by the database profile
- one QC thumbnail or overlay
- one provenance record or one row in a tissue provenance table

## Required tissue provenance fields

Each single-tissue-section OME-TIFF derivative must record:

- parent source image path or identifier
- parent source checksum
- parent VSI/ETS association
- tissue section ID
- crop bounds in parent pixel coordinates
- crop bounds in parent physical coordinates, when available
- output pixel size and units
- output array-to-physical transform
- any flip, rotation, or resampling applied
- segmentation method and config hash
- approval status
- reviewer identity or reviewer note, if available
- conversion software version
- conversion timestamp
- output checksum

## Blocking conditions

Conversion must be blocked when:

- source image cannot be read
- matching ETS data are missing when required
- physical pixel size is missing
- tissue section has no parent-coordinate mapping
- output storage is insufficient
- database-required metadata are missing
- OME-TIFF validation fails
- sidecar metadata conflict with OME metadata
- tissue section has not been approved
