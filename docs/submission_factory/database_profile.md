# Database Profile Guide

A database profile is a YAML contract that describes what a target database
requires from a submission batch. PR 1 includes a default profile at
`configs/database_profiles/national_database_ometiff.yaml`.

## What The Profile Controls

The profile defines:

- accepted input extensions
- whether ETS companion data are required
- whether raw files must be read-only
- output mode and extension
- OME-TIFF structural expectations
- required metadata fields
- QC and review gates
- output naming template
- validation gates

PR 1 validates this structure only. It does not validate real OME-TIFF files,
read VSI/ETS metadata, or write outputs.

## Required Metadata

The default profile requires physical pixel size, physical units, parent source
checksum, parent crop bounds in pixels, and child array-to-physical transform.
Missing physical pixel size should block future conversion unless an expert
records an explicit repair.

## Output Naming

The default naming template is:

```text
sub-{specimen_id}_slide-{slide_id}_tissue-{tissue_id}.ome.tif
```

Profiles must include these placeholders:

- `{specimen_id}`
- `{slide_id}`
- `{tissue_id}`

This keeps each output name traceable to the submission records.

## Sidecar Policy

The default profile requires JSON sidecars, QC PNGs, per-tissue provenance,
batch manifests, and checksums. Sidecars must remain consistent with OME
metadata in a future validation workflow.

## Validation Gates

The default profile asks future workflows to validate:

- OME-TIFF metadata
- sidecar consistency
- checksums
- required output success for all approved tissues

PR 1 stores and structurally validates these gate settings. Operational
validation belongs to later implementation PRs.

## Creating A New Profile

Admins can create a new profile by copying the default YAML and changing the
database-specific values. A valid profile should keep the required top-level
sections and required keys, use lowercase file extensions beginning with `.`,
and include all required naming placeholders.
