# Database Profile Guide

A database profile is a YAML contract that describes what a target database
requires from a submission batch. PR 1 includes a default profile at
`configs/database_profiles/national_database_ometiff.yaml`.

## What The Profile Controls

The profile defines:

- accepted input extensions and optional mode-specific extension policy
- whether ETS companion data are required
- whether raw files must be read-only
- output mode and extension
- OME-TIFF structural expectations
- required metadata fields
- QC and review gates
- output naming template
- validation gates

The profile loader validates this structure. Preflight uses the profile for
manifest/path checks, but it does not validate real OME-TIFF files, read VSI/ETS
metadata, or write outputs.


## Mode-Specific Input Extensions

Profiles may include `input.workflow_mode_extensions` to define input-extension
policy by submission workflow mode. The default profile uses this to allow
existing OME-TIFF upload rows to point at `.ome.tif` or `.ome.tiff` files while
conversion and extraction modes continue to use `.vsi` or `.ets` source files.

`input.accepted_extensions` remains the backward-compatible fallback for older
profiles and for commands that do not select a workflow mode. Generic `.tif` or
`.tiff` files are accepted for `existing-ometiff-upload` only when the profile
explicitly lists those generic TIFF suffixes for that mode.

## Requirement Phases

Profiles may optionally include a `requirement_phases` mapping that classifies
when each database requirement can be validated:

- `preflight_manifest`: must be present in the manifest now.
- `source_file_preflight`: can be checked from source path/URI information.
- `source_metadata_validation`: deferred until source metadata inspection.
- `derivative_export`: deferred until tissue detection, cropping, or conversion.
- `upload_validation`: deferred until final validation/package preparation.

Profiles without this mapping remain valid. Unknown phase names fail profile
validation with a clear error.

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

The profile stores and structurally validates these gate settings. Operational
validation beyond manifest/profile preflight belongs to later implementation PRs.

## Creating A New Profile

Admins can create a new profile by copying the default YAML and changing the
database-specific values. A valid profile should keep the required top-level
sections and required keys, use lowercase file extensions beginning with `.`,
and include all required naming placeholders.
