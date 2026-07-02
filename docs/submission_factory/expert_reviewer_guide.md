# Expert Reviewer Guide

Expert review is the planned escalation path for cases that an operator should
not resolve alone. PR 1 records review states and provenance requirements only;
it does not provide review UI, tissue detection, or conversion.

## When Expert Review Is Needed

Expert review is required when a source slide or tissue section has:

- ambiguous tissue boundaries
- missing or suspect physical pixel size
- missing physical units
- uncertain orientation
- warning-state segmentation results
- missing parent-coordinate mapping
- metadata that must be repaired before conversion

## Ambiguous Tissue Masks

Future tissue detection may split one parent WSI into multiple tissue-section
records. Expert reviewers should resolve cases where tissue sections are merged,
over-split, clipped, or confused with debris. Review decisions must be recorded
so future conversion can preserve why a derivative was approved or rejected.

## Missing Metadata

Missing physical pixel size is blocking for conversion unless an expert records
a repair in a future metadata workflow. Any repair must include who made the
decision, what value was applied, and why the value is trusted.

## Orientation Ambiguity

If flip or rotation corrections are needed, the derivative record must preserve
those operations. Orientation overrides should never be implicit because the
database consumer needs to understand how the child OME-TIFF maps back to the
parent WSI.

## Warning-State Tissue Sections

Warning-state tissue sections may still become valid derivatives, but only
after review. Expert reviewers can approve, reject, defer, or require additional
review. Overrides must be recorded with notes because they become part of the
submission provenance.

## Provenance Expectations

For every approved derivative, expert reviewers should expect traceability to:

- parent source path or identifier
- parent checksum when required
- parent pixel crop bounds
- physical crop bounds when available
- child array-to-physical transform
- flip, rotation, and resampling flags
- conversion profile and version
- output checksum when available
