# wsi_pipeline.submission

Submission-oriented workflow code for preparing database-upload-ready OME-TIFF
derivatives from large VSI/ETS whole-slide image batches.

This package should remain viewer-agnostic. GUI apps, QuPath adapters, napari
plugins, and CLI tools should call into this workflow layer rather than
duplicating submission logic.
