# WSI Tissue Pipeline

A reproducible whole-slide imaging (WSI) pipeline: **VSI/ETS ingest → tissue
extraction → OME-Zarr + Neuroglancer export → EM-LDDMM 3D reconstruction.**

The pipeline processes large whole-slide images (potentially 0.5 TB+ per specimen)
through staged preprocessing and reconstruction, with quality-control artifacts,
provenance, and MLflow tracking at every stage.

## What it does

- **Convert** VSI/ETS Olympus data into flat image assets and metadata.
- **Segment** tissue, extract sections, and plate them for downstream analysis.
- **Write** OME-Zarr pyramids and optional Neuroglancer-precomputed volumes.
- **Prepare** the EM-LDDMM dataset root and manifest (`step4`).
- **Reconstruct** in 3D with the reproducible EM-LDDMM registration workflow
  (`step5`) — atlas-free self-alignment, optional atlas registration, optional
  between-slice upsampling, plus QC reports, logs, provenance, and replay artifacts.

## Key outputs

- **OME-Zarr pyramids** — multi-resolution NGFF for each tissue section.
- **Neuroglancer precomputed** — chunked volumes for browser-based 3D viewing.
- **QC contact sheets** — tissue-extraction grids and registration stage PNGs.
- **Registration reports** — `registration_report.html` / `.json`,
  `registration_summary.json`, and `resolved_run_plan.json`.
- **MLflow artifacts** — experiment logs, metrics, and tracked run artifacts.
- **Provenance / replay** — `run_provenance.json` and `reproduce_step5_command.txt`.

## Getting started tips

- New here? Start with the [Installation guide](installation.md), then skim
  [Configuration](configuration.md) for the YAML surface.
- Want zero local setup? See the [Colab guide](colab_guide.md).
- Reproducible local runs? See the [Docker guide](docker_guide.md).
- Running at JHU? See the [SciServer guide](sciserver_guide.md).
- Working on registration? See the [Pipeline overview](pipeline_overview.md) and
  [EM-LDDMM registration](emlddmm_registration.md).

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
configuration
```

```{toctree}
:maxdepth: 2
:caption: Deployment Guides

colab_guide
docker_guide
sciserver_guide
```

```{toctree}
:maxdepth: 2
:caption: Pipeline

pipeline_overview
emlddmm_registration
emlddmm_notebook_parity
ometiff_benchmark_runbook
source_level_0_production_runbook
```

```{toctree}
:maxdepth: 2
:caption: Examples

examples/index
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Project

contributing
credits
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
