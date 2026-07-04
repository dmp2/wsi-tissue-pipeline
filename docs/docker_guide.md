# Docker Guide

Best for development and reproducible batch runs — full control over environment
and dependencies. These instructions are pulled from the project README.

## Build and run

```bash
# Clone the repository
git clone https://github.com/dmp2/wsi-tissue-pipeline.git
cd wsi-tissue-pipeline

# Optional but recommended for local bind mounts:
# keep machine-specific settings in your untracked .env
cp .env.example .env
# then set DATA_DIR / OUTPUT_DIR to absolute host paths outside this repo,
# and set APP_UID / APP_GID from: id -u ; id -g

# Build and run with Docker Compose
docker compose -f docker/docker-compose.yml up --build

# Access services:
# - Jupyter Lab: http://localhost:8888
# - MLflow UI: http://localhost:5000
```

## Docker directory contract

| Host path in `.env` | Container path to use in notebooks/CLI | Purpose |
|---|---|---|
| `DATA_DIR=/absolute/path/outside/repo/wsi-data` | `/data` | Read-mostly input tree |
| `$DATA_DIR/input` | `/data/input` | Raw WSI/flat image inputs used by notebooks |
| `$DATA_DIR/resources` | `/data/resources` | Optional atlas, labels, configs, precomputed inputs, or jars |
| `OUTPUT_DIR=/absolute/path/outside/repo/wsi-output` | `/output` | Writable outputs and durable notebook artifacts |
| `$OUTPUT_DIR/tissue_sections` | `/output/tissue_sections` | Default notebook and staged-runner dataset root |

Create the host directories before starting Docker, for example:

```bash
mkdir -p /absolute/path/outside/repo/wsi-data/input
mkdir -p /absolute/path/outside/repo/wsi-data/resources
mkdir -p /absolute/path/outside/repo/wsi-output
```

Inside Docker, always pass the in-container paths (`/data/...`, `/output/...`) to
notebooks and CLI commands. Keeping `DATA_DIR` and `OUTPUT_DIR` outside the git
checkout avoids putting private slide data in the repository. If a bind-mount
source path is missing, Docker may create it on the host, sometimes with ownership
that your user cannot write.

## Notebook defaults in Docker

- `notebooks/01_wsi_to_tissue_sections.ipynb`, `02_quality_control.ipynb`, and
  `04_emlddmm_preparation.ipynb` use `/data` and `/output`.
- Notebook 01 auto-generates demo PNG inputs in `/data/input` when that directory
  is empty.
- Notebook 03 is separate from the TIFF tile workflow and auto-generates a tiny
  demo NGFF plate when `/output/per_tissue_ngff` is empty.
- Docker clones `https://github.com/twardlab/emlddmm.git`, adds it to `PYTHONPATH`,
  and installs the extra runtime dependencies notebook 04 needs without requiring
  the full pinned upstream requirements set.
- Jupyter starts in `/home/appuser/app`, and the repo notebooks are mounted there.
  Save durable executed notebooks, logs, and scratch files under
  `/output/notebook_runs` rather than in repo-internal paths.
- To smoke-test notebook 02 non-interactively after notebook 01 populates
  `/output/tissue_sections`, run:
  `docker compose -f docker/docker-compose.yml run --rm pipeline jupyter nbconvert --to notebook --execute /home/appuser/app/notebooks/02_quality_control.ipynb --output 02_quality_control.executed.ipynb --output-dir /output/notebook_runs`

## Local Docker notes

- `.env` is ignored by git, so keep host-specific values there rather than in
  tracked files.
- Set `APP_UID` and `APP_GID` locally when you want the container's non-root user
  to match your host user for writable bind mounts.
- `DATA_DIR` and `OUTPUT_DIR` are host paths. `/data` and `/output` are the
  matching in-container paths.
- The tracked Compose and Dockerfiles keep generic defaults and do not require
  committing personal UID/GID values.
