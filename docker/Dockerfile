# WSI Tissue Pipeline - Main Dockerfile
# Optimized for CPU-based processing with optional GPU support

# Build arguments
ARG PYTHON_VERSION=3.10
ARG BASE_IMAGE=python:${PYTHON_VERSION}-slim-bookworm
ARG EMLDDMM_GIT_URL=https://github.com/twardlab/emlddmm.git
ARG EMLDDMM_GIT_REF=main
ARG EMLDDMM_INSTALL_REQUIREMENTS=0
ARG APP_UID=1000
ARG APP_GID=1000

FROM ${BASE_IMAGE} as base
ARG APP_UID
ARG APP_GID

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    cmake \
    git \
    # Image processing libraries
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    # HDF5 for large data
    libhdf5-dev \
    # OpenSlide for WSI formats
    libopenslide-dev \
    openjdk-17-jdk-headless \
    # Networking
    curl \
    wget \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user that can be aligned with the host UID/GID for
# writable bind mounts in local Docker workflows.
RUN groupadd --gid "${APP_GID}" appuser \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /bin/bash appuser
WORKDIR /home/appuser

# ============================================
# Builder stage - install Python dependencies
# ============================================
FROM base as builder
ARG APP_UID
ARG APP_GID
ARG EMLDDMM_GIT_URL
ARG EMLDDMM_GIT_REF
ARG EMLDDMM_INSTALL_REQUIREMENTS

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV EMLDDMM_HOME=/opt/emlddmm

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Clone the upstream EM-LDDMM source tree. Its pinned requirements are optional
# because they target an older stack that can break modern image builds.
RUN git clone --depth 1 --branch "${EMLDDMM_GIT_REF}" "${EMLDDMM_GIT_URL}" "${EMLDDMM_HOME}" \
    && if [ "${EMLDDMM_INSTALL_REQUIREMENTS}" = "1" ]; then \
        pip install --no-build-isolation -r "${EMLDDMM_HOME}/requirements.txt"; \
    fi

# Copy only requirements first (for caching)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install package with all dependencies plus explicit notebook runtime deps.
RUN pip install \
    ".[all]" \
    "h5py>=3.10" \
    "nibabel>=5.0" \
    "jupyterlab>=4.0" \
    "ipywidgets>=8.0"

# ============================================
# Runtime stage - final image
# ============================================
FROM base as runtime
ARG APP_UID
ARG APP_GID

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/emlddmm /opt/emlddmm
ENV PATH="/opt/venv/bin:$PATH"
ENV EMLDDMM_HOME=/opt/emlddmm
ENV PYTHONPATH="/opt/emlddmm:${PYTHONPATH}"

# Copy application code
COPY --chown=appuser:appuser . /home/appuser/app
WORKDIR /home/appuser/app

# Create directories for data, outputs, and Jupyter state before switching
# to the non-root runtime user so mounted volumes inherit usable ownership.
RUN mkdir -p \
    /data \
    /output \
    /mlruns \
    /home/appuser/.jupyter \
    /home/appuser/.local/share/jupyter/runtime \
    && chown -R appuser:appuser \
        /data \
        /output \
        /mlruns \
        /home/appuser/.jupyter \
        /home/appuser/.local

# Switch to non-root user
USER appuser

# Environment variables for MLflow
ENV MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db
ENV MLFLOW_ARTIFACT_ROOT=/mlruns

# Default port for Jupyter
EXPOSE 8888
# Port for MLflow UI
EXPOSE 5000
# Port for Neuroglancer
EXPOSE 9999

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import wsi_pipeline; print('OK')" || exit 1

# Default command: start Jupyter Lab without auth prompts in the local Docker workflow
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--IdentityProvider.token="]

# ============================================
# Development stage - includes dev tools
# ============================================
FROM runtime as development

USER root

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install development dependencies
RUN pip install \
    ipdb \
    jupyter-server-proxy \
    jupyterlab-git

USER appuser

# Development command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--IdentityProvider.token="]
