# WSI Tissue Pipeline - Main Dockerfile
# Optimized for CPU-based processing with optional GPU support

# Build arguments
ARG PYTHON_VERSION=3.11
ARG BASE_IMAGE=python:${PYTHON_VERSION}-slim-bookworm

FROM ${BASE_IMAGE} as base

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
    # Networking
    curl \
    wget \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser

# ============================================
# Builder stage - install Python dependencies
# ============================================
FROM base as builder

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy only requirements first (for caching)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install package with all dependencies
RUN pip install ".[all]"

# ============================================
# Runtime stage - final image
# ============================================
FROM base as runtime

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser . /home/appuser/app
WORKDIR /home/appuser/app

# Create directories for data and outputs
RUN mkdir -p /data /output /mlruns \
    && chown -R appuser:appuser /data /output /mlruns

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

# Default command: start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]

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
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
