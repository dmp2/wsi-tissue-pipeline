"""
Google Colab Setup Module

This module provides utilities for setting up the WSI Tissue Pipeline
in Google Colab environments. It handles:
- Package installation
- Google Drive mounting
- GPU detection and configuration
- MLflow setup

Usage in Colab:
    # Clone and install
    !git clone https://github.com/dmp2/wsi-tissue-pipeline.git
    %cd wsi-tissue-pipeline
    
    # Run setup
    from notebooks.colab_setup import setup_colab
    config = setup_colab()
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def is_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information if available."""
    gpu_info = {
        "available": False,
        "name": None,
        "memory_mb": None,
        "cuda_version": None,
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["name"] = torch.cuda.get_device_name(0)
            gpu_info["memory_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            gpu_info["cuda_version"] = torch.version.cuda
    except ImportError:
        # Try nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                gpu_info["available"] = True
                gpu_info["name"] = parts[0].strip()
                if len(parts) > 1:
                    mem_str = parts[1].strip().replace(" MiB", "")
                    gpu_info["memory_mb"] = int(mem_str)
        except FileNotFoundError:
            pass
    
    return gpu_info


def install_dependencies(
    use_gpu: bool = True,
    install_optional: bool = False,
    quiet: bool = False,
) -> None:
    """
    Install required dependencies.
    
    Parameters
    ----------
    use_gpu : bool
        Install GPU-accelerated packages (PyTorch with CUDA).
    install_optional : bool
        Install optional packages (neuroglancer, tiatoolbox).
    quiet : bool
        Suppress installation output.
    """
    quiet_flag = "-q" if quiet else ""
    
    # Install the package
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", quiet_flag],
        check=True,
    )
    
    # Install PyTorch with CUDA if GPU available
    if use_gpu:
        gpu_info = get_gpu_info()
        if gpu_info["available"]:
            print(f"GPU detected: {gpu_info['name']}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", 
                 "torch", "torchvision", "--index-url", 
                 "https://download.pytorch.org/whl/cu121", quiet_flag],
                check=True,
            )
    
    if install_optional:
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "neuroglancer", "tiatoolbox", quiet_flag],
            check=True,
        )


def mount_google_drive(mount_point: str = "/content/drive") -> Path:
    """
    Mount Google Drive in Colab.
    
    Parameters
    ----------
    mount_point : str
        Mount point for Google Drive.
    
    Returns
    -------
    Path
        Path to the mounted drive's MyDrive folder.
    """
    if not is_colab():
        raise RuntimeError("Google Drive mounting only works in Colab")
    
    from google.colab import drive
    drive.mount(mount_point)
    
    return Path(mount_point) / "MyDrive"


def setup_mlflow_colab(
    experiment_name: str = "wsi-tissue-pipeline",
    tracking_dir: Optional[Path] = None,
) -> str:
    """
    Setup MLflow for Colab environment.
    
    Parameters
    ----------
    experiment_name : str
        Name for the MLflow experiment.
    tracking_dir : Path, optional
        Directory for MLflow tracking. If None, uses ./mlruns.
    
    Returns
    -------
    str
        MLflow tracking URI.
    """
    import mlflow
    
    if tracking_dir is None:
        tracking_dir = Path.cwd() / "mlruns"
    
    tracking_uri = f"file://{tracking_dir}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    return tracking_uri


def setup_colab(
    mount_drive: bool = True,
    use_gpu: bool = True,
    install_optional: bool = False,
    mlflow_experiment: str = "wsi-tissue-pipeline",
    data_dir_name: str = "wsi-data",
) -> Dict[str, Any]:
    """
    Complete Colab setup.
    
    Parameters
    ----------
    mount_drive : bool
        Mount Google Drive.
    use_gpu : bool
        Use GPU if available.
    install_optional : bool
        Install optional dependencies.
    mlflow_experiment : str
        MLflow experiment name.
    data_dir_name : str
        Name of data directory in Google Drive.
    
    Returns
    -------
    dict
        Configuration dictionary with paths and settings.
    """
    print("=" * 60)
    print("WSI Tissue Pipeline - Colab Setup")
    print("=" * 60)
    
    config = {
        "is_colab": is_colab(),
        "gpu": get_gpu_info(),
        "paths": {},
        "mlflow": {},
    }
    
    # Print environment info
    print(f"\n📍 Environment: {'Google Colab' if config['is_colab'] else 'Local'}")
    
    if config["gpu"]["available"]:
        print(f"🎮 GPU: {config['gpu']['name']} ({config['gpu']['memory_mb']} MB)")
    else:
        print("💻 GPU: Not available (CPU mode)")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    install_dependencies(use_gpu=use_gpu, install_optional=install_optional, quiet=True)
    print("✅ Dependencies installed")
    
    # Mount Google Drive
    if mount_drive and config["is_colab"]:
        print("\n📂 Mounting Google Drive...")
        try:
            drive_path = mount_google_drive()
            config["paths"]["drive"] = drive_path
            config["paths"]["data"] = drive_path / data_dir_name
            config["paths"]["data"].mkdir(exist_ok=True)
            print(f"✅ Drive mounted at: {drive_path}")
            print(f"📁 Data directory: {config['paths']['data']}")
        except Exception as e:
            print(f"⚠️ Drive mount failed: {e}")
            config["paths"]["data"] = Path.cwd() / "data"
    else:
        config["paths"]["data"] = Path.cwd() / "data"
    
    # Ensure data directories exist
    config["paths"]["data"].mkdir(exist_ok=True)
    (config["paths"]["data"] / "input").mkdir(exist_ok=True)
    (config["paths"]["data"] / "output").mkdir(exist_ok=True)
    
    config["paths"]["input"] = config["paths"]["data"] / "input"
    config["paths"]["output"] = config["paths"]["data"] / "output"
    
    # Setup MLflow
    print("\n📊 Setting up MLflow...")
    mlruns_dir = config["paths"]["data"] / "mlruns"
    config["mlflow"]["tracking_uri"] = setup_mlflow_colab(
        experiment_name=mlflow_experiment,
        tracking_dir=mlruns_dir,
    )
    config["mlflow"]["experiment_name"] = mlflow_experiment
    print(f"✅ MLflow tracking: {config['mlflow']['tracking_uri']}")
    
    print("\n" + "=" * 60)
    print("✨ Setup complete!")
    print("=" * 60)
    
    # Print usage hints
    print("""
Next steps:
1. Upload your VSI/ETS files to: {input_dir}
2. Run the pipeline:
   
   from wsi_pipeline import process_specimen
   
   results = process_specimen(
       input_path="{input_dir}",
       output_path="{output_dir}",
   )

3. View results in MLflow:
   
   !mlflow ui --port 5000 &
   # Then click the URL to open MLflow UI
""".format(
        input_dir=config["paths"]["input"],
        output_dir=config["paths"]["output"],
    ))
    
    return config


def create_sample_notebook() -> str:
    """Generate a sample notebook as a string."""
    return '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# WSI Tissue Pipeline - Quick Start\\n", "\\n", "This notebook demonstrates the WSI processing pipeline."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": ["# Clone and setup\\n", "!git clone https://github.com/dmp2/wsi-tissue-pipeline.git\\n", "%cd wsi-tissue-pipeline\\n", "\\n", "from notebooks.colab_setup import setup_colab\\n", "config = setup_colab()"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": ["# Process a specimen\\n", "from wsi_pipeline import process_specimen\\n", "\\n", "results = process_specimen(\\n", "    input_path=config['paths']['input'],\\n", "    output_path=config['paths']['output'],\\n", ")"]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''


if __name__ == "__main__":
    # When run directly, perform setup
    config = setup_colab()
    print(f"\nConfiguration: {config}")
