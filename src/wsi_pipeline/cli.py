"""
Command-Line Interface for WSI Tissue Pipeline

Provides CLI commands for processing, QC, and visualization.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import PipelineConfig, create_default_config, load_config

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="wsi-pipeline")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG) logging.",
)
@click.pass_context
def main(ctx: click.Context, verbose: bool):
    """
    WSI Tissue Pipeline - Whole-slide image processing for tissue extraction.

    Process large microscopy images into individual tissue sections
    with experiment tracking and quality control.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input WSI file path (.vsi, .ets, .jpg, .tiff, etc.)",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for tissue tiles",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file (YAML)",
)
@click.option(
    "--backend",
    type=click.Choice(["local-entropy", "local-otsu"]),
    default="local-entropy",
    help="Segmentation backend",
)
@click.option(
    "--mlflow",
    "mlflow_enabled",
    is_flag=True,
    help="Enable MLflow tracking (requires mlflow to be installed)",
)
def process(
    input_path: Path,
    output_dir: Path,
    config_path: Path | None,
    backend: str,
    mlflow_enabled: bool,
):
    """Process a single WSI file and extract tissue sections."""
    from .wsi_processing import process_wsi

    # Load config
    if config_path:
        config = load_config(config_path)
    else:
        config = PipelineConfig()

    # Override backend if specified
    config.segmentation.backend = backend

    # Enable MLflow only if explicitly requested
    if mlflow_enabled:
        config.mlflow.enabled = True

    console.print(f"[bold blue]Processing:[/] {input_path.name}")
    console.print(f"[bold blue]Output:[/] {output_dir}")

    # Process with optional MLflow tracking
    if config.mlflow.enabled:
        from .mlflow_utils import MLflowContext, log_processing_run

        with MLflowContext(
            run_name=f"process_{input_path.stem}",
            config=config,
        ):
            results = process_wsi(input_path, output_dir, config=config)
            log_processing_run(results)
    else:
        results = process_wsi(input_path, output_dir, config=config)

    # Display results
    console.print(f"\n[bold green]Extracted {len(results.get('output_paths', []))} tissue tiles")


@main.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input directory containing WSI files",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for tissue tiles",
)
@click.option(
    "--pattern",
    "-p",
    default="*.vsi",
    help="Glob pattern for input files (default: *.vsi)",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file (YAML)",
)
@click.option(
    "--mlflow",
    "mlflow_enabled",
    is_flag=True,
    help="Enable MLflow tracking (requires mlflow to be installed)",
)
def batch(
    input_dir: Path,
    output_dir: Path,
    pattern: str,
    config_path: Path | None,
    mlflow_enabled: bool,
):
    """Batch process all WSI files in a directory."""
    from .wsi_processing import process_specimen

    # Load config
    if config_path:
        config = load_config(config_path)
    else:
        config = PipelineConfig()

    # Enable MLflow only if explicitly requested
    if mlflow_enabled:
        config.mlflow.enabled = True

    files = list(input_dir.glob(pattern))
    console.print(f"[bold blue]Found {len(files)} files matching '{pattern}'")
    console.print(f"[bold blue]Output:[/] {output_dir}")

    # Process with optional MLflow tracking
    if config.mlflow.enabled:
        from .mlflow_utils import MLflowContext, log_processing_run

        with MLflowContext(
            run_name=f"batch_{input_dir.name}",
            config=config,
        ):
            results = process_specimen(
                input_dir,
                output_dir,
                config=config,
                pattern=pattern,
            )
            log_processing_run(results)
    else:
        results = process_specimen(
            input_dir,
            output_dir,
            config=config,
            pattern=pattern,
        )

    console.print(f"\n[bold green]Processed {results['n_inputs']} files")
    console.print(f"[bold green]Created {results['n_outputs']} tissue tiles")


@main.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input directory containing tissue images",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for QC grids",
)
@click.option(
    "--thumb-size",
    default=256,
    type=int,
    help="Thumbnail size in pixels",
)
@click.option(
    "--columns",
    default="auto",
    help="Number of columns ('auto' or integer)",
)
@click.option(
    "--no-master",
    is_flag=True,
    help="Don't create master contact sheet",
)
def qc(
    input_dir: Path,
    output_dir: Path,
    thumb_size: int,
    columns: str,
    no_master: bool,
):
    """Generate QC contact sheets for tissue images."""
    from .qc_grid import run_qc_workflow

    console.print(f"[bold blue]Building QC grids for:[/] {input_dir}")

    # Parse columns
    cols = "auto" if columns == "auto" else int(columns)

    result = run_qc_workflow(
        input_dir=input_dir,
        output_dir=output_dir,
        thumb_size=thumb_size,
        padding=1,
        columns=cols,
        label_mode="slice",
        backend="pil",
        write_master=not no_master,
        write_per_slide=True,
        write_stats=True,
    )

    n_outputs = len(result.artifacts.per_slide_grids) + int(
        result.artifacts.master_contact_sheet is not None
    )
    console.print(f"\n[bold green]Created {n_outputs} QC images")


@main.command()
@click.option(
    "--zarr-dir",
    "-z",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing OME-Zarr files",
)
@click.option(
    "--port",
    default=9999,
    type=int,
    help="Neuroglancer server port",
)
@click.option(
    "--http-port",
    default=8000,
    type=int,
    help="HTTP file server port",
)
def visualize(
    zarr_dir: Path,
    port: int,
    http_port: int,
):
    """Start Neuroglancer visualization server."""
    try:
        from .neuroglancer import start_neuroglancer_server

        console.print("[bold blue]Starting Neuroglancer server...")
        console.print(f"[bold blue]Zarr directory:[/] {zarr_dir}")
        console.print(f"[bold blue]Neuroglancer:[/] http://localhost:{port}")
        console.print(f"[bold blue]File server:[/] http://localhost:{http_port}")

        start_neuroglancer_server(
            zarr_dir,
            ng_port=port,
            http_port=http_port,
        )
    except ImportError:
        console.print("[bold red]Error:[/] Neuroglancer not installed.")
        console.print("Install with: pip install wsi-tissue-pipeline[visualization]")
        sys.exit(1)


@main.command()
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    help="Output path for config file",
)
def init_config(output_path: Path | None):
    """Create a default configuration file."""
    if output_path is None:
        output_path = Path("config.yaml")

    path = create_default_config(output_path)
    console.print(f"[bold green]Created configuration file:[/] {path}")


@main.command()
@click.option(
    "--tracking-uri",
    default="sqlite:///mlflow.db",
    help="MLflow tracking server URI",
)
@click.option(
    "--experiment",
    default="wsi-tissue-pipeline",
    help="Experiment name",
)
def init_tracking(tracking_uri: str, experiment: str):
    """Initialize MLflow tracking."""
    from .mlflow_utils import init_mlflow

    success = init_mlflow(tracking_uri=tracking_uri, experiment_name=experiment)

    if success:
        console.print("[bold green]MLflow initialized")
        console.print(f"[bold blue]Tracking URI:[/] {tracking_uri}")
        console.print(f"[bold blue]Experiment:[/] {experiment}")
    else:
        console.print("[bold red]Failed to initialize MLflow")
        console.print("Install with: pip install wsi-tissue-pipeline[mlflow]")
        sys.exit(1)


@main.command()
@click.option(
    "--strict",
    is_flag=True,
    help="Exit non-zero when Bio-Formats prerequisites are not ready.",
)
def doctor(strict: bool):
    """Check optional runtime dependencies and Java/JDK setup."""
    import os

    from .bioformats_runtime import (
        BIOFORMATS_DOWNLOAD_ENV,
        BIOFORMATS_JAR_ENV,
        _default_jar_path,
        _verify_managed_jar,
        discover_java_runtime,
    )

    failures: list[str] = []
    table = Table(title="WSI Pipeline Doctor")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    java_status = discover_java_runtime()
    if java_status["is_jdk"]:
        table.add_row("JDK", "[green]OK", f"JAVA_HOME={java_status['java_home']}")
        table.add_row("java", "[green]OK", str(java_status["java"]))
        table.add_row("javac", "[green]OK", str(java_status["javac"]))
        if java_status.get("has_libjvm"):
            table.add_row("libjvm", "[green]OK", str(java_status["jvm_path"]))
        else:
            table.add_row(
                "libjvm", "[red]Missing", "Set JVM_PATH or install a full OpenJDK package."
            )
            failures.append("libjvm")
    else:
        java_detail = str(java_status["java"] or "not found")
        javac_detail = str(java_status["javac"] or "not found")
        table.add_row("JDK", "[red]Missing", f"java={java_detail}; javac={javac_detail}")
        failures.append("JDK with javac")

    for module_name in ("jnius", "jnius_config"):
        if importlib.util.find_spec(module_name) is None:
            table.add_row(module_name, "[red]Missing", "Install with the bioformats extra.")
            failures.append(module_name)
        else:
            table.add_row(module_name, "[green]OK", "importable")

    jar_override = os.getenv(BIOFORMATS_JAR_ENV)
    jar_path = Path(jar_override).expanduser() if jar_override else _default_jar_path()
    if jar_path.exists():
        if jar_override:
            table.add_row("Bio-Formats jar", "[yellow]Custom", str(jar_path))
        else:
            try:
                _verify_managed_jar(jar_path)
            except RuntimeError as exc:
                table.add_row("Bio-Formats jar", "[red]Invalid", str(exc))
                failures.append("Bio-Formats jar")
            else:
                table.add_row("Bio-Formats jar", "[green]OK", str(jar_path))
    else:
        downloads_disabled = os.getenv(BIOFORMATS_DOWNLOAD_ENV, "").strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }
        status = "[red]Missing" if downloads_disabled else "[yellow]Not cached"
        detail = (
            f"{jar_path}; automatic downloads disabled"
            if downloads_disabled
            else f"{jar_path}; will download on first Bio-Formats use"
        )
        table.add_row("Bio-Formats jar", status, detail)
        if downloads_disabled:
            failures.append("Bio-Formats jar")

    console.print(table)

    if strict and failures:
        console.print(
            "[bold red]Doctor found missing prerequisites:[/] " + ", ".join(sorted(set(failures)))
        )
        sys.exit(1)


@main.command()
def info():
    """Display pipeline information and configuration."""
    from . import __version__

    table = Table(title="WSI Tissue Pipeline")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Version", __version__)
    table.add_row("Python", sys.version.split()[0])

    # Check optional dependencies
    try:
        import torch

        table.add_row("PyTorch", torch.__version__)
    except ImportError:
        table.add_row("PyTorch", "[red]Not installed")

    try:
        import mlflow

        table.add_row("MLflow", mlflow.__version__)
    except ImportError:
        table.add_row("MLflow", "[red]Not installed")

    try:
        if importlib.util.find_spec("neuroglancer") is None:
            raise ImportError
        table.add_row("Neuroglancer", "[green]Available")
    except ImportError:
        table.add_row("Neuroglancer", "[red]Not installed")

    console.print(table)


if __name__ == "__main__":
    main()
