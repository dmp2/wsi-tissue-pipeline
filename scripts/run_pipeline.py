#!/usr/bin/env python3
"""
WSI Tissue Pipeline — Staged Runner
====================================

Run the full pipeline or individual stages from the command line.
Each stage mirrors a notebook in ``notebooks/``.

Stages:
    step1 (segment)       WSI segmentation, tile extraction, renaming
    step2 (qc)            Quality-control contact sheets
    step3 (visualize)     Interactive Neuroglancer visualization
    step4 (emlddmm-prep)  EM-LDDMM metadata preparation
    step5 (reconstruct)   EM-LDDMM reconstruction (placeholder)
    all                   Run steps 1 → 2 → 4 sequentially

Usage:
    python run_pipeline.py step1 -i /data/wsi -o /data/tiles --pattern "*.jpg"
    python run_pipeline.py step2 -o /data/tiles
    python run_pipeline.py step3 -o /data/zarr_plate
    python run_pipeline.py step4 -o /data/tiles --dv 35.05 35.05 16.0
    python run_pipeline.py step5 -o /data/tiles
    python run_pipeline.py all   -i /data/wsi -o /data/tiles -c configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Allow direct execution without pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def _load_config(args):
    """Load PipelineConfig from --config or use defaults."""
    from wsi_pipeline.config import PipelineConfig, load_config

    if getattr(args, "config", None):
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        config = PipelineConfig()
        if getattr(args, "verbose", False):
            print("Using default configuration")

    # Apply CLI overrides
    if getattr(args, "backend", None):
        config.segmentation.backend = args.backend
    if getattr(args, "no_mlflow", False):
        config.mlflow.enabled = False
    if getattr(args, "verbose", False):
        config.verbose = True

    return config


# ---------------------------------------------------------------------------
# Step 1: Segment
# ---------------------------------------------------------------------------

def step1_segment(args, config) -> int:
    """WSI segmentation → tile extraction → global index renaming."""
    from wsi_pipeline.wsi_processing import process_wsi, process_specimen
    from wsi_pipeline.tiles.naming import rename_outputs_by_overall_index
    from wsi_pipeline.mlflow_utils import MLflowContext, log_processing_run

    _banner("Step 1: WSI Segmentation & Tile Extraction")

    input_path = args.input
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = args.pattern
    mlflow_enabled = config.mlflow.enabled

    print(f"Input:    {input_path}")
    print(f"Output:   {output_dir}")
    print(f"Pattern:  {pattern}")
    print(f"Backend:  {config.segmentation.backend}")
    print(f"MLflow:   {'enabled' if mlflow_enabled else 'disabled'}")
    print()

    # --- Process ---
    if input_path.is_dir():
        # Batch: use process_specimen() (wraps process_directory, adds specimen metadata)
        run_name = (
            f"batch_{input_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        if mlflow_enabled:
            with MLflowContext(run_name=run_name, config=config):
                results = process_specimen(
                    input_path, output_dir, config=config, pattern=pattern
                )
                log_processing_run(results)
        else:
            results = process_specimen(
                input_path, output_dir, config=config, pattern=pattern
            )
        print(f"Processed {results['n_inputs']} files -> {results['n_outputs']} tiles")
    else:
        # Single file: use process_wsi()
        run_name = f"process_{input_path.stem}"
        if mlflow_enabled:
            with MLflowContext(run_name=run_name, config=config):
                results = process_wsi(input_path, output_dir, config=config)
                log_processing_run(results)
        else:
            results = process_wsi(input_path, output_dir, config=config)
        n_tiles = len(results.get("output_paths", []))
        print(f"Created {n_tiles} tissue tiles")

    # --- Rename with global indices ---
    if not getattr(args, "no_rename", False):
        spacing = (
            args.spacing if getattr(args, "spacing", None) is not None
            else config.specimen_spacing
        )
        print(f"\nRenaming with global indices (spacing={spacing})...")
        renames = rename_outputs_by_overall_index(
            output_dir, pattern=pattern, spacing=spacing, pad=4, start=1
        )
        print(f"Renamed {len(renames)} files")

    print(f"\nOutput directory: {output_dir}")
    return 0


# ---------------------------------------------------------------------------
# Step 2: QC
# ---------------------------------------------------------------------------

def step2_qc(args, config) -> int:
    """Generate quality-control contact sheets."""
    from wsi_pipeline.qc_grid import build_qc_grids

    _banner("Step 2: Quality Control Contact Sheets")

    output_dir = args.output
    qc_dir = output_dir / "_qc_grids"

    thumb_size = getattr(args, "thumb_size", 256)
    columns = getattr(args, "columns", "auto")

    print(f"Image directory: {output_dir}")
    print(f"QC output:       {qc_dir}")
    print(f"Thumb size:      {thumb_size}")
    print()

    grid_paths = build_qc_grids(
        output_dir,
        qc_dir,
        thumb_size=thumb_size,
        columns=columns,
        create_master=True,
    )

    print(f"Generated {len(grid_paths)} QC grid(s) in {qc_dir}")
    return 0


# ---------------------------------------------------------------------------
# Step 3: Visualize
# ---------------------------------------------------------------------------

def step3_visualize(args, config) -> int:
    """Launch interactive Neuroglancer viewer."""
    from wsi_pipeline.neuroglancer import (
        open_neuroglancer_plate_view,
        open_neuroglancer_precomputed,
        stop_cors_server,
    )

    _banner("Step 3: Neuroglancer Visualization")

    output_dir = args.output
    fmt = getattr(args, "format", "zarr")
    ng_port = getattr(args, "ng_port", 9999)
    http_port = getattr(args, "http_port", 8000)

    if fmt == "precomputed":
        viewer, httpd = open_neuroglancer_precomputed(
            f"precomputed://file:///{output_dir}",
            mode="remote",
            http_host="localhost",
            http_port=http_port,
            ng_host="localhost",
            ng_port=ng_port,
        )
    else:
        viewer, httpd = open_neuroglancer_plate_view(
            output_dir,
            mode="remote",
            http_host="localhost",
            http_port=http_port,
            ng_host="localhost",
            ng_port=ng_port,
            visible_first_only=True,
        )

    print("\nViewer is running. Press Ctrl-C (or Enter) to stop.\n")
    try:
        # signal.pause() is not available on Windows
        import signal
        signal.pause()
    except (KeyboardInterrupt, AttributeError):
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            pass
    finally:
        stop_cors_server(httpd)
        print("Viewer stopped.")

    return 0


# ---------------------------------------------------------------------------
# Step 4: EM-LDDMM Preparation
# ---------------------------------------------------------------------------

def step4_emlddmm_prep(args, config) -> int:
    """Generate samples.tsv and JSON sidecar metadata for EM-LDDMM."""
    from wsi_pipeline.emlddmm_prep import nuke_json_sidecars, set_up_hist_for_emlddmm

    _banner("Step 4: EM-LDDMM Metadata Preparation")

    output_dir = args.output

    # Clean existing sidecars
    if getattr(args, "clean_sidecars", True):
        nuke_json_sidecars(output_dir)

    # Build dv (voxel sizes in micrometers, XYZ)
    dv = getattr(args, "dv", None)
    if dv is None:
        level = getattr(args, "pyramid_level", None)
        base_px = getattr(args, "base_pixel_size", None)
        dz = getattr(args, "dz", 16.0)
        if level is not None and base_px is not None:
            scale = base_px * pow(2, level)
            dv = [scale, scale, dz]
        else:
            print(
                "Error: provide --dv DX DY DZ  or  --pyramid-level + --base-pixel-size"
            )
            return 1

    species = getattr(args, "species", "Homo Sapiens")
    ext = getattr(args, "ext", "")
    space = getattr(args, "space", "right-inferior-posterior")

    histsetup_config = {
        "subject_dir": str(output_dir),
        "output_dir": str(output_dir),
        "species_name": species,
        "ext": ext,
        "slice_down": 1,
        "res_down": 1,
        "max_slice": None,
        "dv": dv,
        "space": space,
    }

    set_up_hist_for_emlddmm(histsetup_config)
    return 0


# ---------------------------------------------------------------------------
# Step 5: Reconstruct (placeholder)
# ---------------------------------------------------------------------------

def step5_reconstruct(args, config) -> int:
    """Placeholder for EM-LDDMM reconstruction (not yet implemented)."""
    _banner("Step 5: EM-LDDMM Reconstruction")

    print("This step is not yet implemented in the automated pipeline.\n")
    print("To run EM-LDDMM reconstruction manually:")
    print("  1. Ensure steps 1-4 have been completed")
    print("  2. Verify samples.tsv and JSON sidecars in the output directory")
    print("  3. Use the emlddmm package for registration:")
    print("       python -m emlddmm --config <config.json>")
    print()
    print("See: https://github.com/tward/emlddmm")

    emlddmm_config = getattr(args, "emlddmm_config", None)
    if emlddmm_config:
        print(f"\nNote: --emlddmm-config={emlddmm_config} was provided.")
        print("      This will be used when reconstruction is implemented.")

    return 0


# ---------------------------------------------------------------------------
# All: run steps 1 → 2 → 4
# ---------------------------------------------------------------------------

def run_all(args, config) -> int:
    """Run steps 1, 2, and 4 sequentially."""
    _banner("Full Pipeline: Steps 1 -> 2 -> 4")

    rc = step1_segment(args, config)
    if rc != 0:
        return rc

    rc = step2_qc(args, config)
    if rc != 0:
        return rc

    # Step 3 is interactive — skip in batch mode
    # Step 4 requires --dv or --pyramid-level; only run if provided
    dv = getattr(args, "dv", None)
    level = getattr(args, "pyramid_level", None)
    if dv is not None or level is not None:
        rc = step4_emlddmm_prep(args, config)
        if rc != 0:
            return rc
    else:
        print("\nStep 4 skipped (no --dv or --pyramid-level provided).")

    print(f"\n{'=' * 60}")
    print("  Pipeline complete (steps 1-2, optionally 4)")
    print(f"{'=' * 60}")
    print("\nNext steps:")
    print("  - Run 'step3' separately for interactive Neuroglancer visualization")
    print("  - Run 'step5' when EM-LDDMM reconstruction is available")
    return 0


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _add_common_io(parser, input_required=True):
    """Add -i/--input and -o/--output to a subparser."""
    if input_required:
        parser.add_argument(
            "-i", "--input", type=Path, required=True,
            help="Input WSI file or directory",
        )
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Output directory",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WSI Tissue Pipeline — staged runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-c", "--config", type=Path, default=None,
        help="Configuration file (YAML): default.yaml, colab.yaml, sciserver.yaml",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output",
    )

    sub = parser.add_subparsers(dest="step", help="Pipeline stage to run")

    # --- step1 / segment ---
    p1 = sub.add_parser("step1", aliases=["segment"],
                         help="WSI segmentation, tile extraction, renaming")
    _add_common_io(p1)
    p1.add_argument("--pattern", type=str, default="*.jpg",
                     help="Glob pattern for input files (default: *.jpg)")
    p1.add_argument("--backend", type=str, default=None,
                     choices=["local-entropy", "local-otsu", "tiatoolbox", "pathml"],
                     help="Segmentation backend override")
    p1.add_argument("--spacing", type=int, default=None,
                     help="Specimen spacing for renaming (default: from config)")
    p1.add_argument("--no-mlflow", action="store_true",
                     help="Disable MLflow tracking")
    p1.add_argument("--no-rename", action="store_true",
                     help="Skip global index renaming")

    # --- step2 / qc ---
    p2 = sub.add_parser("step2", aliases=["qc"],
                         help="Generate QC contact sheets")
    _add_common_io(p2, input_required=False)
    p2.add_argument("--thumb-size", type=int, default=256,
                     help="Thumbnail size in pixels (default: 256)")
    p2.add_argument("--columns", type=str, default="auto",
                     help="Grid columns: integer or 'auto' (default: auto)")

    # --- step3 / visualize ---
    p3 = sub.add_parser("step3", aliases=["visualize"],
                         help="Interactive Neuroglancer visualization")
    _add_common_io(p3, input_required=False)
    p3.add_argument("--format", type=str, default="zarr",
                     choices=["zarr", "precomputed"],
                     help="Data format (default: zarr)")
    p3.add_argument("--ng-port", type=int, default=9999,
                     help="Neuroglancer server port (default: 9999)")
    p3.add_argument("--http-port", type=int, default=8000,
                     help="HTTP file server port (default: 8000)")

    # --- step4 / emlddmm-prep ---
    p4 = sub.add_parser("step4", aliases=["emlddmm-prep"],
                         help="EM-LDDMM metadata preparation")
    _add_common_io(p4, input_required=False)
    p4.add_argument("--species", type=str, default="Homo Sapiens",
                     help="Species name for samples.tsv")
    p4.add_argument("--ext", type=str, default="",
                     help="Image extension to filter (auto-detect if empty)")
    p4.add_argument("--dv", type=float, nargs=3, default=None,
                     metavar=("DX", "DY", "DZ"),
                     help="Voxel size in micrometers (XYZ)")
    p4.add_argument("--pyramid-level", type=int, default=None,
                     help="Pyramid level (alternative to --dv)")
    p4.add_argument("--base-pixel-size", type=float, default=None,
                     help="Base pixel size in um (used with --pyramid-level)")
    p4.add_argument("--dz", type=float, default=16.0,
                     help="Z spacing in um (used with --pyramid-level, default: 16.0)")
    p4.add_argument("--space", type=str, default="right-inferior-posterior",
                     help="Anatomical coordinate space")
    p4.add_argument("--no-clean-sidecars", action="store_true",
                     dest="no_clean_sidecars",
                     help="Don't delete existing JSON sidecars before generating")

    # --- step5 / reconstruct ---
    p5 = sub.add_parser("step5", aliases=["reconstruct"],
                         help="EM-LDDMM reconstruction (placeholder)")
    _add_common_io(p5, input_required=False)
    p5.add_argument("--emlddmm-config", type=Path, default=None,
                     help="Path to EM-LDDMM JSON config file (future use)")

    # --- all ---
    pa = sub.add_parser("all", help="Run steps 1 -> 2 -> 4 sequentially")
    _add_common_io(pa)
    pa.add_argument("--pattern", type=str, default="*.jpg",
                     help="Glob pattern for input files (default: *.jpg)")
    pa.add_argument("--backend", type=str, default=None,
                     choices=["local-entropy", "local-otsu", "tiatoolbox", "pathml"],
                     help="Segmentation backend override")
    pa.add_argument("--spacing", type=int, default=None,
                     help="Specimen spacing for renaming (default: from config)")
    pa.add_argument("--no-mlflow", action="store_true",
                     help="Disable MLflow tracking")
    pa.add_argument("--no-rename", action="store_true",
                     help="Skip global index renaming")
    pa.add_argument("--thumb-size", type=int, default=256,
                     help="QC thumbnail size (default: 256)")
    pa.add_argument("--columns", type=str, default="auto",
                     help="QC grid columns (default: auto)")
    pa.add_argument("--dv", type=float, nargs=3, default=None,
                     metavar=("DX", "DY", "DZ"),
                     help="Voxel size in um (XYZ) for step 4")
    pa.add_argument("--pyramid-level", type=int, default=None,
                     help="Pyramid level for step 4")
    pa.add_argument("--base-pixel-size", type=float, default=None,
                     help="Base pixel size in um for step 4")
    pa.add_argument("--dz", type=float, default=16.0,
                     help="Z spacing in um for step 4 (default: 16.0)")
    pa.add_argument("--species", type=str, default="Homo Sapiens",
                     help="Species name for step 4")
    pa.add_argument("--ext", type=str, default="",
                     help="Image extension for step 4 (auto-detect if empty)")
    pa.add_argument("--space", type=str, default="right-inferior-posterior",
                     help="Anatomical coordinate space for step 4")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DISPATCH = {
    "step1": step1_segment,
    "segment": step1_segment,
    "step2": step2_qc,
    "qc": step2_qc,
    "step3": step3_visualize,
    "visualize": step3_visualize,
    "step4": step4_emlddmm_prep,
    "emlddmm-prep": step4_emlddmm_prep,
    "step5": step5_reconstruct,
    "reconstruct": step5_reconstruct,
    "all": run_all,
}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.step is None:
        parser.print_help()
        return 1

    # Normalize clean_sidecars flag for step4
    if hasattr(args, "no_clean_sidecars"):
        args.clean_sidecars = not args.no_clean_sidecars

    config = _load_config(args)

    handler = DISPATCH.get(args.step)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args, config)


if __name__ == "__main__":
    sys.exit(main())
