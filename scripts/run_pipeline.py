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
    step5 (reconstruct)   EM-LDDMM registration / reconstruction
    all                   Run steps 1 → 2 → 4 sequentially

Usage:
    python run_pipeline.py step1 -i /data/wsi -o /data/tiles --pattern "*.jpg"
    python run_pipeline.py step2 -o /data/tiles
    python run_pipeline.py step3 -o /data/zarr_plate
    python run_pipeline.py step4 -o /data/tiles --dv 35.05 35.05 16.0
    python run_pipeline.py step5 --dataset-root /data/tiles
    python run_pipeline.py all   -i /data/wsi -o /data/tiles -c configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
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


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


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
# Step 5: Reconstruct
# ---------------------------------------------------------------------------

def step5_reconstruct(args, config) -> int:
    """Run the notebook-derived EM-LDDMM workflow."""
    from wsi_pipeline.registration import run_emlddmm_workflow
    from wsi_pipeline.registration.orientation import (
        list_valid_orientation_codes,
        orientation_validation_rule,
    )

    _banner("Step 5: EM-LDDMM Reconstruction")

    if getattr(args, "list_orientations", False):
        print(orientation_validation_rule())
        print()
        print("Valid backend orientation codes:")
        print(" ".join(list_valid_orientation_codes()))
        print()
        print("These codes are passed to emlddmm.orientation_to_orientation.")
        return 0

    dataset_root = getattr(args, "dataset_root", None)
    legacy_output = getattr(args, "output", None)
    if dataset_root is None and legacy_output is None:
        raise ValueError("step5 requires --dataset-root or legacy --output")
    if (
        dataset_root is not None
        and legacy_output is not None
        and dataset_root.resolve() != legacy_output.resolve()
    ):
        raise ValueError("--dataset-root and --output must match when both are provided")
    dataset_root = (dataset_root or legacy_output).resolve()
    used_legacy_output_alias = legacy_output is not None and dataset_root is not None and getattr(args, "dataset_root", None) is None
    if used_legacy_output_alias:
        logging.getLogger(__name__).warning(
            "Deprecated --output alias was used for step5; prefer --dataset-root."
        )

    target_source = getattr(args, "target_source", None) or dataset_root
    registration_output = getattr(args, "registration_output", None)
    if registration_output is None:
        registration_output = dataset_root / "emlddmm"

    print(f"Prepared dataset:    {dataset_root}")
    print(f"Target source:       {target_source}")
    print(f"Target source mode:  {args.target_source_format}")
    print(f"Registration output: {registration_output}")
    print(f"Atlas mode:          {'atlas_to_target' if args.atlas else 'atlas_free'}")
    print(f"Dry run:             {'yes' if args.dry_run else 'no'}")
    if args.atlas:
        print(f"Atlas:               {args.atlas}")
    if args.label:
        print(f"Labels:              {args.label}")
    print()

    result = run_emlddmm_workflow(
        dataset_root=dataset_root,
        output_dir=args.output,
        target_source=target_source,
        target_source_format=args.target_source_format,
        registration_output=registration_output,
        atlas=args.atlas,
        label=args.label,
        emlddmm_config=args.emlddmm_config,
        preset=args.preset,
        init_affine=args.init_affine,
        orientation_from=args.orientation_from,
        orientation_to=args.orientation_to,
        atlas_unit_scale=args.atlas_unit_scale,
        target_unit_scale=args.target_unit_scale,
        device=args.device,
        precomputed_manifest=args.precomputed_manifest,
        upsample_between_slices=args.upsample_between_slices,
        upsample_mode=args.upsample_mode,
        skip_self_alignment=args.skip_self_alignment,
        run_transformation_graph=args.run_transformation_graph,
        transformation_graph_script=args.transformation_graph_script,
        write_notebook_bundle=args.write_notebook_bundle,
        write_qc_report=args.write_qc_report,
        used_legacy_output_alias=used_legacy_output_alias,
        original_cli_argv=list(sys.argv[1:]),
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    resolved_plan = getattr(result, "resolved_plan", None)
    print(f"Mode:         {result.mode}")
    if resolved_plan is not None:
        print(f"Backend:      {resolved_plan.backend_name}")
        print(f"Resolution:   {resolved_plan.working_resolution_um} um")
        print(
            f"Unit scales:  atlas={resolved_plan.atlas_unit_scale} "
            f"target={resolved_plan.target_unit_scale}"
        )
        print(f"Atlas init:   {resolved_plan.atlas_init_mode}")
        enabled = ", ".join(resolved_plan.enabled_stages) or "none"
        skipped = ", ".join(
            f"{name} ({reason})" for name, reason in resolved_plan.skipped_stages.items()
        ) or "none"
        print(f"Enabled:      {enabled}")
        print(f"Skipped:      {skipped}")
    print(f"Plan JSON:    {result.plan_path}")
    print(f"Output root:  {result.registration_output}")
    print(f"Summary JSON: {result.summary_path}")
    log_path = getattr(result, "log_path", None)
    if log_path is not None:
        print(f"Log file:     {log_path}")
    provenance_path = getattr(result, "provenance_path", None)
    if provenance_path is not None:
        print(f"Provenance:   {provenance_path}")
    replay_path = getattr(result, "reproduce_command_path", None)
    if replay_path is not None:
        print(f"Replay cmd:   {replay_path}")
    report_path = getattr(result, "report_path", None)
    if report_path is not None:
        print(f"QC report:    {report_path}")
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
    print("  - Run 'step5' for atlas-free reconstruction or atlas registration")
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
                         help="EM-LDDMM registration / reconstruction")
    p5.add_argument(
        "--dataset-root", type=Path, default=None,
        help="Prepared dataset root generated by earlier pipeline steps",
    )
    p5.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Deprecated alias for --dataset-root",
    )
    p5.add_argument("--emlddmm-config", type=Path, default=None,
                     help="Path to EM-LDDMM workflow JSON override")
    p5.add_argument("--atlas", type=Path, default=None,
                     help="Optional atlas image path; omit for atlas-free reconstruction")
    p5.add_argument("--label", type=Path, default=None,
                     help="Optional atlas label path (valid only with --atlas)")
    p5.add_argument("--target-source", type=Path, default=None,
                      help="Prepared-dir or precomputed target source (default: --dataset-root)")
    p5.add_argument("--target-source-format", type=str, default="auto",
                     choices=["auto", "prepared-dir", "precomputed"],
                     help="Target source format (default: auto)")
    p5.add_argument("--registration-output", type=Path, default=None,
                       help="Registration output root (default: <dataset-root>/emlddmm)")
    p5.add_argument("--preset", type=str, default="macaque-notebook",
                      help="Notebook-aligned workflow defaults (default: macaque-notebook)")
    p5.add_argument("--init-affine", type=Path, default=None,
                      help="Initial 4x4 affine for atlas registration")
    p5.add_argument("--orientation-from", type=str, default=None,
                      help="Backend orientation code for the atlas volume when deriving the init affine")
    p5.add_argument("--orientation-to", type=str, default=None,
                       help="Backend orientation code for the target volume when deriving the init affine")
    p5.add_argument("--list-orientations", action="store_true",
                      help="Print the valid backend orientation codes and exit")
    p5.add_argument("--atlas-unit-scale", type=float, default=None,
                      help="Scale atlas axes into micrometers (default preset assumes atlas coordinates are in mm)")
    p5.add_argument("--target-unit-scale", type=float, default=None,
                      help="Scale target axes into micrometers (default preset assumes the target is already in um)")
    p5.add_argument("--device", type=str, default=None,
                      help="Execution device override: auto, cpu, cuda, or cuda:N")
    p5.add_argument("--precomputed-manifest", type=Path, default=None,
                      help="Manifest required for precomputed target input")
    p5.add_argument("--upsample-between-slices", action="store_true", default=None,
                     help="Enable optional between-slice filling on the self-aligned target")
    p5.add_argument("--upsample-mode", type=str, default=None,
                     choices=["seg", "img"],
                     help="Between-slice filling mode (default: preset or seg)")
    p5.add_argument("--skip-self-alignment", action="store_true",
                      help="Skip atlas-free self-alignment and use identity slice transforms")
    p5.add_argument("--run-transformation-graph", action="store_true",
                      help="Execute transformation_graph_v01.py from the external emlddmm package after atlas registration")
    p5.add_argument("--transformation-graph-script", type=Path, default=None,
                      help="Explicit path to transformation_graph_v01.py; overrides external emlddmm package discovery")
    p5.add_argument("--write-notebook-bundle", action="store_true",
                      help="Write a debug notebook-style bundle summarizing stage payloads")
    p5.add_argument("--write-qc-report", action="store_true",
                      help="Write registration_report.html and registration_report.json with QC images when present")
    p5.add_argument("--dry-run", action="store_true",
                      help="Resolve and validate the run plan without executing stages")

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

    _configure_logging(getattr(args, "verbose", False))
    config = _load_config(args)

    handler = DISPATCH.get(args.step)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args, config)


if __name__ == "__main__":
    sys.exit(main())
