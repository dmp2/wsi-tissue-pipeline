"""Registration report helpers for the staged EM-LDDMM workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
import logging
from pathlib import Path
from typing import Any

from .config import EmlddmmResolvedPlan, SCHEMA_VERSION

logger = logging.getLogger(__name__)

_IMAGE_LIMIT = 24
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _relative_to_root(path: str | Path | None, root: Path) -> str | None:
    if path in (None, ""):
        return None
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(candidate)


def _relativize_nested(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _relativize_nested(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_relativize_nested(item, root) for item in value]
    if isinstance(value, Path):
        return _relative_to_root(value, root) or str(value)
    if isinstance(value, str):
        return _relative_to_root(value, root) or value
    return value


def _self_alignment_candidates(stage_dir: Path) -> list[Path]:
    return [
        stage_dir / "qc" / "input_target_overview.png",
        stage_dir / "qc" / "registered_target_overview.png",
        stage_dir / "qc" / "atlas_free_template_overview.png",
    ]


def _upsampling_candidates(stage_dir: Path) -> list[Path]:
    return [
        stage_dir / "filled_volume_overview.png",
        stage_dir / "nearest_slice_reference_overview.png",
    ]


def _discover_stage_images(stage_name: str, stage_dir: Path, root: Path) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if stage_name == "self_alignment":
        candidates = [path for path in _self_alignment_candidates(stage_dir) if path.exists()]
    elif stage_name == "upsampling":
        candidates = [path for path in _upsampling_candidates(stage_dir) if path.exists()]
    else:
        candidates = sorted(
            path
            for path in stage_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )
    total = len(candidates)
    selected = candidates[:_IMAGE_LIMIT]
    if total > _IMAGE_LIMIT:
        warnings.append(
            f"QC report truncated images for stage '{stage_name}' to {_IMAGE_LIMIT} of {total} files."
        )
    logger.debug(
        "Discovered %d QC images for stage=%s, selected=%d",
        total,
        stage_name,
        min(total, _IMAGE_LIMIT),
    )
    images = [
        {
            "path": _relative_to_root(path, root),
            "name": path.name,
            "caption": path.stem.replace("_", " "),
        }
        for path in selected
    ]
    gallery: dict[str, Any] = {
        "selected_count": len(images),
        "total_available_count": total,
        "truncated": total > len(images),
        "images": images,
    }
    if not images:
        gallery["note"] = "No reportable images produced for this stage."
    return gallery, warnings


def build_registration_report_manifest(
    *,
    registration_output: Path,
    plan: EmlddmmResolvedPlan,
    summary_payload: dict[str, Any],
    stage_results: dict[str, Any],
    artifacts: dict[str, Any],
    plan_path: Path,
    summary_path: Path,
    log_path: Path,
    provenance_payload: dict[str, Any],
    provenance_path: Path,
    reproduce_command_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Build a machine-readable report manifest from workflow outputs."""

    root = registration_output.resolve()
    warnings = list(summary_payload.get("warnings", []))
    report_warnings: list[str] = []
    stages: list[dict[str, Any]] = []

    for stage_name in ("self_alignment", "atlas_registration", "upsampling"):
        stage_dir = root / stage_name
        gallery, gallery_warnings = _discover_stage_images(stage_name, stage_dir, root)
        report_warnings.extend(gallery_warnings)
        stage_entry = {
            "name": stage_name,
            "status": stage_results.get(stage_name, {}).get("status", "unknown"),
            "reason": stage_results.get(stage_name, {}).get("reason"),
            "artifacts": _relativize_nested(artifacts.get(stage_name, {}), root),
            "gallery": gallery,
            "output_dir": _relative_to_root(stage_dir, root),
        }
        stages.append(stage_entry)

    warnings.extend(report_warnings)
    provenance_summary = {
        "pipeline_version": provenance_payload.get("pipeline", {}).get("version"),
        "git_short_sha": provenance_payload.get("git", {}).get("short_sha"),
        "backend_origin": provenance_payload.get("backend", {}).get("origin_type"),
        "transformation_graph_script_source": provenance_payload.get("backend", {}).get(
            "transformation_graph_script_source"
        ),
        "provenance_path": _relative_to_root(provenance_path, root),
        "reproduce_command_path": _relative_to_root(reproduce_command_path, root),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registration_output": root.name,
        "plan_path": _relative_to_root(plan_path, root),
        "summary_path": _relative_to_root(summary_path, root),
        "log_path": _relative_to_root(log_path, root),
        "provenance_path": _relative_to_root(provenance_path, root),
        "reproduce_command_path": _relative_to_root(reproduce_command_path, root),
        "warnings": warnings,
        "overview": {
            "mode": plan.mode,
            "backend_name": plan.backend_name,
            "backend_origin": provenance_summary["backend_origin"],
            "target_source": str(plan.target_source),
            "target_source_format": plan.target_source_format,
            "atlas_path": str(plan.atlas_path) if plan.atlas_path is not None else None,
            "label_path": str(plan.label_path) if plan.label_path is not None else None,
            "atlas_init_mode": plan.atlas_init_mode,
            "working_resolution_um": plan.working_resolution_um,
            "atlas_unit_scale": plan.atlas_unit_scale,
            "target_unit_scale": plan.target_unit_scale,
            "enabled_stages": list(plan.enabled_stages),
            "skipped_stages": dict(plan.skipped_stages),
            "transformation_graph_script": (
                str(plan.transformation_graph_script)
                if plan.transformation_graph_script is not None
                else None
            ),
            "transformation_graph_script_source": provenance_summary[
                "transformation_graph_script_source"
            ],
            "pipeline_version": provenance_summary["pipeline_version"],
            "git_short_sha": provenance_summary["git_short_sha"],
        },
        "provenance_summary": provenance_summary,
        "timings": dict(summary_payload.get("timings", {})),
        "stage_timeline": _relativize_nested(list(summary_payload.get("stage_timeline", [])), root),
        "artifacts": _relativize_nested(artifacts, root),
        "stages": stages,
    }
    logger.debug("Built QC report manifest with %d stage sections", len(stages))
    return manifest, report_warnings


def _html_list(items: list[str]) -> str:
    if not items:
        return "<p>None</p>"
    rows = "".join(f"<li>{escape(item)}</li>" for item in items)
    return f"<ul>{rows}</ul>"


def _render_stage(stage: dict[str, Any]) -> str:
    gallery = stage["gallery"]
    images_html = ""
    for image in gallery["images"]:
        path = image["path"]
        caption = image["caption"]
        images_html += (
            "<figure>"
            f"<img src=\"{escape(path)}\" alt=\"{escape(caption)}\" loading=\"lazy\">"
            f"<figcaption>{escape(caption)}</figcaption>"
            "</figure>"
        )
    if not images_html:
        images_html = f"<p>{escape(gallery.get('note', 'No images found.'))}</p>"
    reason_html = ""
    if stage.get("reason"):
        reason_html = f"<p><strong>Reason:</strong> {escape(str(stage['reason']))}</p>"
    return (
        "<section class=\"stage\">"
        f"<h3>{escape(stage['name'])}</h3>"
        f"<p><strong>Status:</strong> {escape(str(stage['status']))}</p>"
        f"{reason_html}"
        f"<p><strong>Gallery:</strong> {gallery['selected_count']} shown"
        f" / {gallery['total_available_count']} available</p>"
        f"<div class=\"gallery\">{images_html}</div>"
        "</section>"
    )


def render_registration_report_html(manifest: dict[str, Any]) -> str:
    """Render the registration report as a small standalone HTML page."""

    overview = manifest["overview"]
    warnings_html = _html_list([str(item) for item in manifest.get("warnings", [])])
    stages_html = "".join(_render_stage(stage) for stage in manifest["stages"])
    skipped = [
        f"{name}: {reason}"
        for name, reason in manifest["overview"].get("skipped_stages", {}).items()
    ]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>EM-LDDMM Registration Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #222; }}
    h1, h2, h3 {{ margin-bottom: 0.4rem; }}
    section {{ margin-bottom: 2rem; }}
    .gallery {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }}
    figure {{ margin: 0; border: 1px solid #ddd; padding: 0.75rem; background: #fafafa; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    figcaption {{ margin-top: 0.5rem; font-size: 0.9rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #ddd; padding: 0.5rem; vertical-align: top; }}
    code {{ background: #f2f2f2; padding: 0.1rem 0.25rem; }}
  </style>
</head>
<body>
  <h1>EM-LDDMM Registration Report</h1>
  <section>
    <h2>Overview</h2>
    <table>
      <tr><th>Mode</th><td>{escape(str(overview['mode']))}</td></tr>
      <tr><th>Backend</th><td>{escape(str(overview['backend_name']))}</td></tr>
      <tr><th>Backend Origin</th><td>{escape(str(overview.get('backend_origin')))}</td></tr>
      <tr><th>Target Source</th><td>{escape(str(overview['target_source']))}</td></tr>
      <tr><th>Target Format</th><td>{escape(str(overview['target_source_format']))}</td></tr>
      <tr><th>Pipeline Version</th><td>{escape(str(overview.get('pipeline_version')))}</td></tr>
      <tr><th>Git SHA</th><td>{escape(str(overview.get('git_short_sha')))}</td></tr>
      <tr><th>Working Resolution (um)</th><td>{escape(str(overview['working_resolution_um']))}</td></tr>
      <tr><th>Atlas Unit Scale</th><td>{escape(str(overview['atlas_unit_scale']))}</td></tr>
      <tr><th>Target Unit Scale</th><td>{escape(str(overview['target_unit_scale']))}</td></tr>
      <tr><th>Atlas Init Mode</th><td>{escape(str(overview['atlas_init_mode']))}</td></tr>
      <tr><th>Transformation Graph Script</th><td>{escape(str(overview['transformation_graph_script']))}</td></tr>
      <tr><th>Transformation Graph Source</th><td>{escape(str(overview.get('transformation_graph_script_source')))}</td></tr>
      <tr><th>Plan JSON</th><td><code>{escape(str(manifest['plan_path']))}</code></td></tr>
      <tr><th>Summary JSON</th><td><code>{escape(str(manifest['summary_path']))}</code></td></tr>
      <tr><th>Log File</th><td><code>{escape(str(manifest['log_path']))}</code></td></tr>
      <tr><th>Provenance JSON</th><td><code>{escape(str(manifest.get('provenance_path')))}</code></td></tr>
      <tr><th>Replay Command</th><td><code>{escape(str(manifest.get('reproduce_command_path')))}</code></td></tr>
    </table>
  </section>
  <section>
    <h2>Resolved Assumptions</h2>
    <p><strong>Enabled Stages:</strong> {escape(', '.join(overview['enabled_stages']) or 'None')}</p>
    <p><strong>Skipped Stages:</strong></p>
    {_html_list(skipped)}
  </section>
  <section>
    <h2>Warnings and Notes</h2>
    {warnings_html}
  </section>
  <section>
    <h2>Stage Timeline</h2>
    <table>
      <tr><th>Stage</th><th>Status</th><th>Duration (s)</th><th>Reason</th></tr>
      {''.join(f"<tr><td>{escape(str(item.get('name')))}</td><td>{escape(str(item.get('status')))}</td><td>{escape(str(item.get('duration_seconds')))}</td><td>{escape(str(item.get('reason')))}</td></tr>" for item in manifest.get('stage_timeline', []))}
    </table>
  </section>
  <section>
    <h2>QC Gallery</h2>
    {stages_html}
  </section>
</body>
</html>
"""


def write_registration_report(
    *,
    registration_output: Path,
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    """Write the JSON manifest and HTML report to the workflow output root."""

    registration_output.mkdir(parents=True, exist_ok=True)
    json_path = registration_output / "registration_report.json"
    html_path = registration_output / "registration_report.html"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    html_path.write_text(render_registration_report_html(manifest), encoding="utf-8")
    logger.info("Wrote registration QC report -> %s", html_path)
    return json_path, html_path
