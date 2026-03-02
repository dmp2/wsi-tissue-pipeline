"""Run-provenance helpers for EM-LDDMM step-5 executions."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from wsi_pipeline import __version__

from .config import EmlddmmResolvedPlan, EmlddmmRunProvenance, SCHEMA_VERSION

_HASH_LIMIT_BYTES = 2 * 1024 * 1024


def _find_repo_root(start: Path) -> Path | None:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / ".git").exists():
            return parent
    return None


def _run_git(args: list[str], repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def collect_git_metadata(start: Path) -> tuple[dict[str, Any], list[str]]:
    """Collect lightweight git metadata when the source tree is available."""

    repo_root = _find_repo_root(start)
    if repo_root is None:
        return {
            "repo_root": None,
            "commit_sha": None,
            "short_sha": None,
            "dirty": None,
            "branch": None,
        }, ["Git metadata unavailable; repository root could not be determined."]

    commit_sha = _run_git(["rev-parse", "HEAD"], repo_root)
    short_sha = _run_git(["rev-parse", "--short", "HEAD"], repo_root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    status = _run_git(["status", "--porcelain"], repo_root)
    warnings: list[str] = []
    if commit_sha is None:
        warnings.append("Git metadata unavailable; could not resolve the current commit SHA.")
    return {
        "repo_root": str(repo_root),
        "commit_sha": commit_sha,
        "short_sha": short_sha,
        "dirty": None if status is None else bool(status),
        "branch": branch,
    }, warnings


def compute_sha256(path: Path) -> str:
    """Compute a SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path, repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return None


def file_metadata(
    path: str | Path | None,
    *,
    repo_root: Path | None,
    hash_if_small: bool = False,
) -> dict[str, Any] | None:
    """Collect reproducibility metadata for a file or directory path."""

    if path in (None, ""):
        return None
    resolved = Path(path).resolve()
    exists = resolved.exists()
    payload: dict[str, Any] = {
        "path": str(resolved),
        "repo_relative_path": _repo_relative(resolved, repo_root),
        "exists": exists,
        "type": "directory" if exists and resolved.is_dir() else "file",
    }
    if not exists:
        return payload
    stat = resolved.stat()
    payload["size_bytes"] = stat.st_size
    payload["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    if resolved.is_file() and hash_if_small and stat.st_size <= _HASH_LIMIT_BYTES:
        payload["sha256"] = compute_sha256(resolved)
    return payload


def infer_script_source(
    script_path: Path | None,
    *,
    explicit_path: Path | None = None,
    backend_origin: str | None = None,
) -> str | None:
    """Classify a resolved transformation-graph script path."""

    if script_path is None:
        return None
    resolved = script_path.resolve()
    if explicit_path is not None and resolved == explicit_path.resolve():
        return "explicit override path"
    if backend_origin == "installed":
        return "external emlddmm package"
    return "workspace-local development fallback"


def build_reproduce_command(
    *,
    plan: EmlddmmResolvedPlan,
    config_override_path: Path | None,
    used_legacy_output_alias: bool,
) -> str:
    """Build a canonical replay command using --dataset-root terminology."""

    command: list[str] = [sys.executable, "scripts/run_pipeline.py", "step5", "--dataset-root", str(plan.dataset_root)]
    if plan.target_source != plan.dataset_root:
        command.extend(["--target-source", str(plan.target_source)])
    if plan.target_source_format != "prepared-dir":
        command.extend(["--target-source-format", plan.target_source_format])
    default_registration_output = Path(plan.dataset_root) / "emlddmm"
    if Path(plan.registration_output).resolve() != default_registration_output.resolve():
        command.extend(["--registration-output", str(plan.registration_output)])
    if plan.atlas_path is not None:
        command.extend(["--atlas", str(plan.atlas_path)])
    if plan.label_path is not None:
        command.extend(["--label", str(plan.label_path)])
    if config_override_path is not None:
        command.extend(["--emlddmm-config", str(config_override_path)])
    if plan.atlas_init_mode == "matrix":
        init_affine_path = plan.workflow_config.get("init_affine_path")
        if init_affine_path:
            command.extend(["--init-affine", str(init_affine_path)])
    elif plan.atlas_init_mode == "orientation":
        if plan.orientation_resolution.orientation_from:
            command.extend(["--orientation-from", plan.orientation_resolution.orientation_from])
        if plan.orientation_resolution.orientation_to:
            command.extend(["--orientation-to", plan.orientation_resolution.orientation_to])
    if plan.workflow_config.get("units", {}).get("atlas_unit_scale") != 1000.0:
        command.extend(["--atlas-unit-scale", str(plan.atlas_unit_scale)])
    if plan.workflow_config.get("units", {}).get("target_unit_scale") != 1.0:
        command.extend(["--target-unit-scale", str(plan.target_unit_scale)])
    if plan.workflow_config.get("device") not in (None, "auto"):
        command.extend(["--device", str(plan.workflow_config["device"])])
    if plan.manifest_path is not None:
        command.extend(["--precomputed-manifest", str(plan.manifest_path)])
    if "upsampling" in plan.enabled_stages:
        command.append("--upsample-between-slices")
        upsample_mode = plan.workflow_config.get("upsampling", {}).get("mode")
        if upsample_mode is not None:
            command.extend(["--upsample-mode", str(upsample_mode)])
    if "self_alignment" not in plan.enabled_stages:
        command.append("--skip-self-alignment")
    if plan.workflow_config.get("transformation_graph", {}).get("execute"):
        command.append("--run-transformation-graph")
    script_path = plan.transformation_graph_script
    if script_path is not None and infer_script_source(
        script_path,
        explicit_path=Path(plan.workflow_config.get("transformation_graph", {}).get("script_path"))
        if plan.workflow_config.get("transformation_graph", {}).get("script_path")
        else None,
        backend_origin=None,
    ) == "explicit override path":
        command.extend(["--transformation-graph-script", str(script_path)])
    if plan.workflow_config.get("debug", {}).get("write_notebook_bundle"):
        command.append("--write-notebook-bundle")
    if plan.workflow_config.get("outputs", {}).get("write_qc_report"):
        command.append("--write-qc-report")
    if plan.dry_run:
        command.append("--dry-run")
    command_str = subprocess.list2cmdline(command)
    if used_legacy_output_alias:
        return f"# Original invocation used deprecated --output.\n{command_str}\n"
    return f"{command_str}\n"


def build_runtime_metadata() -> dict[str, Any]:
    """Collect runtime metadata useful for reproducing the control plane."""

    torch_version = None
    cuda_available = None
    selected_device = None
    try:
        import torch

        torch_version = getattr(torch, "__version__", None)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            selected_device = f"cuda:{torch.cuda.current_device()}"
    except Exception:
        pass
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "selected_device": selected_device,
    }


def build_run_provenance(
    *,
    plan: EmlddmmResolvedPlan,
    backend: Any,
    warnings: list[str],
    config_override_path: Path | None,
    original_cli_argv: list[str] | None,
    used_legacy_output_alias: bool,
    report_manifest_path: Path | None,
    report_path: Path | None,
    replay_command_path: Path,
    repo_root_hint: Path,
    generated_control_files: list[Path],
) -> EmlddmmRunProvenance:
    """Build structured run-provenance metadata for a step-5 output."""

    git_metadata, git_warnings = collect_git_metadata(repo_root_hint)
    repo_root = Path(git_metadata["repo_root"]) if git_metadata.get("repo_root") else None
    backend_module_file = getattr(getattr(backend, "module", None), "__file__", None)
    backend_package_version = getattr(backend, "package_version", None)
    transformation_graph_metadata = file_metadata(
        plan.transformation_graph_script,
        repo_root=repo_root,
        hash_if_small=True,
    )
    if backend_package_version is None:
        try:
            backend_package_version = importlib.metadata.version("emlddmm")
        except importlib.metadata.PackageNotFoundError:
            backend_package_version = None
    backend_origin = getattr(backend, "origin_type", None)
    transformed_source = infer_script_source(
        plan.transformation_graph_script,
        explicit_path=Path(plan.workflow_config.get("transformation_graph", {}).get("script_path"))
        if plan.workflow_config.get("transformation_graph", {}).get("script_path")
        else None,
        backend_origin=backend_origin,
    )
    effective_warnings = list(dict.fromkeys([*warnings, *git_warnings]))
    if backend_origin == "vendored":
        effective_warnings.append(
            "Registration is using the vendored EM-LDDMM backend rather than an installed external emlddmm package."
        )
    if backend_package_version is None:
        effective_warnings.append(
            "External emlddmm package version metadata is unavailable for this run."
        )
    if transformed_source == "workspace-local development fallback":
        effective_warnings.append(
            "Transformation graph script resolved from a workspace-local development fallback."
        )
    effective_warnings = list(dict.fromkeys(effective_warnings))
    generated_controls = [
        file_metadata(path, repo_root=repo_root, hash_if_small=True)
        for path in generated_control_files
        if path is not None
    ]
    inputs = {
        "dataset_root": file_metadata(plan.dataset_root, repo_root=repo_root, hash_if_small=False),
        "target_source": file_metadata(plan.target_source, repo_root=repo_root, hash_if_small=False),
        "target_source_format": plan.target_source_format,
        "atlas_path": file_metadata(plan.atlas_path, repo_root=repo_root, hash_if_small=False),
        "label_path": file_metadata(plan.label_path, repo_root=repo_root, hash_if_small=False),
        "init_affine_path": file_metadata(
            plan.workflow_config.get("init_affine_path"),
            repo_root=repo_root,
            hash_if_small=True,
        ),
        "precomputed_manifest": file_metadata(plan.manifest_path, repo_root=repo_root, hash_if_small=True),
        "config_override_path": file_metadata(config_override_path, repo_root=repo_root, hash_if_small=True),
    }
    artifacts = {
        "plan_path": file_metadata(plan.registration_output / "resolved_run_plan.json", repo_root=repo_root, hash_if_small=True),
        "summary_path": file_metadata(plan.registration_output / "registration_summary.json", repo_root=repo_root, hash_if_small=True),
        "log_path": file_metadata(plan.log_path, repo_root=repo_root, hash_if_small=False),
        "report_manifest_path": file_metadata(report_manifest_path, repo_root=repo_root, hash_if_small=True),
        "report_path": file_metadata(report_path, repo_root=repo_root, hash_if_small=False),
        "reproduce_command_path": file_metadata(replay_command_path, repo_root=repo_root, hash_if_small=True),
        "generated_control_files": [item for item in generated_controls if item is not None],
    }
    replay_command = build_reproduce_command(
        plan=plan,
        config_override_path=config_override_path,
        used_legacy_output_alias=used_legacy_output_alias,
    )
    return EmlddmmRunProvenance(
        schema_version=SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        warnings=effective_warnings,
        pipeline={
            "name": "wsi_pipeline.step5",
            "version": __version__,
            "mode": plan.mode,
            "preset": plan.preset,
        },
        git=git_metadata,
        runtime=build_runtime_metadata(),
        backend={
            "backend_name": getattr(backend, "name", None),
            "module_path": str(Path(backend_module_file).resolve()) if backend_module_file else None,
            "origin_type": backend_origin,
            "emlddmm_package_version": backend_package_version,
            "transformation_graph_script": transformation_graph_metadata,
            "transformation_graph_script_source": transformed_source,
        },
        inputs=inputs,
        resolved_cli={
            "original_argv": list(original_cli_argv) if original_cli_argv else None,
            "normalized_command": replay_command.strip(),
            "used_legacy_output_alias": used_legacy_output_alias,
        },
        resolved_workflow_config=plan.workflow_config,
        artifacts=artifacts,
        parity={
            "preset": plan.preset,
            "atlas_init_mode": plan.atlas_init_mode,
            "orientation_resolution": plan.orientation_resolution.model_dump(mode="python"),
            "wrote_transformation_graph_execution_config": any(
                str(path).endswith("transformation_graph_execution_config.json")
                for path in plan.expected_outputs.get("atlas_registration", [])
            ),
        },
    )
