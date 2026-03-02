"""Backend resolution for EM-LDDMM workflow execution."""

from __future__ import annotations

import importlib.metadata
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass(frozen=True)
class EmlddmmBackend:
    """Normalized access to an EM-LDDMM backend implementation."""

    name: str
    module: ModuleType
    origin_type: str | None
    module_path: Path | None
    package_version: str | None
    read_data: Callable
    atlas_free_reconstruction: Callable
    emlddmm_multiscale: Callable
    write_transform_outputs: Callable
    write_qc_outputs: Callable
    orientation_to_orientation: Callable
    write_vtk_data: Callable
    downsample_image_domain: Callable
    write_matrix_data: Callable | None = None
    read_matrix_data: Callable | None = None
    labels_to_rgb: Callable | None = None


_REQUIRED_ATTRS = (
    "read_data",
    "atlas_free_reconstruction",
    "emlddmm_multiscale",
    "write_transform_outputs",
    "write_qc_outputs",
    "orientation_to_orientation",
    "write_vtk_data",
    "downsample_image_domain",
)


def _coerce_backend_module(imported: ModuleType) -> ModuleType | None:
    """Handle both `import emlddmm` and `from emlddmm import emlddmm` layouts."""

    candidates: list[ModuleType] = []
    if hasattr(imported, "emlddmm"):
        candidate = imported.emlddmm
        if isinstance(candidate, ModuleType):
            candidates.append(candidate)
    if isinstance(imported, ModuleType):
        candidates.append(imported)

    for candidate in candidates:
        if all(hasattr(candidate, attr) for attr in _REQUIRED_ATTRS):
            return candidate
    return None


def _build_backend(
    name: str,
    module: ModuleType,
    *,
    origin_type: str | None = None,
    package_version: str | None = None,
) -> EmlddmmBackend:
    """Construct a normalized backend wrapper."""

    missing = [attr for attr in _REQUIRED_ATTRS if not hasattr(module, attr)]
    if missing:
        raise ImportError(
            f"EM-LDDMM backend '{name}' is missing required attributes: {', '.join(missing)}"
        )

    return EmlddmmBackend(
        name=name,
        module=module,
        origin_type=origin_type,
        module_path=Path(module.__file__).resolve() if getattr(module, "__file__", None) else None,
        package_version=package_version,
        read_data=module.read_data,
        atlas_free_reconstruction=module.atlas_free_reconstruction,
        emlddmm_multiscale=module.emlddmm_multiscale,
        write_transform_outputs=module.write_transform_outputs,
        write_qc_outputs=module.write_qc_outputs,
        orientation_to_orientation=module.orientation_to_orientation,
        write_vtk_data=module.write_vtk_data,
        downsample_image_domain=module.downsample_image_domain,
        write_matrix_data=getattr(module, "write_matrix_data", None),
        read_matrix_data=getattr(module, "read_matrix_data", None),
        labels_to_rgb=getattr(module, "labels_to_rgb", None),
    )


def _find_vendored_backend_path() -> Path:
    """Locate the vendored legacy EM-LDDMM backend in the workspace."""

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "legacy_scripts" / "emlddmm.py"
        if candidate.exists():
            return candidate
    raise ImportError("Could not locate vendored legacy_scripts/emlddmm.py")


def _load_vendored_backend() -> EmlddmmBackend:
    """Load the vendored backend via importlib."""

    vendored_path = _find_vendored_backend_path()
    spec = importlib.util.spec_from_file_location(
        "wsi_pipeline._vendored_emlddmm",
        vendored_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {vendored_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _build_backend(
        f"vendored:{vendored_path}",
        module,
        origin_type="vendored",
        package_version=None,
    )


def resolve_emlddmm_backend() -> EmlddmmBackend:
    """Resolve the installed backend first, then fall back to the vendored copy."""

    try:
        import emlddmm as imported_backend  # type: ignore[import-not-found]

        module = _coerce_backend_module(imported_backend)
        if module is not None:
            try:
                package_version = importlib.metadata.version("emlddmm")
            except importlib.metadata.PackageNotFoundError:
                package_version = None
            return _build_backend(
                "installed:emlddmm",
                module,
                origin_type="installed",
                package_version=package_version,
            )
    except ImportError:
        pass

    return _load_vendored_backend()
