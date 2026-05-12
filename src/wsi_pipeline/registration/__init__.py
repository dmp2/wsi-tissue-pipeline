"""Registration utilities."""

from .config import (
    EmlddmmResolvedPlan,
    EmlddmmRunProvenance,
    EmlddmmWorkflowConfig,
    EmlddmmWorkflowResult,
    OrientationResolution,
)
from .workflow import plan_emlddmm_workflow, run_emlddmm_workflow


def emlddmm_multiscale_symmetric_N(*args, **kwargs):
    """Lazy symmetric-registration export."""

    from .symmetric import emlddmm_multiscale_symmetric_N as impl

    return impl(*args, **kwargs)


def upsample_between_slices(*args, **kwargs):
    """Lazy between-slice upsampling export."""

    from .upsample import upsample_between_slices as impl

    return impl(*args, **kwargs)


def diagnose_self_alignment_run(*args, **kwargs):
    """Lazy self-alignment diagnostics export."""

    from .sweep import diagnose_self_alignment_run as impl

    return impl(*args, **kwargs)


def run_ea2d_self_alignment_sweep(*args, **kwargs):
    """Lazy ``eA2d`` sweep export."""

    from .sweep import run_ea2d_self_alignment_sweep as impl

    return impl(*args, **kwargs)


def summarize_ea2d_sweep(*args, **kwargs):
    """Lazy ``eA2d`` sweep summary export."""

    from .sweep import summarize_ea2d_sweep as impl

    return impl(*args, **kwargs)


__all__ = [
    "diagnose_self_alignment_run",
    "EmlddmmResolvedPlan",
    "EmlddmmRunProvenance",
    "EmlddmmWorkflowConfig",
    "EmlddmmWorkflowResult",
    "OrientationResolution",
    "emlddmm_multiscale_symmetric_N",
    "plan_emlddmm_workflow",
    "run_ea2d_self_alignment_sweep",
    "run_emlddmm_workflow",
    "summarize_ea2d_sweep",
    "upsample_between_slices",
]
