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


__all__ = [
    "EmlddmmResolvedPlan",
    "EmlddmmRunProvenance",
    "EmlddmmWorkflowConfig",
    "EmlddmmWorkflowResult",
    "OrientationResolution",
    "emlddmm_multiscale_symmetric_N",
    "plan_emlddmm_workflow",
    "run_emlddmm_workflow",
    "upsample_between_slices",
]
