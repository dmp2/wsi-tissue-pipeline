"""
SciServer Integration (optional, deployment-specific)

Utilities for running the WSI pipeline on Johns Hopkins SciServer -- a
cloud-based scientific computing platform. Provides environment detection,
storage path configuration, MLflow integration, and data lineage tracking
specific to SciServer.

Not needed for local, Docker, or Colab deployments.

Public API
----------
SciServerPipeline        Unified interface: environment detection + storage + tracking
StorageConfig            Dataclass for SciServer/local storage paths
setup_sciserver_tracking Quick-setup convenience function
is_sciserver_environment Detect whether running inside SciServer Compute
get_sciserver_user       Return authenticated SciServer username
"""

from wsi_pipeline.sciserver.environment import (
    get_sciserver_user,
    is_sciserver_environment,
)

__all__ = [
    "is_sciserver_environment",
    "get_sciserver_user",
]
