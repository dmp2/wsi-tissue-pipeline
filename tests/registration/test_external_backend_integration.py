from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from wsi_pipeline.registration.backend import resolve_emlddmm_backend
from wsi_pipeline.registration.workflow import _resolve_transformation_graph_script
from wsi_pipeline.registration.config import EmlddmmWorkflowConfig


pytestmark = pytest.mark.skipif(
    not os.environ.get("WSI_PIPELINE_RUN_EXTERNAL_EMLDDMM_TESTS")
    or importlib.util.find_spec("emlddmm") is None,
    reason="External emlddmm integration tests are opt-in and require an installed emlddmm package.",
)


def test_external_backend_resolution_prefers_installed_package():
    backend = resolve_emlddmm_backend()

    assert backend.origin_type == "installed"
    assert backend.module_path is not None


def test_external_transformation_graph_script_resolves_from_installed_package():
    backend = resolve_emlddmm_backend()
    config = EmlddmmWorkflowConfig()
    config.mode = "atlas_to_target"
    config.atlas_path = Path("atlas.vtk")
    config.transformation_graph.execute = True

    script_path, warnings, source = _resolve_transformation_graph_script(config, backend)

    assert script_path is not None
    assert source == "external emlddmm package"
    assert warnings == [] or all(isinstance(item, str) for item in warnings)
