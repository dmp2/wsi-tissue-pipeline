from __future__ import annotations

import importlib

import pytest

from wsi_pipeline.registration.provenance import build_runtime_metadata


@pytest.mark.parametrize(
    "module_name",
    [
        "wsi_pipeline.registration",
        "wsi_pipeline.registration.workflow",
        "wsi_pipeline.registration.report",
        "wsi_pipeline.registration.provenance",
    ],
)
def test_registration_modules_import_on_supported_python_versions(module_name):
    module = importlib.import_module(module_name)

    assert module is not None


def test_build_runtime_metadata_uses_utc_offset_timestamp():
    metadata = build_runtime_metadata()
    timestamp_utc = metadata["timestamp_utc"]

    assert isinstance(timestamp_utc, str)
    assert timestamp_utc.endswith("+00:00")
