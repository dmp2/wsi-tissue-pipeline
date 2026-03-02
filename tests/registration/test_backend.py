from __future__ import annotations

import builtins
import sys
from types import ModuleType

import pytest

import wsi_pipeline.registration.backend as backend_module


def _fake_backend_module() -> ModuleType:
    module = ModuleType("emlddmm")
    module.read_data = lambda *args, **kwargs: None
    module.atlas_free_reconstruction = lambda *args, **kwargs: None
    module.emlddmm_multiscale = lambda *args, **kwargs: None
    module.write_transform_outputs = lambda *args, **kwargs: None
    module.write_qc_outputs = lambda *args, **kwargs: None
    module.orientation_to_orientation = lambda *args, **kwargs: None
    module.write_vtk_data = lambda *args, **kwargs: None
    module.downsample_image_domain = lambda *args, **kwargs: None
    module.write_matrix_data = lambda *args, **kwargs: None
    return module


def test_resolve_backend_prefers_installed_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "emlddmm", _fake_backend_module())

    resolved = backend_module.resolve_emlddmm_backend()

    assert resolved.name == "installed:emlddmm"
    assert resolved.write_matrix_data is not None


def test_resolve_backend_falls_back_to_vendored(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "emlddmm":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    fake_backend = backend_module.EmlddmmBackend(
        name="vendored:test",
        module=_fake_backend_module(),
        read_data=lambda *args, **kwargs: None,
        atlas_free_reconstruction=lambda *args, **kwargs: None,
        emlddmm_multiscale=lambda *args, **kwargs: None,
        write_transform_outputs=lambda *args, **kwargs: None,
        write_qc_outputs=lambda *args, **kwargs: None,
        orientation_to_orientation=lambda *args, **kwargs: None,
        write_vtk_data=lambda *args, **kwargs: None,
        downsample_image_domain=lambda *args, **kwargs: None,
    )

    monkeypatch.delitem(sys.modules, "emlddmm", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(backend_module, "_load_vendored_backend", lambda: fake_backend)

    resolved = backend_module.resolve_emlddmm_backend()

    assert resolved is fake_backend


def test_resolve_backend_raises_when_both_paths_fail(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "emlddmm":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "emlddmm", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(
        backend_module,
        "_load_vendored_backend",
        lambda: (_ for _ in ()).throw(ImportError("vendored missing")),
    )

    with pytest.raises(ImportError, match="vendored missing"):
        backend_module.resolve_emlddmm_backend()
