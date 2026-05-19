from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from wsi_pipeline import bioformats_runtime


@pytest.fixture(autouse=True)
def _reset_bioformats_runtime_state(monkeypatch):
    monkeypatch.setattr(bioformats_runtime, "_CONFIGURED_JAR_PATH", None)
    monkeypatch.delenv(bioformats_runtime.BIOFORMATS_JAR_ENV, raising=False)
    monkeypatch.delenv(bioformats_runtime.BIOFORMATS_CACHE_DIR_ENV, raising=False)
    monkeypatch.delenv(bioformats_runtime.BIOFORMATS_DOWNLOAD_ENV, raising=False)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.delenv("JVM_PATH", raising=False)
    monkeypatch.delitem(sys.modules, "jnius", raising=False)


def test_explicit_bioformats_jar_override_bypasses_download(tmp_path, monkeypatch):
    jar_path = tmp_path / "custom.jar"
    jar_path.write_bytes(b"custom")
    monkeypatch.setenv(bioformats_runtime.BIOFORMATS_JAR_ENV, str(jar_path))
    monkeypatch.setattr(
        bioformats_runtime,
        "_download_jar",
        lambda destination: pytest.fail(f"unexpected download for {destination}"),
    )

    assert bioformats_runtime._resolve_bioformats_jar_path() == jar_path


def test_cached_bioformats_jar_skips_download(tmp_path, monkeypatch):
    jar_path = tmp_path / "cache" / bioformats_runtime.BIOFORMATS_JAR_NAME
    jar_path.parent.mkdir(parents=True, exist_ok=True)
    jar_path.write_bytes(b"cached")
    monkeypatch.setattr(bioformats_runtime, "_default_jar_path", lambda: jar_path)
    monkeypatch.setattr(bioformats_runtime, "_verify_managed_jar", lambda path: None)
    monkeypatch.setattr(
        bioformats_runtime,
        "_download_jar",
        lambda destination: pytest.fail(f"unexpected download for {destination}"),
    )

    assert bioformats_runtime._resolve_bioformats_jar_path() == jar_path


def test_download_disabled_without_cached_or_overridden_jar_fails(tmp_path, monkeypatch):
    jar_path = tmp_path / "missing" / bioformats_runtime.BIOFORMATS_JAR_NAME
    monkeypatch.setattr(bioformats_runtime, "_default_jar_path", lambda: jar_path)
    monkeypatch.setenv(bioformats_runtime.BIOFORMATS_DOWNLOAD_ENV, "0")

    with pytest.raises(RuntimeError, match="disabled"):
        bioformats_runtime._resolve_bioformats_jar_path()


def test_file_lock_prevents_duplicate_downloads(tmp_path, monkeypatch):
    jar_path = tmp_path / "cache" / bioformats_runtime.BIOFORMATS_JAR_NAME
    download_count = 0
    state_lock = threading.Lock()

    @contextmanager
    def fake_file_lock(_lock_path):
        with state_lock:
            yield

    def fake_download(destination: Path) -> None:
        nonlocal download_count
        download_count += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"downloaded")

    monkeypatch.setattr(bioformats_runtime, "_default_jar_path", lambda: jar_path)
    monkeypatch.setattr(bioformats_runtime, "_verify_managed_jar", lambda path: None)
    monkeypatch.setattr(bioformats_runtime, "_file_lock", fake_file_lock)
    monkeypatch.setattr(bioformats_runtime, "_download_jar", fake_download)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _idx: bioformats_runtime._resolve_bioformats_jar_path(), range(2)))

    assert results == [jar_path, jar_path]
    assert download_count == 1


def test_ensure_bioformats_jnius_sets_classpath_before_import(tmp_path, monkeypatch):
    jar_path = tmp_path / bioformats_runtime.BIOFORMATS_JAR_NAME
    jar_path.write_bytes(b"jar")
    events: list[str] = []

    def fake_import_module(name: str):
        if name == "jnius_config":
            return SimpleNamespace(
                set_classpath=lambda *paths: events.append(f"set_classpath:{paths[0]}")
            )
        if name == "jnius":
            events.append("import_jnius")
            return SimpleNamespace(autoclass=lambda name: name)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(bioformats_runtime, "_validate_java_runtime", lambda: Path("java"))
    monkeypatch.setattr(bioformats_runtime, "_resolve_bioformats_jar_path", lambda: jar_path)
    monkeypatch.setattr(bioformats_runtime.importlib, "import_module", fake_import_module)

    result = bioformats_runtime.ensure_bioformats_jnius()

    assert result.autoclass("loci.formats.ImageReader") == "loci.formats.ImageReader"
    assert events == [f"set_classpath:{jar_path}", "import_jnius"]


def _write_java_tool(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)


def _write_libjvm(java_home: Path) -> Path:
    libjvm = java_home / "lib" / "server" / bioformats_runtime._libjvm_name()
    libjvm.parent.mkdir(parents=True, exist_ok=True)
    libjvm.write_bytes(b"jvm")
    return libjvm


def test_discover_java_runtime_prefers_active_conda_jdk(tmp_path, monkeypatch):
    conda_prefix = tmp_path / "env"
    java_home = conda_prefix / "lib" / "jvm"
    _write_java_tool(java_home / "bin" / "java")
    _write_java_tool(java_home / "bin" / "javac")
    libjvm = _write_libjvm(java_home)

    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(bioformats_runtime, "_jdk_search_roots", lambda: [])
    monkeypatch.setattr(bioformats_runtime.sys, "prefix", str(tmp_path / "python"))

    status = bioformats_runtime.discover_java_runtime()

    assert status["is_jdk"] is True
    assert status["java_home"] == str(java_home)
    assert status["javac"] == str(java_home / "bin" / "javac")
    assert status["jvm_path"] == str(libjvm)
    assert status["has_libjvm"] is True


def test_validate_java_runtime_sets_java_home_from_discovered_jdk(tmp_path, monkeypatch):
    root = tmp_path / "usr-lib-jvm"
    jdk_home = root / "java-21-openjdk-amd64"
    _write_java_tool(jdk_home / "bin" / "java")
    _write_java_tool(jdk_home / "bin" / "javac")
    libjvm = _write_libjvm(jdk_home)

    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(bioformats_runtime, "_jdk_search_roots", lambda: [root])
    monkeypatch.setattr(bioformats_runtime.sys, "prefix", str(tmp_path / "python"))

    java = bioformats_runtime._validate_java_runtime()

    assert java == jdk_home / "bin" / "java"
    assert Path(bioformats_runtime.os.environ["JAVA_HOME"]) == jdk_home
    assert Path(bioformats_runtime.os.environ["JVM_PATH"]) == libjvm


def test_validate_java_runtime_rejects_jdk_without_libjvm(tmp_path, monkeypatch):
    jdk_home = tmp_path / "jdk"
    _write_java_tool(jdk_home / "bin" / "java")
    _write_java_tool(jdk_home / "bin" / "javac")

    monkeypatch.setenv("JAVA_HOME", str(jdk_home))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(bioformats_runtime, "_jdk_search_roots", lambda: [])
    monkeypatch.setattr(bioformats_runtime.sys, "prefix", str(tmp_path / "python"))

    with pytest.raises(RuntimeError, match="JVM shared library"):
        bioformats_runtime._validate_java_runtime()


def test_validate_java_runtime_rejects_jre_without_javac(tmp_path, monkeypatch):
    java_bin = tmp_path / "jre" / "bin"
    _write_java_tool(java_bin / "java")

    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv("PATH", str(java_bin))
    monkeypatch.setattr(bioformats_runtime, "_jdk_search_roots", lambda: [])
    monkeypatch.setattr(bioformats_runtime.sys, "prefix", str(tmp_path / "python"))

    with pytest.raises(RuntimeError, match="no JDK compiler"):
        bioformats_runtime._validate_java_runtime()


def test_ensure_bioformats_jnius_fails_if_jnius_already_imported(monkeypatch):
    monkeypatch.setitem(sys.modules, "jnius", SimpleNamespace())

    with pytest.raises(RuntimeError, match="before the Bio-Formats classpath was configured"):
        bioformats_runtime.ensure_bioformats_jnius()
