"""
Bio-Formats runtime bootstrap helpers.

This module owns the optional Pyjnius and Bio-Formats runtime setup used for
VSI physical-metadata extraction. The classpath must be configured before
``jnius`` is imported, so the bootstrap logic is centralized here.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import sys
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

BIOFORMATS_VERSION = "8.4.0"
BIOFORMATS_JAR_NAME = "bioformats_package.jar"
BIOFORMATS_JAR_URL = (
    f"https://downloads.openmicroscopy.org/bio-formats/{BIOFORMATS_VERSION}/artifacts/"
    f"{BIOFORMATS_JAR_NAME}"
)
BIOFORMATS_JAR_SHA256 = "109225564fe6a2db3908f8cb4c651623d5f4232a8e21c51b451ace0453e92362"

BIOFORMATS_JAR_ENV = "WSI_PIPELINE_BIOFORMATS_JAR"
BIOFORMATS_CACHE_DIR_ENV = "WSI_PIPELINE_BIOFORMATS_CACHE_DIR"
BIOFORMATS_DOWNLOAD_ENV = "WSI_PIPELINE_BIOFORMATS_DOWNLOAD"

_CONFIGURED_JAR_PATH: Path | None = None


def _download_enabled() -> bool:
    """Return whether automatic jar download is enabled."""
    raw = os.getenv(BIOFORMATS_DOWNLOAD_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _default_cache_dir() -> Path:
    """Return the cache root used for the managed Bio-Formats jar."""
    override = os.getenv(BIOFORMATS_CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()

    try:
        platformdirs = importlib.import_module("platformdirs")
    except ImportError:
        local_appdata = os.getenv("LOCALAPPDATA")
        if local_appdata:
            return Path(local_appdata) / "wsi-pipeline"
        return Path.home() / ".cache" / "wsi-pipeline"

    return Path(platformdirs.user_cache_dir("wsi-pipeline"))


def _default_jar_path() -> Path:
    """Return the canonical cache location for the managed Bio-Formats jar."""
    return _default_cache_dir() / "bioformats" / BIOFORMATS_VERSION / BIOFORMATS_JAR_NAME


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_managed_jar(path: Path) -> None:
    """Verify that the managed jar matches the pinned Bio-Formats artifact."""
    if _sha256_file(path) != BIOFORMATS_JAR_SHA256:
        raise RuntimeError(
            f"Cached Bio-Formats jar at {path} does not match the pinned SHA-256."
        )


@contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Serialize first-use downloads across concurrent processes."""
    try:
        filelock = importlib.import_module("filelock")
    except ImportError:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
    else:
        with filelock.FileLock(str(lock_path)):
            yield


def _download_jar(destination: Path) -> None:
    """Download the pinned Bio-Formats jar to the managed cache."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(BIOFORMATS_JAR_URL, timeout=60) as response:
            with temp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        _verify_managed_jar(temp_path)
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _resolve_bioformats_jar_path() -> Path:
    """Resolve the local jar path, downloading the managed artifact when needed."""
    override = os.getenv(BIOFORMATS_JAR_ENV)
    if override:
        jar_path = Path(override).expanduser()
        if not jar_path.exists():
            raise RuntimeError(
                f"{BIOFORMATS_JAR_ENV} points to '{jar_path}', but that file does not exist."
            )
        return jar_path

    jar_path = _default_jar_path()
    lock_path = jar_path.with_suffix(jar_path.suffix + ".lock")

    with _file_lock(lock_path):
        if jar_path.exists():
            try:
                _verify_managed_jar(jar_path)
            except RuntimeError as exc:
                if not _download_enabled():
                    raise RuntimeError(
                        "Cached Bio-Formats jar failed verification and automatic downloads are disabled."
                    ) from exc
                jar_path.unlink()
            else:
                return jar_path

        if not _download_enabled():
            raise RuntimeError(
                "Bio-Formats jar is not cached and automatic downloads are disabled. "
                f"Set {BIOFORMATS_JAR_ENV} to a local jar or enable downloads."
            )

        _download_jar(jar_path)
        return jar_path


def _validate_java_runtime() -> Path:
    """Ensure a Java runtime is available before importing Pyjnius."""
    executable = "java.exe" if os.name == "nt" else "java"
    java_home = os.getenv("JAVA_HOME")

    candidates: list[Path] = []
    if java_home:
        candidates.append(Path(java_home) / "bin" / executable)

    on_path = shutil.which("java")
    if on_path:
        candidates.append(Path(on_path))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "Java runtime not found. Install Java or set JAVA_HOME. "
        "The curated Conda and Docker environments include Java automatically."
    )


def ensure_bioformats_jnius():
    """
    Configure the Bio-Formats classpath and import ``jnius``.

    Returns
    -------
    module
        The imported ``jnius`` module.

    Raises
    ------
    RuntimeError
        If Java, Pyjnius, or the managed Bio-Formats jar are unavailable, or
        if ``jnius`` was imported before the classpath was configured.
    """
    global _CONFIGURED_JAR_PATH

    if "jnius" in sys.modules:
        if _CONFIGURED_JAR_PATH is None:
            raise RuntimeError(
                "Pyjnius was imported before the Bio-Formats classpath was configured. "
                "Call get_vsi_metadata() or ensure_bioformats_jnius() before importing jnius."
            )
        return sys.modules["jnius"]

    _validate_java_runtime()
    jar_path = _resolve_bioformats_jar_path()

    try:
        jnius_config = importlib.import_module("jnius_config")
    except ImportError as exc:
        raise RuntimeError(
            "Bio-Formats metadata backend requires the optional 'bioformats' dependencies. "
            "Install the package with `pip install -e \".[bioformats]\"`."
        ) from exc

    if _CONFIGURED_JAR_PATH is None:
        jnius_config.set_classpath(str(jar_path))
        _CONFIGURED_JAR_PATH = jar_path
    elif _CONFIGURED_JAR_PATH != jar_path:
        raise RuntimeError(
            "Bio-Formats runtime is already configured with a different jar path. "
            f"Current: {_CONFIGURED_JAR_PATH}; requested: {jar_path}."
        )

    return importlib.import_module("jnius")
