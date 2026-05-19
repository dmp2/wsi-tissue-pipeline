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

_JDK_INSTALL_HINT = (
    "Install a JDK, not only a JRE. Conda users can run "
    "`conda install -n wsi-pipeline -c conda-forge openjdk>=17`; "
    "system users can install `openjdk-17-jdk-headless` or `openjdk-21-jdk-headless`."
)


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
        raise RuntimeError(f"Cached Bio-Formats jar at {path} does not match the pinned SHA-256.")


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


def _executable_name(name: str) -> str:
    """Return platform-specific Java executable name."""
    return f"{name}.exe" if os.name == "nt" else name


def _candidate_home(home: str | os.PathLike[str] | None) -> Path | None:
    """Normalize a possible Java home path."""
    if not home:
        return None
    path = Path(home).expanduser()
    return path if path.exists() else None


def _candidate_home_variants(home: str | os.PathLike[str] | None) -> list[Path]:
    """Return plausible Java homes for an env or prefix path."""
    path = _candidate_home(home)
    if path is None:
        return []

    variants: list[Path] = []
    if path.name == "bin":
        variants.append(path.parent)

    # Conda's openjdk package exposes java/javac in <env>/bin, but the real
    # JAVA_HOME for libjvm discovery is <env>/lib/jvm.
    conda_jvm = path / "lib" / "jvm"
    if conda_jvm.exists():
        variants.append(conda_jvm)

    variants.append(path)
    return variants


def _jdk_search_roots() -> list[Path]:
    """Return platform roots that commonly contain JDK installations."""
    if os.name == "nt":
        return []
    return [Path("/usr/lib/jvm")]


def _java_tool(home: Path, name: str) -> Path:
    """Return the expected path for a Java tool under a candidate home."""
    return home / "bin" / _executable_name(name)


def _has_jdk_tools(home: Path) -> bool:
    """Return whether a candidate home contains both java and javac."""
    return _java_tool(home, "java").exists() and _java_tool(home, "javac").exists()


def _libjvm_name() -> str:
    if os.name == "nt":
        return "jvm.dll"
    if sys.platform == "darwin":
        return "libjvm.dylib"
    return "libjvm.so"


def _find_libjvm(home: Path) -> Path | None:
    """Find the JVM shared library under a candidate Java home."""
    libjvm = _libjvm_name()
    candidates = [
        home / "lib" / "server" / libjvm,
        home / "jre" / "lib" / "server" / libjvm,
        home / "jre" / "lib" / "amd64" / "server" / libjvm,
        home / "lib" / "jvm" / "lib" / "server" / libjvm,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _candidate_jdk_homes() -> list[Path]:
    """Return ordered JDK home candidates from env vars, PATH, and system roots."""
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path | None) -> None:
        if path is None:
            return
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(path)

    for variant in _candidate_home_variants(os.getenv("JAVA_HOME")):
        add(variant)
    for variant in _candidate_home_variants(os.getenv("CONDA_PREFIX")):
        add(variant)
    for variant in _candidate_home_variants(sys.prefix):
        add(variant)

    javac_on_path = shutil.which(_executable_name("javac"))
    if javac_on_path:
        for variant in _candidate_home_variants(Path(javac_on_path).resolve().parent.parent):
            add(variant)

    for root in _jdk_search_roots():
        if not root.exists():
            continue
        for child in sorted(root.glob("*")):
            if child.is_dir():
                for variant in _candidate_home_variants(child):
                    add(variant)

    return candidates


def discover_java_runtime() -> dict[str, str | bool | None]:
    """
    Discover Java/JDK availability without importing PyJNIus.

    Returns a small status dictionary with ``java_home``, ``java``, ``javac``,
    ``is_jdk``, ``jvm_path``, and ``has_libjvm`` keys. ``is_jdk`` is true only
    when both ``java`` and ``javac`` are discoverable from the same Java home.
    """
    tool_only_status: dict[str, str | bool | None] | None = None
    for home in _candidate_jdk_homes():
        if _has_jdk_tools(home):
            jvm_path = _find_libjvm(home)
            status: dict[str, str | bool | None] = {
                "java_home": str(home),
                "java": str(_java_tool(home, "java")),
                "javac": str(_java_tool(home, "javac")),
                "is_jdk": True,
                "jvm_path": str(jvm_path) if jvm_path else None,
                "has_libjvm": jvm_path is not None,
            }
            if jvm_path is not None:
                return status
            if tool_only_status is None:
                tool_only_status = status

    if tool_only_status is not None:
        return tool_only_status

    java_on_path = shutil.which(_executable_name("java"))
    javac_on_path = shutil.which(_executable_name("javac"))
    return {
        "java_home": None,
        "java": java_on_path,
        "javac": javac_on_path,
        "is_jdk": False,
        "jvm_path": None,
        "has_libjvm": False,
    }


def _validate_java_runtime() -> Path:
    """Ensure a JDK is available before importing Pyjnius."""
    status = discover_java_runtime()
    if status["is_jdk"] and status.get("has_libjvm"):
        java_home = str(status["java_home"])
        os.environ["JAVA_HOME"] = java_home
        os.environ["JVM_PATH"] = str(status["jvm_path"])
        bin_dir = str(Path(java_home) / "bin")
        path_parts = os.getenv("PATH", "").split(os.pathsep) if os.getenv("PATH") else []
        if bin_dir not in path_parts:
            os.environ["PATH"] = os.pathsep.join([bin_dir, *path_parts])
        return Path(str(status["java"]))

    if status["is_jdk"]:
        raise RuntimeError(
            "JDK tools were found, but PyJNIus could not find the JVM shared library "
            f"({_libjvm_name()}). Set JVM_PATH to the absolute libjvm path or install a full "
            f"OpenJDK package. {_JDK_INSTALL_HINT}"
        )

    if status["java"]:
        raise RuntimeError(
            "Java runtime found, but no JDK compiler (`javac`) was found. "
            "Bio-Formats metadata via PyJNIus requires a JDK. "
            f"{_JDK_INSTALL_HINT}"
        )

    raise RuntimeError(
        f"Java/JDK not found. Bio-Formats metadata via PyJNIus requires Java. {_JDK_INSTALL_HINT}"
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
            'Install the package with `pip install -e ".[bioformats]"`.'
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
