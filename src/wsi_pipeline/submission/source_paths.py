"""Shared source path helpers for submission filesystem checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class SourceLocation:
    """Resolved source location for local and non-local manifest values."""

    is_local: bool
    path_to_check: Path | None = None


def source_location(source_value: str, manifest_path: Path) -> SourceLocation:
    """Resolve a manifest source path or URI without opening the target."""
    parsed = urlparse(source_value)
    if parsed.scheme and parsed.scheme != "file":
        return SourceLocation(is_local=False)

    if parsed.scheme == "file":
        path_text = unquote(parsed.path)
        if parsed.netloc and parsed.netloc != "localhost":
            path_text = f"//{parsed.netloc}{path_text}"
        return SourceLocation(is_local=True, path_to_check=Path(path_text))

    source_path = Path(source_value)
    path_to_check = source_path if source_path.is_absolute() else manifest_path.parent / source_path
    return SourceLocation(is_local=True, path_to_check=path_to_check)


def source_target_for_extension(source_value: str) -> str:
    """Return the path-like target used for suffix checks without touching files."""
    parsed = urlparse(source_value)
    if parsed.scheme and parsed.path:
        return unquote(parsed.path)
    return source_value


__all__ = ["SourceLocation", "source_location", "source_target_for_extension"]
