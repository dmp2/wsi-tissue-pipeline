"""
Zarr compatibility helpers used by OME-Zarr writer paths.

This module centralizes small API differences between zarr-python v2 and v3:
- array creation methods (``create_array`` vs ``create_dataset``)
- compression kwargs (``compressor`` vs ``compressors``)
"""

from __future__ import annotations

from typing import Any

import zarr


def _zarr_major_version() -> int:
    """Return the installed zarr major version, defaulting to v2 on parse errors."""
    version = str(getattr(zarr, "__version__", "2"))
    try:
        return int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        return 2


def _coerce_blosc_shuffle_name(value: Any) -> str | None:
    """Map numcodecs Blosc shuffle values to zarr v3 string values."""
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"noshuffle", "shuffle", "bitshuffle"}:
            return normalized
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        return None
    return {
        0: "noshuffle",
        1: "shuffle",
        2: "bitshuffle",
    }.get(as_int)


def _to_v3_bytes_codec(compressor: Any) -> Any:
    """
    Convert supported v2 compressors to zarr v3 codec objects.

    Currently this supports numcodecs Blosc, which is the only compressor
    type used by this repository's manual writer path.
    """
    module_name = compressor.__class__.__module__
    if module_name.startswith("zarr.codecs"):
        return compressor

    cname = getattr(compressor, "cname", None)
    clevel = getattr(compressor, "clevel", None)
    shuffle = getattr(compressor, "shuffle", None)
    blocksize = getattr(compressor, "blocksize", 0)
    if cname is None or clevel is None:
        raise TypeError(
            "Unsupported compressor for zarr v3 array creation. "
            "Pass a zarr codec instance or numcodecs Blosc."
        )

    from zarr.codecs import BloscCodec

    return BloscCodec(
        cname=str(cname),
        clevel=int(clevel),
        shuffle=_coerce_blosc_shuffle_name(shuffle),
        blocksize=int(blocksize or 0),
    )


def compression_kwargs(compressor: Any | None) -> dict[str, Any]:
    """Return zarr-version-appropriate compression kwargs for group array creation."""
    if compressor is None:
        return {}
    if _zarr_major_version() >= 3:
        return {"compressors": [_to_v3_bytes_codec(compressor)]}
    return {"compressor": compressor}


def create_group_array(
    group: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: Any,
    compressor: Any | None = None,
    overwrite: bool = True,
) -> Any:
    """
    Create an array on a zarr group across v2/v3 APIs.

    Uses ``create_array`` when available and falls back to ``create_dataset``.
    """
    kwargs: dict[str, Any] = {
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
        "overwrite": overwrite,
    }
    kwargs.update(compression_kwargs(compressor))

    create_array = getattr(group, "create_array", None)
    if callable(create_array):
        try:
            return create_array(name, **kwargs)
        except TypeError as exc:
            if "overwrite" not in str(exc):
                raise
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("overwrite", None)
            return create_array(name, **retry_kwargs)
    try:
        return group.create_dataset(name, **kwargs)
    except TypeError as exc:
        if "overwrite" not in str(exc):
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs.pop("overwrite", None)
        return group.create_dataset(name, **retry_kwargs)
