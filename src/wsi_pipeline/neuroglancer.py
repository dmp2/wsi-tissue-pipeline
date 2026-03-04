"""
Neuroglancer Module
===================

Provides Neuroglancer integration for 3D visualization of tissue stacks,
including:

- CORS + HTTP Range file server for efficient remote access
- Plate-level OME-Zarr viewer (one layer per child tissue)
- Precomputed volume viewer (``precomputed://`` URL scheme)
- Shareable Neuroglancer link generation
- Neuroglancer JSON state file generation

This module consolidates ``neuroglancer_utils`` (state file generation) and
``visualization`` (server and viewer) into a single place.

Usage
-----
**State generation (no Neuroglancer install required):**

>>> from wsi_pipeline.neuroglancer import emit_ng_state_for_ngff_plate
>>> emit_ng_state_for_ngff_plate("/data/plate", "http://localhost:8000")

**Interactive viewer (requires ``pip install wsi-tissue-pipeline[visualization]``):**

>>> from wsi_pipeline.neuroglancer import open_neuroglancer_plate_view
>>> viewer, httpd = open_neuroglancer_plate_view("/data/plate")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

try:
    import neuroglancer as ng
    NEUROGLANCER_AVAILABLE = True
except ImportError:
    NEUROGLANCER_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Shader constants
# GLSL shaders passed to Neuroglancer to control how channel data is displayed.
# RGB_SHADER_PLAIN: simple normalized RGB composite.
# RGB_SHADER: interactive UI controls for per-channel brightness/contrast.
# R_SHADER / G_SHADER / B_SHADER: single-channel overlays.
# =============================================================================

RGB_SHADER_PLAIN = r"""
void main() {
    emitRGB(vec3(
        toNormalized(getDataValue(0)),
        toNormalized(getDataValue(1)),
        toNormalized(getDataValue(2))
    ));
}
"""

RGB_SHADER = r"""
#uicontrol invlerp red(channel=0)
#uicontrol invlerp green(channel=1)
#uicontrol invlerp blue(channel=2)
void main() {
    emitRGB(vec3(red(), green(), blue()));
}
"""

R_SHADER = """void main(){ emitRGB( vec3( toNormalized(getDataValue(0)), 0.0, 0.0 ) ); }"""
G_SHADER = """void main(){ emitRGB( vec3( 0.0, toNormalized(getDataValue(1)), 0.0 ) ); }"""
B_SHADER = """void main(){ emitRGB( vec3( 0.0, 0.0, toNormalized(getDataValue(2)) ) ); }"""


# =============================================================================
# STATE GENERATION
# Functions that write Neuroglancer JSON state files (.json). These files
# describe which layers to display and how -- they are file-path references,
# not data servers. No Neuroglancer Python package is required.
# =============================================================================

def emit_ng_state_for_ngff_plate(
    plate_root: str | Path,
    base_http_url: str,
    out_state_path: str | Path = "state_ngff_plate.json",
    layer_prefix: str = "",
) -> Path:
    """Create a Neuroglancer state with one ``image`` layer per child ``*.ome.zarr``.

    Each layer source uses the pattern ``zarr://{base_http_url}/{child}``.
    The state JSON can be loaded into the Neuroglancer browser viewer.

    Parameters
    ----------
    plate_root : str or Path
        Directory containing child ``*.ome.zarr`` directories.
    base_http_url : str
        HTTP base URL serving ``plate_root`` (e.g. ``http://localhost:8000``).
    out_state_path : str or Path
        Output path for the JSON state file.
    layer_prefix : str
        Optional prefix for layer names.

    Returns
    -------
    Path
        Path to the written state JSON file.
    """
    plate_root = Path(plate_root)
    children = sorted(p for p in plate_root.iterdir() if p.is_dir() and p.name.endswith(".ome.zarr"))
    from .omezarr.metadata import _is_ngff_image_group

    layers = []
    for idx, child in enumerate(children):
        if not _is_ngff_image_group(child):
            continue
        name = f"{layer_prefix}{child.stem}"
        source = f"zarr://{base_http_url.rstrip('/')}/{child.name}"
        layers.append({
            "type": "image",
            "name": name,
            "source": source,
            "shader": R_SHADER,
            "visible": (idx == 0),  # only first is visible initially
        })

    state = {
        "layers": layers,
        "crossSectionScale": 1.0,
        "perspectiveZoom": 50.0,
        "selectedLayer": {"visible": True, "layer": layers[0]["name"]} if layers else {},
    }

    out_state_path = Path(out_state_path)
    out_state_path.write_text(json.dumps(state, indent=2))
    logger.info("Wrote Neuroglancer state -> %s", out_state_path)
    return out_state_path


def emit_ng_state_for_precomputed_plate(
    precomputed_url: str,
    out_state_path: str | Path = "state_precomputed_plate.json",
    layer_name: str = "TissuePlate",
) -> Path:
    """Create a Neuroglancer state JSON for a single precomputed image layer.

    Parameters
    ----------
    precomputed_url : str
        Neuroglancer-style precomputed source URL (e.g.
        ``precomputed://file:///abs/path/to/volume``).
    out_state_path : str or Path
        Output path for the JSON state file.
    layer_name : str
        Name for the Neuroglancer layer.

    Returns
    -------
    Path
        Path to the written state JSON file.
    """
    state = {
        "layers": [{
            "type": "image",
            "name": layer_name,
            "source": precomputed_url.rstrip("/"),
        }],
        "crossSectionScale": 1.0,
        "perspectiveZoom": 50.0,
        "selectedLayer": {"visible": True, "layer": layer_name},
    }
    out_state_path = Path(out_state_path)
    out_state_path.write_text(json.dumps(state, indent=2))
    logger.info("Wrote Neuroglancer state -> %s", out_state_path)
    return out_state_path


# =============================================================================
# SERVER & VIEWER
# CORS HTTP server and NeuroglancerViewer class for serving local files to the
# Neuroglancer browser application. Requires the ``neuroglancer`` Python package
# (``pip install wsi-tissue-pipeline[visualization]``).
# =============================================================================

class CORSRequestHandler(SimpleHTTPRequestHandler):
    """Static file server with permissive CORS and HTTP Range support.

    Supports ``206 Partial Content`` responses so that Neuroglancer can
    efficiently fetch byte-range slices of large Zarr chunks or precomputed
    shards.
    """

    server_version = "CORSRangeHTTP/0.2"

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()

        ctype = self.guess_type(path)
        if not os.path.exists(path):
            self.send_error(404, "File not found")
            return None

        f = open(path, "rb")
        fs = os.fstat(f.fileno())
        size = fs.st_size
        last_mod = self.date_time_string(fs.st_mtime)

        range_header = self.headers.get("Range")
        if range_header and range_header.strip().lower().startswith("bytes="):
            try:
                rng = range_header.strip().split("=")[1]
                start_str, end_str = (rng.split("-", 1) + [""])[:2]
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else size - 1
                if start < 0 or end >= size or start > end:
                    raise ValueError
            except Exception:
                self.send_error(416, "Invalid Range")
                f.close()
                return None

            self.send_response(206, "Partial Content")
            self.send_header("Content-Type", ctype)
            self.send_header("Last-Modified", last_mod)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            self._range = (start, end)
            return f
        else:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Last-Modified", last_mod)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            self._range = None
            return f

    def copyfile(self, source, outputfile):
        rng = getattr(self, "_range", None)
        if rng is None:
            return shutil.copyfileobj(source, outputfile)
        start, end = rng
        source.seek(start)
        remaining = end - start + 1
        bufsize = 64 * 1024
        while remaining > 0:
            chunk = source.read(min(bufsize, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def log_message(self, format, *args):
        # Suppress per-request access log noise; use logger for server start/stop
        pass


def start_cors_server(
    root_dir: str | Path,
    host: str = "localhost",
    port: int = 8000,
) -> HTTPServer:
    """Start a CORS + Range-enabled HTTP file server on a background thread.

    Parameters
    ----------
    root_dir : str or Path
        Directory to serve.
    host : str
        Server hostname.
    port : int
        Server port.

    Returns
    -------
    HTTPServer
        Running server instance.
    """
    root_dir = str(Path(root_dir).resolve())
    Handler = partial(CORSRequestHandler, directory=root_dir)
    server = HTTPServer((host, port), Handler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    logger.info("[CORS] Serving %s at http://%s:%d", root_dir, host, port)
    return server


def stop_cors_server(server: HTTPServer | None) -> None:
    """Stop a running CORS server gracefully."""
    if server is None:
        return
    try:
        server.shutdown()
    finally:
        server.server_close()


def get_zarr_physical_scale(zarr_path: Path) -> tuple[float, float]:
    """Read physical pixel scale from OME-Zarr root attrs.

    Parameters
    ----------
    zarr_path : Path
        Path to OME-Zarr directory.

    Returns
    -------
    tuple
        ``(phys_x, phys_y)`` in micrometers.
    """
    try:
        import zarr

        root = zarr.open_group(str(zarr_path), mode="r")
        attrs = dict(root.attrs)
        ms = attrs.get("multiscales", [{}])[0]
        datasets = ms.get("datasets", [{}])
        if datasets:
            transforms = datasets[0].get("coordinateTransformations", [{}])
            if transforms:
                scale = transforms[0].get("scale", [1.0, 1.0, 1.0])
                return (float(scale[-1]), float(scale[-2]))
    except Exception:
        pass

    return (1.0, 1.0)


def find_zarr_children(root_dir: Path) -> list[Path]:
    """Find all OME-Zarr directories in a root directory."""
    from .omezarr.metadata import _is_ngff_image_group

    children = []
    for p in sorted(root_dir.iterdir()):
        if p.is_dir() and _is_ngff_image_group(p):
            children.append(p)
    return children


def _require_neuroglancer():
    if not NEUROGLANCER_AVAILABLE:
        raise ImportError(
            "Neuroglancer not installed. Install with: "
            "pip install wsi-tissue-pipeline[visualization]"
        )


def open_neuroglancer_plate_view(
    plate_root: str | Path,
    *,
    mode: str = "remote",
    http_host: str = "localhost",
    http_port: int = 8000,
    ng_host: str = "localhost",
    ng_port: int = 9999,
    shader: str = RGB_SHADER_PLAIN,
    visible_first_only: bool = False,
) -> tuple:
    """Launch Neuroglancer with one layer per child ``.ome.zarr`` in a plate directory.

    Parameters
    ----------
    plate_root : str or Path
        Directory containing child ``*.ome.zarr`` (or ``*.zarr``) directories.
    mode : str
        ``"remote"`` starts a CORS HTTP server; ``"local"`` uses
        ``ng.LocalVolume`` (only works if Neuroglancer runs on the same host).
    http_host, http_port : str, int
        Bind address for the CORS file server (remote mode only).
    ng_host, ng_port : str, int
        Bind address for the Neuroglancer Python server.
    shader : str
        GLSL shader string.  See module-level constants
        ``RGB_SHADER_PLAIN``, ``RGB_SHADER``, ``R_SHADER``, etc.
    visible_first_only : bool
        If *True*, only the first layer is visible; others start hidden.

    Returns
    -------
    tuple
        ``(viewer, httpd)`` -- the Neuroglancer ``Viewer`` and the
        ``HTTPServer`` (or *None* in local mode).
    """
    _require_neuroglancer()

    plate_root = Path(plate_root).resolve()
    children = sorted(
        p
        for p in plate_root.iterdir()
        if p.is_dir() and (p.name.endswith(".ome.zarr") or p.name.endswith(".zarr"))
    )

    httpd = None
    if mode == "remote":
        httpd = start_cors_server(str(plate_root.parent), host=http_host, port=http_port)
        base_http = f"http://{http_host}:{http_port}/{plate_root.name}"
    else:
        base_http = None

    ng.set_server_bind_address(bind_address=ng_host, bind_port=ng_port)
    viewer = ng.Viewer()

    with viewer.txn() as s:
        dims_set = False
        for idx, child in enumerate(children):
            phys_x, phys_y = get_zarr_physical_scale(child)
            dims = ng.CoordinateSpace(
                names=["c^", "y", "x"],
                units=["", "um", "um"],
                scales=[1, phys_y, phys_x],
            )

            if not dims_set:
                s.dimensions = dims
                dims_set = True

            layer_name = child.stem

            if mode == "remote":
                src = f"zarr://{base_http}/{child.name}"
                s.layers[layer_name] = ng.ImageLayer(source=src, shader=shader)
            else:
                import zarr
                z = zarr.open(child / "s0", mode="r")
                s.layers[layer_name] = ng.ImageLayer(
                    source=ng.LocalVolume(
                        data=z,
                        dimensions=dims,
                        voxel_offset=(0, 0, 0),
                    ),
                    shader=shader,
                )

            if visible_first_only and idx > 0:
                s.layers[layer_name].visible = False

    logger.info("Neuroglancer URL: %s", viewer)
    return viewer, httpd


def open_neuroglancer_precomputed(
    precomp_url: str,
    *,
    mode: str = "remote",
    http_host: str = "localhost",
    http_port: int = 8000,
    ng_host: str = "localhost",
    ng_port: int = 9999,
    shader: str = RGB_SHADER_PLAIN,
) -> tuple:
    """Launch Neuroglancer for a single precomputed volume.

    *precomp_url* can use any of these schemes::

        precomputed://http(s)://host/path/to/volume
        precomputed://gs://bucket/path
        precomputed://s3://bucket/path
        precomputed://file:///ABS/PATH/volume   (local disk)

    When a ``file://`` URL is given and *mode* is ``"remote"``, a local CORS
    HTTP server is started automatically and the URL is rewritten to use HTTP.

    Parameters
    ----------
    precomp_url : str
        Neuroglancer-style precomputed source URL.
    mode : str
        ``"remote"`` (default) serves ``file://`` paths over HTTP.
    http_host, http_port : str, int
        Bind address for the CORS file server.
    ng_host, ng_port : str, int
        Bind address for the Neuroglancer Python server.
    shader : str
        GLSL shader string.

    Returns
    -------
    tuple
        ``(viewer, httpd)``.
    """
    _require_neuroglancer()

    httpd = None
    src = precomp_url.rstrip("/")

    if src.startswith("precomputed://file://"):
        parsed = urlparse(src.replace("precomputed://", ""))
        fs_path = Path(parsed.path)
        if mode == "remote":
            parent = fs_path.parent
            httpd = start_cors_server(str(parent), host=http_host, port=http_port)
            src = f"precomputed://http://{http_host}:{http_port}/{fs_path.name}"
        else:
            raise ValueError(
                "Neuroglancer requires HTTP-accessible data. "
                "Use mode='remote' for file-based precomputed data."
            )

    ng.set_server_bind_address(bind_address=ng_host, bind_port=ng_port)
    viewer = ng.Viewer()
    with viewer.txn() as s:
        s.layers["precomp"] = ng.ImageLayer(source=src, shader=shader)
    logger.info("Neuroglancer URL: %s", viewer)
    return viewer, httpd


def start_neuroglancer_server(
    zarr_dir: str | Path,
    ng_host: str = "localhost",
    ng_port: int = 9999,
    http_host: str = "localhost",
    http_port: int = 8000,
    mode: str = "remote",
):
    """Start Neuroglancer with Zarr data.

    .. deprecated::
        Use :func:`open_neuroglancer_plate_view` for multi-tissue plate
        viewing, or :func:`open_neuroglancer_precomputed` for precomputed
        volumes.

    Parameters
    ----------
    zarr_dir : str or Path
        Directory containing OME-Zarr files.
    ng_host : str
        Neuroglancer server host.
    ng_port : int
        Neuroglancer server port.
    http_host : str
        HTTP file server host.
    http_port : int
        HTTP file server port.
    mode : str
        "remote" to use HTTP server, "local" for direct file access.

    Returns
    -------
    tuple
        (viewer, httpd) - Neuroglancer viewer and HTTP server (if remote).
    """
    _require_neuroglancer()

    zarr_dir = Path(zarr_dir)

    children = find_zarr_children(zarr_dir)
    if not children:
        logger.warning("No OME-Zarr files found in %s", zarr_dir)
        return None, None

    httpd = None
    if mode == "remote":
        httpd = start_cors_server(zarr_dir.parent, http_host, http_port)
        base_http = f"http://{http_host}:{http_port}/{zarr_dir.name}"

    ng.set_server_bind_address(bind_address=ng_host, bind_port=ng_port)
    viewer = ng.Viewer()

    with viewer.txn() as state:
        for child in children:
            phys_x, phys_y = get_zarr_physical_scale(child)

            dims = ng.CoordinateSpace(
                names=["c^", "y", "x"],
                units=["", "um", "um"],
                scales=[1, phys_y, phys_x],
            )

            layer_name = child.stem

            if mode == "remote":
                src = f"zarr://{base_http}/{child.name}/s0"
                state.layers[layer_name] = ng.ImageLayer(
                    source=src,
                    shader=RGB_SHADER_PLAIN,
                )
            else:
                import zarr
                z = zarr.open(child / "s0", mode="r")
                vol = ng.LocalVolume(data=z, dimensions=dims)
                state.layers[layer_name] = ng.ImageLayer(source=vol)

            if state.dimensions is None:
                state.dimensions = dims

    url = viewer.get_viewer_url()
    logger.info("Neuroglancer URL: %s", url)

    return viewer, httpd


def create_neuroglancer_link(
    zarr_url: str,
    layer_name: str = "data",
    base_url: str = "https://neuroglancer-demo.appspot.com",
) -> str:
    """Create a shareable Neuroglancer link.

    Parameters
    ----------
    zarr_url : str
        URL to OME-Zarr data (must be publicly accessible).
    layer_name : str
        Name for the layer.
    base_url : str
        Neuroglancer instance URL.

    Returns
    -------
    str
        Shareable Neuroglancer URL.
    """
    import urllib.parse

    state = {
        "layers": [
            {
                "type": "image",
                "source": f"zarr://{zarr_url}",
                "name": layer_name,
            }
        ],
    }

    state_json = json.dumps(state)
    encoded = urllib.parse.quote(state_json)

    return f"{base_url}/#!{encoded}"


class NeuroglancerViewer:
    """Wrapper class for Neuroglancer visualization.

    Convenience wrapper around :func:`start_neuroglancer_server` for users who
    prefer object-oriented usage with context manager support. For one-off
    calls, prefer :func:`open_neuroglancer_plate_view` directly.

    Examples
    --------
    >>> viewer = NeuroglancerViewer(zarr_dir="/data/specimen.zarr")
    >>> viewer.start()
    >>> print(viewer.url)
    >>> # In Jupyter:
    >>> viewer.show_iframe()

    Context manager usage::

        with NeuroglancerViewer("/data/specimen.zarr") as viewer:
            print(viewer.url)
    """

    def __init__(
        self,
        zarr_dir: str | Path,
        ng_port: int = 9999,
        http_port: int = 8000,
    ):
        self.zarr_dir = Path(zarr_dir)
        self.ng_port = ng_port
        self.http_port = http_port
        self._viewer = None
        self._httpd = None

    def start(self, mode: str = "remote"):
        """Start the viewer."""
        self._viewer, self._httpd = start_neuroglancer_server(
            self.zarr_dir,
            ng_port=self.ng_port,
            http_port=self.http_port,
            mode=mode,
        )

    def stop(self):
        """Stop the viewer and server."""
        if self._httpd:
            stop_cors_server(self._httpd)
            self._httpd = None

    @property
    def url(self) -> str | None:
        """Get the viewer URL."""
        if self._viewer:
            return self._viewer.get_viewer_url()
        return None

    def show_iframe(self, width: int = 800, height: int = 600):
        """Display viewer in Jupyter iframe."""
        from IPython.display import IFrame, display

        if self.url:
            display(IFrame(self.url, width=width, height=height))
        else:
            logger.warning("Viewer not started. Call start() first.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
