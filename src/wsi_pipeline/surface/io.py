"""Surface mesh I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            'PyVista/VTK are not installed. Install with: pip install -e ".[visualization]"'
        ) from exc
    return pv


def _validate_vertices(vertices) -> np.ndarray:
    vertices_np = np.asarray(vertices, dtype=np.float64)
    if vertices_np.ndim != 2 or vertices_np.shape[1] != 3:
        raise ValueError(f"Expected vertices with shape (n, 3), got {vertices_np.shape}")
    if not np.all(np.isfinite(vertices_np)):
        raise ValueError("Surface vertices must be finite")
    return vertices_np


def _validate_triangle_faces(faces, *, vertex_count: int) -> np.ndarray:
    faces_np = np.asarray(faces)
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError(f"Expected triangular faces with shape (n, 3), got {faces_np.shape}")
    if faces_np.size:
        faces_float = faces_np.astype(np.float64)
        if not np.all(np.isfinite(faces_float)):
            raise ValueError("Surface faces must be finite integer indices")
        if not np.all(faces_float == np.round(faces_float)):
            raise ValueError("Surface faces must be integer indices")
    faces_int = faces_np.astype(np.int64)
    if faces_int.size:
        min_index = int(np.min(faces_int))
        max_index = int(np.max(faces_int))
        if min_index < 0 or max_index >= vertex_count:
            raise ValueError(
                "Surface face indices are out of bounds for "
                f"{vertex_count} vertices: min={min_index}, max={max_index}"
            )
    return faces_int


def _read_vtk_line(handle, *, context: str) -> str:
    line = handle.readline()
    if not line:
        raise ValueError(f"Unexpected end of file while reading {context}")
    try:
        return line.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Could not decode VTK {context} line as ASCII") from exc


def _read_vtk_data_line(handle, *, context: str) -> str:
    while True:
        line = _read_vtk_line(handle, context=context)
        if line:
            return line


def _vtk_point_dtype(vtk_type: str) -> np.dtype:
    vtk_type = vtk_type.lower()
    if vtk_type == "float":
        return np.dtype(">f4")
    if vtk_type == "double":
        return np.dtype(">f8")
    raise ValueError(f"Unsupported VTK point data type {vtk_type!r}")


def read_surface(filename: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a triangular legacy VTK ``POLYDATA`` surface mesh."""

    filename = Path(filename)
    with open(filename, "rb") as f:
        header = _read_vtk_line(f, context="header")
        if not header.startswith("# vtk DataFile"):
            raise ValueError(f"{filename} is not a legacy VTK file")

        _read_vtk_line(f, context="title")
        encoding = _read_vtk_line(f, context="encoding").upper()
        dataset = _read_vtk_data_line(f, context="dataset").upper()
        if dataset != "DATASET POLYDATA":
            raise ValueError(f"Expected DATASET POLYDATA, got {dataset!r}")

        points_line = _read_vtk_data_line(f, context="POINTS")
        points_parts = points_line.split()
        if len(points_parts) != 3 or points_parts[0].upper() != "POINTS":
            raise ValueError(f"Expected POINTS line, got {points_line!r}")
        point_count = int(points_parts[1])
        point_dtype = _vtk_point_dtype(points_parts[2])

        if encoding == "BINARY":
            point_bytes = point_count * 3 * point_dtype.itemsize
            point_payload = f.read(point_bytes)
            if len(point_payload) != point_bytes:
                raise ValueError("Unexpected end of file while reading VTK points")
            vertices = np.frombuffer(point_payload, dtype=point_dtype).reshape(point_count, 3)

            polygons_line = _read_vtk_data_line(f, context="POLYGONS")
            polygons_parts = polygons_line.split()
            if len(polygons_parts) != 3 or polygons_parts[0].upper() != "POLYGONS":
                raise ValueError(f"Expected POLYGONS line, got {polygons_line!r}")
            polygon_count = int(polygons_parts[1])
            polygon_size = int(polygons_parts[2])
            polygon_payload_bytes = polygon_size * np.dtype(">i4").itemsize
            polygon_payload = f.read(polygon_payload_bytes)
            if len(polygon_payload) != polygon_payload_bytes:
                raise ValueError("Unexpected end of file while reading VTK polygons")
            polygon_cells = np.frombuffer(polygon_payload, dtype=">i4")
        elif encoding == "ASCII":
            points = []
            while len(points) < point_count * 3:
                points.extend(_read_vtk_data_line(f, context="ASCII points").split())
            vertices = np.asarray(points[: point_count * 3], dtype=np.float64).reshape(
                point_count,
                3,
            )

            polygons_line = _read_vtk_data_line(f, context="POLYGONS")
            polygons_parts = polygons_line.split()
            if len(polygons_parts) != 3 or polygons_parts[0].upper() != "POLYGONS":
                raise ValueError(f"Expected POLYGONS line, got {polygons_line!r}")
            polygon_count = int(polygons_parts[1])
            polygon_size = int(polygons_parts[2])
            polygons = []
            while len(polygons) < polygon_size:
                polygons.extend(_read_vtk_data_line(f, context="ASCII polygons").split())
            polygon_cells = np.asarray(polygons[:polygon_size], dtype=np.int64)
        else:
            raise ValueError(f"Unsupported legacy VTK encoding {encoding!r}")

    if polygon_count == 0:
        faces = np.empty((0, 3), dtype=np.int64)
    elif polygon_cells.size != polygon_count * 4:
        raise ValueError("Only triangular VTK POLYDATA polygon cells are supported")
    else:
        polygon_cells = polygon_cells.reshape(polygon_count, 4)
        if not np.all(polygon_cells[:, 0] == 3):
            raise ValueError("Only triangular VTK POLYDATA polygon cells are supported")
        faces = polygon_cells[:, 1:].astype(np.int64, copy=False)

    vertices = _validate_vertices(vertices)
    return vertices, _validate_triangle_faces(faces, vertex_count=vertices.shape[0])


def write_surface(vertices, faces, filename: str | Path) -> Path:
    """Write a triangle surface mesh as legacy binary VTK ``POLYDATA``."""

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    vertices = _validate_vertices(vertices)
    faces = _validate_triangle_faces(faces, vertex_count=vertices.shape[0])

    polygon_cells = np.empty((faces.shape[0], 4), dtype=np.int32)
    polygon_cells[:, 0] = 3
    polygon_cells[:, 1:] = faces.astype(np.int32)

    with open(filename, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(b"surface_mesh\n")
        f.write(b"BINARY\n")
        f.write(b"DATASET POLYDATA\n")
        f.write(f"POINTS {vertices.shape[0]} float\n".encode("ascii"))
        f.write(vertices.astype(">f4", copy=False).tobytes(order="C"))
        f.write(b"\n")
        f.write(f"POLYGONS {faces.shape[0]} {faces.shape[0] * 4}\n".encode("ascii"))
        f.write(polygon_cells.astype(">i4", copy=False).tobytes(order="C"))
        f.write(b"\n")
    return filename
