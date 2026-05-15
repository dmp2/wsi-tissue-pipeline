from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np

from wsi_pipeline.surface import prepare_registration_surface_mesh
from wsi_pipeline.surface import registration as surface_registration
from wsi_pipeline.surface import visualization as surface_visualization
from wsi_pipeline.surface.io import read_surface, write_surface


def _write_vtk(
    path: Path,
    *,
    z_size: int = 3,
    y_size: int = 4,
    x_size: int = 5,
    origin: tuple[float, float, float] = (10.0, 20.0, 30.0),
    spacing: tuple[float, float, float] = (2.0, 3.0, 4.0),
    value_offset: float = 0.0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.arange(z_size * y_size * x_size, dtype=np.float32).reshape(
        z_size,
        y_size,
        x_size,
    )
    header = (
        "# vtk DataFile Version 3.0\n"
        "synthetic\n"
        "BINARY\n"
        "DATASET STRUCTURED_POINTS\n"
        f"DIMENSIONS {x_size} {y_size} {z_size}\n"
        f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n"
        f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\n"
        f"POINT_DATA {x_size * y_size * z_size}\n"
    ).encode()
    payload = bytearray(header)
    for name, array in (
        ("data_000(b)", base + value_offset + 3.0),
        ("data_001(g)", base + value_offset + 2.0),
        ("data_002(r)", base + value_offset + 1.0),
    ):
        payload.extend(f"SCALARS {name} float\nLOOKUP_TABLE default\n".encode())
        payload.extend(np.asarray(array, dtype=">f4").tobytes(order="C"))
        payload.extend(b"\n")
    path.write_bytes(bytes(payload))
    return path


def _write_manifest(path: Path, statuses: list[str]) -> Path:
    payload = {
        "version": 1,
        "entries": [
            {"status": status, "grid_index": idx, "sample_id": f"slice_{idx:04d}.tif"}
            for idx, status in enumerate(statuses)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_plan(registration_output: Path, manifest_path: Path) -> None:
    (registration_output / "resolved_run_plan.json").write_text(
        json.dumps({"manifest_path": str(manifest_path)}),
        encoding="utf-8",
    )


def _read_vtk_polydata(path: Path):
    payload = path.read_bytes()
    offset = 0

    def read_line() -> str:
        nonlocal offset
        newline = payload.index(b"\n", offset)
        line = payload[offset:newline].decode("ascii")
        offset = newline + 1
        return line

    header = [read_line() for _ in range(5)]
    point_count = int(header[4].split()[1])
    point_bytes = point_count * 3 * np.dtype(">f4").itemsize
    points = np.frombuffer(payload[offset : offset + point_bytes], dtype=">f4").reshape(
        point_count,
        3,
    )
    offset += point_bytes
    assert payload[offset : offset + 1] == b"\n"
    offset += 1
    polygons_line = read_line()
    polygon_count = int(polygons_line.split()[1])
    polygon_bytes = polygon_count * 4 * np.dtype(">i4").itemsize
    polygons = np.frombuffer(payload[offset : offset + polygon_bytes], dtype=">i4").reshape(
        polygon_count,
        4,
    )
    return header, points, polygons


def _patch_mesher_and_writer(monkeypatch):
    calls = {}

    def fake_mesher(image, options):
        calls["image"] = np.asarray(image).copy()
        calls["options"] = dict(options)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        vertices_zyx = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        return faces, vertices_zyx

    def fake_write_surface(vertices, faces, filename):
        calls["vertices"] = np.asarray(vertices).copy()
        calls["faces"] = np.asarray(faces).copy()
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text("mesh", encoding="utf-8")
        return filename

    monkeypatch.setattr(surface_registration, "restricted_delaunay_from_image", fake_mesher)
    monkeypatch.setattr(surface_registration, "write_surface", fake_write_surface)
    return calls


def test_surface_modules_import_without_optional_visualization_dependencies(monkeypatch):
    for module_name in ("pyvista", "matplotlib", "nibabel", "torch", "pytorch3d"):
        monkeypatch.setitem(sys.modules, module_name, None)

    for module_name in (
        "wsi_pipeline.surface",
        "wsi_pipeline.surface.surface_generation",
        "wsi_pipeline.surface.resampling",
        "wsi_pipeline.surface.resampling_utils",
    ):
        importlib.import_module(module_name)


def test_write_surface_writes_legacy_vtk_polydata_without_pyvista(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyvista", None)
    vertices = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    output_path = write_surface(vertices, faces, tmp_path / "surface.vtk")

    header, points, polygons = _read_vtk_polydata(output_path)
    assert header == [
        "# vtk DataFile Version 3.0",
        "surface_mesh",
        "BINARY",
        "DATASET POLYDATA",
        "POINTS 3 float",
    ]
    assert np.allclose(points, vertices)
    assert np.array_equal(polygons, [[3, 0, 1, 2]])

    read_vertices, read_faces = read_surface(output_path)
    assert np.allclose(read_vertices, vertices)
    assert np.array_equal(read_faces, faces)


def test_plot_surface_accepts_vtk_path_with_pyvista_backend(tmp_path, monkeypatch):
    vertices = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    output_path = write_surface(vertices, faces, tmp_path / "surface.vtk")

    class FakePolyData:
        def __init__(self, points, cells):
            self.points = np.asarray(points).copy()
            self.cells = np.asarray(cells).copy()

    class FakePlotter:
        def __init__(self):
            self.mesh = None
            self.shown = False

        def add_mesh(self, mesh, **_kwargs):
            self.mesh = mesh

        def add_axes(self):
            pass

        def show_bounds(self):
            pass

        def show(self, **kwargs):
            self.show_kwargs = dict(kwargs)
            self.shown = True

    class FakePyVista:
        PolyData = FakePolyData
        Plotter = FakePlotter

    monkeypatch.setattr(surface_visualization, "_require_pyvista", lambda: FakePyVista)

    plotter = surface_visualization.plot_surface(output_path, backend="pyvista", show=False)

    assert np.allclose(plotter.mesh.points, vertices)
    assert np.array_equal(plotter.mesh.cells.reshape(-1, 4), [[3, 0, 1, 2]])
    assert plotter.shown is False

    plotter = surface_visualization.plot_surface(
        output_path,
        backend="pyvista",
        jupyter_backend="html",
    )
    assert plotter.shown is True
    assert plotter.show_kwargs == {"jupyter_backend": "html"}


def test_prepare_registration_surface_mesh_prefers_filled_volume(tmp_path, monkeypatch):
    registration_output = tmp_path / "emlddmm"
    _write_vtk(
        registration_output / "self_alignment" / "images" / "target_registered.vtk",
        z_size=3,
        value_offset=0.0,
    )
    _write_vtk(
        registration_output / "upsampling" / "filled_volume.vtk",
        z_size=5,
        spacing=(5.0, 6.0, 7.0),
        value_offset=100.0,
    )
    calls = _patch_mesher_and_writer(monkeypatch)

    output_path = prepare_registration_surface_mesh(registration_output)

    assert output_path == registration_output / "visualization" / "surface_mesh.vtk"
    assert output_path.exists()
    assert calls["image"].shape == (5, 4, 5)
    assert np.allclose(calls["options"]["dx"], [5.0, 6.0, 7.0])
    assert calls["options"]["isoval"] == 0.5
    assert np.isclose(calls["options"]["minD"], np.sqrt(5.0**2 + 6.0**2 + 7.0**2))
    assert np.allclose(
        calls["vertices"],
        [
            [10.0, 20.0, 30.0],
            [13.0, 22.0, 31.0],
            [16.0, 25.0, 34.0],
        ],
    )


def test_prepare_registration_surface_mesh_falls_back_to_registered_without_z_filling(
    tmp_path,
    monkeypatch,
):
    registration_output = tmp_path / "emlddmm"
    _write_vtk(registration_output / "self_alignment" / "images" / "target_registered.vtk")
    manifest_path = _write_manifest(
        tmp_path / "emlddmm_dataset_manifest.json",
        [
            "present",
            "missing",
            "present",
        ],
    )
    _write_plan(registration_output, manifest_path)
    calls = _patch_mesher_and_writer(monkeypatch)

    prepare_registration_surface_mesh(registration_output)

    assert calls["image"].shape == (3, 4, 5)
    assert np.any(calls["image"][0] > 0)
    assert np.all(calls["image"][1] == 0)
    assert np.any(calls["image"][2] > 0)
    assert np.allclose(calls["options"]["dx"], [2.0, 3.0, 4.0])


def test_prepare_registration_surface_mesh_does_not_require_pyvista_to_write(
    tmp_path,
    monkeypatch,
):
    registration_output = tmp_path / "emlddmm"
    _write_vtk(registration_output / "self_alignment" / "images" / "target_registered.vtk")
    calls = {"mesher": 0}

    def fake_mesher(_image, _options):
        calls["mesher"] += 1
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        vertices_zyx = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        return faces, vertices_zyx

    monkeypatch.setattr(surface_registration, "restricted_delaunay_from_image", fake_mesher)
    monkeypatch.setitem(sys.modules, "pyvista", None)

    def missing_pyvista():
        raise ImportError(
            'PyVista/VTK are not installed. Install with: pip install -e ".[visualization]"'
        )

    monkeypatch.setattr(surface_registration, "_require_pyvista", missing_pyvista)

    output_path = prepare_registration_surface_mesh(registration_output)

    assert output_path.exists()
    assert calls["mesher"] == 1
