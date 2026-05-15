"""Restricted-Delaunay surface generation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from scipy.spatial import Voronoi
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.transform import resize

from wsi_pipeline.surface.io import write_surface
from wsi_pipeline.surface.resampling import grid_based_sampling_old, robust_delaunay_points
from wsi_pipeline.surface.resampling_utils import (
    _check_tetrahedra,
    _check_topology,
    _deduplicate_faces_and_remove_unused_vertices,
    _tetrahedron_circumcenters,
    orient_surface,
)


def restricted_delaunay(image, points, isoval, dx, verbose=True):
    """Compute restricted-Delaunay faces for an isosurface point cloud."""

    image = np.asarray(image, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    N = 3 # The dimensions of our points
    delaunay_options = 'Qt Qbb Qc' if N <= 3 else 'Qt Qbb Qc Qx' # Set the QHull options
    voronoi_options = 'Qbb' if N <= 3 else 'Qbb Qx' # Set the QHull options

    x_axis = np.arange(image.shape[2], dtype=np.float64) * dx[0]
    y_axis = np.arange(image.shape[1], dtype=np.float64) * dx[1]
    z_axis = np.arange(image.shape[0], dtype=np.float64) * dx[2]

    add_hull_points = True
    triangulation, points_all, mn, span = robust_delaunay_points(
        points,
        qh_base_opts=delaunay_options,
        use_Qx=True,
        first_try_with_QJ=True,
        add_small_hull=add_hull_points,
        verbose=verbose,
    )
    original_point_count = points_all.shape[0] - 8 if add_hull_points else points_all.shape[0]
    tets = triangulation.simplices

    circumcenters_method = "voronoi"
    if circumcenters_method == "voronoi":
        voronoi = Voronoi(triangulation.points, qhull_options=voronoi_options)
        if not add_hull_points:
            circumcenters = voronoi.vertices * span + mn
        else:
            point_region = voronoi.point_region
            regions = voronoi.regions
            voronoi_vertices = voronoi.vertices * span + mn
            tet_to_voronoi_vertex = np.full(tets.shape[0], -1, dtype=int)
            for idx, tet in enumerate(tets):
                shared = None
                for point_idx in tet:
                    region_idx = point_region[point_idx]
                    if region_idx == -1 or len(regions[region_idx]) == 0:
                        shared = set()
                        break
                    candidate = {value for value in regions[region_idx] if value != -1}
                    shared = candidate if shared is None else shared & candidate
                    if not shared:
                        break
                if shared is not None and len(shared) == 1:
                    tet_to_voronoi_vertex[idx] = next(iter(shared))
            circumcenters = np.full((tets.shape[0], 3), np.nan)
            ok = tet_to_voronoi_vertex >= 0
            circumcenters[ok] = voronoi_vertices[tet_to_voronoi_vertex[ok]]
    else:
        circumcenters = _tetrahedron_circumcenters(points, tets)

    faces = _check_tetrahedra(
        image,
        triangulation,
        circumcenters,
        x_axis,
        y_axis,
        z_axis,
        isoval,
        verbose=verbose,
    )
    if add_hull_points and faces.size:
        faces = faces[np.all(faces < original_point_count, axis=1)]

    if verbose:
        chi, genus = _check_topology(points, faces)
        print(f"Euler number before face de-duplication: {chi}")
        print(f"Genus before face de-duplication: {genus}")

    vertices, faces = _deduplicate_faces_and_remove_unused_vertices(points, faces)
    if verbose:
        chi, genus = _check_topology(vertices, faces)
        print(f"Euler number after face de-duplication: {chi}")
        print(f"Genus after face de-duplication: {genus}")
    return faces, vertices


def restricted_delaunay_from_image(image, OPT=None):
    """Compute a restricted-Delaunay triangulation from a 3D segmentation image."""

    if OPT is None:
        OPT = {}

    dx = np.asarray(OPT.get("dx", [1, 1, 1]), dtype=np.float64)
    image = np.asarray(image, dtype=np.float64)
    isoval = float(OPT.get("isoval", np.max(image) / 2))
    dilate = int(OPT.get("dilate", 0))
    min_distance = float(OPT.get("minD", 0.0))
    verbose = bool(OPT.get("verbose", False))

    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image for surface generation, got {image.shape}")
    if not np.isfinite(image).any() or np.nanmax(image) <= isoval:
        raise ValueError("Cannot generate a surface from an empty or sub-threshold image")

    for _ in range(dilate):
        image = convolve(image, np.ones((3, 3, 3)), mode="constant")
        image = (image > 0).astype(np.float64)

    spacing_zyx = (dx[2], dx[1], dx[0])
    verts, faces, _, _ = marching_cubes(image, level=isoval, spacing=spacing_zyx)
    if verbose:
        surface_area = mesh_surface_area(verts, faces)
        print(f"Marching-cubes vertices: {verts.shape[0]}")
        print(f"Marching-cubes faces: {faces.shape[0]}")
        print(f"Marching-cubes surface area: {surface_area}")

    points = grid_based_sampling_old(verts, min_distance)
    if points.shape[0] < 4:
        raise ValueError(
            "Restricted Delaunay needs at least 4 sampled surface points; "
            f"got {points.shape[0]}."
        )

    faces_out, vertices_out = restricted_delaunay(image, points, isoval, dx, verbose=verbose)
    if faces_out.size == 0:
        raise ValueError("Restricted Delaunay produced no surface faces")

    oriented = orient_surface(vertices_out, faces_out)
    if isinstance(oriented, tuple):
        faces_out, vertices_out = oriented
    else:
        faces_out = oriented

    volume = np.sum(
        np.sum(
            vertices_out[faces_out[:, 0]]
            * np.cross(vertices_out[faces_out[:, 1]], vertices_out[faces_out[:, 2]]),
            axis=1,
        )
    ) / 6
    if volume < 0:
        faces_out = faces_out[:, [0, 2, 1]]

    if verbose:
        print("Finished restricted_delaunay_from_image")
    return faces_out, vertices_out


def seg2surf(*args):
    """Use restricted Delaunay triangulation to create surfaces from Analyze images."""

    import nibabel as nib

    filenames = list(args)
    try:
        min_distance = float(filenames[-1])
        filenames = filenames[:-1]
    except (TypeError, ValueError):
        min_distance = np.nan

    faces_result = None
    vertices_result = None
    for filename in filenames:
        filename = str(filename)
        if not filename.endswith(".img"):
            print(f"Warning: {filename} is not an Analyze .img file, skipping")
            continue
        if not Path(filename).is_file():
            print(f"Warning: {filename} does not exist, skipping")
            continue

        img = nib.load(filename)
        image_data = img.get_fdata()
        labels = np.unique(image_data)
        faces_saved = []
        vertices_saved = []
        for label in labels:
            if label == 0:
                continue
            seg_max = 100
            if len(labels) > seg_max:
                image = image_data.astype(np.float64)
            else:
                image = (image_data == label).astype(np.float64)

            dx = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
            z_nonzero = np.nonzero(np.sum(image, axis=(0, 1)))[0]
            y_nonzero = np.nonzero(np.sum(image, axis=(1, 2)))[0]
            x_nonzero = np.nonzero(np.sum(image, axis=(0, 2)))[0]
            if z_nonzero.size == 0 or y_nonzero.size == 0 or x_nonzero.size == 0:
                continue

            zmin = max(z_nonzero[0] - 3, 0)
            zmax = min(z_nonzero[-1] + 3, image.shape[2] - 1)
            ymin = max(y_nonzero[0] - 3, 0)
            ymax = min(y_nonzero[-1] + 3, image.shape[0] - 1)
            xmin = max(x_nonzero[0] - 3, 0)
            xmax = min(x_nonzero[-1] + 3, image.shape[1] - 1)
            image = image[ymin : ymax + 1, xmin : xmax + 1, zmin : zmax + 1]
            origin = np.array([xmin, ymin, zmin], dtype=np.float64) * dx

            image = gaussian_filter(image, sigma=1)
            dx_min = min(dx)
            dx_new = dx_min / 2.0
            new_shape = (
                int(image.shape[0] * dx[1] / dx_new),
                int(image.shape[1] * dx[0] / dx_new),
                int(image.shape[2] * dx[2] / dx_new),
            )
            image = np.squeeze(
                resize(image, new_shape, order=1, mode="constant", cval=0, anti_aliasing=True)
            )
            dx = np.array([dx_new, dx_new, dx_new])
            opts = {
                "dx": dx,
                "minD": np.sqrt(np.sum(np.square(dx)))
                if np.isnan(min_distance)
                else float(min_distance),
                "verbose": False,
            }
            faces, vertices = restricted_delaunay_from_image(image, opts)
            vertices += origin[[2, 1, 0]]

            base_filename = Path(filename).stem
            if len(labels) > seg_max:
                output_filename = f"{base_filename}.vtk"
                write_surface(vertices, faces, output_filename)
                faces_result = faces
                vertices_result = vertices
                break
            label_str = f"_label{int(label) + 1:02d}"
            output_filename = f"{base_filename}{label_str}.vtk"
            write_surface(vertices, faces, output_filename)
            faces_saved.append(faces)
            vertices_saved.append(vertices)

        if len(labels) > 2:
            faces_result = faces_saved
            vertices_result = vertices_saved
    return faces_result, vertices_result
