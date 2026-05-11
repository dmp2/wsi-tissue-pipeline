"""Utilities for restricted-Delaunay surface generation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def orient_surface(
    vertices,
    faces,
    *,
    backend: str | None = "auto",
    indexing: str = "zero",
    epsilon: float = 1e-9,
    rng=None,
):
    """Orient triangle faces consistently using a ray-parity test."""

    vertices = np.asarray(vertices, dtype=float)
    faces_in = np.asarray(faces, dtype=int)
    if faces_in.size == 0:
        return faces_in.reshape(0, 3)

    if indexing not in {"auto", "zero", "one"}:
        raise ValueError('indexing must be "auto", "zero", or "one"')

    restore_one_based = False
    faces_work = faces_in.copy()
    if indexing == "auto":
        if faces_work.min() >= 1 and faces_work.max() <= len(vertices):
            faces_work = faces_work - 1
            restore_one_based = True
    elif indexing == "one":
        faces_work = faces_work - 1
        restore_one_based = True

    if backend == "trimesh":
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError("trimesh is not available; cannot use trimesh backend.") from exc

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces_work, process=False)
        try:
            trimesh.repair.fix_winding(mesh)
        except Exception:
            try:
                mesh.fix_normals()
            except Exception:
                pass
        faces_oriented = mesh.faces.copy()
        if restore_one_based:
            faces_oriented = faces_oriented + 1
        return faces_oriented, np.asarray(mesh.vertices, dtype=np.float64)

    rng = rng if rng is not None else np.random.default_rng()
    ray = rng.normal(size=3)
    norm = np.linalg.norm(ray)
    ray = np.array([1.0, 0.0, 0.0]) if norm == 0 else ray / norm

    v0 = vertices[faces_work[:, 0], :]
    v1 = vertices[faces_work[:, 1], :]
    v2 = vertices[faces_work[:, 2], :]
    centroids = (v0 + v1 + v2) / 3.0

    normals = np.cross(v1 - v0, v2 - v0) * 0.5
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    safe = normal_norm.squeeze() > 0
    unit_normals = np.zeros_like(normals)
    unit_normals[safe] = normals[safe] / normal_norm[safe]
    test_points = centroids + epsilon * unit_normals

    counts = np.zeros(faces_work.shape[0], dtype=np.int64)
    for idx in range(faces_work.shape[0]):
        q = vertices[faces_work[idx, 0], :]
        side_1 = vertices[faces_work[idx, 1], :] - vertices[faces_work[idx, 0], :]
        side_2 = vertices[faces_work[idx, 2], :] - vertices[faces_work[idx, 0], :]
        system = np.vstack((-ray, side_1, side_2))
        try:
            solved = np.linalg.solve(system.T, (test_points - q).T).T
        except np.linalg.LinAlgError:
            continue

        distance = solved[:, 0]
        bary_1 = solved[:, 1]
        bary_2 = solved[:, 2]
        bary = np.column_stack([1.0 - (bary_1 + bary_2), bary_1, bary_2])
        inside = (np.abs(bary - 0.5) <= 0.5).sum(axis=1) == 3
        counts += (inside & (distance >= 0.0)).astype(np.int64)

    flip_mask = (counts % 2) == 1
    faces_oriented = faces_work.copy()
    faces_oriented[flip_mask] = faces_oriented[flip_mask][:, [1, 0, 2]]
    if restore_one_based:
        faces_oriented = faces_oriented + 1
    return faces_oriented


def calculate_surface_area(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    cross_prod = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    return float(0.5 * np.linalg.norm(cross_prod, axis=1).sum())


def identify_thin_surfaces(vertices, faces, thickness_threshold=0.5):
    """Identify faces whose projected mesh thickness is below a threshold."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    thin_faces = []
    for face in faces:
        face_normal = np.cross(
            vertices[face[1]] - vertices[face[0]],
            vertices[face[2]] - vertices[face[0]],
        )
        norm = np.linalg.norm(face_normal)
        if norm == 0:
            thin_faces.append(face)
            continue
        face_normal /= norm
        thickness = np.max(np.abs(np.dot(vertices - vertices[face[0]], face_normal)))
        if thickness < thickness_threshold:
            thin_faces.append(face)
    return thin_faces


def _normalize_unit_box(points):
    points = np.asarray(points, np.float64)
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    span = mx - mn
    span[span == 0.0] = 1.0
    return (points - mn) / span, mn, span


def _denormalize_unit_box(points, mn, span):
    return points * span + mn


def _unique_rows_tol_grid(points, tol=1e-12):
    """Cull near-duplicate rows by quantizing with a bbox-relative tolerance."""

    points = np.asarray(points, np.float64)
    if points.size == 0:
        return points
    diag = np.linalg.norm(np.ptp(points, axis=0))
    scale = tol * (diag if diag > 0 else 1.0)
    if scale == 0:
        return points
    quantized = np.round(points / scale)
    _, idx = np.unique(quantized, axis=0, return_index=True)
    return points[np.sort(idx)]


def _tiny_jitter(points, mag=1e-9, seed=0):
    points = np.asarray(points, np.float64)
    rng = np.random.default_rng(seed)
    span = np.ptp(points, axis=0)
    scale = np.linalg.norm(span) or 1.0
    return points + rng.standard_normal(points.shape) * (mag * scale)


def _tetrahedron_circumcenters(points, tets, eps_vol=0.0):
    """Compute tetrahedron circumcenters for ``points[tets]``."""

    points = np.asarray(points, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    p1 = points[tets[:, 0], :]
    p2 = points[tets[:, 1], :]
    p3 = points[tets[:, 2], :]
    p4 = points[tets[:, 3], :]
    a = p2 - p1
    b = p3 - p1
    c = p4 - p1
    vols6 = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
    tol = max(1e-15, np.finfo(np.float64).eps)

    matrix = np.stack([a, b, c], axis=1).astype(np.float64, copy=False)
    rhs = 0.5 * np.stack(
        [
            np.einsum("ij,ij->i", a, a),
            np.einsum("ij,ij->i", b, b),
            np.einsum("ij,ij->i", c, c),
        ],
        axis=1,
    ).astype(np.float64, copy=False)

    offsets = np.empty_like(rhs)
    keep = vols6 > (eps_vol if eps_vol > 0.0 else tol)
    offsets[~keep] = np.nan
    if keep.any():
        offsets[keep] = np.linalg.solve(matrix[keep], rhs[keep, :, None])[..., 0]
    if (~keep).any():
        bad = ~keep
        u, s, vt = np.linalg.svd(matrix[bad], full_matrices=False)
        u_rhs = np.einsum("...ij,...j->...i", u.transpose(0, 2, 1), rhs[bad])
        s_inv = np.where(s > tol, 1.0 / s, 0.0)
        offsets[bad] = np.einsum("...ij,...j->...i", vt.transpose(0, 2, 1), u_rhs * s_inv)
    return p1 + offsets


def _check_topology(vertices, faces):
    faces = np.asarray(faces, dtype=np.int64)
    vertices = np.asarray(vertices)
    if faces.size == 0:
        return int(vertices.shape[0]), 1 - int(vertices.shape[0]) / 2
    edges = np.sort(
        np.vstack([faces[:, [0, 1]], faces[:, [0, 2]], faces[:, [1, 2]]]),
        axis=1,
    )
    unique_edges = np.unique(edges, axis=0)
    chi = vertices.shape[0] - unique_edges.shape[0] + faces.shape[0]
    genus = 1 - chi / 2
    return chi, genus


def _deduplicate_faces_and_remove_unused_vertices(vertices, faces):
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return vertices[:0], faces.reshape(0, 3).astype(np.int32)
    faces_sorted = np.sort(faces, axis=1)
    faces_unique = np.unique(faces_sorted, axis=0)
    used, inv = np.unique(faces_unique.ravel(), return_inverse=True)
    faces_out = inv.reshape(faces_unique.shape).astype(np.int32)
    return vertices[used], faces_out


def segment_crosses_isovalue(
    p0: np.ndarray,
    p1: np.ndarray,
    interpolator,
    isoval: float,
    n: int = 2,
    eps: float = 1e-12,
    coord_map: Callable[[np.ndarray], np.ndarray] | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
    """Vectorized n-point crossing test for segments against an isovalue."""

    if n < 2:
        raise ValueError("n must be >= 2")
    if p1.shape != p0.shape or p0.shape[1] != 3:
        raise ValueError("p0 and p1 must be shaped (K, 3)")

    p0 = np.asarray(p0, dtype=np.float64, order="C")
    p1 = np.asarray(p1, dtype=np.float64, order="C")
    if coord_map is None:
        def coord_map(array):
            return array

    t = np.linspace(0.0, 1.0, n, dtype=np.float64).reshape(n, 1, 1)

    def process(lo: int, hi: int) -> np.ndarray:
        start = p0[lo:hi]
        delta = p1[lo:hi] - start
        points = coord_map((start[None, :, :] + t * delta[None, :, :]).reshape(-1, 3))
        values = interpolator(points).reshape(n, -1).T
        if n == 2:
            s0 = values[:, 0] - isoval
            s1 = values[:, 1] - isoval
            return ((s0 > eps) & (s1 < -eps)) | ((s0 < -eps) & (s1 > eps))
        return (values.min(axis=1) <= isoval - eps) & (values.max(axis=1) >= isoval + eps)

    if batch_size is None or batch_size >= p0.shape[0]:
        return process(0, p0.shape[0])

    keep = np.zeros(p0.shape[0], dtype=bool)
    for lo in range(0, p0.shape[0], batch_size):
        hi = min(lo + batch_size, p0.shape[0])
        keep[lo:hi] = process(lo, hi)
    return keep


def _check_tetrahedra(image, triangulation, circumcenters, x_axis, y_axis, z_axis, isoval, verbose=False):
    """Keep Delaunay faces whose dual circumcenter edge crosses the image isovalue."""

    tets = triangulation.simplices
    neighbors = triangulation.neighbors
    interpolator = RegularGridInterpolator(
        (z_axis, y_axis, x_axis),
        image,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )

    faces_idx = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    faces_all = tets[:, faces_idx]
    pairs_all = np.stack(
        [np.repeat(np.arange(tets.shape[0]), 4), neighbors.ravel()],
        axis=1,
    )
    mask_interior = (pairs_all[:, 1] != -1) & (pairs_all[:, 0] < pairs_all[:, 1])
    face_candidates = faces_all.reshape(-1, 3)[mask_interior]
    pair_used = pairs_all[mask_interior]
    if face_candidates.size == 0:
        return face_candidates.reshape(0, 3)

    dual_edges = np.stack(
        [circumcenters[pair_used[:, 0]], circumcenters[pair_used[:, 1]]],
        axis=1,
    )
    finite = np.isfinite(dual_edges).all(axis=(1, 2))
    if not np.any(finite):
        return face_candidates[:0]

    keep = segment_crosses_isovalue(
        dual_edges[finite, 0, :],
        dual_edges[finite, 1, :],
        interpolator=interpolator,
        isoval=isoval,
        n=3,
        eps=1e-12,
        batch_size=200_000,
    )
    out = face_candidates[finite][keep]
    if verbose:
        print(f"Kept {out.shape[0]} restricted-Delaunay faces")
    return out
