"""Point resampling helpers for surface generation."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import Delaunay, KDTree, QhullError, cKDTree
from skimage.measure import mesh_surface_area

from wsi_pipeline.surface.resampling_utils import (
    _normalize_unit_box,
    _tiny_jitter,
    _unique_rows_tol_grid,
)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def select_vertices_min_distance(
    verts,
    faces,
    minD,
    nIdeal,
    tol=1e-13,
    smooth=True,
    workers=1,
    verbose=False,
):
    """Select a vertex subset with an approximate minimum-distance constraint."""

    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    surface_area = mesh_surface_area(verts, faces)
    dmin = max(float(np.sqrt(surface_area / nIdeal)), float(minD))
    nmax = int(surface_area / dmin / dmin * 3)

    if verbose:
        print("Surface Area:", surface_area)
        print("Estimated dmin:", dmin)
        print("nmax:", nmax)

    tree = KDTree(verts)
    selected_indices: list[int] = []
    remaining_indices = np.arange(verts.shape[0])
    while len(selected_indices) < nmax and len(remaining_indices) > 0:
        idx = int(remaining_indices[0])
        selected_indices.append(idx)
        neighbors = tree.query_ball_point(verts[idx], r=dmin, eps=tol, workers=workers)
        remaining_indices = np.setdiff1d(remaining_indices, neighbors, assume_unique=True)

    selected = verts[selected_indices]
    if smooth and len(selected) > 1:
        tree = cKDTree(selected)
        _, closest = tree.query(verts, workers=workers)
        selected = np.vstack(
            [
                verts[closest == idx].mean(axis=0) if np.any(closest == idx) else selected[idx]
                for idx in range(len(selected))
            ]
        )
    return selected


def radius_greedy_mis_fast(verts, dmin, workers=1, seed=None):
    """Randomized greedy maximal independent set on a radius graph."""

    verts = np.asarray(verts, np.float32)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(verts))
    tree = cKDTree(verts)
    alive = np.ones(len(verts), dtype=bool)
    chosen = []
    for idx in order:
        if not alive[idx]:
            continue
        chosen.append(idx)
        alive[tree.query_ball_point(verts[idx], dmin - 1e-7, workers=workers)] = False
    return verts[np.array(chosen, dtype=np.int64)]


def select_vertices_sequential(verts, faces, minD, nIdeal, smooth=True, tol=1e-13, verbose=False):
    """Sequential minimum-distance vertex selection."""

    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    surface_area = mesh_surface_area(verts, faces)
    dmin = max(float(np.sqrt(surface_area / nIdeal)), float(minD))
    dmin2 = dmin * dmin
    nmax = int(surface_area / dmin / dmin * 3)

    selected = np.zeros((nmax, 3), dtype=np.float32)
    selected[0, :] = verts[0, :]
    count = 1
    dist2 = np.full((verts.shape[0],), np.inf)
    while count < nmax:
        new_dist2 = np.sum((verts - selected[count - 1, :]) ** 2, axis=1)
        dist2 = np.minimum(dist2, new_dist2)
        candidates = np.where(dist2 > dmin2 + tol)[0]
        if len(candidates) == 0:
            break
        idx = candidates[np.argmin(dist2[candidates])]
        selected[count, :] = verts[idx, :]
        count += 1

    selected = selected[:count, :]
    if smooth and count > 1:
        tree = KDTree(selected)
        _, closest = tree.query(verts, workers=3)
        selected = np.vstack(
            [
                verts[closest == idx].mean(axis=0) if np.any(closest == idx) else selected[idx]
                for idx in range(count)
            ]
        )
    if verbose:
        print(f"Selected {selected.shape[0]} vertices")
    return selected


def select_vertices_original(verts, faces, minD, nIdeal, average=True, tol=1e-13, verbose=False):
    """Original notebook-style distance-threshold sampler."""

    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    surface_area = mesh_surface_area(verts, faces)
    dmin = max(float(np.sqrt(surface_area / nIdeal)), float(minD))
    dmin2 = dmin * dmin
    nmax = max(1, int(surface_area / dmin / dmin * 3))

    selected = np.empty((nmax, 3), dtype=np.float32)
    selected[0, :] = verts[0, :]
    count = 1
    dist2 = np.empty((nmax, verts.shape[0]), dtype=np.float32)
    while count < nmax:
        dist2[count - 1, :] = np.sum((selected[count - 1, :].reshape(3, 1) - verts.T) ** 2, axis=0)
        mindist = np.min(dist2[:count, :], axis=0)
        valid = mindist > dmin2
        if not np.any(valid):
            break
        min_valid = np.min(mindist[valid])
        candidates = np.flatnonzero(np.abs(mindist - min_valid) < tol)
        if len(candidates) == 0:
            break
        selected[count, :] = verts[int(candidates[0]), :]
        count += 1

    selected = selected[:count, :]
    if average and count > 1:
        tree = KDTree(selected)
        _, closest = tree.query(verts, workers=3)
        selected = np.vstack(
            [
                verts[closest == idx].mean(axis=0) if np.any(closest == idx) else selected[idx]
                for idx in range(count)
            ]
        )
    if verbose:
        print(f"Selected {count} vertices")
    return selected, count


def _hexpack_fill(surface_area, minD=None, k=None):
    """Make minimum distance and sample count roughly consistent by hex packing."""

    if minD is None and k is None:
        raise ValueError("Provide at least one of minD or k.")
    if minD is not None and k is not None:
        return float(minD), int(k)
    if minD is None:
        dmin = math.sqrt((2.0 / math.sqrt(3.0)) * surface_area / float(k))
        return float(dmin), int(k)
    k_est = (2.0 / math.sqrt(3.0)) * surface_area / (float(minD) ** 2)
    return float(minD), int(round(k_est))


def fps_threshold(
    verts,
    faces,
    minD=None,
    k=None,
    seed=0,
    device=None,
    return_indices=False,
    verbose=False,
    eps=1e-7,
):
    """Farthest-point sampling with early stopping at minD or fixed k."""

    verts_np = np.asarray(verts, dtype=np.float32)
    if verts_np.shape[0] == 0:
        empty_idx = np.array([], dtype=int)
        return (verts_np, empty_idx) if return_indices else verts_np

    if faces is not None:
        surface_area = float(mesh_surface_area(verts_np, np.asarray(faces, dtype=np.int64)))
        minD, k = _hexpack_fill(surface_area, minD=minD, k=k)
        if verbose:
            print(f"[fps] SA={surface_area:.3f}, minD={minD:.6g}, k={k}")

    use_torch = _torch_available() and device not in {"numpy", "none"}
    if not use_torch:
        if device not in {None, "cpu", "numpy", "none"}:
            raise RuntimeError("PyTorch is not available for the requested FPS device.")
        centroid = verts_np.mean(axis=0)
        d2 = np.sum((verts_np - centroid) ** 2, axis=1)
        start = int(np.argmax(d2))
        chosen = [start]
        d2 = np.sum((verts_np - verts_np[start]) ** 2, axis=1)
        r2 = None if minD is None else float(minD * minD)
        kmax = verts_np.shape[0] if k is None else int(k)
        while len(chosen) < kmax:
            idx = int(np.argmax(d2))
            if r2 is not None and d2[idx] < r2 - eps:
                break
            chosen.append(idx)
            diff = verts_np - verts_np[idx]
            d2 = np.minimum(d2, np.einsum("ij,ij->i", diff, diff))
        indices = np.array(chosen, dtype=np.int64)
        points = verts_np[indices]
        return (points, indices) if return_indices else points

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    verts_t = torch.as_tensor(verts_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        centroid = verts_t.mean(dim=0)
        d2 = torch.sum((verts_t - centroid) ** 2, dim=1)
        start = int(torch.argmax(d2))
        chosen = [start]
        d2 = torch.sum((verts_t - verts_t[start]) ** 2, dim=1)
        r2 = None if minD is None else float(minD * minD)
        kmax = verts_np.shape[0] if k is None else int(k)
        while len(chosen) < kmax:
            idx = int(torch.argmax(d2))
            if r2 is not None and float(d2[idx]) < r2 - eps:
                break
            chosen.append(idx)
            diff = verts_t - verts_t[idx]
            d2 = torch.minimum(d2, torch.sum(diff * diff, dim=1))
        indices_t = torch.tensor(chosen, dtype=torch.int64, device=device)
        points = verts_t.index_select(0, indices_t).detach().cpu().numpy()
        indices = indices_t.cpu().numpy()
    return (points, indices) if return_indices else points


def _grid_hash(points, cell_size):
    import torch

    pmin = points.min(0).values
    gcoords = torch.floor((points - pmin) / cell_size).to(torch.int64)
    gmin = gcoords.min(0).values
    shifted = gcoords - gmin
    grng = shifted.max(0).values + 1
    sy = max(int(torch.ceil(torch.log2(grng[0:1].clamp_min(2).float())).item()), 21)
    sz = max(int(torch.ceil(torch.log2(grng[1:2].clamp_min(2).float())).item()) + sy, 42)
    keys = (shifted[:, 0] << sy) ^ shifted[:, 1] ^ (shifted[:, 2] << sz)
    return keys, shifted, pmin, (sy, sz)


def _group_by_key(keys):
    import torch

    uniq, inv = torch.unique(keys, sorted=True, return_inverse=True)
    order = torch.argsort(inv)
    inv_sorted = inv[order]
    _, counts = torch.unique_consecutive(inv_sorted, return_counts=True)
    starts = torch.cat([torch.tensor([0], device=keys.device), counts.cumsum(0)[:-1]])
    return uniq, inv, order, starts, counts


def _enumerate_neighbor_codes(cell_coords_unique, shifts, halo_cells=1):
    import torch

    device = cell_coords_unique.device
    sy, sz = shifts
    coord_to_id = {}
    for idx in range(cell_coords_unique.shape[0]):
        cx, cy, cz = cell_coords_unique[idx].tolist()
        coord_to_id[(int(cx) << sy) ^ int(cy) ^ (int(cz) << sz)] = idx

    offsets = torch.stack(
        torch.meshgrid(
            torch.arange(-halo_cells, halo_cells + 1, device=device),
            torch.arange(-halo_cells, halo_cells + 1, device=device),
            torch.arange(-halo_cells, halo_cells + 1, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    neighbors = [[] for _ in range(cell_coords_unique.shape[0])]
    for idx in range(cell_coords_unique.shape[0]):
        base = cell_coords_unique[idx]
        for offset in offsets:
            nb = base + offset
            key = (int(nb[0]) << sy) ^ int(nb[1]) ^ (int(nb[2]) << sz)
            neighbor_idx = coord_to_id.get(key)
            if neighbor_idx is not None:
                neighbors[idx].append(neighbor_idx)
    return neighbors


def _pack_padded(points_sorted, order, starts, counts):
    import torch

    device = points_sorted.device
    rows = counts.numel()
    maxlen = int(counts.max().item())
    padded = points_sorted.new_zeros((rows, maxlen, 3))
    idx_padded = torch.full((rows, maxlen), -1, dtype=torch.int64, device=device)
    for row in range(rows):
        count = int(counts[row].item())
        if count == 0:
            continue
        start = starts[row]
        padded[row, :count] = points_sorted[start : start + count]
        idx_padded[row, :count] = order[start : start + count]
    return padded, idx_padded, counts.clone()


def _distribute_quotas(counts, target_K, weights=None):
    import torch

    device = counts.device
    weights_t = counts.float() if weights is None else weights.float().clamp_min(0)
    if weights_t.sum() == 0:
        quotas = torch.zeros_like(counts, dtype=torch.int64)
        quotas[counts.argmax()] = min(int(target_K), int(counts.max().item()))
        return quotas

    raw = weights_t / weights_t.sum() * target_K
    quotas = torch.minimum(torch.floor(raw).to(torch.int64), counts.to(torch.int64))
    remainder = target_K - int(quotas.sum().item())
    if remainder > 0:
        frac = raw - quotas.float()
        cap_mask = quotas < counts
        frac = torch.where(cap_mask, frac, torch.tensor(-1.0, device=device))
        k = min(remainder, int(cap_mask.sum().item()))
        if k > 0:
            quotas[torch.topk(frac, k=k, largest=True).indices] += 1
    return quotas


def _radius_groups(points, ref_idx, radius, cell_size=None):
    import torch

    if cell_size is None:
        cell_size = max(radius, 1e-6)
    _, gcoords, _pmin, shifts = _grid_hash(points, cell_size)
    sy, sz = shifts
    key_all = (gcoords[:, 0] << sy) ^ gcoords[:, 1] ^ (gcoords[:, 2] << sz)
    _, inv_point_to_cell = torch.unique(key_all, return_inverse=True)
    uniq_coords = torch.unique(gcoords, dim=0)
    neighbors = _enumerate_neighbor_codes(uniq_coords, shifts, halo_cells=1)
    groups = []
    r2 = radius * radius
    for idx in ref_idx.tolist():
        cid = int(inv_point_to_cell[idx].item())
        cand_points = [
            torch.nonzero(inv_point_to_cell == nb, as_tuple=False).squeeze(1)
            for nb in (neighbors[cid] if cid < len(neighbors) else [cid])
        ]
        cand_idx = torch.cat(cand_points) if cand_points else torch.empty(0, dtype=torch.int64)
        if cand_idx.numel() == 0:
            groups.append(torch.tensor([idx], device=points.device, dtype=torch.int64))
            continue
        d2 = (points[cand_idx] - points[idx]).pow(2).sum(-1)
        keep = cand_idx[d2 <= r2]
        groups.append(keep if keep.numel() else torch.tensor([idx], device=points.device))
    return groups


def _surface_lloyd_one_step(points, chosen_idx, radius, snap_to_input=True):
    if chosen_idx.numel() == 0:
        return chosen_idx
    groups = _radius_groups(points, chosen_idx, radius)
    new_pos = [points[group].mean(dim=0, keepdim=True) for group in groups]
    new_pos = points.new_zeros((0, 3)) if not new_pos else __import__("torch").cat(new_pos, dim=0)
    if not snap_to_input:
        return chosen_idx
    snapped = []
    for idx, group in enumerate(groups):
        diff = points[group] - new_pos[idx]
        snapped.append(group[__import__("torch").argmin((diff * diff).sum(-1))])
    return __import__("torch").stack(snapped, dim=0)


def hybrid_fps_pytorch3d(
    points,
    K: int,
    dmin: float,
    *,
    device: str | None = None,
    cell_size: float | None = None,
    halo_cells: int = 1,
    use_halo: bool = True,
    area_weights_per_point=None,
    run_surface_lloyd: bool = False,
    lloyd_radius_factor: float = 1.0,
):
    """Grid-partitioned PyTorch3D farthest-point sampling."""

    try:
        import torch
        from pytorch3d.ops import sample_farthest_points
    except ImportError as exc:
        raise ImportError("PyTorch3D is required for hybrid_fps_pytorch3d.") from exc

    with torch.no_grad():
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if K >= points.shape[0]:
            return torch.arange(points.shape[0], device=points.device)

        device = device or points.device
        points = points.to(device)
        if area_weights_per_point is not None:
            area_weights_per_point = area_weights_per_point.to(device)
        cell_size = float(dmin) if cell_size is None else float(cell_size)

        keys, gcoords, _pmin, shifts = _grid_hash(points, cell_size)
        _uniq_keys, inv, order, starts, counts = _group_by_key(keys)
        if area_weights_per_point is None:
            per_cell_weight = counts.float().clone()
        else:
            per_cell_weight = torch.zeros_like(counts, dtype=torch.float32)
            per_cell_weight.scatter_add_(0, inv[order], area_weights_per_point[order])

        if use_halo and halo_cells > 0:
            cell_first = order[starts]
            uniq_cell_coords = gcoords[cell_first]
            neighbors = _enumerate_neighbor_codes(uniq_cell_coords, shifts, halo_cells=halo_cells)
            cell_to_points = []
            for _cid, neighbor_ids in enumerate(neighbors):
                mask = torch.zeros(points.shape[0], dtype=torch.bool, device=device)
                for nb in neighbor_ids:
                    mask |= inv == nb
                cell_to_points.append(torch.nonzero(mask, as_tuple=False).squeeze(1))
            maxlen = (
                max(int(indices.numel()) for indices in cell_to_points) if cell_to_points else 0
            )
            padded = points.new_zeros((len(cell_to_points), maxlen, 3))
            idx_padded = torch.full(
                (len(cell_to_points), maxlen),
                -1,
                dtype=torch.int64,
                device=device,
            )
            lengths = torch.zeros(len(cell_to_points), dtype=torch.int64, device=device)
            for row, indices in enumerate(cell_to_points):
                n_row = int(indices.numel())
                if n_row == 0:
                    continue
                padded[row, :n_row] = points[indices]
                idx_padded[row, :n_row] = indices
                lengths[row] = n_row
            quotas = _distribute_quotas(counts, K, weights=per_cell_weight)
            keep = lengths > 0
            padded = padded[keep]
            idx_padded = idx_padded[keep]
            lengths = lengths[keep]
            quotas = quotas[keep]
        else:
            points_sorted = points[order]
            padded, idx_padded, lengths = _pack_padded(points_sorted, order, starts, counts)
            quotas = _distribute_quotas(counts, K, weights=per_cell_weight)

        if lengths.numel() == 0 or int(lengths.sum().item()) == 0 or quotas.sum() == 0:
            _, idx = sample_farthest_points(points[None, ...], K=K)
            return idx[0]

        k_max = int(quotas.max().item())
        _, local_idx = sample_farthest_points(padded, K=k_max, lengths=lengths)
        candidates = []
        for row in range(padded.shape[0]):
            quota = int(quotas[row].item())
            if quota <= 0:
                continue
            candidates.append(idx_padded[row].gather(0, local_idx[row, :quota]))
        if not candidates:
            _, idx = sample_farthest_points(points[None, ...], K=K)
            return idx[0]
        candidates = torch.unique(torch.cat(candidates), sorted=False)
        if run_surface_lloyd and candidates.numel() > 0:
            candidates = _surface_lloyd_one_step(
                points,
                candidates,
                radius=float(lloyd_radius_factor) * float(dmin),
                snap_to_input=True,
            )
            candidates = torch.unique(candidates, sorted=False)
        k_final = min(K, candidates.numel())
        if k_final == 0:
            _, idx = sample_farthest_points(points[None, ...], K=K)
            return idx[0]
        _, idx_final_local = sample_farthest_points(points[candidates][None, :, :], K=k_final)
        return candidates[idx_final_local[0]]


def grid_based_sampling_old(verts, dmin):
    """Grid-based Poisson-disk-like sampling."""

    verts = np.asarray(verts, dtype=np.float32)
    if verts.size == 0:
        return verts.reshape(0, 3)
    if dmin <= 0:
        return verts
    cell_size = dmin / np.sqrt(3)
    min_corner = verts.min(axis=0)
    grid = {}
    selected = []
    for point in verts:
        grid_coord = tuple(((point - min_corner) / cell_size).astype(int))
        is_valid = True
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor = (grid_coord[0] + dx, grid_coord[1] + dy, grid_coord[2] + dz)
                    if neighbor in grid and np.linalg.norm(point - grid[neighbor]) < dmin:
                        is_valid = False
                        break
                if not is_valid:
                    break
            if not is_valid:
                break
        if is_valid:
            grid[grid_coord] = point
            selected.append(point)
    return np.asarray(selected, dtype=np.float32)


def grid_based_sampling(verts, dmin, seed=None):
    """Greedy grid-based culling with a correct neighbor reach."""

    verts = np.asarray(verts, dtype=np.float32)
    if verts.size == 0 or dmin <= 0:
        return verts
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(verts))
    shuffled = verts[order]
    cell = dmin / np.sqrt(3.0)
    inv_cell = 1.0 / cell
    r2 = dmin * dmin
    min_corner = shuffled.min(axis=0) - rng.uniform(0.0, cell, size=3).astype(np.float32)
    reach = int(math.ceil(dmin / cell))
    grid = {}
    chosen_idx = []
    for shuffled_idx, point in enumerate(shuffled):
        coord = ((point - min_corner) * inv_cell).astype(np.int32)
        ok = True
        for dx in range(-reach, reach + 1):
            if not ok:
                break
            for dy in range(-reach, reach + 1):
                if not ok:
                    break
                for dz in range(-reach, reach + 1):
                    neighbor = grid.get((coord[0] + dx, coord[1] + dy, coord[2] + dz))
                    if neighbor is not None and np.dot(point - neighbor, point - neighbor) < r2:
                        ok = False
                        break
        if ok:
            grid[(coord[0], coord[1], coord[2])] = point
            chosen_idx.append(order[shuffled_idx])
    return verts[np.array(chosen_idx, dtype=np.int64)]


def robust_delaunay_points(
    points_in,
    qh_base_opts="Qt Qbb Qc",
    use_Qx=True,
    first_try_with_QJ=True,
    tol_dedupe=1e-12,
    jitter_seq=(1e-10, 5e-10, 1e-9),
    add_small_hull=False,
    hull_scale=1.5,
    verbose=False,
):
    """Run Delaunay with normalization, de-duplication, and Qhull fallbacks."""

    points = np.asarray(points_in, np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 4:
        raise ValueError("Need at least 4 points for 3D Delaunay")

    normalized, mn, span = _normalize_unit_box(points)
    normalized = _unique_rows_tol_grid(normalized, tol=tol_dedupe)
    if add_small_hull:
        corners = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=np.float64,
        )
        normalized_all = np.vstack([normalized, 0.5 + (corners - 0.5) * hull_scale])
    else:
        normalized_all = normalized

    opts = qh_base_opts
    if use_Qx and "Qx" not in opts:
        opts = f"{opts} Qx"
    opts_try = f"{opts} QJ" if first_try_with_QJ and "QJ" not in opts else opts

    def try_delaunay(candidate_points, candidate_opts):
        if verbose:
            print(f"[Delaunay] trying opts='{candidate_opts}' on n={len(candidate_points)}")
        return Delaunay(candidate_points, qhull_options=candidate_opts)

    try:
        triangulation = try_delaunay(normalized_all, opts_try)
    except QhullError as exc:
        if verbose:
            print(f"[Delaunay] failed with opts='{opts_try}': {type(exc).__name__}")
        triangulation = None
        for mag in jitter_seq:
            try:
                triangulation = try_delaunay(_tiny_jitter(normalized_all, mag=mag), opts_try)
                break
            except QhullError:
                triangulation = None
        if triangulation is None:
            try:
                final_opts = opts if "QJ" in opts else f"{opts} QJ"
                triangulation = try_delaunay(normalized_all, final_opts)
            except QhullError as final_exc:
                raise RuntimeError("Delaunay failed after robust fallbacks") from final_exc

    return triangulation, normalized_all, mn, span
