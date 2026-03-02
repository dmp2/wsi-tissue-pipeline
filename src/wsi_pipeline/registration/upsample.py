#!/usr/bin/env python3

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .symmetric import emlddmm_multiscale_symmetric_N


def _validate_and_reshape_image_time_series(series, name):
    series = np.asarray(series)
    if series.ndim == 5 and series.shape[0] == 1:
        series = series[0]
    if series.ndim != 4:
        raise ValueError(f"{name} must have shape (T, C, Y, X); got {series.shape}")
    return series


def _validate_and_reshape_jacobian_time_series(series, name):
    series = np.asarray(series)
    if series.ndim == 4 and series.shape[0] == 1:
        series = series[0]
    if series.ndim != 3:
        raise ValueError(f"{name} must have shape (T, Y, X); got {series.shape}")
    return series


def _register_pair(ind0, ind1, xJ, J, W, config):
    """Register a pair of observed slices and cache time-resolved trajectories."""
    xy_axes = [np.asarray(xJ[1]), np.asarray(xJ[2])]
    I0 = np.asarray(J[:, ind0], dtype=np.float32)
    I1 = np.asarray(J[:, ind1], dtype=np.float32)
    W0 = None if W is None else np.asarray(W[ind0], dtype=np.float32)
    W1 = None if W is None else np.asarray(W[ind1], dtype=np.float32)

    out_fwd = emlddmm_multiscale_symmetric_N(
        xI=xy_axes,
        I=I0,
        xJ=xy_axes,
        J=I1,
        W0=W0,
        **config,
    )
    out_bwd = emlddmm_multiscale_symmetric_N(
        xI=xy_axes,
        I=I1,
        xJ=xy_axes,
        J=I0,
        W0=W1,
        **config,
    )

    I0t = _validate_and_reshape_image_time_series(out_fwd["It"], "forward It")
    I1t = np.flip(_validate_and_reshape_image_time_series(out_bwd["It"], "backward It"), axis=0)
    J0t = _validate_and_reshape_jacobian_time_series(
        out_fwd["det_jac_phi_I"],
        "forward det_jac_phi_I",
    )
    J1t = np.flip(
        _validate_and_reshape_jacobian_time_series(
            out_bwd["det_jac_phi_J"],
            "backward det_jac_phi_J",
        ),
        axis=0,
    )

    return ind0, ind1, I0t, I1t, J0t, J1t


def _validate_inputs(xJ, J, W, mode, config):
    if config is None or "nt" not in config:
        raise ValueError("config must include nt (time steps)")
    if mode not in {"seg", "img"}:
        raise ValueError(f"Unsupported mode '{mode}'")

    if len(xJ) != 3:
        raise ValueError("xJ must be a list of 3 coordinate axes [z, y, x]")
    if any(np.asarray(axis).ndim != 1 for axis in xJ):
        raise ValueError("Each axis in xJ must be a 1D array")

    J = np.asarray(J)
    if J.ndim != 4:
        raise ValueError(f"J must have shape (C, Z, Y, X); got {J.shape}")

    z_axis, y_axis, x_axis = [np.asarray(axis) for axis in xJ]
    if len(z_axis) != J.shape[1] or len(y_axis) != J.shape[2] or len(x_axis) != J.shape[3]:
        raise ValueError(
            "xJ axis lengths must match J spatial dimensions "
            f"({len(z_axis)}, {len(y_axis)}, {len(x_axis)}) vs {J.shape[1:]}"
        )

    if W is not None:
        W = np.asarray(W)
        if W.shape != J.shape[1:]:
            raise ValueError(f"W must have shape (Z, Y, X); got {W.shape}")

    return J, W


def _resolve_present_mask(J, W, mode, present_mask):
    z_dim = J.shape[1]

    if present_mask is not None:
        present_mask = np.asarray(present_mask)
        if present_mask.ndim != 1 or len(present_mask) != z_dim:
            raise ValueError(
                f"present_mask must have shape ({z_dim},); got {present_mask.shape}"
            )
        if present_mask.dtype.kind != "b":
            raise ValueError("present_mask must be a boolean array")
        return present_mask

    if W is not None:
        return np.any(W > 0, axis=(1, 2))

    if mode == "seg":
        return np.any(J != 0, axis=(0, 2, 3))

    raise ValueError("mode='img' requires W or present_mask for stable slice detection")


def upsample_between_slices(
    xJ,
    J,
    W=None,
    *,
    present_mask=None,
    slice_spacing=None,
    n_resample=None,
    mode="seg",
    tissue_idx=None,
    config,
    parallel=False,
    max_workers=None,
):
    """
    Fill missing planes on the existing global z lattice by interpolating between observed slices.

    Parameters
    ----------
    xJ : list[np.ndarray]
        Three 1D coordinate axes [z_axis, y_axis, x_axis].
    J : np.ndarray
        Channel-first data with shape (C, Z, Y, X).
    W : np.ndarray | None
        Optional scalar weight map with shape (Z, Y, X).
    present_mask : np.ndarray | None
        Optional boolean mask of length Z identifying which global slices are observed.
    slice_spacing : float | None
        Ignored compatibility argument from the notebook prototype.
    n_resample : int | None
        Ignored compatibility argument from the notebook prototype. The repaired
        implementation fills planes on the existing global z lattice.
    mode : {"seg", "img"}
        Segmentation mode rounds the blended slices. Image mode uses Jacobian-weighted blending.
    tissue_idx : int | None
        Ignored compatibility argument from the notebook prototype.
    config : dict
        Configuration for `emlddmm_multiscale_symmetric_N`. Must include `nt`.

    Returns
    -------
    dict
        Dictionary containing the original grid, filled data, and interpolation metadata.
    """
    _ = slice_spacing, n_resample, tissue_idx
    J, W = _validate_inputs(xJ, J, W, mode, config)

    z_axis = np.asarray(xJ[0], dtype=np.float32)
    nt = int(config["nt"])
    present_mask = _resolve_present_mask(J, W, mode, present_mask)
    slices_with_data = np.flatnonzero(present_mask).astype(int)
    pairs = [
        (int(i0), int(i1))
        for i0, i1 in zip(slices_with_data[:-1], slices_with_data[1:], strict=False)
    ]

    J_filled = np.array(J, dtype=np.float32, copy=True)
    J_nearest_bad = np.array(J, dtype=np.float32, copy=True)

    if not pairs:
        return {
            "xJ_out": [np.array(axis, copy=True) for axis in xJ],
            "J_filled": J_filled,
            "J_nearest_bad": J_nearest_bad,
            "slices_with_data": slices_with_data,
            "pairs": pairs,
        }

    max_gap = max(i1 - i0 for i0, i1 in pairs)
    if nt < max_gap:
        raise ValueError(
            f"config['nt']={nt} must be at least the largest pair gap ({max_gap}) "
            "measured in global z steps"
        )

    results = []
    if parallel and len(pairs) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_register_pair, i0, i1, xJ, J, W, config): (i0, i1)
                for i0, i1 in pairs
            }
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for i0, i1 in pairs:
            results.append(_register_pair(i0, i1, xJ, J, W, config))

    cache = {(i0, i1): (I0t, I1t, J0t, J1t) for i0, i1, I0t, I1t, J0t, J1t in results}

    for i0, i1 in pairs:
        I0t, I1t, J0t, J1t = cache[(i0, i1)]
        z0 = float(z_axis[i0])
        z1 = float(z_axis[i1])
        delta_z = z1 - z0

        for k in range(i0, i1 + 1):
            if k == i0:
                J_filled[:, k] = J[:, i0]
                J_nearest_bad[:, k] = J[:, i0]
                continue
            if k == i1:
                J_filled[:, k] = J[:, i1]
                J_nearest_bad[:, k] = J[:, i1]
                continue

            p = 0.0 if delta_z == 0.0 else (float(z_axis[k]) - z0) / delta_z
            p = float(np.clip(p, 0.0, 1.0))
            J_nearest_bad[:, k] = J[:, i0] if p <= 0.5 else J[:, i1]

            # MATLAB builds an endpoint-inclusive stack outside the symmetric solver:
            # time_use = round(p * nT + 1) on 1-based indexing. Since this helper
            # returns the nT non-identity frames only, convert to 0-based indexing.
            t_idx = int(np.floor(p * nt + 0.5)) - 1
            t_idx = int(np.clip(t_idx, 0, nt - 1))

            I0_now = I0t[t_idx]
            I1_now = I1t[t_idx]
            J0t_now = J0t[t_idx]
            J1t_now = J1t[t_idx]

            if mode == "seg":
                It = np.round((1.0 - p) * I0_now + p * I1_now)
            else:
                if W is None:
                    num = ((1.0 - p) * J0t_now * I0_now) + (p * J1t_now * I1_now)
                    den = ((1.0 - p) * J0t_now) + (p * J1t_now) + 1e-8
                else:
                    W0 = W[i0]
                    W1 = W[i1]
                    num = ((1.0 - p) * W0 * J0t_now * I0_now) + (p * W1 * J1t_now * I1_now)
                    den = ((1.0 - p) * W0 * J0t_now) + (p * W1 * J1t_now) + 1e-8
                It = num / den

            J_filled[:, k] = It.astype(np.float32)

    return {
        "xJ_out": [np.array(axis, copy=True) for axis in xJ],
        "J_filled": J_filled,
        "J_nearest_bad": J_nearest_bad,
        "slices_with_data": slices_with_data,
        "pairs": pairs,
    }
