#!/usr/bin/env python3

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import hashlib
import json

import numpy as np
import torch

from . import symmetric as _symmetric
from .symmetric import emlddmm_multiscale_symmetric_N
from warnings import warn

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


def _schedule_entries(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if value is None:
        return [None]
    if not isinstance(value, (list, tuple)):
        return [value]
    if len(value) == 0:
        return []
    if any(isinstance(entry, (list, tuple, np.ndarray)) or entry is None for entry in value):
        return [entry.tolist() if isinstance(entry, np.ndarray) else entry for entry in value]
    return [list(value)]


def _validate_2d_downsampling_config(name, value):
    if value is None:
        return
    for entry in _schedule_entries(value):
        if entry is None:
            continue
        if isinstance(entry, np.ndarray):
            entry = entry.tolist()
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(
                f"{name} for between-slice pair registration must use 2D [y, x] "
                f"entries; got {value!r}"
            )
        for factor in entry:
            try:
                numeric_factor = float(factor)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} entries must contain positive integer downsampling factors; "
                    f"got {value!r}"
                ) from exc
            if not numeric_factor.is_integer() or numeric_factor < 1:
                raise ValueError(
                    f"{name} entries must contain positive integer downsampling factors; "
                    f"got {value!r}"
                )


def _validate_unsupported_2d_pair_option(name, value):
    if value is None:
        return
    for entry in _schedule_entries(value):
        if entry is None:
            continue
        raise ValueError(
            f"{name} is not supported for 2D between-slice pair registration; "
            f"leave it unset or use [None], got {value!r}"
        )


def _validate_pair_registration_config(config):
    _validate_2d_downsampling_config("downI", config.get("downI"))
    _validate_2d_downsampling_config("downJ", config.get("downJ"))
    _validate_unsupported_2d_pair_option("local_contrast", config.get("local_contrast"))
    _validate_unsupported_2d_pair_option("up_vector", config.get("up_vector"))
    _validate_no_affine_pair_config(config)
    _validate_out_of_plane_pair_config(config)


def _validate_zero_schedule(name, value):
    if value is None:
        return
    if isinstance(value, np.ndarray):
        value = value.tolist()
    entries = list(value) if isinstance(value, (list, tuple)) else [value]
    for entry in entries:
        if entry is None:
            continue
        try:
            numeric_entry = float(entry)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be zero for between-slice upsampling; got {value!r}"
            ) from exc
        if numeric_entry != 0.0:
            raise ValueError(f"{name} must be zero for between-slice upsampling; got {value!r}")


def _validate_false_schedule(name, value):
    if value is None:
        return
    if isinstance(value, np.ndarray):
        value = value.tolist()
    entries = list(value) if isinstance(value, (list, tuple)) else [value]
    for entry in entries:
        if entry not in {False, None}:
            raise ValueError(f"{name} must be false for between-slice upsampling; got {value!r}")


def _validate_out_of_plane_pair_config(config):
    """Validate out_of_plane for between-slice pair registration.

    Between-slice registration is intrinsically 2D.  The registration helper
    temporarily extrudes the pair into a synthetic 3D slab only because the
    legacy EM-LDDMM backend expects a 3D domain.

    Therefore:
      - False is the recommended setting: velocity is projected into the
        physical image plane.
      - True remains supported for backwards compatibility.
      - unset is allowed; the symmetric 2D adapter defaults it to False.
    """
    value = config.get("out_of_plane")

    if value is None:
        return

    if isinstance(value, np.ndarray):
        value = value.tolist()

    entries = (
        list(value)
        if isinstance(value, (list, tuple))
        else [value]
    )

    # Normalize bool-like values while rejecting arbitrary numbers/objects.
    normalized = []
    for entry in entries:
        if isinstance(entry, (bool, np.bool_)):
            normalized.append(bool(entry))
        elif entry in (0, 1):
            normalized.append(bool(entry))
        else:
            raise ValueError(
                "out_of_plane must be true, false, or unset for "
                f"between-slice upsampling; got {value!r}"
            )

    if not normalized:
        raise ValueError(
            "out_of_plane must not be an empty sequence for "
            "between-slice upsampling"
        )

    if all(entry is False for entry in normalized):
        # Recommended behavior for an intrinsically 2D registration.
        return

    if all(entry is True for entry in normalized):
        warnings.warn(
            "out_of_plane=True is supported for between-slice upsampling "
            "for backwards compatibility, but False is recommended because "
            "the synthetic third dimension is numerical scaffolding rather "
            "than a physical deformation direction.",
            UserWarning,
            stacklevel=2,
        )
        return

    raise ValueError(
        "out_of_plane must be consistently true or consistently false "
        f"for between-slice upsampling; got {value!r}"
    )


def _validate_identity_matrix(name, value, size):
    if value is None:
        return
    matrix = np.asarray(value, dtype=np.float32)
    identity = np.eye(size, dtype=np.float32)
    if matrix.shape == (size, size):
        is_identity = np.allclose(matrix, identity)
    elif matrix.ndim == 3 and matrix.shape[1:] == (size, size):
        is_identity = all(np.allclose(mat, identity) for mat in matrix)
    else:
        is_identity = False
    if not is_identity:
        raise ValueError(
            f"{name} must be None or identity for between-slice upsampling; got shape {matrix.shape}"
        )


def _validate_no_affine_pair_config(config):
    _validate_zero_schedule("eA", config.get("eA"))
    _validate_zero_schedule("eA2d", config.get("eA2d"))
    _validate_zero_schedule("Amode", config.get("Amode"))
    _validate_false_schedule("slice_matching", config.get("slice_matching"))
    _validate_identity_matrix("A", config.get("A"), 4)
    _validate_identity_matrix("A2d", config.get("A2d"), 3)


def _normalize_pair_registration_config(config):
    normalized = dict(config)
    normalized.setdefault("downI", [[1, 1]])
    normalized.setdefault("downJ", [[1, 1]])
    normalized.setdefault("local_contrast", [None])
    normalized.setdefault("up_vector", [None])
    normalized.setdefault("n_draw", 0)
    normalized.setdefault("dtype", "float32")
    normalized.setdefault("A", np.eye(4, dtype=np.float32))
    normalized.setdefault("A2d", None)
    normalized.setdefault("Amode", 0)
    normalized.setdefault("eA", [0.0])
    normalized.setdefault("eA2d", [0.0])
    normalized.setdefault("slice_matching", [False])
    return normalized


def _validate_pseudotimes(pseudotimes):
    if pseudotimes is None:
        return None
    s = np.atleast_1d(np.asarray(pseudotimes, dtype=np.float64))
    if s.ndim != 1 or s.size == 0:
        raise ValueError(f"pseudotimes must be a non-empty 1D sequence; got shape {s.shape}")
    if not np.all(np.isfinite(s)) or np.any((s < 0.0) | (s > 1.0)):
        raise ValueError(f"pseudotimes must lie in [0, 1]; got {s.tolist()}")
    return s


def _partial_inverse_flow(xv, v, s, *, emlddmm_module, grid_sample_kwargs=None):
    """Return phi^{-1}_s on the velocity grid for a normalized time s in [0, 1].

    The solver treats v as piecewise constant, using v[k] on [k/nt, (k+1)/nt).
    For s = (m + f)/nt we take m full steps and one step of length f/nt with
    v[m]. Scaling v[m] by f keeps the solver's dt = 1/nt, and state m+1 of
    `_integrate_inverse_flow` never reads later steps, so the existing
    integrator is reused unchanged. On-frame times (including s=0 and s=1)
    use the unmodified velocity and return the stored state exactly.
    """
    v = torch.as_tensor(v)
    nt = v.shape[0]
    k = float(s) * nt
    m = int(round(k))
    if abs(k - m) <= 1e-6:
        v_eff = v
    else:
        m = int(np.floor(k))
        v_eff = v.clone()
        v_eff[m] = v_eff[m] * (k - m)
        m += 1
    phis = _symmetric._integrate_inverse_flow(
        xv,
        v_eff,
        emlddmm_module=emlddmm_module,
        interp2d=v.shape[1] == 2,
        grid_sample_kwargs=grid_sample_kwargs,
    )
    return phis[m]


def _pull_back_weight(x, weight, phis, *, emlddmm_module, grid_sample_kwargs=None):
    """Sample a weight map through the maps that warp its image: W o phi^{-1}.

    A weight is a scalar confidence, not a density, so no Jacobian factor is
    applied (the blend applies the image's Jacobian once). Bilinear sampling
    keeps weights nonnegative and zero padding gives no confidence to points
    mapped from outside the image grid.
    """
    kwargs = {**(grid_sample_kwargs or {}), "padding_mode": "zeros"}
    return _symmetric._warp_time_series(
        x,
        np.asarray(weight)[None],
        phis,
        emlddmm_module=emlddmm_module,
        interp2d=True,
        grid_sample_kwargs=kwargs,
    )[:, 0]


def _warp_at_times(
    out, image, xy_axes, times, *, device, grid_sample_kwargs=None, weight=None
):
    """Warp `image` along the stored `v_symmetric` of `out` to each time in `times`.

    Mirrors the solver's `phi_I` path: integrate on the velocity grid, resample
    to the image grid, warp, and take det(D phi^{-1}) on the image grid. An
    optional `weight` is pulled back through the same maps (None otherwise).
    """
    emlddmm_module = _symmetric._resolve_emlddmm_module()
    v = torch.as_tensor(out["v_symmetric"], device=device)
    xv = tuple(torch.as_tensor(axis, device=device, dtype=v.dtype) for axis in out["xv"])
    x = tuple(torch.as_tensor(axis, device=device, dtype=v.dtype) for axis in xy_axes)
    phis_v = torch.stack(
        [
            _partial_inverse_flow(
                xv,
                v,
                t,
                emlddmm_module=emlddmm_module,
                grid_sample_kwargs=grid_sample_kwargs,
            )
            for t in times
        ]
    )
    phis = _symmetric._resample_transform_to_domain(
        xv,
        phis_v,
        x,
        emlddmm_module=emlddmm_module,
        interp2d=True,
        grid_sample_kwargs=grid_sample_kwargs,
    )
    warped = _symmetric._warp_time_series(
        x,
        image,
        phis,
        emlddmm_module=emlddmm_module,
        interp2d=True,
        grid_sample_kwargs=grid_sample_kwargs,
    )
    spacing = tuple(float(abs(axis[1] - axis[0])) for axis in xy_axes)
    det = _symmetric._calculate_determinant_of_jacobian(phis, spacing=spacing)
    if weight is not None:
        weight = _pull_back_weight(
            x, weight, phis, emlddmm_module=emlddmm_module, grid_sample_kwargs=grid_sample_kwargs
        ).detach().cpu().numpy()
    return warped.detach().cpu().numpy(), det.detach().cpu().numpy(), weight


def _evaluate_pair_pseudotimes(
    out_fwd, out_bwd, I0, I1, xy_axes, pseudotimes, config, path_mode="two-sided",
    W0=None, W1=None,
):
    """Evaluate both endpoint warps and their Jacobians at arbitrary pseudotimes.

    Returns (I0s, I1s, J0s, J1s, W0s, W1s). W0s / W1s are W0 / W1 pulled back
    through the same maps as I0 / I1, so each weight stays on the image it
    describes in the intermediate frame; they are None when W0 / W1 are None.

    I0 follows the forward solve's velocity to time s (as in I0t / J0t).

    path_mode="two-sided": I1 follows the backward solve's velocity to time
    1 - s (as in I1t), and J1s is the Jacobian of that same I1 warp. The two
    velocities come from independent optimizations and are not time-reversals
    of each other, so this is two-sided deformation-based interpolation: each
    endpoint is carried part way along its own estimated flow, not along one
    diffeomorphic path joining the sections.

    path_mode="single-forward": both endpoints use the forward flow phi_t of v.
    I1 is sampled by phi_1 o phi_s^{-1}. With w(r) = -v(1 - r), whose flow is
    chi_r = phi_{1-r} o phi_1^{-1}, this map is chi_{1-s}^{-1}, so it is the
    inverse flow of flip(-v) at time 1 - s (the solver's phi_J path). Both
    warps then live in the same intermediate frame phi_s(I0 coordinates).
    """
    runs = {"two-sided": (("forward", out_fwd), ("backward", out_bwd)),
            "single-forward": (("forward", out_fwd),)}[path_mode]
    for name, out in runs:
        if "v_symmetric" not in out or "xv" not in out:
            raise ValueError(
                f"pseudotimes require the {name} pair solve to return 'v_symmetric' and 'xv'"
            )
    device = torch.device(config.get("device") or "cpu")
    grid_sample_kwargs = config.get("grid_sample_kwargs")
    s = np.asarray(pseudotimes, dtype=np.float64)
    I0s, J0s, W0s = _warp_at_times(
        out_fwd, I0, xy_axes, s, device=device, grid_sample_kwargs=grid_sample_kwargs, weight=W0
    )
    if path_mode == "single-forward":
        v = torch.as_tensor(out_fwd["v_symmetric"])
        out_i1 = {"v_symmetric": torch.flip(-v, [0]), "xv": out_fwd["xv"]}
    else:
        out_i1 = out_bwd
    I1s, J1s, W1s = _warp_at_times(
        out_i1, I1, xy_axes, 1.0 - s, device=device, grid_sample_kwargs=grid_sample_kwargs,
        weight=W1,
    )
    return I0s, I1s, J0s, J1s, W0s, W1s


def _blend_pair(mode, p, I0_now, I1_now, J0t_now, J1t_now, W0=None, W1=None):
    if mode == "seg":
        return np.round((1.0 - p) * I0_now + p * I1_now)
    if W0 is None:
        num = ((1.0 - p) * J0t_now * I0_now) + (p * J1t_now * I1_now)
        den = ((1.0 - p) * J0t_now) + (p * J1t_now) + 1e-8
    else:
        num = ((1.0 - p) * W0 * J0t_now * I0_now) + (p * W1 * J1t_now * I1_now)
        den = ((1.0 - p) * W0 * J0t_now) + (p * W1 * J1t_now) + 1e-8
    return num / den


def _pair_endpoint_images(J, ind0, ind1, mode):
    """Return the endpoint images exactly as they are registered."""
    from scipy.signal import convolve2d

    I0 = np.asarray(J[:, ind0], dtype=np.float32)
    I1 = np.asarray(J[:, ind1], dtype=np.float32)

    if mode == "seg":
        # Try smoothing before mapping
        down=8 # filter size
        kernel = np.ones((down, down), dtype=np.float32) / (down**2)

        I0 = np.stack(
            [convolve2d(channel, kernel, mode="same") for channel in I0],
            axis=0,
        ).astype(np.float32)

        I1 = np.stack(
            [convolve2d(channel, kernel, mode="same") for channel in I1],
            axis=0,
        ).astype(np.float32)

    return I0, I1


def _register_pair(ind0, ind1, xJ, J, W, config, mode=None):
    """Register a pair of observed slices and cache time-resolved trajectories."""
    xy_axes = (np.asarray(xJ[1]), np.asarray(xJ[2]))
    I0, I1 = _pair_endpoint_images(J, ind0, ind1, mode)

    W0 = None if W is None else np.asarray(W[ind0], dtype=np.float32)
    W1 = None if W is None else np.asarray(W[ind1], dtype=np.float32)

    print("mode", mode)
    print("I0 min/max", I0.min(), I0.max())
    print("I0 unique-ish", np.unique(I0)[:20])
    print("fraction nonbinary", np.mean((I0 > 0) & (I0 < 1)))

    # EM-LDDMM weights the matching error on the target grid, so each solve
    # takes the target's weights as W0 and the source's as WI (the target of
    # its internal backward solve).
    out_fwd = emlddmm_multiscale_symmetric_N(
        xI=xy_axes,
        I=I0,
        xJ=xy_axes,
        J=I1,
        W0=W1,
        WI=W0,
        **config,
    )

    print("mode", mode)
    print("I0 min/max", I1.min(), I1.max())
    print("I0 unique-ish", np.unique(I1)[:20])
    print("fraction nonbinary", np.mean((I1 > 0) & (I1 < 1)))

    out_bwd = emlddmm_multiscale_symmetric_N(
        xI=xy_axes,
        I=I1,
        xJ=xy_axes,
        J=I0,
        W0=W0,
        WI=W1,
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

    # Only the image-mode blend uses weights. Pull each back through the maps
    # that produced its image frames: I0t from forward phi_I, I1t from the
    # flipped backward phi_I (both include the identity state at index 0).
    W0t = W1t = None
    if W is not None and mode != "seg":
        emlddmm_module = _symmetric._resolve_emlddmm_module()
        grid_sample_kwargs = config.get("grid_sample_kwargs")
        W0t = _pull_back_weight(
            xy_axes, W0, torch.as_tensor(out_fwd["phi_I"])[1:],
            emlddmm_module=emlddmm_module, grid_sample_kwargs=grid_sample_kwargs,
        ).numpy()
        W1t = np.flip(
            _pull_back_weight(
                xy_axes, W1, torch.as_tensor(out_bwd["phi_I"])[1:],
                emlddmm_module=emlddmm_module, grid_sample_kwargs=grid_sample_kwargs,
            ).numpy(),
            axis=0,
        )

    velocities = {
        "v_forward": out_fwd.get("v_symmetric"),
        "v_backward": out_bwd.get("v_symmetric"),
        "xv_forward": out_fwd.get("xv"),
        "xv_backward": out_bwd.get("xv"),
    }

    return ind0, ind1, I0t, I1t, J0t, J1t, W0t, W1t, velocities


PAIR_VELOCITY_FORMAT = "wsi_pipeline.pair_velocities/2"

# /1 archives carry no input fingerprints, and their weighted solves used the
# source slice's weights for the target.
_UNVERIFIABLE_PAIR_VELOCITY_FORMATS = ("wsi_pipeline.pair_velocities/1",)

PAIR_VELOCITY_CONVENTIONS = (
    "Pair p = pairs[p] = (i0, i1) registers I0 = J[:, i0] and I1 = J[:, i1] on the "
    "image grid (y, x); in 'seg' mode both are first 8x8 box-smoothed. "
    "v_forward[p] is the I0 -> I1 velocity and v_backward[p] the I1 -> I0 velocity, "
    "each (nt, 2, len(xv_y), len(xv_x)) with components (y, x) on the grid (xv_y, xv_x). "
    "v[k] is constant on [k/nt, (k+1)/nt); phi_t^{-1} is the semi-Lagrangian "
    "integration of symmetric._integrate_inverse_flow, so I0 o phi_t^{-1} moves I0 "
    "toward I1. z[p] holds the physical z of (i0, i1). The two solves are independent. "
    "W[i] (same frame as J) weights slice i as a registration target and is pulled "
    "back with its image for blending. Input identity: J_sha256[k] and W_sha256[k] are "
    "SHA-256 digests of J[:, input_slices[k]] and W[input_slices[k]] as registered, "
    "little-endian float32 in C order prefixed by dtype and shape (see "
    "upsample._slice_sha256); W_sha256 is empty when no weights were used. They identify "
    "the inputs, not the software environment that produced the velocities."
)


def _slice_sha256(array):
    """SHA-256 of one registration input slice in the float32 form the solver sees.

    The dtype and shape are hashed first, so equal bytes with another layout
    differ. Hashing one slice at a time bounds the copy to a single slice.
    """
    array = np.ascontiguousarray(array, dtype="<f4")
    digest = hashlib.sha256(f"<f4{array.shape}".encode())
    digest.update(array)
    return digest.hexdigest()


def _input_fingerprints(J, W, pairs):
    """Fingerprint every slice that enters a pair solve (J, and W when used)."""
    slices = np.unique(np.asarray(pairs, dtype=np.int64))
    no_digests = np.zeros((0,), dtype="<U64")
    return {
        "input_slices": slices,
        "J_sha256": np.array([_slice_sha256(J[:, i]) for i in slices], dtype="<U64"),
        "W_sha256": (
            no_digests if W is None
            else np.array([_slice_sha256(W[i]) for i in slices], dtype="<U64")
        ),
    }


def _verify_input_fingerprints(J, W, pair_velocities):
    """Reject J / W that differ from the stack the velocities were estimated on."""
    slices = pair_velocities["input_slices"]
    stored_w = pair_velocities["W_sha256"]
    if slices.size and W is None and stored_w.size:
        raise ValueError("the velocities were registered with weights W; pass the same W")
    if slices.size and W is not None and not stored_w.size:
        raise ValueError("the velocities were registered without weights; pass W=None")
    checks = [("J", pair_velocities["J_sha256"], lambda i: J[:, i])]
    if W is not None:
        checks.append(("W", stored_w, lambda i: W[i]))
    for name, stored, data in checks:
        bad = [
            int(i) for i, digest in zip(slices, stored, strict=True)
            if _slice_sha256(data(int(i))) != str(digest)
        ]
        if bad:
            raise ValueError(
                f"{name} differs from the registration inputs of these velocities at "
                f"slices {bad} (SHA-256 of the float32 values)"
            )


def _check_pair_velocity_format(pair_velocities, source="pair_velocities"):
    fmt = str(pair_velocities.get("format"))
    if fmt in _UNVERIFIABLE_PAIR_VELOCITY_FORMATS:
        raise ValueError(
            f"{source} has format {fmt!r}, which records no input fingerprints and "
            "predates the target-weight correction, so its inputs cannot be verified. "
            "Re-run upsample_between_slices(..., return_velocities=True) to write a "
            f"{PAIR_VELOCITY_FORMAT!r} archive."
        )
    if fmt != PAIR_VELOCITY_FORMAT:
        raise ValueError(f"{source} is not a {PAIR_VELOCITY_FORMAT!r} file; got format {fmt!r}")


def _collect_pair_velocities(xJ, J, W, pairs, velocities, mode, config):
    """Stack every pair's stored velocities into one flat dict of plain arrays."""
    v_forward, v_backward, xv = [], [], None
    for pair in pairs:
        entry = velocities[pair]
        for direction in ("forward", "backward"):
            if entry[f"v_{direction}"] is None or entry[f"xv_{direction}"] is None:
                raise ValueError(
                    f"pair {pair}: the {direction} pair solve must return 'v_symmetric' and 'xv'"
                )
            axes = tuple(np.asarray(axis) for axis in entry[f"xv_{direction}"])
            if xv is None:
                xv = axes
            elif any(a.shape != b.shape or not np.array_equal(a, b) for a, b in zip(axes, xv)):
                raise ValueError(f"pair {pair}: velocity grid differs from the other pair solves")
        v_forward.append(np.asarray(entry["v_forward"]))
        v_backward.append(np.asarray(entry["v_backward"]))

    pair_array = np.array(pairs, dtype=np.int64).reshape(len(pairs), 2)
    empty = np.zeros((0,), dtype=np.float32)
    return {
        "format": np.array(PAIR_VELOCITY_FORMAT),
        "conventions": np.array(PAIR_VELOCITY_CONVENTIONS),
        "mode": np.array(mode),
        "grid_sample_kwargs": np.array(json.dumps(config.get("grid_sample_kwargs"))),
        "pairs": pair_array,
        "z": np.asarray(xJ[0], dtype=np.float64)[pair_array],
        "y": np.asarray(xJ[1]),
        "x": np.asarray(xJ[2]),
        "xv_y": empty if xv is None else xv[0],
        "xv_x": empty if xv is None else xv[1],
        "v_forward": np.stack(v_forward) if pairs else empty,
        "v_backward": np.stack(v_backward) if pairs else empty,
        **_input_fingerprints(J, W, pair_array),
    }


def _assemble_pseudotime_samples(J, W, pair_velocities, pseudotimes, path_mode, device):
    """Build pseudotime samples for every stored pair from its velocities."""
    mode = str(pair_velocities["mode"])
    pairs = [(int(i0), int(i1)) for i0, i1 in pair_velocities["pairs"]]
    xy_axes = (pair_velocities["y"], pair_velocities["x"])
    xv = (pair_velocities["xv_y"], pair_velocities["xv_x"])
    eval_config = {
        "device": device,
        "grid_sample_kwargs": json.loads(str(pair_velocities["grid_sample_kwargs"])),
    }

    n_samples = len(pairs) * len(pseudotimes)
    samples = {
        "pair_indices": np.array(
            [pair for pair in pairs for _ in pseudotimes], dtype=int
        ).reshape(n_samples, 2),
        "pseudotimes": np.tile(pseudotimes, len(pairs)),
        "path_mode": path_mode,
        "z": np.empty(n_samples, dtype=np.float64),
        "J": np.empty((J.shape[0], n_samples, *J.shape[2:]), dtype=np.float32),
    }

    for pair_idx, (i0, i1) in enumerate(pairs):
        z0, z1 = (float(z) for z in pair_velocities["z"][pair_idx])
        I0, I1 = _pair_endpoint_images(J, i0, i1, mode)
        weighted = W is not None and mode != "seg"
        I0s, I1s, J0s, J1s, W0s, W1s = _evaluate_pair_pseudotimes(
            {"v_symmetric": pair_velocities["v_forward"][pair_idx], "xv": xv},
            {"v_symmetric": pair_velocities["v_backward"][pair_idx], "xv": xv},
            I0,
            I1,
            xy_axes,
            pseudotimes,
            eval_config,
            path_mode,
            np.asarray(W[i0], dtype=np.float32) if weighted else None,
            np.asarray(W[i1], dtype=np.float32) if weighted else None,
        )
        for j, s in enumerate(pseudotimes):
            n = pair_idx * len(pseudotimes) + j
            samples["z"][n] = z0 + s * (z1 - z0)
            if s == 0.0:
                samples["J"][:, n] = J[:, i0]
            elif s == 1.0:
                samples["J"][:, n] = J[:, i1]
            else:
                samples["J"][:, n] = _blend_pair(
                    mode,
                    float(s),
                    I0s[j],
                    I1s[j],
                    J0s[j],
                    J1s[j],
                    None if W0s is None else W0s[j],
                    None if W1s is None else W1s[j],
                ).astype(np.float32)

    return samples


def _validate_path_mode(path_mode):
    if path_mode not in {"two-sided", "single-forward"}:
        raise ValueError(f"Unsupported path_mode {path_mode!r}")


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
    if len(y_axis) < 2 or len(x_axis) < 2:
        raise ValueError(
            "2D between-slice pair registration requires y and x axes with at least "
            f"two points; got y={len(y_axis)}, x={len(x_axis)} for J shape {J.shape}"
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
            raise ValueError(f"present_mask must have shape ({z_dim},); got {present_mask.shape}")
        if present_mask.dtype.kind != "b":
            raise ValueError("present_mask must be a boolean array")
        return present_mask

    if W is not None:
        w_present_mask = np.any(W > 0, axis=(1, 2))
        if mode == "img" and np.all(w_present_mask):
            # In image mode, some callers pass dense per-slice weights even when
            # the repaired global z lattice still contains zero-filled gaps.
            # Fall back to image content when W alone cannot distinguish missing
            # planes, while still honoring sparse W masks when they are present.
            return np.any(J != 0, axis=(0, 2, 3))
        return w_present_mask

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
    pseudotimes=None,
    path_mode="two-sided",
    return_velocities=False,
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
        Optional scalar weight map with shape (Z, Y, X), in the same frame as J
        (for the workflow, Wd resampled with the A2d that produced Jr). W[i]
        weights slice i wherever it is a registration target, and in the
        image-mode blend.
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
    pseudotimes : sequence of float | None
        Optional normalized times s in [0, 1] to evaluate between every adjacent
        observed pair, independent of the z lattice. The pair's forward
        (I0 -> I1) and backward (I1 -> I0) velocities from this call are
        integrated to s and 1 - s (with a fractional final step), so no extra
        optimization is run. The solves are independent, so samples are a
        two-sided blend of two warped endpoints rather than points on a single
        diffeomorphic path. s=0 and s=1 return the observed slices.
    path_mode : {"two-sided", "single-forward"}
        How `pseudotimes` samples are built; ignored without `pseudotimes`.
        "two-sided" (default) is described above. "single-forward" samples
        both endpoints along the forward flow phi_t only: I0 by phi_s^{-1} and
        I1 by phi_1 o phi_s^{-1}, so both lie in one intermediate frame.
        In both modes, and on the z lattice, W[i0] and W[i1] are pulled back
        through the same maps as their images before blending.
    return_velocities : bool
        Also return ``"pair_velocities"``: every pair's forward and backward
        velocity with its grids, as a flat dict of plain arrays. Save it with
        ``np.savez_compressed(path, **pair_velocities)``, reload it with
        `load_pair_velocities`, and evaluate new pseudotimes with
        `evaluate_pseudotimes` without re-registering. Pseudotime samples are
        always built from these arrays, so both routes give identical results.

    Returns
    -------
    dict
        Dictionary containing the original grid, filled data, and interpolation metadata.
        When `pseudotimes` is given, also contains ``"pseudotime_samples"`` with
        ``pair_indices`` (N, 2), ``pseudotimes`` (N,), ``z`` (N,) physical positions
        ``z0 + s * (z1 - z0)``, ``J`` (C, N, Y, X), ordered pair-major, and
        ``path_mode``. When `return_velocities` is true, also contains
        ``"pair_velocities"`` (see `PAIR_VELOCITY_CONVENTIONS`).
    """
    _ = slice_spacing, n_resample, tissue_idx
    pseudotimes = _validate_pseudotimes(pseudotimes)
    _validate_path_mode(path_mode)
    J, W = _validate_inputs(xJ, J, W, mode, config)
    config = _normalize_pair_registration_config(config)
    _validate_pair_registration_config(config)

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

    result = {
        "xJ_out": [np.array(axis, copy=True) for axis in xJ],
        "J_filled": J_filled,
        "J_nearest_bad": J_nearest_bad,
        "slices_with_data": slices_with_data,
        "pairs": pairs,
    }

    # not a requirement
    # max_gap = max(i1 - i0 for i0, i1 in pairs)
    # if nt < max_gap:
    #     raise ValueError(
    #         f"config['nt']={nt} must be at least the largest pair gap ({max_gap}) "
    #         "measured in global z steps"
    #     )

    results = []
    if parallel and len(pairs) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_register_pair, i0, i1, xJ, J, W, config, mode): (i0, i1)
                for i0, i1 in pairs
            }
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for i0, i1 in pairs:
            results.append(_register_pair(i0, i1, xJ, J, W, config, mode))

    cache = {(i0, i1): frames for i0, i1, *frames, _ in results}

    for i0, i1 in pairs:
        I0t, I1t, J0t, J1t, W0t, W1t = cache[(i0, i1)]
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

            It = _blend_pair(
                mode,
                p,
                I0_now,
                I1_now,
                J0t_now,
                J1t_now,
                None if W0t is None else W0t[t_idx],
                None if W1t is None else W1t[t_idx],
            )

            J_filled[:, k] = It.astype(np.float32)

    if pseudotimes is not None or return_velocities:
        velocities = {(i0, i1): stored for i0, i1, *_, stored in results}
        pair_velocities = _collect_pair_velocities(xJ, J, W, pairs, velocities, mode, config)
        if pseudotimes is not None:
            result["pseudotime_samples"] = _assemble_pseudotime_samples(
                J, W, pair_velocities, pseudotimes, path_mode, config.get("device")
            )
        if return_velocities:
            result["pair_velocities"] = pair_velocities

    return result


def load_pair_velocities(path):
    """Load pair velocities written with ``np.savez_compressed(path, **pair_velocities)``."""
    with np.load(path, allow_pickle=False) as data:
        pair_velocities = {key: data[key] for key in data.files}
    _check_pair_velocity_format(pair_velocities, str(path))
    return pair_velocities


def evaluate_pseudotimes(
    xJ,
    J,
    W=None,
    *,
    pair_velocities,
    pseudotimes,
    path_mode="two-sided",
    device=None,
):
    """Evaluate pseudotime samples from stored pair velocities, without registering.

    `xJ`, `J`, and `W` must be the stack the velocities were estimated on (for the
    workflow, self-alignment ``Jr`` and ``Wr``, written as
    ``images/target_registered.vtk`` and ``images/target_registered_weights.vtk``).
    The grid is checked against the stored axes and every slice entering a pair
    against its stored SHA-256 fingerprint; mismatches raise ValueError. The
    registration mode stored with the velocities is reused, so seg-mode endpoints
    are smoothed exactly as before registration. Returns the same dict as
    ``upsample_between_slices(...)["pseudotime_samples"]``.
    """
    pseudotimes = _validate_pseudotimes(pseudotimes)
    if pseudotimes is None:
        raise ValueError("pseudotimes are required")
    _validate_path_mode(path_mode)
    _check_pair_velocity_format(pair_velocities)

    J = np.asarray(J)
    if J.ndim != 4:
        raise ValueError(f"J must have shape (C, Z, Y, X); got {J.shape}")
    if W is not None:
        W = np.asarray(W)
        if W.shape != J.shape[1:]:
            raise ValueError(f"W must have shape (Z, Y, X); got {W.shape}")
    for name, axis, stored in (
        ("y", xJ[1], pair_velocities["y"]),
        ("x", xJ[2], pair_velocities["x"]),
    ):
        axis = np.asarray(axis)
        if axis.shape != stored.shape or not np.allclose(axis, stored):
            raise ValueError(f"xJ {name} axis differs from the stored image grid")
    if J.shape[2:] != (len(pair_velocities["y"]), len(pair_velocities["x"])):
        raise ValueError(f"J spatial shape {J.shape[2:]} differs from the stored image grid")
    pairs = np.asarray(pair_velocities["pairs"], dtype=int)
    if pairs.size and (pairs.min() < 0 or pairs.max() >= J.shape[1]):
        raise ValueError(f"stored pairs index beyond the {J.shape[1]} slices of J")
    z_axis = np.asarray(xJ[0], dtype=np.float64)
    if not np.allclose(z_axis[pairs], pair_velocities["z"]):
        raise ValueError("xJ z axis differs from the stored pair positions")
    _verify_input_fingerprints(J, W, pair_velocities)

    return _assemble_pseudotime_samples(
        J, W, pair_velocities, pseudotimes, path_mode, device
    )
