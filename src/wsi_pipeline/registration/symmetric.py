#!/usr/bin/env python3
"""Symmetric upsampling helpers carried over from the notebook-era EM-LDDMM code.

This module is intentionally close to the 2025 notebook-derived implementation used
during the `tb_macaque_emlddmm.ipynb` workflow migration. In this cleanup pass, the
surrounding pipeline gains clearer docs, logging, and report outputs, but the
underlying numerical behavior here is intentionally left unchanged.
"""

import torch

from .backend import resolve_emlddmm_backend

_SYMMETRIC_BACKEND_ATTRS = ("emlddmm_multiscale", "interp")

def _resolve_emlddmm_module():
    """Resolve the EM-LDDMM backend module used by symmetric registration."""
    module = resolve_emlddmm_backend().module
    missing = [attr for attr in _SYMMETRIC_BACKEND_ATTRS if not hasattr(module, attr)]
    if missing:
        raise ImportError(
            "EM-LDDMM backend for symmetric registration is missing required attributes: "
            f"{', '.join(missing)}"
        )
    return module


def _integrate_inverse_flow(xv, v, *, emlddmm_module, interp2d=None, grid_sample_kwargs=None):
    """Integrate v_t -> phi^{-1}_t, storing all time steps."""
    v = torch.as_tensor(v)
    xv = [torch.as_tensor(x, device=v.device, dtype=v.dtype) for x in xv]
    ndim = v.shape[1]
    interp2d = bool(interp2d) if interp2d is not None else ndim == 2
    mesh = torch.stack(torch.meshgrid(xv[:ndim], indexing="ij"))
    phi = mesh
    phis = [phi]
    dt = 1.0 / v.shape[0]
    for t in range(v.shape[0]):
        Xs = mesh - v[t] * dt
        phi = (
            emlddmm_module.interp(
                xv[:ndim],
                phi - mesh,
                Xs,
                interp2d=interp2d,
                **(grid_sample_kwargs or {}),
            )
            + Xs
        )
        phis.append(phi)
    return torch.stack(phis)


def _warp_time_series(x, image, phis, *, emlddmm_module, interp2d=None, grid_sample_kwargs=None):
    """Warp an image along a stored phi^{-1}_t trajectory."""
    x = tuple(
        torch.as_tensor(axis, device=phis.device, dtype=phis.dtype) for axis in x
    )
    image = torch.as_tensor(image, device=phis.device, dtype=phis.dtype)
    warped = []
    for t in range(phis.shape[0]):
        warped.append(
            emlddmm_module.interp(
                x,
                image,
                phis[t],
                interp2d=interp2d,
                **(grid_sample_kwargs or {}),
            )
        )
    return torch.stack(warped)


def _resample_transform_to_domain(
    xv,
    phis,
    x,
    *,
    emlddmm_module,
    interp2d=None,
    grid_sample_kwargs=None,
):
    """Evaluate a transform integrated on the velocity grid at image-domain points."""
    phis = torch.as_tensor(phis)
    ndim = phis.shape[1]
    interp2d = bool(interp2d) if interp2d is not None else ndim == 2
    xv = [torch.as_tensor(axis, device=phis.device, dtype=phis.dtype) for axis in xv[:ndim]]
    x = [torch.as_tensor(axis, device=phis.device, dtype=phis.dtype) for axis in x[:ndim]]

    if len(xv) == len(x) and all(
        a.shape == b.shape and torch.equal(a, b) for a, b in zip(xv, x, strict=False)
    ):
        return phis

    velocity_mesh = torch.stack(torch.meshgrid(xv, indexing="ij"))
    image_mesh = torch.stack(torch.meshgrid(x, indexing="ij"))
    resampled = []
    for t in range(phis.shape[0]):
        displacement = phis[t] - velocity_mesh
        resampled.append(
            emlddmm_module.interp(
                xv,
                displacement,
                image_mesh,
                interp2d=interp2d,
                **(grid_sample_kwargs or {}),
            )
            + image_mesh
        )
    return torch.stack(resampled)


def _calculate_determinant_of_jacobian(phi, spacing=None):
    """
    Parameters
    ----------
    phi : torch.Tensor
        Tensor of shape (nt+1, 2, H, W) storing the inverse transform at each time step.
    spacing : tuple or list, optional
        Physical spacing along (row, col). Defaults to 1 for both axes.

    Returns
    -------
    det : torch.Tensor
        Tensor of shape (nt+1, H, W) with det(D phi) at every time step.
    """
    if spacing is None:
        spacing = (1.0, 1.0)
    # grad_phi[c][d] = \partial_{dim d} phi^{c}
    edge_order = 2 if min(phi.shape[-2:]) >= 3 else 1
    grads = []
    for c in range(phi.shape[1]):
        gx, gy = torch.gradient(
            phi[:, c],
            spacing=spacing,
            dim=(-2, -1),
            edge_order=edge_order,
        )
        grads.append((gx, gy))

    # determinant: \partial_x phi_x * \partial_y phi_y - \partial_y phi_x * \partial_x phi_y
    det = grads[0][0] * grads[1][1] - grads[0][1] * grads[1][0]
    return det


def _to_plain_value(value):
    return value.tolist() if hasattr(value, "tolist") else value


def _is_sequence(value):
    return isinstance(value, (list, tuple))


def _prepend_synthetic_axis_to_2d_schedule(value, synthetic_value):
    value = _to_plain_value(value)
    if value is None or not _is_sequence(value) or len(value) == 0:
        return value
    entries = [_to_plain_value(entry) for entry in value]
    if any(_is_sequence(entry) or entry is None for entry in entries):
        promoted = []
        for entry in entries:
            if entry is None:
                promoted.append(entry)
            elif _is_sequence(entry) and len(entry) == 2:
                promoted.append([synthetic_value, *entry])
            else:
                promoted.append(entry)
        return promoted
    if len(entries) == 2:
        return [[synthetic_value, *entries]]
    return value


def _mean_spacing_from_axes(*axis_groups):
    spacings = []
    for axes in axis_groups:
        for axis in axes[-2:]:
            axis = torch.as_tensor(axis)
            if axis.numel() >= 2:
                spacings.append(float(torch.abs(axis[1] - axis[0]).detach().cpu()))
    if not spacings:
        return 1.0
    return float(sum(spacings) / len(spacings))


import math


def _flatten_numeric_values(value):
    """Return all scalar numeric values contained in a nested config value."""
    value = _to_plain_value(value)

    if value is None:
        return []

    if _is_sequence(value):
        out = []
        for item in value:
            out.extend(_flatten_numeric_values(item))
        return out

    return [float(value)]


# def _synthetic_dv_z_from_config(config, synthetic_spacing):
#     """Extract the coarsest synthetic-z dv across all requested scales."""
#     dv = _to_plain_value(config.get("dv"))

#     if dv is None:
#         return float(synthetic_spacing)

#     # Scalar dv -> legacy backend uses the same spacing in z/y/x.
#     if not _is_sequence(dv):
#         value = abs(float(dv))
#         if value == 0:
#             raise ValueError("dv must be nonzero")
#         return value

#     entries = [_to_plain_value(v) for v in dv]

#     # Nested schedule: [[z,y,x], [z,y,x], ...]
#     if entries and all(_is_sequence(entry) for entry in entries):
#         z_values = []

#         for entry in entries:
#             entry = list(entry)

#             if len(entry) == 3:
#                 z_values.append(abs(float(entry[0])))
#             elif len(entry) == 2:
#                 # Still in native 2D form.  We have no user-specified z dv,
#                 # so use the characteristic synthetic-plane spacing.
#                 z_values.append(float(synthetic_spacing))
#             else:
#                 raise ValueError(
#                     "Expected each dv scale to contain 2 or 3 spatial values; "
#                     f"got {entry}"
#                 )

#         return max(z_values)

#     # One explicit 3D vector.
#     if len(entries) == 3:
#         return abs(float(entries[0]))

#     # One native 2D vector. z does not exist scientifically, so give the
#     # synthetic direction its natural image spacing.
#     if len(entries) == 2:
#         return float(synthetic_spacing)

#     # One-element multiscale/scalar schedule.
#     if len(entries) == 1:
#         return abs(float(entries[0]))

#     raise ValueError(f"Unsupported dv specification for 2D promotion: {dv!r}")

def _promote_2d_dv_schedule(dv, synthetic_spacing):
    """Promote native 2D dv settings to backend 3D dv settings.

    The synthetic z direction is an implementation detail, so its velocity-grid
    spacing is fixed internally to ``synthetic_spacing``.  User/config values
    control only the real in-plane y/x directions.

    Examples
    --------
    2D scalar:
        2.0 -> [[synthetic_spacing, 2.0, 2.0]]

    one 2D scale:
        [[2.0, 1.0]] -> [[synthetic_spacing, 2.0, 1.0]]

    multiple 2D scales:
        [[4.0, 4.0], [2.0, 2.0], [1.0, 1.0]]
        ->
        [
            [synthetic_spacing, 4.0, 4.0],
            [synthetic_spacing, 2.0, 2.0],
            [synthetic_spacing, 1.0, 1.0],
        ]
    """
    dv = _to_plain_value(dv)

    if dv is None:
        return [[
            float(synthetic_spacing),
            float(synthetic_spacing),
            float(synthetic_spacing),
        ]]

    if not _is_sequence(dv):
        value = float(dv)
        return [[
            float(synthetic_spacing),
            value,
            value,
        ]]

    entries = [_to_plain_value(v) for v in dv]

    # Nested multiscale schedule.
    if entries and all(_is_sequence(entry) for entry in entries):
        promoted = []

        for entry in entries:
            entry = list(entry)

            if len(entry) == 2:
                dy, dx = map(float, entry)

            elif len(entry) == 3:
                # Accept already-promoted input for compatibility, but ignore
                # the caller's synthetic-z value.  z is internal to the adapter.
                _, dy, dx = map(float, entry)

            else:
                raise ValueError(
                    "Each dv scale must contain 2 in-plane values "
                    f"(or legacy/promoted 3D values); got {entry!r}"
                )

            promoted.append([
                float(synthetic_spacing),
                dy,
                dx,
            ])

        return promoted

    # Flat native 2D vector.
    if len(entries) == 2:
        dy, dx = map(float, entries)
        return [[
            float(synthetic_spacing),
            dy,
            dx,
        ]]

    # Flat legacy/promoted 3D vector. Ignore its z value.
    if len(entries) == 3:
        _, dy, dx = map(float, entries)
        return [[
            float(synthetic_spacing),
            dy,
            dx,
        ]]

    # One-value schedule behaves like an isotropic 2D scalar.
    if len(entries) == 1:
        value = float(entries[0])
        return [[
            float(synthetic_spacing),
            value,
            value,
        ]]

    raise ValueError(f"Unsupported dv specification: {dv!r}")


def _make_synthetic_z_axis(
    *,
    synthetic_spacing,
    dv_z,
    device,
    dtype,
    min_velocity_intervals=4,
):
    """Construct a symmetric artificial z axis for a promoted 2D registration.

    Parameters
    ----------
    synthetic_spacing
        Spacing between repeated image planes.  Normally derived from the
        actual y/x image spacing.

    dv_z
        Effective deformation-grid sampling interval in the artificial z
        direction.

    min_velocity_intervals
        Minimum physical z extent expressed in deformation-grid intervals.
        Four is deliberately conservative for the legacy 3D FFT/Sobolev
        backend and avoids the previously observed singleton z velocity grid.

    Returns
    -------
    torch.Tensor
        Symmetric z coordinates centered at zero.

    Notes
    -----
    This axis has no anatomical meaning.  Its only purpose is to give the
    legacy 3D backend a nondegenerate numerical domain while solving a 2D
    registration problem.
    """
    synthetic_spacing = float(synthetic_spacing)
    dv_z = float(dv_z)

    if not math.isfinite(synthetic_spacing) or synthetic_spacing <= 0:
        raise ValueError(
            f"synthetic_spacing must be finite and positive; got {synthetic_spacing}"
        )

    if not math.isfinite(dv_z) or dv_z <= 0:
        raise ValueError(f"dv_z must be finite and positive; got {dv_z}")

    if min_velocity_intervals < 2:
        raise ValueError("min_velocity_intervals must be >= 2")

    # Require the artificial slab to span at least this much physical extent.
    required_extent = min_velocity_intervals * dv_z

    # For N equally spaced image planes:
    #
    #     extent = (N - 1) * synthetic_spacing
    #
    # Choose an odd N so that there is exactly one central z=0 plane.
    required_intervals = math.ceil(required_extent / synthetic_spacing)
    n_planes = required_intervals + 1

    if n_planes % 2 == 0:
        n_planes += 1

    # Never use a trivial slab even when dv_z is very fine.
    n_planes = max(n_planes, 5)

    half = n_planes // 2

    return (
        torch.arange(
            -half,
            half + 1,
            device=device,
            dtype=dtype,
        )
        * synthetic_spacing
    )


def _extrude_2d_pair_for_3d_backend(
    I_t,
    J_t,
    W0_t,
    xI_t,
    xJ_t,
    config,
    *,
    min_velocity_intervals=4,
):
    """Embed a 2D image pair in a numerically nondegenerate synthetic 3D slab.

    The same 2D image is copied identically through artificial z.  No new
    anatomical/morphological information is introduced.

    Returns
    -------
    xI_backend, xJ_backend, I_backend, J_backend, W0_backend, synthetic_axis
    """
    device = I_t.device
    dtype = I_t.dtype

    synthetic_spacing = _mean_spacing_from_axes(xI_t, xJ_t)
    dv_z = float(synthetic_spacing)

    synthetic_axis = _make_synthetic_z_axis(
        synthetic_spacing=synthetic_spacing,
        dv_z=dv_z,
        device=device,
        dtype=dtype,
        min_velocity_intervals=min_velocity_intervals,
    )
    # dv_z = _synthetic_dv_z_from_config(
    #     config,
    #     synthetic_spacing,
    # ) # slice stencil amenable to 3D fourier transforms in em-lddmm

    # synthetic_axis = _make_synthetic_z_axis(
    #     synthetic_spacing=synthetic_spacing,
    #     dv_z=dv_z,
    #     device=device,
    #     dtype=dtype,
    #     min_velocity_intervals=min_velocity_intervals,
    # ) 

    n_planes = int(synthetic_axis.numel())

    xI_backend = (synthetic_axis, *xI_t)
    xJ_backend = (synthetic_axis, *xJ_t)

    I_backend = (
        I_t[:, None]
        .expand(-1, n_planes, -1, -1)
        .contiguous()
    )

    J_backend = (
        J_t[:, None]
        .expand(-1, n_planes, -1, -1)
        .contiguous()
    )

    W0_backend = (
        W0_t[None]
        .expand(n_planes, -1, -1)
        .contiguous()
    )

    return (
        xI_backend,
        xJ_backend,
        I_backend,
        J_backend,
        W0_backend,
        synthetic_axis,
        synthetic_spacing,
    )


def _promote_2d_pair_config_for_backend(config, dtype, device):
    backend_config = dict(config)
    backend_config["device"] = str(device)
    backend_config["dtype"] = str(dtype).replace("torch.", "")
    backend_config["downI"] = _prepend_synthetic_axis_to_2d_schedule(
        backend_config.get("downI", [[1, 1]]),
        1,
    )
    backend_config["downJ"] = _prepend_synthetic_axis_to_2d_schedule(
        backend_config.get("downJ", [[1, 1]]),
        1,
    )
    backend_config.setdefault("out_of_plane", False) # does 'True' make sense? It works and has very small nonzero v_z value - let's keep false to ensure v_z=0
    backend_config.setdefault("dtype", dtype)
    backend_config["A"] = torch.eye(4, dtype=dtype)
    backend_config["A2d"] = None
    backend_config["Amode"] = 0
    backend_config["eA"] = 0.0
    backend_config["eA2d"] = 0.0
    backend_config["slice_matching"] = False
    backend_config["update_A"] = False
    backend_config["update_matching_weights"] = False

    return backend_config


def _extract_in_plane_velocity(output):
    last = output[-1] if isinstance(output, list) else output
    v3d = torch.as_tensor(last["v"])
    if v3d.ndim != 5 or v3d.shape[1] != 3:
        raise ValueError(
            f"Expected backend velocity with shape (T, 3, Z, Y, X); got {tuple(v3d.shape)}"
        )
    # Make the 3d registration into a 2d problem again
    v2d = v3d[:, 1:].mean(dim=2) # drops v_z; indep. averages v_x, v_y over the synthetic z-axis
    xv = tuple(
        torch.as_tensor(axis, device=v3d.device, dtype=v3d.dtype) for axis in last["xv"][-2:]
    )
    return last, v2d, xv


def _lift_in_plane_velocity_to_backend(v2d, z_size):
    # Make the 2D registration into a 3D problem by 
    # synthetically stacking the 2D velocity along the z-axis.
    v3d = torch.zeros(
        (v2d.shape[0], 3, z_size, v2d.shape[-2], v2d.shape[-1]),
        device=v2d.device,
        dtype=v2d.dtype,
    )
    v3d[:, 1:] = v2d[:, :, None].expand(-1, -1, z_size, -1, -1)
    return v3d


def emlddmm_multiscale_symmetric_N(  # noqa: E741
    xI,
    I,
    xJ,
    J,
    W0=None,
    *,
    combine_velocities="forward", # "average", "backward"
    grid_sample_kwargs=None,
    **config,
):
    r"""
    Symmetric EM-LDDMM: forward + backward passes with a shared symmetric velocity.
    Returns forward/backward outputs plus time-resolved warped images for both paths.

    Deform image I to match J.
    The diffeomorphic regularization energy is \int_X | Lv |^{2}_{L^2}, L = (Id - alpha^2 Laplacian)^2
    The matching energy is \int_X | I(phi^{-1}) - J |^{2}_{L^2} / (2*sigma^2)
    The flow is discretized to nT timesteps.
    The energy is optimized using gradient descent with stepsize epsilon for nIter steps.
    The energy gradient is -(I - J)grad(I)det(D phi_{1t})
    In the multiimage setting, images I and J each have slices.
    The velocity is discretized with nT timesteps. Stored transforms include the
    identity state, so internal trajectories have nT+1 states, while returned
    It/Jt match the MATLAB convention and contain the nT non-identity samples.

    Example
    -------
    pair_out = emlddmm_multiscale_symmetric_N(
        xI=[xJ], I=J[:, idx0, :, :],  # forward atlas -> target
        xJ=[xJ], J=J[:, idx1, :, :],
        W0=W, nt=config["nt"], **config
    )
    t_mid = config["nt"] // 2
    J_mid = pair_out["It"][t_mid]       # atlas slice flowed halfway toward target
    J_mid_w = pair_out["Jt"][t_mid]     # target slice flowed halfway toward atlas
    """
    # Initialize
    emlddmm_module = _resolve_emlddmm_module()
    requested_device = config.get("device", None)

    if requested_device is None:
        # Preserve existing behavior when no device is explicitly requested.
        if torch.is_tensor(I):
            device = I.device
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    # Preserve the input floating dtype unless config explicitly specifies one.
    dtype_name = config.get("dtype", None)

    if dtype_name is None:
        if torch.is_tensor(I) and I.is_floating_point():
            dtype = I.dtype
        else:
            dtype = torch.float32
    else:
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
        }
        try:
            dtype = dtype_map[str(dtype_name)]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported registration dtype {dtype_name!r}"
            ) from exc

    I_t = torch.as_tensor(
        I,
        device=device,
        dtype=dtype,
    )

    J_t = torch.as_tensor(
        J,
        device=device,
        dtype=dtype,
    )

    W0_t = (
        torch.ones_like(
            I_t[0],
            device=device,
            dtype=dtype,
        )
        if W0 is None
        else torch.as_tensor(
            W0,
            device=device,
            dtype=dtype,
        )
    )

    # emlddmm_multiscale interprets Python lists as per-scale schedules. The
    # coordinate axes for a 2D pair are a fixed domain, so keep them as tuples.
    xI_t = tuple(
        torch.as_tensor(
            x,
            device=device,
            dtype=dtype,
        )
        for x in xI
    )

    xJ_t = tuple(
        torch.as_tensor(
            x,
            device=device,
            dtype=dtype,
        )
        for x in xJ
    )

    is_2d_pair = (I_t.ndim == 3)
    if is_2d_pair:
        # The installed EM-LDDMM optimizer builds 3D affine/velocity domains even
        # for 2D interpolation. Promote only the backend solve, then average the
        # synthetic z support back to a 2D velocity before warping the original pair.
        
        # build a synthetic z axis with spacing equal to the mean of the in-plane spacings
        # note that the synthetic domain must be thick enough that the em-lddmm's deformation 
        # lattice has a nondegenerate z dimension. Otherwise, v_z will be NaN and v_x,v_y will be 0.
        # Promote the 2D to a 3D registration with the synthetic z axis.
        (
            xI_backend, # extruded domain domain
            xJ_backend, # extruded target domain
            I_backend, # extruded source image
            J_backend, # extruded target image
            W0_backend, # extruded target image mask/weights
            synthetic_axis, # z-axis amenable to 3D fourier transforms in em-lddmm
            synthetic_spacing,
        ) = _extrude_2d_pair_for_3d_backend(
            I_t,
            J_t,
            W0_t,
            xI_t,
            xJ_t,
            config,
        ) # this creates a 5-slice sandwich of identity slices, centered on the observed 2d slice
        # this is a hack; there's no anatomical meaning to this synthetically extruded axis
        # there's no new information in the z-direction, and the meaningful info remains in-plane

        backend_cfg = _promote_2d_pair_config_for_backend(config, dtype, device)
        backend_cfg["dv"] = _promote_2d_dv_schedule(
            config.get("dv"),
            synthetic_spacing,
        )

    else:
        xI_backend = xI_t
        xJ_backend = xJ_t
        I_backend = I_t
        J_backend = J_t
        W0_backend = W0_t
        backend_cfg = dict(config)

    # Flow forward
    fwd_cfg = dict(backend_cfg)
    # # DEBUG
    # print(
    #     "WSI 2D backend config:",
    #     "dv=", backend_cfg.get("dv"),
    #     "out_of_plane=", backend_cfg.get("out_of_plane"),
    # )
    out_fwd = emlddmm_module.emlddmm_multiscale(
        xI=xI_backend,
        I=I_backend,
        xJ=xJ_backend,
        J=J_backend,
        W0=W0_backend,
        **fwd_cfg,
    )
    if is_2d_pair:
        fwd_last, v_fwd, xv = _extract_in_plane_velocity(out_fwd)
    else:
        fwd_last = out_fwd[-1] if isinstance(out_fwd, list) else out_fwd  # last scale output
        v_fwd, xv = fwd_last["v"], [x.clone() for x in fwd_last["xv"]]

    # Now for the symmetric part: flip and negate the forward velocity to get a guess for the backward velocity
    v_init_back = torch.flip(-v_fwd, [0])

    # Flow backward
    back_cfg = dict(backend_cfg)
    if is_2d_pair:
        back_cfg.setdefault(
            "v",
            _lift_in_plane_velocity_to_backend(v_init_back, fwd_last["v"].shape[2]),
        )
        out_bwd = emlddmm_module.emlddmm_multiscale(
            xI=xJ_backend,
            I=J_backend,
            xJ=xI_backend,
            J=I_backend,
            W0=W0_backend,
            **back_cfg,
        )
        back_last, v_back_raw, _ = _extract_in_plane_velocity(out_bwd)
        v_back = torch.flip(-v_back_raw, [0])
    else:
        back_cfg.setdefault("v", v_init_back)
        out_bwd = emlddmm_module.emlddmm_multiscale(
            xI=xJ_t,
            I=J_t,
            xJ=xI_t,
            J=I_t,
            W0=W0_t,
            **back_cfg,
        )
        back_last = out_bwd[-1] if isinstance(out_bwd, list) else out_bwd  # last scale output
        v_back = torch.flip(-back_last["v"], [0])

    # DEBUG: compare raw backend velocities with extracted 2D velocities.
    fwd_debug_last, v_fwd_debug, xv_fwd_debug = _extract_in_plane_velocity(out_fwd)
    bwd_debug_last, v_bwd_raw_debug, xv_bwd_debug = _extract_in_plane_velocity(out_bwd)

    raw_fwd = torch.as_tensor(fwd_debug_last["v"])
    raw_bwd = torch.as_tensor(bwd_debug_last["v"])

    print(
        "RAW BACKEND VELOCITIES:",
        "fwd shape=", tuple(raw_fwd.shape),
        "fwd max=", float(torch.max(torch.abs(raw_fwd))),
        "fwd rms=", float(torch.sqrt(torch.mean(raw_fwd**2))),
        "bwd shape=", tuple(raw_bwd.shape),
        "bwd max=", float(torch.max(torch.abs(raw_bwd))),
        "bwd rms=", float(torch.sqrt(torch.mean(raw_bwd**2))),
    )

    print(
        "EXTRACTED 2D VELOCITIES:",
        "fwd shape=", tuple(v_fwd_debug.shape),
        "fwd max=", float(torch.max(torch.abs(v_fwd_debug))),
        "fwd rms=", float(torch.sqrt(torch.mean(v_fwd_debug**2))),
        "bwd raw shape=", tuple(v_bwd_raw_debug.shape),
        "bwd raw max=", float(torch.max(torch.abs(v_bwd_raw_debug))),
        "bwd raw rms=", float(torch.sqrt(torch.mean(v_bwd_raw_debug**2))),
    )

    print("RAW FORWARD COMPONENT MAXIMA:")
    for component in range(raw_fwd.shape[1]):
        for z in range(raw_fwd.shape[2]):
            slab = raw_fwd[:, component, z]
            print(
                f"  component={component}, z={z}: "
                f"max={float(torch.max(torch.abs(slab))):.6g}, "
                f"mean={float(torch.mean(slab)):.6g}"
            )

    v_bwd_debug = torch.flip(-v_bwd_raw_debug, [0])

    print(
        "ORIENTED RELATIONS:",
        "max |fwd-bwd|=",
        float(torch.max(torch.abs(v_fwd_debug - v_bwd_debug))),
        "max |fwd+bwd|=",
        float(torch.max(torch.abs(v_fwd_debug + v_bwd_debug))),
    )

    if combine_velocities == "average":
        # I actually don't think this makes sense to do
        v_sym = 0.5 * (v_fwd + v_back) 
    elif combine_velocities == "forward":
        v_sym = v_fwd
    elif combine_velocities == "backward":
        v_sym = v_back
    else:
        raise ValueError(f"Unknown combine_velocities='{combine_velocities}'")

    # Now compute the forward and backward image flows using the symmetric velocity
    # No need for tissue weighting here because we assume that's been properly handled by the forward and reverse mappings
    interp2d = is_2d_pair  # (C, H, W) -> 2D, otherwise 3D
    phi_I_velocity = _integrate_inverse_flow(
        xv,
        v_sym,
        emlddmm_module=emlddmm_module,
        interp2d=interp2d,
        grid_sample_kwargs=grid_sample_kwargs,
    )
    phi_J_velocity = _integrate_inverse_flow(
        xv,
        torch.flip(-v_sym, [0]),
        emlddmm_module=emlddmm_module,
        interp2d=interp2d,
        grid_sample_kwargs=grid_sample_kwargs,
    )
    xI_flow = xI_t[: phi_I_velocity.shape[1]]
    xJ_flow = xJ_t[: phi_J_velocity.shape[1]]
    phi_I = _resample_transform_to_domain(
        xv,
        phi_I_velocity,
        xI_flow,
        emlddmm_module=emlddmm_module,
        interp2d=interp2d,
        grid_sample_kwargs=grid_sample_kwargs,
    )
    phi_J = _resample_transform_to_domain(
        xv,
        phi_J_velocity,
        xJ_flow,
        emlddmm_module=emlddmm_module,
        interp2d=interp2d,
        grid_sample_kwargs=grid_sample_kwargs,
    )
    It_flow_all = _warp_time_series(
        xI_flow,
        I_t,
        phi_I,
        emlddmm_module=emlddmm_module,
        interp2d=interp2d,
        grid_sample_kwargs=grid_sample_kwargs,
    )
    Jt_flow_all = _warp_time_series(
        xJ_flow,
        J_t,
        phi_J,
        emlddmm_module=emlddmm_module,
        interp2d=interp2d,
        grid_sample_kwargs=grid_sample_kwargs,
    )

    # Calculate jacobian determinants at each time step
    spacing_I = tuple(float(abs(x[1] - x[0])) for x in xI[-2:])
    spacing_J = tuple(float(abs(x[1] - x[0])) for x in xJ[-2:])
    det_jac_phi_I_all = _calculate_determinant_of_jacobian(phi_I, spacing=spacing_I)
    det_jac_phi_J_all = _calculate_determinant_of_jacobian(phi_J, spacing=spacing_J)

    # TODO: what should I do about the weights? For now I will output the weights from the forward and reverse mappings in forward and backward

    out = {
        "forward": out_fwd,
        "backward": out_bwd,
        "v_symmetric": v_sym.detach().cpu(),
        "phi_I": phi_I.detach().cpu(),
        "phi_J": phi_J.detach().cpu(),
        "It": It_flow_all[1:].detach().cpu(),
        "Jt": Jt_flow_all[1:].detach().cpu(),
        "ItAll": It_flow_all.detach().cpu(),
        "JtAll": Jt_flow_all.detach().cpu(),
        "det_jac_phi_I": det_jac_phi_I_all[1:].detach().cpu(),
        "det_jac_phi_J": det_jac_phi_J_all[1:].detach().cpu(),
        "det_jac_phi_I_all": det_jac_phi_I_all.detach().cpu(),
        "det_jac_phi_J_all": det_jac_phi_J_all.detach().cpu(),
    }

    return out
