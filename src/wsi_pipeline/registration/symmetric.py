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
        return [synthetic_value, *entries]
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


def _promote_2d_pair_config_for_backend(config, dtype):
    backend_config = dict(config)
    backend_config["downI"] = _prepend_synthetic_axis_to_2d_schedule(
        backend_config.get("downI", [1, 1]),
        1,
    )
    backend_config["downJ"] = _prepend_synthetic_axis_to_2d_schedule(
        backend_config.get("downJ", [1, 1]),
        1,
    )
    backend_config["out_of_plane"] = True
    backend_config.setdefault("dtype", dtype)
    backend_config["A"] = torch.eye(4, dtype=dtype)
    backend_config["A2d"] = None
    backend_config["Amode"] = 0
    backend_config["eA"] = 0.0
    backend_config["eA2d"] = 0.0
    backend_config["slice_matching"] = False
    return backend_config


def _extract_in_plane_velocity(output):
    last = output[-1] if isinstance(output, list) else output
    v3d = torch.as_tensor(last["v"])
    if v3d.ndim != 5 or v3d.shape[1] != 3:
        raise ValueError(
            f"Expected backend velocity with shape (T, 3, Z, Y, X); got {tuple(v3d.shape)}"
        )
    v2d = v3d[:, 1:].mean(dim=2)
    xv = tuple(
        torch.as_tensor(axis, device=v3d.device, dtype=v3d.dtype) for axis in last["xv"][-2:]
    )
    return last, v2d, xv


def _lift_in_plane_velocity_to_backend(v2d, z_size):
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
    combine_velocities="average",
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
    I_t = torch.as_tensor(I)
    J_t = torch.as_tensor(J)
    device, dtype = I_t.device, I_t.dtype
    W0_t = (
        torch.ones_like(I_t[0], device=device, dtype=dtype)
        if W0 is None
        else torch.as_tensor(W0, device=device, dtype=dtype)
    )
    # emlddmm_multiscale interprets Python lists as per-scale schedules. The
    # coordinate axes for a 2D pair are a fixed domain, so keep them as tuples.
    xI_t = tuple(torch.as_tensor(x, device=device, dtype=dtype) for x in xI)
    xJ_t = tuple(torch.as_tensor(x, device=device, dtype=dtype) for x in xJ)
    is_2d_pair = I_t.ndim == 3
    if is_2d_pair:
        # The installed EM-LDDMM optimizer builds 3D affine/velocity domains even
        # for 2D interpolation. Promote only the backend solve, then average the
        # synthetic z support back to a 2D velocity before warping the original pair.
        synthetic_spacing = _mean_spacing_from_axes(xI_t, xJ_t)
        synthetic_axis = torch.tensor(
            [0.0, synthetic_spacing],
            device=device,
            dtype=dtype,
        )
        xI_backend = (synthetic_axis, *xI_t)
        xJ_backend = (synthetic_axis, *xJ_t)
        I_backend = I_t[:, None].expand(-1, 2, -1, -1).contiguous()
        J_backend = J_t[:, None].expand(-1, 2, -1, -1).contiguous()
        W0_backend = W0_t[None].expand(2, -1, -1).contiguous()
        backend_cfg = _promote_2d_pair_config_for_backend(config, dtype)
    else:
        xI_backend = xI_t
        xJ_backend = xJ_t
        I_backend = I_t
        J_backend = J_t
        W0_backend = W0_t
        backend_cfg = dict(config)

    # Flow forward
    fwd_cfg = dict(backend_cfg)
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
        out_back = emlddmm_module.emlddmm_multiscale(
            xI=xJ_backend,
            I=J_backend,
            xJ=xI_backend,
            J=I_backend,
            W0=W0_backend,
            **back_cfg,
        )
        back_last, v_back_raw, _ = _extract_in_plane_velocity(out_back)
        v_back = torch.flip(-v_back_raw, [0])
    else:
        back_cfg.setdefault("v", v_init_back)
        out_back = emlddmm_module.emlddmm_multiscale(
            xI=xJ_t,
            I=J_t,
            xJ=xI_t,
            J=I_t,
            W0=W0_t,
            **back_cfg,
        )
        back_last = out_back[-1] if isinstance(out_back, list) else out_back  # last scale output
        v_back = torch.flip(-back_last["v"], [0])

    if combine_velocities == "average":
        v_sym = 0.5 * (v_fwd + v_back)
    elif combine_velocities == "forward":
        v_sym = v_fwd
    elif combine_velocities == "backward":
        v_sym = v_back
    else:
        raise ValueError(f"Unknown combine_velocities='{combine_velocities}'")

    # Now compute the forward and backward image flows using the average symmetric velocity
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
        "backward": out_back,
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
