#!/usr/bin/env python3

# This code just copy-and-pastes the codes from 11/18/2025.
# There are no edits yet as of 02/27/2026

# Imports
import torch
import numpy as np
from emlddmm import emlddmm 

# Dom: added 11/18/2025
def _integrate_inverse_flow(xv, v, *, interp2d=None, grid_sample_kwargs=None):
    """Integrate v_t -> phi^{-1}_t, storing all time steps."""
    v = torch.as_tensor(v)
    xv = [torch.as_tensor(x, device=v.device, dtype=v.dtype) for x in xv]
    ndim = v.shape[1]
    interp2d = bool(interp2d) if interp2d is not None else ndim == 2
    mesh = torch.stack(torch.meshgrid(xv[:ndim], indexing='ij'))
    phi = mesh
    phis = [phi]
    dt = 1.0 / v.shape[0]
    for t in range(v.shape[0]):
        Xs = mesh - v[t] * dt
        phi = emlddmm.interp(xv[:ndim], phi - mesh, Xs, interp2d=interp2d, **(grid_sample_kwargs or {})) + Xs
        phis.append(phi)
    return torch.stack(phis)


def _warp_time_series(x, image, phis, *, interp2d=None, grid_sample_kwargs=None):
    """Warp an image along a stored phi^{-1}_t trajectory."""
    image = torch.as_tensor(image, device=phis.device)
    warped = []
    for t in range(phis.shape[0]):
        warped.append(emlddmm.interp(x, image, phis[t], interp2d=interp2d, **(grid_sample_kwargs or {})))
    return torch.stack(warped)


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
    grads = []
    for c in range(phi.shape[1]):
        gx, gy = torch.gradient(phi[:, c], spacing, dim=(-2, -1), edge_order=2)
        grads.append((gx, gy))

    # determinant: \partial_x phi_x * \partial_y phi_y - \partial_y phi_x * \partial_x phi_y
    det = grads[0][0] * grads[1][1] - grads[0][1] * grads[1][0]
    return det

def emlddmm_multiscale_symmetric_N(xI, I, xJ, J, W0=None, *, combine_velocities="average",
                                   grid_sample_kwargs=None, **config):
    """
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
    I_t = torch.as_tensor(I)
    J_t = torch.as_tensor(J)
    device, dtype = I_t.device, I_t.dtype
    W0_t = torch.ones_like(I_t[0], device=device, dtype=dtype) if W0 is None else torch.as_tensor(W0, device=device, dtype=dtype)
    xI_t = [torch.as_tensor(x, device=device, dtype=dtype) for x in xI]
    xJ_t = [torch.as_tensor(x, device=device, dtype=dtype) for x in xJ]
    # Flow forward
    fwd_cfg = dict(config)
    out_fwd = emlddmm.emlddmm_multiscale(xI=xI_t, I=I_t, xJ=xJ_t, J=J_t, W0=W0_t, **fwd_cfg)
    fwd_last = out_fwd[-1] if isinstance(out_fwd, list) else out_fwd # last scale output
    v_fwd, xv = fwd_last["v"], [x.clone() for x in fwd_last["xv"]]

    # Now for the symmetric part: flip and negate the forward velocity to get a guess for the backward velocity 
    v_init_back = torch.flip(-v_fwd, [0]) 

    # Flow backward
    back_cfg = dict(config)
    back_cfg.setdefault("v", v_init_back)
    out_back = emlddmm.emlddmm_multiscale(xI=xJ_t, I=J_t, xJ=xI_t, J=I_t, W0=W0_t, **back_cfg)
    back_last = out_back[-1] if isinstance(out_back, list) else out_back # last scale output
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
    interp2d = (I_t.ndim == 3)  # (C, H, W) -> 2D, otherwise 3D
    phi_I = _integrate_inverse_flow(xv, v_sym, interp2d=interp2d, grid_sample_kwargs=grid_sample_kwargs)
    phi_J = _integrate_inverse_flow(xv, torch.flip(-v_sym, [0]), interp2d=interp2d, grid_sample_kwargs=grid_sample_kwargs)
    It_flow_all = _warp_time_series(xI_t[:phi_I.shape[1]], I_t, phi_I, interp2d=interp2d, grid_sample_kwargs=grid_sample_kwargs)
    Jt_flow_all = _warp_time_series(xJ_t[:phi_J.shape[1]], J_t, phi_J, interp2d=interp2d, grid_sample_kwargs=grid_sample_kwargs)

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
