"""Arbitrary-pseudotime evaluation of the stored between-slice trajectories.

These tests run the real symmetric post-processing (flow integration, grid
resampling, warping, Jacobians) on prescribed velocities. The optimizer is
replaced by a stub, and `interp` is a copy of EM-LDDMM's grid_sample wrapper,
so no registration optimization or EM-LDDMM checkout is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import wsi_pipeline.registration.symmetric as symmetric_module
import wsi_pipeline.registration.upsample as upsample_module


def _interp(x, I, phii, interp2d=False, **kwargs):
    """Same normalization and grid_sample call as emlddmm.interp."""
    I = torch.as_tensor(I)
    phii = torch.clone(torch.as_tensor(phii))
    ndim = 2 if interp2d else 3
    for i in range(ndim):
        phii[i] -= x[i][0]
        phii[i] /= x[i][-1] - x[i][0]
    phii = phii * 2.0 - 1.0
    kwargs.setdefault("padding_mode", "border")
    perm = (1, 2, 0) if interp2d else (1, 2, 3, 0)
    phii = phii.flip(0).permute(perm)[None]
    return torch.nn.functional.grid_sample(I[None], phii, align_corners=True, **kwargs)[0]


def _use_backend(monkeypatch, multiscale=None):
    backend = SimpleNamespace(
        interp=_interp,
        emlddmm_multiscale=multiscale or (lambda **kwargs: None),
    )
    monkeypatch.setattr(symmetric_module, "_resolve_emlddmm_module", lambda: backend)
    return backend


def _axes2d(n=9):
    axis = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
    return (axis, axis.clone())


def _mesh(xv):
    return torch.stack(torch.meshgrid(xv, indexing="ij"))


PSEUDOTIMES = [0.1, 0.25, 0.5, 0.73, 0.9]


def test_zero_velocity_is_identity_at_arbitrary_pseudotimes(monkeypatch):
    backend = _use_backend(monkeypatch)
    xv = _axes2d()
    v = torch.zeros((8, 2, 9, 9), dtype=torch.float64)

    for s in PSEUDOTIMES:
        phi = upsample_module._partial_inverse_flow(xv, v, s, emlddmm_module=backend)
        assert torch.allclose(phi, _mesh(xv), atol=1e-12)

    image = np.random.default_rng(0).random((2, 9, 9))
    out = {"v_symmetric": v, "xv": xv}
    warped, det, _ = upsample_module._warp_at_times(
        out, image, [a.numpy() for a in xv], PSEUDOTIMES, device=torch.device("cpu")
    )
    assert np.allclose(warped, image[None], atol=1e-12)
    assert np.allclose(det, 1.0, atol=1e-12)


def test_constant_translation_gives_fractional_displacement(monkeypatch):
    backend = _use_backend(monkeypatch)
    xv = _axes2d()
    nt = 8
    c = torch.tensor([0.3, -0.2], dtype=torch.float64)
    v = c[None, :, None, None].expand(nt, 2, 9, 9).clone()

    for s in PSEUDOTIMES:
        phi = upsample_module._partial_inverse_flow(xv, v, s, emlddmm_module=backend)
        expected = _mesh(xv) - s * c[:, None, None]
        assert torch.allclose(phi, expected, atol=1e-12), s


def test_time_dependent_velocity_uses_fractional_final_step(monkeypatch):
    """v_t(x) = a_t x gives phi^{-1}_s(x) = x prod_{k<m}(1 - a_k dt) (1 - f a_m dt)."""
    backend = _use_backend(monkeypatch)
    xv = _axes2d()
    nt = 8
    dt = 1.0 / nt
    a = torch.linspace(0.2, 1.6, nt, dtype=torch.float64)
    v = a[:, None, None, None] * _mesh(xv)[None]

    for s in PSEUDOTIMES:
        k = s * nt
        m = int(np.floor(k))
        f = k - m
        scale = float(torch.prod(1.0 - a[:m] * dt))
        if f > 0:
            scale *= 1.0 - f * float(a[m]) * dt
        phi = upsample_module._partial_inverse_flow(xv, v, s, emlddmm_module=backend)
        assert torch.allclose(phi, scale * _mesh(xv), atol=1e-12), s

        # Scaling the whole time-dependent velocity by s gives a different map.
        rescaled = float(torch.prod(1.0 - s * a * dt))
        assert not np.isclose(scale, rescaled, atol=1e-4)

    # Jacobian of the same map on the image grid: det = scale**2 for 2D scaling.
    out = {"v_symmetric": v, "xv": xv}
    _, det, _ = upsample_module._warp_at_times(
        out, np.ones((1, 9, 9)), [ax.numpy() for ax in xv], [0.73], device=torch.device("cpu")
    )
    scale = float(torch.prod(1.0 - a[:5] * dt)) * (1.0 - 0.84 * float(a[5]) * dt)
    assert np.allclose(det, scale**2, atol=1e-10)


def test_partial_flow_endpoints_are_exact(monkeypatch):
    backend = _use_backend(monkeypatch)
    xv = _axes2d()
    rng = torch.Generator().manual_seed(0)
    v = 0.1 * torch.randn((6, 2, 9, 9), generator=rng, dtype=torch.float64)
    stored = symmetric_module._integrate_inverse_flow(xv, v, emlddmm_module=backend)

    phi0 = upsample_module._partial_inverse_flow(xv, v, 0.0, emlddmm_module=backend)
    phi1 = upsample_module._partial_inverse_flow(xv, v, 1.0, emlddmm_module=backend)
    assert torch.equal(phi0, _mesh(xv))
    assert torch.equal(phi1, stored[-1])
    for j in range(7):
        phij = upsample_module._partial_inverse_flow(xv, v, j / 6, emlddmm_module=backend)
        assert torch.equal(phij, stored[j])


# -------------------------------------------------------------------------
# Full pair path: real symmetric helper with a stub optimizer
# -------------------------------------------------------------------------

NT = 4
SIZE = 16


def _stub_multiscale(**kwargs):
    """Return a smooth, time-dependent, solve-dependent velocity on a coarse grid."""
    z, y, x = kwargs["xI"]
    yv = torch.linspace(float(y[0]), float(y[-1]), 9, dtype=y.dtype)
    xv = torch.linspace(float(x[0]), float(x[-1]), 7, dtype=x.dtype)
    sign = 1.0 if float(kwargs["I"].mean()) < float(kwargs["J"].mean()) else -1.0
    Y, X = torch.meshgrid(yv, xv, indexing="ij")
    v = torch.zeros((NT, 3, z.numel(), yv.numel(), xv.numel()), dtype=y.dtype)
    for t in range(NT):
        v[t, 1] = sign * (1.0 + t) * torch.sin(X / 4.0)[None]
        v[t, 2] = sign * (2.0 - 0.3 * t) * torch.cos(Y / 5.0)[None]
    return {"v": v, "xv": [z, yv, xv]}


def _pair_stack():
    rng = np.random.default_rng(1)
    J = np.zeros((1, 5, SIZE, SIZE), dtype=np.float32)
    J[:, 0] = rng.random((SIZE, SIZE))
    J[:, 4] = 1.0 + rng.random((SIZE, SIZE))
    xJ = [
        np.arange(5, dtype=np.float32) * 2.5,
        np.arange(SIZE, dtype=np.float32),
        np.arange(SIZE, dtype=np.float32),
    ]
    present = np.array([True, False, False, False, True])
    return xJ, J, present


def _register_and_evaluate(xJ, J, config, pseudotimes, path_mode="two-sided"):
    """Register pair (0, 4), then evaluate its stored velocities at `pseudotimes`."""
    _, _, I0t, I1t, J0t, J1t, _, _, stored = upsample_module._register_pair(
        0, 4, xJ, J, None, config, mode="img"
    )
    I0, I1 = upsample_module._pair_endpoint_images(J, 0, 4, "img")
    samples = upsample_module._evaluate_pair_pseudotimes(
        {"v_symmetric": stored["v_forward"], "xv": stored["xv_forward"]},
        {"v_symmetric": stored["v_backward"], "xv": stored["xv_backward"]},
        I0,
        I1,
        (xJ[1], xJ[2]),
        pseudotimes,
        config,
        path_mode,
    )
    return (I0t, I1t, J0t, J1t), samples[:4]


@pytest.fixture
def captured_solves(monkeypatch):
    _use_backend(monkeypatch, _stub_multiscale)
    captured = []

    def capturing(*args, **kwargs):
        out = symmetric_module.emlddmm_multiscale_symmetric_N(*args, **kwargs)
        captured.append(out)
        return out

    monkeypatch.setattr(upsample_module, "emlddmm_multiscale_symmetric_N", capturing)
    return captured


def test_pseudotimes_on_frames_match_stored_trajectories(captured_solves):
    xJ, J, _ = _pair_stack()
    config = upsample_module._normalize_pair_registration_config({"nt": NT})
    frames = [j / NT for j in range(NT + 1)]

    (I0t, I1t, J0t, J1t), (I0s, I1s, J0s, J1s) = _register_and_evaluate(xJ, J, config, frames)
    out_bwd = captured_solves[1]
    assert float(np.abs(I0t[-1] - J[:, 0]).max()) > 1e-2  # the stub velocity moves I0

    for j in range(NT):
        # I0t[j] / J0t[j] are forward frame j+1, i.e. s = (j+1)/nt.
        assert np.allclose(I0s[j + 1], I0t[j], atol=1e-5)
        assert np.allclose(J0s[j + 1], J0t[j], atol=1e-5)
        # I1t[j] is backward frame nt-j, i.e. s = j/nt.
        assert np.allclose(I1s[j], I1t[j], atol=1e-5)
        # J1s is the Jacobian of the same backward I1 warp.
        assert np.allclose(J1s[j], np.asarray(out_bwd["det_jac_phi_I_all"])[NT - j], atol=1e-5)

    assert np.allclose(I0s[0], J[:, 0], atol=1e-5)
    assert np.allclose(I1s[NT], J[:, 4], atol=1e-5)
    assert np.allclose(J0s[0], 1.0, atol=1e-5)
    assert np.allclose(J1s[NT], 1.0, atol=1e-5)


@pytest.mark.parametrize("mode", ["img", "seg"])
def test_pseudotime_samples_are_returned_separately(captured_solves, mode):
    xJ, J, present = _pair_stack()
    W = np.ones(J.shape[1:], dtype=np.float32) if mode == "img" else None
    pseudotimes = [0.0, 0.1, 0.25, 0.5, 0.73, 1.0]

    baseline = upsample_module.upsample_between_slices(
        xJ, J, W, present_mask=present, mode=mode, config={"nt": NT}
    )
    out = upsample_module.upsample_between_slices(
        xJ, J, W, present_mask=present, mode=mode, config={"nt": NT}, pseudotimes=pseudotimes
    )

    assert "pseudotime_samples" not in baseline
    assert set(out) == set(baseline) | {"pseudotime_samples"}
    assert np.array_equal(out["J_filled"], baseline["J_filled"])
    assert np.array_equal(out["J_nearest_bad"], baseline["J_nearest_bad"])

    samples = out["pseudotime_samples"]
    assert samples["J"].shape == (1, len(pseudotimes), SIZE, SIZE)
    assert samples["pair_indices"].tolist() == [[0, 4]] * len(pseudotimes)
    assert np.allclose(samples["pseudotimes"], pseudotimes)
    assert np.allclose(samples["z"], 10.0 * np.asarray(pseudotimes))
    assert np.array_equal(samples["J"][:, 0], J[:, 0])
    assert np.array_equal(samples["J"][:, -1], J[:, 4])
    interior = samples["J"][:, 1:-1]
    assert np.all(np.isfinite(interior))
    if mode == "seg":
        assert np.array_equal(interior, np.round(interior))
    else:
        # Jacobian-weighted blend of endpoint intensities stays within their range.
        assert interior.min() >= J[:, 0].min() - 1e-4
        assert interior.max() <= J[:, 4].max() + 1e-4


def test_pseudotimes_require_stored_velocity(monkeypatch):
    def no_velocity_helper(xI, I, xJ, J, W0=None, **config):
        nt = int(config["nt"])
        return {
            "It": np.zeros((nt, *np.asarray(I).shape), dtype=np.float32),
            "det_jac_phi_I": np.ones((nt, *np.asarray(I).shape[1:]), dtype=np.float32),
            "det_jac_phi_J": np.ones((nt, *np.asarray(I).shape[1:]), dtype=np.float32),
        }

    monkeypatch.setattr(upsample_module, "emlddmm_multiscale_symmetric_N", no_velocity_helper)
    xJ, J, present = _pair_stack()
    with pytest.raises(ValueError, match="v_symmetric"):
        upsample_module.upsample_between_slices(
            xJ, J, present_mask=present, mode="seg", config={"nt": NT}, pseudotimes=[0.5]
        )


@pytest.mark.parametrize("bad", [[], [-0.1], [1.5], [np.nan], [[0.2, 0.4]]])
def test_pseudotimes_are_validated(bad):
    xJ, J, present = _pair_stack()
    with pytest.raises(ValueError, match="pseudotimes"):
        upsample_module.upsample_between_slices(
            xJ, J, present_mask=present, mode="seg", config={"nt": NT}, pseudotimes=bad
        )


# -------------------------------------------------------------------------
# path_mode="single-forward": both endpoints along the forward flow
# -------------------------------------------------------------------------

CPU = torch.device("cpu")


def _single_forward(v, xv, I0, I1, pseudotimes):
    return upsample_module._evaluate_pair_pseudotimes(
        {"v_symmetric": v, "xv": xv},
        None,
        I0,
        I1,
        [axis.numpy() for axis in xv],
        pseudotimes,
        {},
        "single-forward",
    )[:4]


def _map_to_i1(xv, v, s, backend):
    """Sampling map intermediate(s) -> I1, i.e. phi_1 o phi_s^{-1}."""
    return upsample_module._partial_inverse_flow(
        xv, torch.flip(-v, [0]), 1.0 - s, emlddmm_module=backend
    )


def _compose(xv, outer, inner):
    """outer o inner for maps stored as positions on the grid xv."""
    return _interp(xv, outer - _mesh(xv), inner, interp2d=True) + inner


def test_single_forward_zero_velocity_is_identity(monkeypatch):
    _use_backend(monkeypatch)
    xv = _axes2d()
    v = torch.zeros((8, 2, 9, 9), dtype=torch.float64)
    rng = np.random.default_rng(2)
    I0, I1 = rng.random((1, 9, 9)), rng.random((1, 9, 9))

    I0s, I1s, J0s, J1s = _single_forward(v, xv, I0, I1, PSEUDOTIMES)
    assert np.allclose(I0s, I0[None], atol=1e-12)
    assert np.allclose(I1s, I1[None], atol=1e-12)
    assert np.allclose(J0s, 1.0, atol=1e-12)
    assert np.allclose(J1s, 1.0, atol=1e-12)


def test_single_forward_translation_maps_to_i1_by_remaining_fraction(monkeypatch):
    backend = _use_backend(monkeypatch)
    xv = _axes2d()
    c = torch.tensor([0.3, -0.2], dtype=torch.float64)
    v = c[None, :, None, None].expand(8, 2, 9, 9).clone()

    for s in PSEUDOTIMES:
        # phi_s^{-1}(x) = x - s c and phi_1 o phi_s^{-1}(x) = x + (1 - s) c.
        expected = _mesh(xv) + (1.0 - s) * c[:, None, None]
        assert torch.allclose(_map_to_i1(xv, v, s, backend), expected, atol=1e-12), s


def test_single_forward_endpoints_share_intermediate_frame(monkeypatch):
    """If I1 = I0 o phi_1^{-1} exactly, both warped endpoints coincide at every s."""
    _use_backend(monkeypatch)
    n = 24
    axis = torch.arange(n, dtype=torch.float64)
    xv = (axis, axis.clone())
    c = torch.tensor([2.0, -3.0], dtype=torch.float64)  # whole pixels
    v = c[None, :, None, None].expand(8, 2, n, n).clone()

    yy, xx = np.mgrid[:n, :n]
    I0 = np.exp(-((yy - 10.0) ** 2 + (xx - 13.0) ** 2) / 6.0)[None]
    I1 = np.roll(I0, (2, -3), axis=(1, 2))  # I0(x - c)
    I0s, I1s, _, _ = _single_forward(v, xv, I0, I1, PSEUDOTIMES)
    assert np.allclose(I0s, I1s, atol=1e-6)

    # Two-sided with an independent backward flow does not share the frame.
    I0b, I1b, *_ = upsample_module._evaluate_pair_pseudotimes(
        {"v_symmetric": v, "xv": xv},
        {"v_symmetric": torch.zeros_like(v), "xv": xv},
        I0,
        I1,
        [axis.numpy()] * 2,
        PSEUDOTIMES,
        {},
        "two-sided",
    )
    assert np.allclose(I0b, I0s)
    assert np.abs(I0b - I1b).max() > 0.1


def _scaling_field(nt, n=101, lo=0.1, hi=0.5):
    # Fine grid: bilinear sampling spreads the border-clamped band ~one cell per
    # step, so it stays outside the |x| <= 0.6 interior used for exact checks.
    xv = _axes2d(n)
    a = torch.linspace(lo, hi, nt, dtype=torch.float64)
    v = a[:, None, None, None] * _mesh(xv)[None]
    interior = (_mesh(xv).abs() <= 0.6).all(dim=0)
    return xv, a, v, interior


def test_single_forward_time_dependent_reverse_partial_step(monkeypatch):
    """v_t(x) = a_t x: the I1 map takes a (1 - f) step with v[m], then v[m+1:]."""
    backend = _use_backend(monkeypatch)
    nt = 8
    dt = 1.0 / nt
    xv, a, v, interior = _scaling_field(nt)

    for s in PSEUDOTIMES:
        m = int(np.floor(s * nt))
        f = s * nt - m
        scale = float(torch.prod(1.0 + a[m + 1 :] * dt)) * (1.0 + (1.0 - f) * float(a[m]) * dt)
        phi = _map_to_i1(xv, v, s, backend)
        assert torch.allclose(phi[:, interior], scale * _mesh(xv)[:, interior], atol=1e-12), s

        # A fractional step of f (instead of 1 - f) with v[m] would be wrong.
        wrong = float(torch.prod(1.0 + a[m + 1 :] * dt)) * (1.0 + f * float(a[m]) * dt)
        if not np.isclose(f, 0.5):
            assert not np.isclose(scale, wrong, atol=1e-6)

    rng = np.random.default_rng(3)
    _, _, _, J1s = _single_forward(v, xv, rng.random((1, 101, 101)), rng.random((1, 101, 101)), [0.73])
    m, f = 5, 0.73 * nt - 5
    scale = float(torch.prod(1.0 + a[m + 1 :] * dt)) * (1.0 + (1.0 - f) * float(a[m]) * dt)
    assert np.allclose(J1s[0][interior.numpy()], scale**2, atol=1e-10)


def test_single_forward_maps_compose_to_full_flow(monkeypatch):
    """phi_1^{-1} o (phi_1 o phi_s^{-1}) = phi_s^{-1}, up to O(dt) discretization.

    The reverse map is integrated, not composed, so consistency is first order
    in dt until the O(h^2) interpolation floor; the grid is fine enough to see it.
    """
    backend = _use_backend(monkeypatch)
    xv = _axes2d(129)
    X = _mesh(xv)
    interior = (X.abs() <= 0.6).all(dim=0)
    s = 0.4
    errors = []
    for nt in (8, 16, 32):
        t = (torch.arange(nt, dtype=torch.float64) + 0.5) / nt
        v = torch.stack(
            [
                (0.3 + 0.4 * t)[:, None, None] * torch.sin(2.0 * X[1])[None],
                (0.2 - 0.3 * t)[:, None, None] * torch.cos(1.5 * X[0])[None],
            ],
            dim=1,
        )
        phi_s_inv = upsample_module._partial_inverse_flow(xv, v, s, emlddmm_module=backend)
        phi_1_inv = upsample_module._partial_inverse_flow(xv, v, 1.0, emlddmm_module=backend)
        to_i1 = _map_to_i1(xv, v, s, backend)
        err = (_compose(xv, phi_1_inv, to_i1) - phi_s_inv)[:, interior].abs().max()
        errors.append(float(err))

    assert errors[0] < 5e-3
    assert errors[1] < 0.6 * errors[0]
    assert errors[2] < 0.6 * errors[1]


def test_single_forward_endpoints(monkeypatch):
    backend = _use_backend(monkeypatch)
    xv = _axes2d()
    v = 0.1 * torch.randn((6, 2, 9, 9), generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    rng = np.random.default_rng(5)
    I0, I1 = rng.random((1, 9, 9)), rng.random((1, 9, 9))

    I0s, I1s, J0s, J1s = _single_forward(v, xv, I0, I1, [0.0, 1.0])
    assert np.allclose(I0s[0], I0, atol=1e-12) and np.allclose(J0s[0], 1.0, atol=1e-12)
    assert np.allclose(I1s[1], I1, atol=1e-12) and np.allclose(J1s[1], 1.0, atol=1e-12)
    # s=0: I1 is pulled back by the full reverse flow, phi_1.
    to_i1 = symmetric_module._integrate_inverse_flow(xv, torch.flip(-v, [0]), emlddmm_module=backend)
    assert torch.equal(_map_to_i1(xv, v, 0.0, backend), to_i1[-1])
    # s=1: I0 is warped by the full forward flow, as in two-sided mode.
    two_sided = upsample_module._evaluate_pair_pseudotimes(
        {"v_symmetric": v, "xv": xv}, {"v_symmetric": v, "xv": xv},
        I0, I1, [ax.numpy() for ax in xv], [1.0], {}, "two-sided",
    )
    assert np.array_equal(I0s[1], two_sided[0][0])


def test_single_forward_on_frames_matches_stored_forward_solve(captured_solves):
    xJ, J, _ = _pair_stack()
    config = upsample_module._normalize_pair_registration_config({"nt": NT})
    frames = [j / NT for j in range(NT + 1)]

    _, two_sided = _register_and_evaluate(xJ, J, config, frames)
    _, single = _register_and_evaluate(xJ, J, config, frames, "single-forward")
    out_fwd = captured_solves[2]
    for j in range(NT + 1):
        assert np.array_equal(single[0][j], two_sided[0][j])
        assert np.array_equal(single[2][j], two_sided[2][j])
        # Same operations as the solver's phi_J path (flip(-v), resample, warp, det).
        assert np.allclose(single[1][j], np.asarray(out_fwd["JtAll"])[NT - j], atol=1e-6)
        assert np.allclose(
            single[3][j], np.asarray(out_fwd["det_jac_phi_J_all"])[NT - j], atol=1e-6
        )


@pytest.mark.parametrize("mode", ["img", "seg"])
def test_path_mode_leaves_default_outputs_unchanged(captured_solves, mode):
    xJ, J, present = _pair_stack()
    W = np.ones(J.shape[1:], dtype=np.float32) if mode == "img" else None
    kwargs = dict(present_mask=present, mode=mode, config={"nt": NT})
    s = [0.0, 0.3, 0.73, 1.0]

    lattice = upsample_module.upsample_between_slices(xJ, J, W, **kwargs)
    default = upsample_module.upsample_between_slices(xJ, J, W, pseudotimes=s, **kwargs)
    explicit = upsample_module.upsample_between_slices(
        xJ, J, W, pseudotimes=s, path_mode="two-sided", **kwargs
    )
    single = upsample_module.upsample_between_slices(
        xJ, J, W, pseudotimes=s, path_mode="single-forward", **kwargs
    )

    for out in (default, explicit, single):
        assert np.array_equal(out["J_filled"], lattice["J_filled"])
    assert np.array_equal(default["pseudotime_samples"]["J"], explicit["pseudotime_samples"]["J"])
    assert default["pseudotime_samples"]["path_mode"] == "two-sided"
    assert single["pseudotime_samples"]["path_mode"] == "single-forward"
    for key in ("pair_indices", "pseudotimes", "z"):
        assert np.array_equal(single["pseudotime_samples"][key], default["pseudotime_samples"][key])
    Js = single["pseudotime_samples"]["J"]
    assert np.array_equal(Js[:, 0], J[:, 0]) and np.array_equal(Js[:, -1], J[:, 4])
    if mode == "img":
        assert not np.allclose(Js[:, 1:-1], default["pseudotime_samples"]["J"][:, 1:-1])


def test_path_mode_is_validated():
    xJ, J, present = _pair_stack()
    with pytest.raises(ValueError, match="path_mode"):
        upsample_module.upsample_between_slices(
            xJ, J, present_mask=present, mode="seg", config={"nt": NT},
            pseudotimes=[0.5], path_mode="single",
        )


# -------------------------------------------------------------------------
# Stored pair velocities: return, save, reload, re-evaluate
# -------------------------------------------------------------------------


def _three_slice_stack():
    """Two pairs, (0, 2) and (2, 5), on a nonuniform z axis."""
    rng = np.random.default_rng(6)
    J = np.zeros((2, 6, SIZE, SIZE), dtype=np.float32)
    for k, offset in ((0, 0.0), (2, 1.0), (5, 2.0)):
        J[:, k] = offset + rng.random((2, SIZE, SIZE))
    xJ = [
        np.array([0.0, 1.0, 3.0, 4.0, 4.5, 7.0], dtype=np.float32),
        np.arange(SIZE, dtype=np.float32),
        np.arange(SIZE, dtype=np.float32),
    ]
    present = np.array([True, False, True, False, False, True])
    return xJ, J, present


def test_return_velocities_matches_solver_outputs(captured_solves):
    xJ, J, present = _three_slice_stack()
    kwargs = dict(present_mask=present, mode="img", config={"nt": NT})

    default = upsample_module.upsample_between_slices(xJ, J, **kwargs)
    out = upsample_module.upsample_between_slices(xJ, J, return_velocities=True, **kwargs)

    assert "pair_velocities" not in default
    assert np.array_equal(out["J_filled"], default["J_filled"])
    vel = out["pair_velocities"]
    assert str(vel["format"]) == upsample_module.PAIR_VELOCITY_FORMAT
    assert str(vel["mode"]) == "img"
    assert vel["pairs"].tolist() == [[0, 2], [2, 5]]
    assert np.allclose(vel["z"], [[0.0, 3.0], [3.0, 7.0]])
    assert vel["v_forward"].shape == (2, NT, 2, 9, 7)
    # captured_solves[4:] are this call's solves: (fwd, bwd) per pair.
    for p in range(2):
        fwd, bwd = captured_solves[4 + 2 * p], captured_solves[5 + 2 * p]
        assert np.array_equal(vel["v_forward"][p], np.asarray(fwd["v_symmetric"]))
        assert np.array_equal(vel["v_backward"][p], np.asarray(bwd["v_symmetric"]))
        assert np.array_equal(vel["xv_y"], np.asarray(fwd["xv"][0]))
        assert np.array_equal(vel["xv_x"], np.asarray(bwd["xv"][1]))


@pytest.mark.parametrize("mode", ["img", "seg"])
@pytest.mark.parametrize("path_mode", ["two-sided", "single-forward"])
def test_saved_velocities_reproduce_samples_without_registration(
    captured_solves, tmp_path, mode, path_mode
):
    xJ, J, present = _three_slice_stack()
    W = np.ones(J.shape[1:], dtype=np.float32) if mode == "img" else None
    kwargs = dict(present_mask=present, mode=mode, config={"nt": NT}, path_mode=path_mode)
    first = [0.0, 0.25, 0.73, 1.0]
    later = [0.1, 0.5, 0.9]

    out = upsample_module.upsample_between_slices(
        xJ, J, W, pseudotimes=first, return_velocities=True, **kwargs
    )
    reference = upsample_module.upsample_between_slices(xJ, J, W, pseudotimes=later, **kwargs)

    path = tmp_path / "pair_velocities.npz"
    np.savez_compressed(path, **out["pair_velocities"])
    loaded = upsample_module.load_pair_velocities(path)
    n_solves = len(captured_solves)

    again = upsample_module.evaluate_pseudotimes(
        xJ, J, W, pair_velocities=loaded, pseudotimes=first, path_mode=path_mode
    )
    new = upsample_module.evaluate_pseudotimes(
        xJ, J, W, pair_velocities=loaded, pseudotimes=later, path_mode=path_mode
    )

    assert len(captured_solves) == n_solves  # no registration was run
    for got, want in ((again, out["pseudotime_samples"]), (new, reference["pseudotime_samples"])):
        assert got["path_mode"] == want["path_mode"] == path_mode
        for key in ("pair_indices", "pseudotimes", "z", "J"):
            assert np.array_equal(got[key], want[key]), key


def test_saved_velocities_are_plain_arrays_with_conventions(captured_solves, tmp_path):
    xJ, J, present = _three_slice_stack()
    out = upsample_module.upsample_between_slices(
        xJ, J, present_mask=present, mode="seg", config={"nt": NT}, return_velocities=True
    )
    path = tmp_path / "v.npz"
    np.savez_compressed(path, **out["pair_velocities"])
    with np.load(path, allow_pickle=False) as data:
        assert all(data[key].dtype != object for key in data.files)
        assert "I0 -> I1" in str(data["conventions"])
        assert str(data["mode"]) == "seg"


def test_evaluate_pseudotimes_rejects_mismatched_stack(captured_solves, tmp_path):
    xJ, J, present = _three_slice_stack()
    vel = upsample_module.upsample_between_slices(
        xJ, J, present_mask=present, mode="img", config={"nt": NT}, return_velocities=True
    )["pair_velocities"]
    W = np.ones(J.shape[1:], dtype=np.float32)

    shifted_z = [xJ[0] + 1.0, xJ[1], xJ[2]]
    with pytest.raises(ValueError, match="z axis"):
        upsample_module.evaluate_pseudotimes(shifted_z, J, W, pair_velocities=vel, pseudotimes=[0.5])
    with pytest.raises(ValueError, match="image grid"):
        upsample_module.evaluate_pseudotimes(
            [xJ[0], xJ[1][:-1], xJ[2]], J[:, :, :-1], W[:, :-1], pair_velocities=vel, pseudotimes=[0.5]
        )
    with pytest.raises(ValueError, match="slices"):
        upsample_module.evaluate_pseudotimes(
            [xJ[0][:3], xJ[1], xJ[2]], J[:, :3], W[:3], pair_velocities=vel, pseudotimes=[0.5]
        )

    np.savez(tmp_path / "other.npz", v=np.zeros(3))
    with pytest.raises(ValueError, match="format"):
        upsample_module.load_pair_velocities(tmp_path / "other.npz")


def test_return_velocities_with_no_pairs():
    xJ, J, _ = _three_slice_stack()
    present = np.array([True, False, False, False, False, False])
    out = upsample_module.upsample_between_slices(
        xJ, J, present_mask=present, mode="seg", config={"nt": NT}, return_velocities=True
    )
    vel = out["pair_velocities"]
    assert vel["pairs"].shape == (0, 2) and vel["v_forward"].size == 0
    samples = upsample_module.evaluate_pseudotimes(xJ, J, pair_velocities=vel, pseudotimes=[0.5])
    assert samples["J"].shape == (2, 0, SIZE, SIZE)


# -------------------------------------------------------------------------
# Spatial weights: each weight belongs to its image and shares its frame
# -------------------------------------------------------------------------


def test_each_solve_weights_its_target(monkeypatch):
    """EM-LDDMM weights the target: I0 -> I1 uses W[i1], its internal reverse W[i0]."""
    seen = []

    def recording_multiscale(**kwargs):
        seen.append((float(kwargs["I"].mean()), float(kwargs["J"].mean()), float(kwargs["W0"].mean())))
        return _stub_multiscale(**kwargs)

    _use_backend(monkeypatch, recording_multiscale)
    xJ, J, _ = _pair_stack()  # mean(J[:, 0]) ~ 0.5 < mean(J[:, 4]) ~ 1.5
    W = np.zeros(J.shape[1:], dtype=np.float32)
    W[0], W[4] = 2.0, 7.0  # distinct per-slice constants still matter
    config = upsample_module._normalize_pair_registration_config({"nt": NT})
    upsample_module._register_pair(0, 4, xJ, J, W, config, mode="img")

    i0, i1 = float(J[:, 0].mean()), float(J[:, 4].mean())
    # Backend solves: (fwd, its reverse), then (bwd, its reverse).
    expected = [(i0, i1, 7.0), (i1, i0, 2.0), (i1, i0, 2.0), (i0, i1, 7.0)]
    assert np.allclose(seen, expected, atol=1e-5)


def _translation(n=24, c=(2.0, -3.0), nt=8):
    axis = torch.arange(n, dtype=torch.float64)
    xv = (axis, axis.clone())
    c = torch.tensor(c, dtype=torch.float64)
    return xv, c[None, :, None, None].expand(nt, 2, n, n).clone()


def test_weights_are_pulled_back_with_their_image(monkeypatch):
    """W o phi_1^{-1} for a whole-pixel translation, with zero weight from outside."""
    _use_backend(monkeypatch)
    xv, v = _translation()
    n = xv[0].numel()
    yy, xx = np.mgrid[:n, :n]
    W = (1.0 + np.sin(yy / 3.0) * np.cos(xx / 4.0)) * ((yy > 4) & (xx < 18))
    image = np.random.default_rng(7).random((1, n, n))

    def warp(weight):
        return upsample_module._warp_at_times(
            {"v_symmetric": v, "xv": xv}, image, [a.numpy() for a in xv], [0.5, 1.0],
            device=CPU, weight=weight,
        )

    warped, det, Ws = warp(W)
    _, _, ones = warp(np.ones((n, n)))
    assert np.allclose(det, 1.0, atol=1e-12)  # weights carry no Jacobian of their own
    assert Ws.min() >= 0.0
    # s = 1: phi^{-1}(x) = x - (2, -3), so both become W(y - 2, x + 3).
    assert np.allclose(Ws[1, 2:, :-3], W[:-2, 3:], atol=1e-12)
    assert np.allclose(warped[1, 0, 2:, :-3], image[0, :-2, 3:], atol=1e-12)
    # Points sampled from outside the grid get zero weight; the image is border-clamped.
    assert np.allclose(ones[1, :2], 0.0, atol=1e-12) and np.allclose(ones[1, :, -3:], 0.0, atol=1e-12)
    assert np.allclose(ones[1, 2:, :-3], 1.0, atol=1e-12)
    assert np.allclose(warped[1, 0, 0, :-3], image[0, 0, 3:], atol=1e-12)
    # s = 0.5 is a half-pixel shift in x: bilinear weights stay in [0, 1].
    assert ones[0].min() >= 0.0 and ones[0].max() <= 1.0 + 1e-12 and 0.1 < ones[0, 5, -2] < 0.9


@pytest.mark.parametrize("path_mode", ["two-sided", "single-forward"])
def test_pseudotime_weights_use_the_maps_of_their_images(monkeypatch, path_mode):
    """With W = I (positive, single channel), warped W equals warped I where in bounds."""
    _use_backend(monkeypatch)
    xv, v = _translation()
    n = xv[0].numel()
    rng = np.random.default_rng(8)
    I0, I1 = 0.5 + rng.random((1, n, n)), 0.5 + rng.random((1, n, n))
    v_bwd = {"v_symmetric": -0.5 * v, "xv": xv}
    I0s, I1s, J0s, J1s, W0s, W1s = upsample_module._evaluate_pair_pseudotimes(
        {"v_symmetric": v, "xv": xv}, v_bwd, I0, I1, [a.numpy() for a in xv],
        [0.25, 0.5, 0.75], {}, path_mode, W0=I0[0], W1=I1[0],
    )
    interior = (slice(3, -3), slice(3, -3))  # every map here moves < 3 pixels
    assert np.allclose(W0s[(slice(None), *interior)], I0s[(slice(None), 0, *interior)], atol=1e-12)
    assert np.allclose(W1s[(slice(None), *interior)], I1s[(slice(None), 0, *interior)], atol=1e-12)
    # Unwarped weights would sit in the section frames, not the intermediate one.
    assert not np.allclose(W1s[(slice(None), *interior)], I1[(0, *interior)], atol=1e-3)
    # The blend uses these warped weights and the Jacobians exactly once.
    blended = upsample_module._blend_pair("img", 0.5, I0s[1], I1s[1], J0s[1], J1s[1], W0s[1], W1s[1])
    num = 0.5 * W0s[1] * J0s[1] * I0s[1] + 0.5 * W1s[1] * J1s[1] * I1s[1]
    den = 0.5 * W0s[1] * J0s[1] + 0.5 * W1s[1] * J1s[1] + 1e-8
    assert np.allclose(blended, num / den)


def test_lattice_weights_follow_their_image_frames(captured_solves):
    """Legacy z-lattice blend: W0t / W1t use the same maps and frame order as I0t / I1t."""
    xJ, J, _ = _pair_stack()
    W = np.zeros(J.shape[1:], dtype=np.float32)
    W[0], W[4] = J[0, 0], J[0, 4]  # weights equal to the images
    config = upsample_module._normalize_pair_registration_config({"nt": NT})
    _, _, I0t, I1t, _, _, W0t, W1t, _ = upsample_module._register_pair(
        0, 4, xJ, J, W, config, mode="img"
    )
    out_fwd, out_bwd = captured_solves

    def in_bounds(phi):
        phi = np.asarray(phi)
        return (phi >= 0).all(axis=0) & (phi <= SIZE - 1).all(axis=0)

    for t in range(NT):
        fwd = in_bounds(out_fwd["phi_I"][t + 1])
        bwd = in_bounds(out_bwd["phi_I"][NT - t])
        assert fwd.any() and not fwd.all()  # the stub moves points across the boundary
        assert np.allclose(W0t[t][fwd], I0t[t][0][fwd], atol=1e-5)
        assert np.allclose(W1t[t][bwd], I1t[t][0][bwd], atol=1e-5)

    *_, W0t_seg, W1t_seg, _ = upsample_module._register_pair(0, 4, xJ, J, W, config, mode="seg")
    assert W0t_seg is None and W1t_seg is None  # the seg blend uses no weights


@pytest.mark.parametrize("path_mode", ["two-sided", "single-forward"])
def test_uniform_weights_match_unweighted_blend_in_bounds(captured_solves, path_mode):
    xJ, J, present = _pair_stack()
    kwargs = dict(present_mask=present, mode="img", config={"nt": NT}, path_mode=path_mode)
    out = upsample_module.upsample_between_slices(xJ, J, pseudotimes=[0.3, 0.6], **kwargs)
    ones = upsample_module.upsample_between_slices(
        xJ, J, np.ones(J.shape[1:], dtype=np.float32), pseudotimes=[0.3, 0.6], **kwargs
    )
    # Uniform weights are zero only where a point comes from outside the grid,
    # so they leave the blend unchanged wherever both warps stay in bounds.
    a, b = out["pseudotime_samples"]["J"], ones["pseudotime_samples"]["J"]
    assert a.shape == b.shape and np.all(np.isfinite(b))
    assert np.allclose(a[..., 4:-4, 4:-4], b[..., 4:-4, 4:-4], atol=1e-5)


# -------------------------------------------------------------------------
# Input identity of stored velocities
# -------------------------------------------------------------------------


def _weighted_archive(tmp_path, W=None):
    xJ, J, present = _three_slice_stack()
    out = upsample_module.upsample_between_slices(
        xJ, J, W, present_mask=present, mode="img", config={"nt": NT}, return_velocities=True
    )
    path = tmp_path / "pair_velocities.npz"
    np.savez_compressed(path, **out["pair_velocities"])
    return xJ, J, upsample_module.load_pair_velocities(path)


def _spatial_weights(shape):
    yy, xx = np.mgrid[: shape[1], : shape[2]]
    return np.broadcast_to(0.2 + (yy + xx) / (2.0 * SIZE), shape).astype(np.float32).copy()


def test_archive_records_input_fingerprints(captured_solves, tmp_path):
    xJ, J, present = _three_slice_stack()
    W = _spatial_weights(J.shape[1:])
    _, _, vel = _weighted_archive(tmp_path, W)

    assert str(vel["format"]) == "wsi_pipeline.pair_velocities/2"
    assert vel["input_slices"].tolist() == [0, 2, 5]
    assert vel["J_sha256"].dtype.kind == vel["W_sha256"].dtype.kind == "U"
    assert vel["J_sha256"][1] == upsample_module._slice_sha256(J[:, 2])
    assert vel["W_sha256"][2] == upsample_module._slice_sha256(W[5])
    assert "SHA-256" in str(vel["conventions"])


def test_slice_digest_is_canonical_float32_with_shape():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    digest = upsample_module._slice_sha256(a)
    assert digest == upsample_module._slice_sha256(a.astype(np.float64))  # the solver sees float32
    assert digest == upsample_module._slice_sha256(np.asfortranarray(a))
    assert digest != upsample_module._slice_sha256(a.reshape(4, 3))  # same bytes, other shape
    assert digest != upsample_module._slice_sha256(a.reshape(1, 3, 4))


def test_evaluate_accepts_identical_inputs_and_new_pseudotimes(captured_solves, tmp_path):
    xJ, J, present = _three_slice_stack()
    W = _spatial_weights(J.shape[1:])
    _, _, vel = _weighted_archive(tmp_path, W)
    for path_mode in ("two-sided", "single-forward"):
        reference = upsample_module.upsample_between_slices(
            xJ, J, W, present_mask=present, mode="img", config={"nt": NT},
            pseudotimes=[0.2, 0.65], path_mode=path_mode,
        )["pseudotime_samples"]
        got = upsample_module.evaluate_pseudotimes(
            xJ, J.astype(np.float64), W.copy(), pair_velocities=vel,
            pseudotimes=[0.2, 0.65], path_mode=path_mode,
        )
        assert np.array_equal(got["J"], reference["J"])

    # Slices outside every pair never reach a solve, so they are not part of the identity.
    J_gap = J.copy()
    J_gap[:, 3] = 9.0
    upsample_module.evaluate_pseudotimes(xJ, J_gap, W, pair_velocities=vel, pseudotimes=[0.5])


def test_evaluate_rejects_modified_inputs(captured_solves, tmp_path):
    xJ, J, present = _three_slice_stack()
    W = _spatial_weights(J.shape[1:])
    _, _, vel = _weighted_archive(tmp_path, W)

    J_bad = J.copy()
    J_bad[1, 2, 5, 5] += 1e-3  # same shape, axes, and dtype
    with pytest.raises(ValueError, match=r"J differs .* slices \[2\]"):
        upsample_module.evaluate_pseudotimes(xJ, J_bad, W, pair_velocities=vel, pseudotimes=[0.5])

    W_bad = W.copy()
    W_bad[5, 0, 0] = 0.0
    with pytest.raises(ValueError, match=r"W differs .* slices \[5\]"):
        upsample_module.evaluate_pseudotimes(xJ, J, W_bad, pair_velocities=vel, pseudotimes=[0.5])

    with pytest.raises(ValueError, match="registered with weights"):
        upsample_module.evaluate_pseudotimes(xJ, J, None, pair_velocities=vel, pseudotimes=[0.5])


def test_evaluate_rejects_weights_for_an_unweighted_archive(captured_solves, tmp_path):
    xJ, J, vel = _weighted_archive(tmp_path)
    assert vel["W_sha256"].size == 0
    upsample_module.evaluate_pseudotimes(xJ, J, None, pair_velocities=vel, pseudotimes=[0.5])
    with pytest.raises(ValueError, match="without weights"):
        upsample_module.evaluate_pseudotimes(
            xJ, J, np.ones(J.shape[1:], dtype=np.float32), pair_velocities=vel, pseudotimes=[0.5]
        )


def test_version_1_archives_are_rejected_as_unverifiable(captured_solves, tmp_path):
    xJ, J, vel = _weighted_archive(tmp_path)
    legacy = {k: v for k, v in vel.items() if k not in {"input_slices", "J_sha256", "W_sha256"}}
    legacy["format"] = np.array("wsi_pipeline.pair_velocities/1")
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, **legacy)

    with pytest.raises(ValueError, match="cannot be verified"):
        upsample_module.load_pair_velocities(path)
    with pytest.raises(ValueError, match="cannot be verified"):
        upsample_module.evaluate_pseudotimes(xJ, J, pair_velocities=legacy, pseudotimes=[0.5])
