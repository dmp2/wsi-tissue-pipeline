from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import wsi_pipeline.registration.symmetric as symmetric_module
import wsi_pipeline.registration.upsample as upsample_module


def _axes(z_values, size_y=2, size_x=2):
    return [
        np.asarray(z_values, dtype=np.float32),
        np.arange(size_y, dtype=np.float32),
        np.arange(size_x, dtype=np.float32),
    ]


def _fake_segmentation_helper(xI, I, xJ, J, W0=None, **config):
    nt = int(config["nt"])
    fill_value = 0.0 if float(np.mean(I)) <= float(np.mean(J)) else 10.0
    It = np.full((nt, I.shape[0], I.shape[1], I.shape[2]), fill_value, dtype=np.float32)
    det = np.ones((nt, I.shape[1], I.shape[2]), dtype=np.float32)
    return {
        "It": It,
        "det_jac_phi_I": det,
        "det_jac_phi_J": det,
    }


def _fake_img_helper(xI, I, xJ, J, W0=None, **config):
    nt = int(config["nt"])
    image_value = 10.0 if float(np.mean(I)) <= float(np.mean(J)) else 20.0
    jac_value = 1.0 if float(np.mean(I)) <= float(np.mean(J)) else 3.0
    It = np.full((nt, I.shape[0], I.shape[1], I.shape[2]), image_value, dtype=np.float32)
    det = np.full((nt, I.shape[1], I.shape[2]), jac_value, dtype=np.float32)
    return {
        "It": It,
        "det_jac_phi_I": det,
        "det_jac_phi_J": det,
    }


def test_segmentation_gap_fill_on_global_grid(monkeypatch):
    monkeypatch.setattr(
        upsample_module,
        "emlddmm_multiscale_symmetric_N",
        _fake_segmentation_helper,
    )

    xJ = _axes(np.arange(11, dtype=np.float32))
    J = np.zeros((1, 11, 2, 2), dtype=np.float32)
    J[:, 0] = 5.0
    J[:, 10] = 9.0

    out = upsample_module.upsample_between_slices(
        xJ,
        J,
        slice_spacing=1.0,
        n_resample=10,
        mode="seg",
        tissue_idx=0,
        config={"nt": 10},
        parallel=False,
    )

    assert out["pairs"] == [(0, 10)]
    assert out["J_filled"].shape == J.shape
    assert np.all(out["J_filled"][:, 0] == 5.0)
    assert np.all(out["J_filled"][:, 10] == 9.0)
    assert np.array_equal(out["J_filled"][0, 1:10, 0, 0], np.arange(1, 10, dtype=np.float32))


def test_descending_z_axis_is_orientation_safe(monkeypatch):
    monkeypatch.setattr(
        upsample_module,
        "emlddmm_multiscale_symmetric_N",
        _fake_segmentation_helper,
    )

    xJ = _axes(np.arange(10, -1, -1, dtype=np.float32))
    J = np.zeros((1, 11, 2, 2), dtype=np.float32)
    J[:, 0] = 5.0
    J[:, 10] = 9.0

    out = upsample_module.upsample_between_slices(
        xJ,
        J,
        mode="seg",
        config={"nt": 10},
        parallel=False,
    )

    assert np.array_equal(out["J_filled"][0, 1:10, 0, 0], np.arange(1, 10, dtype=np.float32))


def test_nt_must_cover_largest_global_gap(monkeypatch):
    monkeypatch.setattr(
        upsample_module,
        "emlddmm_multiscale_symmetric_N",
        _fake_segmentation_helper,
    )

    xJ = _axes(np.arange(11, dtype=np.float32))
    J = np.zeros((1, 11, 2, 2), dtype=np.float32)
    J[:, 0] = 1.0
    J[:, 10] = 2.0

    with pytest.raises(ValueError, match="largest pair gap"):
        upsample_module.upsample_between_slices(
            xJ,
            J,
            mode="seg",
            config={"nt": 9},
            parallel=False,
        )


def test_planes_outside_observed_range_remain_unchanged(monkeypatch):
    monkeypatch.setattr(
        upsample_module,
        "emlddmm_multiscale_symmetric_N",
        _fake_segmentation_helper,
    )

    xJ = _axes(np.arange(11, dtype=np.float32))
    J = np.zeros((1, 11, 2, 2), dtype=np.float32)
    J[:, 2] = 3.0
    J[:, 8] = 7.0

    out = upsample_module.upsample_between_slices(
        xJ,
        J,
        mode="seg",
        config={"nt": 10},
        parallel=False,
    )

    assert np.all(out["J_filled"][:, :2] == 0.0)
    assert np.all(out["J_filled"][:, 9:] == 0.0)
    assert np.all(out["J_filled"][:, 2] == 3.0)
    assert np.all(out["J_filled"][:, 8] == 7.0)


def test_img_mode_uses_jacobian_weighted_blend(monkeypatch):
    monkeypatch.setattr(
        upsample_module,
        "emlddmm_multiscale_symmetric_N",
        _fake_img_helper,
    )

    xJ = _axes(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    J = np.zeros((1, 3, 2, 2), dtype=np.float32)
    J[:, 0] = 1.0
    J[:, 2] = 2.0
    W = np.ones((3, 2, 2), dtype=np.float32)

    out = upsample_module.upsample_between_slices(
        xJ,
        J,
        W=W,
        mode="img",
        config={"nt": 2},
        parallel=False,
    )

    assert np.allclose(out["J_filled"][:, 1], 17.5)


def test_img_mode_uses_w_for_slice_detection(monkeypatch):
    monkeypatch.setattr(
        upsample_module,
        "emlddmm_multiscale_symmetric_N",
        _fake_img_helper,
    )

    xJ = _axes(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    J = np.zeros((1, 3, 2, 2), dtype=np.float32)
    J[:, 2] = 2.0
    W = np.zeros((3, 2, 2), dtype=np.float32)
    W[0] = 1.0
    W[2] = 1.0

    out = upsample_module.upsample_between_slices(
        xJ,
        J,
        W=W,
        mode="img",
        config={"nt": 2},
        parallel=False,
    )

    assert np.array_equal(out["slices_with_data"], np.array([0, 2]))
    assert out["pairs"] == [(0, 2)]


def test_img_mode_requires_w_or_present_mask(monkeypatch):
    monkeypatch.setattr(
        upsample_module,
        "emlddmm_multiscale_symmetric_N",
        _fake_img_helper,
    )

    xJ = _axes(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    J = np.zeros((1, 3, 2, 2), dtype=np.float32)
    J[:, 0] = 1.0
    J[:, 2] = 2.0

    with pytest.raises(
        ValueError,
        match="mode='img' requires W or present_mask for stable slice detection",
    ):
        upsample_module.upsample_between_slices(
            xJ,
            J,
            mode="img",
            config={"nt": 2},
            parallel=False,
        )


def test_present_mask_overrides_image_content(monkeypatch):
    monkeypatch.setattr(
        upsample_module,
        "emlddmm_multiscale_symmetric_N",
        _fake_img_helper,
    )

    xJ = _axes(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    J = np.zeros((1, 3, 2, 2), dtype=np.float32)
    present_mask = np.array([True, False, True], dtype=bool)

    out = upsample_module.upsample_between_slices(
        xJ,
        J,
        present_mask=present_mask,
        mode="img",
        config={"nt": 2},
        parallel=False,
    )

    assert np.array_equal(out["slices_with_data"], np.array([0, 2]))
    assert out["pairs"] == [(0, 2)]


def test_symmetric_helper_returns_both_jacobian_keys(monkeypatch):
    nt = 4

    def fake_multiscale(**kwargs):
        return {
            "v": torch.zeros((nt, 2, 2, 2), dtype=torch.float32),
            "xv": [
                torch.arange(2, dtype=torch.float32),
                torch.arange(2, dtype=torch.float32),
            ],
        }

    def fake_integrate_inverse_flow(xv, v, *, interp2d=None, grid_sample_kwargs=None):
        return torch.zeros((nt + 1, 2, 2, 2), dtype=torch.float32)

    def fake_warp_time_series(x, image, phis, *, interp2d=None, grid_sample_kwargs=None):
        base = torch.arange(phis.shape[0], dtype=torch.float32)[:, None, None, None]
        channels = torch.as_tensor(image).shape[0]
        return base.expand(phis.shape[0], channels, 2, 2)

    def fake_det(phi, spacing=None):
        base = torch.arange(phi.shape[0], dtype=torch.float32)[:, None, None]
        return base.expand(phi.shape[0], 2, 2)

    monkeypatch.setattr(symmetric_module.emlddmm, "emlddmm_multiscale", fake_multiscale)
    monkeypatch.setattr(symmetric_module, "_integrate_inverse_flow", fake_integrate_inverse_flow)
    monkeypatch.setattr(symmetric_module, "_warp_time_series", fake_warp_time_series)
    monkeypatch.setattr(symmetric_module, "_calculate_determinant_of_jacobian", fake_det)

    out = symmetric_module.emlddmm_multiscale_symmetric_N(
        xI=[np.arange(2, dtype=np.float32), np.arange(2, dtype=np.float32)],
        I=np.ones((1, 2, 2), dtype=np.float32),
        xJ=[np.arange(2, dtype=np.float32), np.arange(2, dtype=np.float32)],
        J=np.ones((1, 2, 2), dtype=np.float32) * 2.0,
        nt=nt,
    )

    assert "det_jac_phi_I" in out
    assert "det_jac_phi_J" in out
    assert "ItAll" in out
    assert "JtAll" in out
    assert out["It"].shape[0] == nt
    assert out["ItAll"].shape[0] == nt + 1


def test_symmetric_helper_uses_domain_specific_axes_and_spacing(monkeypatch):
    nt = 4
    warp_axes = []
    det_spacings = []

    def fake_multiscale(**kwargs):
        return {
            "v": torch.zeros((nt, 2, 2, 2), dtype=torch.float32),
            "xv": [
                torch.arange(2, dtype=torch.float32),
                torch.arange(2, dtype=torch.float32),
            ],
        }

    def fake_integrate_inverse_flow(xv, v, *, interp2d=None, grid_sample_kwargs=None):
        return torch.zeros((nt + 1, 2, 2, 2), dtype=torch.float32)

    def fake_warp_time_series(x, image, phis, *, interp2d=None, grid_sample_kwargs=None):
        warp_axes.append([np.asarray(axis) for axis in x])
        channels = torch.as_tensor(image).shape[0]
        return torch.zeros((phis.shape[0], channels, 2, 2), dtype=torch.float32)

    def fake_det(phi, spacing=None):
        det_spacings.append(tuple(float(s) for s in spacing))
        return torch.ones((phi.shape[0], 2, 2), dtype=torch.float32)

    monkeypatch.setattr(symmetric_module.emlddmm, "emlddmm_multiscale", fake_multiscale)
    monkeypatch.setattr(symmetric_module, "_integrate_inverse_flow", fake_integrate_inverse_flow)
    monkeypatch.setattr(symmetric_module, "_warp_time_series", fake_warp_time_series)
    monkeypatch.setattr(symmetric_module, "_calculate_determinant_of_jacobian", fake_det)

    xI = [
        np.array([0.0, 2.0], dtype=np.float32),
        np.array([0.0, 4.0], dtype=np.float32),
    ]
    xJ = [
        np.array([10.0, 14.0], dtype=np.float32),
        np.array([5.0, 8.0], dtype=np.float32),
    ]

    symmetric_module.emlddmm_multiscale_symmetric_N(
        xI=xI,
        I=np.ones((1, 2, 2), dtype=np.float32),
        xJ=xJ,
        J=np.ones((1, 2, 2), dtype=np.float32) * 2.0,
        nt=nt,
    )

    assert len(warp_axes) == 2
    assert np.array_equal(warp_axes[0][0], xI[0])
    assert np.array_equal(warp_axes[0][1], xI[1])
    assert np.array_equal(warp_axes[1][0], xJ[0])
    assert np.array_equal(warp_axes[1][1], xJ[1])
    assert det_spacings == [(2.0, 4.0), (4.0, 3.0)]
