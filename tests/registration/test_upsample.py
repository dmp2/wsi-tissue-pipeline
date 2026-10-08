from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import wsi_pipeline.registration.symmetric as symmetric_module
import wsi_pipeline.registration.upsample as upsample_module


# -------------------------------------------------------------------------
# Real-registration / visual-QC integration tests
# -------------------------------------------------------------------------

from pathlib import Path


def _make_binary_square_circle(
    size: int = 128,
    square_half_width: int = 26,
    circle_radius: int = 30,
):
    """Create centered binary square and circle images.

    Returns
    -------
    square, circle : np.ndarray
        Arrays with shape (1, H, W), matching the registration helper's
        channel-first 2D image convention.
    """
    if size < 32:
        raise ValueError("size must be large enough for a meaningful registration test")

    cy = cx = (size - 1) / 2.0

    yy, xx = np.mgrid[:size, :size]

    square_mask = (
        (np.abs(xx - cx) <= square_half_width)
        & (np.abs(yy - cy) <= square_half_width)
    )

    circle_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= circle_radius**2

    square = square_mask.astype(np.float32)[None, ...]
    circle = circle_mask.astype(np.float32)[None, ...]

    return square, circle


def _artifact_dir():
    """Return a deterministic repo-local directory for visual test artifacts."""
    path = Path("tests") / "artifacts" / "registration"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_flow_montage(
    frames,
    output_path,
    *,
    title,
    threshold=False,
):
    """Save all temporal samples as a horizontal montage.

    Parameters
    ----------
    frames : array-like
        Expected shape is either (T, C, H, W) or (T, H, W).
    output_path : pathlib.Path
        Destination PNG.
    title : str
        Figure title.
    threshold : bool
        If True, show binary-thresholded frames. Otherwise show the actual
        floating-point registration output.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = np.asarray(frames)

    if frames.ndim == 4:
        # T, C, H, W -> use first/only segmentation channel.
        frames = frames[:, 0]

    if frames.ndim != 3:
        raise AssertionError(
            f"Expected frames with shape (T,H,W) or (T,C,H,W), got {frames.shape}"
        )

    n_frames = frames.shape[0]

    fig, axes = plt.subplots(
        1,
        n_frames,
        figsize=(2.2 * n_frames, 2.8),
        squeeze=False,
    )
    axes = axes[0]

    for t, ax in enumerate(axes):
        frame = frames[t]

        if threshold:
            frame = frame >= 0.5

        ax.imshow(
            frame,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        if n_frames == 1:
            tau = 0.0
        else:
            tau = t / (n_frames - 1)

        ax.set_title(f"t={tau:.2f}")
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _centroid(image):
    """Intensity-weighted centroid in pixel coordinates."""
    image = np.asarray(image, dtype=np.float64)

    if image.ndim == 3:
        image = image[0]

    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]

    mass = image.sum()

    if mass <= 0:
        return np.array([np.nan, np.nan], dtype=np.float64)

    return np.array(
        [
            (yy * image).sum() / mass,
            (xx * image).sum() / mass,
        ],
        dtype=np.float64,
    )


def test_real_2d_square_to_circle_registration_writes_montage():
    """Exercise the real symmetric registration path on two binary 2D images.

    This deliberately does NOT monkeypatch:
      - emlddmm_multiscale
      - inverse-flow integration
      - image warping
      - Jacobian calculation

    The primary purpose is to establish that the real registration backend
    produces an auditable temporal deformation from square -> circle.
    """
    size = 128

    square, circle = _make_binary_square_circle(size=size)

    axes_2d = (
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32),
    )

    # nt is currently required by the registration machinery. This value
    # provides a useful number of temporal samples for visual inspection.
    #
    # IMPORTANT:
    # This is NOT an endorsement of the current global-gap/nt coupling in
    # upsample_between_slices.
    nt = 8

    config = _fast_real_registration_config(nt=8)
    from scipy.signal import convolve2d

    kernel = np.ones((8, 8), dtype=np.float32) / 64.0

    square_smooth = np.stack(
        [convolve2d(ch, kernel, mode="same") for ch in square],
        axis=0,
    ).astype(np.float32)

    circle_smooth = np.stack(
        [convolve2d(ch, kernel, mode="same") for ch in circle],
        axis=0,
    ).astype(np.float32)
    out = symmetric_module.emlddmm_multiscale_symmetric_N(
        xI=axes_2d,
        I=square_smooth,
        xJ=axes_2d,
        J=circle_smooth,
        **config,
    )
    
    # The existing unit tests establish that the symmetric helper exposes
    # full endpoint-inclusive time series as ItAll/JtAll.
    assert "ItAll" in out
    assert "JtAll" in out

    It_all = np.asarray(out["ItAll"])
    Jt_all = np.asarray(out["JtAll"])
    v = np.asarray(out["v_symmetric"])
    phi_I = np.asarray(out["phi_I"])

    disp = phi_I - phi_I[0:1]

    v_abs_max = float(np.max(np.abs(v)))
    v_rms = float(np.sqrt(np.mean(v**2)))
    disp_abs_max = float(np.max(np.abs(disp[-1])))
    disp_rms = float(np.sqrt(np.mean(disp[-1] ** 2)))

    initial = It_all[0]
    final = It_all[-1]

    mse_initial = float(np.mean((initial - circle) ** 2))
    mse_final = float(np.mean((final - circle) ** 2))

    print("v abs max:", v_abs_max)
    print("v RMS:", v_rms)
    print("final displacement abs max:", disp_abs_max)
    print("final displacement RMS:", disp_rms)
    print("MSE initial -> circle:", mse_initial)
    print("MSE final   -> circle:", mse_final)

    assert v_abs_max > 1e-5, "Registration produced essentially zero velocity"
    assert disp_abs_max > 1e-3, "Final transform remained essentially identity"
    assert mse_final < mse_initial, (
        f"Registration did not move square toward circle: "
        f"{mse_initial=} {mse_final=}"
    )

    assert It_all.ndim == 4
    assert Jt_all.ndim == 4

    assert It_all.shape[1:] == square.shape
    assert Jt_all.shape[1:] == circle.shape

    # nt velocity intervals should yield nt + 1 temporal states.
    assert It_all.shape[0] == nt + 1
    assert Jt_all.shape[0] == nt + 1

    # Basic sanity: registration must produce finite images.
    assert np.all(np.isfinite(It_all))
    assert np.all(np.isfinite(Jt_all))

    # Nothing should disappear completely.
    assert np.all(It_all.sum(axis=(1, 2, 3)) > 0)
    assert np.all(Jt_all.sum(axis=(1, 2, 3)) > 0)

    # Because source and target are centered, gross translation is not expected.
    # Keep this deliberately loose: this is a diagnostic integration test,
    # not a claim about exact LDDMM trajectories.
    center = np.array([(size - 1) / 2.0, (size - 1) / 2.0])

    for frame in It_all:
        c = _centroid(frame)
        assert np.linalg.norm(c - center) < 8.0

    artifact_dir = _artifact_dir()

    _save_flow_montage(
        It_all,
        artifact_dir / "square_to_circle_2d_ItAll_soft.png",
        title="Real 2D registration: square → circle, source-side flow",
        threshold=False,
    )

    _save_flow_montage(
        It_all,
        artifact_dir / "square_to_circle_2d_ItAll_binary.png",
        title="Real 2D registration: square → circle, thresholded source-side flow",
        threshold=True,
    )

    _save_flow_montage(
        Jt_all,
        artifact_dir / "square_to_circle_2d_JtAll_soft.png",
        title="Real 2D registration: square → circle, target-side flow",
        threshold=False,
    )

def test_2d_downsampling_schedule_promotion_preserves_scale_dimension():
    assert symmetric_module._prepend_synthetic_axis_to_2d_schedule(
        [1, 1], 1
    ) == [[1, 1, 1]]

    assert symmetric_module._prepend_synthetic_axis_to_2d_schedule(
        [[1, 1]], 1
    ) == [[1, 1, 1]]

    assert symmetric_module._prepend_synthetic_axis_to_2d_schedule(
        [[2, 2], [1, 1]], 1
    ) == [[1, 2, 2], [1, 1, 1]]



def test_synthetic_z_axis_expands_with_dv():
    axis_fine = symmetric_module._make_synthetic_z_axis(
        synthetic_spacing=1.0,
        dv_z=1.0,
        device="cpu",
        dtype=torch.float32,
        min_velocity_intervals=4,
    )

    axis_coarse = symmetric_module._make_synthetic_z_axis(
        synthetic_spacing=1.0,
        dv_z=2.0,
        device="cpu",
        dtype=torch.float32,
        min_velocity_intervals=4,
    )

    assert axis_fine.numel() >= 5
    assert axis_coarse.numel() > axis_fine.numel()

    assert torch.isclose(axis_fine.mean(), torch.tensor(0.0))
    assert torch.isclose(axis_coarse.mean(), torch.tensor(0.0))


def test_synthetic_z_extent_covers_requested_velocity_support():
    spacing = 1.0
    dv_z = 2.0
    minimum_intervals = 4

    axis = symmetric_module._make_synthetic_z_axis(
        synthetic_spacing=spacing,
        dv_z=dv_z,
        device="cpu",
        dtype=torch.float32,
        min_velocity_intervals=minimum_intervals,
    )

    extent = float(axis[-1] - axis[0])

    assert extent >= minimum_intervals * dv_z


def test_synthetic_extrusion_repeats_without_changing_image():
    I = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    J = I + 1
    W = torch.ones(4, 4)

    axes = (
        torch.arange(4, dtype=torch.float32),
        torch.arange(4, dtype=torch.float32),
    )

    (
        xI3,
        xJ3,
        I3,
        J3,
        W3,
        z,
    ) = symmetric_module._extrude_2d_pair_for_3d_backend(
        I,
        J,
        W,
        axes,
        axes,
        {"dv": 1.0},
    )

    assert I3.ndim == 4
    assert J3.ndim == 4

    for zi in range(len(z)):
        assert torch.equal(I3[:, zi], I)
        assert torch.equal(J3[:, zi], J)
        assert torch.equal(W3[zi], W)

    assert len(xI3) == 3
    assert len(xJ3) == 3


def _fast_real_registration_config(nt=8):
    return {
        "nt": nt,
        "n_iter": [100],
        "v_start": [0],
        "ev": [5e-1],
        "a": [2.0],
        "dv": [[1.0, 1.0]], # changed to 2d spec; old: "dv": [[0.25, 1.0, 1.0]], [[0.5, 1.0, 1.0]], [[1.0, 2.0, 2.0]], # works
        "sigmaR": [5e0],
        "n_draw": [0],
        "n_reduce_step": [1000],
        "device": "cuda:0",
        "dtype": "float32",
        "A": np.eye(4, dtype=np.float32),
        "update_A": False,
        "eA": [0.0],
        "update_matching_weights": False,
        "out_of_plane": False,
    }


def test_promote_2d_dv_uses_internal_synthetic_z_spacing():
    promoted = symmetric_module._promote_2d_dv_schedule(
        [[0.5, 2.0]],
        synthetic_spacing=1.0,
    )

    assert promoted == [[1.0, 0.5, 2.0]]

def test_promote_2d_dv_ignores_user_supplied_synthetic_z_value():
    promoted = symmetric_module._promote_2d_dv_schedule(
        [[99.0, 0.5, 2.0]],
        synthetic_spacing=1.0,
    )

    assert promoted == [[1.0, 0.5, 2.0]]


def test_pair_config_accepts_out_of_plane_false():
    _validate_out_of_plane_pair_config(
        {"out_of_plane": False}
    )


def test_pair_config_accepts_unset_out_of_plane():
    _validate_out_of_plane_pair_config({})


def test_pair_config_warns_for_out_of_plane_true():
    with pytest.warns(
        UserWarning,
        match="False is recommended",
    ):
        _validate_out_of_plane_pair_config(
            {"out_of_plane": True}
        )


def test_pair_config_rejects_invalid_out_of_plane():
    with pytest.raises(
        ValueError,
        match="out_of_plane must be",
    ):
        _validate_out_of_plane_pair_config(
            {"out_of_plane": "sometimes"}
        )

def test_pair_config_accepts_consistent_out_of_plane_sequence():
    _validate_out_of_plane_pair_config(
        {"out_of_plane": [False, False]}
    )


def test_pair_config_rejects_mixed_out_of_plane_sequence():
    with pytest.raises(
        ValueError,
        match="consistently true or consistently false",
    ):
        _validate_out_of_plane_pair_config(
            {"out_of_plane": [False, True]}
        )


def test_real_square_to_circle_between_3d_slices_writes_montage(monkeypatch):
    """Exercise real upsampling between binary 2D slices in a 3D stack.

    The data representation is:

        J.shape == (C, Z, Y, X)

        z = 0       : binary square
        z = 1..Z-2  : missing
        z = Z-1     : binary circle

    `upsample_between_slices` must extract the observed planes as 2D images,
    register them using the real symmetric-registration helper, and populate
    the missing global Z planes.
    """
    size = 128

    # Keep nine planes so that the resulting montage has useful temporal
    # resolution without becoming unwieldy.
    nz = 9

    square, circle = _make_binary_square_circle(size=size)

    J = np.zeros(
        (1, nz, size, size),
        dtype=np.float32,
    )

    J[:, 0] = square
    J[:, -1] = circle

    present_mask = np.zeros(nz, dtype=bool)
    present_mask[0] = True
    present_mask[-1] = True

    xJ = [
        np.arange(nz, dtype=np.float32),
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32),
    ]

    # Temporary compatibility with the CURRENT implementation:
    # the current upsampler requires nt to cover the largest global gap.
    #
    # That coupling is not being defended here and should be redesigned
    # separately. We satisfy it only so this test reaches the registration
    # and interpolation path we actually want to inspect.
    nt = nz - 1
    config = _fast_real_registration_config(nt=8)

    real_symmetric = symmetric_module.emlddmm_multiscale_symmetric_N
    captured = []

    def capturing_symmetric(*args, **kwargs):
        result = real_symmetric(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(
        upsample_module,
        "emlddmm_multiscale_symmetric_N",
        capturing_symmetric,
    )

    out = upsample_module.upsample_between_slices(
        xJ,
        J,
        present_mask=present_mask,
        mode="seg",
        config = config,
        parallel=False,
    )

    assert len(captured) == 2

    max_v = max(
        float(np.max(np.abs(np.asarray(pair_out["v_symmetric"]))))
        for pair_out in captured
    )

    assert max_v > 1e-5, (
        "Upsampling produced visually changing slices but registration "
        "learned zero velocity; result is only endpoint blending."
    )

    for i, pair_out in enumerate(captured):
        v = np.asarray(pair_out["v_symmetric"])
        phi = np.asarray(pair_out["phi_I"])

        disp = phi - phi[0:1]

        v_abs_max = float(np.max(np.abs(v)))
        v_rms = float(np.sqrt(np.mean(v**2)))
        disp_abs_max = float(np.max(np.abs(disp[-1])))

        print(
            f"symmetric call {i}: "
            f"v_abs_max={v_abs_max:.6g}, "
            f"v_rms={v_rms:.6g}, "
            f"disp_abs_max={disp_abs_max:.6g}"
        )

    assert out["pairs"] == [(0, nz - 1)]

    filled = np.asarray(out["J_filled"])

    assert filled.shape == J.shape
    assert np.all(np.isfinite(filled))

    # Observed planes must be preserved.
    assert np.array_equal(filled[:, 0], square)
    assert np.array_equal(filled[:, -1], circle)

    # Every plane between the observed slices should now contain tissue.
    for z in range(1, nz - 1):
        assert filled[:, z].sum() > 0

    # Gross sanity check: because square and circle are both centered,
    # the interpolated anatomy should remain approximately centered.
    center = np.array([(size - 1) / 2.0, (size - 1) / 2.0])

    for z in range(nz):
        c = _centroid(filled[:, z])
        assert np.linalg.norm(c - center) < 8.0

    artifact_dir = _artifact_dir()

    # Convert C,Z,Y,X -> Z,C,Y,X for the montage helper.
    filled_time = np.moveaxis(filled, 1, 0)

    _save_flow_montage(
        filled_time,
        artifact_dir / "square_to_circle_3d_slices_soft.png",
        title="Real upsampling: square slice → circle slice",
        threshold=False,
    )

    _save_flow_montage(
        filled_time,
        artifact_dir / "square_to_circle_3d_slices_binary.png",
        title="Real upsampling: square slice → circle slice, thresholded",
        threshold=True,
    )

def _fake_interp(x, image, phii, interp2d=False, **kwargs):
    image_t = torch.as_tensor(image)
    phii_t = torch.as_tensor(phii)
    return torch.zeros(
        (image_t.shape[0], *phii_t.shape[1:]),
        device=phii_t.device,
        dtype=image_t.dtype,
    )


def _fake_symmetric_backend(multiscale=None):
    return SimpleNamespace(
        emlddmm_multiscale=multiscale or (lambda **kwargs: None),
        interp=_fake_interp,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_warp_time_series_moves_axes_to_transform_device():
    seen_devices = []

    def checking_interp(x, image, phii, **kwargs):
        seen_devices.append(
            (
                tuple(axis.device.type for axis in x),
                image.device.type,
                phii.device.type,
            )
        )
        return torch.zeros(
            (image.shape[0], *phii.shape[1:]),
            device=phii.device,
            dtype=image.dtype,
        )

    backend = SimpleNamespace(interp=checking_interp)
    axes = (torch.arange(2), torch.arange(3))
    image = torch.ones((1, 2, 3))
    phis = torch.zeros((2, 2, 2, 3), device="cuda")

    result = symmetric_module._warp_time_series(axes, image, phis, emlddmm_module=backend)

    assert result.device.type == "cuda"
    assert seen_devices == [
        (("cuda", "cuda"), "cuda", "cuda"),
        (("cuda", "cuda"), "cuda", "cuda"),
    ]


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


def test_pair_registration_inputs_are_2d_and_identity_downsampling(monkeypatch):
    calls = []

    def fake_pair_helper(xI, I, xJ, J, W0=None, **config):
        calls.append(
            {
                "xI_type": type(xI),
                "xJ_type": type(xJ),
                "xI_lengths": [len(axis) for axis in xI],
                "I_shape": tuple(np.asarray(I).shape),
                "J_shape": tuple(np.asarray(J).shape),
                "W0_shape": None if W0 is None else tuple(np.asarray(W0).shape),
                "config": dict(config),
            }
        )
        assert len(xI) == 2
        assert len(xJ) == 2
        assert np.asarray(I).shape == (1, 3, 4)
        assert np.asarray(J).shape == (1, 3, 4)
        assert np.asarray(W0).shape == (3, 4)
        nt = int(config["nt"])
        It = np.ones((nt, 1, 3, 4), dtype=np.float32)
        det = np.ones((nt, 3, 4), dtype=np.float32)
        return {
            "It": It,
            "det_jac_phi_I": det,
            "det_jac_phi_J": det,
        }

    monkeypatch.setattr(
        upsample_module,
        "emlddmm_multiscale_symmetric_N",
        fake_pair_helper,
    )

    xJ = _axes(np.array([0.0, 1.0, 2.0], dtype=np.float32), size_y=3, size_x=4)
    J = np.zeros((1, 3, 3, 4), dtype=np.float32)
    J[:, 0] = 1.0
    J[:, 2] = 2.0
    W = np.ones((3, 3, 4), dtype=np.float32)
    present_mask = np.array([True, False, True], dtype=bool)

    upsample_module.upsample_between_slices(
        xJ,
        J,
        W=W,
        present_mask=present_mask,
        mode="seg",
        config={"nt": 2},
        parallel=False,
    )

    assert len(calls) == 2
    assert all(call["xI_type"] is tuple for call in calls)
    assert all(call["xI_lengths"] == [3, 4] for call in calls)
    assert all(call["I_shape"] == (1, 3, 4) for call in calls)
    assert all(call["W0_shape"] == (3, 4) for call in calls)
    assert all(call["config"]["downI"] == [[1, 1]] for call in calls)
    assert all(call["config"]["downJ"] == [[1, 1]] for call in calls)
    assert all(call["config"]["local_contrast"] == [None] for call in calls)
    assert all(call["config"]["up_vector"] == [None] for call in calls)
    assert all(call["config"]["n_draw"] == 0 for call in calls)
    assert all(call["config"]["dtype"] == "float32" for call in calls)
    assert all(np.allclose(call["config"]["A"], np.eye(4, dtype=np.float32)) for call in calls)
    assert all(call["config"]["A2d"] is None for call in calls)
    assert all(call["config"]["Amode"] == 0 for call in calls)
    assert all(call["config"]["eA"] == [0.0] for call in calls)
    assert all(call["config"]["eA2d"] == [0.0] for call in calls)
    assert all(call["config"]["slice_matching"] == [False] for call in calls)


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


def test_pair_registration_rejects_too_short_xy_axes():
    xJ = [
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ]
    J = np.zeros((1, 3, 1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="at least two points"):
        upsample_module.upsample_between_slices(
            xJ,
            J,
            mode="seg",
            config={"nt": 2},
            parallel=False,
        )


def test_pair_registration_rejects_3d_downsampling_config():
    xJ = _axes(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    J = np.zeros((1, 3, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match=r"downI.*2D"):
        upsample_module.upsample_between_slices(
            xJ,
            J,
            mode="seg",
            config={"nt": 2, "downI": [[1, 1, 1]], "downJ": [[1, 1]]},
            parallel=False,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("local_contrast", [[1, 16, 16]]),
        ("up_vector", [[0.0, 0.0, -1.0]]),
    ],
)
def test_pair_registration_rejects_3d_only_options(name, value):
    xJ = _axes(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    J = np.zeros((1, 3, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match=rf"{name}.*not supported"):
        upsample_module.upsample_between_slices(
            xJ,
            J,
            mode="seg",
            config={"nt": 2, name: value},
            parallel=False,
        )


@pytest.mark.parametrize(
    "config",
    [
        {"nt": 2, "eA": [1.0]},
        {"nt": 2, "eA2d": [1.0]},
        {"nt": 2, "Amode": 1},
        {"nt": 2, "slice_matching": [True]},
        {"nt": 2, "A": np.diag([1.0, 1.0, 1.0, 2.0]).astype(np.float32)},
        {"nt": 2, "A2d": np.diag([1.0, 1.0, 2.0]).astype(np.float32)},
    ],
)
def test_pair_registration_rejects_affine_settings(config):
    xJ = _axes(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    J = np.zeros((1, 3, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="upsampling"):
        upsample_module.upsample_between_slices(
            xJ,
            J,
            mode="seg",
            config=config,
            parallel=False,
        )


def test_pair_registration_rejects_out_of_plane_false():
    xJ = _axes(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    J = np.zeros((1, 3, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="out_of_plane=False is not supported"):
        upsample_module.upsample_between_slices(
            xJ,
            J,
            mode="seg",
            config={"nt": 2, "out_of_plane": False},
            parallel=False,
        )


def test_symmetric_helper_returns_both_jacobian_keys(monkeypatch):
    nt = 4

    def fake_multiscale(**kwargs):
        assert [axis.numel() for axis in kwargs["xI"]] == [2, 2, 2]
        assert tuple(kwargs["I"].shape) == (1, 2, 2, 2)
        return {
            "v": torch.zeros((nt, 3, 2, 2, 2), dtype=torch.float32),
            "xv": [
                torch.arange(2, dtype=torch.float32),
                torch.arange(2, dtype=torch.float32),
                torch.arange(2, dtype=torch.float32),
            ],
        }

    def fake_integrate_inverse_flow(
        xv,
        v,
        *,
        emlddmm_module,
        interp2d=None,
        grid_sample_kwargs=None,
    ):
        return torch.zeros((nt + 1, 2, 2, 2), dtype=torch.float32)

    def fake_warp_time_series(
        x,
        image,
        phis,
        *,
        emlddmm_module,
        interp2d=None,
        grid_sample_kwargs=None,
    ):
        base = torch.arange(phis.shape[0], dtype=torch.float32)[:, None, None, None]
        channels = torch.as_tensor(image).shape[0]
        return base.expand(phis.shape[0], channels, 2, 2)

    def fake_det(phi, spacing=None):
        base = torch.arange(phi.shape[0], dtype=torch.float32)[:, None, None]
        return base.expand(phi.shape[0], 2, 2)

    monkeypatch.setattr(
        symmetric_module,
        "_resolve_emlddmm_module",
        lambda: _fake_symmetric_backend(fake_multiscale),
    )
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


def test_symmetric_helper_passes_coordinate_axes_as_tuples(monkeypatch):
    nt = 3
    multiscale_calls = []

    def fake_multiscale(**kwargs):
        multiscale_calls.append(kwargs)
        assert isinstance(kwargs["xI"], tuple)
        assert isinstance(kwargs["xJ"], tuple)
        assert [axis.numel() for axis in kwargs["xI"]] == [2, 3, 4]
        assert [axis.numel() for axis in kwargs["xJ"]] == [2, 3, 4]
        assert tuple(kwargs["I"].shape) == (1, 2, 3, 4)
        assert kwargs["downI"] == [[1, 1, 1]]
        assert kwargs["downJ"] == [[1, 1, 1]]
        assert kwargs["out_of_plane"] is False
        assert kwargs["eA"] == 0.0
        assert kwargs["eA2d"] == 0.0
        assert kwargs["slice_matching"] is False
        assert kwargs["Amode"] == 0
        assert torch.allclose(kwargs["A"], torch.eye(4))
        assert kwargs["A2d"] is None
        assert kwargs["update_A"] is False
        assert kwargs["update_matching_weights"] is False
        return {
            "v": torch.zeros((nt, 3, 2, 3, 4), dtype=torch.float32),
            "xv": [
                torch.arange(2, dtype=torch.float32),
                torch.arange(3, dtype=torch.float32),
                torch.arange(4, dtype=torch.float32),
            ],
        }

    def fake_integrate_inverse_flow(
        xv,
        v,
        *,
        emlddmm_module,
        interp2d=None,
        grid_sample_kwargs=None,
    ):
        return torch.zeros((nt + 1, 2, 3, 4), dtype=torch.float32)

    def fake_warp_time_series(
        x,
        image,
        phis,
        *,
        emlddmm_module,
        interp2d=None,
        grid_sample_kwargs=None,
    ):
        channels = torch.as_tensor(image).shape[0]
        return torch.zeros((phis.shape[0], channels, 3, 4), dtype=torch.float32)

    def fake_det(phi, spacing=None):
        return torch.ones((phi.shape[0], 3, 4), dtype=torch.float32)

    monkeypatch.setattr(
        symmetric_module,
        "_resolve_emlddmm_module",
        lambda: _fake_symmetric_backend(fake_multiscale),
    )
    monkeypatch.setattr(symmetric_module, "_integrate_inverse_flow", fake_integrate_inverse_flow)
    monkeypatch.setattr(symmetric_module, "_warp_time_series", fake_warp_time_series)
    monkeypatch.setattr(symmetric_module, "_calculate_determinant_of_jacobian", fake_det)

    symmetric_module.emlddmm_multiscale_symmetric_N(
        xI=[np.arange(3, dtype=np.float32), np.arange(4, dtype=np.float32)],
        I=np.ones((1, 3, 4), dtype=np.float32),
        xJ=[
            np.arange(3, dtype=np.float32) + 10.0,
            np.arange(4, dtype=np.float32) + 20.0,
        ],
        J=np.ones((1, 3, 4), dtype=np.float32) * 2.0,
        nt=nt,
    )

    assert len(multiscale_calls) == 2
    assert all(isinstance(call["xI"], tuple) for call in multiscale_calls)
    assert all(isinstance(call["xJ"], tuple) for call in multiscale_calls)


def test_symmetric_helper_resamples_velocity_transform_to_image_grid():
    xv = [
        torch.linspace(0.0, 3.0, 5),
        torch.linspace(0.0, 3.0, 5),
    ]
    x = [
        torch.arange(4, dtype=torch.float32),
        torch.arange(4, dtype=torch.float32),
    ]
    velocity_mesh = torch.stack(torch.meshgrid(xv, indexing="ij"))
    phis = velocity_mesh[None].repeat(2, 1, 1, 1)

    out = symmetric_module._resample_transform_to_domain(
        xv,
        phis,
        x,
        emlddmm_module=_fake_symmetric_backend(),
        interp2d=True,
    )

    image_mesh = torch.stack(torch.meshgrid(x, indexing="ij"))
    assert tuple(out.shape) == (2, 2, 4, 4)
    assert torch.allclose(out[0], image_mesh)


def test_symmetric_helper_uses_domain_specific_axes_and_spacing(monkeypatch):
    nt = 4
    warp_axes = []
    det_spacings = []

    def fake_multiscale(**kwargs):
        return {
            "v": torch.zeros((nt, 3, 2, 2, 2), dtype=torch.float32),
            "xv": [
                torch.arange(2, dtype=torch.float32),
                torch.arange(2, dtype=torch.float32),
                torch.arange(2, dtype=torch.float32),
            ],
        }

    def fake_integrate_inverse_flow(
        xv,
        v,
        *,
        emlddmm_module,
        interp2d=None,
        grid_sample_kwargs=None,
    ):
        return torch.zeros((nt + 1, 2, 2, 2), dtype=torch.float32)

    def fake_warp_time_series(
        x,
        image,
        phis,
        *,
        emlddmm_module,
        interp2d=None,
        grid_sample_kwargs=None,
    ):
        warp_axes.append([np.asarray(axis) for axis in x])
        channels = torch.as_tensor(image).shape[0]
        return torch.zeros((phis.shape[0], channels, 2, 2), dtype=torch.float32)

    def fake_det(phi, spacing=None):
        det_spacings.append(tuple(float(s) for s in spacing))
        return torch.ones((phi.shape[0], 2, 2), dtype=torch.float32)


    monkeypatch.setattr(
        symmetric_module,
        "_resolve_emlddmm_module",
        lambda: _fake_symmetric_backend(fake_multiscale),
    )
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
