"""Regression tests for in-memory pyramid construction (``build_mips_from_yxc``).

Generated levels are only needed for data without a native pyramid: derived or
cropped in-memory tiles (plating, VSI small-tile paths, precomputed plates), masks
and synthetic images.  Each level must be the 2x2 block mean of the previous one,
whatever the input's memory layout, dtype, channel count or (odd) size.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from wsi_pipeline.omezarr import build_mips_from_yxc, write_ngff_from_mips


def _reference_2x2(img: np.ndarray) -> np.ndarray:
    """Loop-based 2x2 mean; an edge block averages only the pixels it contains."""
    h, w = img.shape[:2]
    out = np.zeros(((h + 1) // 2, (w + 1) // 2) + img.shape[2:], dtype=np.float64)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = (
                img[2 * i : 2 * i + 2, 2 * j : 2 * j + 2].astype(np.float64).mean(axis=(0, 1))
            )
    return out


def _random_yxc(shape: tuple[int, ...], dtype, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, np.iinfo(dtype).max, size=shape, endpoint=True).astype(dtype)
    return rng.random(shape).astype(dtype)


def _layouts(base: np.ndarray) -> dict[str, np.ndarray]:
    padded = np.zeros((base.shape[0] + 3, base.shape[1] + 5, base.shape[2]), dtype=base.dtype)
    padded[1 : 1 + base.shape[0], 2 : 2 + base.shape[1]] = base
    return {
        "c_order": np.ascontiguousarray(base),
        "fortran_order": np.asfortranarray(base),
        "strided_view": padded[1 : 1 + base.shape[0], 2 : 2 + base.shape[1]],
    }


@pytest.mark.parametrize("shape", [(16, 12), (17, 13), (5, 7), (9, 6), (6, 9), (64, 33)])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_levels_are_2x2_means_for_any_layout(shape, channels, dtype):
    base = _random_yxc(shape + (channels,), dtype)
    tolerance = 0.5 if np.issubdtype(dtype, np.integer) else 1e-6
    for name, img in _layouts(base).items():
        mips = build_mips_from_yxc(img, 3)
        assert len(mips) == 3, name
        np.testing.assert_array_equal(mips[0], base)
        for level in (1, 2):
            expected = _reference_2x2(mips[level - 1])
            assert mips[level].shape == expected.shape, (name, level)
            assert mips[level].dtype == base.dtype, (name, level)
            np.testing.assert_allclose(
                mips[level].astype(np.float64),
                expected,
                atol=tolerance,
                err_msg=f"{name} level {level}",
            )


def test_channels_stay_separate():
    base = np.zeros((10, 14, 3), dtype=np.uint8)
    base[..., 0], base[..., 1], base[..., 2] = 10, 120, 250
    for level in build_mips_from_yxc(base, 4)[1:]:
        assert [np.unique(level[..., c]).tolist() for c in range(3)] == [[10], [120], [250]]


def test_single_level_returns_base():
    base = _random_yxc((7, 5, 3), np.uint8)
    mips = build_mips_from_yxc(base, 1)
    assert len(mips) == 1 and mips[0] is base


def test_written_ngff_levels_match_reference(tmp_path):
    """The in-memory use case end to end: a C-ordered derived tile written as OME-Zarr."""
    base = _random_yxc((33, 47, 3), np.uint8, seed=7)
    write_ngff_from_mips(
        build_mips_from_yxc(base, 2), tmp_path / "tile.ome.zarr", phys_xy_um=(0.5, 0.5)
    )
    s1 = np.moveaxis(
        np.asarray(zarr.open_array(str(tmp_path / "tile.ome.zarr" / "s1"), mode="r")[:]), 0, -1
    )
    np.testing.assert_allclose(s1.astype(np.float64), _reference_2x2(base), atol=0.5)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("shape", [(80, 114), (81, 115), (80, 115), (81, 114)])
def test_hybrid_matches_numpy_exactly_across_layouts(dtype, channels, shape):
    from wsi_pipeline.omezarr.pyramid import _average_pool_2x2_yxc

    base = _random_yxc(shape + (channels,), dtype, seed=19)
    layouts = _layouts(base)
    layouts["reversed"] = base[::-1, ::-1]
    layouts["transposed"] = base.transpose(1, 0, 2)
    layouts["spatial_stride"] = np.repeat(np.repeat(base, 2, axis=0), 2, axis=1)[::2, ::2]
    layouts["channel_stride"] = np.repeat(base, 2, axis=2)[..., ::2]
    for name, img in layouts.items():
        expected = img
        for got in build_mips_from_yxc(img, 9)[1:]:
            expected = _average_pool_2x2_yxc(expected)
            np.testing.assert_array_equal(got, expected, err_msg=name)
            assert got.dtype == img.dtype


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("order", ["C", "F"])
def test_large_inputs_really_use_bounded_native_tiles(monkeypatch, dtype, order):
    import tinybrain.accelerated

    import wsi_pipeline.omezarr.pyramid as pyramid

    base = np.array(_random_yxc((1057, 1091, 3), dtype), order=order)
    native = tinybrain.accelerated.average_pooling_2x2
    calls = []

    def checked(work, num_mips):
        assert work.flags.f_contiguous
        assert work.dtype == np.float32
        assert num_mips == 1
        assert all(s % 2 == 0 for s in work.shape)
        assert work.nbytes <= 4 * 1024**2
        calls.append(work.shape)
        return native(work, num_mips)

    monkeypatch.setattr(tinybrain.accelerated, "average_pooling_2x2", checked)
    expected = base
    for got in build_mips_from_yxc(base, 4)[1:]:
        expected = pyramid._average_pool_2x2_yxc(expected)
        np.testing.assert_array_equal(got, expected)
    assert len(calls) >= 12


def test_rounding_ties_and_per_level_rounding():
    from wsi_pipeline.omezarr.pyramid import _average_pool_2x2_yxc

    # Quarter means 0.5, 1.5, 2.5, 3.5 distinguish truncation and half-up.
    base = np.tile(
        np.array([[0, 0, 1, 1, 2, 2, 3, 3], [1, 1, 2, 2, 3, 3, 4, 4]], np.uint16)[..., None],
        (40, 16, 1),
    )
    got = build_mips_from_yxc(base, 5)
    np.testing.assert_array_equal(got[1][0, :4, 0], [0, 2, 2, 4])
    expected = base
    for level in got[1:]:
        expected = _average_pool_2x2_yxc(expected)
        np.testing.assert_array_equal(level, expected)


@pytest.mark.parametrize("shape", [(1, 115, 3), (81, 1, 1), (1, 1, 3)])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_degenerate_dimensions(shape, dtype):
    from wsi_pipeline.omezarr.pyramid import _average_pool_2x2_yxc

    img = _random_yxc(shape, dtype)
    expected = img
    for level in build_mips_from_yxc(img, 5)[1:]:
        expected = _average_pool_2x2_yxc(expected)
        np.testing.assert_array_equal(level, expected)


def test_float32_extremes_keep_reference_accumulation(monkeypatch):
    import tinybrain.accelerated

    from wsi_pipeline.omezarr.pyramid import _average_pool_2x2_yxc

    def unexpected(*args):
        pytest.fail("float32 must retain NumPy float64 accumulation")

    monkeypatch.setattr(tinybrain.accelerated, "average_pooling_2x2", unexpected)
    img = np.tile(np.array([[1e30, 1], [-1e30, 3]], np.float32)[..., None], (40, 40, 1))
    np.testing.assert_array_equal(build_mips_from_yxc(img, 2)[1], _average_pool_2x2_yxc(img))


def test_missing_tinybrain_uses_numpy(monkeypatch):
    import builtins

    from wsi_pipeline.omezarr.pyramid import _average_pool_2x2_yxc

    real_import = builtins.__import__

    def without_tinybrain(name, *args, **kwargs):
        if name.startswith("tinybrain"):
            raise ImportError("simulated unavailable optional accelerator")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_tinybrain)
    img = _random_yxc((81, 115, 3), np.uint16)
    np.testing.assert_array_equal(build_mips_from_yxc(img, 2)[1], _average_pool_2x2_yxc(img))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("order", ["C", "F"])
def test_hybrid_ngff_roundtrip_all_levels(tmp_path, dtype, order):
    from wsi_pipeline.omezarr.pyramid import _average_pool_2x2_yxc

    base = np.array(_random_yxc((81, 115, 3), dtype), order=order)
    path = tmp_path / "hybrid.ome.zarr"
    write_ngff_from_mips(build_mips_from_yxc(base, 5), path, phys_xy_um=(0.5, 0.5))
    expected = base
    for k in range(5):
        got = np.moveaxis(np.asarray(zarr.open_array(str(path / f"s{k}"), mode="r")[:]), 0, -1)
        np.testing.assert_array_equal(got, expected)
        expected = _average_pool_2x2_yxc(expected)


@pytest.mark.parametrize("dtype,side", [(np.uint8, 32), (np.uint16, 512)])
def test_acceleration_rounds_every_possible_four_pixel_sum(dtype, side):
    limit = np.iinfo(dtype).max
    sums = np.minimum(np.arange(side * side).reshape(side, side), 4 * limit)
    base = np.empty((side * 2, side * 2, 1), dtype=dtype)
    for k, (dy, dx) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        base[dy::2, dx::2, 0] = np.clip(sums - k * limit, 0, limit)
    expected = np.rint(sums / 4).astype(dtype)
    np.testing.assert_array_equal(build_mips_from_yxc(base, 2)[1][..., 0], expected)


@pytest.mark.parametrize("dtype", [np.int16, np.uint32, np.float64, np.dtype(">u2")])
def test_other_dtypes_keep_numpy_fallback(monkeypatch, dtype):
    import tinybrain.accelerated

    from wsi_pipeline.omezarr.pyramid import _average_pool_2x2_yxc

    def unexpected(*args):
        pytest.fail("unsupported native dtype must use NumPy")

    monkeypatch.setattr(tinybrain.accelerated, "average_pooling_2x2", unexpected)
    base = np.arange(80 * 114 * 3).reshape(80, 114, 3).astype(dtype)
    expected = base
    for got in build_mips_from_yxc(base, 4)[1:]:
        expected = _average_pool_2x2_yxc(expected)
        np.testing.assert_array_equal(got, expected)
