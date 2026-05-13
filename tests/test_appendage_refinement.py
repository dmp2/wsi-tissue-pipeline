from __future__ import annotations

import numpy as np
from skimage import measure


def _canvas(shape: tuple[int, int] = (160, 240)) -> tuple[np.ndarray, np.ndarray]:
    img = np.ones((*shape, 3), dtype=np.uint8) * 245
    mask = np.zeros(shape, dtype=bool)
    return img, mask


def test_adaptive_he_features_separate_tissue_from_pale_artifact():
    from wsi_pipeline.segmentation.stain import he_features, he_stain_mask

    img = np.ones((64, 96, 3), dtype=np.uint8) * 245
    img[12:52, 10:35] = [160, 80, 145]   # hematoxylin/purple tissue
    img[12:52, 60:85] = [220, 145, 175]  # eosin/pink tissue
    img[4:10, 5:90] = [235, 235, 225]    # pale background-like strip
    img[28:36, 35:60] = [180, 180, 180]  # neutral high-OD artifact

    features = he_features(img)
    mask = he_stain_mask(img, mode="adaptive-he")

    assert features.he_signal[20:40, 15:30].mean() > features.he_signal[4:10, 5:90].mean()
    assert mask[20:40, 15:30].mean() > 0.95
    assert mask[20:40, 65:80].mean() > 0.95
    assert mask[4:10, 5:90].sum() == 0
    assert mask[28:36, 35:60].sum() == 0


def test_appendage_refinement_trims_attached_pale_strip():
    from wsi_pipeline.segmentation.appendage import refine_appendages

    img, mask = _canvas()
    img[50:130, 70:145] = [160, 80, 145]
    mask[50:130, 70:145] = True
    img[35:43, 65:170] = [235, 235, 225]
    mask[35:43, 65:170] = True
    mask[42:55, 95:120] = True  # short neck attaching strip to tissue
    img[42:55, 95:120] = [235, 235, 225]

    refined, info = refine_appendages(img, mask, min_area_px=500)

    assert info["n_appendages_trimmed"] >= 1
    assert info["trimmed_area_px"] > 0
    assert refined[35:43, 65:170].sum() == 0
    assert refined[65:120, 80:135].mean() > 0.95


def test_appendage_refinement_preserves_high_he_thin_tissue():
    from wsi_pipeline.segmentation.appendage import refine_appendages

    img, mask = _canvas()
    img[50:130, 70:145] = [160, 80, 145]
    mask[50:130, 70:145] = True
    img[35:43, 65:170] = [160, 80, 145]
    mask[35:43, 65:170] = True
    img[42:55, 95:120] = [160, 80, 145]
    mask[42:55, 95:120] = True

    refined, info = refine_appendages(img, mask, min_area_px=500)

    assert info["n_appendages_trimmed"] == 0
    assert refined[35:43, 65:170].mean() > 0.95


def test_appendage_refinement_removes_pale_bridge_between_tissue_bodies():
    from wsi_pipeline.segmentation.appendage import refine_appendages

    img, mask = _canvas()
    img[55:125, 35:95] = [160, 80, 145]
    img[55:125, 145:205] = [220, 145, 175]
    mask[55:125, 35:95] = True
    mask[55:125, 145:205] = True
    img[80:88, 95:145] = [235, 235, 225]
    mask[80:88, 95:145] = True

    assert measure.label(mask, connectivity=2).max() == 1
    refined, info = refine_appendages(img, mask, min_area_px=500)

    assert info["n_appendages_trimmed"] >= 1
    assert refined[80:88, 95:145].sum() == 0
    assert measure.label(refined, connectivity=2).max() == 2
