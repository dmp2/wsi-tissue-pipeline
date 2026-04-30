from __future__ import annotations

import numpy as np


def _blank_canvas(shape: tuple[int, int] = (220, 320)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = np.ones((*shape, 3), dtype=np.uint8) * 245
    mask = np.zeros(shape, dtype=bool)
    stain_mask = np.zeros(shape, dtype=bool)
    return img, mask, stain_mask


def _add_rect(
    img: np.ndarray,
    mask: np.ndarray,
    stain_mask: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    *,
    color: tuple[int, int, int],
    stained: bool,
) -> None:
    img[y0:y1, x0:x1] = color
    mask[y0:y1, x0:x1] = True
    stain_mask[y0:y1, x0:x1] = stained


def test_component_qc_keeps_large_tissue_like_components():
    from wsi_pipeline.segmentation.component_qc import score_components

    img, mask, stain_mask = _blank_canvas()
    for x0 in (20, 120, 220):
        _add_rect(
            img,
            mask,
            stain_mask,
            40,
            145,
            x0,
            x0 + 70,
            color=(168, 90, 145),
            stained=True,
        )

    records = score_components(img, mask, stain_mask=stain_mask)

    assert len(records) == 3
    assert [record.tile_index_on_source for record in records] == [0, 1, 2]
    assert not any(record.artifact_likely for record in records)


def test_component_qc_flags_long_thin_low_stain_strip():
    from wsi_pipeline.segmentation.component_qc import score_components

    img, mask, stain_mask = _blank_canvas()
    for x0 in (20, 120, 220):
        _add_rect(
            img,
            mask,
            stain_mask,
            70,
            150,
            x0,
            x0 + 55,
            color=(168, 90, 145),
            stained=True,
        )
    _add_rect(
        img,
        mask,
        stain_mask,
        20,
        30,
        40,
        200,
        color=(235, 235, 225),
        stained=False,
    )

    records = score_components(img, mask, stain_mask=stain_mask)
    artifact_records = [record for record in records if record.artifact_likely]

    assert len(artifact_records) == 1
    assert "thin_low_stain_component" in artifact_records[0].artifact_reason


def test_component_qc_flags_small_low_od_fragment():
    from wsi_pipeline.segmentation.component_qc import score_components

    img, mask, stain_mask = _blank_canvas()
    for x0 in (20, 120, 220):
        _add_rect(
            img,
            mask,
            stain_mask,
            55,
            155,
            x0,
            x0 + 65,
            color=(168, 90, 145),
            stained=True,
        )
    _add_rect(
        img,
        mask,
        stain_mask,
        180,
        192,
        50,
        62,
        color=(242, 242, 235),
        stained=False,
    )

    records = score_components(img, mask, stain_mask=stain_mask)
    artifact_records = [record for record in records if record.artifact_likely]

    assert len(artifact_records) == 1
    assert "tiny_low_od_fragment" in artifact_records[0].artifact_reason


def test_component_qc_does_not_flag_high_stain_component_for_aspect_ratio_only():
    from wsi_pipeline.segmentation.component_qc import score_components

    img, mask, stain_mask = _blank_canvas()
    _add_rect(
        img,
        mask,
        stain_mask,
        100,
        115,
        40,
        230,
        color=(168, 90, 145),
        stained=True,
    )

    records = score_components(img, mask, stain_mask=stain_mask)

    assert len(records) == 1
    assert records[0].aspect_ratio >= 7.0
    assert records[0].stain_fraction == 1.0
    assert not records[0].artifact_likely
