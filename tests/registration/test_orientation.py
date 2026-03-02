from __future__ import annotations

import pytest

from wsi_pipeline.registration.orientation import (
    list_valid_orientation_codes,
    validate_orientation_code,
)


def test_orientation_code_list_is_complete():
    codes = list_valid_orientation_codes()

    assert len(codes) == 48
    assert codes == sorted(codes)
    assert "RAS" in codes
    assert "LPI" in codes
    assert "PIR" in codes
    assert "RIP" in codes


@pytest.mark.parametrize("code", ["ras", "LPI", "pir", "RIP"])
def test_validate_orientation_code_accepts_valid_examples(code):
    assert validate_orientation_code(code) == code.upper()


@pytest.mark.parametrize("code", ["", "RA", "RRS", "RAX", "RASS"])
def test_validate_orientation_code_rejects_invalid_examples(code):
    with pytest.raises(ValueError, match="Orientation codes must be exactly 3 letters"):
        validate_orientation_code(code)
