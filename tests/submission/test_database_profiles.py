from __future__ import annotations

from pathlib import Path

import pytest

from wsi_pipeline.submission import ProfileValidationError, load_database_profile

PROFILE_PATH = Path("configs/database_profiles/national_database_ometiff.yaml")


def test_database_profile_yaml_loads_successfully():
    profile = load_database_profile(PROFILE_PATH)

    assert profile.profile_name == "national_database_ometiff"
    assert profile.profile_version == "0.1.0"
    assert profile.input.accepted_extensions == [".vsi"]
    assert profile.output.extension == ".ome.tif"
    assert profile.naming.template == "sub-{specimen_id}_slide-{slide_id}_tissue-{tissue_id}.ome.tif"


def test_profile_loader_fails_clearly_when_required_sections_are_missing(tmp_path):
    profile_path = tmp_path / "missing_sections.yaml"
    profile_path.write_text(
        """
profile_name: invalid
profile_version: 0.1.0
input:
  accepted_extensions:
    - .vsi
  require_ets_companion: true
  raw_files_read_only: true
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ProfileValidationError, match="missing required top-level keys") as excinfo:
        load_database_profile(profile_path)

    assert "output" in str(excinfo.value)
    assert "validation" in str(excinfo.value)


def test_profile_naming_template_requires_submission_placeholders(tmp_path):
    profile_path = tmp_path / "missing_tissue_placeholder.yaml"
    profile_text = PROFILE_PATH.read_text(encoding="utf-8")
    profile_path.write_text(
        profile_text.replace("_tissue-{tissue_id}", "_tissue-fixed"),
        encoding="utf-8",
    )

    with pytest.raises(ProfileValidationError, match="tissue_id"):
        load_database_profile(profile_path)
