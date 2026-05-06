from wsi_pipeline.emlddmm_prep import resolve_emlddmm_checkout


def test_resolve_emlddmm_checkout_from_notebooks_cwd(tmp_path, monkeypatch):
    repo = tmp_path / "wsi-tissue-pipeline"
    notebooks = repo / "notebooks"
    checkout = tmp_path / "emlddmm"
    notebooks.mkdir(parents=True)
    checkout.mkdir()
    (checkout / "histsetup.py").write_text("", encoding="utf-8")
    (checkout / "emlddmm.py").write_text("", encoding="utf-8")

    monkeypatch.delenv("EMLDDMM_HOME", raising=False)
    monkeypatch.delenv("EMLDDMM_REPO", raising=False)
    monkeypatch.chdir(notebooks)

    assert resolve_emlddmm_checkout() == checkout


def test_resolve_emlddmm_checkout_from_env(tmp_path, monkeypatch):
    checkout = tmp_path / "custom-emlddmm"
    checkout.mkdir()
    (checkout / "histsetup.py").write_text("", encoding="utf-8")
    (checkout / "emlddmm.py").write_text("", encoding="utf-8")

    monkeypatch.setenv("EMLDDMM_HOME", str(checkout))

    assert resolve_emlddmm_checkout() == checkout
