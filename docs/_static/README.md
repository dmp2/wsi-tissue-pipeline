# Static assets

This directory holds static assets served with the documentation site.

Drop-in slots (not yet referenced anywhere, so their absence does not break the
`-W` build):

- `logo.png` — project logo. Once added, wire it up in `docs/conf.py` via
  `html_logo = "_static/logo.png"`.
- `pipeline.png` — an overview diagram you can embed in `docs/index.md` or
  `docs/pipeline_overview.md` with `![Pipeline](_static/pipeline.png)`.

Only reference these files after they actually exist here — the docs build runs
with warnings-as-errors (`-W`), so a link to a missing image will fail the build.
