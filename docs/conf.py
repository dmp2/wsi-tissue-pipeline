"""Sphinx configuration for the WSI Tissue Pipeline documentation.

The docs build imports the package with ``pip install --no-deps -e .`` and mocks
the heavy/runtime-sensitive dependencies (see ``autodoc_mock_imports``) so the
full WSI runtime stack is not required just to render the API reference.
"""

import os
import sys

# Make the package importable for autodoc/autosummary without installing deps.
sys.path.insert(0, os.path.abspath("../src"))

# Several package modules use runtime-evaluated PEP 604 unions with heavy types,
# e.g. ``def f(x: np.ndarray | da.Array)``. When those heavy modules are mocked
# (see ``autodoc_mock_imports``), the mock object must support ``|`` or importing
# the module raises ``TypeError: unsupported operand type(s) for |``. Teach the
# Sphinx mock objects to participate in unions so autosummary can import cleanly.
from sphinx.ext.autodoc.mock import _MockObject as _SphinxMockObject


def _mock_union(self, other):  # noqa: ANN001
    return self


_SphinxMockObject.__or__ = _mock_union
_SphinxMockObject.__ror__ = _mock_union

# -- Project information -----------------------------------------------------

project = "WSI Tissue Pipeline"
author = "Dominic Padova"
copyright = f"2025, {author}"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

templates_path = ["_templates"]

# Test modules and build artifacts must stay out of autosummary/API generation
# and the source tree, otherwise the -W build treats them as orphans or import
# failures.
exclude_patterns = [
    "_build",
    "_static",
    "_templates",
    "**.ipynb_checkpoints",
    "**/test_*.py",
    "test_*.py",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Under -W these two categories would otherwise fail the build without any doc
# issue on our side:
#   * misc.highlighting_failure -- some existing ```python blocks contain Jupyter
#     line magics (`!pip ...`, `%cd ...`) that the Pygments python lexer can't
#     tokenize; it falls back to plain text, which is fine.
#   * myst.xref_missing -- the included CONTRIBUTING.md links to the repo-root
#     LICENSE file, which is not part of the Sphinx source tree.
#   * docutils -- some package docstrings use loose RST (short title underlines,
#     stray inline markup, unexpected indentation). We render them as-is rather
#     than editing package source, so these parse nits are not build blockers.
suppress_warnings = [
    "misc.highlighting_failure",
    "myst.xref_missing",
    "docutils",
]

# -- MyST (Markdown) ---------------------------------------------------------

# Note: GFM-style tables are enabled by default in myst-parser, so "tables" is
# not a named enable_extension (listing it errors out on recent myst-parser).
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
]
myst_heading_anchors = 3

# -- Notebooks (nbsphinx) ----------------------------------------------------

# The example notebooks need large datasets and heavy optional dependencies, so
# they must NOT run at build time -- render them from their saved outputs.
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
# Do NOT force ``:members:`` globally: that makes the top-level package page
# re-document every re-exported name in ``wsi_pipeline.__all__`` (which also
# lives in its real submodule page), producing duplicate object descriptions and
# ambiguous cross-references. Member documentation is driven by the autosummary
# templates instead (module.rst / class.rst), which document each object once on
# its own stub page.
autosummary_imported_members = False
autodoc_typehints = "description"
autodoc_class_signature = "mixed"

# Install-tier deps (numpy, pydantic, click, rich, pyyaml, ...) are present in
# docs/requirements.txt for autodoc fidelity. Only genuinely heavy or
# runtime-sensitive imports are mocked here.
autodoc_mock_imports = [
    "cv2",
    "skimage",
    "dask",
    "dask_image",
    "distributed",
    "tensorstore",
    "torch",
    "torchvision",
    "neuroglancer",
    "mlflow",
    "cloudvolume",
    "cloud_volume",
    "pyvista",
    "vtk",
    "jnius",
    "jpype",
    "ngff_zarr",
    "zarr",
    "numcodecs",
    "tifffile",
    "matplotlib",
    "tinybrain",
    "imageio",
    "scipy",
    "pandas",
    "PIL",
    "sklearn",
]

# -- Napoleon (Google/NumPy docstrings) --------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
# Render documented "Attributes" as :ivar: fields inside the class description
# rather than standalone .. attribute:: directives. This avoids duplicate object
# descriptions when a class both documents an attribute in its docstring and
# defines a real property of the same name (e.g. ETSFile.compression_str).
napoleon_use_ivar = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "WSI Tissue Pipeline"
# NOTE: html_logo / html_favicon are intentionally unset -- drop logo.png into
# docs/_static/ and wire it here once it exists (referencing a missing file
# would fail the -W build).

# -- Copybutton --------------------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
