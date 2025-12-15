# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'annolid'
copyright = '2021, Chen Yang'
author = 'Chen Yang'

def _read_release() -> str:
    """Return the Annolid version for docs rendering.

    Prefer installed package metadata, then fall back to pyproject.toml.
    """
    try:
        from importlib import metadata
        return metadata.version("annolid")
    except Exception:
        pass

    try:
        import tomllib  # py3.11+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore
        except Exception:
            return "unknown"

    try:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        return str(data.get("project", {}).get("version", "unknown"))
    except Exception:
        return "unknown"


# The full version, including alpha/beta/rc tags.
release = _read_release()
version = release


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['./_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Legacy API stubs that assume a different packaging/layout.
    "_templates/*.rst",
    "modules.rst",
    "setup.rst",
    "tests.rst",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['./_static']

try:
    import sphinxcontrib.spelling
except ImportError:
    pass

else:
    extensions.append("sphinxcontrib.spelling")

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
