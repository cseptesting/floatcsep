# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
from sphinx_gallery.sorting import FileNameSortKey

sys.path.insert(0, os.path.abspath(".."))

#  NOTE: To expose sphinx warnings, comment `setup_logger()` in floatcsep.commands.main


# -- Project information -----------------------------------------------------

project = "floatCSEP"
copyright = "2024, Pablo Iturrieta"
author = "Pablo Iturrieta"
release = "v0.2.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

# language = 'en'
autosummary_generate = False
autoclass_content = "both"
suppress_warnings = [
    "autosummary",
    "autosummary.missing",
]
templates_path = ["_templates"]
source_suffix = ".rst"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "default"
autodoc_typehints = "description"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pycsep": ("https://docs.cseptesting.org/", None),
}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "display_version": True,
    "prev_next_buttons_location": "both",
    "sticky_navigation": True,
    "collapse_navigation": True,
    "style_nav_header_background": "#343131ff",
    "logo_only": True,
}
html_logo = "_static/floatcsep_logo.svg"
html_js_files = [
    "custom.js",
]

todo_include_todos = False

copybutton_prompt_text = "$ "  # Text to ignore when copying (for shell commands)
copybutton_only_copy_prompt_lines = False


rst_epilog = """
.. raw:: html

    <hr />
    <div style="text-align: center;">
        <a href="https://github.com/cseptesting/floatcsep">GitHub</a> |
        <a href="https://cseptesting.org">CSEP Website</a> |
        <a href="https://github.com/sceccode/pycsep">pyCSEP</a> |
        <a href="https://floatcsep.readthedocs.io/_/downloads/en/latest/pdf/">Download PDF</a>
    </div>
"""
