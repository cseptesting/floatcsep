# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
from sphinx_gallery.sorting import FileNameSortKey

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'floatCSEP'
copyright = '2022, Pablo Iturrieta'
author = 'Pablo Iturrieta'
release = 'v0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    # 'sphinx_gallery.gen_gallery',
    # 'sphinx.ext.githubpages'
]

templates_path = ['_templates']
source_suffix = '.rst'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'default'  # todo
autodoc_typehints = 'none'
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ('http://docs.scipy.org/doc/scipy/reference', None),
    "matplotlib": ('http://matplotlib.sourceforge.net/', None),
    'pycsep': ('https://docs.cseptesting.org/', None)
}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'display_version': True,
    'prev_next_buttons_location': 'both',
    'collapse_navigation': False,
    'style_nav_header_background': '#343131ff',
    'logo_only': True,
}
html_logo = '_static/floatcsep_logo.svg'
todo_include_todos = False
