# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'feCSEP'
copyright = '2022, Pablo Iturrieta'
author = 'Pablo Iturrieta'
release = 'v0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'autoapi.extension'
]

autoapi_dirs = ['../../fecsep']
templates_path = ['_templates']
exclude_patterns = []
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
