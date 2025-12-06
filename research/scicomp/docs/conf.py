# Configuration file for the Sphinx documentation builder.
# SciComp Documentation
import os
import sys
import datetime
# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../Python'))
# -- Project information -----------------------------------------------------
project = 'SciComp'
copyright = f'{datetime.datetime.now().year}, University of California, Berkeley'
author = 'Meshal Alawein, UC Berkeley SciComp Team'
release = '1.0.0'
version = '1.0.0'
# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
# -- Options for HTML output -------------------------------------------------
html_theme = 'basic'
# Berkeley color scheme
html_theme_options = {
    'nosidebar': False,
}
# Berkeley branding
html_title = f"SciComp v{version}"
# -- Extension configuration -------------------------------------------------
# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}
# MathJax configuration
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    }
}
# Source file extensions
source_suffix = '.rst'
# Master document
master_doc = 'index'
# Berkeley-specific footer
html_last_updated_fmt = '%b %d, %Y'
html_show_sphinx = False
html_show_copyright = True