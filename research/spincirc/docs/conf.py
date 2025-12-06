# SpinCirc Documentation Configuration
# Sphinx configuration for generating comprehensive documentation

import os
import sys
from datetime import datetime

# Add project root to path for autodoc
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../python'))

# -- Project information -----------------------------------------------------

project = 'SpinCirc'
copyright = f'2025, Meshal Alawein, UC Berkeley'
author = 'Meshal Alawein'
release = '1.0.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinxcontrib.matlab',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_rtd_theme',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#003262',  # Berkeley Blue
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Custom CSS
html_static_path = ['_static']
html_css_files = [
    'css/berkeley_theme.css',
]

# Logo
html_logo = '_static/images/spincirc_logo.png'
html_favicon = '_static/images/favicon.ico'

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

# MATLAB settings
matlab_src_dir = os.path.abspath('../matlab')
matlab_keep_package_prefix = False

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# Math settings
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'macros': {
            'bm': ['\\boldsymbol{#1}', 1],
            'vb': ['\\mathbf{#1}', 1],
            'vc': ['\\boldsymbol{#1}', 1],
            'hbar': '\\hbar',
            'kb': 'k_\\mathrm{B}',
            'muB': '\\mu_\\mathrm{B}',
            'TMR': '\\mathrm{TMR}',
        }
    }
}

# Copy button configuration
copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: '
copybutton_prompt_is_regexp = True

# MyST parser settings
myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
    'html_admonition',
    'html_image',
    'linkify',
    'replacements',
    'smartquotes',
    'substitution',
    'tasklist',
]

# -- Custom setup -----------------------------------------------------------

def setup(app):
    """Custom Sphinx setup function"""
    app.add_css_file('css/berkeley_theme.css')
    
    # Add custom roles
    app.add_role('matlab', matlab_role)
    app.add_role('python', python_role)
    
def matlab_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Custom role for MATLAB code snippets"""
    from docutils import nodes
    node = nodes.literal(rawtext, text, **options)
    node['classes'] = ['matlab-code']
    return [node], []

def python_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Custom role for Python code snippets"""
    from docutils import nodes
    node = nodes.literal(rawtext, text, **options)
    node['classes'] = ['python-code']
    return [node], []

# -- LaTeX output settings --------------------------------------------------

latex_engine = 'pdflatex'
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    'preamble': r'''
        \usepackage{amsmath}
        \usepackage{amsfonts}
        \usepackage{amssymb}
        \usepackage{bm}
        \usepackage{physics}
        \definecolor{berkeleyblue}{RGB}{0,50,98}
        \definecolor{californiagold}{RGB}{253,181,21}
    ''',
    'fncychap': '\\usepackage[Bjornstrup]{fncychap}',
    'printindex': '\\footnotesize\\raggedright\\printindex',
}

latex_documents = [
    (master_doc, 'SpinCirc.tex', 'SpinCirc Documentation',
     'Meshal Alawein', 'manual'),
]

# -- Epub output settings ---------------------------------------------------

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ['search.html']

# -- Custom directives ------------------------------------------------------

from sphinx.directives import SphinxDirective
from sphinx.util.docutils import docutils_namespace
from docutils import nodes
from docutils.parsers.rst import directives

class MATLABExample(SphinxDirective):
    """Directive for MATLAB code examples with output"""
    
    has_content = True
    required_arguments = 0
    optional_arguments = 1
    option_spec = {
        'linenos': directives.flag,
        'caption': directives.unchanged,
    }
    
    def run(self):
        code = '\n'.join(self.content)
        literal = nodes.literal_block(code, code)
        literal['language'] = 'matlab'
        literal['linenos'] = 'linenos' in self.options
        
        if 'caption' in self.options:
            caption = nodes.caption('', self.options['caption'])
            literal = nodes.container('', caption, literal)
            
        return [literal]

class PhysicsNote(SphinxDirective):
    """Directive for physics explanations"""
    
    has_content = True
    required_arguments = 0
    optional_arguments = 1
    
    def run(self):
        content = '\n'.join(self.content)
        note = nodes.admonition()
        note['classes'] = ['physics-note']
        
        title = nodes.title('', 'Physics Note')
        if self.arguments:
            title = nodes.title('', self.arguments[0])
            
        note += title
        
        from sphinx.parsers.rst import Parser
        from sphinx.util.docutils import new_document
        
        parser = Parser()
        parser.set_application(self.env.app)
        
        doc = new_document('<physics-note>')
        parser.parse(content, doc)
        
        note.extend(doc.children)
        return [note]

# Register custom directives
def setup_directives(app):
    app.add_directive('matlab-example', MATLABExample)
    app.add_directive('physics-note', PhysicsNote)
    
# -- Version information ----------------------------------------------------

# Get git information for version
try:
    import subprocess
    git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
    version_info = f"{version} ({git_branch}@{git_hash})"
except:
    version_info = version

html_context = {
    'display_github': True,
    'github_user': 'alawein',
    'github_repo': 'SpinCirc',
    'github_version': 'main',
    'conf_py_path': '/docs/',
    'version_info': version_info,
}

# -- Build hooks ------------------------------------------------------------

def setup_build_hooks(app):
    """Setup build hooks for documentation generation"""
    
    def generate_matlab_api(app, config):
        """Generate MATLAB API documentation"""
        import glob
        
        matlab_files = glob.glob('../matlab/**/*.m', recursive=True)
        
        # Generate API stubs for MATLAB files
        api_dir = '_matlab_api'
        os.makedirs(api_dir, exist_ok=True)
        
        for matlab_file in matlab_files:
            # Parse MATLAB file and generate RST
            pass  # Simplified for this example
    
    app.connect('config-inited', generate_matlab_api)

# Final setup
def complete_setup(app):
    setup_directives(app)
    setup_build_hooks(app)
    
setup = complete_setup