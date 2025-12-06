GOLDEN_STRUCTURE = {
    'src': {
        '{product_name}': {
            '__init__.py': '"""Package initialization."""\n\nfrom .main import main\n\n__version__ = "0.1.0"\n',
            'main.py': '# Main module - rename from original',
            'models.py': '# Data models (dataclasses)',
            'utils.py': '# Utility functions',
            'constants.py': '# Constants and configuration'
        }
    },
    'tests': {
        '__init__.py': '',
        'test_{product_name}.py': '# Unit tests'
    },
    'examples': {
        'README.md': '# Examples\n\nHow to use this product.\n',
        'example_basic.py': '# Basic usage example'
    },
    'docs': {
        'API.md': '# API Reference\n',
        'CHANGELOG.md': '# Changelog\n\n## [0.1.0] - {date}\n- Initial release\n'
    },
    'README.md': '# {product_title}\n\n{description}\n',
    'pyproject.toml': None,
    'LICENSE': 'MIT License\n',
    '.gitignore': '*.pyc\n__pycache__/\n*.json\n.venv/\n',
}

NAMING_RULES = {
    'files': {'pattern': r'^[a-z][a-z0-9_]*\.py$'},
    'classes': {'pattern': r'^[A-Z][a-zA-Z0-9]*$'},
    'functions': {'pattern': r'^[a-z][a-z0-9_]*$'},
    'constants': {'pattern': r'^[A-Z][A-Z0-9_]*$'},
    'private': {'pattern': r'^_[a-z][a-z0-9_]*$'},
}

CODE_STANDARDS = {
    'line_length': 100,
    'indent': 4,
    'docstring_style': 'google',
    'type_hints': True,
    'imports_order': ['stdlib', 'third_party', 'local'],
    'blank_lines': {'top_level': 2, 'method': 1},
}

