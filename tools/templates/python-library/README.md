# Python Library Template

A production-ready Python library template with modern tooling.

## Features

- ✅ `pyproject.toml` with Poetry/pip support
- ✅ pytest with coverage reporting
- ✅ mypy strict type checking
- ✅ ruff linting and formatting
- ✅ GitHub Actions CI/CD
- ✅ Sphinx documentation ready
- ✅ PyPI publishing workflow

## Usage

```bash
# Copy template
cp -r templates/python-library my-new-library
cd my-new-library

# Replace placeholders
sed -i 's/{{PROJECT_NAME}}/my-new-library/g' **/*
sed -i 's/{{AUTHOR}}/Meshal Alawein/g' **/*

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest --cov
```

## Structure

```
my-library/
├── src/
│   └── my_library/
│       ├── __init__.py
│       └── core.py
├── tests/
│   ├── __init__.py
│   └── test_core.py
├── docs/
│   ├── conf.py
│   └── index.rst
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
├── README.md
└── LICENSE
```

