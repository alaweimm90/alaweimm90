# Contributing to the Governance System

Thank you for your interest in contributing to the alaweimm90 GitHub Governance System!

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Contribution Types](#contribution-types)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

This project follows a standard code of conduct. Be respectful, inclusive, and constructive in all interactions.

## How to Contribute

### Reporting Issues

1. Check existing issues to avoid duplicates
2. Use the appropriate issue template
3. Provide clear reproduction steps
4. Include relevant logs or screenshots

### Suggesting Enhancements

1. Open a GitHub issue with the `enhancement` label
2. Describe the use case and expected behavior
3. Consider backward compatibility

## Development Setup

### Prerequisites

- Python 3.11+
- Git
- pre-commit (`pip install pre-commit`)

### Setup Steps

```bash
# Clone the repository
git clone https://github.com/alaweimm90/GitHub.git
cd GitHub

# Install Python dependencies
pip install -r .metaHub/scripts/requirements.txt
pip install pytest pytest-cov ruff mypy

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
```

### Running Governance Scripts

```bash
# Enforcement
python .metaHub/scripts/enforce.py ./organizations/my-org/

# Catalog generation
python .metaHub/scripts/catalog.py

# Meta auditor
python .metaHub/scripts/meta.py scan-projects
```

## Contribution Types

### 1. Policy Changes (`.metaHub/policies/`)

OPA/Rego policy changes require:

- Clear documentation of the policy intent
- Test cases demonstrating the policy behavior
- Consideration of backward compatibility
- Review by a maintainer

**Example policy structure:**
```rego
package my_policy

# Deny rule with clear message
deny[msg] {
    # condition
    msg := "Clear explanation of violation"
}

# Warn rule for non-blocking issues
warn[msg] {
    # condition
    msg := "Suggestion for improvement"
}
```

### 2. Schema Changes (`.metaHub/schemas/`)

Schema changes to `repo-schema.json` require:

- Backward compatibility (new fields should be optional)
- Migration guide for existing repositories
- Updated documentation
- Validation tests

### 3. Template Changes (`.metaHub/templates/`)

Template changes require:

- Testing with multiple project types
- Documentation updates
- Consideration of all supported languages

### 4. Script Changes (`.metaHub/scripts/`)

Python script changes require:

- Type hints for all functions
- Docstrings for public functions
- Unit tests with >80% coverage
- No breaking changes to CLI interfaces

### 5. Workflow Changes (`.github/workflows/`)

Workflow changes require:

- Explicit permissions block
- Pinned action versions
- Testing in a fork first
- Security review for sensitive operations

## Pull Request Process

### Before Submitting

1. **Run tests locally**: `pytest tests/ -v`
2. **Run linting**: `ruff check .metaHub/scripts/ scripts/`
3. **Run pre-commit**: `pre-commit run --all-files`
4. **Update documentation** if needed

### PR Requirements

- Clear title following conventional commits (`feat:`, `fix:`, `docs:`, etc.)
- Description of changes and motivation
- Link to related issues
- All CI checks passing

### Review Process

1. Automated checks must pass
2. At least one maintainer approval required
3. All comments must be resolved
4. Squash merge preferred

## Style Guidelines

### Python

- Use `ruff` for linting and formatting
- Type hints required for all functions
- Docstrings in Google style
- Maximum line length: 100 characters


```python
def my_function(param: str, optional: int = 0) -> Dict[str, Any]:
    """Brief description.

    Args:
        param: Description of param.
        optional: Description of optional param.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param is invalid.
    """
    pass
```

### YAML

- 2-space indentation
- No trailing whitespace
- Explicit quotes for strings that could be misinterpreted

### Markdown

- ATX-style headers (`#`)
- Fenced code blocks with language specifier
- One sentence per line (for better diffs)

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(policies): add kubernetes pod security policy`
- `fix(enforce): handle missing metadata gracefully`
- `docs(readme): update installation instructions`

## Questions?

- Open a GitHub issue for general questions
- Tag maintainers for urgent matters
- Check existing documentation first

Thank you for contributing!
