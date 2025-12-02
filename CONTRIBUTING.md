# Contributing

<img src="https://img.shields.io/badge/PRs-Welcome-A855F7?style=flat-square&labelColor=1a1b27" alt="PRs Welcome"/>
<img src="https://img.shields.io/badge/Code-Python_3.11+-EC4899?style=flat-square&labelColor=1a1b27" alt="Python"/>
<img src="https://img.shields.io/badge/Style-Ruff-4CC9F0?style=flat-square&labelColor=1a1b27" alt="Ruff"/>

---

> Thank you for your interest in contributing to the governance system.

## Quick Links

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Contribution Types](#contribution-types)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

---

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

---

## Development Setup

### Prerequisites

- Python 3.11+
- Git
- pre-commit

### Setup

```bash
# Clone
git clone https://github.com/alaweimm90/alaweimm90.git
cd alaweimm90

# Install dependencies
pip install -r .metaHub/scripts/requirements.txt
pip install pytest pytest-cov ruff mypy

# Install hooks
pre-commit install

# Run tests
pytest tests/ -v
```

### Run Scripts

```bash
python .metaHub/scripts/enforce.py ./organizations/my-org/   # Enforcement
python .metaHub/scripts/catalog.py                           # Catalog
python .metaHub/scripts/meta.py scan-projects                # Audit
```

---

## Contribution Types

### Policies (`.metaHub/policies/`)

- Clear documentation of policy intent
- Test cases demonstrating behavior
- Backward compatibility consideration
- Maintainer review required

### Schemas (`.metaHub/schemas/`)

- New fields should be optional (backward compatible)
- Migration guide for existing repos
- Validation tests required

### Scripts (`.metaHub/scripts/`)

- Type hints for all functions
- Docstrings for public functions
- Unit tests with >80% coverage
- No breaking CLI changes

### Workflows (`.github/workflows/`)

- Explicit permissions block
- Pinned action versions
- Test in fork first
- Security review for sensitive operations

---

## Pull Request Process

### Before Submitting

```bash
pytest tests/ -v                    # Run tests
ruff check .metaHub/scripts/        # Lint
pre-commit run --all-files          # Pre-commit
```

### Requirements

| Requirement | Description                                           |
| ----------- | ----------------------------------------------------- |
| Title       | Conventional commit format (`feat:`, `fix:`, `docs:`) |
| Description | Changes and motivation                                |
| Issues      | Link to related issues                                |
| CI          | All checks passing                                    |

### Review

1. Automated checks must pass
2. One maintainer approval required
3. All comments resolved
4. Squash merge preferred

---

## Style Guidelines

### Python

```python
def my_function(param: str, optional: int = 0) -> Dict[str, Any]:
    """Brief description.

    Args:
        param: Description.
        optional: Description.

    Returns:
        Description.
    """
    pass
```

- Use `ruff` for linting/formatting
- Type hints required
- Google-style docstrings
- Max line length: 100

### Commit Messages

```
<type>(<scope>): <description>

[optional body]
```

| Type       | Description      |
| ---------- | ---------------- |
| `feat`     | New feature      |
| `fix`      | Bug fix          |
| `docs`     | Documentation    |
| `refactor` | Code restructure |
| `test`     | Tests            |
| `chore`    | Maintenance      |

**Examples:**

- `feat(policies): add kubernetes pod security policy`
- `fix(enforce): handle missing metadata gracefully`
- `docs(readme): update installation instructions`

---

## Questions?

- Open a GitHub issue
- Check existing documentation first

---

**Maintainer:** [@alaweimm90](https://github.com/alaweimm90)
