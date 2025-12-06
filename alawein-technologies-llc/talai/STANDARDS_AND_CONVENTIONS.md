# IDEAS Product Suite - Standards and Conventions

**Version:** 1.0
**Last Updated:** 2025-11-16
**Status:** Active - All new products must follow these standards

This document defines the golden standards for all products in the IDEAS suite. These conventions ensure consistency, maintainability, and quality across the entire codebase.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Naming Conventions](#naming-conventions)
3. [Code Style](#code-style)
4. [Documentation Standards](#documentation-standards)
5. [Testing Requirements](#testing-requirements)
6. [Refactoring Agents](#refactoring-agents)
7. [Quality Metrics](#quality-metrics)

---

## Repository Structure

### Golden Template

Every product MUST follow this structure:

```
product-name/
├── src/
│   └── product_name/          # Package directory (underscores, not hyphens)
│       ├── __init__.py        # Package initialization
│       ├── main.py            # Main entry point
│       ├── models.py          # Data models (dataclasses)
│       ├── utils.py           # Utility functions
│       └── constants.py       # Constants and configuration
│
├── tests/
│   ├── __init__.py
│   └── test_product_name.py  # Unit tests
│
├── examples/
│   ├── README.md              # Usage examples
│   └── example_basic.py       # Basic usage example
│
├── docs/
│   ├── API.md                 # API reference
│   └── CHANGELOG.md           # Version history
│
├── README.md                  # Main documentation
├── pyproject.toml             # Build configuration
├── LICENSE                    # MIT License
└── .gitignore                 # Git ignore rules
```

### Directory Naming

- **Product directories:** Use hyphens (e.g., `failure-db`, `research-pricer`)
- **Python packages:** Use underscores (e.g., `failure_db`, `research_pricer`)
- **All lowercase:** No capital letters in directory names

### File Organization

**src/product_name/main.py:**
- Contains CLI interface
- Imports from other modules
- Main entry point: `def main():`

**src/product_name/models.py:**
- All dataclass definitions
- Type definitions
- Data structures

**src/product_name/utils.py:**
- Helper functions
- Shared utilities
- No business logic

**src/product_name/constants.py:**
- Configuration values
- Magic numbers
- Domain knowledge (like SCIENTISTS dict in ghost-researcher)

---

## Naming Conventions

### Files

**Pattern:** `^[a-z][a-z0-9_]*\.py$`

✅ **Good:**
- `main.py`
- `models.py`
- `test_example.py`
- `data_loader.py`

❌ **Bad:**
- `Main.py` (capitalized)
- `testExample.py` (camelCase)
- `data-loader.py` (hyphens)

### Classes

**Pattern:** `^[A-Z][a-zA-Z0-9]*$` (PascalCase)

✅ **Good:**
```python
class FailureDB:
    pass

class PowerAnalysis:
    pass

class ROIPrediction:
    pass
```

❌ **Bad:**
```python
class failureDB:       # lowercase start
class power_analysis:  # snake_case
class Roi_Prediction:  # mixed
```

### Functions and Methods

**Pattern:** `^[a-z][a-z0-9_]*$` (snake_case)

✅ **Good:**
```python
def calculate_roi():
    pass

def generate_protocol():
    pass

def consult_scientist():
    pass
```

❌ **Bad:**
```python
def calculateROI():        # camelCase
def GenerateProtocol():    # PascalCase
def Consult_Scientist():   # mixed
```

### Constants

**Pattern:** `^[A-Z][A-Z0-9_]*$` (SCREAMING_SNAKE_CASE)

✅ **Good:**
```python
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
API_VERSION = "1.0"
```

❌ **Bad:**
```python
max_retries = 3        # lowercase
MaxRetries = 3         # PascalCase
Max_Retries = 3        # mixed
```

### Private Members

**Pattern:** `^_[a-z][a-z0-9_]*$` (leading underscore)

✅ **Good:**
```python
def _internal_helper():
    pass

_cache = {}
```

---

## Code Style

### General Standards

- **Line length:** Max 100 characters
- **Indentation:** 4 spaces (no tabs)
- **Encoding:** UTF-8
- **Line endings:** LF (Unix-style)

### Imports

**Order:**
1. Standard library
2. Third-party packages
3. Local imports

**Format:**
```python
# Standard library
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Third-party (if any)
import numpy as np

# Local
from .models import Failure, PredictionMarket
from .utils import validate_input
```

### Type Hints

**Required** for all public functions:

```python
def calculate_roi(
    funding: float,
    duration: int,
    team_size: int
) -> ROIPrediction:
    """Calculate ROI for research proposal."""
    pass
```

**Optional** but recommended for private functions:

```python
def _estimate_publications(proposal: ResearchProposal) -> float:
    pass
```

### Docstrings

**Required** for:
- All public classes
- All public functions
- All modules

**Style:** Google-style docstrings

```python
def calculate_power(effect_size: float, alpha: float = 0.05) -> PowerAnalysis:
    """Calculate statistical power and sample size.

    Args:
        effect_size: Cohen's d effect size (0.2 small, 0.5 medium, 0.8 large)
        alpha: Significance level (default: 0.05)

    Returns:
        PowerAnalysis object with sample size and assumptions

    Raises:
        ValueError: If effect_size is not positive

    Examples:
        >>> power = calculate_power(0.5)
        >>> print(power.required_sample_size)
        128
    """
    if effect_size <= 0:
        raise ValueError("Effect size must be positive")

    # Implementation
    pass
```

### Dataclasses

**Use dataclasses** for all data structures:

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Collision:
    """A domain collision."""
    collision_id: int
    source_domain: str
    target_domain: str
    novelty_score: float
    feasibility_score: float
    applications: List[str]
    created_at: str
```

### Error Handling

**Be specific** with exceptions:

```python
# Good
if proposal_id not in self.proposals:
    raise ValueError(f"Proposal {proposal_id} not found")

# Bad
if proposal_id not in self.proposals:
    raise Exception("Not found")
```

---

## Documentation Standards

### README.md Structure

Every product README must have:

1. **Title and Description** (1-2 sentences)
2. **Concept** (Problem + Solution)
3. **Features** (3-5 bullet points)
4. **Usage** (Code examples with actual commands)
5. **Output Examples** (Real output from the tool)
6. **Use Cases** (Who would use this and why)
7. **Pricing Model** (Revenue strategy)
8. **Build Info** (Time, LOC, credit, status)
9. **Future Enhancements** (Roadmap)

**Template:**

```markdown
# ProductName - One Line Description

Brief overview of what it does (2-3 sentences).

## Concept

**The Problem:** Clear problem statement

**The Solution:** How this tool solves it

## Features

- Feature 1
- Feature 2
- Feature 3

## Usage

### Basic Command

```bash
python product.py command --arg value
```

### Advanced Example

```bash
python product.py advanced --option
```

## Output Examples

```
[Actual output from the tool]
```

## Use Cases

### For Researchers
- Use case 1
- Use case 2

### For Students
- Use case 1
- Use case 2

## Pricing Model

- Free tier: X
- Pro: $Y/month
- Enterprise: $Z/month

## Build Info

- Build time: X hours
- Credit used: ~$Y
- Lines of code: Z
- Status: [Functional prototype | Production ready]

## Future Enhancements

- Enhancement 1
- Enhancement 2
```

### API Documentation

**docs/API.md** should document:

- All public classes
- All public functions
- Parameters and return types
- Usage examples

### CHANGELOG.md

Follow semantic versioning:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-11-16

### Added
- Initial release
- Feature X
- Feature Y

### Changed
- Improvement Z

### Fixed
- Bug fix W
```

---

## Testing Requirements

### Minimum Requirements

Every product MUST have:

1. **tests/** directory
2. **test_product_name.py** file
3. At least 3 basic tests

### Test Structure

```python
import pytest
from product_name.main import ProductClass

def test_basic_functionality():
    """Test basic feature works."""
    obj = ProductClass()
    result = obj.method()
    assert result is not None

def test_error_handling():
    """Test error cases are handled."""
    obj = ProductClass()
    with pytest.raises(ValueError):
        obj.method(invalid_input)

def test_example_workflow():
    """Test complete workflow from README examples."""
    # Simulate the exact workflow from README
    pass
```

### Running Tests

```bash
# From product root
pytest tests/

# With coverage
pytest --cov=src/product_name tests/
```

---

## Refactoring Agents

### Built-in Agents

The IDEAS suite includes automated refactoring agents:

**StructureAgent:**
- Ensures golden template structure
- Creates missing directories
- Generates pyproject.toml

**CodeStyleAgent:**
- Fixes import order
- Adds type hints (where possible)
- Checks docstrings

**DocAgent:**
- Consolidates documentation
- Creates missing docs
- Generates API.md

**NamingAgent:**
- Checks naming conventions
- Reports violations
- Suggests fixes

**QualityAgent:**
- Measures quality metrics
- Generates score (0-100)
- Tracks improvements

**ConsolidationAgent:**
- Removes clutter files
- Merges duplicates
- Cleans up

### Using Agents

**Refactor single product:**
```bash
python refactor_agents.py --product failure-db
```

**Refactor all products:**
```bash
python refactor_agents.py --all --report
```

**Check quality only:**
```bash
python refactor_agents.py --product failure-db --quality
```

---

## Quality Metrics

### Score Breakdown

**Overall Score = weighted average:**
- Structure Compliance: 30%
- Docstring Coverage: 25%
- Type Hint Coverage: 20%
- Naming Compliance: 25%

### Target Scores

- **Minimum (MVP):** 60/100
- **Good (Production):** 80/100
- **Excellent:** 90+/100

### Current Status (Session 2 Products)

| Product | LOC | Files | Score | Status |
|---------|-----|-------|-------|--------|
| failure_db | 1,297 | 4 | 81/100 | ✅ Good |
| research_pricer | 1,155 | 4 | 81/100 | ✅ Good |
| experiment_designer | 1,809 | 4 | 81/100 | ✅ Good |
| chaos_engine | 1,049 | 4 | 81/100 | ✅ Good |
| ghost_researcher | 1,255 | 4 | 81/100 | ✅ Good |

All products meet "Good" production quality threshold!

---

## pyproject.toml Template

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "product-name"
version = "0.1.0"
description = "Part of IDEAS research tools suite"
authors = [
    {name = "IDEAS Team"}
]
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

[project.scripts]
product-name = "product_name.main:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"

[tool.black]
line-length = 100
target-version = ['py38']

[tool.isort]
profile = "black"
line_length = 100
```

---

## Git Ignore Template

```gitignore
# Python
*.pyc
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
env/

# Data files
*.json
*.csv
*.sqlite

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Build
dist/
build/
*.egg-info/
```

---

## Enforcement

### Automatic Checks

Run before every commit:

```bash
# Check style
python refactor_agents.py --product my-product

# Run tests
cd my-product && pytest

# Check naming
python refactor_agents.py --product my-product --naming
```

### Pre-commit Hook

**Recommended .git/hooks/pre-commit:**

```bash
#!/bin/bash
# Run quality checks before commit

python refactor_agents.py --changed --quick

if [ $? -ne 0 ]; then
    echo "❌ Quality checks failed"
    exit 1
fi

echo "✅ Quality checks passed"
```

---

## Migration Guide

### Migrating Existing Products

For products that don't follow these standards:

**Step 1: Run Structure Agent**
```bash
python refactor_agents.py --product old-product
```

**Step 2: Manual Cleanup**
- Move code to src/product_name/
- Split large files into main.py, models.py, utils.py
- Update imports

**Step 3: Add Documentation**
- Create missing docs/
- Add docstrings
- Update README to match template

**Step 4: Add Tests**
- Create tests/ directory
- Write basic tests
- Verify examples work

**Step 5: Quality Check**
```bash
python refactor_agents.py --product old-product --report
```

Target: 80/100 or higher

---

## Best Practices

### DO

✅ Use dataclasses for data structures
✅ Type hint all public functions
✅ Write docstrings for public APIs
✅ Keep functions under 50 lines
✅ Use descriptive variable names
✅ Follow the golden template
✅ Run refactor agents regularly
✅ Write tests for new features
✅ Update CHANGELOG.md

### DON'T

❌ Mix tabs and spaces
❌ Use global variables
❌ Write functions over 100 lines
❌ Skip type hints
❌ Ignore naming conventions
❌ Commit without testing
❌ Create files without docstrings
❌ Use magic numbers (use constants)

---

## FAQ

**Q: Do all 17 products follow these standards?**
A: Session 2 products (5 newest) now follow them. Session 1 products (12 older) are being migrated.

**Q: Can I use a different structure for special cases?**
A: Only with documented justification. Consistency > cleverness.

**Q: What if my product needs additional directories?**
A: Add them, but keep the golden template structure as base.

**Q: How often should I run refactor agents?**
A: After major changes, before releases, and weekly during active development.

**Q: What if the agents break my code?**
A: Agents create backups and report changes. Review reports before accepting.

---

## Version History

### 1.0 (2025-11-16)
- Initial standards document
- Golden template defined
- 6 refactoring agents created
- Quality metrics established
- All Session 2 products refactored to standards

---

**Status:** ✅ Active and Enforced

All new products MUST follow these standards.
All existing products SHOULD be migrated by end of Q1 2025.

**For questions or suggestions:** Submit issue or update this document.
