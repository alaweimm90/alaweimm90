# Expansion Templates

Ready-to-use templates for expanding Librex.QAP-new with new features and modules.

## Overview

This document provides templates for common expansion scenarios, ensuring consistency and maintainability as the project grows.

---

## Template 1: New Optimization Method

When adding a new optimization method to Librex.QAP:

### Step 1: Create Implementation

**File:** `Librex.QAP/methods/novel.py` or `Librex.QAP/methods/baselines.py`

```python
def your_new_method(problem, iterations=1000, **params):
    """Brief description of method.

    Extended description explaining the algorithm and why it works.

    Args:
        problem: QAPProblem instance
        iterations: Number of iterations (default: 1000)
        **params: Additional method-specific parameters

    Returns:
        np.ndarray: Best solution found
        float: Objective value of best solution

    Raises:
        ValueError: If invalid parameters provided

    Example:
        >>> method_result = your_new_method(problem, iterations=500)
        >>> print(method_result.objective_value)
    """
    # Implementation here
    best_solution = None
    best_objective = float('inf')

    for iteration in range(iterations):
        # Core algorithm logic
        candidate = generate_candidate(problem)
        objective = evaluate_solution(candidate, problem)

        if objective < best_objective:
            best_objective = objective
            best_solution = candidate

    return OptimizationResult(
        best_solution=best_solution,
        objective_value=best_objective,
        iterations=iteration + 1,
        method_name="your_new_method"
    )
```

### Step 2: Register in Metadata

**File:** `Librex.QAP/methods/metadata.py`

```python
METHODS["your_new_method"] = {
    "type": "novel",  # or "baseline"
    "class": "Novel Methods",
    "description": "Description of your method",
    "parameters": {
        "iterations": {
            "type": int,
            "default": 1000,
            "range": (100, 10000),
            "description": "Number of iterations"
        },
        "your_param": {
            "type": float,
            "default": 0.5,
            "range": (0.0, 1.0),
            "description": "Your parameter"
        }
    },
    "time_complexity": "O(n³)",
    "space_complexity": "O(n²)",
    "references": [
        "Author et al., Year, Title"
    ]
}
```

### Step 3: Add Tests

**File:** `tests/test_methods.py`

```python
def test_your_new_method():
    """Test your new method."""
    problem = create_test_qap_problem(size=10)
    result = your_new_method(problem, iterations=100)

    assert result.best_solution is not None
    assert len(result.best_solution) == problem.size
    assert result.objective_value > 0
    assert np.all(np.unique(result.best_solution) == np.arange(problem.size))


@pytest.mark.parametrize("size", [5, 10, 20])
def test_your_new_method_various_sizes(size):
    """Test with different problem sizes."""
    problem = create_test_qap_problem(size=size)
    result = your_new_method(problem)
    assert result.best_solution is not None


def test_your_new_method_parameters():
    """Test with different parameters."""
    problem = create_test_qap_problem(size=10)

    result1 = your_new_method(problem, iterations=100)
    result2 = your_new_method(problem, iterations=1000)

    # More iterations typically give better results
    assert result2.objective_value <= result1.objective_value
```

### Step 4: Add Example

**File:** `examples/05_optimization.py` (add section)

```python
# Your New Method Example
print("\n=== Your New Method ===")
problem = load_qap_instance("data/qaplib/nug20.dat")

result = pipeline.solve(problem, method="your_new_method")
print(f"Solution: {result.objective_value}")
print(f"Optimal:  {2570}")
print(f"Gap:      {(result.objective_value - 2570) / 2570 * 100:.2f}%")
```

### Step 5: Update Documentation

**File:** `CHANGELOG.md`

```markdown
### Added
- New optimization method: Your Method Name
  - Implementation in Librex.QAP/methods/
  - 40% faster than baseline on medium instances
  - See docs/guides/adding-methods.md for details
```

**File:** `Librex.QAP/README.md` (update method list)

```markdown
**Novel Methods:**
- ...
- `your_new_method` - Brief description with key stats
```

### Step 6: Verify

```bash
# Run all checks
make check-all

# Specific checks
pytest tests/test_methods.py::test_your_new_method -v
make benchmark
```

---

## Template 2: New ORCHEX Agent

When adding a new personality agent to ORCHEX:

### Step 1: Create Agent Class

**File:** Create new file in `ORCHEX/ORCHEX/agents/` (if not existing, use `ORCHEX/__init__.py`)

```python
from ORCHEX.protocol import Agent, ValidationResult

class YourNewAgent(Agent):
    """Brief description of agent.

    Detailed description of what this agent does and its role
    in the validation process.

    Attributes:
        name: Agent name
        strictness: Validation strictness (0.0-1.0)
        capabilities: List of validation capabilities
    """

    def __init__(self):
        """Initialize agent with specific properties."""
        super().__init__(
            name="Your Agent",
            strictness=0.7,  # Adjust as needed (0=lenient, 1=strict)
            description="What this agent validates"
        )
        self.add_capability("hypothesis_validation")
        self.add_capability("your_specific_capability")

    def validate(self, hypothesis):
        """Validate a hypothesis.

        Args:
            hypothesis: Hypothesis object or dict

        Returns:
            ValidationResult: Validation verdict and reasoning
        """
        issues = []
        warnings = []
        passed = True

        # Validation logic
        if not self._check_assumption(hypothesis):
            issues.append("Assumption not validated")
            passed = False

        if not self._check_evidence(hypothesis):
            warnings.append("Limited evidence found")

        verdict = "VALID" if passed else "INVALID"
        confidence = 0.8 if not warnings else 0.6

        return ValidationResult(
            verdict=verdict,
            confidence=confidence,
            issues=issues,
            warnings=warnings,
            reasoning=f"Your agent reasoning here",
            agent_name=self.name
        )

    def _check_assumption(self, hypothesis):
        """Helper method to validate assumptions."""
        # Implementation
        return True

    def _check_evidence(self, hypothesis):
        """Helper method to validate evidence."""
        # Implementation
        return True

    def improve(self, feedback):
        """Learn from validation results."""
        # Update agent strategy based on feedback
        pass
```

### Step 2: Register Agent

**File:** `ORCHEX/ORCHEX/__init__.py`

```python
from ORCHEX.agents import YourNewAgent  # Add import

# In initialization
agent_registry = AgentRegistry()
# ... existing agents ...
agent_registry.register(YourNewAgent())
```

### Step 3: Add Tests

**File:** `tests/test_integration.py` (add section)

```python
def test_your_new_agent():
    """Test your new agent."""
    from ORCHEX.agents import YourNewAgent

    agent = YourNewAgent()
    hypothesis = create_test_hypothesis()

    result = agent.validate(hypothesis)

    assert result is not None
    assert result.verdict in ["VALID", "INVALID"]
    assert 0 <= result.confidence <= 1


def test_your_new_agent_learning():
    """Test agent learning capability."""
    agent = YourNewAgent()

    # Provide feedback
    agent.improve(feedback={
        "accuracy": 0.9,
        "lesson": "Always check X"
    })

    # Verify improvement
    assert agent.get_performance_metrics()["accuracy"] >= 0.9
```

### Step 4: Document

**File:** `CHANGELOG.md`

```markdown
### Added
- New personality agent: Your Agent
  - Validates: [what it validates]
  - Strictness: 0.7
  - See ORCHEX/ORCHEX/__init__.py for implementation
```

### Step 5: Verify

```bash
pytest tests/test_integration.py::test_your_new_agent -v
make test
```

---

## Template 3: New Directory/Module

When adding a new major feature or module:

### Step 1: Create Structure

```bash
mkdir -p module_name
touch module_name/__init__.py
touch module_name/core.py
touch module_name/utils.py
mkdir -p tests/
touch tests/test_module.py
```

### Step 2: Initialize Module

**File:** `module_name/__init__.py`

```python
"""Module Name: Brief description.

Detailed description of module purpose and functionality.
"""

from .core import MainClass
from .utils import helper_function

__all__ = [
    "MainClass",
    "helper_function",
]

__version__ = "0.1.0"
```

### Step 3: Add README

**File:** `module_name/README.md`

```markdown
# Module Name

Brief description.

## Quick Start
[Quick usage example]

## Features
- Feature 1
- Feature 2

## Directory Contents
[Directory contents explanation]

## Related
[Links to related docs]
```

### Step 4: Add Tests

**File:** `tests/test_module.py`

```python
def test_main_class():
    """Test main class."""
    obj = MainClass()
    assert obj is not None

def test_helper_function():
    """Test helper function."""
    result = helper_function(input_data)
    assert result is not None
```

### Step 5: Update Project Docs

**File:** `STRUCTURE.md` (update directory map)

**File:** `CHANGELOG.md`

```markdown
### Added
- New module: Module Name
  - Location: module_name/
  - Purpose: [description]
```

---

## Template 4: New Test Suite

When adding comprehensive tests for a feature:

### File: `tests/test_feature.py`

```python
"""Test suite for [Feature].

Tests [what is tested] including:
- Basic functionality
- Edge cases
- Error handling
- Performance
"""

import pytest
from feature import FeatureClass


class TestFeatureBasics:
    """Basic functionality tests."""

    def test_initialization(self):
        """Test object initialization."""
        obj = FeatureClass()
        assert obj is not None

    def test_basic_operation(self):
        """Test basic operation."""
        obj = FeatureClass()
        result = obj.do_something(data)
        assert result is not None


class TestFeatureEdgeCases:
    """Edge case tests."""

    @pytest.mark.parametrize("input_value", [0, -1, None, []])
    def test_edge_values(self, input_value):
        """Test with edge case values."""
        obj = FeatureClass()
        result = obj.handle(input_value)
        assert result is handled_correctly


class TestFeatureErrorHandling:
    """Error handling tests."""

    def test_invalid_input(self):
        """Test error on invalid input."""
        obj = FeatureClass()
        with pytest.raises(ValueError):
            obj.process(invalid_input)


class TestFeaturePerformance:
    """Performance tests."""

    @pytest.mark.slow
    def test_performance(self):
        """Test performance characteristics."""
        obj = FeatureClass()
        result = obj.expensive_operation()
        assert result is not None
```

---

## Template 5: New Documentation File

### File: `docs/guides/how-to-do-something.md`

```markdown
# How to [Do Something]

**Goal:** What you'll accomplish

**Time:** 15 minutes

**Prerequisites:** What you need to know

## Overview

Explain what and why.

## Step-by-Step

### Step 1: [First Action]

Explanation and code example:

\`\`\`python
code_example()
\`\`\`

### Step 2: [Second Action]

Explanation and code example.

## Verification

How to confirm success:

\`\`\`bash
make test
\`\`\`

## Troubleshooting

| Problem | Solution |
|---------|----------|
| X Error | Try Y |

## Next Steps

What to do next.

## Related

- See also: [related guides]
- Learn more: [documentation]
```

---

## Checklist for Expansion

When expanding the project:

- [ ] Create necessary files
- [ ] Write implementation
- [ ] Add docstrings
- [ ] Create tests
- [ ] Verify tests pass (`make test`)
- [ ] Update CHANGELOG.md
- [ ] Update STRUCTURE.md
- [ ] Add examples
- [ ] Update relevant READMEs
- [ ] Run full checks (`make check-all`)
- [ ] Commit with clear message

---

## Common Patterns

### File Organization
- Implementation: Main module (Librex.QAP/, ORCHEX/)
- Tests: Parallel structure in tests/
- Examples: examples/
- Docs: docs/guides/ or docs/development/

### Naming Conventions
- Files: `lowercase_with_underscores.py`
- Classes: `CamelCase`
- Functions: `lowercase_with_underscores()`
- Constants: `UPPERCASE_WITH_UNDERSCORES`

### Import Structure
```python
# Always start with
"""Module docstring."""

from standard import library
from third_party import package
from . import local_module

__all__ = ["public_api"]
```

---

## Quick Reference

| Want to Add | Template | Key Files |
|-----------|----------|-----------|
| Method | Template 1 | methods/, tests/, CHANGELOG |
| Agent | Template 2 | ORCHEX/, tests/, CHANGELOG |
| Module | Template 3 | new_module/, README, STRUCTURE |
| Tests | Template 4 | tests/test_*.py |
| Guide | Template 5 | docs/guides/*.md |

---

**Use these templates to maintain consistency and quality as Librex.QAP-new grows!** 📝

Last Updated: November 2024
