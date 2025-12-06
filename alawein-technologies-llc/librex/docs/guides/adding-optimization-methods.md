# How to Add a New Optimization Method

**Goal:** Add a custom optimization method to Librex.QAP and have it ready for benchmarking

**Time Required:** 45 minutes (including testing)

**Prerequisites:**
- Python 3.9+
- Understanding of numpy and QAP concepts
- Familiarity with the optimization pipeline

---

## Overview

This guide walks you through adding a new optimization method to Librex.QAP that integrates seamlessly with the existing framework. You'll implement the algorithm, test it, benchmark it, and document it.

## Prerequisites Check

```bash
# Verify your environment
python -c "from Librex.QAP.core import OptimizationPipeline; print('✓ Ready')"
pip list | grep numpy  # Should be installed
```

---

## Step 1: Understand the Method Interface

All optimization methods follow this interface:

```python
def my_method(problem, iterations=1000, **params):
    """Brief description.

    Args:
        problem: QAPProblem with distance_matrix and flow_matrix
        iterations: Number of iterations to run
        **params: Method-specific parameters

    Returns:
        OptimizationResult with best_solution and objective_value
    """
    # Implementation
    return OptimizationResult(...)
```

**Key Requirements:**
- Accept a `problem` parameter (QAPProblem object)
- Return `OptimizationResult` with solution and objective value
- Handle iterations parameter
- Include comprehensive docstring
- Be deterministic (same seed = same result)

---

## Step 2: Implement Your Method

### Option A: Novel Method (High Impact)

**File:** `Librex.QAP/methods/novel.py`

```python
import numpy as np
from Librex.QAP.utils import evaluate_solution, validate_solution
from Librex.QAP.core.pipeline import OptimizationResult


def your_novel_method(problem, iterations=1000, learning_rate=0.1,
                      momentum=0.9, **params):
    """Your Novel Optimization Method.

    This method implements [Algorithm Name] which combines [Key Features].

    Algorithm Overview:
    1. Initialize population with random solutions
    2. Compute gradients using [technique]
    3. Update solutions with momentum-based approach
    4. Apply local optimization
    5. Track best solution

    Mathematical Foundation:
    - Cost function: f(x) = x^T * D * x (QAP formulation)
    - Update rule: x(t+1) = x(t) - α*∇f(x) + β*v(t)
    - Where α=learning_rate, β=momentum

    Time Complexity: O(n² * iterations)
    Space Complexity: O(n²)

    References:
    - Your Citation Here (Year)
    - Related Work (Year)

    Args:
        problem: QAPProblem instance
        iterations: Number of iterations (default: 1000)
        learning_rate: Gradient descent step size (default: 0.1)
        momentum: Momentum coefficient (default: 0.9)
        **params: Additional parameters (random_seed, verbose, etc.)

    Returns:
        OptimizationResult containing:
            - best_solution: Best permutation found
            - objective_value: Objective value of best solution
            - iterations: Actual iterations performed
            - convergence_history: Objective values over iterations
            - method_name: Name of this method

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If optimization fails

    Example:
        >>> problem = load_qap_instance("data/qaplib/nug20.dat")
        >>> result = your_novel_method(problem, iterations=500,
        ...                             learning_rate=0.05)
        >>> print(f"Best: {result.objective_value}")
        >>> print(f"Solution: {result.best_solution[:5]}")  # First 5 elements
    """
    n = problem.size
    D = problem.distance_matrix
    F = problem.flow_matrix

    # Set random seed for reproducibility
    seed = params.get('random_seed', None)
    if seed is not None:
        np.random.seed(seed)

    # Initialize
    best_solution = np.random.permutation(n)
    best_objective = evaluate_solution(best_solution, problem)

    # Algorithm state
    velocity = np.zeros(n)  # For momentum
    convergence_history = [best_objective]

    verbose = params.get('verbose', False)

    try:
        for iteration in range(iterations):
            # Generate candidate solution
            candidate = _generate_candidate(problem, best_solution,
                                           learning_rate, momentum, velocity)

            # Evaluate
            objective = evaluate_solution(candidate, problem)

            # Update best if better
            if objective < best_objective:
                best_solution = candidate.copy()
                best_objective = objective
                if verbose:
                    print(f"Iteration {iteration}: {best_objective:.2f} ✓")

            # Update velocity for momentum
            velocity = momentum * velocity + (1 - momentum) * (candidate - best_solution)

            convergence_history.append(best_objective)

        # Verify solution validity
        if not validate_solution(best_solution, problem):
            raise RuntimeError("Final solution is invalid")

        result = OptimizationResult(
            best_solution=best_solution,
            objective_value=best_objective,
            iterations=iterations,
            convergence_history=convergence_history,
            method_name="your_novel_method",
            metadata={
                "learning_rate": learning_rate,
                "momentum": momentum,
                "final_convergence_rate": (
                    convergence_history[-10:] if len(convergence_history) > 10
                    else convergence_history
                )
            }
        )

        return result

    except Exception as e:
        # Provide helpful error message
        raise RuntimeError(
            f"Optimization failed at iteration {iteration}: {str(e)}"
        ) from e


def _generate_candidate(problem, current, learning_rate, momentum, velocity):
    """Helper: Generate next candidate solution.

    Uses your specific algorithm logic.
    """
    n = problem.size

    # Your algorithm-specific logic here
    candidate = current.copy().astype(float)

    # Apply gradient-like update
    candidate = candidate - learning_rate * velocity

    # Convert back to valid permutation
    candidate = np.argsort(candidate)

    return candidate
```

### Option B: Baseline Method (For Comparison)

**File:** `Librex.QAP/methods/baselines.py`

```python
def your_baseline_method(problem, iterations=1000, **params):
    """Baseline Method: Your Algorithm.

    A classical approach using [technique].

    Args:
        problem: QAPProblem instance
        iterations: Number of iterations
        **params: Additional parameters

    Returns:
        OptimizationResult
    """
    n = problem.size
    best_solution = np.random.permutation(n)
    best_objective = evaluate_solution(best_solution, problem)

    for iteration in range(iterations):
        # Simple local search
        candidate = best_solution.copy()

        # Random swap
        i, j = np.random.choice(n, 2, replace=False)
        candidate[[i, j]] = candidate[[j, i]]

        objective = evaluate_solution(candidate, problem)

        if objective < best_objective:
            best_solution = candidate
            best_objective = objective

    return OptimizationResult(
        best_solution=best_solution,
        objective_value=best_objective,
        iterations=iterations,
        method_name="your_baseline_method"
    )
```

---

## Step 3: Register Your Method

**File:** `Librex.QAP/methods/metadata.py`

Add to the `METHODS` dictionary:

```python
METHODS["your_novel_method"] = {
    "type": "novel",  # or "baseline"
    "class": "Novel Methods" if "novel" else "Baseline Algorithms",
    "version": "1.0",
    "author": "Your Name",
    "date_added": "2024-11-19",

    "description": "Brief one-sentence description",
    "long_description": """
    Extended description of method.
    Explain the algorithm, key innovations, and use cases.
    """,

    "parameters": {
        "iterations": {
            "type": int,
            "default": 1000,
            "range": (100, 10000),
            "description": "Number of optimization iterations"
        },
        "learning_rate": {
            "type": float,
            "default": 0.1,
            "range": (0.001, 1.0),
            "description": "Learning rate for gradient descent"
        },
        "momentum": {
            "type": float,
            "default": 0.9,
            "range": (0.0, 1.0),
            "description": "Momentum coefficient"
        }
    },

    "complexity": {
        "time": "O(n² * iterations)",
        "space": "O(n²)",
        "description": "Linear with respect to problem size and iterations"
    },

    "performance": {
        "expected_speedup": 2.5,  # Relative to simulated annealing
        "scalability": "good",  # excellent/good/fair/poor
        "quality": "high",  # excellent/high/fair/low
        "stability": "stable"  # stable/variable
    },

    "references": [
        "Author et al., 2023, Journal Name, 'Paper Title'",
        "Your Citation Style Here"
    ],

    "notes": "Any special notes about implementation or usage",
    "status": "validated",  # experimental/validated/production
    "tested_on": ["nug20.dat", "tai30a.dat", "tai40a.dat"]
}
```

---

## Step 4: Write Comprehensive Tests

**File:** `tests/test_methods.py`

Add your tests:

```python
@pytest.fixture
def nug20_problem():
    """Load nug20 instance for testing."""
    return load_qap_instance("data/qaplib/nug20.dat")


class TestYourNovelMethod:
    """Test suite for your_novel_method."""

    def test_basic_operation(self, nug20_problem):
        """Test basic method operation."""
        result = your_novel_method(nug20_problem, iterations=10)

        assert result is not None
        assert result.best_solution is not None
        assert result.objective_value > 0
        assert result.iterations == 10

    def test_solution_validity(self, nug20_problem):
        """Test that returned solution is valid."""
        result = your_novel_method(nug20_problem, iterations=100)

        # Check size
        assert len(result.best_solution) == nug20_problem.size

        # Check permutation (all values 0 to n-1, no duplicates)
        assert set(result.best_solution) == set(range(nug20_problem.size))
        assert len(set(result.best_solution)) == len(result.best_solution)

        # Validate solution
        assert validate_solution(result.best_solution, nug20_problem)

    @pytest.mark.parametrize("size", [5, 10, 20])
    def test_various_sizes(self, size):
        """Test with different problem sizes."""
        problem = create_test_qap_problem(size=size)
        result = your_novel_method(problem, iterations=50)

        assert result.best_solution is not None
        assert len(result.best_solution) == size

    @pytest.mark.parametrize("iterations", [10, 50, 100, 500])
    def test_parameter_sensitivity(self, nug20_problem, iterations):
        """Test with different iteration counts."""
        result = your_novel_method(nug20_problem, iterations=iterations)

        assert result.iterations == iterations
        # More iterations should generally give better results

    def test_reproducibility(self, nug20_problem):
        """Test that same seed produces same result."""
        result1 = your_novel_method(nug20_problem, iterations=100,
                                    random_seed=42)
        result2 = your_novel_method(nug20_problem, iterations=100,
                                    random_seed=42)

        assert np.array_equal(result1.best_solution, result2.best_solution)
        assert result1.objective_value == result2.objective_value

    def test_convergence_history(self, nug20_problem):
        """Test that convergence history is tracked."""
        result = your_novel_method(nug20_problem, iterations=100)

        assert hasattr(result, 'convergence_history')
        assert len(result.convergence_history) > 0
        # Should be decreasing or stable (not increasing randomly)

    def test_parameter_variations(self, nug20_problem):
        """Test with different parameter values."""
        result1 = your_novel_method(nug20_problem, iterations=100,
                                    learning_rate=0.1)
        result2 = your_novel_method(nug20_problem, iterations=100,
                                    learning_rate=0.5)

        # Both should produce valid results
        assert result1.best_solution is not None
        assert result2.best_solution is not None
        assert result1.objective_value > 0
        assert result2.objective_value > 0

    def test_error_handling(self, nug20_problem):
        """Test error handling."""
        # Invalid iterations
        with pytest.raises(ValueError):
            your_novel_method(nug20_problem, iterations=-1)

        # Invalid learning rate
        with pytest.raises(ValueError):
            your_novel_method(nug20_problem, learning_rate=2.0)

    @pytest.mark.slow
    def test_performance(self, nug20_problem):
        """Test performance on larger problem."""
        import time

        start = time.time()
        result = your_novel_method(nug20_problem, iterations=1000)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 10.0  # Max 10 seconds
        assert result.best_solution is not None


class TestYourNovelMethodComparison:
    """Comparative tests against baselines."""

    def test_versus_baseline(self, nug20_problem):
        """Compare against simulated annealing."""
        novel_result = your_novel_method(nug20_problem, iterations=100)
        baseline_result = simulated_annealing(nug20_problem, iterations=100)

        # Novel method should be competitive or better
        # (May not always be true on all instances)
        assert novel_result.objective_value > 0
        assert baseline_result.objective_value > 0
```

---

## Step 5: Add Example Usage

**File:** `examples/05_optimization.py` (add section)

```python
# Example: Your Novel Method
print("\n" + "="*60)
print("Example: Your Novel Method")
print("="*60)

from Librex.QAP.methods import your_novel_method

problem = load_qap_instance("data/qaplib/nug20.dat")

# Run optimization
result = your_novel_method(
    problem,
    iterations=500,
    learning_rate=0.1,
    momentum=0.9,
    random_seed=42,
    verbose=True
)

# Display results
optimal = 2570  # Known optimal for nug20.dat
gap = (result.objective_value - optimal) / optimal * 100

print(f"\nResults:")
print(f"  Best Solution: {result.best_solution}")
print(f"  Objective Value: {result.objective_value:.2f}")
print(f"  Known Optimal: {optimal}")
print(f"  Gap: {gap:.2f}%")
print(f"  Iterations: {result.iterations}")

# Plot convergence
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(result.convergence_history, linewidth=2)
plt.axhline(y=optimal, color='r', linestyle='--', label='Known Optimal')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.title('Your Novel Method - Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Step 6: Update Documentation

### Update Changelog

**File:** `CHANGELOG.md`

```markdown
### Added
- New optimization method: Your Novel Method
  - Implementation in `Librex.QAP/methods/novel.py`
  - ~2.5x speedup over simulated annealing on medium instances
  - Comprehensive tests with 8 test cases (100% passing)
  - Example usage in `examples/05_optimization.py`
  - See `docs/guides/adding-optimization-methods.md` for details
```

### Update Method List

**File:** `Librex.QAP/README.md`

Add to "Novel Methods" section:

```markdown
- `your_novel_method` - Your Novel Method
  - O(n² * iterations) complexity
  - 2.5x faster than baseline
  - Excellent on medium instances (n=20-40)
  - Paper: [Citation]
```

---

## Step 7: Benchmark Your Method

```bash
# Run benchmarks
pytest tests/test_methods.py::TestYourNovelMethod -v

# Run performance test
pytest tests/test_methods.py::TestYourNovelMethodComparison -v

# Full method test
pytest tests/test_methods.py -k "your_novel_method" -v
```

---

## Step 8: Create How-To Guide (Optional)

If your method is unique, create `docs/guides/your-method-guide.md`:

```markdown
# Guide: Your Novel Method

## Overview
[Detailed explanation]

## Mathematical Foundation
[Equations and theory]

## Implementation Strategy
[How it's implemented]

## Tuning for Best Results
[Parameter guidance]

## When to Use
[Best use cases]

## Limitations
[What it doesn't do well]

## References
[Citations]
```

---

## Verification Checklist

Before committing, verify:

- [ ] Implementation is complete and working
- [ ] Method signature matches interface
- [ ] Docstrings are comprehensive (150+ lines)
- [ ] Registered in `metadata.py`
- [ ] All tests pass: `pytest tests/test_methods.py::Test<YourMethod> -v`
- [ ] Convergence history is tracked
- [ ] Solution validity is verified
- [ ] Example runs without errors
- [ ] CHANGELOG.md is updated
- [ ] README.md is updated
- [ ] Code follows PEP 8: `black your_file.py`
- [ ] No type errors: `mypy Librex.QAP/methods/novel.py`
- [ ] Full test suite passes: `make check-all`

---

## Final Steps

### 1. Run All Checks

```bash
# Format code
black Librex.QAP/methods/novel.py

# Check types
mypy Librex.QAP/methods/novel.py

# Run tests
pytest tests/test_methods.py -v

# Full validation
make check-all
```

### 2. Commit

```bash
git checkout -b feature/your-novel-method
git add Librex.QAP/methods/novel.py tests/test_methods.py examples/05_optimization.py
git commit -m "Add: Your Novel Method for QAP optimization

- O(n² * iterations) time complexity
- 2.5x faster than simulated annealing
- Comprehensive test suite (8 tests, 100% passing)
- Example usage and convergence visualization
- Full documentation with 150+ line docstrings
- See docs/guides/adding-optimization-methods.md"

git push origin feature/your-novel-method
```

### 3. Optional: Create Pull Request

```bash
# Create PR on GitHub for review
gh pr create --title "Add Your Novel Method" \
             --body "Implements novel optimization using [technique]"
```

---

## Troubleshooting

### Solution Not Improving

```python
# Check learning rate
result1 = your_novel_method(problem, learning_rate=0.01)
result2 = your_novel_method(problem, learning_rate=0.5)

# Visualize to understand behavior
plt.plot(result.convergence_history)
plt.show()
```

### Tests Failing

```bash
# Run with verbose output
pytest tests/test_methods.py::TestYourNovelMethod -vv -s

# Drop into debugger
pytest tests/test_methods.py::TestYourNovelMethod --pdb
```

### Solution Validity Issues

```python
# Debug solution format
result = your_novel_method(problem, iterations=10)
print(f"Type: {type(result.best_solution)}")
print(f"Shape: {result.best_solution.shape}")
print(f"Values: {sorted(result.best_solution)}")
print(f"Expected: {list(range(problem.size))}")
```

---

## Next Steps

1. **Compare Performance** - Run benchmarks against other methods
2. **Optimize Hyperparameters** - Tune learning_rate and momentum
3. **Write Research Paper** - Document and publish findings
4. **Contribute** - Submit PR to project
5. **Extend** - Add variants or hybrid approaches

---

## Related Guides

- See `CONTRIBUTING.md` for broader contribution workflow
- See `EXPANSION_TEMPLATES.md` Template 1 for quick reference
- See `.archive/docs/Librex.QAP/FORMULA_REFERENCES.md` for QAP theory

---

**Happy implementing!** Your method is now ready for production use, testing, and optimization. 🚀

Last Updated: November 2024
