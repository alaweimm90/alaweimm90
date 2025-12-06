# Tests: Quality Assurance Suite

Comprehensive test suite for Librex.QAP-new with 149 passing tests.

## What Is This Directory?

The **tests/** directory contains the complete quality assurance suite. These are automated tests that validate the correctness of Librex.QAP and ORCHEX components.

**Quick Facts:**
- 149 total tests
- 6 test files
- ~2,100 lines of test code
- 40%+ overall coverage
- 91% critical module coverage
- All tests passing ✅

## Test Files

| File | Tests | Size | Purpose |
|------|-------|------|---------|
| `test_pipeline_exhaustive.py` | ~60 | 512 LOC | Pipeline validation |
| `test_methods.py` | ~40 | 477 LOC | Individual methods |
| `test_integration.py` | ~30 | 206 LOC | ORCHEX-Librex.QAP integration |
| `test_utils_core.py` | ~15 | 593 LOC | Core utilities |
| `test_benchmarks.py` | ~10 | 70 LOC | Benchmarking |
| `test_validation.py` | ~4 | 281 LOC | Validation framework |

**Total: 149 tests**

## Running Tests

### Quick Test Run

```bash
# Fast test (no coverage)
make test-fast

# Output:
# ✓ 149 tests passed in 2.3s
```

### Full Test Run with Coverage

```bash
# Full test with coverage report
make test

# Output:
# ✓ 149 tests passed
# ✓ 40%+ coverage
# ✓ Report: htmlcov/index.html
```

### Run Specific Test

```bash
# Run single file
pytest tests/test_methods.py -v

# Run single test
pytest tests/test_methods.py::test_fft_laplace -v

# Run with output (print statements)
pytest tests/ -v -s
```

### Verbose Output

```bash
# Detailed output with timing
make test-verbose

# Shows each test with timing
# test_pipeline_exhaustive.py::test_basic_pipeline PASSED [0.25s]
# test_methods.py::test_fft_laplace PASSED [0.50s]
# ...
```

## Test Coverage

### Current Coverage

```
Overall:         40%+
Critical:        91%
├── pipeline.py: 100%
├── methods/:    100%
├── core/:       95%
└── utils.py:    85%

Experimental:    20%
├── ORCHEX:       25%
└── Integration: 30%
```

### Generate Coverage Report

```bash
make coverage

# View report
open htmlcov/index.html
```

## Test Categories

### 1. Pipeline Tests (test_pipeline_exhaustive.py)

Tests the main optimization pipeline:

```python
def test_basic_pipeline():
    """Basic pipeline operation."""
    pipeline = OptimizationPipeline(size=10)
    result = pipeline.solve(problem, method="fft_laplace")
    assert result.best_solution is not None

def test_pipeline_with_different_sizes():
    """Test various problem sizes."""
    for size in [10, 20, 30]:
        pipeline = OptimizationPipeline(size=size)
        result = pipeline.solve(test_problem)
        assert result.objective_value > 0

def test_error_handling():
    """Test error conditions."""
    with pytest.raises(ValueError):
        OptimizationPipeline(size=-1)
```

**Key Tests:**
- Initialization with various sizes
- Method selection
- Results validity
- Edge cases and errors
- Performance monitoring

### 2. Method Tests (test_methods.py)

Tests individual optimization methods:

```python
def test_fft_laplace():
    """Test FFT-Laplace method."""
    result = fft_laplace_method(problem)
    assert result.objective_value < baseline_value

def test_reverse_time_escape():
    """Test reverse-time saddle escape."""
    result = reverse_time_method(problem)
    assert validates_solution(result)

def test_simulated_annealing():
    """Test baseline method."""
    result = simulated_annealing(problem)
    assert result is not None
```

**Key Tests:**
- Method correctness
- Solution validity
- Parameter sensitivity
- Convergence behavior
- Comparative performance

### 3. Integration Tests (test_integration.py)

Tests ORCHEX-Librex.QAP integration:

```python
def test_atlas_validates_method():
    """ORCHEX validates optimization result."""
    result = pipeline.solve(problem, method="fft_laplace")
    validation = orchestrator.validate(result)
    assert validation.is_valid

def test_learning_from_validation():
    """System learns from validation."""
    initial_score = agent.get_validation_score()
    orchestrator.validate_methods()
    new_score = agent.get_validation_score()
    assert new_score >= initial_score
```

**Key Tests:**
- Method-agent coordination
- Hypothesis validation
- Learning mechanisms
- Full workflow validation

### 4. Utility Tests (test_utils_core.py)

Tests core utility functions:

```python
def test_load_qap_instance():
    """Load benchmark instance."""
    problem = load_qap_instance("data/qaplib/nug20.dat")
    assert problem.size == 20
    assert problem.distance_matrix is not None

def test_evaluate_solution():
    """Evaluate solution quality."""
    objective = evaluate_solution(solution, problem)
    assert objective > 0

def test_validate_solution():
    """Check solution validity."""
    is_valid = validate_solution(solution, problem)
    assert is_valid  # Or raises exception
```

**Key Tests:**
- Problem loading
- Solution evaluation
- Validation checks
- Permutation utilities
- Caching mechanisms

### 5. Benchmark Tests (test_benchmarks.py)

Tests benchmarking infrastructure:

```python
def test_benchmark_all_methods():
    """Run benchmarks on all methods."""
    results = benchmark_suite.run_all(problems)
    assert len(results) == num_methods

def test_benchmark_timing():
    """Verify timing accuracy."""
    assert results.timing_data is not None
```

### 6. Validation Tests (test_validation.py)

Tests validation framework:

```python
def test_solution_validity():
    """Validate solution format."""
    assert validate_solution(solution, problem)

def test_constraint_checking():
    """Check problem constraints."""
    assert check_constraints(solution, problem)
```

## Test Structure

Each test follows a pattern:

```python
def test_something():
    """Test description."""
    # Arrange: Setup test data
    pipeline = OptimizationPipeline(size=20)
    problem = create_test_problem()

    # Act: Execute test
    result = pipeline.solve(problem, method="fft_laplace")

    # Assert: Verify results
    assert result.best_solution is not None
    assert result.objective_value > 0
    assert len(result.best_solution) == problem.size
```

## Writing New Tests

### Adding a Test

1. **Create test file** if needed:
   ```bash
   touch tests/test_my_feature.py
   ```

2. **Write test function**:
   ```python
   def test_my_feature():
       """Test description."""
       # Arrange
       obj = MyClass()

       # Act
       result = obj.my_method()

       # Assert
       assert result == expected
   ```

3. **Run your test**:
   ```bash
   pytest tests/test_my_feature.py -v
   ```

4. **Ensure coverage**:
   ```bash
   make coverage
   ```

### Test Naming Conventions

- **Test files:** `test_*.py`
- **Test functions:** `test_something()`
- **Test classes:** `TestSomething`
- **Test methods in classes:** `test_something()`

### Using Pytest Features

```python
import pytest

# Parametrize tests
@pytest.mark.parametrize("size", [10, 20, 30])
def test_various_sizes(size):
    pipeline = OptimizationPipeline(size=size)
    # Test logic

# Skip tests
@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    pass

# Mark as slow
@pytest.mark.slow
def test_long_running():
    pass

# Expect exceptions
def test_error():
    with pytest.raises(ValueError):
        bad_function()

# Fixtures for setup/teardown
@pytest.fixture
def test_problem():
    return create_test_problem()

def test_with_fixture(test_problem):
    assert test_problem is not None
```

## Test Execution

### Before Committing

```bash
# Run all checks
make check-all

# This runs:
# 1. Format check
# 2. Linters
# 3. All tests
```

### During Development

```bash
# Quick feedback
make test-fast

# Detailed output
make test-verbose

# Specific test
pytest tests/test_methods.py::test_fft_laplace -v
```

### With Coverage

```bash
# Generate coverage
make coverage

# View detailed report
open htmlcov/index.html
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example: GitHub Actions
- name: Run tests
  run: make test

- name: Check coverage
  run: pytest --cov
```

## Troubleshooting Tests

### Tests failing locally?

```bash
# Verify environment
pip install -e ".[dev]"

# Run with verbose output
pytest tests/ -v -s

# Check imports
python -c "import Librex.QAP; import ORCHEX"
```

### Specific test failing?

```bash
# Run just that test
pytest tests/test_methods.py::test_fft_laplace -v -s

# Show local variables on failure
pytest --tb=long tests/test_methods.py

# Drop into debugger on failure
pytest --pdb tests/test_methods.py
```

### Coverage not meeting goals?

```bash
# Check what's not covered
make coverage
open htmlcov/index.html  # See red areas

# Add tests for uncovered code
# Run coverage again
```

## Test Quality Metrics

### Pass Rate
- Current: **100%** (149/149 tests passing)
- Goal: **100%** (maintain)

### Coverage
- Current: **40%+ overall, 91% critical**
- Goal: **50%+ overall, 95% critical**

### Test Performance
- Total execution time: ~2-3 seconds
- Slowest test: ~1 second
- Fastest test: ~10ms

## Testing Best Practices

✅ **DO:**
- Write tests for new code
- Use descriptive test names
- Test both happy and sad paths
- Use fixtures for setup
- Keep tests independent
- Aim for high coverage of critical code

❌ **DON'T:**
- Skip tests
- Write tests without assertions
- Make tests depend on order
- Use sleep() for synchronization
- Test implementation details
- Write tests that are flaky

## Related Documentation

- **DEVELOPMENT.md** - Development workflow
- **CONTRIBUTING.md** - Contribution guidelines
- **PROJECT.md** - Project overview
- **Makefile** - Testing commands

## Quick Reference

### Common Commands

```bash
make test              # Run all tests with coverage
make test-fast         # Quick test run
make test-verbose      # Detailed output
make coverage          # Generate coverage report
pytest -v             # Run with verbose output
pytest -s             # Show print statements
pytest --pdb          # Drop into debugger
```

### Test Statistics

```
Total Tests:       149 ✅
Test Files:        6
Lines of Tests:    ~2,100
Coverage:          40%+ overall, 91% critical
Pass Rate:         100%
Execution Time:    ~2-3 seconds
```

---

**Happy testing!** 🚀

Questions? Check `CONTRIBUTING.md` for how to write better tests.

Last Updated: November 2024
