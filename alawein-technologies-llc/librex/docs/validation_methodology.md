# Librex Validation Methodology

## Overview

The Librex framework employs a comprehensive validation methodology to ensure the correctness, performance, and reliability of all optimization algorithms. This document describes our validation approach, test suite, and continuous quality assurance processes.

## Validation Philosophy

Our validation approach is based on four pillars:

1. **Correctness**: Algorithms find optimal or near-optimal solutions for problems with known optima
2. **Performance**: Algorithms converge efficiently within reasonable time and resource constraints
3. **Robustness**: Algorithms perform consistently across different problem types and dimensions
4. **Reproducibility**: Results can be reproduced given the same inputs and random seeds

## Test Problem Suite

### Analytical Test Problems (28+ Functions)

We validate against problems with known analytical solutions, organized by categories:

#### Convex Functions
- **Sphere Function**: f(x) = Σx²ᵢ
  - Global optimum: x* = 0, f(x*) = 0
  - Tests basic convergence

- **Convex Quadratic**: f(x) = ½x^T Q x + b^T x + c
  - Analytical solution: x* = -Q⁻¹b
  - Tests handling of quadratic forms

#### Unimodal Functions
- **Rosenbrock**: Tests navigation of narrow valleys
- **Zakharov**: Tests handling of coupled variables
- **Dixon-Price**: Tests asymmetric landscapes

#### Multimodal Functions
- **Rastrigin**: Highly multimodal with regular structure
- **Ackley**: Multimodal with flat regions
- **Griewank**: Product of cosines with many local minima
- **Schwefel**: Deceptive global structure
- **Levy**: Complex multimodal landscape

#### 2D Test Functions (14 problems)
Special functions for detailed visualization and analysis:
- Beale, Goldstein-Price, Booth, Matyas
- Himmelblau (4 global minima)
- Easom (sharp global minimum)
- Cross-in-tray, Eggholder, Holder Table
- McCormick, Schaffer N.2/N.4
- Three-hump/Six-hump Camel

### Benchmark Collections

#### QAPLIB (138 instances)
- Quadratic Assignment Problems
- Sizes: 12 to 256 facilities
- Known best solutions available

#### TSPLIB
- Traveling Salesman Problems
- Various sizes and structures

#### CEC Benchmarks
- Competition benchmarks
- Shifted, rotated, and hybrid functions

## Validation Framework Components

### 1. Test Problem Module (`test_problems.py`)

```python
class TestProblem:
    name: str
    dimension: int
    bounds: Tuple[float, float]
    optimal_solution: np.ndarray
    optimal_value: float

    def evaluate(self, x: np.ndarray) -> float
```

### 2. Validator Module (`validator.py`)

```python
class OptimizationValidator:
    def validate_method(method, dimensions, problems) -> Dict
    def compare_methods(methods) -> pd.DataFrame
    def generate_report(results, output_path) -> None
```

### 3. Continuous Validation (`continuous_validation.py`)

```python
class ContinuousValidator:
    def run_validation_suite(level) -> Dict
    def run_regression_test(baseline) -> bool
    def generate_benchmark_report() -> None
```

## Validation Metrics

### Primary Metrics

1. **Success Rate**: Percentage of runs achieving target accuracy
   - Target: > 70% for standard problems
   - Target: > 90% for convex problems

2. **Optimality Gap**: |f(x) - f(x*)|
   - Absolute tolerance: 1e-3
   - Relative tolerance: 1%

3. **Solution Quality**: ||x - x*||₂
   - Tolerance: 1e-2 (problem-dependent)

### Secondary Metrics

4. **Convergence Speed**: Iterations to reach tolerance
5. **Function Evaluations**: Total objective function calls
6. **Runtime**: Wall-clock time
7. **Memory Usage**: Peak memory consumption

## Validation Levels

### Quick Validation (CI on every commit)
- 3 methods × 2 dimensions × 10 problems = 60 tests
- Runtime: ~2 minutes
- Success threshold: 70%

### Standard Validation (CI on PR)
- 5 methods × 3 dimensions × 20 problems = 300 tests
- Runtime: ~10 minutes
- Success threshold: 75%

### Comprehensive Validation (Release)
- 10 methods × 4 dimensions × all problems = 1000+ tests
- Runtime: ~1 hour
- Success threshold: 80%

### Nightly Validation
- All methods × 5 dimensions × all problems
- Multiple runs per configuration
- Statistical significance testing
- Runtime: ~4 hours

## Statistical Validation

### Performance Comparison
- Wilcoxon signed-rank test for paired comparisons
- Kruskal-Wallis test for multiple methods
- Effect size calculation (Cohen's d)

### Reliability Testing
- Bootstrap confidence intervals
- Cross-validation across problem sets
- Sensitivity analysis for parameters

## Continuous Integration

### GitHub Actions Workflow

```yaml
validation:
  - quick: on every commit
  - standard: on pull request
  - comprehensive: on release tag
  - nightly: scheduled at 2 AM UTC
```

### Regression Detection

```python
if current_performance < baseline_performance - 0.05:
    raise RegressionError("Performance degraded")
```

## Validation Reports

### HTML Report Contents
1. Overall success rates
2. Method comparison charts
3. Problem difficulty ranking
4. Convergence plots
5. Performance heatmaps
6. Detailed result tables

### Report Locations
- CI artifacts: Available for 30 days
- Release reports: Permanently archived
- Nightly reports: Rolling 7-day window

## Quality Gates

### Method Acceptance Criteria
- Success rate ≥ 70% on standard test suite
- No regression vs baseline
- Passes unit tests
- Documentation complete

### Release Criteria
- All methods pass validation
- Overall success rate ≥ 80%
- Benchmarks show no regression
- Test coverage ≥ 95%

## Known Limitations

### Test Coverage Gaps
- Limited stochastic optimization testing
- Constrained optimization needs expansion
- Multi-objective problems underrepresented

### Future Improvements
- Add noisy function tests
- Implement dynamic problem tests
- Expand constraint handling tests
- Add real-world problem instances

## Validation Best Practices

### For Contributors
1. Run quick validation before committing
2. Add tests for new methods
3. Update baselines when improving performance
4. Document any validation exceptions

### For Users
1. Check validation reports for method selection
2. Run custom validation for specific problems
3. Report unexpected failures
4. Contribute new test problems

## References

1. Jamil, M., & Yang, X. S. (2013). A literature survey of benchmark functions for global optimization problems.
2. Suganthan, P. N., et al. (2005). Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
3. Wolpert, D. H., & Macready, W. G. (1997). No free lunch theorems for optimization.

---

**Document Version**: 1.0.0
**Last Updated**: November 2024
**Contact**: Librex@alawein.com