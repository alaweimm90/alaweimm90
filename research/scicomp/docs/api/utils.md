# utils
**Module:** `Python/Optimization/utils.py`
## Overview
Optimization Utilities and Benchmark Functions
==============================================
This module provides utility functions, benchmark test problems,
and analysis tools for optimization algorithms.
Author: Berkeley SciComp Team
Date: 2024
## Constants
- **`BERKELEY_BLUE`**
- **`CALIFORNIA_GOLD`**
- **`BERKELEY_LIGHT_BLUE`**
## Functions
### `benchmark_algorithm(algorithm, problems, n_runs, max_iterations)`
Benchmark an optimization algorithm on multiple test problems.
Args:
algorithm: Optimization algorithm to test
problems: List of optimization problems
n_runs: Number of runs per problem
max_iterations: Maximum iterations per run
Returns:
List of benchmark results
**Source:** [Line 495](Python/Optimization/utils.py#L495)
### `demo()`
Demonstrate optimization utilities and benchmark functions.
**Source:** [Line 590](Python/Optimization/utils.py#L590)
## Classes
### `BenchmarkResult`
Results from benchmark testing.
Attributes:
function_name: Name of test function
algorithm_name: Name of optimization algorithm
result: OptimizationResult object
success_rate: Success rate over multiple runs
avg_iterations: Average number of iterations
avg_function_evaluations: Average number of function evaluations
avg_execution_time: Average execution time
best_value_found: Best function value found
convergence_data: Convergence history data
**Class Source:** [Line 24](Python/Optimization/utils.py#L24)
### `OptimizationProblem`
Represents an optimization problem with objective, constraints, and bounds.
This class provides a standardized way to define optimization problems
and evaluate different algorithms on them.
#### Methods
##### `__init__(self, name, objective, gradient, hessian, bounds, constraints, global_minimum, optimal_point, dimension)`
Initialize optimization problem.
Args:
name: Problem name
objective: Objective function
gradient: Gradient function (optional)
hessian: Hessian function (optional)
bounds: Variable bounds
constraints: List of constraints
global_minimum: Known global minimum value
optimal_point: Known optimal point
dimension: Problem dimension
**Source:** [Line 57](Python/Optimization/utils.py#L57)
##### `evaluate(self, x)`
Evaluate objective function.
**Source:** [Line 85](Python/Optimization/utils.py#L85)
##### `evaluate_gradient(self, x)`
Evaluate gradient if available.
**Source:** [Line 89](Python/Optimization/utils.py#L89)
##### `evaluate_hessian(self, x)`
Evaluate Hessian if available.
**Source:** [Line 93](Python/Optimization/utils.py#L93)
##### `is_feasible(self, x)`
Check if point is feasible.
**Source:** [Line 97](Python/Optimization/utils.py#L97)
##### `distance_to_optimum(self, x)`
Calculate distance to known optimum.
**Source:** [Line 115](Python/Optimization/utils.py#L115)
**Class Source:** [Line 49](Python/Optimization/utils.py#L49)
### `BenchmarkFunctions`
Collection of standard benchmark functions for optimization testing.
Includes unimodal, multimodal, and separable/non-separable test functions
commonly used in optimization literature.
#### Methods
##### `sphere(dimension)`
Sphere function: f(x) = sum(x_i^2)
Global minimum: f(0) = 0
Properties: Unimodal, separable, convex
**Source:** [Line 131](Python/Optimization/utils.py#L131)
##### `rosenbrock(dimension)`
Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
Global minimum: f(1) = 0
Properties: Unimodal, non-separable, valley-shaped
**Source:** [Line 162](Python/Optimization/utils.py#L162)
##### `rastrigin(dimension)`
Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
Global minimum: f(0) = 0
Properties: Multimodal, separable, many local minima
**Source:** [Line 192](Python/Optimization/utils.py#L192)
##### `ackley(dimension)`
Ackley function: f(x) = -a*exp(-b*sqrt(sum(x_i^2)/n)) - exp(sum(cos(c*x_i))/n) + a + exp(1)
Global minimum: f(0) = 0
Properties: Multimodal, non-separable, many local minima
**Source:** [Line 221](Python/Optimization/utils.py#L221)
##### `griewank(dimension)`
Griewank function: f(x) = sum(x_i^2)/4000 - prod(cos(x_i/sqrt(i+1))) + 1
Global minimum: f(0) = 0
Properties: Multimodal, non-separable
**Source:** [Line 267](Python/Optimization/utils.py#L267)
##### `schwefel(dimension)`
Schwefel function: f(x) = 418.9829*n - sum(x_i * sin(sqrt(|x_i|)))
Global minimum: f(420.9687) ≈ 0
Properties: Multimodal, separable
**Source:** [Line 312](Python/Optimization/utils.py#L312)
##### `get_all_functions(dimension)`
Get all benchmark functions.
**Source:** [Line 344](Python/Optimization/utils.py#L344)
**Class Source:** [Line 122](Python/Optimization/utils.py#L122)
### `ConvergenceAnalysis`
Tools for analyzing convergence behavior of optimization algorithms.
Provides methods to track and analyze convergence properties,
including convergence rate estimation and performance metrics.
#### Methods
##### `analyze_convergence(fitness_history, true_optimum, tolerance)`
Analyze convergence behavior from fitness history.
Args:
fitness_history: List of function values over iterations
true_optimum: Known optimal value (optional)
tolerance: Convergence tolerance
Returns:
Dictionary with convergence analysis results
**Source:** [Line 364](Python/Optimization/utils.py#L364)
##### `compare_algorithms(results)`
Compare multiple algorithm results.
Args:
results: List of BenchmarkResult objects
Returns:
Comparison analysis
**Source:** [Line 438](Python/Optimization/utils.py#L438)
**Class Source:** [Line 355](Python/Optimization/utils.py#L355)