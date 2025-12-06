# linear_programming
**Module:** `Python/Optimization/linear_programming.py`
## Overview
Linear Programming Solvers
==========================
This module implements linear programming algorithms including the
simplex method, interior point methods, and revised simplex for
solving linear optimization problems.
Author: Berkeley SciComp Team
Date: 2024
## Constants
- **`BERKELEY_BLUE`**
- **`CALIFORNIA_GOLD`**
- **`BERKELEY_LIGHT_BLUE`**
## Functions
### `demo()`
Demonstrate linear programming solvers.
**Source:** [Line 625](Python/Optimization/linear_programming.py#L625)
## Classes
### `LinearProgram`
Represents a linear programming problem in standard form:
minimize    c^T x
subject to  A x = b
x >= 0
Attributes:
c: Objective function coefficients
A: Constraint matrix
b: Right-hand side vector
bounds: Variable bounds (optional)
sense: Optimization sense ('min' or 'max')
#### Methods
##### `__post_init__(self)`
Validate LP problem data.
**Source:** [Line 48](Python/Optimization/linear_programming.py#L48)
**Class Source:** [Line 27](Python/Optimization/linear_programming.py#L27)
### `LinearProgrammingSolver`
Abstract base class for linear programming solvers.
#### Methods
##### `__init__(self, max_iterations, tolerance, verbose)`
Initialize LP solver.
Args:
max_iterations: Maximum number of iterations
tolerance: Numerical tolerance
verbose: Whether to print progress
**Source:** [Line 67](Python/Optimization/linear_programming.py#L67)
##### `solve(self, lp)`
Solve linear programming problem.
Args:
lp: Linear programming problem
Returns:
OptimizationResult object
**Source:** [Line 82](Python/Optimization/linear_programming.py#L82)
**Class Source:** [Line 62](Python/Optimization/linear_programming.py#L62)
### `SimplexMethod`
Simplex method for linear programming.
Implements the two-phase simplex algorithm with Bland's pivoting
rule to prevent cycling.
#### Methods
##### `__init__(self, pivoting_rule)`
Initialize Simplex method.
Args:
pivoting_rule: Pivoting rule ('bland', 'dantzig', 'largest_improvement')
**Source:** [Line 102](Python/Optimization/linear_programming.py#L102)
##### `solve(self, lp)`
Solve LP using simplex method.
Args:
lp: Linear programming problem
Returns:
OptimizationResult
**Source:** [Line 112](Python/Optimization/linear_programming.py#L112)
##### `_two_phase_simplex(self, c, A, b)`
Two-phase simplex method.
**Source:** [Line 149](Python/Optimization/linear_programming.py#L149)
##### `_simplex_phase2(self, c, A, b)`
Single-phase simplex for problems already in standard form.
**Source:** [Line 218](Python/Optimization/linear_programming.py#L218)
##### `_simplex_solve(self, c, A, b, basic_vars)`
Core simplex algorithm.
**Source:** [Line 237](Python/Optimization/linear_programming.py#L237)
##### `_create_tableau(self, c, A, b)`
Create simplex tableau.
**Source:** [Line 294](Python/Optimization/linear_programming.py#L294)
##### `_compute_reduced_costs(self, tableau, basic_vars)`
Compute reduced costs.
**Source:** [Line 308](Python/Optimization/linear_programming.py#L308)
##### `_select_entering_variable(self, reduced_costs)`
Select entering variable using specified pivoting rule.
**Source:** [Line 312](Python/Optimization/linear_programming.py#L312)
##### `_ratio_test(self, tableau, entering_var)`
Perform ratio test to find leaving variable.
**Source:** [Line 335](Python/Optimization/linear_programming.py#L335)
##### `_pivot(self, tableau, pivot_row, pivot_col)`
Perform pivot operation.
**Source:** [Line 358](Python/Optimization/linear_programming.py#L358)
##### `_extract_solution(self, tableau, basic_vars, n)`
Extract solution from tableau.
**Source:** [Line 374](Python/Optimization/linear_programming.py#L374)
##### `_find_initial_basis(self, A)`
Find initial basis (simplified implementation).
**Source:** [Line 384](Python/Optimization/linear_programming.py#L384)
**Class Source:** [Line 94](Python/Optimization/linear_programming.py#L94)
### `InteriorPointLP`
Interior Point Method for Linear Programming.
Implements the primal-dual interior point method with barrier functions.
#### Methods
##### `__init__(self, barrier_parameter, centering_parameter)`
Initialize Interior Point method.
Args:
barrier_parameter: Initial barrier parameter
centering_parameter: Centering parameter for path following
**Source:** [Line 411](Python/Optimization/linear_programming.py#L411)
##### `solve(self, lp)`
Solve LP using interior point method.
Args:
lp: Linear programming problem
Returns:
OptimizationResult
**Source:** [Line 424](Python/Optimization/linear_programming.py#L424)
##### `_find_initial_point(self, c, A, b)`
Find initial feasible point.
**Source:** [Line 517](Python/Optimization/linear_programming.py#L517)
##### `_solve_newton_system(self, A, x, s, c, b, mu)`
Solve Newton system for interior point method.
**Source:** [Line 544](Python/Optimization/linear_programming.py#L544)
##### `_line_search_primal(self, x, s, dx, ds)`
Line search for primal variables.
**Source:** [Line 602](Python/Optimization/linear_programming.py#L602)
##### `_line_search_dual(self, y, dy)`
Line search for dual variables.
**Source:** [Line 618](Python/Optimization/linear_programming.py#L618)
**Class Source:** [Line 404](Python/Optimization/linear_programming.py#L404)