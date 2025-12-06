# unconstrained
**Module:** `Python/Optimization/unconstrained.py`
## Overview
Unconstrained Optimization Algorithms
====================================
This module implements various unconstrained optimization algorithms
including gradient-based methods, Newton-type methods, and quasi-Newton
approaches with Berkeley SciComp framework integration.
Author: Berkeley SciComp Team
Date: 2024
## Constants
- **`BERKELEY_BLUE`**
- **`CALIFORNIA_GOLD`**
- **`BERKELEY_LIGHT_BLUE`**
## Functions
### `demo()`
Demonstrate unconstrained optimization algorithms.
**Source:** [Line 825](Python/Optimization/unconstrained.py#L825)
## Classes
### `OptimizationResult`
Represents the result of an optimization algorithm.
Attributes:
x: Solution vector
fun: Function value at solution
grad: Gradient at solution (if available)
hess: Hessian at solution (if available)
nit: Number of iterations
nfev: Number of function evaluations
ngev: Number of gradient evaluations
nhev: Number of Hessian evaluations
success: Whether optimization was successful
message: Description of termination condition
execution_time: Time taken for optimization
path: Optimization path (if tracked)
**Class Source:** [Line 26](Python/Optimization/unconstrained.py#L26)
### `UnconstrainedOptimizer`
Abstract base class for unconstrained optimization algorithms.
This class provides a common interface for all unconstrained
optimization methods in the Berkeley SciComp framework.
#### Methods
##### `__init__(self, max_iterations, tolerance, track_path, verbose)`
Initialize optimizer.
Args:
max_iterations: Maximum number of iterations
tolerance: Convergence tolerance
track_path: Whether to track optimization path
verbose: Whether to print progress information
**Source:** [Line 65](Python/Optimization/unconstrained.py#L65)
##### `minimize(self, objective, x0, gradient, hessian)`
Minimize the objective function.
Args:
objective: Objective function to minimize
x0: Initial guess
gradient: Gradient function (optional)
hessian: Hessian function (optional)
**kwargs: Additional algorithm-specific parameters
Returns:
OptimizationResult object
**Source:** [Line 87](Python/Optimization/unconstrained.py#L87)
##### `_evaluate_function(self, func, x)`
Evaluate function with counter increment.
**Source:** [Line 106](Python/Optimization/unconstrained.py#L106)
##### `_evaluate_gradient(self, grad_func, objective, x, h)`
Evaluate gradient (analytical or numerical).
**Source:** [Line 111](Python/Optimization/unconstrained.py#L111)
##### `_evaluate_hessian(self, hess_func, grad_func, objective, x, h)`
Evaluate Hessian (analytical or numerical).
**Source:** [Line 130](Python/Optimization/unconstrained.py#L130)
**Class Source:** [Line 57](Python/Optimization/unconstrained.py#L57)
### `GradientDescent`
Gradient Descent optimization algorithm.
Implements steepest descent with various line search strategies
and adaptive step size control.
#### Methods
##### `__init__(self, learning_rate, line_search, c1, rho, max_line_search)`
Initialize Gradient Descent optimizer.
Args:
learning_rate: Initial learning rate
line_search: Line search method ('fixed', 'backtracking', 'armijo')
c1: Armijo condition parameter
rho: Backtracking reduction factor
max_line_search: Maximum line search iterations
**Source:** [Line 195](Python/Optimization/unconstrained.py#L195)
##### `minimize(self, objective, x0, gradient, hessian)`
Minimize function using gradient descent.
Args:
objective: Function to minimize
x0: Initial point
gradient: Gradient function (optional)
hessian: Not used in gradient descent
Returns:
OptimizationResult
**Source:** [Line 217](Python/Optimization/unconstrained.py#L217)
##### `_backtracking_line_search(self, objective, x, grad, f_val)`
Backtracking line search with Armijo condition.
**Source:** [Line 286](Python/Optimization/unconstrained.py#L286)
##### `_armijo_line_search(self, objective, x, grad, f_val)`
Armijo line search.
**Source:** [Line 303](Python/Optimization/unconstrained.py#L303)
**Class Source:** [Line 187](Python/Optimization/unconstrained.py#L187)
### `NewtonMethod`
Newton's Method for unconstrained optimization.
Uses second-order Taylor approximation with Hessian information
for quadratic convergence near the optimum.
#### Methods
##### `__init__(self, damping, regularization)`
Initialize Newton's Method.
Args:
damping: Damping factor for Newton step
regularization: Regularization for Hessian inversion
**Source:** [Line 316](Python/Optimization/unconstrained.py#L316)
##### `minimize(self, objective, x0, gradient, hessian)`
Minimize function using Newton's method.
Args:
objective: Function to minimize
x0: Initial point
gradient: Gradient function
hessian: Hessian function
Returns:
OptimizationResult
**Source:** [Line 329](Python/Optimization/unconstrained.py#L329)
**Class Source:** [Line 308](Python/Optimization/unconstrained.py#L308)
### `BFGS`
Broyden-Fletcher-Goldfarb-Shanno (BFGS) quasi-Newton method.
Approximates the Hessian using gradient information from
previous iterations, avoiding expensive Hessian computations.
#### Methods
##### `__init__(self, initial_hessian, line_search, c1, c2)`
Initialize BFGS optimizer.
Args:
initial_hessian: Initial Hessian approximation
line_search: Line search method ('wolfe', 'backtracking')
c1: Armijo condition parameter
c2: Curvature condition parameter
**Source:** [Line 409](Python/Optimization/unconstrained.py#L409)
##### `minimize(self, objective, x0, gradient, hessian)`
Minimize function using BFGS.
Args:
objective: Function to minimize
x0: Initial point
gradient: Gradient function
hessian: Not used in BFGS
Returns:
OptimizationResult
**Source:** [Line 427](Python/Optimization/unconstrained.py#L427)
##### `_line_search(self, objective, gradient, x, p, f_val, grad)`
Perform line search to find step size.
**Source:** [Line 524](Python/Optimization/unconstrained.py#L524)
**Class Source:** [Line 401](Python/Optimization/unconstrained.py#L401)
### `ConjugateGradient`
Conjugate Gradient method for unconstrained optimization.
Particularly effective for quadratic functions and large-scale
optimization problems where Hessian computation is expensive.
#### Methods
##### `__init__(self, beta_method, restart_threshold)`
Initialize Conjugate Gradient optimizer.
Args:
beta_method: Method for computing beta ('fletcher-reeves', 'polak-ribiere', 'hestenes-stiefel')
restart_threshold: Iterations after which to restart (default: n)
**Source:** [Line 558](Python/Optimization/unconstrained.py#L558)
##### `minimize(self, objective, x0, gradient, hessian)`
Minimize function using Conjugate Gradient.
Args:
objective: Function to minimize
x0: Initial point
gradient: Gradient function
hessian: Not used in CG
Returns:
OptimizationResult
**Source:** [Line 571](Python/Optimization/unconstrained.py#L571)
##### `_compute_beta(self, grad_old, grad_new, p_old)`
Compute beta parameter for conjugate direction.
**Source:** [Line 649](Python/Optimization/unconstrained.py#L649)
##### `_line_search_cg(self, objective, x, p, f_val, grad)`
Simple line search for conjugate gradient.
**Source:** [Line 661](Python/Optimization/unconstrained.py#L661)
**Class Source:** [Line 550](Python/Optimization/unconstrained.py#L550)
### `TrustRegion`
Trust Region method for unconstrained optimization.
Uses a trust region approach with quadratic model approximation
and adaptive region size adjustment.
#### Methods
##### `__init__(self, initial_radius, max_radius, eta1, eta2, gamma1, gamma2)`
Initialize Trust Region optimizer.
Args:
initial_radius: Initial trust region radius
max_radius: Maximum trust region radius
eta1: Threshold for shrinking trust region
eta2: Threshold for expanding trust region
gamma1: Shrinking factor
gamma2: Expansion factor
**Source:** [Line 687](Python/Optimization/unconstrained.py#L687)
##### `minimize(self, objective, x0, gradient, hessian)`
Minimize function using Trust Region method.
Args:
objective: Function to minimize
x0: Initial point
gradient: Gradient function
hessian: Hessian function
Returns:
OptimizationResult
**Source:** [Line 709](Python/Optimization/unconstrained.py#L709)
##### `_solve_trust_region_subproblem(self, grad, hess, radius)`
Solve trust region subproblem using Cauchy point method.
This is a simplified implementation. More sophisticated methods
like dogleg or exact trust region solutions could be implemented.
**Source:** [Line 793](Python/Optimization/unconstrained.py#L793)
**Class Source:** [Line 679](Python/Optimization/unconstrained.py#L679)