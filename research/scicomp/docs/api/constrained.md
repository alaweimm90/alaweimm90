# constrained
**Module:** `Python/Optimization/constrained.py`
## Overview
Constrained Optimization Algorithms
==================================
This module implements various constrained optimization algorithms including
penalty methods, barrier methods, Lagrange multipliers, and sequential
quadratic programming for handling equality and inequality constraints.
Author: Berkeley SciComp Team
Date: 2024
## Constants
- **`BERKELEY_BLUE`**
- **`CALIFORNIA_GOLD`**
- **`BERKELEY_LIGHT_BLUE`**
## Functions
### `demo()`
Demonstrate constrained optimization algorithms.
**Source:** [Line 807](Python/Optimization/constrained.py#L807)
## Classes
### `Constraint`
Represents a constraint in optimization problem.
Attributes:
fun: Constraint function
jac: Jacobian of constraint function
type: Type of constraint ('eq' for equality, 'ineq' for inequality)
args: Additional arguments for constraint function
**Class Source:** [Line 27](Python/Optimization/constrained.py#L27)
### `ConstrainedOptimizer`
Abstract base class for constrained optimization algorithms.
This class provides a common interface for all constrained
optimization methods in the Berkeley SciComp framework.
#### Methods
##### `__init__(self, max_iterations, tolerance, track_path, verbose)`
Initialize constrained optimizer.
Args:
max_iterations: Maximum number of iterations
tolerance: Convergence tolerance
track_path: Whether to track optimization path
verbose: Whether to print progress information
**Source:** [Line 50](Python/Optimization/constrained.py#L50)
##### `minimize(self, objective, x0, constraints, bounds, gradient, hessian)`
Minimize the constrained objective function.
Args:
objective: Objective function to minimize
x0: Initial guess
constraints: List of constraint objects
bounds: Variable bounds
gradient: Gradient function (optional)
hessian: Hessian function (optional)
**kwargs: Additional algorithm-specific parameters
Returns:
OptimizationResult object
**Source:** [Line 72](Python/Optimization/constrained.py#L72)
##### `_evaluate_function(self, func, x)`
Evaluate function with counter increment.
**Source:** [Line 95](Python/Optimization/constrained.py#L95)
##### `_evaluate_constraints(self, constraints, x)`
Evaluate constraints and separate into equality and inequality.
Returns:
Tuple of (equality_constraints, inequality_constraints)
**Source:** [Line 100](Python/Optimization/constrained.py#L100)
##### `_evaluate_constraint_jacobians(self, constraints, x)`
Evaluate constraint Jacobians.
Returns:
Tuple of (equality_jacobian, inequality_jacobian)
**Source:** [Line 120](Python/Optimization/constrained.py#L120)
##### `_numerical_jacobian(self, func, x, h)`
Compute numerical Jacobian.
**Source:** [Line 148](Python/Optimization/constrained.py#L148)
##### `_clip_to_bounds(self, x, bounds)`
Clip variables to bounds.
**Source:** [Line 163](Python/Optimization/constrained.py#L163)
**Class Source:** [Line 42](Python/Optimization/constrained.py#L42)
### `PenaltyMethod`
Penalty Method for constrained optimization.
Converts constrained problem to unconstrained by adding penalty terms
for constraint violations to the objective function.
#### Methods
##### `__init__(self, penalty_parameter, penalty_increase, unconstrained_solver)`
Initialize Penalty Method.
Args:
penalty_parameter: Initial penalty parameter
penalty_increase: Factor to increase penalty parameter
unconstrained_solver: Unconstrained solver to use
**Source:** [Line 185](Python/Optimization/constrained.py#L185)
##### `minimize(self, objective, x0, constraints, bounds, gradient, hessian)`
Minimize function using penalty method.
Args:
objective: Objective function
x0: Initial point
constraints: List of constraints
bounds: Variable bounds
gradient: Gradient function
hessian: Hessian function
Returns:
OptimizationResult
**Source:** [Line 200](Python/Optimization/constrained.py#L200)
**Class Source:** [Line 177](Python/Optimization/constrained.py#L177)
### `BarrierMethod`
Barrier Method (Interior Point Method) for constrained optimization.
Uses logarithmic barrier functions to keep iterates in the feasible region
while solving inequality-constrained problems.
#### Methods
##### `__init__(self, barrier_parameter, barrier_decrease, unconstrained_solver)`
Initialize Barrier Method.
Args:
barrier_parameter: Initial barrier parameter
barrier_decrease: Factor to decrease barrier parameter
unconstrained_solver: Unconstrained solver to use
**Source:** [Line 312](Python/Optimization/constrained.py#L312)
##### `minimize(self, objective, x0, constraints, bounds, gradient, hessian)`
Minimize function using barrier method.
Args:
objective: Objective function
x0: Initial point (must be feasible)
constraints: List of inequality constraints
bounds: Variable bounds
gradient: Gradient function
hessian: Hessian function
Returns:
OptimizationResult
**Source:** [Line 327](Python/Optimization/constrained.py#L327)
**Class Source:** [Line 304](Python/Optimization/constrained.py#L304)
### `AugmentedLagrangian`
Augmented Lagrangian Method for constrained optimization.
Combines Lagrange multipliers with penalty terms to handle
both equality and inequality constraints effectively.
#### Methods
##### `__init__(self, penalty_parameter, penalty_increase, multiplier_update, unconstrained_solver)`
Initialize Augmented Lagrangian Method.
Args:
penalty_parameter: Initial penalty parameter
penalty_increase: Factor to increase penalty parameter
multiplier_update: Multiplier update rule ('standard', 'safeguarded')
unconstrained_solver: Unconstrained solver to use
**Source:** [Line 436](Python/Optimization/constrained.py#L436)
##### `minimize(self, objective, x0, constraints, bounds, gradient, hessian)`
Minimize function using augmented Lagrangian method.
Args:
objective: Objective function
x0: Initial point
constraints: List of constraints
bounds: Variable bounds
gradient: Gradient function
hessian: Hessian function
Returns:
OptimizationResult
**Source:** [Line 454](Python/Optimization/constrained.py#L454)
**Class Source:** [Line 428](Python/Optimization/constrained.py#L428)
### `SequentialQuadraticProgramming`
Sequential Quadratic Programming (SQP) for constrained optimization.
Solves a sequence of quadratic programming subproblems to approximate
the constrained optimization problem.
#### Methods
##### `__init__(self, hessian_update, merit_function, line_search)`
Initialize SQP Method.
Args:
hessian_update: Hessian approximation method ('BFGS', 'SR1')
merit_function: Merit function for line search ('l1', 'l2')
line_search: Whether to use line search
**Source:** [Line 586](Python/Optimization/constrained.py#L586)
##### `minimize(self, objective, x0, constraints, bounds, gradient, hessian)`
Minimize function using SQP method.
Args:
objective: Objective function
x0: Initial point
constraints: List of constraints
bounds: Variable bounds
gradient: Gradient function
hessian: Hessian function
Returns:
OptimizationResult
**Source:** [Line 601](Python/Optimization/constrained.py#L601)
##### `_solve_qp_subproblem(self, grad_f, B, eq_constraints, ineq_constraints, eq_jac, ineq_jac)`
Solve QP subproblem (simplified implementation).
This is a basic implementation using least squares.
A real SQP implementation would use a proper QP solver.
**Source:** [Line 723](Python/Optimization/constrained.py#L723)
##### `_line_search_sqp(self, objective, constraints, x, p, f_val)`
Simple line search for SQP.
**Source:** [Line 771](Python/Optimization/constrained.py#L771)
##### `_numerical_gradient(self, func, x, h)`
Compute numerical gradient.
**Source:** [Line 793](Python/Optimization/constrained.py#L793)
**Class Source:** [Line 578](Python/Optimization/constrained.py#L578)