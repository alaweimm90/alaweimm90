# nonlinear_solvers
**Module:** `Python/ODE_PDE/nonlinear_solvers.py`
## Overview
Nonlinear Solvers for ODEs and PDEs.
This module provides comprehensive nonlinear solving capabilities including
Newton methods, continuation methods, and specialized solvers for nonlinear
differential equations.
Classes:
NonlinearSolver: Base class for nonlinear solvers
NewtonSolver: Newton-Raphson method
DampedNewtonSolver: Damped Newton method
ContinuationSolver: Continuation/homotopy methods
FixedPointSolver: Fixed point iteration
Functions:
newton_raphson: Newton-Raphson solver
solve_nonlinear_ode: Nonlinear ODE solver
solve_nonlinear_pde: Nonlinear PDE solver
continuation_method: Parameter continuation
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `newton_raphson(f, x0, jacobian, tolerance, max_iterations)`
Convenience function for Newton-Raphson method.
Args:
f: Nonlinear function F(x)
x0: Initial guess
jacobian: Jacobian function (optional)
tolerance: Convergence tolerance
max_iterations: Maximum iterations
Returns:
NonlinearResult
**Source:** [Line 568](Python/ODE_PDE/nonlinear_solvers.py#L568)
### `solve_nonlinear_ode(dydt, y0, t, dt, method)`
Solve nonlinear ODE using implicit method.
Args:
dydt: Right-hand side function dy/dt = f(t, y)
y0: Initial condition
t: Current time
dt: Time step
method: Nonlinear solver method
Returns:
Solution at t + dt
**Source:** [Line 588](Python/ODE_PDE/nonlinear_solvers.py#L588)
### `solve_nonlinear_pde(pde_residual, u0, jacobian, method, tolerance)`
Solve nonlinear PDE system.
Args:
pde_residual: PDE residual function R(u) = 0
u0: Initial guess
jacobian: Jacobian function (optional)
method: Nonlinear solver method
tolerance: Convergence tolerance
Returns:
NonlinearResult
**Source:** [Line 640](Python/ODE_PDE/nonlinear_solvers.py#L640)
### `continuation_method(f, jacobian, x0, param_range, n_steps)`
Convenience function for parameter continuation.
Args:
f: Function F(x, p)
jacobian: Jacobian dF/dx(x, p)
x0: Initial solution
param_range: Parameter range
n_steps: Number of continuation steps
Returns:
ContinuationResult
**Source:** [Line 668](Python/ODE_PDE/nonlinear_solvers.py#L668)
### `picard_iteration(f, x0, max_iterations, tolerance, relaxation)`
Picard iteration for solving x = f(x).
Args:
f: Function defining x = f(x)
x0: Initial guess
max_iterations: Maximum iterations
tolerance: Convergence tolerance
relaxation: Relaxation parameter
Returns:
NonlinearResult
**Source:** [Line 687](Python/ODE_PDE/nonlinear_solvers.py#L687)
### `analyze_nonlinear_convergence(convergence_history)`
Analyze convergence behavior of nonlinear solver.
Args:
convergence_history: History of residual norms
Returns:
Dictionary with convergence analysis
**Source:** [Line 707](Python/ODE_PDE/nonlinear_solvers.py#L707)
## Classes
### `NonlinearResult`
Result of nonlinear solver.
**Class Source:** [Line 36](Python/ODE_PDE/nonlinear_solvers.py#L36)
### `ContinuationResult`
Result of continuation method.
**Class Source:** [Line 51](Python/ODE_PDE/nonlinear_solvers.py#L51)
### `NonlinearSolver`
Abstract base class for nonlinear solvers.
#### Methods
##### `__init__(self, tolerance, max_iterations)`
Initialize nonlinear solver.
Args:
tolerance: Convergence tolerance
max_iterations: Maximum number of iterations
**Source:** [Line 65](Python/ODE_PDE/nonlinear_solvers.py#L65)
##### `solve(self, f, x0, jacobian)`
Solve nonlinear system F(x) = 0.
Args:
f: Nonlinear function F(x)
x0: Initial guess
jacobian: Jacobian function dF/dx (optional)
Returns:
NonlinearResult
**Source:** [Line 80](Python/ODE_PDE/nonlinear_solvers.py#L80)
##### `finite_difference_jacobian(self, f, x, eps)`
Compute Jacobian using finite differences.
Args:
f: Function F(x)
x: Point to evaluate Jacobian
eps: Finite difference step size
Returns:
Jacobian matrix
**Source:** [Line 94](Python/ODE_PDE/nonlinear_solvers.py#L94)
**Class Source:** [Line 62](Python/ODE_PDE/nonlinear_solvers.py#L62)
### `NewtonSolver`
Newton-Raphson method for nonlinear systems.
#### Methods
##### `__init__(self, tolerance, max_iterations, use_line_search)`
Initialize Newton solver.
Args:
tolerance: Convergence tolerance
max_iterations: Maximum iterations
use_line_search: Whether to use line search
**Source:** [Line 126](Python/ODE_PDE/nonlinear_solvers.py#L126)
##### `solve(self, f, x0, jacobian)`
Solve using Newton-Raphson method.
Args:
f: Nonlinear function F(x)
x0: Initial guess
jacobian: Jacobian function (optional)
Returns:
NonlinearResult
**Source:** [Line 138](Python/ODE_PDE/nonlinear_solvers.py#L138)
##### `_line_search(self, f, x, f_x, delta_x, alpha_max)`
Simple backtracking line search.
Args:
f: Function
x: Current point
f_x: Function value at x
delta_x: Search direction
alpha_max: Maximum step size
Returns:
Step size alpha
**Source:** [Line 230](Python/ODE_PDE/nonlinear_solvers.py#L230)
**Class Source:** [Line 123](Python/ODE_PDE/nonlinear_solvers.py#L123)
### `DampedNewtonSolver`
Damped Newton method with adaptive damping.
#### Methods
##### `__init__(self, tolerance, max_iterations, initial_damping, min_damping)`
Initialize damped Newton solver.
Args:
tolerance: Convergence tolerance
max_iterations: Maximum iterations
initial_damping: Initial damping factor
min_damping: Minimum damping factor
**Source:** [Line 268](Python/ODE_PDE/nonlinear_solvers.py#L268)
##### `solve(self, f, x0, jacobian)`
Solve using damped Newton method.
Args:
f: Nonlinear function F(x)
x0: Initial guess
jacobian: Jacobian function (optional)
Returns:
NonlinearResult
**Source:** [Line 282](Python/ODE_PDE/nonlinear_solvers.py#L282)
**Class Source:** [Line 265](Python/ODE_PDE/nonlinear_solvers.py#L265)
### `FixedPointSolver`
Fixed point iteration solver.
#### Methods
##### `__init__(self, tolerance, max_iterations, relaxation)`
Initialize fixed point solver.
Args:
tolerance: Convergence tolerance
max_iterations: Maximum iterations
relaxation: Relaxation parameter (0 < relaxation <= 1)
**Source:** [Line 389](Python/ODE_PDE/nonlinear_solvers.py#L389)
##### `solve(self, g, x0, jacobian)`
Solve using fixed point iteration x = g(x).
Args:
g: Fixed point function x = g(x)
x0: Initial guess
jacobian: Not used for fixed point iteration
Returns:
NonlinearResult
**Source:** [Line 401](Python/ODE_PDE/nonlinear_solvers.py#L401)
**Class Source:** [Line 386](Python/ODE_PDE/nonlinear_solvers.py#L386)
### `ContinuationSolver`
Continuation/homotopy method solver.
#### Methods
##### `__init__(self, tolerance, max_iterations, max_continuation_steps)`
Initialize continuation solver.
Args:
tolerance: Convergence tolerance
max_iterations: Maximum Newton iterations per step
max_continuation_steps: Maximum continuation steps
**Source:** [Line 464](Python/ODE_PDE/nonlinear_solvers.py#L464)
##### `solve_parameter_continuation(self, f, jacobian, x0, param_range, n_steps)`
Solve F(x, p) = 0 for parameter p in given range.
Args:
f: Function F(x, p)
jacobian: Jacobian dF/dx(x, p)
x0: Initial solution at p0
param_range: Parameter range (p0, pf)
n_steps: Number of continuation steps
Returns:
ContinuationResult
**Source:** [Line 477](Python/ODE_PDE/nonlinear_solvers.py#L477)
**Class Source:** [Line 461](Python/ODE_PDE/nonlinear_solvers.py#L461)