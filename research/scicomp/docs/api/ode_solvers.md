# ode_solvers
**Module:** `Python/ODE_PDE/ode_solvers.py`
## Overview
Ordinary Differential Equation Solvers.
This module provides comprehensive ODE solving capabilities including
explicit and implicit methods, adaptive time stepping, and stiff equation solvers.
Classes:
ODESolver: Base class for ODE solvers
ExplicitEuler: Forward Euler method
ImplicitEuler: Backward Euler method
RungeKutta4: Classical 4th order Runge-Kutta
RungeKuttaFehlberg: RK45 with adaptive stepping
AdamsBashforth: Multi-step explicit method
AdamsMoulton: Multi-step implicit method
BDF: Backward differentiation formulas for stiff ODEs
Functions:
solve_ode: General ODE solving interface
solve_ivp: Initial value problem solver
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `solve_ode(fun, y0, t_span, method, t_eval, dense_output, events, vectorized, args)`
Solve initial value problem for ordinary differential equations.
Args:
fun: Right-hand side function dy/dt = f(t, y)
y0: Initial conditions
t_span: Time span (t0, tf)
method: Solution method ('euler', 'rk4', 'rk45', 'adams', 'bdf')
t_eval: Times at which to evaluate solution
dense_output: Whether to compute dense output
events: Event functions for detection
vectorized: Whether function is vectorized
args: Additional arguments to pass to function
**options: Additional solver options
Returns:
ODEResult containing solution
**Source:** [Line 632](Python/ODE_PDE/ode_solvers.py#L632)
## Classes
### `ODEResult`
Result of ODE integration.
**Class Source:** [Line 34](Python/ODE_PDE/ode_solvers.py#L34)
### `ODEOptions`
Options for ODE solvers.
**Class Source:** [Line 49](Python/ODE_PDE/ode_solvers.py#L49)
### `ODESolver`
Abstract base class for ODE solvers.
Provides common interface for all ODE solving methods.
#### Methods
##### `__init__(self, fun, t0, y0, t_bound, options)`
Initialize ODE solver.
Args:
fun: Right-hand side function dy/dt = f(t, y)
t0: Initial time
y0: Initial conditions
t_bound: Final time
options: Solver options
**Source:** [Line 68](Python/ODE_PDE/ode_solvers.py#L68)
##### `step(self)`
Take one integration step.
Returns:
True if step successful, False if failed
**Source:** [Line 99](Python/ODE_PDE/ode_solvers.py#L99)
##### `dense_output(self, t)`
Compute solution at arbitrary points (if supported).
**Source:** [Line 107](Python/ODE_PDE/ode_solvers.py#L107)
##### `solve(self, t_eval)`
Solve the ODE.
Args:
t_eval: Times at which to evaluate solution
Returns:
ODEResult containing solution
**Source:** [Line 111](Python/ODE_PDE/ode_solvers.py#L111)
**Class Source:** [Line 62](Python/ODE_PDE/ode_solvers.py#L62)
### `ExplicitEuler`
Forward Euler method (1st order explicit).
Simple explicit method: y_{n+1} = y_n + h * f(t_n, y_n)
#### Methods
##### `__init__(self, fun, t0, y0, t_bound, options)`
*No documentation available.*
**Source:** [Line 186](Python/ODE_PDE/ode_solvers.py#L186)
##### `step(self)`
Take one Euler step.
**Source:** [Line 191](Python/ODE_PDE/ode_solvers.py#L191)
**Class Source:** [Line 180](Python/ODE_PDE/ode_solvers.py#L180)
### `ImplicitEuler`
Backward Euler method (1st order implicit).
Implicit method: y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
Requires solving nonlinear system at each step.
#### Methods
##### `__init__(self, fun, t0, y0, t_bound, options)`
*No documentation available.*
**Source:** [Line 215](Python/ODE_PDE/ode_solvers.py#L215)
##### `step(self)`
Take one implicit Euler step.
**Source:** [Line 220](Python/ODE_PDE/ode_solvers.py#L220)
**Class Source:** [Line 208](Python/ODE_PDE/ode_solvers.py#L208)
### `RungeKutta4`
Classical 4th order Runge-Kutta method.
High-accuracy explicit method with 4 function evaluations per step.
#### Methods
##### `__init__(self, fun, t0, y0, t_bound, options)`
*No documentation available.*
**Source:** [Line 251](Python/ODE_PDE/ode_solvers.py#L251)
##### `step(self)`
Take one RK4 step.
**Source:** [Line 256](Python/ODE_PDE/ode_solvers.py#L256)
**Class Source:** [Line 245](Python/ODE_PDE/ode_solvers.py#L245)
### `RungeKuttaFehlberg`
Runge-Kutta-Fehlberg method (RK45) with adaptive step size.
Embedded Runge-Kutta method that provides both 4th and 5th order
approximations for error estimation and step size control.
#### Methods
##### `__init__(self, fun, t0, y0, t_bound, options)`
*No documentation available.*
**Source:** [Line 302](Python/ODE_PDE/ode_solvers.py#L302)
##### `step(self)`
Take one adaptive RK45 step.
**Source:** [Line 309](Python/ODE_PDE/ode_solvers.py#L309)
**Class Source:** [Line 281](Python/ODE_PDE/ode_solvers.py#L281)
### `AdamsBashforth`
Adams-Bashforth multi-step explicit method.
Uses function values from previous steps to achieve higher order
with only one function evaluation per step.
#### Methods
##### `__init__(self, fun, t0, y0, t_bound, options, order)`
*No documentation available.*
**Source:** [Line 362](Python/ODE_PDE/ode_solvers.py#L362)
##### `step(self)`
Take one Adams-Bashforth step.
**Source:** [Line 382](Python/ODE_PDE/ode_solvers.py#L382)
**Class Source:** [Line 355](Python/ODE_PDE/ode_solvers.py#L355)
### `AdamsMoulton`
Adams-Moulton multi-step implicit method.
Implicit multi-step method that includes the current function value
in the formula, requiring solution of nonlinear system.
#### Methods
##### `__init__(self, fun, t0, y0, t_bound, options, order)`
*No documentation available.*
**Source:** [Line 429](Python/ODE_PDE/ode_solvers.py#L429)
##### `step(self)`
Take one Adams-Moulton step.
**Source:** [Line 448](Python/ODE_PDE/ode_solvers.py#L448)
**Class Source:** [Line 422](Python/ODE_PDE/ode_solvers.py#L422)
### `BDF`
Backward Differentiation Formulas for stiff ODEs.
Implicit multi-step methods particularly suited for stiff
differential equations with excellent stability properties.
#### Methods
##### `__init__(self, fun, t0, y0, t_bound, options, order)`
*No documentation available.*
**Source:** [Line 506](Python/ODE_PDE/ode_solvers.py#L506)
##### `step(self)`
Take one BDF step.
**Source:** [Line 529](Python/ODE_PDE/ode_solvers.py#L529)
##### `_newton_solve(self, residual, y_guess, t_new, max_iter)`
Solve nonlinear system using Newton's method.
**Source:** [Line 579](Python/ODE_PDE/ode_solvers.py#L579)
##### `_compute_jacobian(self, t, y)`
Compute Jacobian matrix using finite differences.
**Source:** [Line 606](Python/ODE_PDE/ode_solvers.py#L606)
**Class Source:** [Line 499](Python/ODE_PDE/ode_solvers.py#L499)