# adaptive_methods
**Module:** `Python/ODE_PDE/adaptive_methods.py`
## Overview
Adaptive Methods for ODE and PDE Solvers.
This module provides adaptive time stepping, mesh refinement, and error control
methods for robust and efficient ODE/PDE solving.
Classes:
AdaptiveTimeStepper: Adaptive time stepping for ODEs
AdaptiveMeshRefiner: Adaptive mesh refinement for PDEs
ErrorEstimator: Error estimation and control
Functions:
adaptive_rk_step: Single adaptive RK step
estimate_local_error: Local error estimation
compute_optimal_timestep: Optimal time step computation
refine_mesh: Mesh refinement based on error indicators
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `adaptive_rk_step(f, t, y, dt, rtol, atol)`
Single adaptive Runge-Kutta step.
Args:
f: Right-hand side function
t: Current time
y: Current solution
dt: Current time step
rtol: Relative tolerance
atol: Absolute tolerance
Returns:
Tuple of (y_new, accept, dt_new)
**Source:** [Line 471](Python/ODE_PDE/adaptive_methods.py#L471)
### `estimate_local_error(y1, y2, order)`
Estimate local truncation error.
Args:
y1: Solution from method of order p
y2: Solution from method of order p+1
order: Order of lower order method
Returns:
Local error estimate
**Source:** [Line 498](Python/ODE_PDE/adaptive_methods.py#L498)
### `compute_optimal_timestep(error, dt_current, tolerance, order, safety_factor)`
Compute optimal time step based on error estimate.
Args:
error: Current error estimate
dt_current: Current time step
tolerance: Error tolerance
order: Method order
safety_factor: Safety factor
Returns:
Optimal time step
**Source:** [Line 514](Python/ODE_PDE/adaptive_methods.py#L514)
### `refine_mesh_uniform(mesh, refinement_factor)`
Uniform mesh refinement.
Args:
mesh: Current mesh
refinement_factor: Refinement factor
Returns:
Refined mesh
**Source:** [Line 538](Python/ODE_PDE/adaptive_methods.py#L538)
### `estimate_convergence_rate(errors, step_sizes)`
Estimate convergence rate from error sequence.
Args:
errors: Sequence of errors
step_sizes: Corresponding step sizes
Returns:
Estimated convergence rate
**Source:** [Line 564](Python/ODE_PDE/adaptive_methods.py#L564)
### `adaptive_pde_solve(pde_solver, initial_mesh, initial_solution, target_error, max_iterations)`
Adaptive PDE solving with mesh refinement.
Args:
pde_solver: PDE solver function
initial_mesh: Initial mesh
initial_solution: Initial solution
target_error: Target error tolerance
max_iterations: Maximum refinement iterations
Returns:
Dictionary with final solution and mesh
**Source:** [Line 589](Python/ODE_PDE/adaptive_methods.py#L589)
## Classes
### `AdaptiveResult`
Result of adaptive computation.
**Class Source:** [Line 30](Python/ODE_PDE/adaptive_methods.py#L30)
### `RefinementResult`
Result of mesh refinement.
**Class Source:** [Line 44](Python/ODE_PDE/adaptive_methods.py#L44)
### `AdaptiveTimeStepper`
Adaptive time stepping for ODE integration.
#### Methods
##### `__init__(self, rtol, atol, dt_min, dt_max, safety_factor, max_increase_factor, max_decrease_factor)`
Initialize adaptive time stepper.
Args:
rtol: Relative tolerance
atol: Absolute tolerance
dt_min: Minimum time step
dt_max: Maximum time step
safety_factor: Safety factor for step size adjustment
max_increase_factor: Maximum step size increase factor
max_decrease_factor: Maximum step size decrease factor
**Source:** [Line 57](Python/ODE_PDE/adaptive_methods.py#L57)
##### `step_rk45(self, f, t, y, dt)`
Adaptive Runge-Kutta 4(5) step with error estimation.
Args:
f: Right-hand side function
t: Current time
y: Current solution
dt: Current time step
Returns:
Tuple of (y_new, error_estimate, dt_new)
**Source:** [Line 86](Python/ODE_PDE/adaptive_methods.py#L86)
##### `integrate(self, f, t_span, y0, dt0)`
Integrate ODE with adaptive time stepping.
Args:
f: Right-hand side function dy/dt = f(t, y)
t_span: Time span (t0, tf)
y0: Initial condition
dt0: Initial time step
Returns:
AdaptiveResult with solution
**Source:** [Line 150](Python/ODE_PDE/adaptive_methods.py#L150)
**Class Source:** [Line 54](Python/ODE_PDE/adaptive_methods.py#L54)
### `AdaptiveMeshRefiner`
Adaptive mesh refinement for PDEs.
#### Methods
##### `__init__(self, refinement_criterion, refinement_threshold, max_refinement_levels, min_element_size)`
Initialize adaptive mesh refiner.
Args:
refinement_criterion: Criterion for refinement ('gradient', 'residual', 'error')
refinement_threshold: Threshold for refinement
max_refinement_levels: Maximum refinement levels
min_element_size: Minimum element size
**Source:** [Line 226](Python/ODE_PDE/adaptive_methods.py#L226)
##### `compute_error_indicators(self, mesh, solution)`
Compute error indicators for mesh refinement.
Args:
mesh: Current mesh points
solution: Solution values at mesh points
Returns:
Error indicators for each element
**Source:** [Line 243](Python/ODE_PDE/adaptive_methods.py#L243)
##### `refine_mesh(self, mesh, solution, max_new_points)`
Refine mesh based on error indicators.
Args:
mesh: Current mesh points
solution: Solution values at mesh points
max_new_points: Maximum number of new points to add
Returns:
RefinementResult with refined mesh and solution
**Source:** [Line 295](Python/ODE_PDE/adaptive_methods.py#L295)
**Class Source:** [Line 223](Python/ODE_PDE/adaptive_methods.py#L223)
### `ErrorEstimator`
Error estimation for ODE/PDE solutions.
#### Methods
##### `__init__(self, method)`
Initialize error estimator.
Args:
method: Error estimation method ('richardson', 'embedding', 'residual')
**Source:** [Line 373](Python/ODE_PDE/adaptive_methods.py#L373)
##### `richardson_extrapolation(self, u_coarse, u_fine, refinement_ratio, order)`
Richardson extrapolation for error estimation.
Args:
u_coarse: Solution on coarse grid
u_fine: Solution on fine grid (subsampled to match coarse)
refinement_ratio: Grid refinement ratio
order: Expected order of accuracy
Returns:
Dictionary with error estimates
**Source:** [Line 381](Python/ODE_PDE/adaptive_methods.py#L381)
##### `embedded_error_estimate(self, y_low, y_high)`
Error estimate from embedded method.
Args:
y_low: Lower order solution
y_high: Higher order solution
Returns:
Dictionary with error estimates
**Source:** [Line 425](Python/ODE_PDE/adaptive_methods.py#L425)
##### `residual_error_estimate(self, residual, operator_norm)`
Residual-based error estimate.
Args:
residual: Residual vector
operator_norm: Estimate of operator norm
Returns:
Dictionary with error estimates
**Source:** [Line 449](Python/ODE_PDE/adaptive_methods.py#L449)
**Class Source:** [Line 370](Python/ODE_PDE/adaptive_methods.py#L370)