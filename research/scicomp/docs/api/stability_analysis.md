# stability_analysis
**Module:** `Python/ODE_PDE/stability_analysis.py`
## Overview
Stability Analysis for ODE and PDE Methods.
This module provides comprehensive stability analysis tools for numerical
methods including linear stability analysis, von Neumann analysis,
and eigenvalue-based stability assessment.
Classes:
StabilityAnalyzer: Main stability analysis class
LinearStabilityAnalyzer: Linear stability analysis
VonNeumannAnalyzer: Fourier stability analysis
Functions:
analyze_rk_stability: Runge-Kutta stability analysis
von_neumann_analysis: Von Neumann stability analysis
compute_stability_region: Stability region computation
cfl_analysis: CFL condition analysis
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `analyze_rk_stability(order, z_values)`
Analyze Runge-Kutta stability.
Args:
order: Order of RK method
z_values: Scaled eigenvalues
Returns:
StabilityResult
**Source:** [Line 380](Python/ODE_PDE/stability_analysis.py#L380)
### `von_neumann_analysis(pde_type, parameters, dx, dt, method)`
General von Neumann stability analysis.
Args:
pde_type: Type of PDE ('heat', 'wave', 'advection')
parameters: PDE parameters
dx: Spatial step size
dt: Time step
method: Time integration method
Returns:
StabilityResult
**Source:** [Line 395](Python/ODE_PDE/stability_analysis.py#L395)
### `compute_stability_region(method, real_range, imag_range, resolution)`
Compute stability region for numerical method.
Args:
method: Numerical method name
real_range: Range for real axis
imag_range: Range for imaginary axis
resolution: Grid resolution
Returns:
Tuple of (real_grid, imag_grid, stability_matrix)
**Source:** [Line 425](Python/ODE_PDE/stability_analysis.py#L425)
### `cfl_analysis(pde_type, parameters, dx, target_cfl)`
Analyze CFL conditions and recommend time step.
Args:
pde_type: Type of PDE
parameters: PDE parameters
dx: Spatial step size
target_cfl: Target CFL number
Returns:
Dictionary with CFL analysis results
**Source:** [Line 456](Python/ODE_PDE/stability_analysis.py#L456)
### `plot_stability_region(method, save_path, title)`
Plot stability region for numerical method.
Args:
method: Numerical method name
save_path: Path to save figure
title: Plot title
Returns:
Figure object
**Source:** [Line 513](Python/ODE_PDE/stability_analysis.py#L513)
### `stability_comparison(methods, z_test)`
Compare stability properties of different methods.
Args:
methods: List of method names
z_test: Test point in complex plane
Returns:
Dictionary with comparison results
**Source:** [Line 568](Python/ODE_PDE/stability_analysis.py#L568)
## Classes
### `StabilityResult`
Result of stability analysis.
**Class Source:** [Line 33](Python/ODE_PDE/stability_analysis.py#L33)
### `AmplificationMatrix`
Amplification matrix for stability analysis.
**Class Source:** [Line 46](Python/ODE_PDE/stability_analysis.py#L46)
### `StabilityAnalyzer`
General stability analyzer for numerical methods.
#### Methods
##### `__init__(self, method_name, order)`
Initialize stability analyzer.
Args:
method_name: Name of numerical method
order: Order of the method
**Source:** [Line 58](Python/ODE_PDE/stability_analysis.py#L58)
##### `analyze_method_stability(self, lambda_values)`
Analyze stability for given eigenvalues.
Args:
lambda_values: Eigenvalues of the spatial discretization
Returns:
StabilityResult
**Source:** [Line 68](Python/ODE_PDE/stability_analysis.py#L68)
##### `_get_stability_function(self)`
Get stability function R(z) for the method.
**Source:** [Line 99](Python/ODE_PDE/stability_analysis.py#L99)
**Class Source:** [Line 55](Python/ODE_PDE/stability_analysis.py#L55)
### `LinearStabilityAnalyzer`
Linear stability analysis for ODEs and PDEs.
#### Methods
##### `__init__(self)`
Initialize linear stability analyzer.
**Source:** [Line 128](Python/ODE_PDE/stability_analysis.py#L128)
##### `analyze_ode_system(self, jacobian, dt, method)`
Analyze stability of ODE system du/dt = f(u).
Args:
jacobian: Jacobian matrix df/du
dt: Time step
method: Time integration method
Returns:
StabilityResult
**Source:** [Line 132](Python/ODE_PDE/stability_analysis.py#L132)
##### `analyze_pde_discretization(self, spatial_operator, dx, dt, method)`
Analyze stability of PDE spatial discretization.
Args:
spatial_operator: Spatial discretization matrix
dx: Spatial step size
dt: Time step
method: Time integration method
Returns:
StabilityResult
**Source:** [Line 165](Python/ODE_PDE/stability_analysis.py#L165)
**Class Source:** [Line 125](Python/ODE_PDE/stability_analysis.py#L125)
### `VonNeumannAnalyzer`
Von Neumann (Fourier) stability analysis for PDEs.
#### Methods
##### `__init__(self)`
Initialize von Neumann analyzer.
**Source:** [Line 201](Python/ODE_PDE/stability_analysis.py#L201)
##### `analyze_heat_equation(self, alpha, dx, dt, method)`
Von Neumann analysis for heat equation ∂u/∂t = α∂²u/∂x².
Args:
alpha: Thermal diffusivity
dx: Spatial step size
dt: Time step
method: Time integration method
Returns:
StabilityResult
**Source:** [Line 205](Python/ODE_PDE/stability_analysis.py#L205)
##### `analyze_wave_equation(self, c, dx, dt)`
Von Neumann analysis for wave equation ∂²u/∂t² = c²∂²u/∂x².
Args:
c: Wave speed
dx: Spatial step size
dt: Time step
Returns:
StabilityResult
**Source:** [Line 268](Python/ODE_PDE/stability_analysis.py#L268)
##### `analyze_advection_equation(self, v, dx, dt, scheme)`
Von Neumann analysis for advection equation ∂u/∂t + v∂u/∂x = 0.
Args:
v: Advection velocity
dx: Spatial step size
dt: Time step
scheme: Spatial discretization scheme
Returns:
StabilityResult
**Source:** [Line 315](Python/ODE_PDE/stability_analysis.py#L315)
**Class Source:** [Line 198](Python/ODE_PDE/stability_analysis.py#L198)