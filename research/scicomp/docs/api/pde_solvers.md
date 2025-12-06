# pde_solvers
**Module:** `Python/ODE_PDE/pde_solvers.py`
## Overview
Partial Differential Equation Solvers.
This module provides comprehensive PDE solving capabilities including
finite difference, finite element, and spectral methods for various
types of PDEs including heat, wave, Poisson, and Navier-Stokes equations.
Classes:
PDESolver: Base class for PDE solvers
FiniteDifferencePDE: Finite difference method base
HeatEquationSolver: Heat/diffusion equation solver
WaveEquationSolver: Wave equation solver
PoissonSolver: Poisson equation solver
AdvectionDiffusionSolver: Advection-diffusion equation
NavierStokesSolver: Navier-Stokes equations
Functions:
solve_pde: General PDE solving interface
solve_heat_equation: Heat equation solver
solve_wave_equation: Wave equation solver
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `solve_pde(pde_type, domain, boundary_conditions, initial_condition, time_span, dt, source_term, parameters, options)`
General PDE solving interface.
Args:
pde_type: Type of PDE ('heat', 'wave', 'poisson', 'advection_diffusion')
domain: Spatial domain specification
boundary_conditions: Boundary conditions
initial_condition: Initial condition (for time-dependent)
time_span: Time interval (for time-dependent)
dt: Time step (for time-dependent)
source_term: Source term function
parameters: PDE-specific parameters
options: Solver options
Returns:
PDEResult containing solution
**Source:** [Line 692](Python/ODE_PDE/pde_solvers.py#L692)
### `solve_heat_equation(domain, boundary_conditions, initial_condition, time_span, thermal_diffusivity, source_term)`
Solve heat equation with default settings.
**Source:** [Line 767](Python/ODE_PDE/pde_solvers.py#L767)
### `solve_wave_equation(domain, boundary_conditions, initial_condition, initial_velocity, time_span, wave_speed, source_term)`
Solve wave equation with default settings.
**Source:** [Line 779](Python/ODE_PDE/pde_solvers.py#L779)
## Classes
### `PDEResult`
Result of PDE solution.
**Class Source:** [Line 36](Python/ODE_PDE/pde_solvers.py#L36)
### `PDEOptions`
Options for PDE solvers.
**Class Source:** [Line 50](Python/ODE_PDE/pde_solvers.py#L50)
### `PDESolver`
Abstract base class for PDE solvers.
#### Methods
##### `__init__(self, domain, boundary_conditions, options)`
Initialize PDE solver.
Args:
domain: Spatial domain specification
boundary_conditions: Boundary conditions
options: Solver options
**Source:** [Line 65](Python/ODE_PDE/pde_solvers.py#L65)
##### `solve_steady(self, source_term)`
Solve steady-state PDE.
**Source:** [Line 102](Python/ODE_PDE/pde_solvers.py#L102)
##### `solve_transient(self, initial_condition, time_span, dt, source_term)`
Solve time-dependent PDE.
**Source:** [Line 107](Python/ODE_PDE/pde_solvers.py#L107)
##### `_apply_boundary_conditions(self, matrix, rhs)`
Apply boundary conditions to system.
**Source:** [Line 114](Python/ODE_PDE/pde_solvers.py#L114)
**Class Source:** [Line 62](Python/ODE_PDE/pde_solvers.py#L62)
### `FiniteDifferencePDE`
Finite difference method for PDEs.
#### Methods
##### `__init__(self, domain, boundary_conditions, options)`
*No documentation available.*
**Source:** [Line 135](Python/ODE_PDE/pde_solvers.py#L135)
##### `build_laplacian_1d(self)`
Build 1D Laplacian matrix.
**Source:** [Line 140](Python/ODE_PDE/pde_solvers.py#L140)
##### `build_laplacian_2d(self)`
Build 2D Laplacian matrix using finite differences.
**Source:** [Line 158](Python/ODE_PDE/pde_solvers.py#L158)
##### `build_gradient_1d(self)`
Build 1D gradient matrix (central differences).
**Source:** [Line 183](Python/ODE_PDE/pde_solvers.py#L183)
**Class Source:** [Line 132](Python/ODE_PDE/pde_solvers.py#L132)
### `HeatEquationSolver`
Solver for heat/diffusion equation: ∂u/∂t = α∇²u + f.
#### Methods
##### `__init__(self, domain, boundary_conditions, thermal_diffusivity, options)`
*No documentation available.*
**Source:** [Line 200](Python/ODE_PDE/pde_solvers.py#L200)
##### `solve_steady(self, source_term)`
Solve steady-state heat equation: -α∇²u = f.
**Source:** [Line 207](Python/ODE_PDE/pde_solvers.py#L207)
##### `solve_transient(self, initial_condition, time_span, dt, source_term)`
Solve transient heat equation.
**Source:** [Line 262](Python/ODE_PDE/pde_solvers.py#L262)
**Class Source:** [Line 197](Python/ODE_PDE/pde_solvers.py#L197)
### `WaveEquationSolver`
Solver for wave equation: ∂²u/∂t² = c²∇²u + f.
#### Methods
##### `__init__(self, domain, boundary_conditions, wave_speed, options)`
*No documentation available.*
**Source:** [Line 359](Python/ODE_PDE/pde_solvers.py#L359)
##### `solve_steady(self, source_term)`
Wave equation has no meaningful steady state.
**Source:** [Line 366](Python/ODE_PDE/pde_solvers.py#L366)
##### `solve_transient(self, initial_condition, initial_velocity, time_span, dt, source_term)`
Solve transient wave equation using central differences.
**Source:** [Line 370](Python/ODE_PDE/pde_solvers.py#L370)
**Class Source:** [Line 356](Python/ODE_PDE/pde_solvers.py#L356)
### `PoissonSolver`
Solver for Poisson equation: -∇²u = f.
#### Methods
##### `__init__(self, domain, boundary_conditions, options)`
*No documentation available.*
**Source:** [Line 473](Python/ODE_PDE/pde_solvers.py#L473)
##### `solve_steady(self, source_term)`
Solve Poisson equation.
**Source:** [Line 478](Python/ODE_PDE/pde_solvers.py#L478)
##### `solve_transient(self, initial_condition, time_span, dt, source_term)`
Poisson equation is elliptic, no time dependence.
**Source:** [Line 529](Python/ODE_PDE/pde_solvers.py#L529)
**Class Source:** [Line 470](Python/ODE_PDE/pde_solvers.py#L470)
### `AdvectionDiffusionSolver`
Solver for advection-diffusion equation: ∂u/∂t + v·∇u = D∇²u + f.
#### Methods
##### `__init__(self, domain, boundary_conditions, velocity, diffusivity, options)`
*No documentation available.*
**Source:** [Line 540](Python/ODE_PDE/pde_solvers.py#L540)
##### `solve_steady(self, source_term)`
Solve steady advection-diffusion: v·∇u = D∇²u + f.
**Source:** [Line 549](Python/ODE_PDE/pde_solvers.py#L549)
##### `solve_transient(self, initial_condition, time_span, dt, source_term)`
Solve transient advection-diffusion equation.
**Source:** [Line 600](Python/ODE_PDE/pde_solvers.py#L600)
**Class Source:** [Line 537](Python/ODE_PDE/pde_solvers.py#L537)
### `NavierStokesSolver`
Simplified Navier-Stokes solver for incompressible flow.
#### Methods
##### `__init__(self, domain, boundary_conditions, viscosity, density, options)`
*No documentation available.*
**Source:** [Line 666](Python/ODE_PDE/pde_solvers.py#L666)
##### `solve_steady(self, source_term)`
Solve steady Navier-Stokes (simplified).
**Source:** [Line 675](Python/ODE_PDE/pde_solvers.py#L675)
##### `solve_transient(self, initial_condition, time_span, dt, source_term)`
Solve transient Navier-Stokes (simplified).
**Source:** [Line 682](Python/ODE_PDE/pde_solvers.py#L682)
**Class Source:** [Line 663](Python/ODE_PDE/pde_solvers.py#L663)