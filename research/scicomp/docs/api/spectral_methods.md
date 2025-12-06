# spectral_methods
**Module:** `Python/ODE_PDE/spectral_methods.py`
## Overview
Spectral Methods for PDEs.
This module provides spectral method implementations including Fourier spectral,
Chebyshev spectral, and pseudospectral methods for solving PDEs with high accuracy.
Classes:
SpectralSolver: Base class for spectral methods
FourierSpectral: Fourier spectral method
ChebyshevSpectral: Chebyshev spectral method
SpectralDifferentiation: Spectral differentiation matrices
Functions:
pseudospectral_solve: General pseudospectral solver
fourier_derivative: Fourier derivative computation
chebyshev_derivative: Chebyshev derivative computation
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `pseudospectral_solve(pde_type, domain, n_modes, boundary_conditions, spectral_type)`
General pseudospectral PDE solver.
Args:
pde_type: Type of PDE ('poisson', 'heat', 'wave')
domain: Computational domain
n_modes: Number of spectral modes
boundary_conditions: Boundary conditions
spectral_type: Type of spectral method ('fourier', 'chebyshev')
**kwargs: Additional parameters
Returns:
SpectralResult
**Source:** [Line 446](Python/ODE_PDE/spectral_methods.py#L446)
### `fourier_derivative(u, L, order)`
Compute derivative using Fourier method.
Args:
u: Function values on periodic grid
L: Domain length
order: Derivative order
Returns:
Derivative values
**Source:** [Line 483](Python/ODE_PDE/spectral_methods.py#L483)
### `chebyshev_derivative(u, domain, order)`
Compute derivative using Chebyshev method.
Args:
u: Function values on Chebyshev grid
domain: Physical domain
order: Derivative order
Returns:
Derivative values
**Source:** [Line 509](Python/ODE_PDE/spectral_methods.py#L509)
### `spectral_interpolation(x_data, u_data, x_interp, method)`
Spectral interpolation of data.
Args:
x_data: Data points (assumed to be spectral grid)
u_data: Function values at data points
x_interp: Points to interpolate to
method: Spectral method ('fourier', 'chebyshev')
Returns:
Interpolated values
**Source:** [Line 539](Python/ODE_PDE/spectral_methods.py#L539)
### `compute_spectral_accuracy(coeffs, threshold)`
Analyze spectral accuracy from coefficients.
Args:
coeffs: Spectral coefficients
threshold: Threshold for determining convergence
Returns:
Dictionary with accuracy metrics
**Source:** [Line 596](Python/ODE_PDE/spectral_methods.py#L596)
## Classes
### `SpectralResult`
Result of spectral method solution.
**Class Source:** [Line 31](Python/ODE_PDE/spectral_methods.py#L31)
### `SpectralSolver`
Abstract base class for spectral methods.
#### Methods
##### `__init__(self, n_modes, domain)`
Initialize spectral solver.
Args:
n_modes: Number of spectral modes
domain: Computational domain (a, b)
**Source:** [Line 45](Python/ODE_PDE/spectral_methods.py#L45)
##### `_setup_grid(self)`
Setup computational grid.
**Source:** [Line 65](Python/ODE_PDE/spectral_methods.py#L65)
##### `_setup_differentiation_matrices(self)`
Setup spectral differentiation matrices.
**Source:** [Line 70](Python/ODE_PDE/spectral_methods.py#L70)
##### `solve_pde(self, pde_func, boundary_conditions, initial_condition)`
Solve PDE using spectral method.
**Source:** [Line 75](Python/ODE_PDE/spectral_methods.py#L75)
**Class Source:** [Line 42](Python/ODE_PDE/spectral_methods.py#L42)
### `FourierSpectral`
Fourier spectral method for periodic problems.
#### Methods
##### `_setup_grid(self)`
Setup Fourier grid (periodic).
**Source:** [Line 84](Python/ODE_PDE/spectral_methods.py#L84)
##### `_setup_differentiation_matrices(self)`
Setup Fourier differentiation matrices.
**Source:** [Line 93](Python/ODE_PDE/spectral_methods.py#L93)
##### `fourier_derivative(self, u, order)`
Compute derivative using FFT.
Args:
u: Function values on grid
order: Derivative order
Returns:
Derivative values
**Source:** [Line 105](Python/ODE_PDE/spectral_methods.py#L105)
##### `solve_pde(self, pde_func, boundary_conditions, initial_condition)`
Solve PDE using Fourier spectral method.
Note: Fourier methods require periodic boundary conditions.
**Source:** [Line 131](Python/ODE_PDE/spectral_methods.py#L131)
##### `solve_heat_equation(self, alpha, initial_condition, time_span, n_time_steps)`
Solve heat equation using Fourier spectral method.
Args:
alpha: Thermal diffusivity
initial_condition: Initial condition function
time_span: Time interval (t0, tf)
n_time_steps: Number of time steps
Returns:
SpectralResult with solution
**Source:** [Line 164](Python/ODE_PDE/spectral_methods.py#L164)
**Class Source:** [Line 81](Python/ODE_PDE/spectral_methods.py#L81)
### `ChebyshevSpectral`
Chebyshev spectral method for non-periodic problems.
#### Methods
##### `_setup_grid(self)`
Setup Chebyshev-Gauss-Lobatto grid.
**Source:** [Line 212](Python/ODE_PDE/spectral_methods.py#L212)
##### `_setup_differentiation_matrices(self)`
Setup Chebyshev differentiation matrices.
**Source:** [Line 224](Python/ODE_PDE/spectral_methods.py#L224)
##### `solve_pde(self, pde_func, boundary_conditions, initial_condition)`
Solve PDE using Chebyshev spectral method.
**Source:** [Line 256](Python/ODE_PDE/spectral_methods.py#L256)
##### `solve_bvp(self, differential_operator, boundary_conditions, source_term)`
Solve boundary value problem.
Args:
differential_operator: Function that applies differential operator
boundary_conditions: Boundary condition values
source_term: Source term function
Returns:
SpectralResult
**Source:** [Line 319](Python/ODE_PDE/spectral_methods.py#L319)
**Class Source:** [Line 209](Python/ODE_PDE/spectral_methods.py#L209)
### `SpectralDifferentiation`
Utility class for spectral differentiation.
#### Methods
##### `fourier_differentiation_matrix(N, L)`
Construct Fourier differentiation matrix.
Args:
N: Number of grid points
L: Domain length
Returns:
Differentiation matrix
**Source:** [Line 388](Python/ODE_PDE/spectral_methods.py#L388)
##### `chebyshev_differentiation_matrix(N)`
Construct Chebyshev differentiation matrix.
Args:
N: Number of Chebyshev points
Returns:
Tuple of (grid_points, differentiation_matrix)
**Source:** [Line 411](Python/ODE_PDE/spectral_methods.py#L411)
**Class Source:** [Line 384](Python/ODE_PDE/spectral_methods.py#L384)