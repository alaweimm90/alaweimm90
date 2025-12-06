# heat_equation
**Module:** `Python/Thermal_Transport/core/heat_equation.py`
## Overview
Heat equation solvers for thermal transport simulations.
This module provides numerical solvers for the heat equation using
finite difference and finite element methods.
Classes:
HeatEquationSolver1D: 1D heat equation solver
HeatEquationSolver2D: 2D heat equation solver
Author: Berkeley SciComp Team
Date: 2025
## Classes
### `HeatEquationSolver1D`
One-dimensional heat equation solver.
#### Methods
##### `__init__(self, L, nx, alpha)`
Initialize 1D heat equation solver.
Args:
L: Domain length
nx: Number of grid points
alpha: Thermal diffusivity
**Source:** [Line 24](Python/Thermal_Transport/core/heat_equation.py#L24)
##### `solve(self, T_initial, t_final, dt, boundary_conditions, T_left, T_right)`
Solve 1D heat equation.
Args:
T_initial: Initial temperature distribution
t_final: Final time
dt: Time step
boundary_conditions: Type of boundary conditions
T_left: Left boundary temperature
T_right: Right boundary temperature
Returns:
Final temperature distribution
**Source:** [Line 41](Python/Thermal_Transport/core/heat_equation.py#L41)
**Class Source:** [Line 21](Python/Thermal_Transport/core/heat_equation.py#L21)