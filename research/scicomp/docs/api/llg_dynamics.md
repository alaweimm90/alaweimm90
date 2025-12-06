# llg_dynamics
**Module:** `Python/Spintronics/core/llg_dynamics.py`
## Overview
Landau-Lifshitz-Gilbert dynamics for magnetic moments.
This module provides tools for simulating magnetization dynamics using
the Landau-Lifshitz-Gilbert equation.
Classes:
LLGSolver: Solver for LLG dynamics
Author: Berkeley SciComp Team
Date: 2025
## Classes
### `LLGSolver`
Landau-Lifshitz-Gilbert equation solver.
#### Methods
##### `__init__(self, alpha, gamma)`
Initialize LLG solver.
Args:
alpha: Gilbert damping parameter
gamma: Gyromagnetic ratio (rad/s/T)
**Source:** [Line 22](Python/Spintronics/core/llg_dynamics.py#L22)
##### `llg_equation(self, t, m, H_eff)`
LLG equation RHS.
Args:
t: Time
m: Magnetization vector
H_eff: Effective field
Returns:
Time derivative dm/dt
**Source:** [Line 33](Python/Spintronics/core/llg_dynamics.py#L33)
##### `solve(self, m0, times, H_ext)`
Solve LLG dynamics.
Args:
m0: Initial magnetization
times: Time points
H_ext: External field
Returns:
Magnetization trajectory
**Source:** [Line 58](Python/Spintronics/core/llg_dynamics.py#L58)
**Class Source:** [Line 19](Python/Spintronics/core/llg_dynamics.py#L19)