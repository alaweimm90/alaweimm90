"""
Heat equation solvers for thermal transport simulations.
This module provides numerical solvers for the heat equation using
finite difference and finite element methods.
Classes:
    HeatEquationSolver1D: 1D heat equation solver
    HeatEquationSolver2D: 2D heat equation solver
Author: Berkeley SciComp Team
Date: 2025
"""
import numpy as np
from typing import Union, Optional, Callable
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
class HeatEquationSolver1D:
    """One-dimensional heat equation solver."""
    def __init__(self, L: float, nx: int, alpha: float = 1.0):
        """
        Initialize 1D heat equation solver.
        Args:
            L: Domain length
            nx: Number of grid points
            alpha: Thermal diffusivity
        """
        self.L = L
        self.nx = nx
        self.alpha = alpha
        # Grid setup
        self.dx = L / (nx - 1)
        self.grid = np.linspace(0, L, nx)
    def solve(self,
              T_initial: np.ndarray,
              t_final: float,
              dt: float,
              boundary_conditions: str = "dirichlet",
              T_left: float = 0.0,
              T_right: float = 0.0) -> np.ndarray:
        """
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
        """
        T = T_initial.copy()
        t = 0.0
        # Stability check
        r = self.alpha * dt / self.dx**2
        if r > 0.5:
            print(f"Warning: CFL condition violated (r = {r:.3f} > 0.5)")
        # Time stepping
        while t < t_final:
            T_new = T.copy()
            # Internal points (explicit finite difference)
            for i in range(1, self.nx - 1):
                T_new[i] = T[i] + r * (T[i+1] - 2*T[i] + T[i-1])
            # Boundary conditions
            if boundary_conditions == "dirichlet":
                T_new[0] = T_left
                T_new[-1] = T_right
            elif boundary_conditions == "neumann":
                # Zero flux at boundaries
                T_new[0] = T_new[1]
                T_new[-1] = T_new[-2]
            T = T_new
            t += dt
        return T