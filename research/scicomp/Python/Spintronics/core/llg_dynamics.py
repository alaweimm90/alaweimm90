"""
Landau-Lifshitz-Gilbert dynamics for magnetic moments.
This module provides tools for simulating magnetization dynamics using
the Landau-Lifshitz-Gilbert equation.
Classes:
    LLGSolver: Solver for LLG dynamics
Author: Berkeley SciComp Team
Date: 2025
"""
import numpy as np
from typing import Union, List, Tuple, Optional, Callable
from scipy.integrate import solve_ivp
class LLGSolver:
    """Landau-Lifshitz-Gilbert equation solver."""
    def __init__(self, alpha: float = 0.1, gamma: float = 1.76e11):
        """
        Initialize LLG solver.
        Args:
            alpha: Gilbert damping parameter
            gamma: Gyromagnetic ratio (rad/s/T)
        """
        self.alpha = alpha
        self.gamma = gamma
    def llg_equation(self, t: float, m: np.ndarray, H_eff: np.ndarray) -> np.ndarray:
        """
        LLG equation RHS.
        Args:
            t: Time
            m: Magnetization vector
            H_eff: Effective field
        Returns:
            Time derivative dm/dt
        """
        # Normalize magnetization
        m_norm = np.linalg.norm(m)
        if m_norm > 0:
            m = m / m_norm
        # LLG equation: dm/dt = -γ m × H_eff - α γ m × (m × H_eff)
        mxH = np.cross(m, H_eff)
        mxmxH = np.cross(m, mxH)
        dmdt = -self.gamma * mxH - self.alpha * self.gamma * mxmxH
        return dmdt
    def solve(self,
              m0: np.ndarray,
              times: np.ndarray,
              H_ext: np.ndarray) -> np.ndarray:
        """
        Solve LLG dynamics.
        Args:
            m0: Initial magnetization
            times: Time points
            H_ext: External field
        Returns:
            Magnetization trajectory
        """
        # Normalize initial condition
        m0 = m0 / np.linalg.norm(m0)
        def rhs(t, y):
            return self.llg_equation(t, y, H_ext)
        # Solve ODE
        sol = solve_ivp(rhs, [times[0], times[-1]], m0,
                       t_eval=times, method='RK45', rtol=1e-8)
        # Ensure normalization is preserved
        for i in range(sol.y.shape[1]):
            norm = np.linalg.norm(sol.y[:, i])
            if norm > 0:
                sol.y[:, i] /= norm
        return sol.y.T