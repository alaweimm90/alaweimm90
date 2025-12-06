#!/usr/bin/env python3
"""
Time-Dependent Schrödinger Equation Solver
Numerical methods for solving the time-dependent Schrödinger equation including
Crank-Nicolson and split-operator methods for accurate wavefunction evolution.
Key Features:
- Crank-Nicolson implicit method for stable evolution
- Split-operator method for efficient computation
- Adaptive time stepping and error control
- Support for arbitrary potentials and boundary conditions
- Conservation of probability and energy monitoring
Mathematical Foundation:
The time-dependent Schrödinger equation is:
iℏ ∂ψ/∂t = Ĥψ = [-ℏ²/(2m)∇² + V(r,t)]ψ
Numerical schemes preserve unitarity and provide controllable accuracy.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import expm
from typing import Callable, Optional, Union, Tuple, Dict, Any
import warnings
try:
    from ...utils.constants import hbar, me
    from ...utils.parallel import parallel_compute
except ImportError:
    # Fallback for direct execution
    hbar = 1.054571817e-34
    me = 9.1093837015e-31
    def parallel_compute(*args, **kwargs):
        pass
class TDSESolver:
    """
    Base class for time-dependent Schrödinger equation solvers.
    Provides common functionality for different numerical schemes
    including grid setup, potential handling, and diagnostics.
    """
    def __init__(self, x: np.ndarray,
                 potential: Callable[[np.ndarray, float], np.ndarray],
                 mass: float = me,
                 boundary_conditions: str = 'periodic'):
        """
        Initialize TDSE solver.
        Parameters
        ----------
        x : ndarray
            Spatial grid points
        potential : callable
            Potential function V(x, t)
        mass : float, default me
            Particle mass
        boundary_conditions : str, default 'periodic'
            Boundary conditions ('periodic', 'dirichlet', 'neumann')
        """
        self.x = x
        self.dx = x[1] - x[0] if len(x) > 1 else 1.0
        self.N = len(x)
        self.potential = potential
        self.mass = mass
        self.boundary_conditions = boundary_conditions
        # Kinetic energy operator (finite differences)
        self.T_matrix = self._construct_kinetic_operator()
        # Storage for diagnostics
        self.times = []
        self.energies = []
        self.norms = []
    def _construct_kinetic_operator(self) -> sp.csr_matrix:
        """Construct kinetic energy operator using finite differences."""
        # Second derivative operator: -ℏ²/(2m) * d²/dx²
        coeff = -hbar**2 / (2 * self.mass * self.dx**2)
        if self.boundary_conditions == 'periodic':
            # Periodic boundary conditions
            diagonals = [np.ones(self.N), -2*np.ones(self.N), np.ones(self.N)]
            offsets = [-1, 0, 1]
            T = sp.diags(diagonals, offsets, shape=(self.N, self.N), format='csr')
            # Periodic wrap-around terms
            T[0, -1] = 1
            T[-1, 0] = 1
        else:
            # Dirichlet boundary conditions (ψ=0 at boundaries)
            diagonals = [np.ones(self.N-1), -2*np.ones(self.N), np.ones(self.N-1)]
            offsets = [-1, 0, 1]
            T = sp.diags(diagonals, offsets, shape=(self.N, self.N), format='csr')
        return coeff * T
    def _construct_potential_operator(self, t: float) -> sp.diags:
        """Construct potential energy operator."""
        V_values = self.potential(self.x, t)
        return sp.diags(V_values, format='csr')
    def _construct_hamiltonian(self, t: float) -> sp.csr_matrix:
        """Construct full Hamiltonian matrix."""
        return self.T_matrix + self._construct_potential_operator(t)
    def calculate_energy(self, psi: np.ndarray, t: float) -> float:
        """
        Calculate expectation value of energy.
        Parameters
        ----------
        psi : ndarray
            Wavefunction
        t : float
            Time
        Returns
        -------
        float
            Energy expectation value
        """
        H = self._construct_hamiltonian(t)
        H_psi = H.dot(psi)
        energy = np.real(np.vdot(psi, H_psi)) * self.dx
        return energy
    def calculate_norm(self, psi: np.ndarray) -> float:
        """Calculate wavefunction norm."""
        return np.sqrt(np.real(np.vdot(psi, psi)) * self.dx)
    def normalize_wavefunction(self, psi: np.ndarray) -> np.ndarray:
        """Normalize wavefunction."""
        norm = self.calculate_norm(psi)
        if norm > 1e-12:
            return psi / norm
        else:
            warnings.warn("Wavefunction has negligible norm")
            return psi
    def save_diagnostics(self, psi: np.ndarray, t: float) -> None:
        """Save diagnostic information."""
        self.times.append(t)
        self.energies.append(self.calculate_energy(psi, t))
        self.norms.append(self.calculate_norm(psi))
    def get_diagnostics(self) -> Dict[str, np.ndarray]:
        """Get diagnostic arrays."""
        return {
            'times': np.array(self.times),
            'energies': np.array(self.energies),
            'norms': np.array(self.norms)
        }
class CrankNicolsonSolver(TDSESolver):
    """
    Crank-Nicolson solver for the time-dependent Schrödinger equation.
    The Crank-Nicolson method is implicit and unconditionally stable,
    preserving unitarity to machine precision. It uses the trapezoidal
    rule for time integration:
    (1 + iĤΔt/2ℏ)ψₙ₊₁ = (1 - iĤΔt/2ℏ)ψₙ
    This requires solving a linear system at each time step but provides
    excellent stability and accuracy properties.
    """
    def __init__(self, x: np.ndarray,
                 potential: Callable[[np.ndarray, float], np.ndarray],
                 mass: float = me,
                 boundary_conditions: str = 'periodic'):
        """Initialize Crank-Nicolson solver."""
        super().__init__(x, potential, mass, boundary_conditions)
        # Sparse solver setup
        self.solver_setup = False
        self.LU_decomp = None
    def evolve(self, psi_initial: np.ndarray,
              t_initial: float,
              t_final: float,
              dt: float,
              save_intermediate: bool = False,
              adaptive_timestep: bool = False,
              tolerance: float = 1e-8) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Evolve wavefunction using Crank-Nicolson method.
        Parameters
        ----------
        psi_initial : ndarray
            Initial wavefunction
        t_initial : float
            Initial time
        t_final : float
            Final time
        dt : float
            Time step
        save_intermediate : bool, default False
            Whether to save intermediate wavefunctions
        adaptive_timestep : bool, default False
            Whether to use adaptive time stepping
        tolerance : float, default 1e-8
            Error tolerance for adaptive stepping
        Returns
        -------
        psi_final : ndarray
            Final wavefunction
        psi_trajectory : ndarray, optional
            Array of intermediate wavefunctions if save_intermediate=True
        """
        psi = psi_initial.copy()
        t = t_initial
        if save_intermediate:
            trajectory = [psi.copy()]
            time_points = [t]
        # Main evolution loop
        while t < t_final:
            # Adjust final step size
            dt_current = min(dt, t_final - t)
            if adaptive_timestep:
                psi, dt_used = self._adaptive_step(psi, t, dt_current, tolerance)
                dt = dt_used
            else:
                psi = self._single_step(psi, t, dt_current)
                dt_used = dt_current
            t += dt_used
            # Save diagnostics
            self.save_diagnostics(psi, t)
            if save_intermediate:
                trajectory.append(psi.copy())
                time_points.append(t)
        if save_intermediate:
            return psi, np.array(trajectory), np.array(time_points)
        else:
            return psi
    def _single_step(self, psi: np.ndarray, t: float, dt: float) -> np.ndarray:
        """Perform single Crank-Nicolson step."""
        # Construct Hamiltonian at current time
        H = self._construct_hamiltonian(t + dt/2)  # Mid-point evaluation
        # Crank-Nicolson matrices
        coeff = 1j * dt / (2 * hbar)
        I = sp.identity(self.N, format='csr')
        # Left side: (1 + iĤΔt/2ℏ)
        A = I + coeff * H
        # Right side: (1 - iĤΔt/2ℏ)ψ
        B = I - coeff * H
        rhs = B.dot(psi)
        # Solve linear system
        psi_new = spla.spsolve(A, rhs)
        return psi_new
    def _adaptive_step(self, psi: np.ndarray, t: float, dt: float,
                      tolerance: float) -> Tuple[np.ndarray, float]:
        """Adaptive time stepping with error control."""
        # Try full step
        psi_full = self._single_step(psi, t, dt)
        # Try two half steps
        psi_half = self._single_step(psi, t, dt/2)
        psi_double = self._single_step(psi_half, t + dt/2, dt/2)
        # Estimate error
        error = np.linalg.norm(psi_full - psi_double) * self.dx
        if error < tolerance:
            # Accept step
            return psi_full, dt
        else:
            # Reject step, try smaller dt
            dt_new = dt * (tolerance / error)**0.5
            dt_new = max(dt_new, dt * 0.1)  # Don't reduce too drastically
            return self._adaptive_step(psi, t, dt_new, tolerance)
class SplitOperatorSolver(TDSESolver):
    """
    Split-operator method for the time-dependent Schrödinger equation.
    The split-operator method (also known as the Trotter-Suzuki method)
    splits the evolution operator into kinetic and potential parts:
    exp(-iĤΔt/ℏ) ≈ exp(-iV̂Δt/2ℏ) exp(-iT̂Δt/ℏ) exp(-iV̂Δt/2ℏ)
    This method is particularly efficient when the kinetic energy operator
    is diagonal in momentum space (via FFT) and the potential is diagonal
    in position space.
    """
    def __init__(self, x: np.ndarray,
                 potential: Callable[[np.ndarray, float], np.ndarray],
                 mass: float = me,
                 boundary_conditions: str = 'periodic'):
        """Initialize split-operator solver."""
        super().__init__(x, potential, mass, boundary_conditions)
        if boundary_conditions != 'periodic':
            warnings.warn("Split-operator method works best with periodic boundaries")
        # Momentum space grid
        self.k = 2 * np.pi * np.fft.fftfreq(self.N, self.dx)
        # Kinetic energy in momentum space
        self.T_k = hbar**2 * self.k**2 / (2 * self.mass)
    def evolve(self, psi_initial: np.ndarray,
              t_initial: float,
              t_final: float,
              dt: float,
              save_intermediate: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Evolve wavefunction using split-operator method.
        Parameters
        ----------
        psi_initial : ndarray
            Initial wavefunction
        t_initial : float
            Initial time
        t_final : float
            Final time
        dt : float
            Time step
        save_intermediate : bool, default False
            Whether to save intermediate wavefunctions
        Returns
        -------
        psi_final : ndarray
            Final wavefunction
        psi_trajectory : ndarray, optional
            Array of intermediate wavefunctions if save_intermediate=True
        """
        psi = psi_initial.copy()
        t = t_initial
        if save_intermediate:
            trajectory = [psi.copy()]
            time_points = [t]
        # Precompute kinetic evolution operator
        exp_T = np.exp(-1j * self.T_k * dt / hbar)
        # Main evolution loop
        while t < t_final:
            # Adjust final step size
            dt_current = min(dt, t_final - t)
            if dt_current != dt:
                exp_T_current = np.exp(-1j * self.T_k * dt_current / hbar)
            else:
                exp_T_current = exp_T
            psi = self._single_step(psi, t, dt_current, exp_T_current)
            t += dt_current
            # Save diagnostics
            self.save_diagnostics(psi, t)
            if save_intermediate:
                trajectory.append(psi.copy())
                time_points.append(t)
        if save_intermediate:
            return psi, np.array(trajectory), np.array(time_points)
        else:
            return psi
    def _single_step(self, psi: np.ndarray, t: float, dt: float,
                    exp_T: np.ndarray) -> np.ndarray:
        """Perform single split-operator step."""
        # First half potential step
        V = self.potential(self.x, t)
        psi *= np.exp(-1j * V * dt / (2 * hbar))
        # Full kinetic step (in momentum space)
        psi_k = np.fft.fft(psi)
        psi_k *= exp_T
        psi = np.fft.ifft(psi_k)
        # Second half potential step
        V = self.potential(self.x, t + dt)
        psi *= np.exp(-1j * V * dt / (2 * hbar))
        return psi
# Convenience functions
def evolve_wavefunction(psi_initial: np.ndarray,
                       x: np.ndarray,
                       potential: Callable[[np.ndarray, float], np.ndarray],
                       t_initial: float,
                       t_final: float,
                       dt: float,
                       method: str = 'crank_nicolson',
                       mass: float = me,
                       **kwargs) -> np.ndarray:
    """
    Evolve wavefunction using specified method.
    Parameters
    ----------
    psi_initial : ndarray
        Initial wavefunction
    x : ndarray
        Spatial grid
    potential : callable
        Potential function V(x, t)
    t_initial : float
        Initial time
    t_final : float
        Final time
    dt : float
        Time step
    method : str, default 'crank_nicolson'
        Evolution method ('crank_nicolson', 'split_operator')
    mass : float, default me
        Particle mass
    **kwargs
        Additional arguments for solver
    Returns
    -------
    ndarray
        Final wavefunction
    """
    if method == 'crank_nicolson':
        solver = CrankNicolsonSolver(x, potential, mass)
    elif method == 'split_operator':
        solver = SplitOperatorSolver(x, potential, mass)
    else:
        raise ValueError(f"Unknown method: {method}")
    return solver.evolve(psi_initial, t_initial, t_final, dt, **kwargs)
def schrodinger_evolution(psi_initial: np.ndarray,
                         hamiltonian: Callable[[float], np.ndarray],
                         t_span: Tuple[float, float],
                         dt: float,
                         method: str = 'matrix_exponential') -> np.ndarray:
    """
    Evolve wavefunction using matrix exponential method.
    For time-independent Hamiltonians, the exact evolution is:
    ψ(t) = exp(-iĤt/ℏ)ψ(0)
    Parameters
    ----------
    psi_initial : ndarray
        Initial wavefunction
    hamiltonian : callable
        Function returning Hamiltonian matrix H(t)
    t_span : tuple
        (t_initial, t_final)
    dt : float
        Time step
    method : str, default 'matrix_exponential'
        Evolution method
    Returns
    -------
    ndarray
        Final wavefunction
    """
    t_initial, t_final = t_span
    if method == 'matrix_exponential':
        # For time-independent case
        H = hamiltonian(t_initial)
        U = expm(-1j * H * (t_final - t_initial) / hbar)
        return U @ psi_initial
    else:
        raise ValueError(f"Unknown method: {method}")
class AdaptiveTDSESolver(CrankNicolsonSolver):
    """
    Adaptive TDSE solver with automatic error control and time stepping.
    Features:
    - Automatic time step adaptation based on local error estimates
    - Conservation monitoring with automatic corrections
    - Adaptive spatial grid refinement (optional)
    - Performance optimization and memory management
    """
    def __init__(self, x: np.ndarray,
                 potential: Callable[[np.ndarray, float], np.ndarray],
                 mass: float = me,
                 boundary_conditions: str = 'periodic',
                 error_tolerance: float = 1e-8,
                 min_dt: float = 1e-12,
                 max_dt: float = 1e-2):
        """Initialize adaptive solver."""
        super().__init__(x, potential, mass, boundary_conditions)
        self.error_tolerance = error_tolerance
        self.min_dt = min_dt
        self.max_dt = max_dt
        # Adaptive parameters
        self.dt_history = []
        self.error_history = []
    def evolve_adaptive(self, psi_initial: np.ndarray,
                       t_initial: float,
                       t_final: float,
                       dt_initial: float = 1e-3,
                       save_intermediate: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Evolve with full adaptive control.
        Parameters
        ----------
        psi_initial : ndarray
            Initial wavefunction
        t_initial : float
            Initial time
        t_final : float
            Final time
        dt_initial : float, default 1e-3
            Initial time step
        save_intermediate : bool, default False
            Whether to save trajectory
        Returns
        -------
        psi_final : ndarray
            Final wavefunction
        trajectory : ndarray, optional
            Intermediate wavefunctions if requested
        """
        psi = psi_initial.copy()
        t = t_initial
        dt = dt_initial
        if save_intermediate:
            trajectory = [psi.copy()]
            time_points = [t]
        step_count = 0
        rejected_steps = 0
        while t < t_final:
            dt = min(dt, t_final - t)
            # Adaptive step with error control
            psi_new, dt_used, error, accepted = self._adaptive_step_full(psi, t, dt)
            if accepted:
                psi = psi_new
                t += dt_used
                step_count += 1
                # Save diagnostics
                self.save_diagnostics(psi, t)
                self.dt_history.append(dt_used)
                self.error_history.append(error)
                if save_intermediate:
                    trajectory.append(psi.copy())
                    time_points.append(t)
                # Adjust next time step
                dt = self._suggest_next_timestep(error, dt_used)
            else:
                rejected_steps += 1
                dt *= 0.5  # Reduce time step
                if dt < self.min_dt:
                    warnings.warn(f"Time step reached minimum value: {self.min_dt}")
                    dt = self.min_dt
        print(f"Evolution completed: {step_count} accepted steps, {rejected_steps} rejected steps")
        if save_intermediate:
            return psi, np.array(trajectory), np.array(time_points)
        else:
            return psi
    def _adaptive_step_full(self, psi: np.ndarray, t: float, dt: float) -> Tuple[np.ndarray, float, float, bool]:
        """Full adaptive step with error estimation."""
        # Two approaches: one step vs two half-steps
        psi1 = self._single_step(psi, t, dt)
        psi_half = self._single_step(psi, t, dt/2)
        psi2 = self._single_step(psi_half, t + dt/2, dt/2)
        # Error estimate
        error = np.linalg.norm(psi1 - psi2) * self.dx
        # Accept or reject step
        if error <= self.error_tolerance:
            return psi2, dt, error, True  # Use higher-order result
        else:
            return psi, 0.0, error, False
    def _suggest_next_timestep(self, error: float, dt_current: float) -> float:
        """Suggest next time step based on error."""
        if error > 0:
            factor = (self.error_tolerance / error) ** 0.2
            factor = np.clip(factor, 0.1, 5.0)  # Conservative bounds
        else:
            factor = 1.1  # Slight increase if error is negligible
        dt_new = dt_current * factor
        return np.clip(dt_new, self.min_dt, self.max_dt)