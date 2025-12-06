"""Adaptive Methods for ODE and PDE Solvers.
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
"""
import numpy as np
from typing import Callable, Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from scipy.interpolate import interp1d
@dataclass
class AdaptiveResult:
    """Result of adaptive computation."""
    t: np.ndarray
    y: np.ndarray
    dt_history: np.ndarray
    error_history: np.ndarray
    rejections: int
    success: bool
    message: str
    total_steps: int = 0
    function_evaluations: int = 0
@dataclass
class RefinementResult:
    """Result of mesh refinement."""
    new_mesh: np.ndarray
    refined_solution: np.ndarray
    error_indicators: np.ndarray
    refinement_ratio: float
    success: bool
    message: str
class AdaptiveTimeStepper:
    """Adaptive time stepping for ODE integration."""
    def __init__(self, rtol: float = 1e-6, atol: float = 1e-8,
                 dt_min: float = 1e-12, dt_max: float = 1.0,
                 safety_factor: float = 0.9,
                 max_increase_factor: float = 5.0,
                 max_decrease_factor: float = 0.1):
        """Initialize adaptive time stepper.
        Args:
            rtol: Relative tolerance
            atol: Absolute tolerance
            dt_min: Minimum time step
            dt_max: Maximum time step
            safety_factor: Safety factor for step size adjustment
            max_increase_factor: Maximum step size increase factor
            max_decrease_factor: Maximum step size decrease factor
        """
        self.rtol = rtol
        self.atol = atol
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.safety_factor = safety_factor
        self.max_increase_factor = max_increase_factor
        self.max_decrease_factor = max_decrease_factor
        # Statistics
        self.rejections = 0
        self.total_steps = 0
        self.function_evaluations = 0
    def step_rk45(self, f: Callable, t: float, y: np.ndarray,
                  dt: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Adaptive Runge-Kutta 4(5) step with error estimation.
        Args:
            f: Right-hand side function
            t: Current time
            y: Current solution
            dt: Current time step
        Returns:
            Tuple of (y_new, error_estimate, dt_new)
        """
        # RK45 coefficients (Dormand-Prince)
        a = np.array([
            [0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
        ])
        b4 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
        b5 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
        c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
        # Compute RK stages
        k = np.zeros((7, len(y)))
        k[0] = f(t, y)
        for i in range(1, 6):
            y_temp = y + dt * np.sum(a[i, :i] * k[:i].T, axis=1)
            k[i] = f(t + c[i] * dt, y_temp)
        # 4th order solution
        y4 = y + dt * np.sum(b4 * k[:6].T, axis=1)
        # 5th order solution
        k[6] = f(t + dt, y4)
        y5 = y + dt * np.sum(b5 * k.T, axis=1)
        # Error estimate
        error = y5 - y4
        # Error norm
        scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y5))
        error_norm = np.sqrt(np.mean((error / scale)**2))
        # New time step
        if error_norm > 0:
            dt_new = dt * min(self.max_increase_factor,
                             max(self.max_decrease_factor,
                                 self.safety_factor * (1.0 / error_norm)**(1/5)))
        else:
            dt_new = dt * self.max_increase_factor
        dt_new = min(self.dt_max, max(self.dt_min, dt_new))
        self.function_evaluations += 7
        return y5, error, dt_new
    def integrate(self, f: Callable, t_span: Tuple[float, float],
                  y0: np.ndarray, dt0: float = 0.01) -> AdaptiveResult:
        """Integrate ODE with adaptive time stepping.
        Args:
            f: Right-hand side function dy/dt = f(t, y)
            t_span: Time span (t0, tf)
            y0: Initial condition
            dt0: Initial time step
        Returns:
            AdaptiveResult with solution
        """
        t0, tf = t_span
        t = t0
        y = y0.copy()
        dt = dt0
        # Storage
        times = [t]
        solutions = [y.copy()]
        dt_history = []
        error_history = []
        self.rejections = 0
        self.total_steps = 0
        while t < tf:
            # Ensure we don't overshoot
            if t + dt > tf:
                dt = tf - t
            # Take step
            y_new, error, dt_new = self.step_rk45(f, t, y, dt)
            # Error norm
            scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y_new))
            error_norm = np.sqrt(np.mean((error / scale)**2))
            if error_norm <= 1.0:
                # Accept step
                t += dt
                y = y_new
                times.append(t)
                solutions.append(y.copy())
                dt_history.append(dt)
                error_history.append(error_norm)
                self.total_steps += 1
            else:
                # Reject step
                self.rejections += 1
            # Update time step
            dt = dt_new
            # Safety check
            if dt < self.dt_min:
                warnings.warn(f"Time step {dt} below minimum {self.dt_min}")
                break
        return AdaptiveResult(
            t=np.array(times),
            y=np.array(solutions),
            dt_history=np.array(dt_history),
            error_history=np.array(error_history),
            rejections=self.rejections,
            success=True,
            message="Adaptive integration completed successfully",
            total_steps=self.total_steps,
            function_evaluations=self.function_evaluations
        )
class AdaptiveMeshRefiner:
    """Adaptive mesh refinement for PDEs."""
    def __init__(self, refinement_criterion: str = 'gradient',
                 refinement_threshold: float = 0.1,
                 max_refinement_levels: int = 5,
                 min_element_size: float = 1e-6):
        """Initialize adaptive mesh refiner.
        Args:
            refinement_criterion: Criterion for refinement ('gradient', 'residual', 'error')
            refinement_threshold: Threshold for refinement
            max_refinement_levels: Maximum refinement levels
            min_element_size: Minimum element size
        """
        self.refinement_criterion = refinement_criterion
        self.refinement_threshold = refinement_threshold
        self.max_refinement_levels = max_refinement_levels
        self.min_element_size = min_element_size
    def compute_error_indicators(self, mesh: np.ndarray,
                                solution: np.ndarray) -> np.ndarray:
        """Compute error indicators for mesh refinement.
        Args:
            mesh: Current mesh points
            solution: Solution values at mesh points
        Returns:
            Error indicators for each element
        """
        n_elements = len(mesh) - 1
        error_indicators = np.zeros(n_elements)
        if self.refinement_criterion == 'gradient':
            # Gradient-based refinement
            for i in range(n_elements):
                if i == 0:
                    # Forward difference
                    grad = (solution[i+1] - solution[i]) / (mesh[i+1] - mesh[i])
                elif i == n_elements - 1:
                    # Backward difference
                    grad = (solution[i+1] - solution[i]) / (mesh[i+1] - mesh[i])
                else:
                    # Central difference
                    h_left = mesh[i] - mesh[i-1]
                    h_right = mesh[i+1] - mesh[i]
                    grad_left = (solution[i] - solution[i-1]) / h_left
                    grad_right = (solution[i+1] - solution[i]) / h_right
                    grad = 0.5 * (grad_left + grad_right)
                # Element size weighted indicator
                h = mesh[i+1] - mesh[i]
                error_indicators[i] = h * abs(grad)
        elif self.refinement_criterion == 'second_derivative':
            # Second derivative based refinement
            for i in range(1, n_elements):
                h = mesh[i+1] - mesh[i]
                second_deriv = (solution[i+1] - 2*solution[i] + solution[i-1]) / h**2
                error_indicators[i] = h**2 * abs(second_deriv)
        elif self.refinement_criterion == 'residual':
            # Residual-based refinement (simplified)
            for i in range(n_elements):
                h = mesh[i+1] - mesh[i]
                # Simple residual estimate
                residual = abs(solution[i+1] - solution[i]) / h
                error_indicators[i] = h * residual
        return error_indicators
    def refine_mesh(self, mesh: np.ndarray, solution: np.ndarray,
                   max_new_points: int = 100) -> RefinementResult:
        """Refine mesh based on error indicators.
        Args:
            mesh: Current mesh points
            solution: Solution values at mesh points
            max_new_points: Maximum number of new points to add
        Returns:
            RefinementResult with refined mesh and solution
        """
        # Compute error indicators
        error_indicators = self.compute_error_indicators(mesh, solution)
        # Find elements to refine
        max_error = np.max(error_indicators)
        threshold = self.refinement_threshold * max_error
        refine_elements = error_indicators > threshold
        # Sort by error magnitude (refine worst elements first)
        refine_indices = np.where(refine_elements)[0]
        refine_indices = refine_indices[np.argsort(error_indicators[refine_indices])[::-1]]
        # Limit number of refined elements
        refine_indices = refine_indices[:max_new_points]
        if len(refine_indices) == 0:
            # No refinement needed
            return RefinementResult(
                new_mesh=mesh,
                refined_solution=solution,
                error_indicators=error_indicators,
                refinement_ratio=1.0,
                success=True,
                message="No refinement needed"
            )
        # Create new mesh with midpoint refinement
        new_points = []
        new_mesh_list = [mesh[0]]
        for i in range(len(mesh) - 1):
            new_mesh_list.append(mesh[i+1])
            if i in refine_indices:
                # Check minimum element size
                h_current = mesh[i+1] - mesh[i]
                if h_current > 2 * self.min_element_size:
                    # Add midpoint
                    midpoint = 0.5 * (mesh[i] + mesh[i+1])
                    new_points.append((len(new_mesh_list) - 1, midpoint))
        # Insert new points
        new_mesh_array = np.array(new_mesh_list)
        for insert_idx, point in reversed(new_points):
            new_mesh_array = np.insert(new_mesh_array, insert_idx, point)
        # Interpolate solution to new mesh
        interpolator = interp1d(mesh, solution, kind='cubic',
                               bounds_error=False, fill_value='extrapolate')
        refined_solution = interpolator(new_mesh_array)
        refinement_ratio = len(new_mesh_array) / len(mesh)
        return RefinementResult(
            new_mesh=new_mesh_array,
            refined_solution=refined_solution,
            error_indicators=error_indicators,
            refinement_ratio=refinement_ratio,
            success=True,
            message=f"Mesh refined: {len(mesh)} -> {len(new_mesh_array)} points"
        )
class ErrorEstimator:
    """Error estimation for ODE/PDE solutions."""
    def __init__(self, method: str = 'richardson'):
        """Initialize error estimator.
        Args:
            method: Error estimation method ('richardson', 'embedding', 'residual')
        """
        self.method = method
    def richardson_extrapolation(self, u_coarse: np.ndarray,
                                u_fine: np.ndarray,
                                refinement_ratio: int = 2,
                                order: int = 2) -> Dict[str, Any]:
        """Richardson extrapolation for error estimation.
        Args:
            u_coarse: Solution on coarse grid
            u_fine: Solution on fine grid (subsampled to match coarse)
            refinement_ratio: Grid refinement ratio
            order: Expected order of accuracy
        Returns:
            Dictionary with error estimates
        """
        if len(u_fine) != len(u_coarse):
            # Subsample fine grid solution
            indices = np.linspace(0, len(u_fine) - 1, len(u_coarse), dtype=int)
            u_fine_sub = u_fine[indices]
        else:
            u_fine_sub = u_fine
        # Richardson extrapolation
        # E ≈ (u_fine - u_coarse) / (r^p - 1)
        richardson_factor = refinement_ratio**order - 1
        error_estimate = (u_fine_sub - u_coarse) / richardson_factor
        # Improved solution estimate
        u_improved = u_fine_sub + error_estimate
        # Error norms
        max_error = np.max(np.abs(error_estimate))
        l2_error = np.sqrt(np.mean(error_estimate**2))
        relative_error = l2_error / np.sqrt(np.mean(u_fine_sub**2))
        return {
            'error_estimate': error_estimate,
            'improved_solution': u_improved,
            'max_error': max_error,
            'l2_error': l2_error,
            'relative_error': relative_error,
            'richardson_factor': richardson_factor
        }
    def embedded_error_estimate(self, y_low: np.ndarray,
                               y_high: np.ndarray) -> Dict[str, Any]:
        """Error estimate from embedded method.
        Args:
            y_low: Lower order solution
            y_high: Higher order solution
        Returns:
            Dictionary with error estimates
        """
        error = y_high - y_low
        max_error = np.max(np.abs(error))
        l2_error = np.sqrt(np.mean(error**2))
        relative_error = l2_error / np.sqrt(np.mean(y_high**2))
        return {
            'error_estimate': error,
            'max_error': max_error,
            'l2_error': l2_error,
            'relative_error': relative_error
        }
    def residual_error_estimate(self, residual: np.ndarray,
                               operator_norm: float = 1.0) -> Dict[str, Any]:
        """Residual-based error estimate.
        Args:
            residual: Residual vector
            operator_norm: Estimate of operator norm
        Returns:
            Dictionary with error estimates
        """
        residual_norm = np.sqrt(np.mean(residual**2))
        error_estimate = residual_norm / operator_norm
        return {
            'residual_norm': residual_norm,
            'error_estimate': error_estimate,
            'operator_norm': operator_norm
        }
# Utility functions
def adaptive_rk_step(f: Callable, t: float, y: np.ndarray, dt: float,
                    rtol: float = 1e-6, atol: float = 1e-8) -> Tuple[np.ndarray, bool, float]:
    """Single adaptive Runge-Kutta step.
    Args:
        f: Right-hand side function
        t: Current time
        y: Current solution
        dt: Current time step
        rtol: Relative tolerance
        atol: Absolute tolerance
    Returns:
        Tuple of (y_new, accept, dt_new)
    """
    stepper = AdaptiveTimeStepper(rtol=rtol, atol=atol)
    y_new, error, dt_new = stepper.step_rk45(f, t, y, dt)
    # Error norm
    scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
    error_norm = np.sqrt(np.mean((error / scale)**2))
    accept = error_norm <= 1.0
    return y_new, accept, dt_new
def estimate_local_error(y1: np.ndarray, y2: np.ndarray,
                        order: int = 4) -> float:
    """Estimate local truncation error.
    Args:
        y1: Solution from method of order p
        y2: Solution from method of order p+1
        order: Order of lower order method
    Returns:
        Local error estimate
    """
    error = y2 - y1
    return np.sqrt(np.mean(error**2))
def compute_optimal_timestep(error: float, dt_current: float,
                            tolerance: float, order: int = 4,
                            safety_factor: float = 0.9) -> float:
    """Compute optimal time step based on error estimate.
    Args:
        error: Current error estimate
        dt_current: Current time step
        tolerance: Error tolerance
        order: Method order
        safety_factor: Safety factor
    Returns:
        Optimal time step
    """
    if error == 0:
        return dt_current * 2.0
    factor = safety_factor * (tolerance / error)**(1.0 / (order + 1))
    factor = min(2.0, max(0.1, factor))  # Limit step size changes
    return dt_current * factor
def refine_mesh_uniform(mesh: np.ndarray, refinement_factor: int = 2) -> np.ndarray:
    """Uniform mesh refinement.
    Args:
        mesh: Current mesh
        refinement_factor: Refinement factor
    Returns:
        Refined mesh
    """
    new_points = []
    for i in range(len(mesh) - 1):
        new_points.append(mesh[i])
        # Add intermediate points
        for j in range(1, refinement_factor):
            alpha = j / refinement_factor
            new_point = (1 - alpha) * mesh[i] + alpha * mesh[i + 1]
            new_points.append(new_point)
    new_points.append(mesh[-1])
    return np.array(new_points)
def estimate_convergence_rate(errors: List[float],
                             step_sizes: List[float]) -> float:
    """Estimate convergence rate from error sequence.
    Args:
        errors: Sequence of errors
        step_sizes: Corresponding step sizes
    Returns:
        Estimated convergence rate
    """
    if len(errors) < 2:
        return 0.0
    # Fit log(error) vs log(h)
    log_errors = np.log(errors)
    log_h = np.log(step_sizes)
    # Linear regression
    A = np.vstack([log_h, np.ones(len(log_h))]).T
    rate, _ = np.linalg.lstsq(A, log_errors, rcond=None)[0]
    return rate
def adaptive_pde_solve(pde_solver: Callable, initial_mesh: np.ndarray,
                      initial_solution: np.ndarray,
                      target_error: float = 1e-6,
                      max_iterations: int = 10) -> Dict[str, Any]:
    """Adaptive PDE solving with mesh refinement.
    Args:
        pde_solver: PDE solver function
        initial_mesh: Initial mesh
        initial_solution: Initial solution
        target_error: Target error tolerance
        max_iterations: Maximum refinement iterations
    Returns:
        Dictionary with final solution and mesh
    """
    refiner = AdaptiveMeshRefiner()
    current_mesh = initial_mesh
    current_solution = initial_solution
    for iteration in range(max_iterations):
        # Refine mesh
        refinement_result = refiner.refine_mesh(current_mesh, current_solution)
        if not refinement_result.success:
            break
        # Solve on refined mesh
        refined_solution = pde_solver(refinement_result.new_mesh)
        # Estimate error
        error_indicators = refiner.compute_error_indicators(
            refinement_result.new_mesh, refined_solution)
        max_error = np.max(error_indicators)
        # Update for next iteration
        current_mesh = refinement_result.new_mesh
        current_solution = refined_solution
        # Check convergence
        if max_error < target_error:
            break
    return {
        'mesh': current_mesh,
        'solution': current_solution,
        'error_indicators': error_indicators,
        'iterations': iteration + 1,
        'converged': max_error < target_error
    }