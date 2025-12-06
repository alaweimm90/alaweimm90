"""Nonlinear Solvers for ODEs and PDEs.
This module provides comprehensive nonlinear solving capabilities including
Newton methods, continuation methods, and specialized solvers for nonlinear
differential equations.
Classes:
    NonlinearSolver: Base class for nonlinear solvers
    NewtonSolver: Newton-Raphson method
    DampedNewtonSolver: Damped Newton method
    ContinuationSolver: Continuation/homotopy methods
    FixedPointSolver: Fixed point iteration
Functions:
    newton_raphson: Newton-Raphson solver
    solve_nonlinear_ode: Nonlinear ODE solver
    solve_nonlinear_pde: Nonlinear PDE solver
    continuation_method: Parameter continuation
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
from typing import Callable, Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from scipy.linalg import solve, norm
from scipy.sparse import linalg as spla
from scipy.optimize import fsolve, root
import time
@dataclass
class NonlinearResult:
    """Result of nonlinear solver."""
    x: np.ndarray
    success: bool
    iterations: int
    residual_norm: float
    convergence_history: List[float]
    solve_time: float
    method: str
    message: str
    function_evaluations: int = 0
    jacobian_evaluations: int = 0
@dataclass
class ContinuationResult:
    """Result of continuation method."""
    parameter_values: np.ndarray
    solutions: np.ndarray
    success: bool
    convergence_data: List[NonlinearResult]
    bifurcation_points: List[int]
    method: str
    message: str
class NonlinearSolver(ABC):
    """Abstract base class for nonlinear solvers."""
    def __init__(self, tolerance: float = 1e-8, max_iterations: int = 100):
        """Initialize nonlinear solver.
        Args:
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        # Statistics
        self.function_evaluations = 0
        self.jacobian_evaluations = 0
    @abstractmethod
    def solve(self, f: Callable, x0: np.ndarray,
              jacobian: Optional[Callable] = None) -> NonlinearResult:
        """Solve nonlinear system F(x) = 0.
        Args:
            f: Nonlinear function F(x)
            x0: Initial guess
            jacobian: Jacobian function dF/dx (optional)
        Returns:
            NonlinearResult
        """
        pass
    def finite_difference_jacobian(self, f: Callable, x: np.ndarray,
                                  eps: float = 1e-8) -> np.ndarray:
        """Compute Jacobian using finite differences.
        Args:
            f: Function F(x)
            x: Point to evaluate Jacobian
            eps: Finite difference step size
        Returns:
            Jacobian matrix
        """
        n = len(x)
        f_x = f(x)
        m = len(f_x)
        J = np.zeros((m, n))
        for j in range(n):
            x_plus = x.copy()
            x_plus[j] += eps
            f_plus = f(x_plus)
            J[:, j] = (f_plus - f_x) / eps
            self.function_evaluations += 1
        return J
class NewtonSolver(NonlinearSolver):
    """Newton-Raphson method for nonlinear systems."""
    def __init__(self, tolerance: float = 1e-8, max_iterations: int = 100,
                 use_line_search: bool = False):
        """Initialize Newton solver.
        Args:
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            use_line_search: Whether to use line search
        """
        super().__init__(tolerance, max_iterations)
        self.use_line_search = use_line_search
    def solve(self, f: Callable, x0: np.ndarray,
              jacobian: Optional[Callable] = None) -> NonlinearResult:
        """Solve using Newton-Raphson method.
        Args:
            f: Nonlinear function F(x)
            x0: Initial guess
            jacobian: Jacobian function (optional)
        Returns:
            NonlinearResult
        """
        start_time = time.time()
        x = x0.copy()
        convergence_history = []
        self.function_evaluations = 0
        self.jacobian_evaluations = 0
        for iteration in range(self.max_iterations):
            # Evaluate function
            f_x = f(x)
            self.function_evaluations += 1
            # Check convergence
            residual_norm = norm(f_x)
            convergence_history.append(residual_norm)
            if residual_norm < self.tolerance:
                return NonlinearResult(
                    x=x,
                    success=True,
                    iterations=iteration,
                    residual_norm=residual_norm,
                    convergence_history=convergence_history,
                    solve_time=time.time() - start_time,
                    method="Newton-Raphson",
                    message=f"Converged in {iteration} iterations",
                    function_evaluations=self.function_evaluations,
                    jacobian_evaluations=self.jacobian_evaluations
                )
            # Compute Jacobian
            if jacobian is not None:
                J = jacobian(x)
                self.jacobian_evaluations += 1
            else:
                J = self.finite_difference_jacobian(f, x)
                self.jacobian_evaluations += 1
            # Check for singular Jacobian
            try:
                # Solve J * delta_x = -f_x
                delta_x = solve(J, -f_x)
            except np.linalg.LinAlgError:
                return NonlinearResult(
                    x=x,
                    success=False,
                    iterations=iteration,
                    residual_norm=residual_norm,
                    convergence_history=convergence_history,
                    solve_time=time.time() - start_time,
                    method="Newton-Raphson",
                    message="Singular Jacobian encountered",
                    function_evaluations=self.function_evaluations,
                    jacobian_evaluations=self.jacobian_evaluations
                )
            # Line search (optional)
            if self.use_line_search:
                alpha = self._line_search(f, x, f_x, delta_x)
            else:
                alpha = 1.0
            # Update solution
            x = x + alpha * delta_x
        # Max iterations reached
        return NonlinearResult(
            x=x,
            success=False,
            iterations=self.max_iterations,
            residual_norm=residual_norm,
            convergence_history=convergence_history,
            solve_time=time.time() - start_time,
            method="Newton-Raphson",
            message="Maximum iterations reached",
            function_evaluations=self.function_evaluations,
            jacobian_evaluations=self.jacobian_evaluations
        )
    def _line_search(self, f: Callable, x: np.ndarray, f_x: np.ndarray,
                    delta_x: np.ndarray, alpha_max: float = 1.0) -> float:
        """Simple backtracking line search.
        Args:
            f: Function
            x: Current point
            f_x: Function value at x
            delta_x: Search direction
            alpha_max: Maximum step size
        Returns:
            Step size alpha
        """
        alpha = alpha_max
        rho = 0.5  # Backtracking factor
        c1 = 1e-4  # Armijo constant
        phi_0 = 0.5 * norm(f_x)**2
        phi_prime_0 = np.dot(f_x, f(x + 1e-8 * delta_x) - f_x) / 1e-8
        for _ in range(10):  # Max line search iterations
            x_new = x + alpha * delta_x
            f_new = f(x_new)
            phi_alpha = 0.5 * norm(f_new)**2
            # Armijo condition
            if phi_alpha <= phi_0 + c1 * alpha * phi_prime_0:
                break
            alpha *= rho
        return alpha
class DampedNewtonSolver(NonlinearSolver):
    """Damped Newton method with adaptive damping."""
    def __init__(self, tolerance: float = 1e-8, max_iterations: int = 100,
                 initial_damping: float = 1.0, min_damping: float = 1e-6):
        """Initialize damped Newton solver.
        Args:
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            initial_damping: Initial damping factor
            min_damping: Minimum damping factor
        """
        super().__init__(tolerance, max_iterations)
        self.initial_damping = initial_damping
        self.min_damping = min_damping
    def solve(self, f: Callable, x0: np.ndarray,
              jacobian: Optional[Callable] = None) -> NonlinearResult:
        """Solve using damped Newton method.
        Args:
            f: Nonlinear function F(x)
            x0: Initial guess
            jacobian: Jacobian function (optional)
        Returns:
            NonlinearResult
        """
        start_time = time.time()
        x = x0.copy()
        convergence_history = []
        damping = self.initial_damping
        self.function_evaluations = 0
        self.jacobian_evaluations = 0
        for iteration in range(self.max_iterations):
            # Evaluate function
            f_x = f(x)
            self.function_evaluations += 1
            # Check convergence
            residual_norm = norm(f_x)
            convergence_history.append(residual_norm)
            if residual_norm < self.tolerance:
                return NonlinearResult(
                    x=x,
                    success=True,
                    iterations=iteration,
                    residual_norm=residual_norm,
                    convergence_history=convergence_history,
                    solve_time=time.time() - start_time,
                    method="Damped Newton",
                    message=f"Converged in {iteration} iterations",
                    function_evaluations=self.function_evaluations,
                    jacobian_evaluations=self.jacobian_evaluations
                )
            # Compute Jacobian
            if jacobian is not None:
                J = jacobian(x)
                self.jacobian_evaluations += 1
            else:
                J = self.finite_difference_jacobian(f, x)
                self.jacobian_evaluations += 1
            # Solve with damping: (J + λI) * delta_x = -f_x
            try:
                # Add damping to diagonal
                J_damped = J + damping * np.eye(len(x))
                delta_x = solve(J_damped, -f_x)
            except np.linalg.LinAlgError:
                # Increase damping and try again
                damping *= 10
                if damping > 1e6:
                    return NonlinearResult(
                        x=x,
                        success=False,
                        iterations=iteration,
                        residual_norm=residual_norm,
                        convergence_history=convergence_history,
                        solve_time=time.time() - start_time,
                        method="Damped Newton",
                        message="Cannot solve even with high damping",
                        function_evaluations=self.function_evaluations,
                        jacobian_evaluations=self.jacobian_evaluations
                    )
                continue
            # Try the step
            x_new = x + delta_x
            f_new = f(x_new)
            self.function_evaluations += 1
            # Check if step reduces residual
            if norm(f_new) < residual_norm:
                # Good step, reduce damping
                x = x_new
                damping = max(self.min_damping, damping * 0.5)
            else:
                # Bad step, increase damping
                damping = min(1e6, damping * 2.0)
        # Max iterations reached
        return NonlinearResult(
            x=x,
            success=False,
            iterations=self.max_iterations,
            residual_norm=residual_norm,
            convergence_history=convergence_history,
            solve_time=time.time() - start_time,
            method="Damped Newton",
            message="Maximum iterations reached",
            function_evaluations=self.function_evaluations,
            jacobian_evaluations=self.jacobian_evaluations
        )
class FixedPointSolver(NonlinearSolver):
    """Fixed point iteration solver."""
    def __init__(self, tolerance: float = 1e-8, max_iterations: int = 1000,
                 relaxation: float = 1.0):
        """Initialize fixed point solver.
        Args:
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            relaxation: Relaxation parameter (0 < relaxation <= 1)
        """
        super().__init__(tolerance, max_iterations)
        self.relaxation = relaxation
    def solve(self, g: Callable, x0: np.ndarray,
              jacobian: Optional[Callable] = None) -> NonlinearResult:
        """Solve using fixed point iteration x = g(x).
        Args:
            g: Fixed point function x = g(x)
            x0: Initial guess
            jacobian: Not used for fixed point iteration
        Returns:
            NonlinearResult
        """
        start_time = time.time()
        x = x0.copy()
        convergence_history = []
        self.function_evaluations = 0
        for iteration in range(self.max_iterations):
            # Fixed point iteration
            x_new = g(x)
            self.function_evaluations += 1
            # Relaxation
            x_next = self.relaxation * x_new + (1 - self.relaxation) * x
            # Check convergence
            residual_norm = norm(x_next - x)
            convergence_history.append(residual_norm)
            if residual_norm < self.tolerance:
                return NonlinearResult(
                    x=x_next,
                    success=True,
                    iterations=iteration,
                    residual_norm=residual_norm,
                    convergence_history=convergence_history,
                    solve_time=time.time() - start_time,
                    method="Fixed Point",
                    message=f"Converged in {iteration} iterations",
                    function_evaluations=self.function_evaluations
                )
            x = x_next
        # Max iterations reached
        return NonlinearResult(
            x=x,
            success=False,
            iterations=self.max_iterations,
            residual_norm=residual_norm,
            convergence_history=convergence_history,
            solve_time=time.time() - start_time,
            method="Fixed Point",
            message="Maximum iterations reached",
            function_evaluations=self.function_evaluations
        )
class ContinuationSolver:
    """Continuation/homotopy method solver."""
    def __init__(self, tolerance: float = 1e-8, max_iterations: int = 100,
                 max_continuation_steps: int = 100):
        """Initialize continuation solver.
        Args:
            tolerance: Convergence tolerance
            max_iterations: Maximum Newton iterations per step
            max_continuation_steps: Maximum continuation steps
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.max_continuation_steps = max_continuation_steps
    def solve_parameter_continuation(self,
                                   f: Callable,
                                   jacobian: Callable,
                                   x0: np.ndarray,
                                   param_range: Tuple[float, float],
                                   n_steps: int = 50) -> ContinuationResult:
        """Solve F(x, p) = 0 for parameter p in given range.
        Args:
            f: Function F(x, p)
            jacobian: Jacobian dF/dx(x, p)
            x0: Initial solution at p0
            param_range: Parameter range (p0, pf)
            n_steps: Number of continuation steps
        Returns:
            ContinuationResult
        """
        p0, pf = param_range
        param_values = np.linspace(p0, pf, n_steps + 1)
        solutions = np.zeros((n_steps + 1, len(x0)))
        convergence_data = []
        bifurcation_points = []
        # Initial solution
        solutions[0] = x0
        # Newton solver for each step
        newton_solver = NewtonSolver(self.tolerance, self.max_iterations)
        for i in range(1, n_steps + 1):
            p = param_values[i]
            # Use previous solution as initial guess
            x_guess = solutions[i-1]
            # Predictor step (simple)
            if i > 1:
                # Linear extrapolation
                dp = param_values[i] - param_values[i-1]
                dx = solutions[i-1] - solutions[i-2]
                x_guess = solutions[i-1] + dx * (dp / (param_values[i-1] - param_values[i-2]))
            # Define function for current parameter
            def f_p(x):
                return f(x, p)
            def jac_p(x):
                return jacobian(x, p)
            # Corrector step (Newton)
            result = newton_solver.solve(f_p, x_guess, jac_p)
            convergence_data.append(result)
            if result.success:
                solutions[i] = result.x
                # Simple bifurcation detection (determinant change sign)
                if i > 1:
                    J_prev = jacobian(solutions[i-1], param_values[i-1])
                    J_curr = jacobian(solutions[i], param_values[i])
                    det_prev = np.linalg.det(J_prev)
                    det_curr = np.linalg.det(J_curr)
                    if det_prev * det_curr < 0:
                        bifurcation_points.append(i)
            else:
                # Continuation failed
                return ContinuationResult(
                    parameter_values=param_values[:i],
                    solutions=solutions[:i],
                    success=False,
                    convergence_data=convergence_data,
                    bifurcation_points=bifurcation_points,
                    method="Parameter Continuation",
                    message=f"Continuation failed at step {i}"
                )
        return ContinuationResult(
            parameter_values=param_values,
            solutions=solutions,
            success=True,
            convergence_data=convergence_data,
            bifurcation_points=bifurcation_points,
            method="Parameter Continuation",
            message=f"Continuation completed successfully with {len(bifurcation_points)} bifurcation(s)"
        )
# Utility functions
def newton_raphson(f: Callable, x0: np.ndarray,
                  jacobian: Optional[Callable] = None,
                  tolerance: float = 1e-8,
                  max_iterations: int = 100) -> NonlinearResult:
    """Convenience function for Newton-Raphson method.
    Args:
        f: Nonlinear function F(x)
        x0: Initial guess
        jacobian: Jacobian function (optional)
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
    Returns:
        NonlinearResult
    """
    solver = NewtonSolver(tolerance, max_iterations)
    return solver.solve(f, x0, jacobian)
def solve_nonlinear_ode(dydt: Callable, y0: np.ndarray, t: float,
                       dt: float, method: str = "newton") -> np.ndarray:
    """Solve nonlinear ODE using implicit method.
    Args:
        dydt: Right-hand side function dy/dt = f(t, y)
        y0: Initial condition
        t: Current time
        dt: Time step
        method: Nonlinear solver method
    Returns:
        Solution at t + dt
    """
    # Implicit Euler: y_{n+1} = y_n + dt * f(t_{n+1}, y_{n+1})
    # Rearranged: G(y_{n+1}) = y_{n+1} - y_n - dt * f(t_{n+1}, y_{n+1}) = 0
    def residual(y_new):
        return y_new - y0 - dt * dydt(t + dt, y_new)
    def jacobian(y_new):
        # Finite difference approximation of Jacobian
        eps = 1e-8
        n = len(y_new)
        J = np.zeros((n, n))
        r0 = residual(y_new)
        for j in range(n):
            y_pert = y_new.copy()
            y_pert[j] += eps
            r_pert = residual(y_pert)
            J[:, j] = (r_pert - r0) / eps
        return J
    if method == "newton":
        solver = NewtonSolver()
    elif method == "damped_newton":
        solver = DampedNewtonSolver()
    else:
        raise ValueError(f"Unknown method: {method}")
    result = solver.solve(residual, y0, jacobian)
    if result.success:
        return result.x
    else:
        warnings.warn(f"Nonlinear solve failed: {result.message}")
        return y0  # Return previous value if solve fails
def solve_nonlinear_pde(pde_residual: Callable, u0: np.ndarray,
                       jacobian: Optional[Callable] = None,
                       method: str = "newton",
                       tolerance: float = 1e-8) -> NonlinearResult:
    """Solve nonlinear PDE system.
    Args:
        pde_residual: PDE residual function R(u) = 0
        u0: Initial guess
        jacobian: Jacobian function (optional)
        method: Nonlinear solver method
        tolerance: Convergence tolerance
    Returns:
        NonlinearResult
    """
    if method == "newton":
        solver = NewtonSolver(tolerance)
    elif method == "damped_newton":
        solver = DampedNewtonSolver(tolerance)
    elif method == "fixed_point":
        solver = FixedPointSolver(tolerance)
    else:
        raise ValueError(f"Unknown method: {method}")
    return solver.solve(pde_residual, u0, jacobian)
def continuation_method(f: Callable, jacobian: Callable,
                       x0: np.ndarray, param_range: Tuple[float, float],
                       n_steps: int = 50) -> ContinuationResult:
    """Convenience function for parameter continuation.
    Args:
        f: Function F(x, p)
        jacobian: Jacobian dF/dx(x, p)
        x0: Initial solution
        param_range: Parameter range
        n_steps: Number of continuation steps
    Returns:
        ContinuationResult
    """
    solver = ContinuationSolver()
    return solver.solve_parameter_continuation(f, jacobian, x0, param_range, n_steps)
def picard_iteration(f: Callable, x0: np.ndarray,
                    max_iterations: int = 1000,
                    tolerance: float = 1e-8,
                    relaxation: float = 1.0) -> NonlinearResult:
    """Picard iteration for solving x = f(x).
    Args:
        f: Function defining x = f(x)
        x0: Initial guess
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        relaxation: Relaxation parameter
    Returns:
        NonlinearResult
    """
    solver = FixedPointSolver(tolerance, max_iterations, relaxation)
    return solver.solve(f, x0)
def analyze_nonlinear_convergence(convergence_history: List[float]) -> Dict[str, Any]:
    """Analyze convergence behavior of nonlinear solver.
    Args:
        convergence_history: History of residual norms
    Returns:
        Dictionary with convergence analysis
    """
    if len(convergence_history) < 3:
        return {"error": "Insufficient data for analysis"}
    # Convergence rate estimation
    errors = np.array(convergence_history)
    # Remove zeros to avoid log issues
    nonzero_errors = errors[errors > 1e-16]
    if len(nonzero_errors) < 3:
        return {"error": "Insufficient nonzero errors for analysis"}
    # Estimate convergence rate
    # For quadratic convergence: e_{k+1} ≈ C * e_k^2
    # Taking log: log(e_{k+1}) ≈ log(C) + 2*log(e_k)
    log_errors = np.log(nonzero_errors[:-1])
    log_errors_next = np.log(nonzero_errors[1:])
    if len(log_errors) >= 2:
        # Linear regression
        A = np.vstack([log_errors, np.ones(len(log_errors))]).T
        try:
            rate, log_c = np.linalg.lstsq(A, log_errors_next, rcond=None)[0]
        except:
            rate = 1.0
            log_c = 0.0
    else:
        rate = 1.0
        log_c = 0.0
    # Classify convergence type
    if abs(rate - 2.0) < 0.2:
        convergence_type = "quadratic"
    elif abs(rate - 1.618) < 0.2:  # Golden ratio for superlinear
        convergence_type = "superlinear"
    elif abs(rate - 1.0) < 0.2:
        convergence_type = "linear"
    else:
        convergence_type = "unknown"
    return {
        "convergence_rate": rate,
        "convergence_type": convergence_type,
        "asymptotic_constant": np.exp(log_c),
        "final_residual": convergence_history[-1],
        "reduction_factor": convergence_history[-1] / convergence_history[0],
        "iterations": len(convergence_history) - 1
    }