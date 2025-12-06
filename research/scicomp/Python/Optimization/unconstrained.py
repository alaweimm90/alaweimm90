"""
Unconstrained Optimization Algorithms
====================================
This module implements various unconstrained optimization algorithms
including gradient-based methods, Newton-type methods, and quasi-Newton
approaches with Berkeley SciComp framework integration.
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Dict, Any
import warnings
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
BERKELEY_LIGHT_BLUE = '#3B7EA1'
@dataclass
class OptimizationResult:
    """
    Represents the result of an optimization algorithm.
    Attributes:
        x: Solution vector
        fun: Function value at solution
        grad: Gradient at solution (if available)
        hess: Hessian at solution (if available)
        nit: Number of iterations
        nfev: Number of function evaluations
        ngev: Number of gradient evaluations
        nhev: Number of Hessian evaluations
        success: Whether optimization was successful
        message: Description of termination condition
        execution_time: Time taken for optimization
        path: Optimization path (if tracked)
    """
    x: np.ndarray
    fun: float
    grad: Optional[np.ndarray] = None
    hess: Optional[np.ndarray] = None
    nit: int = 0
    nfev: int = 0
    ngev: int = 0
    nhev: int = 0
    success: bool = False
    message: str = ""
    execution_time: float = 0.0
    path: Optional[List[np.ndarray]] = None
class UnconstrainedOptimizer(ABC):
    """
    Abstract base class for unconstrained optimization algorithms.
    This class provides a common interface for all unconstrained
    optimization methods in the Berkeley SciComp framework.
    """
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6,
                 track_path: bool = False, verbose: bool = False):
        """
        Initialize optimizer.
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            track_path: Whether to track optimization path
            verbose: Whether to print progress information
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.track_path = track_path
        self.verbose = verbose
        # Counters
        self.nfev = 0
        self.ngev = 0
        self.nhev = 0
    @abstractmethod
    def minimize(self, objective: Callable, x0: np.ndarray,
                gradient: Optional[Callable] = None,
                hessian: Optional[Callable] = None,
                **kwargs) -> OptimizationResult:
        """
        Minimize the objective function.
        Args:
            objective: Objective function to minimize
            x0: Initial guess
            gradient: Gradient function (optional)
            hessian: Hessian function (optional)
            **kwargs: Additional algorithm-specific parameters
        Returns:
            OptimizationResult object
        """
        pass
    def _evaluate_function(self, func: Callable, x: np.ndarray) -> float:
        """Evaluate function with counter increment."""
        self.nfev += 1
        return func(x)
    def _evaluate_gradient(self, grad_func: Optional[Callable],
                          objective: Callable, x: np.ndarray,
                          h: float = 1e-8) -> np.ndarray:
        """Evaluate gradient (analytical or numerical)."""
        if grad_func is not None:
            self.ngev += 1
            return grad_func(x)
        else:
            # Numerical gradient using finite differences
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                grad[i] = (self._evaluate_function(objective, x_plus) -
                          self._evaluate_function(objective, x_minus)) / (2 * h)
            return grad
    def _evaluate_hessian(self, hess_func: Optional[Callable],
                         grad_func: Optional[Callable],
                         objective: Callable, x: np.ndarray,
                         h: float = 1e-5) -> np.ndarray:
        """Evaluate Hessian (analytical or numerical)."""
        if hess_func is not None:
            self.nhev += 1
            return hess_func(x)
        else:
            # Numerical Hessian using finite differences
            n = len(x)
            hess = np.zeros((n, n))
            if grad_func is not None:
                # Use gradient function for more accurate Hessian
                for i in range(n):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += h
                    x_minus[i] -= h
                    grad_plus = self._evaluate_gradient(grad_func, objective, x_plus, h)
                    grad_minus = self._evaluate_gradient(grad_func, objective, x_minus, h)
                    hess[:, i] = (grad_plus - grad_minus) / (2 * h)
            else:
                # Use function evaluations
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            x_plus = x.copy()
                            x_minus = x.copy()
                            x_plus[i] += h
                            x_minus[i] -= h
                            f_center = self._evaluate_function(objective, x)
                            f_plus = self._evaluate_function(objective, x_plus)
                            f_minus = self._evaluate_function(objective, x_minus)
                            hess[i, j] = (f_plus - 2*f_center + f_minus) / (h**2)
                        else:
                            x_pp = x.copy()
                            x_pm = x.copy()
                            x_mp = x.copy()
                            x_mm = x.copy()
                            x_pp[i] += h; x_pp[j] += h
                            x_pm[i] += h; x_pm[j] -= h
                            x_mp[i] -= h; x_mp[j] += h
                            x_mm[i] -= h; x_mm[j] -= h
                            f_pp = self._evaluate_function(objective, x_pp)
                            f_pm = self._evaluate_function(objective, x_pm)
                            f_mp = self._evaluate_function(objective, x_mp)
                            f_mm = self._evaluate_function(objective, x_mm)
                            hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h**2)
            # Symmetrize
            hess = (hess + hess.T) / 2
            return hess
class GradientDescent(UnconstrainedOptimizer):
    """
    Gradient Descent optimization algorithm.
    Implements steepest descent with various line search strategies
    and adaptive step size control.
    """
    def __init__(self, learning_rate: float = 0.01,
                 line_search: str = 'backtracking',
                 c1: float = 1e-4, rho: float = 0.5,
                 max_line_search: int = 50,
                 **kwargs):
        """
        Initialize Gradient Descent optimizer.
        Args:
            learning_rate: Initial learning rate
            line_search: Line search method ('fixed', 'backtracking', 'armijo')
            c1: Armijo condition parameter
            rho: Backtracking reduction factor
            max_line_search: Maximum line search iterations
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.line_search = line_search
        self.c1 = c1
        self.rho = rho
        self.max_line_search = max_line_search
    def minimize(self, objective: Callable, x0: np.ndarray,
                gradient: Optional[Callable] = None,
                hessian: Optional[Callable] = None,
                **kwargs) -> OptimizationResult:
        """
        Minimize function using gradient descent.
        Args:
            objective: Function to minimize
            x0: Initial point
            gradient: Gradient function (optional)
            hessian: Not used in gradient descent
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        # Initialize
        x = x0.copy()
        path = [x.copy()] if self.track_path else None
        self.nfev = self.ngev = self.nhev = 0
        for iteration in range(self.max_iterations):
            # Evaluate function and gradient
            f_val = self._evaluate_function(objective, x)
            grad = self._evaluate_gradient(gradient, objective, x)
            # Check convergence
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.tolerance:
                success = True
                message = f"Gradient norm below tolerance: {grad_norm:.2e}"
                break
            # Determine step size
            if self.line_search == 'fixed':
                alpha = self.learning_rate
            elif self.line_search == 'backtracking':
                alpha = self._backtracking_line_search(objective, x, grad, f_val)
            elif self.line_search == 'armijo':
                alpha = self._armijo_line_search(objective, x, grad, f_val)
            else:
                alpha = self.learning_rate
            # Update
            x_new = x - alpha * grad
            if self.verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: f = {f_val:.6e}, ||grad|| = {grad_norm:.6e}, alpha = {alpha:.6e}")
            x = x_new
            if self.track_path:
                path.append(x.copy())
        else:
            success = False
            message = f"Maximum iterations ({self.max_iterations}) reached"
        execution_time = time.time() - start_time
        final_f = self._evaluate_function(objective, x)
        final_grad = self._evaluate_gradient(gradient, objective, x)
        return OptimizationResult(
            x=x, fun=final_f, grad=final_grad, nit=iteration+1,
            nfev=self.nfev, ngev=self.ngev, nhev=self.nhev,
            success=success, message=message, execution_time=execution_time,
            path=path
        )
    def _backtracking_line_search(self, objective: Callable, x: np.ndarray,
                                 grad: np.ndarray, f_val: float) -> float:
        """Backtracking line search with Armijo condition."""
        alpha = self.learning_rate
        for _ in range(self.max_line_search):
            x_new = x - alpha * grad
            f_new = self._evaluate_function(objective, x_new)
            # Armijo condition
            if f_new <= f_val - self.c1 * alpha * np.dot(grad, grad):
                return alpha
            alpha *= self.rho
        return alpha
    def _armijo_line_search(self, objective: Callable, x: np.ndarray,
                           grad: np.ndarray, f_val: float) -> float:
        """Armijo line search."""
        return self._backtracking_line_search(objective, x, grad, f_val)
class NewtonMethod(UnconstrainedOptimizer):
    """
    Newton's Method for unconstrained optimization.
    Uses second-order Taylor approximation with Hessian information
    for quadratic convergence near the optimum.
    """
    def __init__(self, damping: float = 1.0, regularization: float = 1e-8,
                 **kwargs):
        """
        Initialize Newton's Method.
        Args:
            damping: Damping factor for Newton step
            regularization: Regularization for Hessian inversion
        """
        super().__init__(**kwargs)
        self.damping = damping
        self.regularization = regularization
    def minimize(self, objective: Callable, x0: np.ndarray,
                gradient: Optional[Callable] = None,
                hessian: Optional[Callable] = None,
                **kwargs) -> OptimizationResult:
        """
        Minimize function using Newton's method.
        Args:
            objective: Function to minimize
            x0: Initial point
            gradient: Gradient function
            hessian: Hessian function
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        # Initialize
        x = x0.copy()
        path = [x.copy()] if self.track_path else None
        self.nfev = self.ngev = self.nhev = 0
        for iteration in range(self.max_iterations):
            # Evaluate function, gradient, and Hessian
            f_val = self._evaluate_function(objective, x)
            grad = self._evaluate_gradient(gradient, objective, x)
            hess = self._evaluate_hessian(hessian, gradient, objective, x)
            # Check convergence
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.tolerance:
                success = True
                message = f"Gradient norm below tolerance: {grad_norm:.2e}"
                break
            # Regularize Hessian for numerical stability
            hess_reg = hess + self.regularization * np.eye(len(x))
            try:
                # Solve Newton system: H * p = -g
                newton_step = np.linalg.solve(hess_reg, -grad)
            except np.linalg.LinAlgError:
                # Fallback to gradient descent if Hessian is singular
                newton_step = -grad
                warnings.warn("Hessian is singular, using gradient descent step")
            # Update with damping
            x_new = x + self.damping * newton_step
            if self.verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: f = {f_val:.6e}, ||grad|| = {grad_norm:.6e}")
            x = x_new
            if self.track_path:
                path.append(x.copy())
        else:
            success = False
            message = f"Maximum iterations ({self.max_iterations}) reached"
        execution_time = time.time() - start_time
        final_f = self._evaluate_function(objective, x)
        final_grad = self._evaluate_gradient(gradient, objective, x)
        final_hess = self._evaluate_hessian(hessian, gradient, objective, x)
        return OptimizationResult(
            x=x, fun=final_f, grad=final_grad, hess=final_hess,
            nit=iteration+1, nfev=self.nfev, ngev=self.ngev, nhev=self.nhev,
            success=success, message=message, execution_time=execution_time,
            path=path
        )
class BFGS(UnconstrainedOptimizer):
    """
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) quasi-Newton method.
    Approximates the Hessian using gradient information from
    previous iterations, avoiding expensive Hessian computations.
    """
    def __init__(self, initial_hessian: Optional[np.ndarray] = None,
                 line_search: str = 'wolfe', c1: float = 1e-4, c2: float = 0.9,
                 **kwargs):
        """
        Initialize BFGS optimizer.
        Args:
            initial_hessian: Initial Hessian approximation
            line_search: Line search method ('wolfe', 'backtracking')
            c1: Armijo condition parameter
            c2: Curvature condition parameter
        """
        super().__init__(**kwargs)
        self.initial_hessian = initial_hessian
        self.line_search = line_search
        self.c1 = c1
        self.c2 = c2
    def minimize(self, objective: Callable, x0: np.ndarray,
                gradient: Optional[Callable] = None,
                hessian: Optional[Callable] = None,
                **kwargs) -> OptimizationResult:
        """
        Minimize function using BFGS.
        Args:
            objective: Function to minimize
            x0: Initial point
            gradient: Gradient function
            hessian: Not used in BFGS
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        # Initialize
        x = x0.copy()
        n = len(x)
        # Initial Hessian approximation
        if self.initial_hessian is not None:
            B = self.initial_hessian.copy()
        else:
            B = np.eye(n)
        path = [x.copy()] if self.track_path else None
        self.nfev = self.ngev = self.nhev = 0
        # Initial gradient
        grad = self._evaluate_gradient(gradient, objective, x)
        prev_f_val = None
        for iteration in range(self.max_iterations):
            # Check convergence
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.tolerance:
                success = True
                message = f"Gradient norm below tolerance: {grad_norm:.2e}"
                break
            # Get current function value for convergence check
            f_val = self._evaluate_function(objective, x)
            if iteration > 0 and prev_f_val is not None:
                if abs(f_val - prev_f_val) < self.tolerance * max(1.0, abs(f_val)):
                    success = True
                    message = f"Function value converged: df = {abs(f_val - prev_f_val):.2e}"
                    break
            prev_f_val = f_val
            # Compute search direction
            try:
                p = -np.linalg.solve(B, grad)
            except np.linalg.LinAlgError:
                # Fallback to gradient descent
                p = -grad
                B = np.eye(n)  # Reset Hessian approximation
            # Line search (f_val already computed above)
            alpha = self._line_search(objective, gradient, x, p, f_val, grad)
            # Update
            s = alpha * p
            x_new = x + s
            grad_new = self._evaluate_gradient(gradient, objective, x_new)
            y = grad_new - grad
            # BFGS update
            rho = np.dot(y, s)
            if abs(rho) > 1e-10:  # Avoid division by zero
                rho = 1.0 / rho
                I = np.eye(n)
                B = (I - rho * np.outer(s, y)) @ B @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
            if self.verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: f = {f_val:.6e}, ||grad|| = {grad_norm:.6e}, alpha = {alpha:.6e}")
            x = x_new
            grad = grad_new
            if self.track_path:
                path.append(x.copy())
        else:
            success = False
            message = f"Maximum iterations ({self.max_iterations}) reached"
        execution_time = time.time() - start_time
        final_f = self._evaluate_function(objective, x)
        return OptimizationResult(
            x=x, fun=final_f, grad=grad, hess=B, nit=iteration+1,
            nfev=self.nfev, ngev=self.ngev, nhev=self.nhev,
            success=success, message=message, execution_time=execution_time,
            path=path
        )
    def _line_search(self, objective: Callable, gradient: Optional[Callable],
                    x: np.ndarray, p: np.ndarray, f_val: float, grad: np.ndarray) -> float:
        """Perform line search to find step size."""
        alpha = 1.0
        for _ in range(50):  # Max line search iterations
            x_new = x + alpha * p
            f_new = self._evaluate_function(objective, x_new)
            # Armijo condition
            if f_new <= f_val + self.c1 * alpha * np.dot(grad, p):
                if self.line_search == 'backtracking':
                    return alpha
                # Wolfe condition
                grad_new = self._evaluate_gradient(gradient, objective, x_new)
                if np.dot(grad_new, p) >= self.c2 * np.dot(grad, p):
                    return alpha
            alpha *= 0.5
        return alpha
# Alias for common usage
QuasiNewton = BFGS
class ConjugateGradient(UnconstrainedOptimizer):
    """
    Conjugate Gradient method for unconstrained optimization.
    Particularly effective for quadratic functions and large-scale
    optimization problems where Hessian computation is expensive.
    """
    def __init__(self, beta_method: str = 'polak-ribiere', restart_threshold: int = None,
                 **kwargs):
        """
        Initialize Conjugate Gradient optimizer.
        Args:
            beta_method: Method for computing beta ('fletcher-reeves', 'polak-ribiere', 'hestenes-stiefel')
            restart_threshold: Iterations after which to restart (default: n)
        """
        super().__init__(**kwargs)
        self.beta_method = beta_method
        self.restart_threshold = restart_threshold
    def minimize(self, objective: Callable, x0: np.ndarray,
                gradient: Optional[Callable] = None,
                hessian: Optional[Callable] = None,
                **kwargs) -> OptimizationResult:
        """
        Minimize function using Conjugate Gradient.
        Args:
            objective: Function to minimize
            x0: Initial point
            gradient: Gradient function
            hessian: Not used in CG
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        # Initialize
        x = x0.copy()
        n = len(x)
        restart_threshold = self.restart_threshold or n
        path = [x.copy()] if self.track_path else None
        self.nfev = self.ngev = self.nhev = 0
        # Initial gradient and direction
        grad = self._evaluate_gradient(gradient, objective, x)
        p = -grad.copy()  # Initial search direction
        for iteration in range(self.max_iterations):
            # Check convergence
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.tolerance:
                success = True
                message = f"Gradient norm below tolerance: {grad_norm:.2e}"
                break
            # Line search
            f_val = self._evaluate_function(objective, x)
            alpha = self._line_search_cg(objective, x, p, f_val, grad)
            # Update
            x_new = x + alpha * p
            grad_new = self._evaluate_gradient(gradient, objective, x_new)
            # Compute beta
            if iteration % restart_threshold == 0:
                # Restart: use steepest descent direction
                beta = 0.0
            else:
                beta = self._compute_beta(grad, grad_new, p)
            # New search direction
            p_new = -grad_new + beta * p
            if self.verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: f = {f_val:.6e}, ||grad|| = {grad_norm:.6e}, alpha = {alpha:.6e}")
            x = x_new
            grad = grad_new
            p = p_new
            if self.track_path:
                path.append(x.copy())
        else:
            success = False
            message = f"Maximum iterations ({self.max_iterations}) reached"
        execution_time = time.time() - start_time
        final_f = self._evaluate_function(objective, x)
        return OptimizationResult(
            x=x, fun=final_f, grad=grad, nit=iteration+1,
            nfev=self.nfev, ngev=self.ngev, nhev=self.nhev,
            success=success, message=message, execution_time=execution_time,
            path=path
        )
    def _compute_beta(self, grad_old: np.ndarray, grad_new: np.ndarray, p_old: np.ndarray) -> float:
        """Compute beta parameter for conjugate direction."""
        if self.beta_method == 'fletcher-reeves':
            return np.dot(grad_new, grad_new) / np.dot(grad_old, grad_old)
        elif self.beta_method == 'polak-ribiere':
            return np.dot(grad_new, grad_new - grad_old) / np.dot(grad_old, grad_old)
        elif self.beta_method == 'hestenes-stiefel':
            y = grad_new - grad_old
            return np.dot(grad_new, y) / np.dot(p_old, y)
        else:
            return np.dot(grad_new, grad_new) / np.dot(grad_old, grad_old)
    def _line_search_cg(self, objective: Callable, x: np.ndarray, p: np.ndarray,
                       f_val: float, grad: np.ndarray) -> float:
        """Simple line search for conjugate gradient."""
        alpha = 1.0
        c1 = 1e-4
        for _ in range(50):
            x_new = x + alpha * p
            f_new = self._evaluate_function(objective, x_new)
            # Armijo condition
            if f_new <= f_val + c1 * alpha * np.dot(grad, p):
                return alpha
            alpha *= 0.5
        return alpha
class TrustRegion(UnconstrainedOptimizer):
    """
    Trust Region method for unconstrained optimization.
    Uses a trust region approach with quadratic model approximation
    and adaptive region size adjustment.
    """
    def __init__(self, initial_radius: float = 1.0, max_radius: float = 10.0,
                 eta1: float = 0.25, eta2: float = 0.75, gamma1: float = 0.5,
                 gamma2: float = 2.0, **kwargs):
        """
        Initialize Trust Region optimizer.
        Args:
            initial_radius: Initial trust region radius
            max_radius: Maximum trust region radius
            eta1: Threshold for shrinking trust region
            eta2: Threshold for expanding trust region
            gamma1: Shrinking factor
            gamma2: Expansion factor
        """
        super().__init__(**kwargs)
        self.initial_radius = initial_radius
        self.max_radius = max_radius
        self.eta1 = eta1
        self.eta2 = eta2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
    def minimize(self, objective: Callable, x0: np.ndarray,
                gradient: Optional[Callable] = None,
                hessian: Optional[Callable] = None,
                **kwargs) -> OptimizationResult:
        """
        Minimize function using Trust Region method.
        Args:
            objective: Function to minimize
            x0: Initial point
            gradient: Gradient function
            hessian: Hessian function
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        # Initialize
        x = x0.copy()
        radius = self.initial_radius
        path = [x.copy()] if self.track_path else None
        self.nfev = self.ngev = self.nhev = 0
        for iteration in range(self.max_iterations):
            # Evaluate function, gradient, and Hessian
            f_val = self._evaluate_function(objective, x)
            grad = self._evaluate_gradient(gradient, objective, x)
            hess = self._evaluate_hessian(hessian, gradient, objective, x)
            # Check convergence
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.tolerance:
                success = True
                message = f"Gradient norm below tolerance: {grad_norm:.2e}"
                break
            # Solve trust region subproblem
            step = self._solve_trust_region_subproblem(grad, hess, radius)
            # Compute actual and predicted reduction
            x_new = x + step
            f_new = self._evaluate_function(objective, x_new)
            actual_reduction = f_val - f_new
            predicted_reduction = -(np.dot(grad, step) + 0.5 * np.dot(step, hess @ step))
            # Compute reduction ratio
            if abs(predicted_reduction) < 1e-12:
                rho = 0
            else:
                rho = actual_reduction / predicted_reduction
            # Update trust region radius
            if rho < self.eta1:
                radius = self.gamma1 * radius
            elif rho > self.eta2:
                radius = min(self.gamma2 * radius, self.max_radius)
            # Accept or reject step
            if rho > self.eta1:
                x = x_new
                if self.track_path:
                    path.append(x.copy())
            if self.verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: f = {f_val:.6e}, ||grad|| = {grad_norm:.6e}, radius = {radius:.6e}, rho = {rho:.6e}")
        else:
            success = False
            message = f"Maximum iterations ({self.max_iterations}) reached"
        execution_time = time.time() - start_time
        final_f = self._evaluate_function(objective, x)
        final_grad = self._evaluate_gradient(gradient, objective, x)
        final_hess = self._evaluate_hessian(hessian, gradient, objective, x)
        return OptimizationResult(
            x=x, fun=final_f, grad=final_grad, hess=final_hess,
            nit=iteration+1, nfev=self.nfev, ngev=self.ngev, nhev=self.nhev,
            success=success, message=message, execution_time=execution_time,
            path=path
        )
    def _solve_trust_region_subproblem(self, grad: np.ndarray, hess: np.ndarray, radius: float) -> np.ndarray:
        """
        Solve trust region subproblem using Cauchy point method.
        This is a simplified implementation. More sophisticated methods
        like dogleg or exact trust region solutions could be implemented.
        """
        # Cauchy point method
        grad_norm_sq = np.dot(grad, grad)
        if grad_norm_sq == 0:
            return np.zeros_like(grad)
        # Compute Cauchy point
        Bg = hess @ grad
        curvature = np.dot(grad, Bg)
        if curvature <= 0:
            # Negative curvature direction
            tau = 1.0
        else:
            tau = min(1.0, grad_norm_sq / curvature)
        # Scale to trust region boundary if necessary
        cauchy_step = -tau * grad
        cauchy_norm = np.linalg.norm(cauchy_step)
        if cauchy_norm <= radius:
            return cauchy_step
        else:
            return -(radius / np.sqrt(grad_norm_sq)) * grad
def demo():
    """Demonstrate unconstrained optimization algorithms."""
    print("Unconstrained Optimization Demo")
    print("==============================")
    print()
    import matplotlib.pyplot as plt
    # Test function: Rosenbrock
    def rosenbrock(x):
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    def rosenbrock_grad(x):
        return np.array([
            -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
            200 * (x[1] - x[0]**2)
        ])
    def rosenbrock_hess(x):
        return np.array([
            [-400 * (x[1] - 3*x[0]**2) + 2, -400 * x[0]],
            [-400 * x[0], 200]
        ])
    x0 = np.array([-1.0, 1.0])
    # Test different algorithms
    algorithms = {
        'Gradient Descent': GradientDescent(learning_rate=0.001, track_path=True),
        'Newton Method': NewtonMethod(track_path=True),
        'BFGS': BFGS(track_path=True),
        'Conjugate Gradient': ConjugateGradient(track_path=True),
        'Trust Region': TrustRegion(track_path=True)
    }
    results = {}
    for name, optimizer in algorithms.items():
        print(f"Testing {name}...")
        if name == 'Newton Method' or name == 'Trust Region':
            result = optimizer.minimize(rosenbrock, x0, rosenbrock_grad, rosenbrock_hess)
        else:
            result = optimizer.minimize(rosenbrock, x0, rosenbrock_grad)
        results[name] = result
        print(f"  Solution: [{result.x[0]:.6f}, {result.x[1]:.6f}]")
        print(f"  Function value: {result.fun:.6e}")
        print(f"  Iterations: {result.nit}")
        print(f"  Success: {result.success}")
        print()
    print("Demo completed!")
if __name__ == "__main__":
    demo()