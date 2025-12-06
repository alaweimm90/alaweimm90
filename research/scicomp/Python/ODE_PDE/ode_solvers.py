"""Ordinary Differential Equation Solvers.
This module provides comprehensive ODE solving capabilities including
explicit and implicit methods, adaptive time stepping, and stiff equation solvers.
Classes:
    ODESolver: Base class for ODE solvers
    ExplicitEuler: Forward Euler method
    ImplicitEuler: Backward Euler method
    RungeKutta4: Classical 4th order Runge-Kutta
    RungeKuttaFehlberg: RK45 with adaptive stepping
    AdamsBashforth: Multi-step explicit method
    AdamsMoulton: Multi-step implicit method
    BDF: Backward differentiation formulas for stiff ODEs
Functions:
    solve_ode: General ODE solving interface
    solve_ivp: Initial value problem solver
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from scipy.optimize import fsolve
from scipy.linalg import solve
@dataclass
class ODEResult:
    """Result of ODE integration."""
    t: np.ndarray
    y: np.ndarray
    success: bool
    message: str
    nfev: int  # Number of function evaluations
    njev: int = 0  # Number of Jacobian evaluations
    nlu: int = 0   # Number of LU decompositions
    status: int = 0
    t_events: Optional[List[np.ndarray]] = None
    sol: Optional[Callable] = None
@dataclass
class ODEOptions:
    """Options for ODE solvers."""
    rtol: float = 1e-3
    atol: float = 1e-6
    max_step: float = np.inf
    min_step: float = 0.0
    first_step: Optional[float] = None
    dense_output: bool = False
    events: Optional[List[Callable]] = None
    vectorized: bool = False
    args: Tuple = ()
class ODESolver(ABC):
    """Abstract base class for ODE solvers.
    Provides common interface for all ODE solving methods.
    """
    def __init__(self, fun: Callable, t0: float, y0: np.ndarray,
                 t_bound: float, options: ODEOptions):
        """Initialize ODE solver.
        Args:
            fun: Right-hand side function dy/dt = f(t, y)
            t0: Initial time
            y0: Initial conditions
            t_bound: Final time
            options: Solver options
        """
        self.fun = fun
        self.t0 = float(t0)
        self.y0 = np.asarray(y0, dtype=float)
        self.t_bound = float(t_bound)
        self.options = options
        # Current state
        self.t = self.t0
        self.y = self.y0.copy()
        self.h = None  # Step size
        # Statistics
        self.nfev = 0
        self.njev = 0
        self.nlu = 0
        # Integration direction
        self.direction = np.sign(self.t_bound - self.t0)
    @abstractmethod
    def step(self) -> bool:
        """Take one integration step.
        Returns:
            True if step successful, False if failed
        """
        pass
    def dense_output(self, t: np.ndarray) -> np.ndarray:
        """Compute solution at arbitrary points (if supported)."""
        raise NotImplementedError("Dense output not implemented for this solver")
    def solve(self, t_eval: Optional[np.ndarray] = None) -> ODEResult:
        """Solve the ODE.
        Args:
            t_eval: Times at which to evaluate solution
        Returns:
            ODEResult containing solution
        """
        if t_eval is None:
            t_eval = np.array([self.t0, self.t_bound])
        t_eval = np.asarray(t_eval)
        # Storage for solution
        t_list = [self.t0]
        y_list = [self.y0.copy()]
        # Main integration loop
        while True:
            # Check if we've reached the end
            if self.direction * (self.t - self.t_bound) >= 0:
                break
            # Adjust step size if necessary
            if self.h is None:
                self.h = self.options.first_step or 0.01
            if self.direction * (self.t + self.h - self.t_bound) > 0:
                self.h = self.t_bound - self.t
            # Take integration step
            success = self.step()
            if not success:
                return ODEResult(
                    t=np.array(t_list),
                    y=np.array(y_list).T,
                    success=False,
                    message="Integration failed",
                    nfev=self.nfev
                )
            t_list.append(self.t)
            y_list.append(self.y.copy())
        # Interpolate to requested evaluation points
        t_solution = np.array(t_list)
        y_solution = np.array(y_list).T
        if len(t_eval) > 2 or not np.allclose(t_eval, [self.t0, self.t_bound]):
            y_eval = np.zeros((len(self.y0), len(t_eval)))
            for i in range(len(self.y0)):
                y_eval[i] = np.interp(t_eval, t_solution, y_solution[i])
        else:
            t_eval = t_solution
            y_eval = y_solution
        return ODEResult(
            t=t_eval,
            y=y_eval,
            success=True,
            message="Integration completed successfully",
            nfev=self.nfev,
            njev=self.njev,
            nlu=self.nlu
        )
class ExplicitEuler(ODESolver):
    """Forward Euler method (1st order explicit).
    Simple explicit method: y_{n+1} = y_n + h * f(t_n, y_n)
    """
    def __init__(self, fun: Callable, t0: float, y0: np.ndarray,
                 t_bound: float, options: ODEOptions):
        super().__init__(fun, t0, y0, t_bound, options)
        self.h = 0.01  # Default step size
    def step(self) -> bool:
        """Take one Euler step."""
        try:
            # Evaluate function
            f = self.fun(self.t, self.y, *self.options.args)
            self.nfev += 1
            # Update solution
            self.y = self.y + self.h * f
            self.t = self.t + self.h
            return True
        except Exception:
            return False
class ImplicitEuler(ODESolver):
    """Backward Euler method (1st order implicit).
    Implicit method: y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
    Requires solving nonlinear system at each step.
    """
    def __init__(self, fun: Callable, t0: float, y0: np.ndarray,
                 t_bound: float, options: ODEOptions):
        super().__init__(fun, t0, y0, t_bound, options)
        self.h = 0.01
    def step(self) -> bool:
        """Take one implicit Euler step."""
        try:
            t_new = self.t + self.h
            # Define nonlinear system to solve
            def residual(y_new):
                self.nfev += 1
                return y_new - self.y - self.h * self.fun(t_new, y_new, *self.options.args)
            # Solve nonlinear system
            y_new, info, ier, msg = fsolve(residual, self.y, full_output=True)
            if ier != 1:
                return False
            self.y = y_new
            self.t = t_new
            return True
        except Exception:
            return False
class RungeKutta4(ODESolver):
    """Classical 4th order Runge-Kutta method.
    High-accuracy explicit method with 4 function evaluations per step.
    """
    def __init__(self, fun: Callable, t0: float, y0: np.ndarray,
                 t_bound: float, options: ODEOptions):
        super().__init__(fun, t0, y0, t_bound, options)
        self.h = 0.01
    def step(self) -> bool:
        """Take one RK4 step."""
        try:
            h = self.h
            t = self.t
            y = self.y
            # RK4 stages
            k1 = self.fun(t, y, *self.options.args)
            k2 = self.fun(t + h/2, y + h*k1/2, *self.options.args)
            k3 = self.fun(t + h/2, y + h*k2/2, *self.options.args)
            k4 = self.fun(t + h, y + h*k3, *self.options.args)
            self.nfev += 4
            # Update solution
            self.y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            self.t = t + h
            return True
        except Exception:
            return False
class RungeKuttaFehlberg(ODESolver):
    """Runge-Kutta-Fehlberg method (RK45) with adaptive step size.
    Embedded Runge-Kutta method that provides both 4th and 5th order
    approximations for error estimation and step size control.
    """
    # Butcher tableau coefficients for RK45
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/4, 0, 0, 0, 0, 0],
        [3/32, 9/32, 0, 0, 0, 0],
        [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
        [439/216, -8, 3680/513, -845/4104, 0, 0],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]
    ])
    b4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])  # 4th order
    b5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])  # 5th order
    c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    def __init__(self, fun: Callable, t0: float, y0: np.ndarray,
                 t_bound: float, options: ODEOptions):
        super().__init__(fun, t0, y0, t_bound, options)
        self.h = options.first_step or 0.01
        self.h_min = options.min_step
        self.h_max = options.max_step
    def step(self) -> bool:
        """Take one adaptive RK45 step."""
        try:
            while True:
                # Compute RK stages
                k = np.zeros((6, len(self.y)))
                k[0] = self.fun(self.t, self.y, *self.options.args)
                for i in range(1, 6):
                    y_temp = self.y + self.h * np.sum(self.A[i, :i] * k[:i].T, axis=1)
                    k[i] = self.fun(self.t + self.c[i] * self.h, y_temp, *self.options.args)
                self.nfev += 6
                # Compute 4th and 5th order solutions
                y4 = self.y + self.h * np.sum(self.b4 * k.T, axis=1)
                y5 = self.y + self.h * np.sum(self.b5 * k.T, axis=1)
                # Error estimate
                error = np.abs(y5 - y4)
                scale = self.options.atol + self.options.rtol * np.maximum(np.abs(self.y), np.abs(y5))
                error_norm = np.sqrt(np.mean((error / scale)**2))
                # Step size control
                if error_norm <= 1.0:  # Accept step
                    self.y = y5  # Use higher order solution
                    self.t += self.h
                    # Update step size for next step
                    factor = min(2.0, max(0.2, 0.9 * (1.0 / error_norm)**0.2))
                    self.h = min(self.h_max, max(self.h_min, factor * self.h))
                    return True
                else:  # Reject step
                    factor = max(0.1, 0.9 * (1.0 / error_norm)**0.25)
                    self.h = max(self.h_min, factor * self.h)
                    if self.h == self.h_min:
                        warnings.warn("Minimum step size reached")
                        return False
        except Exception:
            return False
class AdamsBashforth(ODESolver):
    """Adams-Bashforth multi-step explicit method.
    Uses function values from previous steps to achieve higher order
    with only one function evaluation per step.
    """
    def __init__(self, fun: Callable, t0: float, y0: np.ndarray,
                 t_bound: float, options: ODEOptions, order: int = 4):
        super().__init__(fun, t0, y0, t_bound, options)
        self.order = min(order, 5)  # Maximum order 5
        self.h = 0.01
        # Storage for previous values
        self.f_history = []
        self.startup_solver = RungeKutta4(fun, t0, y0, t_bound, options)
        self.startup_complete = False
        # Adams-Bashforth coefficients
        self.coeffs = {
            1: [1],
            2: [3/2, -1/2],
            3: [23/12, -16/12, 5/12],
            4: [55/24, -59/24, 37/24, -9/24],
            5: [1901/720, -2774/720, 2616/720, -1274/720, 251/720]
        }
    def step(self) -> bool:
        """Take one Adams-Bashforth step."""
        try:
            # Use RK4 for startup
            if len(self.f_history) < self.order:
                f_current = self.fun(self.t, self.y, *self.options.args)
                self.f_history.append(f_current)
                self.nfev += 1
                # Use RK4 for first few steps
                success = self.startup_solver.step()
                if success:
                    self.t = self.startup_solver.t
                    self.y = self.startup_solver.y
                    self.nfev += self.startup_solver.nfev - len(self.f_history)
                return success
            # Adams-Bashforth step
            f_current = self.fun(self.t, self.y, *self.options.args)
            self.nfev += 1
            # Update history
            self.f_history.append(f_current)
            if len(self.f_history) > self.order:
                self.f_history.pop(0)
            # Compute step
            coeffs = self.coeffs[len(self.f_history)]
            step = sum(c * f for c, f in zip(coeffs, reversed(self.f_history)))
            self.y = self.y + self.h * step
            self.t = self.t + self.h
            return True
        except Exception:
            return False
class AdamsMoulton(ODESolver):
    """Adams-Moulton multi-step implicit method.
    Implicit multi-step method that includes the current function value
    in the formula, requiring solution of nonlinear system.
    """
    def __init__(self, fun: Callable, t0: float, y0: np.ndarray,
                 t_bound: float, options: ODEOptions, order: int = 4):
        super().__init__(fun, t0, y0, t_bound, options)
        self.order = min(order, 5)
        self.h = 0.01
        # Storage for previous values
        self.f_history = []
        self.startup_solver = RungeKutta4(fun, t0, y0, t_bound, options)
        # Adams-Moulton coefficients (including current point)
        self.coeffs = {
            1: [1],
            2: [1/2, 1/2],
            3: [5/12, 8/12, -1/12],
            4: [9/24, 19/24, -5/24, 1/24],
            5: [251/720, 646/720, -264/720, 106/720, -19/720]
        }
    def step(self) -> bool:
        """Take one Adams-Moulton step."""
        try:
            # Startup phase
            if len(self.f_history) < self.order - 1:
                f_current = self.fun(self.t, self.y, *self.options.args)
                self.f_history.append(f_current)
                self.nfev += 1
                success = self.startup_solver.step()
                if success:
                    self.t = self.startup_solver.t
                    self.y = self.startup_solver.y
                return success
            t_new = self.t + self.h
            # Define implicit system
            def residual(y_new):
                f_new = self.fun(t_new, y_new, *self.options.args)
                self.nfev += 1
                # Adams-Moulton formula
                coeffs = self.coeffs[len(self.f_history) + 1]
                f_values = [f_new] + list(reversed(self.f_history))
                step = sum(c * f for c, f in zip(coeffs, f_values))
                return y_new - self.y - self.h * step
            # Solve implicit system
            y_new, info, ier, msg = fsolve(residual, self.y, full_output=True)
            if ier != 1:
                return False
            # Update history
            f_new = self.fun(t_new, y_new, *self.options.args)
            self.f_history.append(f_new)
            if len(self.f_history) > self.order - 1:
                self.f_history.pop(0)
            self.y = y_new
            self.t = t_new
            return True
        except Exception:
            return False
class BDF(ODESolver):
    """Backward Differentiation Formulas for stiff ODEs.
    Implicit multi-step methods particularly suited for stiff
    differential equations with excellent stability properties.
    """
    def __init__(self, fun: Callable, t0: float, y0: np.ndarray,
                 t_bound: float, options: ODEOptions, order: int = 3):
        super().__init__(fun, t0, y0, t_bound, options)
        self.order = min(order, 6)  # Maximum order 6
        self.h = 0.01
        # Storage for previous solutions
        self.y_history = [self.y0.copy()]
        self.startup_solver = RungeKutta4(fun, t0, y0, t_bound, options)
        # BDF coefficients
        self.alpha_coeffs = {
            1: [1, -1],
            2: [3/2, -2, 1/2],
            3: [11/6, -3, 3/2, -1/3],
            4: [25/12, -4, 3, -4/3, 1/4],
            5: [137/60, -5, 5, -10/3, 5/4, -1/5],
            6: [147/60, -6, 15/2, -20/3, 15/4, -6/5, 1/6]
        }
        # Jacobian for Newton's method
        self.jacobian = None
    def step(self) -> bool:
        """Take one BDF step."""
        try:
            # Startup phase using RK4
            if len(self.y_history) < self.order:
                success = self.startup_solver.step()
                if success:
                    self.t = self.startup_solver.t
                    self.y = self.startup_solver.y
                    self.y_history.append(self.y.copy())
                return success
            t_new = self.t + self.h
            order = min(self.order, len(self.y_history))
            # BDF formula: α₀y_{n+1} + α₁y_n + ... + αₖy_{n-k+1} = h*f(t_{n+1}, y_{n+1})
            alphas = self.alpha_coeffs[order]
            # Define nonlinear system for BDF
            def residual(y_new):
                f_new = self.fun(t_new, y_new, *self.options.args)
                self.nfev += 1
                # BDF formula
                bdf_sum = alphas[0] * y_new
                for i in range(1, order + 1):
                    bdf_sum += alphas[i] * self.y_history[-i]
                return bdf_sum - self.h * f_new
            # Newton's method for solving nonlinear system
            y_new = self._newton_solve(residual, self.y, t_new)
            if y_new is None:
                return False
            # Update history
            self.y_history.append(y_new.copy())
            if len(self.y_history) > self.order:
                self.y_history.pop(0)
            self.y = y_new
            self.t = t_new
            return True
        except Exception:
            return False
    def _newton_solve(self, residual: Callable, y_guess: np.ndarray,
                     t_new: float, max_iter: int = 10) -> Optional[np.ndarray]:
        """Solve nonlinear system using Newton's method."""
        y = y_guess.copy()
        for iteration in range(max_iter):
            # Compute residual
            r = residual(y)
            if np.linalg.norm(r) < self.options.atol:
                return y
            # Compute Jacobian (finite differences)
            J = self._compute_jacobian(t_new, y)
            self.njev += 1
            try:
                # Newton step
                dy = solve(J, -r)
                self.nlu += 1
                y = y + dy
            except np.linalg.LinAlgError:
                return None
        return None
    def _compute_jacobian(self, t: float, y: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix using finite differences."""
        n = len(y)
        J = np.zeros((n, n))
        eps = np.sqrt(np.finfo(float).eps)
        f0 = self.fun(t, y, *self.options.args)
        self.nfev += 1
        for i in range(n):
            y_pert = y.copy()
            y_pert[i] += eps
            f_pert = self.fun(t, y_pert, *self.options.args)
            self.nfev += 1
            J[:, i] = (f_pert - f0) / eps
        # Add BDF contribution to Jacobian
        order = min(self.order, len(self.y_history))
        alpha0 = self.alpha_coeffs[order][0]
        J = alpha0 * np.eye(n) - self.h * J
        return J
# Main solving interface
def solve_ode(fun: Callable, y0: Union[float, np.ndarray],
              t_span: Tuple[float, float],
              method: str = 'rk45',
              t_eval: Optional[np.ndarray] = None,
              dense_output: bool = False,
              events: Optional[List[Callable]] = None,
              vectorized: bool = False,
              args: Tuple = (),
              **options) -> ODEResult:
    """Solve initial value problem for ordinary differential equations.
    Args:
        fun: Right-hand side function dy/dt = f(t, y)
        y0: Initial conditions
        t_span: Time span (t0, tf)
        method: Solution method ('euler', 'rk4', 'rk45', 'adams', 'bdf')
        t_eval: Times at which to evaluate solution
        dense_output: Whether to compute dense output
        events: Event functions for detection
        vectorized: Whether function is vectorized
        args: Additional arguments to pass to function
        **options: Additional solver options
    Returns:
        ODEResult containing solution
    """
    # Validate inputs
    y0 = np.asarray(y0, dtype=float)
    if y0.ndim == 0:
        y0 = y0.reshape(1)
    t0, tf = t_span
    # Create options object
    ode_options = ODEOptions(
        dense_output=dense_output,
        events=events,
        vectorized=vectorized,
        args=args,
        **options
    )
    # Select solver
    solver_map = {
        'euler': ExplicitEuler,
        'implicit_euler': ImplicitEuler,
        'rk4': RungeKutta4,
        'rk45': RungeKuttaFehlberg,
        'adams': AdamsBashforth,
        'adams_moulton': AdamsMoulton,
        'bdf': BDF
    }
    if method not in solver_map:
        raise ValueError(f"Unknown method: {method}")
    # Create and run solver
    solver_class = solver_map[method]
    solver = solver_class(fun, t0, y0, tf, ode_options)
    return solver.solve(t_eval)
# Alias for compatibility
solve_ivp = solve_ode