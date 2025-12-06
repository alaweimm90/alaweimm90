"""
Constrained Optimization Algorithms
==================================
This module implements various constrained optimization algorithms including
penalty methods, barrier methods, Lagrange multipliers, and sequential
quadratic programming for handling equality and inequality constraints.
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Dict, Any, Union
from .unconstrained import OptimizationResult, UnconstrainedOptimizer, BFGS
import warnings
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
BERKELEY_LIGHT_BLUE = '#3B7EA1'
@dataclass
class Constraint:
    """
    Represents a constraint in optimization problem.
    Attributes:
        fun: Constraint function
        jac: Jacobian of constraint function
        type: Type of constraint ('eq' for equality, 'ineq' for inequality)
        args: Additional arguments for constraint function
    """
    fun: Callable
    jac: Optional[Callable] = None
    type: str = 'ineq'  # 'eq' or 'ineq'
    args: Tuple = ()
class ConstrainedOptimizer(ABC):
    """
    Abstract base class for constrained optimization algorithms.
    This class provides a common interface for all constrained
    optimization methods in the Berkeley SciComp framework.
    """
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6,
                 track_path: bool = False, verbose: bool = False):
        """
        Initialize constrained optimizer.
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
                constraints: List[Constraint] = None,
                bounds: Optional[List[Tuple[float, float]]] = None,
                gradient: Optional[Callable] = None,
                hessian: Optional[Callable] = None,
                **kwargs) -> OptimizationResult:
        """
        Minimize the constrained objective function.
        Args:
            objective: Objective function to minimize
            x0: Initial guess
            constraints: List of constraint objects
            bounds: Variable bounds
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
    def _evaluate_constraints(self, constraints: List[Constraint], x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate constraints and separate into equality and inequality.
        Returns:
            Tuple of (equality_constraints, inequality_constraints)
        """
        eq_constraints = []
        ineq_constraints = []
        if constraints:
            for constraint in constraints:
                value = constraint.fun(x)
                if constraint.type == 'eq':
                    eq_constraints.append(value)
                else:  # 'ineq'
                    ineq_constraints.append(value)
        return np.array(eq_constraints), np.array(ineq_constraints)
    def _evaluate_constraint_jacobians(self, constraints: List[Constraint], x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate constraint Jacobians.
        Returns:
            Tuple of (equality_jacobian, inequality_jacobian)
        """
        eq_jac = []
        ineq_jac = []
        if constraints:
            for constraint in constraints:
                if constraint.jac is not None:
                    jac_value = constraint.jac(x)
                else:
                    # Numerical Jacobian
                    jac_value = self._numerical_jacobian(constraint.fun, x)
                if constraint.type == 'eq':
                    eq_jac.append(jac_value)
                else:
                    ineq_jac.append(jac_value)
        eq_jac = np.array(eq_jac) if eq_jac else np.empty((0, len(x)))
        ineq_jac = np.array(ineq_jac) if ineq_jac else np.empty((0, len(x)))
        return eq_jac, ineq_jac
    def _numerical_jacobian(self, func: Callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        """Compute numerical Jacobian."""
        n = len(x)
        jac = np.zeros(n)
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            jac[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        return jac
    def _clip_to_bounds(self, x: np.ndarray, bounds: Optional[List[Tuple[float, float]]]) -> np.ndarray:
        """Clip variables to bounds."""
        if bounds is None:
            return x
        clipped = x.copy()
        for i, (low, high) in enumerate(bounds):
            if low is not None:
                clipped[i] = max(clipped[i], low)
            if high is not None:
                clipped[i] = min(clipped[i], high)
        return clipped
class PenaltyMethod(ConstrainedOptimizer):
    """
    Penalty Method for constrained optimization.
    Converts constrained problem to unconstrained by adding penalty terms
    for constraint violations to the objective function.
    """
    def __init__(self, penalty_parameter: float = 1.0, penalty_increase: float = 10.0,
                 unconstrained_solver: str = 'BFGS', **kwargs):
        """
        Initialize Penalty Method.
        Args:
            penalty_parameter: Initial penalty parameter
            penalty_increase: Factor to increase penalty parameter
            unconstrained_solver: Unconstrained solver to use
        """
        super().__init__(**kwargs)
        self.penalty_parameter = penalty_parameter
        self.penalty_increase = penalty_increase
        self.unconstrained_solver = unconstrained_solver
    def minimize(self, objective: Callable, x0: np.ndarray,
                constraints: List[Constraint] = None,
                bounds: Optional[List[Tuple[float, float]]] = None,
                gradient: Optional[Callable] = None,
                hessian: Optional[Callable] = None,
                **kwargs) -> OptimizationResult:
        """
        Minimize function using penalty method.
        Args:
            objective: Objective function
            x0: Initial point
            constraints: List of constraints
            bounds: Variable bounds
            gradient: Gradient function
            hessian: Hessian function
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        x = x0.copy()
        penalty_param = self.penalty_parameter
        path = [x.copy()] if self.track_path else None
        self.nfev = self.ngev = self.nhev = 0
        # Create unconstrained solver
        if self.unconstrained_solver == 'BFGS':
            solver = BFGS(max_iterations=100, tolerance=self.tolerance/10)
        else:
            solver = BFGS(max_iterations=100, tolerance=self.tolerance/10)
        for iteration in range(self.max_iterations):
            # Create penalty function
            def penalty_objective(x_inner):
                f_val = objective(x_inner)
                penalty = 0.0
                if constraints:
                    eq_constraints, ineq_constraints = self._evaluate_constraints(constraints, x_inner)
                    # Equality constraint penalty: (c_eq)^2
                    for eq_val in eq_constraints:
                        penalty += penalty_param * eq_val**2
                    # Inequality constraint penalty: max(0, c_ineq)^2
                    for ineq_val in ineq_constraints:
                        penalty += penalty_param * max(0, ineq_val)**2
                return f_val + penalty
            # Solve unconstrained subproblem
            result = solver.minimize(penalty_objective, x)
            x = result.x
            self.nfev += result.nfev
            self.ngev += result.ngev
            self.nhev += result.nhev
            # Check convergence
            if constraints:
                eq_constraints, ineq_constraints = self._evaluate_constraints(constraints, x)
                # Check constraint satisfaction
                eq_violation = np.max(np.abs(eq_constraints)) if len(eq_constraints) > 0 else 0
                ineq_violation = np.max(np.maximum(0, ineq_constraints)) if len(ineq_constraints) > 0 else 0
                total_violation = max(eq_violation, ineq_violation)
                if total_violation < self.tolerance:
                    success = True
                    message = f"Constraints satisfied: violation = {total_violation:.2e}"
                    break
            else:
                success = True
                message = "No constraints to satisfy"
                break
            # Increase penalty parameter
            penalty_param *= self.penalty_increase
            if self.track_path:
                path.append(x.copy())
            if self.verbose:
                obj_val = objective(x)
                print(f"Iter {iteration}: f = {obj_val:.6e}, penalty = {penalty_param:.2e}, "
                      f"violation = {total_violation:.6e}")
        else:
            success = False
            message = f"Maximum iterations ({self.max_iterations}) reached"
        execution_time = time.time() - start_time
        final_f = objective(x)
        return OptimizationResult(
            x=x, fun=final_f, nit=iteration+1,
            nfev=self.nfev, ngev=self.ngev, nhev=self.nhev,
            success=success, message=message, execution_time=execution_time,
            path=path
        )
class BarrierMethod(ConstrainedOptimizer):
    """
    Barrier Method (Interior Point Method) for constrained optimization.
    Uses logarithmic barrier functions to keep iterates in the feasible region
    while solving inequality-constrained problems.
    """
    def __init__(self, barrier_parameter: float = 1.0, barrier_decrease: float = 0.1,
                 unconstrained_solver: str = 'BFGS', **kwargs):
        """
        Initialize Barrier Method.
        Args:
            barrier_parameter: Initial barrier parameter
            barrier_decrease: Factor to decrease barrier parameter
            unconstrained_solver: Unconstrained solver to use
        """
        super().__init__(**kwargs)
        self.barrier_parameter = barrier_parameter
        self.barrier_decrease = barrier_decrease
        self.unconstrained_solver = unconstrained_solver
    def minimize(self, objective: Callable, x0: np.ndarray,
                constraints: List[Constraint] = None,
                bounds: Optional[List[Tuple[float, float]]] = None,
                gradient: Optional[Callable] = None,
                hessian: Optional[Callable] = None,
                **kwargs) -> OptimizationResult:
        """
        Minimize function using barrier method.
        Args:
            objective: Objective function
            x0: Initial point (must be feasible)
            constraints: List of inequality constraints
            bounds: Variable bounds
            gradient: Gradient function
            hessian: Hessian function
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        # Check initial feasibility
        if constraints:
            _, ineq_constraints = self._evaluate_constraints(constraints, x0)
            if len(ineq_constraints) > 0 and np.any(ineq_constraints >= 0):
                raise ValueError("Initial point must be strictly feasible for barrier method")
        x = x0.copy()
        barrier_param = self.barrier_parameter
        path = [x.copy()] if self.track_path else None
        self.nfev = self.ngev = self.nhev = 0
        # Create unconstrained solver
        if self.unconstrained_solver == 'BFGS':
            solver = BFGS(max_iterations=100, tolerance=self.tolerance/10)
        else:
            solver = BFGS(max_iterations=100, tolerance=self.tolerance/10)
        for iteration in range(self.max_iterations):
            # Create barrier function
            def barrier_objective(x_inner):
                f_val = objective(x_inner)
                barrier = 0.0
                if constraints:
                    _, ineq_constraints = self._evaluate_constraints(constraints, x_inner)
                    # Logarithmic barrier: -μ * sum(log(-c_ineq))
                    for ineq_val in ineq_constraints:
                        if ineq_val >= -1e-12:  # Approaching boundary
                            return float('inf')
                        barrier -= barrier_param * np.log(-ineq_val)
                return f_val + barrier
            # Solve unconstrained subproblem
            try:
                result = solver.minimize(barrier_objective, x)
                x = result.x
                self.nfev += result.nfev
                self.ngev += result.ngev
                self.nhev += result.nhev
            except Exception as e:
                warnings.warn(f"Barrier method failed at iteration {iteration}: {str(e)}")
                success = False
                message = f"Solver failed: {str(e)}"
                break
            # Check convergence
            if barrier_param < self.tolerance:
                success = True
                message = f"Barrier parameter below tolerance: {barrier_param:.2e}"
                break
            # Decrease barrier parameter
            barrier_param *= self.barrier_decrease
            if self.track_path:
                path.append(x.copy())
            if self.verbose:
                obj_val = objective(x)
                print(f"Iter {iteration}: f = {obj_val:.6e}, barrier = {barrier_param:.2e}")
        else:
            success = False
            message = f"Maximum iterations ({self.max_iterations}) reached"
        execution_time = time.time() - start_time
        final_f = objective(x)
        return OptimizationResult(
            x=x, fun=final_f, nit=iteration+1,
            nfev=self.nfev, ngev=self.ngev, nhev=self.nhev,
            success=success, message=message, execution_time=execution_time,
            path=path
        )
class AugmentedLagrangian(ConstrainedOptimizer):
    """
    Augmented Lagrangian Method for constrained optimization.
    Combines Lagrange multipliers with penalty terms to handle
    both equality and inequality constraints effectively.
    """
    def __init__(self, penalty_parameter: float = 1.0, penalty_increase: float = 10.0,
                 multiplier_update: str = 'standard', unconstrained_solver: str = 'BFGS',
                 **kwargs):
        """
        Initialize Augmented Lagrangian Method.
        Args:
            penalty_parameter: Initial penalty parameter
            penalty_increase: Factor to increase penalty parameter
            multiplier_update: Multiplier update rule ('standard', 'safeguarded')
            unconstrained_solver: Unconstrained solver to use
        """
        super().__init__(**kwargs)
        self.penalty_parameter = penalty_parameter
        self.penalty_increase = penalty_increase
        self.multiplier_update = multiplier_update
        self.unconstrained_solver = unconstrained_solver
    def minimize(self, objective: Callable, x0: np.ndarray,
                constraints: List[Constraint] = None,
                bounds: Optional[List[Tuple[float, float]]] = None,
                gradient: Optional[Callable] = None,
                hessian: Optional[Callable] = None,
                **kwargs) -> OptimizationResult:
        """
        Minimize function using augmented Lagrangian method.
        Args:
            objective: Objective function
            x0: Initial point
            constraints: List of constraints
            bounds: Variable bounds
            gradient: Gradient function
            hessian: Hessian function
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        x = x0.copy()
        penalty_param = self.penalty_parameter
        path = [x.copy()] if self.track_path else None
        # Initialize multipliers
        if constraints:
            eq_constraints, ineq_constraints = self._evaluate_constraints(constraints, x)
            lambda_eq = np.zeros(len(eq_constraints))
            lambda_ineq = np.zeros(len(ineq_constraints))
        else:
            lambda_eq = np.array([])
            lambda_ineq = np.array([])
        self.nfev = self.ngev = self.nhev = 0
        # Create unconstrained solver
        if self.unconstrained_solver == 'BFGS':
            solver = BFGS(max_iterations=100, tolerance=self.tolerance/10)
        else:
            solver = BFGS(max_iterations=100, tolerance=self.tolerance/10)
        for iteration in range(self.max_iterations):
            # Create augmented Lagrangian function
            def augmented_lagrangian(x_inner):
                f_val = objective(x_inner)
                augmented_term = 0.0
                if constraints:
                    eq_constraints, ineq_constraints = self._evaluate_constraints(constraints, x_inner)
                    # Equality constraints: λ*c + (ρ/2)*c²
                    for i, eq_val in enumerate(eq_constraints):
                        augmented_term += lambda_eq[i] * eq_val + 0.5 * penalty_param * eq_val**2
                    # Inequality constraints: max(0, λ + ρ*c)*c - λ²/(2*ρ)
                    for i, ineq_val in enumerate(ineq_constraints):
                        multiplier_term = lambda_ineq[i] + penalty_param * ineq_val
                        if multiplier_term > 0:
                            augmented_term += multiplier_term * ineq_val - lambda_ineq[i]**2 / (2 * penalty_param)
                        else:
                            augmented_term -= lambda_ineq[i]**2 / (2 * penalty_param)
                return f_val + augmented_term
            # Solve unconstrained subproblem
            result = solver.minimize(augmented_lagrangian, x)
            x = result.x
            self.nfev += result.nfev
            self.ngev += result.ngev
            self.nhev += result.nhev
            # Evaluate constraints at new point
            if constraints:
                eq_constraints, ineq_constraints = self._evaluate_constraints(constraints, x)
            else:
                eq_constraints, ineq_constraints = np.array([]), np.array([])
            # Update multipliers
            if len(eq_constraints) > 0:
                lambda_eq = lambda_eq + penalty_param * eq_constraints
            if len(ineq_constraints) > 0:
                lambda_ineq = np.maximum(0, lambda_ineq + penalty_param * ineq_constraints)
            # Check convergence
            eq_violation = np.max(np.abs(eq_constraints)) if len(eq_constraints) > 0 else 0
            ineq_violation = np.max(np.maximum(0, ineq_constraints)) if len(ineq_constraints) > 0 else 0
            total_violation = max(eq_violation, ineq_violation)
            if total_violation < self.tolerance:
                success = True
                message = f"Constraints satisfied: violation = {total_violation:.2e}"
                break
            # Update penalty parameter if needed
            prev_violation = getattr(self, '_prev_violation', float('inf'))
            if total_violation > 0.25 * prev_violation:
                penalty_param *= self.penalty_increase
            self._prev_violation = total_violation
            if self.track_path:
                path.append(x.copy())
            if self.verbose:
                obj_val = objective(x)
                print(f"Iter {iteration}: f = {obj_val:.6e}, penalty = {penalty_param:.2e}, "
                      f"violation = {total_violation:.6e}")
        else:
            success = False
            message = f"Maximum iterations ({self.max_iterations}) reached"
        execution_time = time.time() - start_time
        final_f = objective(x)
        return OptimizationResult(
            x=x, fun=final_f, nit=iteration+1,
            nfev=self.nfev, ngev=self.ngev, nhev=self.nhev,
            success=success, message=message, execution_time=execution_time,
            path=path
        )
class SequentialQuadraticProgramming(ConstrainedOptimizer):
    """
    Sequential Quadratic Programming (SQP) for constrained optimization.
    Solves a sequence of quadratic programming subproblems to approximate
    the constrained optimization problem.
    """
    def __init__(self, hessian_update: str = 'BFGS', merit_function: str = 'l1',
                 line_search: bool = True, **kwargs):
        """
        Initialize SQP Method.
        Args:
            hessian_update: Hessian approximation method ('BFGS', 'SR1')
            merit_function: Merit function for line search ('l1', 'l2')
            line_search: Whether to use line search
        """
        super().__init__(**kwargs)
        self.hessian_update = hessian_update
        self.merit_function = merit_function
        self.line_search = line_search
    def minimize(self, objective: Callable, x0: np.ndarray,
                constraints: List[Constraint] = None,
                bounds: Optional[List[Tuple[float, float]]] = None,
                gradient: Optional[Callable] = None,
                hessian: Optional[Callable] = None,
                **kwargs) -> OptimizationResult:
        """
        Minimize function using SQP method.
        Args:
            objective: Objective function
            x0: Initial point
            constraints: List of constraints
            bounds: Variable bounds
            gradient: Gradient function
            hessian: Hessian function
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        x = x0.copy()
        n = len(x)
        # Initialize Hessian approximation
        B = np.eye(n)
        # Initialize multipliers
        if constraints:
            eq_constraints, ineq_constraints = self._evaluate_constraints(constraints, x)
            lambda_eq = np.zeros(len(eq_constraints))
            lambda_ineq = np.zeros(len(ineq_constraints))
        else:
            lambda_eq = np.array([])
            lambda_ineq = np.array([])
            eq_constraints, ineq_constraints = np.array([]), np.array([])
        path = [x.copy()] if self.track_path else None
        self.nfev = self.ngev = self.nhev = 0
        for iteration in range(self.max_iterations):
            # Evaluate objective and constraints
            f_val = objective(x)
            if gradient:
                grad_f = gradient(x)
            else:
                grad_f = self._numerical_gradient(objective, x)
            if constraints:
                eq_constraints, ineq_constraints = self._evaluate_constraints(constraints, x)
                eq_jac, ineq_jac = self._evaluate_constraint_jacobians(constraints, x)
            else:
                eq_constraints, ineq_constraints = np.array([]), np.array([])
                eq_jac, ineq_jac = np.empty((0, n)), np.empty((0, n))
            # Check convergence
            eq_violation = np.max(np.abs(eq_constraints)) if len(eq_constraints) > 0 else 0
            ineq_violation = np.max(np.maximum(0, ineq_constraints)) if len(ineq_constraints) > 0 else 0
            total_violation = max(eq_violation, ineq_violation)
            if total_violation < self.tolerance and np.linalg.norm(grad_f) < self.tolerance:
                success = True
                message = f"Converged: violation = {total_violation:.2e}, ||grad|| = {np.linalg.norm(grad_f):.2e}"
                break
            # Solve QP subproblem (simplified)
            # This is a basic implementation - real SQP would use proper QP solver
            try:
                p = self._solve_qp_subproblem(grad_f, B, eq_constraints, ineq_constraints, eq_jac, ineq_jac)
            except np.linalg.LinAlgError:
                # Fallback to gradient descent step
                p = -grad_f / np.linalg.norm(grad_f) * 0.01
                warnings.warn("QP subproblem failed, using gradient descent step")
            # Line search or full step
            if self.line_search:
                alpha = self._line_search_sqp(objective, constraints, x, p, f_val)
            else:
                alpha = 1.0
            # Update
            x_new = x + alpha * p
            x_new = self._clip_to_bounds(x_new, bounds)
            # Update Hessian approximation
            if iteration > 0:
                s = x_new - x
                if gradient:
                    y = gradient(x_new) - grad_f
                else:
                    y = self._numerical_gradient(objective, x_new) - grad_f
                # BFGS update
                if np.dot(s, y) > 1e-10:
                    rho = 1.0 / np.dot(s, y)
                    I = np.eye(n)
                    B = (I - rho * np.outer(s, y)) @ B @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
            x = x_new
            if self.track_path:
                path.append(x.copy())
            if self.verbose:
                print(f"Iter {iteration}: f = {f_val:.6e}, violation = {total_violation:.6e}, "
                      f"||grad|| = {np.linalg.norm(grad_f):.6e}")
        else:
            success = False
            message = f"Maximum iterations ({self.max_iterations}) reached"
        execution_time = time.time() - start_time
        final_f = objective(x)
        return OptimizationResult(
            x=x, fun=final_f, nit=iteration+1,
            nfev=self.nfev, ngev=self.ngev, nhev=self.nhev,
            success=success, message=message, execution_time=execution_time,
            path=path
        )
    def _solve_qp_subproblem(self, grad_f: np.ndarray, B: np.ndarray,
                            eq_constraints: np.ndarray, ineq_constraints: np.ndarray,
                            eq_jac: np.ndarray, ineq_jac: np.ndarray) -> np.ndarray:
        """
        Solve QP subproblem (simplified implementation).
        This is a basic implementation using least squares.
        A real SQP implementation would use a proper QP solver.
        """
        n = len(grad_f)
        # For simplicity, solve unconstrained QP if no active constraints
        if len(eq_constraints) == 0 and np.all(ineq_constraints < -1e-6):
            # Unconstrained QP: min 0.5*p'*B*p + grad_f'*p
            try:
                p = -np.linalg.solve(B, grad_f)
            except np.linalg.LinAlgError:
                p = -grad_f / (np.linalg.norm(grad_f) + 1e-10)
            return p
        # Handle equality constraints using null space method (simplified)
        if len(eq_constraints) > 0:
            try:
                # Solve: A'*lambda = -grad_f, A*p = -c
                A = eq_jac
                c = eq_constraints
                # Least squares solution
                if A.shape[0] > 0:
                    AtA = A @ A.T
                    if np.linalg.det(AtA) > 1e-12:
                        lambda_eq = np.linalg.solve(AtA, -A @ grad_f - c)
                        p_particular = -A.T @ lambda_eq
                        # Project onto null space (simplified)
                        residual = grad_f + A.T @ lambda_eq
                        p = p_particular - 0.1 * residual
                    else:
                        p = -grad_f / (np.linalg.norm(grad_f) + 1e-10)
                else:
                    p = -grad_f / (np.linalg.norm(grad_f) + 1e-10)
            except np.linalg.LinAlgError:
                p = -grad_f / (np.linalg.norm(grad_f) + 1e-10)
        else:
            p = -grad_f / (np.linalg.norm(grad_f) + 1e-10)
        return p
    def _line_search_sqp(self, objective: Callable, constraints: List[Constraint],
                        x: np.ndarray, p: np.ndarray, f_val: float) -> float:
        """Simple line search for SQP."""
        alpha = 1.0
        for _ in range(20):
            x_new = x + alpha * p
            try:
                f_new = objective(x_new)
                # Check if function value improved
                if f_new < f_val + 1e-4 * alpha * np.dot(p, p):
                    return alpha
            except:
                pass
            alpha *= 0.5
        return alpha
    def _numerical_gradient(self, func: Callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        """Compute numerical gradient."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        return grad
# Alias for common usage
LagrangeMultipliers = AugmentedLagrangian
def demo():
    """Demonstrate constrained optimization algorithms."""
    print("Constrained Optimization Demo")
    print("============================")
    print()
    # Test problem: minimize x^2 + y^2 subject to x + y >= 1
    def objective(x):
        return x[0]**2 + x[1]**2
    def objective_grad(x):
        return np.array([2*x[0], 2*x[1]])
    # Constraint: x + y - 1 >= 0 (reformulated as g(x) = 1 - x - y <= 0)
    constraint = Constraint(fun=lambda x: x[0] + x[1] - 1, type='ineq')
    x0 = np.array([2.0, 2.0])
    # Test algorithms
    algorithms = {
        'Penalty Method': PenaltyMethod(penalty_parameter=1.0, verbose=False),
        'Augmented Lagrangian': AugmentedLagrangian(penalty_parameter=1.0, verbose=False),
        'SQP': SequentialQuadraticProgramming(verbose=False)
    }
    print("Problem: min x² + y² subject to x + y ≥ 1")
    print("Expected solution: x ≈ 0.5, y ≈ 0.5, f ≈ 0.5")
    print("-" * 50)
    for name, optimizer in algorithms.items():
        try:
            result = optimizer.minimize(objective, x0, constraints=[constraint],
                                      gradient=objective_grad)
            print(f"{name:20s}: x = [{result.x[0]:.4f}, {result.x[1]:.4f}], "
                  f"f = {result.fun:.6f}, success = {result.success}")
        except Exception as e:
            print(f"{name:20s}: Error - {str(e)}")
    print()
    # Test problem 2: Equality constraint
    print("Problem 2: min (x-1)² + (y-2)² subject to x + y = 2")
    print("Expected solution: x ≈ 0.5, y ≈ 1.5, f ≈ 0.5")
    print("-" * 50)
    def objective2(x):
        return (x[0] - 1)**2 + (x[1] - 2)**2
    def objective2_grad(x):
        return np.array([2*(x[0] - 1), 2*(x[1] - 2)])
    constraint2 = Constraint(fun=lambda x: x[0] + x[1] - 2, type='eq')
    for name, optimizer in algorithms.items():
        try:
            result = optimizer.minimize(objective2, x0, constraints=[constraint2],
                                      gradient=objective2_grad)
            print(f"{name:20s}: x = [{result.x[0]:.4f}, {result.x[1]:.4f}], "
                  f"f = {result.fun:.6f}, success = {result.success}")
        except Exception as e:
            print(f"{name:20s}: Error - {str(e)}")
    print()
    print("Demo completed!")
if __name__ == "__main__":
    demo()