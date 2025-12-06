"""
Constraint type definitions for optimization problems.

This module provides various constraint types including:
- Equality constraints: h(x) = 0
- Inequality constraints: g(x) ≤ 0
- Box constraints: l ≤ x ≤ u
- Linear constraints: Ax ≤ b
- Nonlinear constraints: general g(x) ≤ 0

All constraints follow the convention that feasible solutions
satisfy the constraint (negative or zero violation).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class Constraint(ABC):
    """
    Abstract base class for optimization constraints.

    All constraints follow the convention:
    - Feasible if violation ≤ 0
    - Infeasible if violation > 0
    """

    def __init__(
        self,
        name: Optional[str] = None,
        tolerance: float = 1e-6
    ):
        """
        Initialize constraint.

        Args:
            name: Optional name for the constraint
            tolerance: Numerical tolerance for feasibility check
        """
        self.name = name or f"{self.__class__.__name__}_{id(self)}"
        self.tolerance = tolerance

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate constraint violation.

        Args:
            x: Decision variable vector

        Returns:
            Constraint violation (≤0 is feasible, >0 is infeasible)
        """
        pass

    def is_feasible(self, x: np.ndarray) -> bool:
        """
        Check if solution is feasible.

        Args:
            x: Decision variable vector

        Returns:
            True if feasible, False otherwise
        """
        return self.evaluate(x) <= self.tolerance

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate constraint gradient.

        Args:
            x: Decision variable vector

        Returns:
            Gradient vector
        """
        pass

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Project solution to satisfy constraint.

        Default implementation returns original solution.
        Override for specific projection methods.

        Args:
            x: Decision variable vector

        Returns:
            Projected solution
        """
        return x


class EqualityConstraint(Constraint):
    """
    Equality constraint: h(x) = 0.

    Violation is calculated as |h(x)|.
    """

    def __init__(
        self,
        function: Callable[[np.ndarray], float],
        gradient_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        name: Optional[str] = None,
        tolerance: float = 1e-6
    ):
        """
        Initialize equality constraint.

        Args:
            function: Constraint function h(x)
            gradient_function: Optional gradient function
            name: Optional constraint name
            tolerance: Tolerance for equality check
        """
        super().__init__(name, tolerance)
        self.function = function
        self.gradient_function = gradient_function

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate equality constraint violation.

        Returns: |h(x)|
        """
        return abs(self.function(x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Calculate constraint gradient."""
        if self.gradient_function is not None:
            return self.gradient_function(x)
        else:
            # Numerical gradient
            return self._numerical_gradient(x)

    def _numerical_gradient(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Calculate numerical gradient using finite differences."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (self.function(x_plus) - self.function(x_minus)) / (2 * eps)
        return grad


class InequalityConstraint(Constraint):
    """
    Inequality constraint: g(x) ≤ 0.

    Violation is calculated as max(0, g(x)).
    """

    def __init__(
        self,
        function: Callable[[np.ndarray], float],
        gradient_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        name: Optional[str] = None,
        tolerance: float = 1e-6
    ):
        """
        Initialize inequality constraint.

        Args:
            function: Constraint function g(x)
            gradient_function: Optional gradient function
            name: Optional constraint name
            tolerance: Tolerance for feasibility check
        """
        super().__init__(name, tolerance)
        self.function = function
        self.gradient_function = gradient_function

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate inequality constraint violation.

        Returns: max(0, g(x))
        """
        return max(0, self.function(x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Calculate constraint gradient."""
        if self.gradient_function is not None:
            grad = self.gradient_function(x)
        else:
            grad = self._numerical_gradient(x)

        # Zero gradient if constraint is satisfied
        if self.function(x) <= 0:
            return np.zeros_like(grad)
        return grad

    def _numerical_gradient(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Calculate numerical gradient using finite differences."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (self.function(x_plus) - self.function(x_minus)) / (2 * eps)
        return grad


class BoxConstraint(Constraint):
    """
    Box constraint: l ≤ x ≤ u.

    Efficiently handles variable bounds.
    """

    def __init__(
        self,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        name: Optional[str] = None,
        tolerance: float = 1e-6
    ):
        """
        Initialize box constraint.

        Args:
            lower_bounds: Lower bounds for each variable
            upper_bounds: Upper bounds for each variable
            name: Optional constraint name
            tolerance: Tolerance for feasibility check
        """
        super().__init__(name, tolerance)
        self.lower_bounds = np.asarray(lower_bounds)
        self.upper_bounds = np.asarray(upper_bounds)

        if len(self.lower_bounds) != len(self.upper_bounds):
            raise ValueError("Lower and upper bounds must have same length")

        if np.any(self.lower_bounds > self.upper_bounds):
            raise ValueError("Lower bounds must not exceed upper bounds")

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate box constraint violation.

        Returns: Maximum violation across all dimensions
        """
        lower_violation = np.maximum(0, self.lower_bounds - x)
        upper_violation = np.maximum(0, x - self.upper_bounds)
        return max(np.max(lower_violation), np.max(upper_violation))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Calculate constraint gradient."""
        grad = np.zeros_like(x)

        # Lower bound violations
        lower_mask = x < self.lower_bounds
        grad[lower_mask] = -1.0

        # Upper bound violations
        upper_mask = x > self.upper_bounds
        grad[upper_mask] = 1.0

        return grad

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Project solution to box constraints.

        Simply clips values to bounds.
        """
        return np.clip(x, self.lower_bounds, self.upper_bounds)

    def random_point(self) -> np.ndarray:
        """Generate random point within box constraints."""
        return np.random.uniform(self.lower_bounds, self.upper_bounds)


class LinearConstraint(Constraint):
    """
    Linear constraint: Ax ≤ b or Ax = b.

    Supports both inequality and equality linear constraints.
    """

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        constraint_type: str = "inequality",
        name: Optional[str] = None,
        tolerance: float = 1e-6
    ):
        """
        Initialize linear constraint.

        Args:
            A: Constraint matrix (m x n)
            b: Constraint vector (m,)
            constraint_type: "inequality" (Ax ≤ b) or "equality" (Ax = b)
            name: Optional constraint name
            tolerance: Tolerance for feasibility check
        """
        super().__init__(name, tolerance)
        self.A = np.asarray(A)
        self.b = np.asarray(b)

        if self.A.ndim == 1:
            self.A = self.A.reshape(1, -1)
        if self.b.ndim == 0:
            self.b = np.array([self.b])

        if constraint_type not in ["inequality", "equality"]:
            raise ValueError("constraint_type must be 'inequality' or 'equality'")
        self.constraint_type = constraint_type

        if self.A.shape[0] != len(self.b):
            raise ValueError("Number of constraints in A must match length of b")

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate linear constraint violation.

        Returns: Maximum violation across all constraints
        """
        Ax = self.A @ x

        if self.constraint_type == "inequality":
            violations = np.maximum(0, Ax - self.b)
        else:  # equality
            violations = np.abs(Ax - self.b)

        return np.max(violations) if len(violations) > 0 else 0.0

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Calculate constraint gradient."""
        Ax = self.A @ x

        if self.constraint_type == "inequality":
            # Find most violated constraint
            violations = Ax - self.b
            if np.all(violations <= 0):
                return np.zeros(len(x))
            max_idx = np.argmax(violations)
            return self.A[max_idx]
        else:  # equality
            # Find constraint with largest absolute violation
            violations = np.abs(Ax - self.b)
            max_idx = np.argmax(violations)
            sign = np.sign(Ax[max_idx] - self.b[max_idx])
            return sign * self.A[max_idx]

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Project solution to satisfy linear constraints.

        Uses quadratic programming for projection.
        """
        try:
            from scipy.optimize import minimize

            # Minimize ||y - x||^2 subject to Ay <= b
            def objective(y):
                return np.sum((y - x) ** 2)

            def objective_grad(y):
                return 2 * (y - x)

            if self.constraint_type == "inequality":
                constraints = {
                    'type': 'ineq',
                    'fun': lambda y: self.b - self.A @ y,
                    'jac': lambda y: -self.A
                }
            else:  # equality
                constraints = {
                    'type': 'eq',
                    'fun': lambda y: self.A @ y - self.b,
                    'jac': lambda y: self.A
                }

            result = minimize(
                objective,
                x,
                jac=objective_grad,
                constraints=constraints,
                method='SLSQP'
            )

            if result.success:
                return result.x
            else:
                logger.warning(f"Projection failed: {result.message}")
                return x

        except Exception as e:
            logger.warning(f"Projection failed with error: {e}")
            return x


class NonlinearConstraint(Constraint):
    """
    General nonlinear constraint: g(x) ≤ 0 or h(x) = 0.

    Supports arbitrary nonlinear constraint functions.
    """

    def __init__(
        self,
        function: Callable[[np.ndarray], Union[float, np.ndarray]],
        gradient_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        hessian_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        constraint_type: str = "inequality",
        name: Optional[str] = None,
        tolerance: float = 1e-6
    ):
        """
        Initialize nonlinear constraint.

        Args:
            function: Constraint function (can return scalar or vector)
            gradient_function: Optional gradient/Jacobian function
            hessian_function: Optional Hessian function
            constraint_type: "inequality" or "equality"
            name: Optional constraint name
            tolerance: Tolerance for feasibility check
        """
        super().__init__(name, tolerance)
        self.function = function
        self.gradient_function = gradient_function
        self.hessian_function = hessian_function

        if constraint_type not in ["inequality", "equality"]:
            raise ValueError("constraint_type must be 'inequality' or 'equality'")
        self.constraint_type = constraint_type

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate nonlinear constraint violation."""
        result = self.function(x)

        if np.isscalar(result):
            result = np.array([result])
        else:
            result = np.asarray(result)

        if self.constraint_type == "inequality":
            violations = np.maximum(0, result)
        else:  # equality
            violations = np.abs(result)

        return np.max(violations) if len(violations) > 0 else 0.0

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Calculate constraint gradient."""
        if self.gradient_function is not None:
            return self.gradient_function(x)
        else:
            return self._numerical_gradient(x)

    def _numerical_gradient(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Calculate numerical gradient using finite differences."""
        grad = np.zeros_like(x)
        f0 = self.evaluate(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (self.evaluate(x_plus) - f0) / eps

        return grad


@dataclass
class ConstraintSet:
    """
    Container for multiple constraints.

    Provides unified interface for evaluating multiple constraints.
    """

    constraints: List[Constraint]

    def evaluate(self, x: np.ndarray) -> Tuple[float, List[float]]:
        """
        Evaluate all constraints.

        Args:
            x: Decision variable vector

        Returns:
            Tuple of (total_violation, individual_violations)
        """
        violations = []
        for constraint in self.constraints:
            violations.append(constraint.evaluate(x))

        total_violation = sum(violations)
        return total_violation, violations

    def is_feasible(self, x: np.ndarray) -> bool:
        """Check if solution satisfies all constraints."""
        return all(constraint.is_feasible(x) for constraint in self.constraints)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate aggregated constraint gradient.

        Returns gradient of most violated constraint.
        """
        max_violation = -float('inf')
        max_gradient = np.zeros_like(x)

        for constraint in self.constraints:
            violation = constraint.evaluate(x)
            if violation > max_violation:
                max_violation = violation
                max_gradient = constraint.gradient(x)

        return max_gradient

    def project(self, x: np.ndarray, max_iterations: int = 10) -> np.ndarray:
        """
        Sequential projection onto constraint sets.

        Uses alternating projections for multiple constraints.
        """
        projected = x.copy()

        for _ in range(max_iterations):
            old_projected = projected.copy()

            # Project onto each constraint
            for constraint in self.constraints:
                if not constraint.is_feasible(projected):
                    projected = constraint.project(projected)

            # Check convergence
            if np.allclose(projected, old_projected):
                break

        return projected

    def add_constraint(self, constraint: Constraint):
        """Add a new constraint to the set."""
        self.constraints.append(constraint)

    def remove_constraint(self, name: str):
        """Remove constraint by name."""
        self.constraints = [c for c in self.constraints if c.name != name]

    def __len__(self):
        """Number of constraints."""
        return len(self.constraints)

    def __iter__(self):
        """Iterate over constraints."""
        return iter(self.constraints)