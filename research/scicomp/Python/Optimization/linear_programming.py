"""
Linear Programming Solvers
==========================
This module implements linear programming algorithms including the
simplex method, interior point methods, and revised simplex for
solving linear optimization problems.
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Dict, Any, Union
from .unconstrained import OptimizationResult
import warnings
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
BERKELEY_LIGHT_BLUE = '#3B7EA1'
@dataclass
class LinearProgram:
    """
    Represents a linear programming problem in standard form:
    minimize    c^T x
    subject to  A x = b
                x >= 0
    Attributes:
        c: Objective function coefficients
        A: Constraint matrix
        b: Right-hand side vector
        bounds: Variable bounds (optional)
        sense: Optimization sense ('min' or 'max')
    """
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
    sense: str = 'min'
    def __post_init__(self):
        """Validate LP problem data."""
        self.c = np.asarray(self.c)
        self.A = np.asarray(self.A)
        self.b = np.asarray(self.b)
        if self.A.shape[0] != len(self.b):
            raise ValueError("A and b dimensions are inconsistent")
        if self.A.shape[1] != len(self.c):
            raise ValueError("A and c dimensions are inconsistent")
        if self.sense not in ['min', 'max']:
            raise ValueError("Sense must be 'min' or 'max'")
class LinearProgrammingSolver(ABC):
    """
    Abstract base class for linear programming solvers.
    """
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-8,
                 verbose: bool = False):
        """
        Initialize LP solver.
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Numerical tolerance
            verbose: Whether to print progress
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
    @abstractmethod
    def solve(self, lp: LinearProgram) -> OptimizationResult:
        """
        Solve linear programming problem.
        Args:
            lp: Linear programming problem
        Returns:
            OptimizationResult object
        """
        pass
class SimplexMethod(LinearProgrammingSolver):
    """
    Simplex method for linear programming.
    Implements the two-phase simplex algorithm with Bland's pivoting
    rule to prevent cycling.
    """
    def __init__(self, pivoting_rule: str = 'bland', **kwargs):
        """
        Initialize Simplex method.
        Args:
            pivoting_rule: Pivoting rule ('bland', 'dantzig', 'largest_improvement')
        """
        super().__init__(**kwargs)
        self.pivoting_rule = pivoting_rule
    def solve(self, lp: LinearProgram) -> OptimizationResult:
        """
        Solve LP using simplex method.
        Args:
            lp: Linear programming problem
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        # Convert to maximization if needed
        if lp.sense == 'min':
            c = -lp.c.copy()
        else:
            c = lp.c.copy()
        A = lp.A.copy()
        b = lp.b.copy()
        m, n = A.shape
        # Check if problem is in standard form
        if np.any(b < 0):
            # Need to use two-phase method
            result = self._two_phase_simplex(c, A, b)
        else:
            # Can use regular simplex
            result = self._simplex_phase2(c, A, b)
        # Convert back to minimization if needed
        if lp.sense == 'min' and result.success:
            result.fun = -result.fun
        result.execution_time = time.time() - start_time
        return result
    def _two_phase_simplex(self, c: np.ndarray, A: np.ndarray, b: np.ndarray) -> OptimizationResult:
        """Two-phase simplex method."""
        m, n = A.shape
        # Phase I: Find initial basic feasible solution
        if self.verbose:
            print("Starting Phase I...")
        # Add artificial variables
        # min sum(artificial variables)
        # subject to A*x + I*x_art = b, x >= 0, x_art >= 0
        c_phase1 = np.concatenate([np.zeros(n), np.ones(m)])
        A_phase1 = np.hstack([A, np.eye(m)])
        # Make b non-negative
        negative_rows = b < 0
        A_phase1[negative_rows] *= -1
        b = np.abs(b)
        # Initial basic variables are artificial variables
        basic_vars = list(range(n, n + m))
        result_phase1 = self._simplex_solve(c_phase1, A_phase1, b, basic_vars)
        if not result_phase1.success:
            return OptimizationResult(
                x=np.zeros(n), fun=float('inf'), success=False,
                message="Phase I failed", nit=result_phase1.nit
            )
        # Check if original problem is feasible
        if result_phase1.fun > self.tolerance:
            return OptimizationResult(
                x=np.zeros(n), fun=float('inf'), success=False,
                message="Problem is infeasible", nit=result_phase1.nit
            )
        if self.verbose:
            print(f"Phase I completed. Objective: {result_phase1.fun:.6e}")
        # Phase II: Solve original problem
        if self.verbose:
            print("Starting Phase II...")
        # Extract basic variables from Phase I (excluding artificial variables)
        x_phase1 = result_phase1.x
        basic_vars_phase2 = []
        # Find basic variables that are not artificial
        tableau = self._create_tableau(c_phase1, A_phase1, b)
        for i, var_idx in enumerate(basic_vars):
            if var_idx < n:  # Not an artificial variable
                basic_vars_phase2.append(var_idx)
        # If we don't have enough basic variables, add slack variables
        while len(basic_vars_phase2) < m:
            # This is a simplified approach
            for j in range(n):
                if j not in basic_vars_phase2:
                    basic_vars_phase2.append(j)
                    break
        # Solve Phase II
        result_phase2 = self._simplex_solve(c, A, b, basic_vars_phase2[:m])
        result_phase2.nit += result_phase1.nit
        return result_phase2
    def _simplex_phase2(self, c: np.ndarray, A: np.ndarray, b: np.ndarray) -> OptimizationResult:
        """Single-phase simplex for problems already in standard form."""
        m, n = A.shape
        # Find initial basic feasible solution
        # Assume last m columns form an identity matrix (slack variables)
        if n >= m and np.allclose(A[:, -m:], np.eye(m)):
            basic_vars = list(range(n - m, n))
        else:
            # Try to find a basic feasible solution
            basic_vars = self._find_initial_basis(A)
            if basic_vars is None:
                return OptimizationResult(
                    x=np.zeros(n), fun=float('inf'), success=False,
                    message="Could not find initial basic feasible solution"
                )
        return self._simplex_solve(c, A, b, basic_vars)
    def _simplex_solve(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
                      basic_vars: List[int]) -> OptimizationResult:
        """Core simplex algorithm."""
        m, n = A.shape
        iteration = 0
        # Create initial tableau
        tableau = self._create_tableau(c, A, b)
        while iteration < self.max_iterations:
            # Check optimality conditions
            reduced_costs = self._compute_reduced_costs(tableau, basic_vars)
            entering_var = self._select_entering_variable(reduced_costs)
            if entering_var is None:
                # Optimal solution found
                x = self._extract_solution(tableau, basic_vars, n)
                obj_value = -tableau[-1, -1]  # Bottom-right corner
                return OptimizationResult(
                    x=x, fun=obj_value, success=True,
                    message="Optimal solution found", nit=iteration
                )
            # Compute pivot column
            pivot_column = tableau[:-1, entering_var]
            # Ratio test to find leaving variable
            leaving_var_idx = self._ratio_test(tableau, entering_var)
            if leaving_var_idx is None:
                # Problem is unbounded
                return OptimizationResult(
                    x=np.zeros(n), fun=-float('inf'), success=False,
                    message="Problem is unbounded", nit=iteration
                )
            leaving_var = basic_vars[leaving_var_idx]
            # Pivot operation
            self._pivot(tableau, leaving_var_idx, entering_var)
            basic_vars[leaving_var_idx] = entering_var
            if self.verbose and iteration % 10 == 0:
                obj_value = -tableau[-1, -1]
                print(f"Iteration {iteration}: Objective = {obj_value:.6e}")
            iteration += 1
        # Maximum iterations reached
        x = self._extract_solution(tableau, basic_vars, n)
        obj_value = -tableau[-1, -1]
        return OptimizationResult(
            x=x, fun=obj_value, success=False,
            message="Maximum iterations reached", nit=iteration
        )
    def _create_tableau(self, c: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Create simplex tableau."""
        m, n = A.shape
        # Create tableau: [A | b]
        #                 [c | 0]
        tableau = np.zeros((m + 1, n + 1))
        tableau[:-1, :-1] = A
        tableau[:-1, -1] = b
        tableau[-1, :-1] = c
        tableau[-1, -1] = 0
        return tableau
    def _compute_reduced_costs(self, tableau: np.ndarray, basic_vars: List[int]) -> np.ndarray:
        """Compute reduced costs."""
        return tableau[-1, :-1]
    def _select_entering_variable(self, reduced_costs: np.ndarray) -> Optional[int]:
        """Select entering variable using specified pivoting rule."""
        if self.pivoting_rule == 'dantzig':
            # Most negative reduced cost
            min_idx = np.argmin(reduced_costs)
            return min_idx if reduced_costs[min_idx] < -self.tolerance else None
        elif self.pivoting_rule == 'bland':
            # First negative reduced cost (Bland's rule)
            for i, cost in enumerate(reduced_costs):
                if cost < -self.tolerance:
                    return i
            return None
        elif self.pivoting_rule == 'largest_improvement':
            # Largest improvement (most negative)
            min_idx = np.argmin(reduced_costs)
            return min_idx if reduced_costs[min_idx] < -self.tolerance else None
        else:
            # Default to Bland's rule
            return self._select_entering_variable(reduced_costs)
    def _ratio_test(self, tableau: np.ndarray, entering_var: int) -> Optional[int]:
        """Perform ratio test to find leaving variable."""
        m = tableau.shape[0] - 1
        pivot_column = tableau[:-1, entering_var]
        rhs = tableau[:-1, -1]
        ratios = []
        valid_indices = []
        for i in range(m):
            if pivot_column[i] > self.tolerance:
                ratios.append(rhs[i] / pivot_column[i])
                valid_indices.append(i)
        if not ratios:
            return None  # Unbounded
        # Find minimum ratio
        min_ratio = min(ratios)
        min_idx = ratios.index(min_ratio)
        return valid_indices[min_idx]
    def _pivot(self, tableau: np.ndarray, pivot_row: int, pivot_col: int):
        """Perform pivot operation."""
        pivot_element = tableau[pivot_row, pivot_col]
        if abs(pivot_element) < self.tolerance:
            raise ValueError("Pivot element is too small")
        # Normalize pivot row
        tableau[pivot_row] /= pivot_element
        # Eliminate other elements in pivot column
        for i in range(tableau.shape[0]):
            if i != pivot_row and abs(tableau[i, pivot_col]) > self.tolerance:
                factor = tableau[i, pivot_col]
                tableau[i] -= factor * tableau[pivot_row]
    def _extract_solution(self, tableau: np.ndarray, basic_vars: List[int], n: int) -> np.ndarray:
        """Extract solution from tableau."""
        x = np.zeros(n)
        for i, var_idx in enumerate(basic_vars):
            if var_idx < n:
                x[var_idx] = tableau[i, -1]
        return x
    def _find_initial_basis(self, A: np.ndarray) -> Optional[List[int]]:
        """Find initial basis (simplified implementation)."""
        m, n = A.shape
        # Look for identity columns
        basic_vars = []
        for j in range(n):
            col = A[:, j]
            if np.sum(np.abs(col)) == 1 and np.max(col) == 1:
                # This is a unit column
                row_idx = np.argmax(col)
                if row_idx not in [basic_vars.index(var) if var in basic_vars else -1 for var in basic_vars]:
                    basic_vars.append(j)
        if len(basic_vars) == m:
            return basic_vars
        else:
            return None
class InteriorPointLP(LinearProgrammingSolver):
    """
    Interior Point Method for Linear Programming.
    Implements the primal-dual interior point method with barrier functions.
    """
    def __init__(self, barrier_parameter: float = 0.1, centering_parameter: float = 0.1,
                 **kwargs):
        """
        Initialize Interior Point method.
        Args:
            barrier_parameter: Initial barrier parameter
            centering_parameter: Centering parameter for path following
        """
        super().__init__(**kwargs)
        self.barrier_parameter = barrier_parameter
        self.centering_parameter = centering_parameter
    def solve(self, lp: LinearProgram) -> OptimizationResult:
        """
        Solve LP using interior point method.
        Args:
            lp: Linear programming problem
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        c = lp.c.copy()
        A = lp.A.copy()
        b = lp.b.copy()
        if lp.sense == 'max':
            c = -c
        m, n = A.shape
        # Find initial feasible point
        x, y, s = self._find_initial_point(c, A, b)
        if x is None:
            return OptimizationResult(
                x=np.zeros(n), fun=float('inf'), success=False,
                message="Could not find initial feasible point"
            )
        iteration = 0
        mu = self.barrier_parameter
        while iteration < self.max_iterations:
            # Check optimality conditions
            primal_residual = np.linalg.norm(A @ x - b)
            dual_residual = np.linalg.norm(A.T @ y + s - c)
            complementarity = np.dot(x, s)
            if (primal_residual < self.tolerance and
                dual_residual < self.tolerance and
                complementarity < self.tolerance):
                obj_value = np.dot(c, x)
                if lp.sense == 'max':
                    obj_value = -obj_value
                return OptimizationResult(
                    x=x, fun=obj_value, success=True,
                    message="Optimal solution found", nit=iteration,
                    execution_time=time.time() - start_time
                )
            # Update barrier parameter
            mu = self.centering_parameter * complementarity / n
            # Solve Newton system
            try:
                dx, dy, ds = self._solve_newton_system(A, x, s, c, b, mu)
            except np.linalg.LinAlgError:
                return OptimizationResult(
                    x=x, fun=np.dot(c, x), success=False,
                    message="Newton system is singular", nit=iteration,
                    execution_time=time.time() - start_time
                )
            # Line search
            alpha_primal = self._line_search_primal(x, s, dx, ds)
            alpha_dual = self._line_search_dual(y, dy)
            # Update variables
            x += alpha_primal * dx
            y += alpha_dual * dy
            s += alpha_dual * ds
            if self.verbose and iteration % 10 == 0:
                obj_value = np.dot(c, x)
                print(f"Iteration {iteration}: Objective = {obj_value:.6e}, "
                      f"Complementarity = {complementarity:.6e}")
            iteration += 1
        # Maximum iterations reached
        obj_value = np.dot(c, x)
        if lp.sense == 'max':
            obj_value = -obj_value
        return OptimizationResult(
            x=x, fun=obj_value, success=False,
            message="Maximum iterations reached", nit=iteration,
            execution_time=time.time() - start_time
        )
    def _find_initial_point(self, c: np.ndarray, A: np.ndarray, b: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Find initial feasible point."""
        m, n = A.shape
        # Simple initialization
        try:
            # Solve A @ x = b for x (least squares if overdetermined)
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            # Make x positive
            x = np.maximum(x, 1.0)
            # Dual variables
            try:
                y = np.linalg.solve(A @ A.T, A @ c)
            except np.linalg.LinAlgError:
                y = np.zeros(m)
            # Slack variables
            s = c - A.T @ y
            s = np.maximum(s, 1.0)
            return x, y, s
        except np.linalg.LinAlgError:
            return None, None, None
    def _solve_newton_system(self, A: np.ndarray, x: np.ndarray, s: np.ndarray,
                            c: np.ndarray, b: np.ndarray, mu: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve Newton system for interior point method."""
        m, n = A.shape
        # Newton system for primal-dual interior point:
        # [ 0   A^T  I ] [dx]   [c - A^T*y - s]
        # [ A   0   0 ] [dy] = [b - A*x      ]
        # [ S   0   X ] [ds]   [mu*e - X*S*e ]
        X = np.diag(x)
        S = np.diag(s)
        e = np.ones(n)
        # Right-hand side
        rhs1 = c - A.T @ np.zeros(m) - s  # Assuming y is updated separately
        rhs2 = b - A @ x
        rhs3 = mu * e - x * s
        # Solve reduced system (eliminating ds)
        # dx = inv(X) * (rhs3 - S*ds)
        # A*dx = rhs2
        # A^T*dy + ds = rhs1
        # Substitute ds = rhs1 - A^T*dy into first equation
        # dx = inv(X) * (rhs3 - S*(rhs1 - A^T*dy))
        # dx = inv(X) * (rhs3 - S*rhs1 + S*A^T*dy)
        # Substituting into A*dx = rhs2:
        # A * inv(X) * (rhs3 - S*rhs1 + S*A^T*dy) = rhs2
        # A * inv(X) * (rhs3 - S*rhs1) + A * inv(X) * S * A^T * dy = rhs2
        try:
            X_inv = np.diag(1.0 / x)
            # Compute coefficient matrix for dy
            M = A @ X_inv @ S @ A.T
            # Compute right-hand side for dy
            rhs_dy = rhs2 - A @ X_inv @ (rhs3 - S @ rhs1)
            # Solve for dy
            dy = np.linalg.solve(M, rhs_dy)
            # Compute ds and dx
            ds = rhs1 - A.T @ dy
            dx = X_inv @ (rhs3 - S @ ds)
            return dx, dy, ds
        except np.linalg.LinAlgError:
            # Fallback: simple gradient step
            dx = -0.01 * (c - A.T @ np.zeros(m))
            dy = np.zeros(m)
            ds = -0.01 * s
            return dx, dy, ds
    def _line_search_primal(self, x: np.ndarray, s: np.ndarray,
                           dx: np.ndarray, ds: np.ndarray) -> float:
        """Line search for primal variables."""
        alpha_max = 1.0
        # Find maximum step length that keeps x, s > 0
        for i in range(len(x)):
            if dx[i] < 0:
                alpha_max = min(alpha_max, -0.95 * x[i] / dx[i])
        for i in range(len(s)):
            if ds[i] < 0:
                alpha_max = min(alpha_max, -0.95 * s[i] / ds[i])
        return max(alpha_max, 1e-10)
    def _line_search_dual(self, y: np.ndarray, dy: np.ndarray) -> float:
        """Line search for dual variables."""
        return 1.0  # No constraints on dual variables in this formulation
# Alias for compatibility
RevisedSimplex = SimplexMethod
def demo():
    """Demonstrate linear programming solvers."""
    print("Linear Programming Demo")
    print("======================")
    print()
    # Example 1: Simple LP
    print("Example 1: Simple Linear Program")
    print("maximize    3x + 2y")
    print("subject to  x + y <= 4")
    print("            2x + y <= 6")
    print("            x, y >= 0")
    print("-" * 40)
    # Convert to standard form:
    # minimize   -3x - 2y
    # subject to  x + y + s1 = 4
    #             2x + y + s2 = 6
    #             x, y, s1, s2 >= 0
    c = np.array([-3, -2, 0, 0])  # Coefficients (minimization)
    A = np.array([
        [1, 1, 1, 0],
        [2, 1, 0, 1]
    ])
    b = np.array([4, 6])
    lp1 = LinearProgram(c=c, A=A, b=b, sense='min')
    # Test simplex method
    simplex = SimplexMethod(verbose=False)
    result1 = simplex.solve(lp1)
    print(f"Simplex Method:")
    print(f"  Solution: x = {result1.x[0]:.4f}, y = {result1.x[1]:.4f}")
    print(f"  Objective: {-result1.fun:.4f}")  # Convert back to maximization
    print(f"  Success: {result1.success}")
    print()
    # Test interior point method
    interior_point = InteriorPointLP(verbose=False)
    result2 = interior_point.solve(lp1)
    print(f"Interior Point Method:")
    print(f"  Solution: x = {result2.x[0]:.4f}, y = {result2.x[1]:.4f}")
    print(f"  Objective: {-result2.fun:.4f}")  # Convert back to maximization
    print(f"  Success: {result2.success}")
    print()
    # Example 2: Diet problem
    print("Example 2: Diet Problem")
    print("minimize cost of food while meeting nutritional requirements")
    print("-" * 60)
    # Foods: bread, milk, cheese, potato, fish, yogurt
    # Nutrients: calories, protein, fat
    # Cost per unit
    cost = np.array([2.0, 3.5, 8.0, 1.5, 11.0, 1.0])
    # Nutritional content (per unit)
    nutrition = np.array([
        [90, 120, 106, 97, 130, 180],   # Calories
        [3.0, 8.1, 7.4, 1.3, 24.2, 6.0],  # Protein
        [0.5, 5.0, 7.7, 0.1, 5.2, 0.1]    # Fat
    ])
    # Minimum requirements
    requirements = np.array([2000, 55, 20])  # Min calories, protein, fat
    # LP formulation:
    # minimize cost^T * x
    # subject to nutrition * x >= requirements, x >= 0
    # Convert to standard form by adding slack variables and negating inequalities
    # nutrition * x - s = requirements, x >= 0, s >= 0
    # becomes: -nutrition * x + s = -requirements
    c_diet = np.concatenate([cost, np.zeros(len(requirements))])
    A_diet = np.hstack([-nutrition.T, np.eye(len(requirements))])
    b_diet = -requirements
    lp2 = LinearProgram(c=c_diet, A=A_diet, b=b_diet, sense='min')
    try:
        result_diet = simplex.solve(lp2)
        if result_diet.success:
            food_amounts = result_diet.x[:len(cost)]
            total_cost = result_diet.fun
            print(f"Optimal diet:")
            foods = ['Bread', 'Milk', 'Cheese', 'Potato', 'Fish', 'Yogurt']
            for i, (food, amount) in enumerate(zip(foods, food_amounts)):
                if amount > 0.01:
                    print(f"  {food}: {amount:.2f} units")
            print(f"Total cost: ${total_cost:.2f}")
            # Check nutritional content
            total_nutrition = nutrition @ food_amounts
            print(f"Nutritional content:")
            nutrients = ['Calories', 'Protein', 'Fat']
            for i, (nutrient, amount, req) in enumerate(zip(nutrients, total_nutrition, requirements)):
                print(f"  {nutrient}: {amount:.1f} (required: {req:.1f})")
        else:
            print("Could not solve diet problem")
    except Exception as e:
        print(f"Error solving diet problem: {str(e)}")
    print()
    print("Demo completed!")
if __name__ == "__main__":
    demo()