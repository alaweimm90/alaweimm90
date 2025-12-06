"""
Feasibility preservation methods for constrained optimization.

This module provides techniques for maintaining and restoring feasibility:
- Feasibility-preserving genetic operators
- Projection to feasible region
- Random feasible solution generation
- Feasibility checking utilities
"""

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

from .constraints import BoxConstraint, Constraint, ConstraintSet

logger = logging.getLogger(__name__)


class FeasibilityPreservingCrossover:
    """
    Crossover operators that preserve feasibility.

    Includes methods designed to produce feasible offspring
    from feasible parents.
    """

    def __init__(
        self,
        constraints: ConstraintSet,
        repair_infeasible: bool = True
    ):
        """
        Initialize feasibility-preserving crossover.

        Args:
            constraints: Set of constraints to satisfy
            repair_infeasible: Whether to repair infeasible offspring
        """
        self.constraints = constraints
        self.repair_infeasible = repair_infeasible

    def blend_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blend crossover (BLX-α) with feasibility preservation.

        Creates offspring in the hypercube defined by parents,
        extended by factor α.

        Args:
            parent1: First parent solution
            parent2: Second parent solution
            alpha: Extension factor (0.5 recommended)

        Returns:
            Tuple of two offspring
        """
        d = np.abs(parent1 - parent2)
        lower = np.minimum(parent1, parent2) - alpha * d
        upper = np.maximum(parent1, parent2) + alpha * d

        # Generate random offspring in extended range
        offspring1 = np.random.uniform(lower, upper)
        offspring2 = np.random.uniform(lower, upper)

        # Check and repair feasibility
        if not self.constraints.is_feasible(offspring1):
            if self.repair_infeasible:
                offspring1 = self._repair_blend(offspring1, parent1, parent2)
            else:
                offspring1 = parent1.copy()  # Fallback to parent

        if not self.constraints.is_feasible(offspring2):
            if self.repair_infeasible:
                offspring2 = self._repair_blend(offspring2, parent1, parent2)
            else:
                offspring2 = parent2.copy()  # Fallback to parent

        return offspring1, offspring2

    def _repair_blend(
        self,
        offspring: np.ndarray,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> np.ndarray:
        """
        Repair infeasible blend crossover offspring.

        Uses binary search on the line between offspring and parents.
        """
        # Try line search to nearest parent
        nearest_parent = parent1 if np.linalg.norm(offspring - parent1) < np.linalg.norm(offspring - parent2) else parent2

        # Binary search for feasible point
        left, right = 0.0, 1.0
        for _ in range(20):  # Max iterations
            mid = (left + right) / 2
            candidate = offspring + mid * (nearest_parent - offspring)

            if self.constraints.is_feasible(candidate):
                right = mid
                if right < 0.01:  # Close enough
                    return candidate
            else:
                left = mid

        # If no feasible point found, return parent
        return nearest_parent

    def arithmetic_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        alpha: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Arithmetic crossover with feasibility checking.

        Creates convex combinations of parents, which preserves
        convex constraints.

        Args:
            parent1: First parent solution
            parent2: Second parent solution
            alpha: Mixing coefficients (random if None)

        Returns:
            Tuple of two offspring
        """
        if alpha is None:
            alpha = np.random.random(len(parent1))

        offspring1 = alpha * parent1 + (1 - alpha) * parent2
        offspring2 = (1 - alpha) * parent1 + alpha * parent2

        # For convex feasible regions, offspring should be feasible
        # if parents are feasible. Check and repair if needed.
        if not self.constraints.is_feasible(offspring1):
            if self.repair_infeasible:
                offspring1 = self.constraints.project(offspring1)
            else:
                offspring1 = parent1.copy()

        if not self.constraints.is_feasible(offspring2):
            if self.repair_infeasible:
                offspring2 = self.constraints.project(offspring2)
            else:
                offspring2 = parent2.copy()

        return offspring1, offspring2

    def simplex_crossover(
        self,
        parents: List[np.ndarray],
        n_offspring: int = 1
    ) -> List[np.ndarray]:
        """
        Simplex crossover (SPX) for multiple parents.

        Creates offspring within simplex defined by parents,
        good for preserving feasibility in convex regions.

        Args:
            parents: List of parent solutions (3+ recommended)
            n_offspring: Number of offspring to generate

        Returns:
            List of offspring solutions
        """
        n_parents = len(parents)
        if n_parents < 3:
            raise ValueError("Simplex crossover requires at least 3 parents")

        # Calculate centroid
        centroid = np.mean(parents, axis=0)

        offspring = []
        for _ in range(n_offspring):
            # Generate random weights (Dirichlet distribution)
            weights = np.random.exponential(1, n_parents)
            weights /= weights.sum()

            # Create offspring as weighted combination
            child = np.zeros_like(parents[0])
            for i, parent in enumerate(parents):
                child += weights[i] * parent

            # Expand around centroid
            expansion_factor = np.random.uniform(0.8, 1.2)
            child = centroid + expansion_factor * (child - centroid)

            # Check feasibility
            if self.constraints.is_feasible(child):
                offspring.append(child)
            elif self.repair_infeasible:
                # Project back to feasible region
                repaired = self.constraints.project(child)
                offspring.append(repaired)
            else:
                # Use random parent as fallback
                offspring.append(parents[np.random.randint(n_parents)].copy())

        return offspring


class FeasibilityPreservingMutation:
    """
    Mutation operators that preserve or restore feasibility.
    """

    def __init__(
        self,
        constraints: ConstraintSet,
        repair_infeasible: bool = True
    ):
        """
        Initialize feasibility-preserving mutation.

        Args:
            constraints: Set of constraints to satisfy
            repair_infeasible: Whether to repair infeasible mutants
        """
        self.constraints = constraints
        self.repair_infeasible = repair_infeasible

    def boundary_mutation(
        self,
        x: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
        mutation_prob: float = 0.1
    ) -> np.ndarray:
        """
        Mutation that respects box constraints.

        Mutates variables to random values within bounds.

        Args:
            x: Solution to mutate
            bounds: (lower_bounds, upper_bounds)
            mutation_prob: Probability of mutating each variable

        Returns:
            Mutated solution
        """
        mutated = x.copy()
        lower_bounds, upper_bounds = bounds

        for i in range(len(x)):
            if np.random.random() < mutation_prob:
                # Random value within bounds
                mutated[i] = np.random.uniform(
                    lower_bounds[i],
                    upper_bounds[i]
                )

        # Check general constraints
        if not self.constraints.is_feasible(mutated):
            if self.repair_infeasible:
                mutated = self._repair_to_feasible(mutated, x)
            else:
                return x.copy()  # Return original if infeasible

        return mutated

    def gaussian_mutation(
        self,
        x: np.ndarray,
        sigma: float = 0.1,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Gaussian mutation with feasibility preservation.

        Args:
            x: Solution to mutate
            sigma: Standard deviation for Gaussian noise
            bounds: Optional box constraints

        Returns:
            Mutated solution
        """
        mutated = x + np.random.normal(0, sigma, size=len(x))

        # Apply box constraints if provided
        if bounds is not None:
            lower_bounds, upper_bounds = bounds
            mutated = np.clip(mutated, lower_bounds, upper_bounds)

        # Check and repair general constraints
        if not self.constraints.is_feasible(mutated):
            if self.repair_infeasible:
                mutated = self._repair_to_feasible(mutated, x)
            else:
                return x.copy()

        return mutated

    def _repair_to_feasible(
        self,
        infeasible: np.ndarray,
        feasible: np.ndarray
    ) -> np.ndarray:
        """
        Repair infeasible solution using line search.

        Searches on line between infeasible and feasible solution.

        Args:
            infeasible: Infeasible solution to repair
            feasible: Known feasible solution

        Returns:
            Repaired feasible solution
        """
        # Binary search on line segment
        left, right = 0.0, 1.0

        for _ in range(20):  # Max iterations
            mid = (left + right) / 2
            candidate = feasible + mid * (infeasible - feasible)

            if self.constraints.is_feasible(candidate):
                left = mid  # Move towards infeasible
                if left > 0.99:  # Close enough to infeasible
                    return candidate
            else:
                right = mid  # Move towards feasible

        # Return point closest to infeasible that's feasible
        return feasible + left * (infeasible - feasible)

    def non_uniform_mutation(
        self,
        x: np.ndarray,
        generation: int,
        max_generation: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        b: float = 5.0
    ) -> np.ndarray:
        """
        Non-uniform mutation that decreases with generations.

        Mutation strength decreases over time for fine-tuning.

        Args:
            x: Solution to mutate
            generation: Current generation
            max_generation: Maximum generations
            bounds: Variable bounds
            b: System parameter controlling decrease rate

        Returns:
            Mutated solution
        """
        mutated = x.copy()
        lower_bounds, upper_bounds = bounds

        for i in range(len(x)):
            if np.random.random() < 0.5:
                # Mutate towards upper bound
                delta = upper_bounds[i] - x[i]
                r = np.random.random()
                t = generation / max_generation
                mutation = delta * (1 - r ** ((1 - t) ** b))
                mutated[i] = x[i] + mutation
            else:
                # Mutate towards lower bound
                delta = x[i] - lower_bounds[i]
                r = np.random.random()
                t = generation / max_generation
                mutation = delta * (1 - r ** ((1 - t) ** b))
                mutated[i] = x[i] - mutation

        # Ensure within bounds
        mutated = np.clip(mutated, lower_bounds, upper_bounds)

        # Check general constraints
        if not self.constraints.is_feasible(mutated):
            if self.repair_infeasible:
                mutated = self.constraints.project(mutated)
            else:
                return x.copy()

        return mutated


def feasibility_check(
    x: np.ndarray,
    constraints: ConstraintSet,
    tolerance: float = 1e-6
) -> Tuple[bool, float, List[str]]:
    """
    Comprehensive feasibility check with detailed information.

    Args:
        x: Solution to check
        constraints: Set of constraints
        tolerance: Feasibility tolerance

    Returns:
        Tuple of:
        - is_feasible: Boolean feasibility status
        - total_violation: Sum of all violations
        - violated_constraints: Names of violated constraints
    """
    total_violation = 0.0
    violated_constraints = []

    for constraint in constraints.constraints:
        violation = constraint.evaluate(x)
        if violation > tolerance:
            total_violation += violation
            violated_constraints.append(constraint.name)

    is_feasible = len(violated_constraints) == 0

    return is_feasible, total_violation, violated_constraints


def project_to_feasible_region(
    x: np.ndarray,
    constraints: ConstraintSet,
    method: str = "sequential",
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> np.ndarray:
    """
    Project solution to feasible region.

    Args:
        x: Solution to project
        constraints: Set of constraints
        method: Projection method ("sequential", "simultaneous")
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Feasible solution (or best attempt)
    """
    if method == "sequential":
        # Sequential projection (alternating projections)
        projected = x.copy()
        for iteration in range(max_iterations):
            old_projected = projected.copy()

            # Project onto each constraint sequentially
            for constraint in constraints.constraints:
                if not constraint.is_feasible(projected):
                    projected = constraint.project(projected)

            # Check convergence
            if np.linalg.norm(projected - old_projected) < tolerance:
                break

        return projected

    elif method == "simultaneous":
        # Simultaneous projection using optimization
        try:
            from scipy.optimize import minimize

            # Minimize distance to original point
            def objective(y):
                return np.sum((y - x) ** 2)

            # Convert constraints to scipy format
            scipy_constraints = []
            for constraint in constraints.constraints:
                scipy_constraints.append({
                    'type': 'ineq',
                    'fun': lambda y, c=constraint: -c.evaluate(y),
                    'jac': lambda y, c=constraint: -c.gradient(y)
                })

            result = minimize(
                objective,
                x,
                constraints=scipy_constraints,
                method='SLSQP',
                options={'maxiter': max_iterations}
            )

            if result.success:
                return result.x
            else:
                logger.warning(f"Projection failed: {result.message}")
                # Fall back to sequential
                return project_to_feasible_region(
                    x, constraints, "sequential", max_iterations, tolerance
                )

        except Exception as e:
            logger.warning(f"Simultaneous projection failed: {e}")
            # Fall back to sequential
            return project_to_feasible_region(
                x, constraints, "sequential", max_iterations, tolerance
            )

    else:
        raise ValueError(f"Unknown projection method: {method}")


def random_feasible_solution(
    constraints: ConstraintSet,
    bounds: Tuple[np.ndarray, np.ndarray],
    max_attempts: int = 10000,
    repair_attempts: int = 10
) -> Optional[np.ndarray]:
    """
    Generate random feasible solution.

    Args:
        constraints: Set of constraints
        bounds: Variable bounds
        max_attempts: Maximum random generation attempts
        repair_attempts: Attempts to repair each infeasible solution

    Returns:
        Feasible solution or None if not found
    """
    lower_bounds, upper_bounds = bounds

    for attempt in range(max_attempts):
        # Generate random solution
        x = np.random.uniform(lower_bounds, upper_bounds)

        # Check feasibility
        if constraints.is_feasible(x):
            return x

        # Try to repair
        if repair_attempts > 0 and attempt % 100 == 0:
            repaired = project_to_feasible_region(
                x,
                constraints,
                max_iterations=repair_attempts
            )
            if constraints.is_feasible(repaired):
                return repaired

    logger.warning(
        f"Could not generate feasible solution after {max_attempts} attempts"
    )
    return None