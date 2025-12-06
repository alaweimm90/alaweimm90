"""
Core multi-objective optimization infrastructure.

This module provides fundamental concepts and algorithms for multi-objective
optimization including Pareto dominance, non-dominated sorting, crowding
distance, and hypervolume calculations.

Mathematical Foundation:
- Pareto Dominance: Solution x dominates y iff:
    ∀i: f_i(x) ≤ f_i(y) and ∃j: f_j(x) < f_j(y) (minimization)
- Non-dominated Sorting: O(MN²) complexity where M is objectives, N is population
- Crowding Distance: Manhattan distance in objective space normalized by range
- Hypervolume: Volume of space dominated by a set relative to reference point
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import distance

logger = logging.getLogger(__name__)


@dataclass
class MultiObjectiveProblem:
    """
    Representation of a multi-objective optimization problem.

    Attributes:
        objectives: List of objective functions to minimize/maximize
        n_objectives: Number of objectives
        n_variables: Number of decision variables
        bounds: Variable bounds as (lower, upper) arrays
        constraints: List of constraint functions g(x) <= 0
        minimize: List of booleans indicating minimize (True) or maximize (False)
        reference_point: Reference point for hypervolume calculation
        metadata: Additional problem-specific metadata
    """
    objectives: List[Callable[[np.ndarray], float]]
    n_objectives: int
    n_variables: int
    bounds: Tuple[np.ndarray, np.ndarray]
    constraints: List[Callable[[np.ndarray], float]] = field(default_factory=list)
    minimize: List[bool] = field(default_factory=list)
    reference_point: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if not self.minimize:
            self.minimize = [True] * self.n_objectives

        if len(self.minimize) != self.n_objectives:
            raise ValueError(
                f"Length of minimize ({len(self.minimize)}) must match "
                f"n_objectives ({self.n_objectives})"
            )

        if self.reference_point is None:
            # Default reference point for minimization problems
            self.reference_point = np.ones(self.n_objectives) * 1e6

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate all objectives for a solution.

        Args:
            x: Decision variable vector

        Returns:
            Array of objective values
        """
        obj_values = np.zeros(self.n_objectives)
        for i, obj_func in enumerate(self.objectives):
            value = obj_func(x)
            # Convert maximization to minimization
            if not self.minimize[i]:
                value = -value
            obj_values[i] = value
        return obj_values

    def evaluate_constraints(self, x: np.ndarray) -> Tuple[bool, float]:
        """
        Evaluate all constraints for a solution.

        Args:
            x: Decision variable vector

        Returns:
            Tuple of (is_feasible, total_violation)
        """
        if not self.constraints:
            return True, 0.0

        violations = []
        for constraint_func in self.constraints:
            violation = max(0, constraint_func(x))
            violations.append(violation)

        total_violation = sum(violations)
        is_feasible = total_violation == 0
        return is_feasible, total_violation


@dataclass
class MultiObjectiveSolution:
    """
    Representation of a solution in multi-objective optimization.

    Attributes:
        variables: Decision variable values
        objectives: Objective function values
        constraint_violation: Total constraint violation (0 if feasible)
        rank: Non-domination rank (0 is best)
        crowding_distance: Crowding distance in objective space
        metadata: Additional solution-specific metadata
    """
    variables: np.ndarray
    objectives: np.ndarray
    constraint_violation: float = 0.0
    rank: int = 0
    crowding_distance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def dominates(self, other: "MultiObjectiveSolution") -> bool:
        """
        Check if this solution dominates another solution.

        Constraint-domination is used: feasible solutions always dominate
        infeasible ones, among infeasible solutions the one with lower
        violation dominates.

        Args:
            other: Another solution to compare

        Returns:
            True if this solution dominates the other
        """
        # Constraint domination
        if self.constraint_violation == 0 and other.constraint_violation > 0:
            return True
        if self.constraint_violation > 0 and other.constraint_violation == 0:
            return False
        if self.constraint_violation > 0 and other.constraint_violation > 0:
            return self.constraint_violation < other.constraint_violation

        # Both feasible - check Pareto dominance
        return dominates(self.objectives, other.objectives)


class ParetoFront:
    """
    Container and utilities for Pareto front management.

    The Pareto front is the set of non-dominated solutions representing
    the best trade-offs between objectives.
    """

    def __init__(self):
        """Initialize empty Pareto front."""
        self.solutions: List[MultiObjectiveSolution] = []
        self._objectives_array: Optional[np.ndarray] = None

    def add(self, solution: MultiObjectiveSolution) -> bool:
        """
        Try to add a solution to the Pareto front.

        The solution is added only if it's not dominated by any existing
        solution. Dominated solutions are removed.

        Args:
            solution: Solution to potentially add

        Returns:
            True if solution was added, False otherwise
        """
        # Check if solution is dominated by any in front
        for existing in self.solutions[:]:
            if existing.dominates(solution):
                return False
            if solution.dominates(existing):
                self.solutions.remove(existing)

        self.solutions.append(solution)
        self._objectives_array = None  # Invalidate cache
        return True

    def get_objectives(self) -> np.ndarray:
        """
        Get objective values of all solutions in the front.

        Returns:
            2D array of shape (n_solutions, n_objectives)
        """
        if self._objectives_array is None and self.solutions:
            self._objectives_array = np.array([s.objectives for s in self.solutions])
        return self._objectives_array if self._objectives_array is not None else np.array([])

    def size(self) -> int:
        """Get number of solutions in the Pareto front."""
        return len(self.solutions)

    def clear(self):
        """Clear all solutions from the Pareto front."""
        self.solutions.clear()
        self._objectives_array = None


def dominates(obj1: np.ndarray, obj2: np.ndarray) -> bool:
    """
    Check if obj1 Pareto-dominates obj2 (for minimization).

    Mathematical definition:
    obj1 dominates obj2 iff:
    - ∀i: obj1[i] <= obj2[i] (obj1 is not worse in any objective)
    - ∃j: obj1[j] < obj2[j] (obj1 is strictly better in at least one)

    Args:
        obj1: First objective vector
        obj2: Second objective vector

    Returns:
        True if obj1 dominates obj2
    """
    not_worse = np.all(obj1 <= obj2)
    strictly_better = np.any(obj1 < obj2)
    return not_worse and strictly_better


def fast_non_dominated_sort(population: List[MultiObjectiveSolution]) -> List[List[int]]:
    """
    Fast non-dominated sorting algorithm from NSGA-II.

    Time Complexity: O(MN²) where M is objectives, N is population size
    Space Complexity: O(N²) for domination lists

    Algorithm:
    1. For each solution p:
       - Count solutions that dominate p (n_p)
       - List solutions dominated by p (S_p)
    2. Solutions with n_p = 0 form first front F_1
    3. For each front F_i:
       - For each p in F_i and q in S_p:
         - Decrement n_q
         - If n_q = 0, add q to F_{i+1}

    Args:
        population: List of solutions to sort

    Returns:
        List of fronts, each front is a list of solution indices
    """
    n = len(population)
    if n == 0:
        return []

    # Initialize data structures
    domination_count = np.zeros(n, dtype=int)  # n_p
    dominated_solutions = [[] for _ in range(n)]  # S_p
    fronts = []
    current_front = []

    # Step 1: Calculate domination relationships
    for i in range(n):
        for j in range(n):
            if i != j:
                if population[i].dominates(population[j]):
                    dominated_solutions[i].append(j)
                elif population[j].dominates(population[i]):
                    domination_count[i] += 1

        # Step 2: Identify first front
        if domination_count[i] == 0:
            population[i].rank = 0
            current_front.append(i)

    # Step 3: Identify subsequent fronts
    front_rank = 0
    while current_front:
        fronts.append(current_front)
        next_front = []

        for i in current_front:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = front_rank + 1
                    next_front.append(j)

        current_front = next_front
        front_rank += 1

    return fronts


def crowding_distance(population: List[MultiObjectiveSolution], indices: List[int]) -> None:
    """
    Calculate crowding distance for solutions.

    Crowding distance estimates the density of solutions in objective space,
    used for diversity preservation in NSGA-II.

    Mathematical formulation:
    For each objective m and solution i:
    distance[i] += (f_m[i+1] - f_m[i-1]) / (f_m_max - f_m_min)

    Args:
        population: Full population of solutions
        indices: Indices of solutions to calculate distance for
    """
    if len(indices) <= 2:
        # Boundary solutions get infinite distance
        for idx in indices:
            population[idx].crowding_distance = float('inf')
        return

    # Reset distances
    for idx in indices:
        population[idx].crowding_distance = 0.0

    n_objectives = len(population[0].objectives)

    for obj_idx in range(n_objectives):
        # Sort by objective
        sorted_indices = sorted(
            indices,
            key=lambda i: population[i].objectives[obj_idx]
        )

        # Get objective range
        obj_min = population[sorted_indices[0]].objectives[obj_idx]
        obj_max = population[sorted_indices[-1]].objectives[obj_idx]

        # Boundary solutions get infinite distance
        population[sorted_indices[0]].crowding_distance = float('inf')
        population[sorted_indices[-1]].crowding_distance = float('inf')

        # Calculate distances for middle solutions
        if obj_max > obj_min:
            for i in range(1, len(sorted_indices) - 1):
                curr_idx = sorted_indices[i]
                next_idx = sorted_indices[i + 1]
                prev_idx = sorted_indices[i - 1]

                distance_contrib = (
                    population[next_idx].objectives[obj_idx] -
                    population[prev_idx].objectives[obj_idx]
                ) / (obj_max - obj_min)

                population[curr_idx].crowding_distance += distance_contrib


def hypervolume(points: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Calculate hypervolume indicator (2D and 3D).

    Hypervolume measures the volume of objective space dominated by a
    Pareto front with respect to a reference point.

    For 2D: Simple rectangle summation algorithm
    For 3D: Inclusion-exclusion principle
    For >3D: Returns approximation (full algorithm is NP-hard)

    Args:
        points: Pareto front points (n_points, n_objectives)
        reference_point: Reference point for hypervolume

    Returns:
        Hypervolume value
    """
    if len(points) == 0:
        return 0.0

    n_objectives = points.shape[1]

    # Filter points that dominate reference point
    valid_points = []
    for point in points:
        if np.all(point <= reference_point):
            valid_points.append(point)

    if not valid_points:
        return 0.0

    points = np.array(valid_points)

    if n_objectives == 2:
        return _hypervolume_2d(points, reference_point)
    elif n_objectives == 3:
        return _hypervolume_3d(points, reference_point)
    else:
        # For higher dimensions, use Monte Carlo approximation
        return _hypervolume_monte_carlo(points, reference_point)


def _hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    """Calculate exact 2D hypervolume."""
    # Sort by first objective
    points = points[points[:, 0].argsort()]

    volume = 0.0
    prev_x = 0.0

    for point in points:
        if point[0] > prev_x:
            volume += (point[0] - prev_x) * (ref[1] - point[1])
            prev_x = point[0]

    # Add last rectangle
    if points[-1, 0] < ref[0]:
        volume += (ref[0] - points[-1, 0]) * (ref[1] - points[-1, 1])

    return volume


def _hypervolume_3d(points: np.ndarray, ref: np.ndarray) -> float:
    """Calculate exact 3D hypervolume using inclusion-exclusion."""
    volume = 0.0

    # Single point contributions
    for p in points:
        vol = np.prod(ref - p)
        volume += vol

    # Pairwise intersections (subtract)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            intersection = np.maximum(points[i], points[j])
            if np.all(intersection < ref):
                vol = np.prod(ref - intersection)
                volume -= vol

    # Triple intersections (add back)
    if len(points) >= 3:
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                for k in range(j + 1, len(points)):
                    intersection = np.maximum(np.maximum(points[i], points[j]), points[k])
                    if np.all(intersection < ref):
                        vol = np.prod(ref - intersection)
                        volume += vol

    return volume


def _hypervolume_monte_carlo(points: np.ndarray, ref: np.ndarray, n_samples: int = 10000) -> float:
    """Monte Carlo approximation for high-dimensional hypervolume."""
    n_objectives = points.shape[1]

    # Generate random samples in the reference box
    samples = np.random.uniform(0, ref, size=(n_samples, n_objectives))

    # Count samples dominated by at least one point
    dominated_count = 0
    for sample in samples:
        for point in points:
            if np.all(point <= sample):
                dominated_count += 1
                break

    # Calculate volume
    reference_volume = np.prod(ref)
    return (dominated_count / n_samples) * reference_volume