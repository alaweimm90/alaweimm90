"""
Quality indicators for multi-objective optimization.

This module provides comprehensive quality indicators for evaluating
the performance of multi-objective optimization algorithms, including:
- Hypervolume (HV): Volume of dominated space
- Inverted Generational Distance (IGD): Distance to reference set
- Generational Distance (GD): Distance from reference set
- Spread: Distribution uniformity measure
- Epsilon Indicator: Minimum translation factor
- Coverage: Set coverage metric
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def calculate_hypervolume(
    pareto_front: np.ndarray,
    reference_point: np.ndarray
) -> float:
    """
    Calculate hypervolume indicator.

    Hypervolume measures the volume of objective space dominated by
    the Pareto front with respect to a reference point.

    Mathematical formulation:
    HV(S, r) = λ(∪_{s∈S} {x | s ≺ x ≺ r})
    where λ is the Lebesgue measure.

    Args:
        pareto_front: Array of objective vectors (n_solutions, n_objectives)
        reference_point: Reference point for hypervolume calculation

    Returns:
        Hypervolume value

    Note:
        For >3D problems, uses Monte Carlo approximation
    """
    from .core import hypervolume
    return hypervolume(pareto_front, reference_point)


def calculate_igd(
    pareto_front: np.ndarray,
    reference_set: np.ndarray,
    p: int = 2
) -> float:
    """
    Calculate Inverted Generational Distance (IGD).

    IGD measures the average distance from each point in the reference
    set to the nearest point in the obtained Pareto front. Lower values
    indicate better convergence and diversity.

    Mathematical formulation:
    IGD(S, R) = (1/|R|) * (Σ_{r∈R} min_{s∈S} d(r,s)^p)^(1/p)

    Args:
        pareto_front: Obtained Pareto front (n_solutions, n_objectives)
        reference_set: True/reference Pareto front (n_ref, n_objectives)
        p: Norm parameter (default: 2 for Euclidean)

    Returns:
        IGD value (lower is better)
    """
    if len(pareto_front) == 0 or len(reference_set) == 0:
        return float('inf')

    # Calculate minimum distances from reference to obtained front
    distances = cdist(reference_set, pareto_front, metric='euclidean')
    min_distances = np.min(distances, axis=1)

    # Calculate IGD
    igd = np.mean(min_distances ** p) ** (1.0 / p)
    return igd


def calculate_igd_plus(
    pareto_front: np.ndarray,
    reference_set: np.ndarray
) -> float:
    """
    Calculate IGD+ (Modified Inverted Generational Distance).

    IGD+ is a Pareto-compliant version of IGD that only considers
    dominated distances, preventing non-Pareto-optimal solutions
    from achieving good IGD+ values.

    Mathematical formulation:
    IGD+(S, R) = (1/|R|) * Σ_{r∈R} min_{s∈S} d+(r,s)
    where d+(r,s) = sqrt(Σ_i max(0, s_i - r_i)²)

    Args:
        pareto_front: Obtained Pareto front (n_solutions, n_objectives)
        reference_set: True/reference Pareto front (n_ref, n_objectives)

    Returns:
        IGD+ value (lower is better)
    """
    if len(pareto_front) == 0 or len(reference_set) == 0:
        return float('inf')

    igd_plus = 0.0
    for ref_point in reference_set:
        # Calculate dominated distance to each solution
        min_distance = float('inf')
        for solution in pareto_front:
            # Only consider dimensions where solution is worse
            diff = np.maximum(0, solution - ref_point)
            distance = np.linalg.norm(diff)
            min_distance = min(min_distance, distance)
        igd_plus += min_distance

    return igd_plus / len(reference_set)


def calculate_gd(
    pareto_front: np.ndarray,
    reference_set: np.ndarray,
    p: int = 2
) -> float:
    """
    Calculate Generational Distance (GD).

    GD measures the average distance from each point in the obtained
    Pareto front to the nearest point in the reference set.

    Mathematical formulation:
    GD(S, R) = (1/|S|) * (Σ_{s∈S} min_{r∈R} d(s,r)^p)^(1/p)

    Args:
        pareto_front: Obtained Pareto front (n_solutions, n_objectives)
        reference_set: True/reference Pareto front (n_ref, n_objectives)
        p: Norm parameter (default: 2 for Euclidean)

    Returns:
        GD value (lower is better)
    """
    if len(pareto_front) == 0 or len(reference_set) == 0:
        return float('inf')

    # Calculate minimum distances from obtained front to reference
    distances = cdist(pareto_front, reference_set, metric='euclidean')
    min_distances = np.min(distances, axis=1)

    # Calculate GD
    gd = np.mean(min_distances ** p) ** (1.0 / p)
    return gd


def calculate_spread(
    pareto_front: np.ndarray,
    reference_set: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Spread (Delta) indicator.

    Spread measures the extent of spread and distribution uniformity
    of the obtained Pareto front.

    Mathematical formulation:
    Δ = (d_f + d_l + Σ|d_i - d̄|) / (d_f + d_l + (N-1)d̄)
    where:
    - d_f, d_l: distances to extreme solutions
    - d_i: consecutive distances between solutions
    - d̄: mean of d_i
    - N: number of solutions

    Args:
        pareto_front: Obtained Pareto front (n_solutions, n_objectives)
        reference_set: Optional reference set for extreme points

    Returns:
        Spread value (lower is better, 0 = perfect distribution)
    """
    if len(pareto_front) < 3:
        return float('inf')

    n_obj = pareto_front.shape[1]

    # Find extreme points
    if reference_set is not None and len(reference_set) > 0:
        # Use reference set extremes
        extreme_points = []
        for i in range(n_obj):
            min_idx = np.argmin(reference_set[:, i])
            max_idx = np.argmax(reference_set[:, i])
            extreme_points.append(reference_set[min_idx])
            extreme_points.append(reference_set[max_idx])
    else:
        # Use pareto front extremes
        extreme_points = []
        for i in range(n_obj):
            min_idx = np.argmin(pareto_front[:, i])
            max_idx = np.argmax(pareto_front[:, i])
            extreme_points.append(pareto_front[min_idx])
            extreme_points.append(pareto_front[max_idx])

    # Sort solutions by first objective for consecutive distance calculation
    sorted_front = pareto_front[pareto_front[:, 0].argsort()]

    # Calculate consecutive distances
    consecutive_distances = []
    for i in range(len(sorted_front) - 1):
        dist = np.linalg.norm(sorted_front[i] - sorted_front[i + 1])
        consecutive_distances.append(dist)

    if not consecutive_distances:
        return float('inf')

    d_mean = np.mean(consecutive_distances)

    # Calculate distances to extreme points
    d_f = np.linalg.norm(sorted_front[0] - extreme_points[0])
    d_l = np.linalg.norm(sorted_front[-1] - extreme_points[-1])

    # Calculate spread
    numerator = d_f + d_l + sum(abs(d - d_mean) for d in consecutive_distances)
    denominator = d_f + d_l + (len(sorted_front) - 1) * d_mean

    if denominator < 1e-10:
        return float('inf')

    spread = numerator / denominator
    return spread


def calculate_epsilon_indicator(
    pareto_front: np.ndarray,
    reference_set: np.ndarray,
    additive: bool = False
) -> float:
    """
    Calculate Epsilon indicator.

    Epsilon indicator measures the minimum factor by which the obtained
    Pareto front needs to be translated/scaled to dominate the reference set.

    Mathematical formulation:
    - Multiplicative: ε = max_{r∈R} min_{s∈S} max_i (r_i/s_i)
    - Additive: ε = max_{r∈R} min_{s∈S} max_i (r_i - s_i)

    Args:
        pareto_front: Obtained Pareto front (n_solutions, n_objectives)
        reference_set: True/reference Pareto front (n_ref, n_objectives)
        additive: If True, use additive epsilon; otherwise multiplicative

    Returns:
        Epsilon value (lower is better)
    """
    if len(pareto_front) == 0 or len(reference_set) == 0:
        return float('inf')

    epsilon = -float('inf')

    for ref_point in reference_set:
        min_epsilon = float('inf')

        for solution in pareto_front:
            if additive:
                # Additive epsilon: max difference
                max_diff = np.max(ref_point - solution)
                min_epsilon = min(min_epsilon, max_diff)
            else:
                # Multiplicative epsilon: max ratio
                # Avoid division by zero
                with np.errstate(divide='ignore'):
                    ratios = np.where(
                        solution > 1e-10,
                        ref_point / solution,
                        float('inf')
                    )
                max_ratio = np.max(ratios)
                min_epsilon = min(min_epsilon, max_ratio)

        epsilon = max(epsilon, min_epsilon)

    return epsilon


def calculate_coverage(
    pareto_front_a: np.ndarray,
    pareto_front_b: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate set coverage (C-metric).

    Coverage C(A,B) measures the proportion of solutions in B that
    are dominated by at least one solution in A.

    Mathematical formulation:
    C(A,B) = |{b ∈ B | ∃a ∈ A : a ≺ b}| / |B|

    Args:
        pareto_front_a: First Pareto front (n_a, n_objectives)
        pareto_front_b: Second Pareto front (n_b, n_objectives)

    Returns:
        Tuple of (C(A,B), C(B,A)) - proportion of dominated solutions
    """
    from .core import dominates

    def coverage_metric(front1: np.ndarray, front2: np.ndarray) -> float:
        if len(front2) == 0:
            return 0.0

        dominated_count = 0
        for sol2 in front2:
            is_dominated = False
            for sol1 in front1:
                if dominates(sol1, sol2):
                    is_dominated = True
                    break
            if is_dominated:
                dominated_count += 1

        return dominated_count / len(front2)

    c_ab = coverage_metric(pareto_front_a, pareto_front_b)
    c_ba = coverage_metric(pareto_front_b, pareto_front_a)

    return c_ab, c_ba


def calculate_spacing(pareto_front: np.ndarray) -> float:
    """
    Calculate Spacing indicator.

    Spacing measures the standard deviation of distances between
    consecutive solutions, indicating distribution uniformity.

    Mathematical formulation:
    S = sqrt((1/(n-1)) * Σ(d̄ - d_i)²)
    where d_i = min_{j≠i} Σ_m |f_m^i - f_m^j|

    Args:
        pareto_front: Pareto front (n_solutions, n_objectives)

    Returns:
        Spacing value (lower is better, 0 = perfect uniformity)
    """
    n = len(pareto_front)
    if n < 2:
        return 0.0

    # Calculate minimum distances for each solution
    distances = []
    for i in range(n):
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                # Manhattan distance in objective space
                dist = np.sum(np.abs(pareto_front[i] - pareto_front[j]))
                min_dist = min(min_dist, dist)
        distances.append(min_dist)

    # Calculate spacing
    d_mean = np.mean(distances)
    spacing = np.sqrt(np.sum((distances - d_mean) ** 2) / (n - 1))

    return spacing


def calculate_maximum_spread(pareto_front: np.ndarray) -> float:
    """
    Calculate Maximum Spread indicator.

    Maximum spread measures the maximum extension of the Pareto front
    in each objective dimension.

    Mathematical formulation:
    MS = sqrt(Σ_m (max_i f_m^i - min_i f_m^i)²)

    Args:
        pareto_front: Pareto front (n_solutions, n_objectives)

    Returns:
        Maximum spread value (higher is better)
    """
    if len(pareto_front) == 0:
        return 0.0

    # Calculate range in each objective
    ranges = np.max(pareto_front, axis=0) - np.min(pareto_front, axis=0)

    # Calculate maximum spread
    max_spread = np.linalg.norm(ranges)

    return max_spread


def calculate_purity(
    pareto_front: np.ndarray,
    reference_set: np.ndarray,
    threshold: float = 0.01
) -> float:
    """
    Calculate Purity indicator.

    Purity measures the proportion of obtained solutions that are
    close to the true Pareto front (within a threshold distance).

    Args:
        pareto_front: Obtained Pareto front (n_solutions, n_objectives)
        reference_set: True Pareto front (n_ref, n_objectives)
        threshold: Distance threshold for considering a solution "pure"

    Returns:
        Purity value [0, 1] (higher is better)
    """
    if len(pareto_front) == 0 or len(reference_set) == 0:
        return 0.0

    pure_count = 0
    for solution in pareto_front:
        min_dist = np.min(
            np.linalg.norm(reference_set - solution, axis=1)
        )
        if min_dist <= threshold:
            pure_count += 1

    purity = pure_count / len(pareto_front)
    return purity


class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator for multi-objective optimization.

    This class provides a unified interface for calculating multiple
    quality indicators simultaneously.
    """

    def __init__(
        self,
        reference_set: Optional[np.ndarray] = None,
        reference_point: Optional[np.ndarray] = None
    ):
        """
        Initialize performance metrics calculator.

        Args:
            reference_set: True/reference Pareto front
            reference_point: Reference point for hypervolume
        """
        self.reference_set = reference_set
        self.reference_point = reference_point

    def evaluate(self, pareto_front: np.ndarray) -> dict:
        """
        Calculate all available metrics for a Pareto front.

        Args:
            pareto_front: Obtained Pareto front

        Returns:
            Dictionary of metric names and values
        """
        metrics = {}

        # Basic metrics
        metrics["n_solutions"] = len(pareto_front)

        if len(pareto_front) > 0:
            metrics["spacing"] = calculate_spacing(pareto_front)
            metrics["maximum_spread"] = calculate_maximum_spread(pareto_front)

            # Metrics requiring reference point
            if self.reference_point is not None:
                metrics["hypervolume"] = calculate_hypervolume(
                    pareto_front,
                    self.reference_point
                )

            # Metrics requiring reference set
            if self.reference_set is not None and len(self.reference_set) > 0:
                metrics["igd"] = calculate_igd(pareto_front, self.reference_set)
                metrics["igd_plus"] = calculate_igd_plus(pareto_front, self.reference_set)
                metrics["gd"] = calculate_gd(pareto_front, self.reference_set)
                metrics["spread"] = calculate_spread(pareto_front, self.reference_set)
                metrics["epsilon"] = calculate_epsilon_indicator(
                    pareto_front,
                    self.reference_set
                )
                metrics["purity"] = calculate_purity(pareto_front, self.reference_set)

        return metrics