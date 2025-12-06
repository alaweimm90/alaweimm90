"""
Performance metrics for benchmarking

Provides utilities for calculating optimization performance metrics.
"""

from typing import List, Optional

import numpy as np


def compute_optimality_gap(
    objective: float,
    optimal: float,
) -> float:
    """
    Compute optimality gap as percentage

    Gap = (objective - optimal) / optimal * 100%

    Args:
        objective: Objective value obtained
        optimal: Known optimal objective value

    Returns:
        Optimality gap as decimal (e.g., 0.05 = 5% gap)

    Example:
        >>> gap = compute_optimality_gap(105.0, 100.0)
        >>> print(f"Gap: {gap:.2%}")  # Gap: 5.00%
    """
    if optimal == 0:
        # Avoid division by zero
        if abs(objective) < 1e-12:
            return 0.0
        else:
            return float('inf')

    gap = (objective - optimal) / abs(optimal)
    return gap


def compute_solution_quality(
    best_objective: float,
    all_objectives: List[float],
) -> float:
    """
    Compute solution quality score based on consistency

    Quality = 1 - (std_dev / mean) for multiple runs
    Higher score indicates more consistent performance

    Args:
        best_objective: Best objective value found
        all_objectives: All objective values from multiple runs

    Returns:
        Quality score in [0, 1], where 1 is perfect consistency

    Example:
        >>> objectives = [100.0, 102.0, 101.0, 99.0, 103.0]
        >>> quality = compute_solution_quality(99.0, objectives)
        >>> print(f"Quality: {quality:.3f}")
    """
    if len(all_objectives) <= 1:
        return 1.0  # Single run is perfectly consistent

    mean_obj = np.mean(all_objectives)
    std_obj = np.std(all_objectives)

    if abs(mean_obj) < 1e-12:
        # Near-zero mean, check if std is also near zero
        return 1.0 if std_obj < 1e-12 else 0.0

    # Coefficient of variation (lower is better)
    cv = std_obj / abs(mean_obj)

    # Convert to quality score (higher is better)
    # Use exponential decay: quality = exp(-cv)
    quality = np.exp(-cv)

    return float(quality)


def compute_convergence_speed(
    history: List[float],
    target_quality: float = 0.95,
) -> Optional[int]:
    """
    Compute convergence speed (iterations to reach target quality)

    Target quality = (optimal - current) / (optimal - initial) <= (1 - target_quality)

    Args:
        history: List of objective values over iterations
        target_quality: Target quality threshold (default 95% of improvement)

    Returns:
        Iteration where target quality was reached, or None if not reached

    Example:
        >>> history = [1000, 800, 600, 520, 510, 505, 502, 500]
        >>> iterations = compute_convergence_speed(history, target_quality=0.95)
        >>> print(f"Converged in {iterations} iterations")
    """
    if len(history) < 2:
        return None

    initial = history[0]
    final = history[-1]

    if abs(initial - final) < 1e-12:
        # No improvement
        return None

    improvement_needed = abs(initial - final)
    target_improvement = target_quality * improvement_needed
    target_value = initial - target_improvement if initial > final else initial + target_improvement

    # Find first iteration where target is reached
    for i, value in enumerate(history):
        current_improvement = abs(initial - value)
        if current_improvement >= target_improvement:
            return i

    return None  # Target not reached


def compute_performance_profile(
    results: List[float],
    tau_max: float = 10.0,
    num_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute performance profile for algorithm comparison

    Performance profile shows the probability that an algorithm's performance
    is within a factor tau of the best performance.

    Args:
        results: List of objective values (one per problem instance)
        tau_max: Maximum performance ratio to consider
        num_points: Number of points in the profile

    Returns:
        Tuple of (tau_values, probabilities)

    Reference:
        Dolan & Moré (2002), "Benchmarking optimization software with
        performance profiles", Mathematical Programming

    Example:
        >>> # Compare two methods on 10 problems
        >>> method1_results = [100, 95, 110, 88, ...]
        >>> method2_results = [102, 93, 115, 90, ...]
        >>>
        >>> tau1, prob1 = compute_performance_profile(method1_results)
        >>> tau2, prob2 = compute_performance_profile(method2_results)
        >>>
        >>> # Plot profiles for comparison
        >>> plt.plot(tau1, prob1, label='Method 1')
        >>> plt.plot(tau2, prob2, label='Method 2')
    """
    results_array = np.array(results)
    n_problems = len(results)

    # Compute performance ratios
    min_results = np.min(results_array)
    if min_results <= 0:
        # Shift to positive values
        results_array = results_array - min_results + 1.0
        min_results = 1.0

    ratios = results_array / min_results

    # Compute profile
    tau_values = np.linspace(1.0, tau_max, num_points)
    probabilities = np.zeros(num_points)

    for i, tau in enumerate(tau_values):
        # Fraction of problems solved within factor tau of best
        probabilities[i] = np.sum(ratios <= tau) / n_problems

    return tau_values, probabilities


__all__ = [
    'compute_optimality_gap',
    'compute_solution_quality',
    'compute_convergence_speed',
    'compute_performance_profile',
]
