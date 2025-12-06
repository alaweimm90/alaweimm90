"""Local Search (Hill Climbing) optimization method"""

import logging
from typing import Any, Dict

import numpy as np

from Librex.core.interfaces import StandardizedProblem

logger = logging.getLogger(__name__)


def local_search_optimize(
    problem: StandardizedProblem,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Local Search (Hill Climbing) optimization

    Iteratively improves solution by accepting only better neighbors.

    Args:
        problem: Standardized optimization problem
        config: Configuration dict with keys:
            - max_iterations: Maximum iterations (default: 5000)
            - restarts: Number of random restarts (default: 10)
            - seed: Random seed (default: None)

    Returns:
        dict: Optimization result
    """
    max_iterations = config.get('max_iterations', 5000)
    restarts = config.get('restarts', 10)
    seed = config.get('seed', None)

    if seed is not None:
        np.random.seed(seed)

    n = problem.dimension
    best_overall = None
    best_overall_obj = float('inf')
    total_iterations = 0

    for restart in range(restarts):
        # Random initialization
        current = np.arange(n)
        np.random.shuffle(current)
        current_obj = problem.objective_function(current)

        improved = True
        iterations = 0

        while improved and iterations < max_iterations // restarts:
            improved = False
            iterations += 1
            total_iterations += 1

            # Try all 2-opt swaps
            for i in range(n - 1):
                for j in range(i + 1, n):
                    neighbor = current.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    neighbor_obj = problem.objective_function(neighbor)

                    if neighbor_obj < current_obj:
                        current = neighbor
                        current_obj = neighbor_obj
                        improved = True
                        break
                if improved:
                    break

        if current_obj < best_overall_obj:
            best_overall = current
            best_overall_obj = current_obj

    return {
        'solution': best_overall,
        'objective': float(best_overall_obj),
        'is_valid': True,
        'iterations': total_iterations,
        'convergence': {'converged': True, 'restarts': restarts},
        'metadata': {'method': 'local_search', 'seed': seed}
    }


__all__ = ['local_search_optimize']
