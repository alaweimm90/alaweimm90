"""Tabu Search optimization method"""

import logging
from typing import Any, Dict, Set, Tuple

import numpy as np

from Librex.core.interfaces import StandardizedProblem

logger = logging.getLogger(__name__)


def tabu_search_optimize(
    problem: StandardizedProblem,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Tabu Search optimization

    Local search with memory to avoid revisiting recent solutions.

    Args:
        problem: Standardized optimization problem
        config: Configuration dict with keys:
            - iterations: Number of iterations (default: 5000)
            - tabu_tenure: Length of tabu list (default: 20)
            - seed: Random seed (default: None)

    Returns:
        dict: Optimization result
    """
    iterations = config.get('iterations', 5000)
    tabu_tenure = config.get('tabu_tenure', 20)
    seed = config.get('seed', None)

    if seed is not None:
        np.random.seed(seed)

    n = problem.dimension

    # Initialize
    current = np.arange(n)
    np.random.shuffle(current)
    current_obj = problem.objective_function(current)

    best = current.copy()
    best_obj = current_obj

    tabu_list: Set[Tuple[int, int]] = set()

    for iteration in range(iterations):
        # Find best non-tabu neighbor
        best_neighbor = None
        best_neighbor_obj = float('inf')
        best_move = None

        for i in range(n - 1):
            for j in range(i + 1, n):
                if (i, j) not in tabu_list:
                    neighbor = current.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    neighbor_obj = problem.objective_function(neighbor)

                    if neighbor_obj < best_neighbor_obj:
                        best_neighbor = neighbor
                        best_neighbor_obj = neighbor_obj
                        best_move = (i, j)

        if best_neighbor is None:
            # All moves are tabu, clear list
            tabu_list.clear()
            continue

        # Move to best neighbor
        current = best_neighbor
        current_obj = best_neighbor_obj

        # Update tabu list
        if best_move is not None:
            tabu_list.add(best_move)
            if len(tabu_list) > tabu_tenure:
                # Remove oldest entry (simplified)
                tabu_list.pop()

        # Update best
        if current_obj < best_obj:
            best = current.copy()
            best_obj = current_obj

    return {
        'solution': best,
        'objective': float(best_obj),
        'is_valid': True,
        'iterations': iterations,
        'convergence': {'converged': len(tabu_list) < tabu_tenure},
        'metadata': {
            'method': 'tabu_search',
            'tabu_tenure': tabu_tenure,
            'seed': seed
        }
    }


__all__ = ['tabu_search_optimize']
