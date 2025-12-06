"""Random Search baseline optimization method"""

import logging
from typing import Any, Dict

import numpy as np

from Librex.core.interfaces import StandardizedProblem

logger = logging.getLogger(__name__)


def random_search_optimize(
    problem: StandardizedProblem,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Random search optimization - baseline method

    Randomly samples solutions and returns the best found.

    Args:
        problem: Standardized optimization problem
        config: Configuration dict with keys:
            - iterations: Number of random samples (default: 1000)
            - seed: Random seed (default: None)

    Returns:
        dict: Result with solution, objective, iterations, etc.
    """
    iterations = config.get('iterations', 1000)
    seed = config.get('seed', None)

    if seed is not None:
        np.random.seed(seed)

    n = problem.dimension
    best_solution = np.arange(n)
    np.random.shuffle(best_solution)
    best_objective = problem.objective_function(best_solution)

    for i in range(iterations - 1):
        solution = np.arange(n)
        np.random.shuffle(solution)
        objective = problem.objective_function(solution)

        if objective < best_objective:
            best_solution = solution.copy()
            best_objective = objective

    return {
        'solution': best_solution,
        'objective': float(best_objective),
        'is_valid': True,
        'iterations': iterations,
        'convergence': {'converged': False, 'method': 'random_search'},
        'metadata': {'method': 'random_search', 'seed': seed}
    }


__all__ = ['random_search_optimize']
