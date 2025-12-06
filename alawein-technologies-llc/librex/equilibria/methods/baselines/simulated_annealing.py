"""Simulated Annealing optimization method"""

import logging
import math
from typing import Any, Dict

import numpy as np

from Librex.core.interfaces import StandardizedProblem

logger = logging.getLogger(__name__)


def simulated_annealing_optimize(
    problem: StandardizedProblem,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simulated Annealing optimization

    Probabilistic method that accepts worse solutions with decreasing probability.

    Args:
        problem: Standardized optimization problem
        config: Configuration dict with keys:
            - iterations: Number of iterations (default: 10000)
            - initial_temp: Starting temperature (default: 100.0)
            - cooling_rate: Temperature decay rate (default: 0.95)
            - seed: Random seed (default: None)

    Returns:
        dict: Optimization result
    """
    iterations = config.get('iterations', 10000)
    temp = config.get('initial_temp', 100.0)
    cooling_rate = config.get('cooling_rate', 0.95)
    seed = config.get('seed', None)

    if seed is not None:
        np.random.seed(seed)

    n = problem.dimension
    current = np.arange(n)
    np.random.shuffle(current)
    current_obj = problem.objective_function(current)

    best = current.copy()
    best_obj = current_obj

    for i in range(iterations):
        # Generate neighbor by swapping two random positions
        neighbor = current.copy()
        idx1, idx2 = np.random.choice(n, 2, replace=False)
        neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]

        neighbor_obj = problem.objective_function(neighbor)

        # Accept or reject
        delta = neighbor_obj - current_obj
        if delta < 0 or np.random.random() < math.exp(-delta / temp):
            current = neighbor
            current_obj = neighbor_obj

            if current_obj < best_obj:
                best = current.copy()
                best_obj = current_obj

        # Cool down
        temp *= cooling_rate

    return {
        'solution': best,
        'objective': float(best_obj),
        'is_valid': True,
        'iterations': iterations,
        'convergence': {'converged': temp < 0.01, 'final_temp': temp},
        'metadata': {'method': 'simulated_annealing', 'seed': seed}
    }


__all__ = ['simulated_annealing_optimize']
