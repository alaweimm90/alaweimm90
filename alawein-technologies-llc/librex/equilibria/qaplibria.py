"""
QAP Convenience API - Simplified interface for QAP-specific workflows

This module provides a thin facade over the universal optimize() function
specifically tailored for Quadratic Assignment Problems.
"""

from typing import Any, Dict, Optional

import numpy as np

from Librex.adapters.qap import QAPAdapter
from Librex.optimize import optimize as universal_optimize


def optimize_qap(
    flow_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    method: str = 'simulated_annealing',
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Optimize a Quadratic Assignment Problem

    Convenience function that wraps the universal optimize() interface
    specifically for QAP problems.

    Args:
        flow_matrix: Flow matrix (n x n numpy array)
        distance_matrix: Distance matrix (n x n numpy array)
        method: Optimization method (default: 'simulated_annealing')
            Options: random_search, simulated_annealing, local_search,
                     genetic_algorithm, tabu_search
        config: Method-specific configuration

    Returns:
        dict: Optimization result with keys:
            - solution: Best permutation found (numpy array)
            - objective: QAP objective value
            - is_valid: Whether solution is valid permutation
            - iterations: Iteration count
            - convergence: Convergence info

    Example:
        >>> import numpy as np
        >>> from Librex.Librex.QAP import optimize_qap
        >>>
        >>> flow = np.array([[0, 5, 2], [5, 0, 3], [2, 3, 0]])
        >>> distance = np.array([[0, 8, 15], [8, 0, 13], [15, 13, 0]])
        >>>
        >>> result = optimize_qap(
        ...     flow, distance,
        ...     method="simulated_annealing",
        ...     config={"iterations": 1000, "seed": 42}
        ... )
        >>>
        >>> print(result["solution"], result["objective"], result["is_valid"])
    """
    # Create QAP problem dict
    problem = {
        'flow_matrix': flow_matrix,
        'distance_matrix': distance_matrix
    }

    # Create adapter
    adapter = QAPAdapter()

    # Call universal optimize function
    result = universal_optimize(problem, adapter, method, config)

    return result


__all__ = ['optimize_qap']
