"""
Variable Neighborhood Search (VNS) for combinatorial optimization

References:
    - Hansen, P., & Mladenović, N. (2001). Variable neighborhood search:
      Principles and applications. European Journal of Operational Research, 130(3), 449-467.
    - Hansen, P., Mladenović, N., & Moreno Pérez, J. A. (2010).
      Variable neighbourhood search: methods and applications. Annals of Operations Research, 175(1), 367-407.
"""

import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from Librex.core.interfaces import StandardizedProblem

logger = logging.getLogger(__name__)


def variable_neighborhood_search_optimize(
    problem: StandardizedProblem,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Variable Neighborhood Search metaheuristic

    VNS systematically explores different neighborhood structures to escape
    local optima. It combines shaking (perturbation) with local search.

    Args:
        problem: Standardized optimization problem
        config: Configuration dict with keys:
            - max_iterations: Maximum iterations (default: 1000)
            - k_max: Maximum neighborhood size (default: 5)
            - local_search_iterations: Local search iterations (default: 100)
            - time_limit: Time limit in seconds (default: None)
            - seed: Random seed (default: None)

    Returns:
        dict: Optimization result with solution, objective, and metadata
    """
    max_iterations = config.get('max_iterations', 1000)
    k_max = config.get('k_max', 5)
    local_search_iterations = config.get('local_search_iterations', 100)
    seed = config.get('seed', None)

    if seed is not None:
        np.random.seed(seed)

    n = problem.dimension

    # Define neighborhood structures
    neighborhoods = _define_neighborhoods(n, k_max)

    # Initialize with random solution
    current = np.random.permutation(n)

    # Evaluate initial solution
    if problem.objective_function:
        current_obj = problem.objective_function(current)
    else:
        current_obj = _evaluate_permutation(current, problem.objective_matrix)

    best_solution = current.copy()
    best_objective = current_obj

    convergence_history = []
    total_iterations = 0

    for iteration in range(max_iterations):
        k = 0  # Neighborhood index

        while k < k_max:
            # Shaking: generate neighbor in k-th neighborhood
            neighbor = _shake(current, neighborhoods[k])

            # Local search
            improved_neighbor, improved_obj, ls_iters = _local_search(
                neighbor, problem, local_search_iterations
            )
            total_iterations += ls_iters

            # Move or not
            if improved_obj < current_obj:
                current = improved_neighbor
                current_obj = improved_obj

                # Update best
                if current_obj < best_objective:
                    best_solution = current.copy()
                    best_objective = current_obj

                k = 0  # Reset to first neighborhood
            else:
                k += 1  # Move to next neighborhood

        convergence_history.append(best_objective)

        # Check convergence
        if len(convergence_history) > 20:
            recent = convergence_history[-20:]
            if max(recent) - min(recent) < 1e-6:
                logger.info(f"VNS converged at iteration {iteration}")
                break

    return {
        'solution': best_solution,
        'objective': float(best_objective),
        'is_valid': True,
        'iterations': total_iterations,
        'convergence': {
            'converged': len(convergence_history) > 20 and
                        max(convergence_history[-20:]) - min(convergence_history[-20:]) < 1e-6,
            'history': convergence_history,
            'final_iteration': iteration + 1,
            'neighborhoods_explored': k_max
        },
        'metadata': {
            'method': 'variable_neighborhood_search',
            'k_max': k_max,
            'local_search_iterations': local_search_iterations,
            'seed': seed
        }
    }


def _define_neighborhoods(n: int, k_max: int) -> List[Callable]:
    """Define neighborhood structures of increasing size"""
    neighborhoods = []

    for k in range(1, k_max + 1):
        # Each neighborhood performs k random swaps/moves
        def make_neighborhood(num_moves):
            def neighborhood(solution):
                result = solution.copy()
                for _ in range(num_moves):
                    move_type = np.random.choice(['swap', 'insert', 'reverse'])
                    if move_type == 'swap':
                        i, j = np.random.choice(n, 2, replace=False)
                        result[i], result[j] = result[j], result[i]
                    elif move_type == 'insert':
                        i, j = np.random.choice(n, 2, replace=False)
                        if i < j:
                            result = np.concatenate([
                                result[:i],
                                result[i+1:j+1],
                                [result[i]],
                                result[j+1:]
                            ])
                        else:
                            result = np.concatenate([
                                result[:j],
                                [result[i]],
                                result[j:i],
                                result[i+1:]
                            ])
                    else:  # reverse
                        i, j = sorted(np.random.choice(n, 2, replace=False))
                        result[i:j+1] = result[i:j+1][::-1]
                return result
            return neighborhood

        neighborhoods.append(make_neighborhood(k))

    return neighborhoods


def _shake(solution: np.ndarray, neighborhood: Callable) -> np.ndarray:
    """Apply shaking in given neighborhood"""
    return neighborhood(solution)


def _local_search(
    solution: np.ndarray,
    problem: StandardizedProblem,
    max_iterations: int
) -> Tuple[np.ndarray, float, int]:
    """
    Perform local search using best improvement strategy

    Returns:
        Tuple of (best_solution, best_objective, iterations_used)
    """
    current = solution.copy()

    # Evaluate current
    if problem.objective_function:
        current_obj = problem.objective_function(current)
    else:
        current_obj = _evaluate_permutation(current, problem.objective_matrix)

    n = len(current)
    iterations = 0
    improved = True

    while improved and iterations < max_iterations:
        improved = False
        best_neighbor = None
        best_neighbor_obj = current_obj

        # Explore 2-opt neighborhood
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Create neighbor by swapping positions i and j
                neighbor = current.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

                # Evaluate
                if problem.objective_function:
                    obj = problem.objective_function(neighbor)
                else:
                    obj = _evaluate_permutation(neighbor, problem.objective_matrix)

                iterations += 1

                # Check if better
                if obj < best_neighbor_obj:
                    best_neighbor = neighbor.copy()
                    best_neighbor_obj = obj
                    improved = True

                if iterations >= max_iterations:
                    break
            if iterations >= max_iterations:
                break

        if improved:
            current = best_neighbor
            current_obj = best_neighbor_obj

    return current, current_obj, iterations


def _evaluate_permutation(permutation: np.ndarray, matrix: np.ndarray) -> float:
    """Evaluate permutation on matrix (for QAP-like problems)"""
    if matrix is None:
        return 0.0

    n = len(permutation)
    cost = 0.0
    for i in range(n):
        for j in range(n):
            cost += matrix[i, j] * matrix[permutation[i], permutation[j]]
    return cost


__all__ = ['variable_neighborhood_search_optimize']