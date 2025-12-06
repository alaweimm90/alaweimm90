"""
Iterated Local Search (ILS) for combinatorial optimization

References:
    - Lourenço, H. R., Martin, O. C., & Stützle, T. (2003). Iterated local search.
      In Handbook of metaheuristics (pp. 320-353). Springer.
    - Stützle, T. (2006). Iterated local search for the quadratic assignment problem.
      European Journal of Operational Research, 174(3), 1519-1539.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from Librex.core.interfaces import StandardizedProblem

logger = logging.getLogger(__name__)


def iterated_local_search_optimize(
    problem: StandardizedProblem,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Iterated Local Search metaheuristic

    ILS iteratively applies local search, perturbation, and acceptance criteria
    to escape local optima and explore the solution space efficiently.

    Algorithm:
    1. Generate initial solution
    2. Apply local search
    3. Repeat:
       - Perturbation
       - Local search
       - Acceptance criterion

    Args:
        problem: Standardized optimization problem
        config: Configuration dict with keys:
            - max_iterations: Maximum iterations (default: 100)
            - local_search_iterations: Local search iterations (default: 100)
            - perturbation_strength: Perturbation strength (1-5) (default: 3)
            - acceptance: Acceptance criterion ('better', 'threshold', 'simulated_annealing') (default: 'better')
            - threshold: Acceptance threshold for 'threshold' criterion (default: 0.01)
            - temperature: Initial temperature for SA acceptance (default: 100.0)
            - cooling_rate: Cooling rate for SA acceptance (default: 0.95)
            - seed: Random seed (default: None)

    Returns:
        dict: Optimization result with solution, objective, and metadata
    """
    max_iterations = config.get('max_iterations', 100)
    local_search_iterations = config.get('local_search_iterations', 100)
    perturbation_strength = config.get('perturbation_strength', 3)
    acceptance = config.get('acceptance', 'better')
    threshold = config.get('threshold', 0.01)
    temperature = config.get('temperature', 100.0)
    cooling_rate = config.get('cooling_rate', 0.95)
    seed = config.get('seed', None)

    if seed is not None:
        np.random.seed(seed)

    n = problem.dimension

    # Generate initial solution
    current = np.random.permutation(n)

    # Apply local search to initial solution
    current, current_obj, ls_iters = _local_search(
        current, problem, local_search_iterations
    )

    best_solution = current.copy()
    best_objective = current_obj

    convergence_history = []
    total_iterations = ls_iters
    no_improve_count = 0

    for iteration in range(max_iterations):
        # Perturbation
        perturbed = _perturbation(current, perturbation_strength)

        # Local search
        candidate, candidate_obj, ls_iters = _local_search(
            perturbed, problem, local_search_iterations
        )
        total_iterations += ls_iters

        # Acceptance criterion
        accept = _accept_solution(
            candidate_obj, current_obj,
            acceptance, threshold, temperature, iteration
        )

        if accept:
            current = candidate
            current_obj = candidate_obj

            # Update best
            if current_obj < best_objective:
                best_solution = current.copy()
                best_objective = current_obj
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        # Adaptive perturbation strength
        if no_improve_count > 10:
            perturbation_strength = min(perturbation_strength + 1, n // 2)
            no_improve_count = 0
        elif no_improve_count == 0:
            perturbation_strength = max(perturbation_strength - 1, 2)

        # Update temperature for SA acceptance
        if acceptance == 'simulated_annealing':
            temperature *= cooling_rate

        convergence_history.append(best_objective)

        # Check convergence
        if len(convergence_history) > 20:
            recent = convergence_history[-20:]
            if max(recent) - min(recent) < 1e-6:
                logger.info(f"ILS converged at iteration {iteration}")
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
            'final_iteration': iteration + 1
        },
        'metadata': {
            'method': 'iterated_local_search',
            'perturbation_strength_final': perturbation_strength,
            'acceptance_criterion': acceptance,
            'local_search_iterations': local_search_iterations,
            'seed': seed
        }
    }


def _local_search(
    solution: np.ndarray,
    problem: StandardizedProblem,
    max_iterations: int
) -> Tuple[np.ndarray, float, int]:
    """
    Perform local search using first improvement strategy

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

        # Random order for exploring neighborhood
        indices = list(range(n))
        np.random.shuffle(indices)

        for i in indices[:n-1]:
            for j in indices[i+1:]:
                # Create neighbor by swapping
                neighbor = current.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

                # Evaluate
                if problem.objective_function:
                    obj = problem.objective_function(neighbor)
                else:
                    obj = _evaluate_permutation(neighbor, problem.objective_matrix)

                iterations += 1

                # First improvement
                if obj < current_obj:
                    current = neighbor
                    current_obj = obj
                    improved = True
                    break

                if iterations >= max_iterations:
                    break

            if improved or iterations >= max_iterations:
                break

    return current, current_obj, iterations


def _perturbation(solution: np.ndarray, strength: int) -> np.ndarray:
    """
    Apply perturbation to escape local optima

    Strength determines number of random moves applied
    """
    perturbed = solution.copy()
    n = len(solution)

    for _ in range(strength):
        move_type = np.random.choice(['double_bridge', 'random_swaps', 'segment_reversal'])

        if move_type == 'double_bridge':
            # 4-opt style perturbation
            positions = sorted(np.random.choice(n, 4, replace=False))
            i, j, k, l = positions

            # Reconnect segments in different order
            perturbed = np.concatenate([
                perturbed[:i],
                perturbed[k:l],
                perturbed[j:k],
                perturbed[i:j],
                perturbed[l:]
            ])

        elif move_type == 'random_swaps':
            # Multiple random swaps
            for _ in range(strength):
                i, j = np.random.choice(n, 2, replace=False)
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

        else:  # segment_reversal
            # Reverse a random segment
            i, j = sorted(np.random.choice(n, 2, replace=False))
            perturbed[i:j+1] = perturbed[i:j+1][::-1]

    return perturbed


def _accept_solution(
    candidate_obj: float,
    current_obj: float,
    criterion: str,
    threshold: float,
    temperature: float,
    iteration: int
) -> bool:
    """
    Determine whether to accept candidate solution

    Criteria:
    - 'better': Accept only if better
    - 'threshold': Accept if within threshold
    - 'simulated_annealing': Accept with SA probability
    """
    if criterion == 'better':
        return candidate_obj < current_obj

    elif criterion == 'threshold':
        return candidate_obj < current_obj * (1 + threshold)

    elif criterion == 'simulated_annealing':
        if candidate_obj < current_obj:
            return True
        else:
            # SA acceptance probability
            delta = candidate_obj - current_obj
            prob = np.exp(-delta / temperature) if temperature > 0 else 0
            return np.random.random() < prob

    else:
        # Default to better only
        return candidate_obj < current_obj


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


__all__ = ['iterated_local_search_optimize']