"""
GRASP - Greedy Randomized Adaptive Search Procedure

References:
    - Feo, T. A., & Resende, M. G. (1995). Greedy randomized adaptive search procedures.
      Journal of Global Optimization, 6(2), 109-133.
    - Resende, M. G., & Ribeiro, C. C. (2016). Optimization by GRASP.
      Springer New York.
    - Festa, P., & Resende, M. G. (2011). GRASP: An annotated bibliography.
      In Essays and surveys in metaheuristics (pp. 325-367). Springer.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Librex.core.interfaces import StandardizedProblem

logger = logging.getLogger(__name__)


def grasp_optimize(
    problem: StandardizedProblem,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    GRASP - Greedy Randomized Adaptive Search Procedure

    GRASP is a multi-start metaheuristic consisting of two phases:
    1. Construction: Build a greedy randomized solution
    2. Local Search: Improve the constructed solution

    Args:
        problem: Standardized optimization problem
        config: Configuration dict with keys:
            - max_iterations: Maximum GRASP iterations (default: 100)
            - alpha: Greediness factor [0,1] (0=greedy, 1=random) (default: 0.2)
            - local_search_iterations: Local search iterations (default: 100)
            - rcl_size: Restricted Candidate List size (default: None, auto)
            - path_relinking: Enable path relinking (default: False)
            - elite_size: Size of elite solution set for path relinking (default: 10)
            - seed: Random seed (default: None)

    Returns:
        dict: Optimization result with solution, objective, and metadata
    """
    max_iterations = config.get('max_iterations', 100)
    alpha = config.get('alpha', 0.2)
    local_search_iterations = config.get('local_search_iterations', 100)
    rcl_size = config.get('rcl_size', None)
    path_relinking = config.get('path_relinking', False)
    elite_size = config.get('elite_size', 10)
    seed = config.get('seed', None)

    if seed is not None:
        np.random.seed(seed)

    n = problem.dimension

    best_solution = None
    best_objective = float('inf')

    elite_solutions = []  # For path relinking
    convergence_history = []
    total_iterations = 0

    for iteration in range(max_iterations):
        # Construction phase
        constructed = _greedy_randomized_construction(
            problem, alpha, rcl_size
        )

        # Local search phase
        improved, improved_obj, ls_iters = _local_search(
            constructed, problem, local_search_iterations
        )
        total_iterations += ls_iters

        # Path relinking (if enabled and elite set available)
        if path_relinking and elite_solutions:
            relinked, relinked_obj, pr_iters = _path_relinking(
                improved, improved_obj, elite_solutions, problem
            )
            if relinked_obj < improved_obj:
                improved = relinked
                improved_obj = relinked_obj
            total_iterations += pr_iters

        # Update best solution
        if improved_obj < best_objective:
            best_solution = improved.copy()
            best_objective = improved_obj

        # Update elite set
        if path_relinking:
            _update_elite_set(
                elite_solutions, improved, improved_obj, elite_size
            )

        convergence_history.append(best_objective)

        # Check convergence
        if len(convergence_history) > 20:
            recent = convergence_history[-20:]
            if max(recent) - min(recent) < 1e-6:
                logger.info(f"GRASP converged at iteration {iteration}")
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
            'method': 'grasp',
            'alpha': alpha,
            'path_relinking': path_relinking,
            'elite_solutions': len(elite_solutions) if path_relinking else 0,
            'seed': seed
        }
    }


def _greedy_randomized_construction(
    problem: StandardizedProblem,
    alpha: float,
    rcl_size: Optional[int]
) -> np.ndarray:
    """
    Construct a greedy randomized solution

    Uses Restricted Candidate List (RCL) approach
    """
    n = problem.dimension
    solution = []
    unassigned = list(range(n))

    # Build solution incrementally
    while unassigned:
        if len(solution) == 0:
            # First element: random choice
            selected = np.random.choice(unassigned)
        else:
            # Calculate greedy scores for all candidates
            scores = []
            for candidate in unassigned:
                # Estimate cost of adding candidate
                if problem.objective_function:
                    # Create partial solution for evaluation
                    partial = solution + [candidate]
                    remaining = [x for x in unassigned if x != candidate]
                    # Fill with remaining in order for evaluation
                    test_solution = np.array(partial + remaining)
                    score = problem.objective_function(test_solution)
                else:
                    # Use matrix-based heuristic if available
                    score = _greedy_score(solution, candidate, problem.objective_matrix)
                scores.append(score)

            scores = np.array(scores)

            # Build Restricted Candidate List (RCL)
            min_score = scores.min()
            max_score = scores.max()
            threshold = min_score + alpha * (max_score - min_score)

            # RCL contains all elements within threshold
            rcl_indices = np.where(scores <= threshold)[0]

            # Apply size limit if specified
            if rcl_size and len(rcl_indices) > rcl_size:
                # Keep best rcl_size candidates
                sorted_indices = np.argsort(scores)
                rcl_indices = sorted_indices[:rcl_size]

            # Randomly select from RCL
            selected_idx = np.random.choice(rcl_indices)
            selected = unassigned[selected_idx]

        solution.append(selected)
        unassigned.remove(selected)

    return np.array(solution)


def _greedy_score(
    partial_solution: List[int],
    candidate: int,
    matrix: Optional[np.ndarray]
) -> float:
    """Calculate greedy score for adding candidate to partial solution"""
    if matrix is None:
        return np.random.random()  # Random if no matrix available

    score = 0.0
    for assigned in partial_solution:
        score += matrix[assigned, candidate] + matrix[candidate, assigned]
    return score


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
                # Create neighbor
                neighbor = current.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

                # Evaluate
                if problem.objective_function:
                    obj = problem.objective_function(neighbor)
                else:
                    obj = _evaluate_permutation(neighbor, problem.objective_matrix)

                iterations += 1

                # Track best improvement
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


def _path_relinking(
    source: np.ndarray,
    source_obj: float,
    elite_solutions: List[Tuple[np.ndarray, float]],
    problem: StandardizedProblem
) -> Tuple[np.ndarray, float, int]:
    """
    Perform path relinking between source and elite solutions

    Returns:
        Tuple of (best_solution, best_objective, iterations_used)
    """
    best_solution = source
    best_objective = source_obj
    iterations = 0

    # Select target from elite (best in elite)
    if not elite_solutions:
        return best_solution, best_objective, iterations

    target, target_obj = min(elite_solutions, key=lambda x: x[1])

    # Generate path from source to target
    current = source.copy()

    while not np.array_equal(current, target):
        # Find differences
        differences = []
        for i in range(len(current)):
            if current[i] != target[i]:
                # Find where target[i] is in current
                j = np.where(current == target[i])[0][0]
                differences.append((i, j))

        if not differences:
            break

        # Try each move and select best
        best_move = None
        best_move_obj = float('inf')

        for i, j in differences:
            # Apply move
            neighbor = current.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

            # Evaluate
            if problem.objective_function:
                obj = problem.objective_function(neighbor)
            else:
                obj = _evaluate_permutation(neighbor, problem.objective_matrix)

            iterations += 1

            if obj < best_move_obj:
                best_move = neighbor
                best_move_obj = obj

        # Apply best move
        current = best_move

        # Update best if improved
        if best_move_obj < best_objective:
            best_solution = current.copy()
            best_objective = best_move_obj

    return best_solution, best_objective, iterations


def _update_elite_set(
    elite_solutions: List[Tuple[np.ndarray, float]],
    solution: np.ndarray,
    objective: float,
    max_size: int
) -> None:
    """Update elite solution set for path relinking"""
    # Check if solution is already in elite
    for elite_sol, _ in elite_solutions:
        if np.array_equal(solution, elite_sol):
            return

    # Add solution
    elite_solutions.append((solution.copy(), objective))

    # Sort by objective
    elite_solutions.sort(key=lambda x: x[1])

    # Keep only best solutions
    if len(elite_solutions) > max_size:
        elite_solutions.pop()


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


__all__ = ['grasp_optimize']