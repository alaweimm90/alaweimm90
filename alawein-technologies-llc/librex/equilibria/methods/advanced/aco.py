"""
Ant Colony Optimization (ACO) for combinatorial optimization problems

References:
    - Dorigo, M., & Stützle, T. (2004). Ant colony optimization. MIT Press.
    - Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: optimization
      by a colony of cooperating agents. IEEE Transactions on Systems, Man,
      and Cybernetics, Part B, 26(1), 29-41.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from Librex.core.interfaces import StandardizedProblem

logger = logging.getLogger(__name__)


def ant_colony_optimize(
    problem: StandardizedProblem,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Ant Colony Optimization metaheuristic

    ACO uses artificial ants to explore solution space, depositing pheromones
    to guide future exploration. Particularly effective for graph-based problems
    like TSP and QAP.

    Args:
        problem: Standardized optimization problem
        config: Configuration dict with keys:
            - n_ants: Number of ants (default: 20)
            - n_iterations: Number of iterations (default: 100)
            - alpha: Pheromone importance (default: 1.0)
            - beta: Heuristic importance (default: 2.0)
            - rho: Pheromone evaporation rate (default: 0.1)
            - q0: Exploration vs exploitation probability (default: 0.9)
            - tau_min: Minimum pheromone level (default: 0.01)
            - tau_max: Maximum pheromone level (default: 10.0)
            - seed: Random seed (default: None)

    Returns:
        dict: Optimization result with solution, objective, and metadata
    """
    n_ants = config.get('n_ants', 20)
    n_iterations = config.get('n_iterations', 100)
    alpha = config.get('alpha', 1.0)
    beta = config.get('beta', 2.0)
    rho = config.get('rho', 0.1)
    q0 = config.get('q0', 0.9)
    tau_min = config.get('tau_min', 0.01)
    tau_max = config.get('tau_max', 10.0)
    seed = config.get('seed', None)

    if seed is not None:
        np.random.seed(seed)

    n = problem.dimension

    # Initialize pheromone matrix
    tau = np.ones((n, n)) * tau_max

    # Initialize heuristic information (inverse of problem matrix if available)
    if problem.objective_matrix is not None:
        # For QAP-like problems, use inverse of matrix as heuristic
        eta = 1.0 / (problem.objective_matrix + 1e-10)
    else:
        # Use uniform heuristic if no matrix provided
        eta = np.ones((n, n))

    best_solution = None
    best_objective = float('inf')
    convergence_history = []

    for iteration in range(n_iterations):
        solutions = []
        objectives = []

        # Construct solutions for all ants
        for ant in range(n_ants):
            solution = _construct_solution_aco(
                n, tau, eta, alpha, beta, q0
            )

            # Evaluate solution
            if problem.objective_function:
                obj = problem.objective_function(solution)
            else:
                obj = _evaluate_permutation_matrix(solution, problem.objective_matrix)

            solutions.append(solution)
            objectives.append(obj)

            # Update best solution
            if obj < best_objective:
                best_solution = solution.copy()
                best_objective = obj

        # Update pheromones
        tau = _update_pheromones(
            tau, solutions, objectives, best_solution, best_objective,
            rho, tau_min, tau_max, n
        )

        convergence_history.append(best_objective)

        # Check convergence
        if len(convergence_history) > 20:
            recent = convergence_history[-20:]
            if max(recent) - min(recent) < 1e-6:
                logger.info(f"ACO converged at iteration {iteration}")
                break

    return {
        'solution': best_solution,
        'objective': float(best_objective),
        'is_valid': True,
        'iterations': (iteration + 1) * n_ants,
        'convergence': {
            'converged': len(convergence_history) > 20 and
                        max(convergence_history[-20:]) - min(convergence_history[-20:]) < 1e-6,
            'history': convergence_history,
            'final_iteration': iteration + 1
        },
        'metadata': {
            'method': 'ant_colony_optimization',
            'n_ants': n_ants,
            'alpha': alpha,
            'beta': beta,
            'rho': rho,
            'seed': seed
        }
    }


def _construct_solution_aco(
    n: int,
    tau: np.ndarray,
    eta: np.ndarray,
    alpha: float,
    beta: float,
    q0: float
) -> np.ndarray:
    """Construct a solution using ACO probabilistic rules"""
    solution = []
    unvisited = list(range(n))

    # Start from random position
    current = np.random.choice(unvisited)
    solution.append(current)
    unvisited.remove(current)

    while unvisited:
        # Calculate probabilities for next position
        probabilities = []
        for next_pos in unvisited:
            prob = (tau[current, next_pos] ** alpha) * (eta[current, next_pos] ** beta)
            probabilities.append(prob)

        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()

        # Select next position
        if np.random.random() < q0:
            # Exploitation: choose best
            next_idx = np.argmax(probabilities)
        else:
            # Exploration: probabilistic choice
            next_idx = np.random.choice(len(unvisited), p=probabilities)

        next_pos = unvisited[next_idx]
        solution.append(next_pos)
        unvisited.remove(next_pos)
        current = next_pos

    return np.array(solution)


def _update_pheromones(
    tau: np.ndarray,
    solutions: List[np.ndarray],
    objectives: List[float],
    best_solution: np.ndarray,
    best_objective: float,
    rho: float,
    tau_min: float,
    tau_max: float,
    n: int
) -> np.ndarray:
    """Update pheromone levels using AS-rank strategy"""
    # Evaporation
    tau = (1 - rho) * tau

    # Sort solutions by objective value
    sorted_indices = np.argsort(objectives)

    # Deposit pheromones from top solutions
    n_elite = min(5, len(solutions))
    for rank, idx in enumerate(sorted_indices[:n_elite]):
        solution = solutions[idx]
        delta_tau = (n_elite - rank) / (objectives[idx] + 1e-10)

        for i in range(n - 1):
            tau[solution[i], solution[i + 1]] += delta_tau
            tau[solution[i + 1], solution[i]] += delta_tau  # Symmetric

    # Extra pheromone for global best
    if best_solution is not None:
        delta_tau_best = n_elite / (best_objective + 1e-10)
        for i in range(n - 1):
            tau[best_solution[i], best_solution[i + 1]] += delta_tau_best
            tau[best_solution[i + 1], best_solution[i]] += delta_tau_best

    # Apply bounds
    tau = np.clip(tau, tau_min, tau_max)

    return tau


def _evaluate_permutation_matrix(permutation: np.ndarray, matrix: np.ndarray) -> float:
    """Evaluate permutation on matrix (for QAP-like problems)"""
    n = len(permutation)
    cost = 0.0
    for i in range(n):
        for j in range(n):
            cost += matrix[i, j] * matrix[permutation[i], permutation[j]]
    return cost


__all__ = ['ant_colony_optimize']