"""
Particle Swarm Optimization (PSO) adapted for permutation problems

References:
    - Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
      Proceedings of IEEE International Conference on Neural Networks, 4, 1942-1948.
    - Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer.
      IEEE International Conference on Evolutionary Computation, 69-73.
    - Clerc, M. (2012). Standard Particle Swarm Optimisation. HAL Open Science.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from Librex.core.interfaces import StandardizedProblem

logger = logging.getLogger(__name__)


def particle_swarm_optimize(
    problem: StandardizedProblem,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Particle Swarm Optimization for permutation problems

    PSO uses a swarm of particles that move through the search space,
    influenced by their own best position and the global best position.
    This implementation is adapted for discrete permutation problems.

    Args:
        problem: Standardized optimization problem
        config: Configuration dict with keys:
            - n_particles: Number of particles (default: 30)
            - n_iterations: Number of iterations (default: 100)
            - w: Inertia weight (default: 0.7)
            - c1: Cognitive coefficient (default: 2.0)
            - c2: Social coefficient (default: 2.0)
            - v_max: Maximum velocity (default: 4.0)
            - seed: Random seed (default: None)

    Returns:
        dict: Optimization result with solution, objective, and metadata
    """
    n_particles = config.get('n_particles', 30)
    n_iterations = config.get('n_iterations', 100)
    w = config.get('w', 0.7)  # Inertia weight
    c1 = config.get('c1', 2.0)  # Cognitive coefficient
    c2 = config.get('c2', 2.0)  # Social coefficient
    v_max = config.get('v_max', 4.0)
    seed = config.get('seed', None)

    if seed is not None:
        np.random.seed(seed)

    n = problem.dimension

    # Initialize particles with random permutations
    particles = []
    velocities = []
    personal_best = []
    personal_best_scores = []

    for _ in range(n_particles):
        # Random permutation
        particle = np.random.permutation(n)
        particles.append(particle)

        # Initialize velocity as empty swap sequence
        velocities.append([])

        # Initialize personal best
        personal_best.append(particle.copy())

        # Evaluate initial position
        if problem.objective_function:
            score = problem.objective_function(particle)
        else:
            score = _evaluate_permutation(particle, problem.objective_matrix)

        personal_best_scores.append(score)

    # Initialize global best
    best_idx = np.argmin(personal_best_scores)
    global_best = personal_best[best_idx].copy()
    global_best_score = personal_best_scores[best_idx]

    convergence_history = []

    for iteration in range(n_iterations):
        # Update each particle
        for i in range(n_particles):
            # Update velocity
            velocity = _update_velocity_discrete(
                particles[i], velocities[i],
                personal_best[i], global_best,
                w, c1, c2, v_max
            )
            velocities[i] = velocity

            # Update position
            new_particle = _apply_velocity(particles[i], velocity)
            particles[i] = new_particle

            # Evaluate new position
            if problem.objective_function:
                score = problem.objective_function(new_particle)
            else:
                score = _evaluate_permutation(new_particle, problem.objective_matrix)

            # Update personal best
            if score < personal_best_scores[i]:
                personal_best[i] = new_particle.copy()
                personal_best_scores[i] = score

                # Update global best
                if score < global_best_score:
                    global_best = new_particle.copy()
                    global_best_score = score

        convergence_history.append(global_best_score)

        # Adaptive inertia weight (linear decrease)
        w = w * 0.99

        # Check convergence
        if len(convergence_history) > 20:
            recent = convergence_history[-20:]
            if max(recent) - min(recent) < 1e-6:
                logger.info(f"PSO converged at iteration {iteration}")
                break

    return {
        'solution': global_best,
        'objective': float(global_best_score),
        'is_valid': True,
        'iterations': (iteration + 1) * n_particles,
        'convergence': {
            'converged': len(convergence_history) > 20 and
                        max(convergence_history[-20:]) - min(convergence_history[-20:]) < 1e-6,
            'history': convergence_history,
            'final_iteration': iteration + 1
        },
        'metadata': {
            'method': 'particle_swarm_optimization',
            'n_particles': n_particles,
            'w_final': w,
            'c1': c1,
            'c2': c2,
            'seed': seed
        }
    }


def _update_velocity_discrete(
    current: np.ndarray,
    velocity: List[Tuple[int, int]],
    p_best: np.ndarray,
    g_best: np.ndarray,
    w: float,
    c1: float,
    c2: float,
    v_max: float
) -> List[Tuple[int, int]]:
    """
    Update velocity for discrete PSO using swap sequences

    Velocity is represented as a list of swap operations (i, j)
    """
    new_velocity = []

    # Inertia: keep some of current velocity
    if velocity:
        n_keep = int(len(velocity) * w)
        if n_keep > 0:
            keep_indices = np.random.choice(len(velocity), n_keep, replace=False)
            new_velocity = [velocity[i] for i in keep_indices]

    # Cognitive component: moves towards personal best
    p_swaps = _get_swaps_between(current, p_best)
    if p_swaps:
        n_add = int(len(p_swaps) * c1 * np.random.random())
        if n_add > 0:
            add_indices = np.random.choice(len(p_swaps), min(n_add, len(p_swaps)), replace=False)
            new_velocity.extend([p_swaps[i] for i in add_indices])

    # Social component: moves towards global best
    g_swaps = _get_swaps_between(current, g_best)
    if g_swaps:
        n_add = int(len(g_swaps) * c2 * np.random.random())
        if n_add > 0:
            add_indices = np.random.choice(len(g_swaps), min(n_add, len(g_swaps)), replace=False)
            new_velocity.extend([g_swaps[i] for i in add_indices])

    # Limit velocity magnitude
    if len(new_velocity) > v_max:
        new_velocity = new_velocity[:int(v_max)]

    return new_velocity


def _get_swaps_between(perm1: np.ndarray, perm2: np.ndarray) -> List[Tuple[int, int]]:
    """Get swap sequence to transform perm1 into perm2"""
    swaps = []
    current = perm1.copy()
    target = perm2.copy()

    # Create inverse mapping for target
    inv_target = np.zeros(len(target), dtype=int)
    for i, val in enumerate(target):
        inv_target[val] = i

    for i in range(len(current)):
        if current[i] != target[i]:
            # Find where target[i] is in current
            j = np.where(current == target[i])[0][0]
            # Swap
            current[i], current[j] = current[j], current[i]
            swaps.append((i, j))

    return swaps


def _apply_velocity(permutation: np.ndarray, velocity: List[Tuple[int, int]]) -> np.ndarray:
    """Apply velocity (swap sequence) to permutation"""
    result = permutation.copy()
    for i, j in velocity:
        if i < len(result) and j < len(result):
            result[i], result[j] = result[j], result[i]
    return result


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


__all__ = ['particle_swarm_optimize']