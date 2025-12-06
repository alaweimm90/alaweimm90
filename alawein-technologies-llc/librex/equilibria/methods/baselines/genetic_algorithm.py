"""Genetic Algorithm optimization method"""

import logging
from typing import Any, Dict, List

import numpy as np

from Librex.core.interfaces import StandardizedProblem

logger = logging.getLogger(__name__)


def genetic_algorithm_optimize(
    problem: StandardizedProblem,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Genetic Algorithm optimization

    Population-based evolutionary optimization using selection, crossover, mutation.

    Args:
        problem: Standardized optimization problem
        config: Configuration dict with keys:
            - population_size: Size of population (default: 100)
            - generations: Number of generations (default: 100)
            - mutation_rate: Probability of mutation (default: 0.1)
            - crossover_rate: Probability of crossover (default: 0.8)
            - seed: Random seed (default: None)

    Returns:
        dict: Optimization result
    """
    pop_size = config.get('population_size', 100)
    generations = config.get('generations', 100)
    mutation_rate = config.get('mutation_rate', 0.1)
    crossover_rate = config.get('crossover_rate', 0.8)
    seed = config.get('seed', None)

    if seed is not None:
        np.random.seed(seed)

    n = problem.dimension

    # Initialize population
    population: List[np.ndarray] = []
    for _ in range(pop_size):
        individual = np.arange(n)
        np.random.shuffle(individual)
        population.append(individual)

    best_solution = None
    best_objective = float('inf')

    for gen in range(generations):
        # Evaluate fitness
        fitness = np.array([problem.objective_function(ind) for ind in population])

        # Track best
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_objective:
            best_solution = population[min_idx].copy()
            best_objective = fitness[min_idx]

        # Selection (tournament)
        new_population: List[np.ndarray] = []
        for _ in range(pop_size):
            idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
            parent = population[idx1] if fitness[idx1] < fitness[idx2] else population[idx2]
            new_population.append(parent.copy())

        # Crossover (order crossover)
        for i in range(0, pop_size - 1, 2):
            if np.random.random() < crossover_rate:
                child1, child2 = _order_crossover(new_population[i], new_population[i + 1])
                new_population[i] = child1
                new_population[i + 1] = child2

        # Mutation (swap mutation)
        for i in range(pop_size):
            if np.random.random() < mutation_rate:
                idx1, idx2 = np.random.choice(n, 2, replace=False)
                new_population[i][idx1], new_population[i][idx2] = (
                    new_population[i][idx2],
                    new_population[i][idx1]
                )

        population = new_population

    return {
        'solution': best_solution,
        'objective': float(best_objective),
        'is_valid': True,
        'iterations': generations * pop_size,
        'convergence': {'converged': False, 'generations': generations},
        'metadata': {
            'method': 'genetic_algorithm',
            'population_size': pop_size,
            'seed': seed
        }
    }


def _order_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple:
    """Order crossover operator for permutations"""
    n = len(parent1)
    start, end = sorted(np.random.choice(n, 2, replace=False))

    child1 = np.full(n, -1)
    child2 = np.full(n, -1)

    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]

    # Fill remaining positions
    _fill_child(child1, parent2, end, n)
    _fill_child(child2, parent1, end, n)

    return child1, child2


def _fill_child(child: np.ndarray, parent: np.ndarray, start: int, n: int) -> None:
    """Helper to fill child with remaining genes from parent"""
    pos = start
    for gene in np.concatenate([parent[start:], parent[:start]]):
        if gene not in child:
            child[pos % n] = gene
            pos += 1


__all__ = ['genetic_algorithm_optimize']
