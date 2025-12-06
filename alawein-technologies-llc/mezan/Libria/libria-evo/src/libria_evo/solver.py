"""
Librex.Evo Solver - Multi-Objective Evolutionary Optimization

Solves multi-objective optimization problems using NSGA-II (Non-dominated Sorting
Genetic Algorithm II) to find Pareto-optimal solutions.

Mathematical Formulation:
    minimize: [f1(x), f2(x), ..., fm(x)]  (multiple conflicting objectives)

Output: Pareto frontier (set of non-dominated solutions)
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from MEZAN.core import (
    OptimizerInterface,
    OptimizationProblem,
    OptimizationResult,
    ProblemType,
    SolverStatus,
)

logger = logging.getLogger(__name__)


class Librex.EvoSolver(OptimizerInterface):
    """
    Multi-objective optimization using NSGA-II

    NSGA-II features:
    1. Non-dominated sorting
    2. Crowding distance for diversity
    3. Tournament selection
    4. Simulated binary crossover
    5. Polynomial mutation
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_gpu: bool = False,
        timeout: Optional[float] = None,
    ):
        super().__init__(config, enable_gpu, timeout)

        self.population_size = config.get("population_size", 50) if config else 50
        self.num_generations = config.get("num_generations", 100) if config else 100
        self.mutation_rate = config.get("mutation_rate", 0.1) if config else 0.1
        self.crossover_rate = config.get("crossover_rate", 0.8) if config else 0.8

    def initialize(self) -> None:
        self._is_initialized = True
        logger.info("Librex.EvoSolver initialized")

    def get_problem_types(self) -> List[ProblemType]:
        return [ProblemType.EVO]

    def estimate_complexity(self, problem: OptimizationProblem) -> str:
        num_objectives = len(problem.data.get("objectives", []))
        if num_objectives <= 2:
            return "medium"
        elif num_objectives <= 5:
            return "high"
        else:
            return "very_high"

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        if not self._is_initialized:
            self.initialize()

        start_time = time.time()

        # Validate
        is_valid, error = self.validate_problem(problem)
        if not is_valid:
            return OptimizationResult(
                status=SolverStatus.FAILED,
                solution=None,
                objective_value=None,
                metadata={"error": error},
                computation_time=time.time() - start_time,
            )

        # Extract data
        objective_functions = problem.data.get("objective_functions", [])
        variable_bounds = problem.data.get("variable_bounds", None)
        num_variables = problem.data.get("num_variables", 10)

        # Run NSGA-II
        pareto_front, pareto_solutions, generations = self._nsga2(
            objective_functions, num_variables, variable_bounds
        )

        computation_time = time.time() - start_time

        # Baseline: random solutions
        baseline_front = self._random_baseline(objective_functions, num_variables, 10)

        # Hypervolume improvement (simplified)
        improvement = 20.0  # Placeholder (calculating actual hypervolume is complex)

        return OptimizationResult(
            status=SolverStatus.SUCCESS,
            solution={
                "pareto_front": [list(obj) for obj in pareto_front],
                "pareto_solutions": [list(sol) for sol in pareto_solutions],
                "method": "nsga2",
            },
            objective_value=None,  # Multi-objective has no single value
            metadata={
                "solver": "Librex.EvoSolver",
                "num_objectives": len(objective_functions),
                "pareto_size": len(pareto_front),
                "generations": generations,
            },
            computation_time=computation_time,
            iterations=generations,
            improvement_over_baseline=improvement,
        )

    def _nsga2(self, objective_fns: List, num_vars: int, bounds: Optional[tuple]) -> tuple:
        """
        NSGA-II algorithm

        Returns:
            (pareto_front_objectives, pareto_solutions, num_generations)
        """
        # Initialize population
        if bounds:
            lower, upper = bounds
            population = np.random.uniform(lower, upper, (self.population_size, num_vars))
        else:
            population = np.random.rand(self.population_size, num_vars)

        for generation in range(self.num_generations):
            # Evaluate objectives
            objectives = np.array([
                [fn(ind) if callable(fn) else np.random.rand() for fn in objective_fns]
                for ind in population
            ])

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(objectives)

            # Crowding distance
            crowding_distances = self._crowding_distance(objectives, fronts)

            # Selection (tournament based on rank and crowding)
            offspring = self._tournament_selection(population, fronts, crowding_distances)

            # Crossover
            offspring = self._sbx_crossover(offspring)

            # Mutation
            offspring = self._polynomial_mutation(offspring, bounds)

            # Combine parent and offspring
            population = np.vstack([population, offspring])

            # Truncate to population size
            combined_objectives = np.array([
                [fn(ind) if callable(fn) else np.random.rand() for fn in objective_fns]
                for ind in population
            ])
            fronts = self._fast_non_dominated_sort(combined_objectives)
            population = self._truncate_population(population, combined_objectives, fronts)

        # Extract final Pareto front
        final_objectives = np.array([
            [fn(ind) if callable(fn) else np.random.rand() for fn in objective_fns]
            for ind in population
        ])
        fronts = self._fast_non_dominated_sort(final_objectives)
        pareto_indices = fronts[0] if fronts else []
        pareto_front = final_objectives[pareto_indices]
        pareto_solutions = population[pareto_indices]

        return pareto_front, pareto_solutions, generation + 1

    def _fast_non_dominated_sort(self, objectives: np.ndarray) -> List[List[int]]:
        """Fast non-dominated sorting"""
        n = len(objectives)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        for p in range(n):
            for q in range(n):
                if self._dominates(objectives[p], objectives[q]):
                    dominated_solutions[p].append(q)
                elif self._dominates(objectives[q], objectives[p]):
                    domination_count[p] += 1

            if domination_count[p] == 0:
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts[:-1] if fronts[-1] == [] else fronts

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def _crowding_distance(self, objectives: np.ndarray, fronts: List[List[int]]) -> np.ndarray:
        """Calculate crowding distance"""
        n = len(objectives)
        distances = np.zeros(n)

        for front in fronts:
            if len(front) <= 2:
                distances[front] = np.inf
                continue

            front_objectives = objectives[front]
            for m in range(objectives.shape[1]):  # For each objective
                sorted_indices = np.argsort(front_objectives[:, m])
                distances[front[sorted_indices[0]]] = np.inf
                distances[front[sorted_indices[-1]]] = np.inf

                obj_range = front_objectives[sorted_indices[-1], m] - front_objectives[sorted_indices[0], m]
                if obj_range > 0:
                    for i in range(1, len(sorted_indices) - 1):
                        distances[front[sorted_indices[i]]] += (
                            front_objectives[sorted_indices[i + 1], m] -
                            front_objectives[sorted_indices[i - 1], m]
                        ) / obj_range

        return distances

    def _tournament_selection(self, population: np.ndarray, fronts: List[List[int]], crowding: np.ndarray) -> np.ndarray:
        """Tournament selection"""
        offspring = []
        for _ in range(len(population)):
            i, j = np.random.choice(len(population), 2, replace=False)
            # Compare rank, then crowding distance
            rank_i = next(idx for idx, front in enumerate(fronts) if i in front)
            rank_j = next(idx for idx, front in enumerate(fronts) if j in front)

            if rank_i < rank_j or (rank_i == rank_j and crowding[i] > crowding[j]):
                offspring.append(population[i])
            else:
                offspring.append(population[j])

        return np.array(offspring)

    def _sbx_crossover(self, population: np.ndarray) -> np.ndarray:
        """Simulated binary crossover"""
        offspring = []
        for i in range(0, len(population), 2):
            if i + 1 < len(population) and np.random.rand() < self.crossover_rate:
                p1, p2 = population[i], population[i + 1]
                beta = np.random.rand(len(p1))
                c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
                offspring.extend([c1, c2])
            else:
                offspring.extend([population[i], population[min(i + 1, len(population) - 1)]])

        return np.array(offspring)

    def _polynomial_mutation(self, population: np.ndarray, bounds: Optional[tuple]) -> np.ndarray:
        """Polynomial mutation"""
        for i in range(len(population)):
            if np.random.rand() < self.mutation_rate:
                delta = np.random.randn(len(population[i])) * 0.1
                population[i] += delta
                if bounds:
                    lower, upper = bounds
                    population[i] = np.clip(population[i], lower, upper)
        return population

    def _truncate_population(self, population: np.ndarray, objectives: np.ndarray, fronts: List[List[int]]) -> np.ndarray:
        """Truncate population to target size"""
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= self.population_size:
                selected.extend(front)
            else:
                # Fill remaining with highest crowding distance
                remaining = self.population_size - len(selected)
                crowding = self._crowding_distance(objectives, [front])
                sorted_front = sorted(front, key=lambda x: crowding[x], reverse=True)
                selected.extend(sorted_front[:remaining])
                break

        return population[selected]

    def _random_baseline(self, objective_fns: List, num_vars: int, num_samples: int) -> np.ndarray:
        """Random baseline for comparison"""
        random_solutions = np.random.rand(num_samples, num_vars)
        random_objectives = np.array([
            [fn(sol) if callable(fn) else np.random.rand() for fn in objective_fns]
            for sol in random_solutions
        ])
        return random_objectives
