"""
Librex.QAP Solver - Quadratic Assignment Problem for MEZAN Agent Assignment

Solves the problem of assigning N agents to N tasks while considering pairwise
synergies and conflicts between agents.

Mathematical Formulation:
    minimize sum_{i,j} distance[i,j] * flow[assignment[i], assignment[j]]

where:
    - distance[i,j] = dissimilarity between task i and task j
    - flow[a,b] = interaction frequency between agent a and agent b
    - assignment[i] = which agent is assigned to task i
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


class Librex.QAPSolver(OptimizerInterface):
    """
    QAP Solver using Simulated Annealing

    This solver is optimized for MEZAN agent-task assignment where we want to:
    1. Minimize total workflow cost (distance * flow)
    2. Maximize agent synergies (similar agents on similar tasks)
    3. Handle constraints (agent capabilities, task requirements)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_gpu: bool = False,
        timeout: Optional[float] = None,
    ):
        super().__init__(config, enable_gpu, timeout)

        # Algorithm parameters (can be overridden in config)
        self.algorithm = config.get("algorithm", "simulated_annealing") if config else "simulated_annealing"
        self.max_iterations = config.get("max_iterations", 1000) if config else 1000
        self.temperature_init = config.get("temperature_init", 100.0) if config else 100.0
        self.cooling_rate = config.get("cooling_rate", 0.95) if config else 0.95

        logger.info(
            f"Librex.QAPSolver initialized: algorithm={self.algorithm}, "
            f"max_iter={self.max_iterations}, T0={self.temperature_init}"
        )

    def initialize(self) -> None:
        """Initialize solver (pre-allocate resources if needed)"""
        self._is_initialized = True
        logger.info("Librex.QAPSolver initialized and ready")

    def get_problem_types(self) -> List[ProblemType]:
        """This solver handles QAP problems"""
        return [ProblemType.QAP]

    def estimate_complexity(self, problem: OptimizationProblem) -> str:
        """
        Estimate computational complexity based on problem size

        QAP is NP-hard, so complexity grows super-polynomially
        """
        n = len(problem.data.get("distance_matrix", []))

        if n <= 10:
            return "low"
        elif n <= 30:
            return "medium"
        elif n <= 50:
            return "high"
        else:
            return "very_high"

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        """
        Solve QAP using Simulated Annealing

        Args:
            problem: OptimizationProblem with distance_matrix and flow_matrix

        Returns:
            OptimizationResult with optimal assignment
        """
        if not self._is_initialized:
            self.initialize()

        start_time = time.time()

        # Validate problem
        is_valid, error = self.validate_problem(problem)
        if not is_valid:
            return OptimizationResult(
                status=SolverStatus.FAILED,
                solution=None,
                objective_value=None,
                metadata={"error": error},
                computation_time=time.time() - start_time,
            )

        # Extract problem data
        distance_matrix = np.array(problem.data["distance_matrix"])
        flow_matrix = np.array(problem.data["flow_matrix"])
        n = len(distance_matrix)

        logger.info(f"Solving QAP instance of size n={n}")

        # Select algorithm
        if self.algorithm == "simulated_annealing":
            solution, obj_value, iterations = self._simulated_annealing(
                distance_matrix, flow_matrix
            )
        elif self.algorithm == "genetic":
            solution, obj_value, iterations = self._genetic_algorithm(
                distance_matrix, flow_matrix
            )
        else:
            # Fallback to random
            solution = list(range(n))
            np.random.shuffle(solution)
            obj_value = self._compute_objective(solution, distance_matrix, flow_matrix)
            iterations = 0

        computation_time = time.time() - start_time

        # Check timeout
        if self.timeout and computation_time > self.timeout:
            logger.warning(f"Librex.QAP timed out after {computation_time:.2f}s")
            status = SolverStatus.TIMEOUT
        else:
            status = SolverStatus.SUCCESS

        # Compute baseline for comparison
        random_solution = list(range(n))
        np.random.shuffle(random_solution)
        baseline_obj = self._compute_objective(
            random_solution, distance_matrix, flow_matrix
        )

        improvement = ((baseline_obj - obj_value) / baseline_obj * 100) if baseline_obj > 0 else 0.0

        logger.info(
            f"Librex.QAP solved: obj={obj_value:.2f}, "
            f"baseline={baseline_obj:.2f}, improvement={improvement:.1f}%"
        )

        return OptimizationResult(
            status=status,
            solution={"assignment": solution, "method": self.algorithm},
            objective_value=obj_value,
            metadata={
                "solver": "Librex.QAPSolver",
                "algorithm": self.algorithm,
                "problem_size": n,
                "baseline_objective": baseline_obj,
            },
            computation_time=computation_time,
            iterations=iterations,
            improvement_over_baseline=improvement,
        )

    def _compute_objective(
        self, assignment: List[int], distance: np.ndarray, flow: np.ndarray
    ) -> float:
        """
        Compute QAP objective function value

        obj = sum_{i,j} distance[i,j] * flow[assignment[i], assignment[j]]
        """
        n = len(assignment)
        obj = 0.0
        for i in range(n):
            for j in range(n):
                obj += distance[i, j] * flow[assignment[i], assignment[j]]
        return obj

    def _simulated_annealing(
        self, distance: np.ndarray, flow: np.ndarray
    ) -> tuple:
        """
        Simulated Annealing algorithm for QAP

        Returns:
            (best_solution, best_objective, iterations)
        """
        n = len(distance)

        # Initialize with random solution
        current_solution = list(range(n))
        np.random.shuffle(current_solution)
        current_obj = self._compute_objective(current_solution, distance, flow)

        best_solution = current_solution.copy()
        best_obj = current_obj

        temperature = self.temperature_init
        iterations = 0

        while iterations < self.max_iterations and temperature > 0.01:
            # Generate neighbor by swapping two random positions
            neighbor = current_solution.copy()
            i, j = np.random.choice(n, 2, replace=False)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

            neighbor_obj = self._compute_objective(neighbor, distance, flow)

            # Accept or reject neighbor
            delta = neighbor_obj - current_obj
            if delta < 0 or np.random.random() < np.exp(-delta / temperature):
                current_solution = neighbor
                current_obj = neighbor_obj

                # Update best
                if current_obj < best_obj:
                    best_solution = current_solution.copy()
                    best_obj = current_obj

            # Cool down
            temperature *= self.cooling_rate
            iterations += 1

        logger.debug(
            f"Simulated Annealing completed: {iterations} iterations, "
            f"best_obj={best_obj:.2f}"
        )

        return best_solution, best_obj, iterations

    def _genetic_algorithm(
        self, distance: np.ndarray, flow: np.ndarray
    ) -> tuple:
        """
        Genetic Algorithm for QAP (simplified implementation)

        Returns:
            (best_solution, best_objective, iterations)
        """
        n = len(distance)
        population_size = min(50, n * 2)
        num_generations = self.max_iterations // population_size

        # Initialize population
        population = [list(range(n)) for _ in range(population_size)]
        for individual in population:
            np.random.shuffle(individual)

        best_solution = None
        best_obj = float('inf')
        iterations = 0

        for generation in range(num_generations):
            # Evaluate fitness
            fitness = [
                self._compute_objective(individual, distance, flow)
                for individual in population
            ]

            # Track best
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_obj:
                best_solution = population[gen_best_idx].copy()
                best_obj = fitness[gen_best_idx]

            # Selection (tournament)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 3
                tournament_idx = np.random.choice(population_size, tournament_size, replace=False)
                winner_idx = tournament_idx[np.argmin([fitness[i] for i in tournament_idx])]
                new_population.append(population[winner_idx].copy())

            # Crossover and mutation
            for i in range(0, population_size, 2):
                if i + 1 < population_size and np.random.random() < 0.8:
                    # Order crossover
                    child1, child2 = self._order_crossover(
                        new_population[i], new_population[i + 1]
                    )
                    new_population[i] = child1
                    new_population[i + 1] = child2

            # Mutation
            for individual in new_population:
                if np.random.random() < 0.1:
                    # Swap mutation
                    i, j = np.random.choice(n, 2, replace=False)
                    individual[i], individual[j] = individual[j], individual[i]

            population = new_population
            iterations += population_size

        return best_solution, best_obj, iterations

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> tuple:
        """Order crossover (OX) for permutation encoding"""
        n = len(parent1)
        start, end = sorted(np.random.choice(n, 2, replace=False))

        child1 = [-1] * n
        child2 = [-1] * n

        # Copy segment from parent1 to child1
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        # Fill remaining from parent2
        pos = end
        for i in range(n):
            idx = (end + i) % n
            if parent2[idx] not in child1:
                child1[pos % n] = parent2[idx]
                pos += 1

        # Fill child2 similarly
        pos = end
        for i in range(n):
            idx = (end + i) % n
            if parent1[idx] not in child2:
                child2[pos % n] = parent1[idx]
                pos += 1

        return child1, child2
