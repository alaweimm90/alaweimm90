"""
MOEA/D: Multi-Objective Evolutionary Algorithm based on Decomposition.

Reference:
Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm
based on decomposition. IEEE Transactions on Evolutionary Computation,
11(6), 712-731.

Key Features:
- Decomposition of multi-objective problem into scalar subproblems
- Neighborhood-based mating and replacement
- Multiple decomposition approaches (Weighted Sum, Tchebycheff, PBI)
- Efficient for regular Pareto fronts
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import (
    MultiObjectiveProblem,
    MultiObjectiveSolution,
    ParetoFront,
)

logger = logging.getLogger(__name__)


class DecompositionMethod(Enum):
    """Decomposition methods for MOEA/D."""
    WEIGHTED_SUM = "weighted_sum"
    TCHEBYCHEFF = "tchebycheff"
    PBI = "pbi"  # Penalty-based Boundary Intersection


class MOEADOptimizer:
    """
    MOEA/D optimizer for multi-objective optimization.

    MOEA/D decomposes a multi-objective problem into N scalar optimization
    subproblems and optimizes them simultaneously. Each subproblem is
    optimized using information from neighboring subproblems.

    Attributes:
        problem: Multi-objective optimization problem
        population_size: Number of subproblems/solutions
        n_generations: Number of generations
        decomposition: Decomposition method to use
        n_neighbors: Size of neighborhood for each subproblem
        neighbor_selection_prob: Probability of selecting from neighborhood
        max_replacements: Maximum replacements in neighborhood
        pbi_theta: Penalty parameter for PBI decomposition
    """

    def __init__(
        self,
        problem: MultiObjectiveProblem,
        population_size: int = 100,
        n_generations: int = 100,
        decomposition: DecompositionMethod = DecompositionMethod.TCHEBYCHEFF,
        n_neighbors: int = 20,
        neighbor_selection_prob: float = 0.9,
        max_replacements: int = 2,
        crossover_prob: float = 1.0,
        mutation_prob: Optional[float] = None,
        eta_crossover: float = 20.0,
        eta_mutation: float = 20.0,
        pbi_theta: float = 5.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize MOEA/D optimizer.

        Args:
            problem: Multi-objective problem to optimize
            population_size: Number of subproblems
            n_generations: Number of generations
            decomposition: Decomposition method
            n_neighbors: Neighborhood size
            neighbor_selection_prob: Probability of neighborhood mating
            max_replacements: Max replacements in neighborhood per offspring
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            eta_crossover: SBX distribution index
            eta_mutation: Polynomial mutation distribution index
            pbi_theta: Penalty parameter for PBI
            seed: Random seed
        """
        self.problem = problem
        self.population_size = population_size
        self.n_generations = n_generations
        self.decomposition = decomposition
        self.n_neighbors = min(n_neighbors, population_size - 1)
        self.neighbor_selection_prob = neighbor_selection_prob
        self.max_replacements = max_replacements
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob or (1.0 / problem.n_variables)
        self.eta_crossover = eta_crossover
        self.eta_mutation = eta_mutation
        self.pbi_theta = pbi_theta

        if seed is not None:
            np.random.seed(seed)

        # Initialize weight vectors
        self.weight_vectors = self._generate_weight_vectors()

        # Calculate neighborhoods based on weight vector distances
        self.neighborhoods = self._calculate_neighborhoods()

        # Initialize population and objectives
        self.population: List[MultiObjectiveSolution] = []
        self.reference_point = np.full(problem.n_objectives, np.inf)
        self.pareto_front = ParetoFront()

        self.history: Dict[str, List] = {
            "hypervolume": [],
            "n_pareto": [],
            "avg_fitness": [],
        }

    def optimize(self) -> ParetoFront:
        """
        Run MOEA/D optimization.

        Returns:
            Pareto front containing non-dominated solutions
        """
        logger.info(
            f"Starting MOEA/D with {self.population_size} subproblems, "
            f"decomposition: {self.decomposition.value}"
        )

        # Initialize population
        self._initialize_population()

        # Main evolutionary loop
        for generation in range(self.n_generations):
            # Update reference point (ideal point)
            self._update_reference_point()

            # Evolve each subproblem
            for i in range(self.population_size):
                # Select mating pool
                if np.random.random() < self.neighbor_selection_prob:
                    # Select from neighborhood
                    mating_pool = self.neighborhoods[i]
                else:
                    # Select from entire population
                    mating_pool = list(range(self.population_size))

                # Generate offspring
                offspring = self._generate_offspring(i, mating_pool)

                # Update reference point with offspring
                self.reference_point = np.minimum(
                    self.reference_point,
                    offspring.objectives
                )

                # Update neighboring solutions
                self._update_neighbors(i, offspring)

            # Update Pareto front
            self._update_pareto_front()

            # Track history
            self._update_history(generation)

            if generation % 10 == 0:
                logger.info(
                    f"Generation {generation}: "
                    f"Pareto front size = {self.pareto_front.size()}, "
                    f"Reference point = {self.reference_point}"
                )

        logger.info(f"Optimization complete. Final Pareto front size: {self.pareto_front.size()}")
        return self.pareto_front

    def _generate_weight_vectors(self) -> np.ndarray:
        """
        Generate uniformly distributed weight vectors.

        Uses systematic approach similar to Das-Dennis for uniform distribution.

        Returns:
            Array of weight vectors (population_size, n_objectives)
        """
        m = self.problem.n_objectives

        if m == 2:
            # For 2 objectives, simple linear distribution
            weights = np.zeros((self.population_size, 2))
            weights[:, 0] = np.linspace(0, 1, self.population_size)
            weights[:, 1] = 1 - weights[:, 0]
        else:
            # For many objectives, use uniform distribution
            # Estimate number of partitions needed
            h = self.population_size
            while True:
                n_vectors = self._count_weight_vectors(m, h)
                if n_vectors >= self.population_size:
                    break
                h += 1

            # Generate weight vectors
            weights = self._generate_uniform_weights(m, h)

            # Select subset if too many
            if len(weights) > self.population_size:
                indices = np.random.choice(
                    len(weights),
                    self.population_size,
                    replace=False
                )
                weights = weights[indices]

        return weights

    def _count_weight_vectors(self, m: int, h: int) -> int:
        """Count number of weight vectors for given dimensions and divisions."""
        from math import factorial
        return factorial(h + m - 1) // (factorial(h) * factorial(m - 1))

    def _generate_uniform_weights(self, m: int, h: int) -> np.ndarray:
        """Generate uniform weight vectors using systematic approach."""
        def generate_recursive(weights, m, h, current):
            if m == 1:
                current.append(h)
                weights.append(np.array(current) / h)
            else:
                for i in range(h + 1):
                    generate_recursive(
                        weights,
                        m - 1,
                        h - i,
                        current + [i]
                    )

        weights = []
        generate_recursive(weights, m, h, [])
        return np.array(weights)

    def _calculate_neighborhoods(self) -> List[List[int]]:
        """
        Calculate neighborhoods based on weight vector distances.

        Each subproblem's neighborhood consists of the T closest
        subproblems in weight vector space.

        Returns:
            List of neighborhood indices for each subproblem
        """
        neighborhoods = []
        n = len(self.weight_vectors)

        # Calculate Euclidean distances between all weight vectors
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(
                    self.weight_vectors[i] - self.weight_vectors[j]
                )

        # Find T nearest neighbors for each subproblem
        for i in range(n):
            # Sort by distance and select nearest (excluding self)
            sorted_indices = np.argsort(distances[i])
            # Include self and T-1 nearest neighbors
            neighborhood = sorted_indices[:self.n_neighbors + 1].tolist()
            neighborhoods.append(neighborhood)

        return neighborhoods

    def _initialize_population(self):
        """Initialize population with random solutions."""
        self.population = []
        lower_bounds, upper_bounds = self.problem.bounds

        for i in range(self.population_size):
            # Random initialization
            variables = np.random.uniform(lower_bounds, upper_bounds)

            # Evaluate
            objectives = self.problem.evaluate(variables)
            is_feasible, violation = self.problem.evaluate_constraints(variables)

            solution = MultiObjectiveSolution(
                variables=variables,
                objectives=objectives,
                constraint_violation=violation,
                metadata={"subproblem_id": i}
            )
            self.population.append(solution)

            # Update reference point
            self.reference_point = np.minimum(self.reference_point, objectives)

    def _generate_offspring(
        self,
        subproblem_id: int,
        mating_pool: List[int]
    ) -> MultiObjectiveSolution:
        """
        Generate offspring for a subproblem.

        Args:
            subproblem_id: Index of current subproblem
            mating_pool: Indices of potential parents

        Returns:
            Generated offspring solution
        """
        # Select two parents from mating pool
        parent_indices = np.random.choice(mating_pool, size=2, replace=False)
        parent1 = self.population[parent_indices[0]]
        parent2 = self.population[parent_indices[1]]

        # Crossover
        if np.random.random() < self.crossover_prob:
            offspring_vars = self._de_crossover(
                self.population[subproblem_id].variables,
                parent1.variables,
                parent2.variables
            )
        else:
            offspring_vars = self.population[subproblem_id].variables.copy()

        # Mutation
        offspring_vars = self._polynomial_mutation(offspring_vars)

        # Evaluate offspring
        objectives = self.problem.evaluate(offspring_vars)
        is_feasible, violation = self.problem.evaluate_constraints(offspring_vars)

        offspring = MultiObjectiveSolution(
            variables=offspring_vars,
            objectives=objectives,
            constraint_violation=violation,
            metadata={"subproblem_id": subproblem_id}
        )

        return offspring

    def _de_crossover(
        self,
        current: np.ndarray,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> np.ndarray:
        """
        Differential Evolution crossover operator.

        Creates offspring: child = current + F * (parent1 - parent2)

        Args:
            current: Current solution variables
            parent1: First parent variables
            parent2: Second parent variables

        Returns:
            Offspring variables
        """
        F = 0.5  # Scaling factor
        CR = 1.0  # Crossover rate

        n = len(current)
        offspring = current.copy()
        lower_bounds, upper_bounds = self.problem.bounds

        j_rand = np.random.randint(n)  # Ensure at least one variable changes

        for j in range(n):
            if np.random.random() < CR or j == j_rand:
                offspring[j] = current[j] + F * (parent1[j] - parent2[j])
                # Ensure within bounds
                offspring[j] = np.clip(offspring[j], lower_bounds[j], upper_bounds[j])

        return offspring

    def _polynomial_mutation(self, variables: np.ndarray) -> np.ndarray:
        """Polynomial mutation operator."""
        mutated = variables.copy()
        lower_bounds, upper_bounds = self.problem.bounds

        for i in range(len(variables)):
            if np.random.random() < self.mutation_prob:
                y = variables[i]
                yl, yu = lower_bounds[i], upper_bounds[i]
                delta1 = (y - yl) / (yu - yl + 1e-10)
                delta2 = (yu - y) / (yu - yl + 1e-10)

                rand = np.random.random()
                mut_pow = 1.0 / (self.eta_mutation + 1.0)

                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.eta_mutation + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.eta_mutation + 1.0))
                    deltaq = 1.0 - val ** mut_pow

                mutated[i] = y + deltaq * (yu - yl)
                mutated[i] = np.clip(mutated[i], yl, yu)

        return mutated

    def _update_neighbors(
        self,
        subproblem_id: int,
        offspring: MultiObjectiveSolution
    ):
        """
        Update neighboring solutions with offspring if better.

        Args:
            subproblem_id: Index of current subproblem
            offspring: Generated offspring solution
        """
        replacements = 0

        # Shuffle neighborhood for random replacement order
        neighborhood = self.neighborhoods[subproblem_id].copy()
        np.random.shuffle(neighborhood)

        for neighbor_id in neighborhood:
            if replacements >= self.max_replacements:
                break

            # Calculate fitness for neighbor's subproblem
            offspring_fitness = self._calculate_fitness(
                offspring.objectives,
                self.weight_vectors[neighbor_id]
            )
            current_fitness = self._calculate_fitness(
                self.population[neighbor_id].objectives,
                self.weight_vectors[neighbor_id]
            )

            # Replace if offspring is better
            if offspring_fitness < current_fitness:
                self.population[neighbor_id] = MultiObjectiveSolution(
                    variables=offspring.variables.copy(),
                    objectives=offspring.objectives.copy(),
                    constraint_violation=offspring.constraint_violation,
                    metadata={"subproblem_id": neighbor_id}
                )
                replacements += 1

    def _calculate_fitness(
        self,
        objectives: np.ndarray,
        weight_vector: np.ndarray
    ) -> float:
        """
        Calculate fitness value for given objectives and weight vector.

        Uses the selected decomposition method to convert multi-objective
        problem to scalar.

        Args:
            objectives: Objective values
            weight_vector: Weight vector for decomposition

        Returns:
            Scalar fitness value
        """
        if self.decomposition == DecompositionMethod.WEIGHTED_SUM:
            return np.dot(weight_vector, objectives)

        elif self.decomposition == DecompositionMethod.TCHEBYCHEFF:
            # Tchebycheff approach: minimize max weighted distance
            # g(x|w,z*) = max_i {w_i * |f_i(x) - z*_i|}
            weighted_diff = weight_vector * np.abs(objectives - self.reference_point)
            return np.max(weighted_diff)

        elif self.decomposition == DecompositionMethod.PBI:
            # Penalty-based Boundary Intersection
            # d1: distance along weight vector
            # d2: perpendicular distance to weight vector
            norm_weight = np.linalg.norm(weight_vector)
            if norm_weight < 1e-10:
                return np.linalg.norm(objectives - self.reference_point)

            diff = objectives - self.reference_point
            d1 = np.abs(np.dot(diff, weight_vector)) / norm_weight

            d2_vec = diff - (d1 * weight_vector / norm_weight)
            d2 = np.linalg.norm(d2_vec)

            return d1 + self.pbi_theta * d2

        else:
            raise ValueError(f"Unknown decomposition method: {self.decomposition}")

    def _update_reference_point(self):
        """Update reference point (ideal point) from current population."""
        for solution in self.population:
            self.reference_point = np.minimum(
                self.reference_point,
                solution.objectives
            )

    def _update_pareto_front(self):
        """Update Pareto front with current population."""
        for solution in self.population:
            self.pareto_front.add(solution)

    def _update_history(self, generation: int):
        """Track optimization history."""
        self.history["n_pareto"].append(self.pareto_front.size())

        # Calculate average fitness across all subproblems
        avg_fitness = 0.0
        for i, solution in enumerate(self.population):
            fitness = self._calculate_fitness(
                solution.objectives,
                self.weight_vectors[i]
            )
            avg_fitness += fitness
        avg_fitness /= self.population_size
        self.history["avg_fitness"].append(avg_fitness)

        # Calculate hypervolume if reference point available
        if self.problem.reference_point is not None:
            from .core import hypervolume
            front_objectives = self.pareto_front.get_objectives()
            if len(front_objectives) > 0:
                hv = hypervolume(front_objectives, self.problem.reference_point)
                self.history["hypervolume"].append(hv)
            else:
                self.history["hypervolume"].append(0.0)