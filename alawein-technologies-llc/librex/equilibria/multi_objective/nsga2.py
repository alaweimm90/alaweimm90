"""
NSGA-II: Non-dominated Sorting Genetic Algorithm II.

Reference:
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
A fast and elitist multiobjective genetic algorithm: NSGA-II.
IEEE Transactions on Evolutionary Computation, 6(2), 182-197.

Key Features:
- Fast non-dominated sorting: O(MN²) complexity
- Crowding distance for diversity preservation
- Elitism through (μ + λ) selection
- Binary tournament selection with crowding comparison
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import (
    MultiObjectiveProblem,
    MultiObjectiveSolution,
    ParetoFront,
    fast_non_dominated_sort,
    crowding_distance,
)

logger = logging.getLogger(__name__)


class NSGA2Optimizer:
    """
    NSGA-II multi-objective optimization algorithm.

    The algorithm maintains diversity through crowding distance while
    converging to the Pareto front using non-dominated sorting.

    Attributes:
        problem: Multi-objective optimization problem
        population_size: Number of solutions in population
        n_generations: Number of generations to evolve
        crossover_prob: Probability of crossover (default: 0.9)
        mutation_prob: Probability of mutation (default: 1/n_variables)
        eta_crossover: Distribution index for SBX crossover (default: 15)
        eta_mutation: Distribution index for polynomial mutation (default: 20)
    """

    def __init__(
        self,
        problem: MultiObjectiveProblem,
        population_size: int = 100,
        n_generations: int = 100,
        crossover_prob: float = 0.9,
        mutation_prob: Optional[float] = None,
        eta_crossover: float = 15.0,
        eta_mutation: float = 20.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize NSGA-II optimizer.

        Args:
            problem: Multi-objective problem to optimize
            population_size: Size of population (even number)
            n_generations: Number of generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability (default: 1/n_vars)
            eta_crossover: SBX distribution index
            eta_mutation: Polynomial mutation distribution index
            seed: Random seed for reproducibility
        """
        self.problem = problem
        self.population_size = population_size if population_size % 2 == 0 else population_size + 1
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob or (1.0 / problem.n_variables)
        self.eta_crossover = eta_crossover
        self.eta_mutation = eta_mutation

        if seed is not None:
            np.random.seed(seed)

        self.population: List[MultiObjectiveSolution] = []
        self.pareto_front = ParetoFront()
        self.history: Dict[str, List] = {
            "hypervolume": [],
            "n_pareto": [],
            "avg_rank": [],
        }

    def optimize(self) -> ParetoFront:
        """
        Run NSGA-II optimization.

        Returns:
            Pareto front containing non-dominated solutions
        """
        logger.info(f"Starting NSGA-II with population size {self.population_size}")

        # Initialize population
        self._initialize_population()

        # Main evolutionary loop
        for generation in range(self.n_generations):
            # Create offspring population
            offspring = self._create_offspring()

            # Combine parent and offspring populations
            combined_population = self.population + offspring

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(combined_population)

            # Select next generation
            self.population = self._environmental_selection(combined_population, fronts)

            # Update Pareto front
            self._update_pareto_front()

            # Track statistics
            self._update_history(generation)

            if generation % 10 == 0:
                logger.info(
                    f"Generation {generation}: "
                    f"Pareto front size = {self.pareto_front.size()}"
                )

        logger.info(f"Optimization complete. Final Pareto front size: {self.pareto_front.size()}")
        return self.pareto_front

    def _initialize_population(self):
        """Initialize population with random solutions."""
        self.population = []
        lower_bounds, upper_bounds = self.problem.bounds

        for _ in range(self.population_size):
            # Random initialization within bounds
            variables = np.random.uniform(lower_bounds, upper_bounds)

            # Evaluate objectives
            objectives = self.problem.evaluate(variables)

            # Evaluate constraints
            is_feasible, violation = self.problem.evaluate_constraints(variables)

            solution = MultiObjectiveSolution(
                variables=variables,
                objectives=objectives,
                constraint_violation=violation,
            )
            self.population.append(solution)

        # Initial non-dominated sorting
        fronts = fast_non_dominated_sort(self.population)

        # Calculate crowding distance for first front
        if fronts:
            crowding_distance(self.population, fronts[0])

    def _create_offspring(self) -> List[MultiObjectiveSolution]:
        """Create offspring population through selection, crossover, and mutation."""
        offspring = []

        while len(offspring) < self.population_size:
            # Binary tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if np.random.random() < self.crossover_prob:
                child1_vars, child2_vars = self._sbx_crossover(
                    parent1.variables,
                    parent2.variables
                )
            else:
                child1_vars = parent1.variables.copy()
                child2_vars = parent2.variables.copy()

            # Mutation
            child1_vars = self._polynomial_mutation(child1_vars)
            child2_vars = self._polynomial_mutation(child2_vars)

            # Create offspring solutions
            for variables in [child1_vars, child2_vars]:
                objectives = self.problem.evaluate(variables)
                is_feasible, violation = self.problem.evaluate_constraints(variables)

                child = MultiObjectiveSolution(
                    variables=variables,
                    objectives=objectives,
                    constraint_violation=violation,
                )
                offspring.append(child)

                if len(offspring) >= self.population_size:
                    break

        return offspring[:self.population_size]

    def _tournament_selection(self) -> MultiObjectiveSolution:
        """
        Binary tournament selection based on rank and crowding distance.

        Selection criteria:
        1. Lower rank (better)
        2. If same rank, higher crowding distance (more diverse)

        Returns:
            Selected solution
        """
        idx1, idx2 = np.random.choice(len(self.population), size=2, replace=False)
        sol1, sol2 = self.population[idx1], self.population[idx2]

        # Compare by rank
        if sol1.rank < sol2.rank:
            return sol1
        elif sol2.rank < sol1.rank:
            return sol2

        # Same rank - compare by crowding distance
        if sol1.crowding_distance > sol2.crowding_distance:
            return sol1
        elif sol2.crowding_distance > sol1.crowding_distance:
            return sol2

        # Equal - random choice
        return sol1 if np.random.random() < 0.5 else sol2

    def _sbx_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulated Binary Crossover (SBX).

        Creates offspring that maintain similar search properties
        to real-coded genetic algorithms.

        Args:
            parent1: First parent variables
            parent2: Second parent variables

        Returns:
            Tuple of two offspring variable vectors
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        lower_bounds, upper_bounds = self.problem.bounds

        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    # Calculate beta
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]

                    rand = np.random.random()
                    beta = 1.0 + (2.0 * (y1 - lower_bounds[i]) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(self.eta_crossover + 1.0))

                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (self.eta_crossover + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_crossover + 1.0))

                    # Create offspring
                    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                    # Ensure within bounds
                    child1[i] = np.clip(c1, lower_bounds[i], upper_bounds[i])
                    child2[i] = np.clip(c2, lower_bounds[i], upper_bounds[i])

        return child1, child2

    def _polynomial_mutation(self, variables: np.ndarray) -> np.ndarray:
        """
        Polynomial mutation operator.

        Args:
            variables: Variable vector to mutate

        Returns:
            Mutated variable vector
        """
        mutated = variables.copy()
        lower_bounds, upper_bounds = self.problem.bounds

        for i in range(len(variables)):
            if np.random.random() < self.mutation_prob:
                delta = min(
                    variables[i] - lower_bounds[i],
                    upper_bounds[i] - variables[i]
                ) / (upper_bounds[i] - lower_bounds[i])

                rand = np.random.random()
                if rand < 0.5:
                    deltaq = (2.0 * rand) ** (1.0 / (self.eta_mutation + 1.0)) - 1.0
                else:
                    deltaq = 1.0 - (2.0 * (1.0 - rand)) ** (1.0 / (self.eta_mutation + 1.0))

                mutated[i] = variables[i] + deltaq * (upper_bounds[i] - lower_bounds[i])
                mutated[i] = np.clip(mutated[i], lower_bounds[i], upper_bounds[i])

        return mutated

    def _environmental_selection(
        self,
        combined_population: List[MultiObjectiveSolution],
        fronts: List[List[int]]
    ) -> List[MultiObjectiveSolution]:
        """
        Select next generation using non-dominated sorting and crowding distance.

        Args:
            combined_population: Combined parent and offspring population
            fronts: Non-dominated fronts

        Returns:
            Selected population for next generation
        """
        next_population = []
        front_idx = 0

        # Add complete fronts that fit in population size
        while front_idx < len(fronts) and len(next_population) + len(fronts[front_idx]) <= self.population_size:
            for idx in fronts[front_idx]:
                next_population.append(combined_population[idx])
            front_idx += 1

        # Handle last front that doesn't completely fit
        if len(next_population) < self.population_size and front_idx < len(fronts):
            last_front_indices = fronts[front_idx]

            # Calculate crowding distance for last front
            crowding_distance(combined_population, last_front_indices)

            # Sort by crowding distance (descending)
            sorted_indices = sorted(
                last_front_indices,
                key=lambda i: combined_population[i].crowding_distance,
                reverse=True
            )

            # Add individuals with highest crowding distance
            remaining_slots = self.population_size - len(next_population)
            for idx in sorted_indices[:remaining_slots]:
                next_population.append(combined_population[idx])

        return next_population

    def _update_pareto_front(self):
        """Update the maintained Pareto front with current population."""
        for solution in self.population:
            if solution.rank == 0:  # Only consider non-dominated solutions
                self.pareto_front.add(solution)

    def _update_history(self, generation: int):
        """
        Track optimization history for analysis.

        Args:
            generation: Current generation number
        """
        # Calculate average rank
        avg_rank = np.mean([sol.rank for sol in self.population])
        self.history["avg_rank"].append(avg_rank)

        # Track Pareto front size
        self.history["n_pareto"].append(self.pareto_front.size())

        # Calculate hypervolume if reference point is available
        if self.problem.reference_point is not None:
            from .core import hypervolume
            front_objectives = self.pareto_front.get_objectives()
            if len(front_objectives) > 0:
                hv = hypervolume(front_objectives, self.problem.reference_point)
                self.history["hypervolume"].append(hv)
            else:
                self.history["hypervolume"].append(0.0)