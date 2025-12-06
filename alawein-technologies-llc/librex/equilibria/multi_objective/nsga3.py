"""
NSGA-III: Reference-point-based Non-dominated Sorting Genetic Algorithm.

Reference:
Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization
algorithm using reference-point-based nondominated sorting approach,
Part I: Solving problems with box constraints.
IEEE Transactions on Evolutionary Computation, 18(4), 577-601.

Key Features:
- Reference point based selection for many-objective problems
- Structured reference points using Das-Dennis method
- Adaptive normalization of objectives
- Association of solutions to reference points
- Niche preservation for diversity
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from .core import (
    MultiObjectiveProblem,
    MultiObjectiveSolution,
    ParetoFront,
    fast_non_dominated_sort,
)

logger = logging.getLogger(__name__)


class NSGA3Optimizer:
    """
    NSGA-III for many-objective optimization (3+ objectives).

    NSGA-III replaces crowding distance with reference-point-based
    selection, making it more effective for problems with many objectives
    where crowding distance becomes less effective.

    Attributes:
        problem: Multi-objective optimization problem
        population_size: Number of solutions (adjusted to match reference points)
        n_generations: Number of generations
        n_partitions: Number of divisions for reference points (or list for layers)
        crossover_prob: Probability of crossover
        mutation_prob: Probability of mutation
    """

    def __init__(
        self,
        problem: MultiObjectiveProblem,
        population_size: Optional[int] = None,
        n_generations: int = 100,
        n_partitions: int = 12,
        crossover_prob: float = 0.9,
        mutation_prob: Optional[float] = None,
        eta_crossover: float = 30.0,
        eta_mutation: float = 20.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize NSGA-III optimizer.

        Args:
            problem: Multi-objective problem to optimize
            population_size: Population size (auto-adjusted to reference points)
            n_generations: Number of generations
            n_partitions: Number of divisions for reference points
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            eta_crossover: SBX distribution index
            eta_mutation: Polynomial mutation distribution index
            seed: Random seed
        """
        self.problem = problem
        self.n_generations = n_generations
        self.n_partitions = n_partitions
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob or (1.0 / problem.n_variables)
        self.eta_crossover = eta_crossover
        self.eta_mutation = eta_mutation

        if seed is not None:
            np.random.seed(seed)

        # Generate reference points
        self.reference_points = self._generate_reference_points()
        n_ref_points = len(self.reference_points)

        # Adjust population size to be multiple of reference points
        if population_size is None:
            self.population_size = n_ref_points
        else:
            # Round to nearest multiple of 4 for better mating pool
            self.population_size = 4 * round(population_size / 4)

        logger.info(
            f"NSGA-III initialized with {n_ref_points} reference points, "
            f"population size: {self.population_size}"
        )

        self.population: List[MultiObjectiveSolution] = []
        self.pareto_front = ParetoFront()
        self.ideal_point = np.full(problem.n_objectives, np.inf)
        self.nadir_point = np.full(problem.n_objectives, -np.inf)
        self.history: Dict[str, List] = {
            "hypervolume": [],
            "n_pareto": [],
            "ideal_point": [],
        }

    def optimize(self) -> ParetoFront:
        """
        Run NSGA-III optimization.

        Returns:
            Pareto front containing non-dominated solutions
        """
        logger.info(f"Starting NSGA-III optimization")

        # Initialize population
        self._initialize_population()

        # Main evolutionary loop
        for generation in range(self.n_generations):
            # Create offspring
            offspring = self._create_offspring()

            # Combine populations
            combined_population = self.population + offspring

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(combined_population)

            # Environmental selection with reference points
            self.population = self._environmental_selection(combined_population, fronts)

            # Update Pareto front
            self._update_pareto_front()

            # Update ideal and nadir points
            self._update_extreme_points()

            # Track history
            self._update_history(generation)

            if generation % 10 == 0:
                logger.info(
                    f"Generation {generation}: "
                    f"Pareto front size = {self.pareto_front.size()}"
                )

        logger.info(f"Optimization complete. Final Pareto front size: {self.pareto_front.size()}")
        return self.pareto_front

    def _generate_reference_points(self) -> np.ndarray:
        """
        Generate structured reference points using Das-Dennis method.

        The Das-Dennis method generates uniformly distributed points
        on the unit simplex.

        Returns:
            Array of reference points
        """
        m = self.problem.n_objectives
        p = self.n_partitions

        # Generate reference points on unit simplex
        ref_points = self._das_dennis_points(m, p)

        # For many objectives, add inner layer for better distribution
        if m >= 5:
            inner_points = self._das_dennis_points(m, p // 2)
            # Scale inner points
            inner_points = 0.5 + 0.5 * inner_points
            ref_points = np.vstack([ref_points, inner_points])

        return ref_points

    def _das_dennis_points(self, n_dim: int, n_partitions: int) -> np.ndarray:
        """
        Generate Das-Dennis points on unit simplex.

        Args:
            n_dim: Number of dimensions (objectives)
            n_partitions: Number of partitions per dimension

        Returns:
            Array of reference points
        """
        def generate_recursive(points, n_dim, n_left, total):
            if n_dim == 1:
                points.append([n_left / total])
            else:
                for i in range(n_left + 1):
                    generate_recursive(
                        points,
                        n_dim - 1,
                        n_left - i,
                        total
                    )
                    if len(points) > 0 and len(points[-1]) == n_dim - 1:
                        points[-1] = [i / total] + points[-1]

        points = []
        generate_recursive(points, n_dim, n_partitions, n_partitions)
        return np.array(points)

    def _initialize_population(self):
        """Initialize population with random solutions."""
        self.population = []
        lower_bounds, upper_bounds = self.problem.bounds

        for _ in range(self.population_size):
            variables = np.random.uniform(lower_bounds, upper_bounds)
            objectives = self.problem.evaluate(variables)
            is_feasible, violation = self.problem.evaluate_constraints(variables)

            solution = MultiObjectiveSolution(
                variables=variables,
                objectives=objectives,
                constraint_violation=violation,
            )
            self.population.append(solution)

        self._update_extreme_points()

    def _create_offspring(self) -> List[MultiObjectiveSolution]:
        """Create offspring using genetic operators."""
        offspring = []

        while len(offspring) < self.population_size:
            # Random selection (no tournament in NSGA-III)
            indices = np.random.choice(
                len(self.population),
                size=2,
                replace=False
            )
            parent1 = self.population[indices[0]]
            parent2 = self.population[indices[1]]

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

            # Create offspring
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

    def _sbx_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (same as NSGA-II)."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        lower_bounds, upper_bounds = self.problem.bounds

        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]

                    rand = np.random.random()
                    beta = 1.0 + (2.0 * (y1 - lower_bounds[i]) / (y2 - y1 + 1e-10))
                    alpha = 2.0 - beta ** (-(self.eta_crossover + 1.0))

                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (self.eta_crossover + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_crossover + 1.0))

                    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                    child1[i] = np.clip(c1, lower_bounds[i], upper_bounds[i])
                    child2[i] = np.clip(c2, lower_bounds[i], upper_bounds[i])

        return child1, child2

    def _polynomial_mutation(self, variables: np.ndarray) -> np.ndarray:
        """Polynomial mutation (same as NSGA-II)."""
        mutated = variables.copy()
        lower_bounds, upper_bounds = self.problem.bounds

        for i in range(len(variables)):
            if np.random.random() < self.mutation_prob:
                delta = min(
                    variables[i] - lower_bounds[i],
                    upper_bounds[i] - variables[i]
                ) / (upper_bounds[i] - lower_bounds[i] + 1e-10)

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
        Reference-point-based environmental selection.

        Algorithm:
        1. Add complete fronts that fit
        2. For last front that doesn't completely fit:
           a. Normalize objectives
           b. Associate solutions to reference points
           c. Select based on niche count
        """
        next_population = []
        front_idx = 0

        # Add complete fronts
        while front_idx < len(fronts) and \
              len(next_population) + len(fronts[front_idx]) <= self.population_size:
            for idx in fronts[front_idx]:
                next_population.append(combined_population[idx])
            front_idx += 1

        # Handle last front with reference point selection
        if len(next_population) < self.population_size and front_idx < len(fronts):
            last_front_indices = fronts[front_idx]
            last_front = [combined_population[i] for i in last_front_indices]

            # Number of solutions to select from last front
            n_select = self.population_size - len(next_population)

            # Normalize objectives
            normalized_pop = self._normalize_objectives(next_population + last_front)

            # Associate to reference points
            associations = self._associate_to_reference_points(
                normalized_pop[len(next_population):]  # Only last front
            )

            # Calculate niche counts (solutions already selected)
            niche_counts = np.zeros(len(self.reference_points))
            if next_population:
                selected_associations = self._associate_to_reference_points(
                    normalized_pop[:len(next_population)]
                )
                for assoc in selected_associations:
                    niche_counts[assoc] += 1

            # Select from last front based on niche preservation
            selected_indices = self._niche_preservation_selection(
                associations,
                niche_counts,
                n_select
            )

            for idx in selected_indices:
                next_population.append(last_front[idx])

        return next_population

    def _normalize_objectives(
        self,
        population: List[MultiObjectiveSolution]
    ) -> np.ndarray:
        """
        Normalize objectives using ideal and nadir points.

        Args:
            population: Population to normalize

        Returns:
            Normalized objective values
        """
        objectives = np.array([sol.objectives for sol in population])

        # Update ideal point (minimum for each objective)
        self.ideal_point = np.min(
            np.vstack([self.ideal_point, objectives]),
            axis=0
        )

        # Estimate nadir point from non-dominated solutions
        non_dominated_mask = np.array([sol.rank == 0 for sol in population])
        if np.any(non_dominated_mask):
            self.nadir_point = np.max(
                objectives[non_dominated_mask],
                axis=0
            )

        # Normalize
        denominator = self.nadir_point - self.ideal_point
        denominator[denominator < 1e-10] = 1.0

        normalized = (objectives - self.ideal_point) / denominator
        return normalized

    def _associate_to_reference_points(
        self,
        normalized_objectives: np.ndarray
    ) -> np.ndarray:
        """
        Associate each solution to closest reference point.

        Uses perpendicular distance from solution to reference line.

        Args:
            normalized_objectives: Normalized objective values

        Returns:
            Array of reference point indices
        """
        n_solutions = len(normalized_objectives)
        distances = np.zeros((n_solutions, len(self.reference_points)))

        for i, ref_point in enumerate(self.reference_points):
            # Calculate perpendicular distance to reference line
            ref_norm = np.linalg.norm(ref_point)
            if ref_norm < 1e-10:
                distances[:, i] = np.linalg.norm(normalized_objectives, axis=1)
            else:
                # Project onto reference line
                projections = np.dot(normalized_objectives, ref_point) / ref_norm
                projections = projections.reshape(-1, 1)
                # Calculate perpendicular distance
                proj_points = projections * ref_point / ref_norm
                distances[:, i] = np.linalg.norm(
                    normalized_objectives - proj_points,
                    axis=1
                )

        # Find closest reference point for each solution
        associations = np.argmin(distances, axis=1)
        return associations

    def _niche_preservation_selection(
        self,
        associations: np.ndarray,
        niche_counts: np.ndarray,
        n_select: int
    ) -> List[int]:
        """
        Select solutions based on niche preservation.

        Selects solutions associated with least crowded reference points.

        Args:
            associations: Reference point associations
            niche_counts: Current niche counts
            n_select: Number of solutions to select

        Returns:
            Indices of selected solutions
        """
        selected = []
        solution_indices = list(range(len(associations)))

        while len(selected) < n_select and solution_indices:
            # Find reference point with minimum niche count
            min_niche_refs = np.where(niche_counts == niche_counts.min())[0]

            for ref_idx in min_niche_refs:
                if len(selected) >= n_select:
                    break

                # Find solutions associated with this reference point
                associated = [
                    i for i in solution_indices
                    if associations[i] == ref_idx
                ]

                if associated:
                    # Select random solution from associated
                    selected_idx = np.random.choice(associated)
                    selected.append(selected_idx)
                    solution_indices.remove(selected_idx)
                    niche_counts[ref_idx] += 1

            # If no solutions for minimum niche refs, increment all
            if not any(associations[solution_indices] == ref_idx
                      for ref_idx in min_niche_refs):
                niche_counts[min_niche_refs] += 1

        return selected

    def _update_extreme_points(self):
        """Update ideal and nadir points from current population."""
        objectives = np.array([sol.objectives for sol in self.population])
        self.ideal_point = np.min(
            np.vstack([self.ideal_point, objectives]),
            axis=0
        )

        # Nadir from non-dominated solutions
        non_dominated = [sol for sol in self.population if sol.rank == 0]
        if non_dominated:
            nd_objectives = np.array([sol.objectives for sol in non_dominated])
            self.nadir_point = np.max(nd_objectives, axis=0)

    def _update_pareto_front(self):
        """Update Pareto front with non-dominated solutions."""
        for solution in self.population:
            if solution.rank == 0:
                self.pareto_front.add(solution)

    def _update_history(self, generation: int):
        """Track optimization history."""
        self.history["n_pareto"].append(self.pareto_front.size())
        self.history["ideal_point"].append(self.ideal_point.copy())

        if self.problem.reference_point is not None:
            from .core import hypervolume
            front_objectives = self.pareto_front.get_objectives()
            if len(front_objectives) > 0:
                hv = hypervolume(front_objectives, self.problem.reference_point)
                self.history["hypervolume"].append(hv)
            else:
                self.history["hypervolume"].append(0.0)