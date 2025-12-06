"""
Ensemble Methods for Robust Optimization

This module implements ensemble optimization strategies that combine
multiple algorithms for improved robustness and performance.

Author: Meshal Alawein
Date: 2025-11-18
"""

import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from Librex.core.interfaces import StandardizedProblem, StandardizedSolution

logger = logging.getLogger(__name__)


@dataclass
class EnsembleMember:
    """Represents a member algorithm in the ensemble."""

    name: str
    weight: float
    solution: Optional[StandardizedSolution] = None
    runtime: float = 0.0
    iterations: int = 0
    diversity_score: float = 0.0


class EnsembleOptimizer:
    """
    Ensemble optimizer that combines multiple algorithms.

    This optimizer runs multiple algorithms in parallel and combines
    their solutions using various voting and consensus strategies.
    """

    def __init__(
        self,
        algorithms: List[str],
        ensemble_size: Optional[int] = None,
        voting_method: str = 'weighted',
        diversity_threshold: float = 0.1
    ):
        """
        Initialize the ensemble optimizer.

        Args:
            algorithms: List of algorithm names to include
            ensemble_size: Size of ensemble (None = use all algorithms)
            voting_method: Method for combining solutions ('weighted', 'majority', 'best')
            diversity_threshold: Minimum diversity required between solutions
        """
        self.algorithms = algorithms
        self.ensemble_size = ensemble_size or len(algorithms)
        self.voting_method = voting_method
        self.diversity_threshold = diversity_threshold

        # Ensemble members
        self.members = []
        self.population_pool = []
        self.consensus_solution = None

    def optimize(
        self,
        problem: StandardizedProblem,
        config: Dict[str, Any],
        n_rounds: int = 3,
        migration_interval: int = 10
    ) -> StandardizedSolution:
        """
        Run ensemble optimization.

        Args:
            problem: Problem to optimize
            config: Configuration for algorithms
            n_rounds: Number of ensemble rounds
            migration_interval: Iterations between population migrations

        Returns:
            Best solution found by the ensemble
        """
        best_solution = None
        best_objective = float('inf')

        for round_idx in range(n_rounds):
            logger.info(f"Ensemble round {round_idx + 1}/{n_rounds}")

            # Initialize or reinitialize ensemble members
            self._initialize_members(problem, config)

            # Run parallel optimization with periodic synchronization
            solutions = self._parallel_optimization(
                problem,
                config,
                migration_interval
            )

            # Combine solutions
            ensemble_solution = self._combine_solutions(
                solutions,
                problem
            )

            # Update best solution
            if ensemble_solution.objective_value < best_objective:
                best_solution = ensemble_solution
                best_objective = ensemble_solution.objective_value

            # Restart strategy for next round
            if round_idx < n_rounds - 1:
                self._adaptive_restart(solutions, problem)

        return best_solution

    def _initialize_members(
        self,
        problem: StandardizedProblem,
        config: Dict[str, Any]
    ):
        """Initialize ensemble members."""
        # Select diverse subset of algorithms
        selected_algorithms = self._select_diverse_algorithms(
            self.algorithms,
            self.ensemble_size
        )

        self.members = []
        for alg_name in selected_algorithms:
            # Calculate initial weight based on historical performance
            weight = self._calculate_initial_weight(alg_name)

            member = EnsembleMember(
                name=alg_name,
                weight=weight
            )
            self.members.append(member)

        logger.info(f"Initialized {len(self.members)} ensemble members")

    def _select_diverse_algorithms(
        self,
        algorithms: List[str],
        n_select: int
    ) -> List[str]:
        """Select diverse subset of algorithms."""
        if n_select >= len(algorithms):
            return algorithms

        # Algorithm categories for diversity
        categories = {
            'evolutionary': ['genetic_algorithm', 'differential_evolution'],
            'swarm': ['particle_swarm', 'ant_colony', 'bee_algorithm'],
            'local_search': ['simulated_annealing', 'tabu_search', 'local_search'],
            'trajectory': ['vns', 'ils', 'grasp'],
            'hybrid': ['memetic', 'scatter_search']
        }

        # Ensure diversity by selecting from different categories
        selected = []
        used_categories = set()

        for alg in algorithms:
            # Find algorithm category
            alg_category = None
            for cat, members in categories.items():
                if alg in members:
                    alg_category = cat
                    break

            # Select if from new category or no category
            if alg_category not in used_categories:
                selected.append(alg)
                if alg_category:
                    used_categories.add(alg_category)

                if len(selected) >= n_select:
                    break

        # Fill remaining with random selection
        remaining = [a for a in algorithms if a not in selected]
        while len(selected) < n_select and remaining:
            idx = np.random.randint(len(remaining))
            selected.append(remaining.pop(idx))

        return selected

    def _calculate_initial_weight(self, algorithm: str) -> float:
        """Calculate initial weight for an algorithm."""
        # Base weights for known algorithms
        base_weights = {
            'genetic_algorithm': 0.8,
            'particle_swarm': 0.7,
            'simulated_annealing': 0.9,
            'tabu_search': 0.75,
            'ant_colony': 0.7,
            'local_search': 0.6,
            'vns': 0.85,
            'ils': 0.8,
            'grasp': 0.75
        }

        return base_weights.get(algorithm, 0.5)

    def _parallel_optimization(
        self,
        problem: StandardizedProblem,
        config: Dict[str, Any],
        migration_interval: int
    ) -> List[StandardizedSolution]:
        """Run algorithms in parallel with periodic migration."""
        solutions = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.members)) as executor:
            # Start all algorithms
            futures = {}
            for member in self.members:
                # Create member-specific config
                member_config = self._create_member_config(
                    member,
                    config,
                    problem
                )

                # Submit optimization task
                future = executor.submit(
                    self._run_member_algorithm,
                    member,
                    problem,
                    member_config,
                    migration_interval
                )
                futures[future] = member

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                member = futures[future]
                try:
                    solution = future.result()
                    member.solution = solution
                    solutions.append(solution)

                    logger.info(
                        f"Member {member.name} completed: "
                        f"objective={solution.objective_value:.6f}"
                    )

                except Exception as e:
                    logger.error(f"Member {member.name} failed: {e}")

        return solutions

    def _run_member_algorithm(
        self,
        member: EnsembleMember,
        problem: StandardizedProblem,
        config: Dict[str, Any],
        migration_interval: int
    ) -> StandardizedSolution:
        """Run a single member algorithm with migration support."""
        start_time = time.time()

        # Get optimization function
        optimize_func = self._get_optimize_function(member.name)

        # Add migration callback if supported
        if member.name in ['genetic_algorithm', 'particle_swarm']:
            config['migration_callback'] = lambda pop: self._migrate_population(
                member.name,
                pop,
                migration_interval
            )

        # Run optimization
        result = optimize_func(problem, config)

        # Update member statistics
        member.runtime = time.time() - start_time
        member.iterations = result.get('iterations', 0)

        # Convert to StandardizedSolution
        if isinstance(result, dict):
            solution = StandardizedSolution(
                vector=result['solution'],
                objective_value=result['objective'],
                is_valid=result.get('is_valid', True),
                metadata={
                    **result.get('metadata', {}),
                    'member': member.name,
                    'runtime': member.runtime
                }
            )
        else:
            solution = result

        return solution

    def _create_member_config(
        self,
        member: EnsembleMember,
        base_config: Dict[str, Any],
        problem: StandardizedProblem
    ) -> Dict[str, Any]:
        """Create configuration for a specific ensemble member."""
        config = base_config.copy()

        # Adjust parameters based on member role
        if member.name == 'genetic_algorithm':
            # Larger population for exploration
            config['population_size'] = min(200, problem.dimension * 10)
            config['crossover_rate'] = 0.9
            config['mutation_rate'] = 0.05

        elif member.name == 'simulated_annealing':
            # Higher temperature for global search
            config['initial_temperature'] = 200.0
            config['cooling_rate'] = 0.98

        elif member.name == 'particle_swarm':
            # Balanced exploration-exploitation
            config['n_particles'] = min(100, problem.dimension * 5)
            config['w'] = 0.7  # Inertia
            config['c1'] = 1.5  # Cognitive
            config['c2'] = 1.5  # Social

        elif member.name == 'tabu_search':
            # Intensification focus
            config['tabu_tenure'] = min(50, problem.dimension)
            config['neighborhood_size'] = 100

        # Add diversity mechanism
        config['diversity_weight'] = member.weight

        return config

    def _migrate_population(
        self,
        algorithm: str,
        population: List[Any],
        interval: int
    ) -> List[Any]:
        """Handle population migration between algorithms."""
        # Check if migration should occur
        if len(self.population_pool) % interval != 0:
            return population

        # Add best individuals to pool
        if algorithm == 'genetic_algorithm':
            # Assuming population is list of (solution, fitness) tuples
            best_individuals = sorted(population, key=lambda x: x[1])[:5]
            self.population_pool.extend(best_individuals)

        # Inject diverse individuals from pool
        if len(self.population_pool) > 10:
            # Replace worst individuals with pool members
            n_replace = min(5, len(population) // 10)
            diverse_individuals = np.random.choice(
                self.population_pool,
                n_replace,
                replace=False
            )

            # Replace worst individuals
            population = sorted(population, key=lambda x: x[1])
            population[-n_replace:] = diverse_individuals

        return population

    def _combine_solutions(
        self,
        solutions: List[StandardizedSolution],
        problem: StandardizedProblem
    ) -> StandardizedSolution:
        """Combine multiple solutions into ensemble solution."""
        if not solutions:
            raise ValueError("No solutions to combine")

        if len(solutions) == 1:
            return solutions[0]

        # Calculate diversity between solutions
        self._calculate_diversity(solutions)

        if self.voting_method == 'best':
            # Simply return the best solution
            return min(solutions, key=lambda s: s.objective_value)

        elif self.voting_method == 'weighted':
            # Weighted combination based on quality and diversity
            return self._weighted_combination(solutions, problem)

        elif self.voting_method == 'majority':
            # Majority voting for discrete problems
            return self._majority_voting(solutions, problem)

        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")

    def _calculate_diversity(self, solutions: List[StandardizedSolution]):
        """Calculate diversity scores for solutions."""
        n_solutions = len(solutions)
        if n_solutions < 2:
            return

        # Calculate pairwise distances
        for i, sol_i in enumerate(solutions):
            distances = []
            for j, sol_j in enumerate(solutions):
                if i != j:
                    # Hamming distance for discrete, Euclidean for continuous
                    if isinstance(sol_i.vector[0], (int, np.integer)):
                        distance = np.sum(sol_i.vector != sol_j.vector) / len(sol_i.vector)
                    else:
                        distance = np.linalg.norm(sol_i.vector - sol_j.vector)
                    distances.append(distance)

            # Average distance to other solutions
            if self.members and i < len(self.members):
                self.members[i].diversity_score = np.mean(distances)

    def _weighted_combination(
        self,
        solutions: List[StandardizedSolution],
        problem: StandardizedProblem
    ) -> StandardizedSolution:
        """Create weighted combination of solutions."""
        # Calculate solution weights based on quality and diversity
        weights = []
        for i, sol in enumerate(solutions):
            # Quality component (inverse of objective for minimization)
            quality_weight = 1.0 / (1.0 + sol.objective_value)

            # Diversity component
            if i < len(self.members):
                diversity_weight = self.members[i].diversity_score
            else:
                diversity_weight = 0.5

            # Member weight
            if i < len(self.members):
                member_weight = self.members[i].weight
            else:
                member_weight = 0.5

            # Combined weight
            weight = quality_weight * (1 + diversity_weight) * member_weight
            weights.append(weight)

        # Normalize weights
        weights = np.array(weights)
        weights /= np.sum(weights)

        # For discrete problems, select probabilistically
        if problem.is_discrete:
            selected_idx = np.random.choice(len(solutions), p=weights)
            return solutions[selected_idx]

        # For continuous problems, weighted average
        combined_vector = np.zeros_like(solutions[0].vector)
        for sol, w in zip(solutions, weights):
            combined_vector += w * sol.vector

        # Evaluate combined solution
        combined_solution = StandardizedSolution(
            vector=combined_vector,
            objective_value=problem.evaluate(combined_vector),
            is_valid=problem.is_valid(combined_vector),
            metadata={'method': 'ensemble_weighted'}
        )

        return combined_solution

    def _majority_voting(
        self,
        solutions: List[StandardizedSolution],
        problem: StandardizedProblem
    ) -> StandardizedSolution:
        """Majority voting for discrete optimization."""
        if not problem.is_discrete:
            # Fall back to weighted for continuous
            return self._weighted_combination(solutions, problem)

        n_vars = len(solutions[0].vector)
        consensus_vector = np.zeros(n_vars, dtype=solutions[0].vector.dtype)

        # Vote on each variable
        for var_idx in range(n_vars):
            # Collect votes for this variable
            votes = {}
            for i, sol in enumerate(solutions):
                value = sol.vector[var_idx]
                if value not in votes:
                    votes[value] = 0

                # Weight vote by solution quality
                weight = 1.0 / (1.0 + sol.objective_value)
                if i < len(self.members):
                    weight *= self.members[i].weight

                votes[value] += weight

            # Select value with most votes
            consensus_vector[var_idx] = max(votes.keys(), key=lambda k: votes[k])

        # Create consensus solution
        consensus_solution = StandardizedSolution(
            vector=consensus_vector,
            objective_value=problem.evaluate(consensus_vector),
            is_valid=problem.is_valid(consensus_vector),
            metadata={'method': 'ensemble_majority'}
        )

        return consensus_solution

    def _adaptive_restart(
        self,
        solutions: List[StandardizedSolution],
        problem: StandardizedProblem
    ):
        """Adaptive restart strategy for next round."""
        # Update member weights based on performance
        for i, member in enumerate(self.members):
            if i < len(solutions) and solutions[i]:
                # Performance-based weight update
                performance = 1.0 / (1.0 + solutions[i].objective_value)
                diversity = member.diversity_score

                # Exponential moving average
                alpha = 0.3
                member.weight = (1 - alpha) * member.weight + alpha * performance * (1 + diversity)

        # Normalize weights
        total_weight = sum(m.weight for m in self.members)
        for member in self.members:
            member.weight /= total_weight

        # Identify underperforming members for replacement
        threshold = 0.5 / len(self.members)  # Below average threshold
        underperformers = [m for m in self.members if m.weight < threshold]

        if underperformers:
            logger.info(f"Replacing {len(underperformers)} underperforming members")

            # Replace with new algorithms or variants
            for member in underperformers:
                # Find replacement from unused algorithms
                used_algorithms = {m.name for m in self.members}
                available = [a for a in self.algorithms if a not in used_algorithms]

                if available:
                    new_algorithm = np.random.choice(available)
                    member.name = new_algorithm
                    member.weight = 0.5  # Reset weight
                    logger.info(f"Replaced {member.name} with {new_algorithm}")

    def _get_optimize_function(self, algorithm: str) -> Callable:
        """Get the optimization function for an algorithm."""
        # Import optimization functions dynamically
        if algorithm == 'simulated_annealing':
            from Librex.methods.baselines.simulated_annealing import (
                simulated_annealing_optimize,
            )
            return simulated_annealing_optimize
        elif algorithm == 'genetic_algorithm':
            from Librex.methods.baselines.genetic_algorithm import (
                genetic_algorithm_optimize,
            )
            return genetic_algorithm_optimize
        elif algorithm == 'tabu_search':
            from Librex.methods.baselines.tabu_search import tabu_search_optimize
            return tabu_search_optimize
        elif algorithm == 'ant_colony':
            from Librex.methods.advanced.aco import ant_colony_optimize
            return ant_colony_optimize
        elif algorithm == 'particle_swarm':
            from Librex.methods.advanced.pso import particle_swarm_optimize
            return particle_swarm_optimize
        elif algorithm == 'local_search':
            from Librex.methods.baselines.local_search import local_search_optimize
            return local_search_optimize
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get summary of ensemble performance."""
        summary = {
            'n_members': len(self.members),
            'voting_method': self.voting_method,
            'members': []
        }

        for member in self.members:
            member_info = {
                'name': member.name,
                'weight': member.weight,
                'diversity_score': member.diversity_score,
                'runtime': member.runtime,
                'iterations': member.iterations
            }
            if member.solution:
                member_info['objective'] = member.solution.objective_value

            summary['members'].append(member_info)

        # Sort members by weight
        summary['members'].sort(key=lambda x: x['weight'], reverse=True)

        return summary