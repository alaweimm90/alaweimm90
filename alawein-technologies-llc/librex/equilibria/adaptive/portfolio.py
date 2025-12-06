"""
Algorithm Portfolio Manager for Adaptive Learning System

This module manages a portfolio of optimization algorithms, dynamically selecting
and allocating resources based on problem characteristics and historical performance.

Author: Meshal Alawein
Date: 2025-11-18
"""

import concurrent.futures
import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from Librex.core.interfaces import StandardizedProblem, StandardizedSolution

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmInstance:
    """Represents a single algorithm instance in the portfolio."""

    name: str
    optimize_func: Callable
    config: Dict[str, Any]
    priority: float
    time_budget: float
    evaluation_budget: int
    performance_history: List[float]
    active: bool = True


@dataclass
class PortfolioResult:
    """Result from portfolio execution."""

    best_solution: StandardizedSolution
    best_method: str
    all_results: Dict[str, StandardizedSolution]
    runtime_allocation: Dict[str, float]
    performance_metrics: Dict[str, Dict[str, float]]


class AlgorithmPortfolioManager:
    """
    Manages a portfolio of optimization algorithms with dynamic resource allocation.

    This manager enables:
    - Parallel execution of multiple algorithms
    - Dynamic time budget allocation based on performance
    - Early stopping of underperforming methods
    - Adaptive algorithm selection based on problem characteristics
    """

    def __init__(
        self,
        algorithms: Optional[List[str]] = None,
        max_parallel: int = 4,
        adaptive_allocation: bool = True
    ):
        """
        Initialize the portfolio manager.

        Args:
            algorithms: List of algorithm names to include in portfolio
            max_parallel: Maximum number of parallel algorithm executions
            adaptive_allocation: Whether to use adaptive resource allocation
        """
        self.algorithms = algorithms or [
            'simulated_annealing',
            'genetic_algorithm',
            'tabu_search',
            'ant_colony',
            'particle_swarm'
        ]
        self.max_parallel = max_parallel
        self.adaptive_allocation = adaptive_allocation

        # Performance tracking
        self.algorithm_stats = {alg: {
            'successes': 0,
            'failures': 0,
            'avg_runtime': 0.0,
            'avg_quality': 0.0,
            'total_evaluations': 0
        } for alg in self.algorithms}

        self.lock = Lock()
        self.active_algorithms: Set[str] = set()

    def optimize_portfolio(
        self,
        problem: StandardizedProblem,
        total_time_budget: float = 60.0,
        total_evaluation_budget: int = 10000,
        early_stopping_rounds: int = 5
    ) -> PortfolioResult:
        """
        Execute portfolio optimization with multiple algorithms.

        Args:
            problem: The standardized problem to optimize
            total_time_budget: Total time budget in seconds
            total_evaluation_budget: Total evaluation budget across all algorithms
            early_stopping_rounds: Number of rounds without improvement before stopping

        Returns:
            PortfolioResult with best solution and performance metrics
        """
        # Initialize algorithm instances
        instances = self._initialize_instances(
            problem,
            total_time_budget,
            total_evaluation_budget
        )

        # Execute algorithms in parallel
        start_time = time.time()
        results = {}
        runtime_allocation = {}
        performance_metrics = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # Submit initial algorithms
            futures = {}
            for instance in instances[:self.max_parallel]:
                future = executor.submit(
                    self._run_algorithm,
                    instance,
                    problem
                )
                futures[future] = instance

            # Process completed algorithms and submit new ones
            completed = 0
            while futures and (time.time() - start_time < total_time_budget):
                # Wait for at least one to complete
                done, pending = concurrent.futures.wait(
                    futures.keys(),
                    timeout=1.0,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )

                for future in done:
                    instance = futures.pop(future)

                    try:
                        result = future.result()
                        results[instance.name] = result
                        runtime_allocation[instance.name] = time.time() - start_time

                        # Update performance metrics
                        performance_metrics[instance.name] = self._compute_metrics(
                            result, instance, problem
                        )

                        # Update algorithm statistics
                        self._update_statistics(instance.name, result)

                        completed += 1
                        logger.info(
                            f"Algorithm {instance.name} completed: "
                            f"objective={result.objective_value:.6f}"
                        )

                    except Exception as e:
                        logger.error(f"Algorithm {instance.name} failed: {e}")
                        self.algorithm_stats[instance.name]['failures'] += 1

                    # Submit next algorithm if available and time permits
                    if completed < len(instances) and (time.time() - start_time < total_time_budget * 0.8):
                        next_instance = instances[completed]
                        if self.adaptive_allocation:
                            # Adjust time budget based on current performance
                            next_instance.time_budget = self._adjust_time_budget(
                                next_instance,
                                results,
                                total_time_budget - (time.time() - start_time)
                            )

                        future = executor.submit(
                            self._run_algorithm,
                            next_instance,
                            problem
                        )
                        futures[future] = next_instance

                # Perform early stopping check
                if self._should_stop_early(results, early_stopping_rounds):
                    logger.info("Early stopping triggered - convergence detected")
                    executor.shutdown(wait=False)
                    break

        # Select best result
        if not results:
            raise RuntimeError("No algorithms completed successfully")

        best_method, best_solution = min(
            results.items(),
            key=lambda x: x[1].objective_value
        )

        return PortfolioResult(
            best_solution=best_solution,
            best_method=best_method,
            all_results=results,
            runtime_allocation=runtime_allocation,
            performance_metrics=performance_metrics
        )

    def _initialize_instances(
        self,
        problem: StandardizedProblem,
        total_time_budget: float,
        total_evaluation_budget: int
    ) -> List[AlgorithmInstance]:
        """Initialize algorithm instances with resource allocation."""
        instances = []

        # Calculate initial resource allocation
        if self.adaptive_allocation:
            allocations = self._compute_adaptive_allocation(problem)
        else:
            # Equal allocation
            n_algorithms = len(self.algorithms)
            allocations = {alg: 1.0 / n_algorithms for alg in self.algorithms}

        for alg_name in self.algorithms:
            # Get optimization function
            optimize_func = self._get_optimize_function(alg_name)

            # Allocate resources
            time_budget = total_time_budget * allocations[alg_name]
            eval_budget = int(total_evaluation_budget * allocations[alg_name])

            # Configure algorithm
            config = self._get_algorithm_config(alg_name, problem)
            config['max_evaluations'] = eval_budget
            config['max_time'] = time_budget

            instances.append(AlgorithmInstance(
                name=alg_name,
                optimize_func=optimize_func,
                config=config,
                priority=allocations[alg_name],
                time_budget=time_budget,
                evaluation_budget=eval_budget,
                performance_history=[]
            ))

        # Sort by priority
        instances.sort(key=lambda x: x.priority, reverse=True)
        return instances

    def _compute_adaptive_allocation(
        self,
        problem: StandardizedProblem
    ) -> Dict[str, float]:
        """Compute adaptive resource allocation based on historical performance."""
        allocations = {}
        total_score = 0.0

        for alg_name in self.algorithms:
            stats = self.algorithm_stats[alg_name]

            # Compute performance score
            if stats['successes'] + stats['failures'] > 0:
                success_rate = stats['successes'] / (stats['successes'] + stats['failures'])
                quality_score = stats['avg_quality'] if stats['avg_quality'] > 0 else 0.5
                efficiency_score = 1.0 / (1.0 + stats['avg_runtime'] / 60.0)  # Normalize runtime

                # Weighted combination
                score = (
                    0.4 * success_rate +
                    0.4 * quality_score +
                    0.2 * efficiency_score
                )
            else:
                # Default score for new algorithms
                score = 0.5

            # Add exploration bonus for less-used algorithms
            n_runs = stats['successes'] + stats['failures']
            exploration_bonus = 0.1 * np.exp(-n_runs / 10.0)
            score += exploration_bonus

            allocations[alg_name] = score
            total_score += score

        # Normalize to sum to 1
        if total_score > 0:
            for alg_name in allocations:
                allocations[alg_name] /= total_score
        else:
            # Equal allocation if no history
            n_algorithms = len(self.algorithms)
            allocations = {alg: 1.0 / n_algorithms for alg in self.algorithms}

        return allocations

    def _run_algorithm(
        self,
        instance: AlgorithmInstance,
        problem: StandardizedProblem
    ) -> StandardizedSolution:
        """Run a single algorithm instance."""
        with self.lock:
            self.active_algorithms.add(instance.name)

        try:
            start_time = time.time()

            # Run optimization
            result = instance.optimize_func(problem, instance.config)

            # Convert to StandardizedSolution
            if isinstance(result, dict):
                solution = StandardizedSolution(
                    vector=result['solution'],
                    objective_value=result['objective'],
                    is_valid=result.get('is_valid', True),
                    metadata={
                        **result.get('metadata', {}),
                        'runtime': time.time() - start_time,
                        'method': instance.name
                    }
                )
            else:
                solution = result

            return solution

        finally:
            with self.lock:
                self.active_algorithms.discard(instance.name)

    def _adjust_time_budget(
        self,
        instance: AlgorithmInstance,
        current_results: Dict[str, StandardizedSolution],
        remaining_time: float
    ) -> float:
        """Adjust time budget based on current performance."""
        if not current_results:
            return instance.time_budget

        # Find current best
        best_objective = min(r.objective_value for r in current_results.values())

        # Estimate potential improvement for this algorithm
        stats = self.algorithm_stats[instance.name]
        if stats['avg_quality'] > 0:
            expected_quality = stats['avg_quality']
            if expected_quality < best_objective:
                # High potential - allocate more time
                return min(instance.time_budget * 1.5, remaining_time * 0.5)
            else:
                # Low potential - reduce time
                return min(instance.time_budget * 0.5, remaining_time * 0.2)

        return min(instance.time_budget, remaining_time * 0.3)

    def _should_stop_early(
        self,
        results: Dict[str, StandardizedSolution],
        rounds: int
    ) -> bool:
        """Check if we should stop early due to convergence."""
        if len(results) < rounds:
            return False

        # Get last n objectives
        objectives = [r.objective_value for r in results.values()]
        if len(objectives) < rounds:
            return False

        recent_objectives = objectives[-rounds:]

        # Check for convergence (small variance)
        variance = np.var(recent_objectives)
        mean = np.mean(recent_objectives)

        if mean > 0:
            coefficient_of_variation = np.sqrt(variance) / mean
            return coefficient_of_variation < 0.01
        return False

    def _compute_metrics(
        self,
        solution: StandardizedSolution,
        instance: AlgorithmInstance,
        problem: StandardizedProblem
    ) -> Dict[str, float]:
        """Compute performance metrics for a solution."""
        return {
            'objective_value': solution.objective_value,
            'runtime': solution.metadata.get('runtime', 0.0),
            'evaluations': solution.metadata.get('n_evaluations', 0),
            'is_valid': float(solution.is_valid),
            'time_efficiency': (
                solution.objective_value / solution.metadata.get('runtime', 1.0)
                if solution.metadata.get('runtime', 0) > 0 else 0.0
            ),
            'eval_efficiency': (
                solution.objective_value / solution.metadata.get('n_evaluations', 1.0)
                if solution.metadata.get('n_evaluations', 0) > 0 else 0.0
            )
        }

    def _update_statistics(self, algorithm: str, solution: StandardizedSolution):
        """Update algorithm statistics based on solution."""
        stats = self.algorithm_stats[algorithm]

        # Update success/failure counts
        if solution.is_valid:
            stats['successes'] += 1
        else:
            stats['failures'] += 1

        # Update averages (exponential moving average)
        alpha = 0.1  # Smoothing factor
        runtime = solution.metadata.get('runtime', 0.0)
        stats['avg_runtime'] = (1 - alpha) * stats['avg_runtime'] + alpha * runtime

        # Normalize quality to [0, 1] range (assuming minimization)
        # This is a simplified normalization - in practice, use problem-specific bounds
        quality = 1.0 / (1.0 + solution.objective_value)
        stats['avg_quality'] = (1 - alpha) * stats['avg_quality'] + alpha * quality

        stats['total_evaluations'] += solution.metadata.get('n_evaluations', 0)

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
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _get_algorithm_config(
        self,
        algorithm: str,
        problem: StandardizedProblem
    ) -> Dict[str, Any]:
        """Get default configuration for an algorithm."""
        # Base configuration
        config = {
            'verbose': False,
            'seed': np.random.randint(0, 2**32)
        }

        # Algorithm-specific defaults
        if algorithm == 'simulated_annealing':
            config.update({
                'initial_temperature': 100.0,
                'cooling_rate': 0.95,
                'min_temperature': 0.01
            })
        elif algorithm == 'genetic_algorithm':
            config.update({
                'population_size': min(100, problem.dimension * 10),
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'elitism_ratio': 0.1
            })
        elif algorithm == 'tabu_search':
            config.update({
                'tabu_tenure': min(20, problem.dimension // 2),
                'neighborhood_size': 50
            })
        elif algorithm == 'ant_colony':
            config.update({
                'n_ants': min(50, problem.dimension * 2),
                'alpha': 1.0,  # Pheromone importance
                'beta': 2.0,   # Heuristic importance
                'evaporation_rate': 0.1
            })
        elif algorithm == 'particle_swarm':
            config.update({
                'n_particles': min(50, problem.dimension * 3),
                'w': 0.7,  # Inertia weight
                'c1': 2.0,  # Cognitive parameter
                'c2': 2.0   # Social parameter
            })

        return config

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the portfolio."""
        summary = {
            'algorithms': self.algorithms,
            'statistics': self.algorithm_stats,
            'active_algorithms': list(self.active_algorithms),
            'total_runs': sum(
                stats['successes'] + stats['failures']
                for stats in self.algorithm_stats.values()
            )
        }

        # Compute rankings
        rankings = []
        for alg, stats in self.algorithm_stats.items():
            if stats['successes'] + stats['failures'] > 0:
                success_rate = stats['successes'] / (stats['successes'] + stats['failures'])
                rankings.append((alg, success_rate * stats['avg_quality']))

        rankings.sort(key=lambda x: x[1], reverse=True)
        summary['rankings'] = rankings

        return summary