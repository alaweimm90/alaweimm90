"""
Core optimization interface for Librex

This module provides the main optimize() function that serves as the
universal entry point for optimization across all domains and methods.
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np

from Librex.core.interfaces import (
    StandardizedProblem,
    StandardizedSolution,
    UniversalOptimizationInterface,
)

logger = logging.getLogger(__name__)


def optimize(
    problem: Union[Dict[str, Any], StandardizedProblem],
    adapter: Optional[UniversalOptimizationInterface] = None,
    method: str = 'simulated_annealing',
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Universal optimization function - main entry point for Librex

    Args:
        problem: Either a domain-specific problem dict, StandardizedProblem,
                or MultiObjectiveProblem for multi-objective optimization
        adapter: Domain adapter (required if problem is dict)
        method: Optimization method name. Options:
            - 'auto': AI-powered automatic method selection
            - 'adaptive': Adaptive learning with online method selection
            Baseline methods:
            - 'random_search': Baseline random sampling
            - 'simulated_annealing': Simulated annealing metaheuristic
            - 'local_search': Hill climbing with restarts
            - 'genetic_algorithm': Population-based evolutionary algorithm
            - 'tabu_search': Memory-based local search
            Advanced methods:
            - 'ant_colony': Ant Colony Optimization (ACO)
            - 'particle_swarm': Particle Swarm Optimization (PSO)
            - 'variable_neighborhood': Variable Neighborhood Search (VNS)
            - 'iterated_local_search': Iterated Local Search (ILS)
            - 'grasp': Greedy Randomized Adaptive Search Procedure
            Multi-objective methods:
            - 'nsga2': Non-dominated Sorting Genetic Algorithm II
            - 'nsga3': Reference-point based NSGA for many objectives
            - 'moead': Multi-Objective Evolutionary Algorithm based on Decomposition
        config: Method-specific configuration parameters

    Returns:
        dict: Optimization result with keys:
            - 'solution': Best solution found
            - 'objective': Objective value
            - 'is_valid': Whether solution satisfies constraints
            - 'iterations': Number of iterations performed
            - 'convergence': Convergence information
            - 'metadata': Additional method-specific metadata

    Raises:
        ValueError: If adapter is None when problem is dict
        NotImplementedError: If method is not implemented

    Example:
        >>> from Librex import optimize
        >>> from Librex.adapters.qap import QAPAdapter
        >>> import numpy as np
        >>>
        >>> problem = {
        ...     'flow_matrix': np.array([[0, 5], [5, 0]]),
        ...     'distance_matrix': np.array([[0, 8], [8, 0]])
        ... }
        >>> adapter = QAPAdapter()
        >>> result = optimize(problem, adapter, method='simulated_annealing')
        >>> print(result['objective'])
    """
    # Set default config
    if config is None:
        config = {}

    # Encode problem if needed
    if isinstance(problem, dict):
        if adapter is None:
            raise ValueError(
                "adapter must be provided when problem is a dictionary. "
                "Use a domain adapter like QAPAdapter or TSPAdapter."
            )
        standardized_problem = adapter.encode_problem(problem)
    else:
        standardized_problem = problem

    # Handle automatic method selection
    if method == 'auto':
        try:
            from Librex.ai import MethodSelector
            selector = MethodSelector()

            # Get AI recommendation
            recommended_method, recommended_config, confidence = selector.recommend_method(
                problem=problem if isinstance(problem, dict) else None,
                adapter=adapter,
                standardized_problem=standardized_problem if not isinstance(problem, dict) else None
            )

            logger.info(
                f"AI selector recommended: {recommended_method} "
                f"(confidence: {confidence:.2%})"
            )

            # Use recommended method and merge configs
            method = recommended_method
            if config is None:
                config = recommended_config
            else:
                # Merge user config with recommended config (user config takes priority)
                config = {**recommended_config, **config}

        except Exception as e:
            logger.warning(f"AI selector failed: {e}. Falling back to simulated_annealing")
            method = 'simulated_annealing'

    # Handle adaptive learning mode
    elif method == 'adaptive':
        try:
            from Librex.adaptive import (
                AlgorithmPortfolioManager,
                OnlineLearner,
                PerformanceDatabase,
            )

            # Extract learning configuration
            learning_config = config.get('learning_config', {})
            learning_mode = learning_config.get('mode', 'online')  # 'online', 'portfolio', 'ensemble'
            time_budget = learning_config.get('time_budget', 300)
            n_parallel = learning_config.get('n_parallel', 4)

            if learning_mode == 'online':
                # Use online learning with multi-armed bandits
                strategy = learning_config.get('strategy', 'ucb1')  # 'ucb1', 'thompson', 'exp3'
                learner = OnlineLearner(
                    algorithms=['simulated_annealing', 'genetic_algorithm', 'tabu_search'],
                    strategy=strategy,
                    contextual=True
                )

                # Extract problem features
                db = PerformanceDatabase()
                if isinstance(problem, dict) and adapter:
                    problem_features = db.extract_features(problem, adapter)
                else:
                    problem_features = np.random.randn(10)  # Default features

                # Select method and configuration
                selected_method, selected_config = learner.select_method(
                    problem_features,
                    time_budget=time_budget
                )

                # Run selected method
                logger.info(f"Adaptive learner selected: {selected_method}")
                method = selected_method
                config = {**config, **selected_config}

            elif learning_mode == 'portfolio':
                # Use portfolio-based optimization
                portfolio = AlgorithmPortfolioManager(
                    algorithms=['simulated_annealing', 'genetic_algorithm', 'tabu_search',
                               'ant_colony', 'particle_swarm'],
                    max_parallel=n_parallel,
                    adaptive_allocation=True
                )

                # Run portfolio optimization
                portfolio_result = portfolio.optimize_portfolio(
                    standardized_problem,
                    total_time_budget=time_budget
                )

                # Return portfolio result
                return {
                    'solution': portfolio_result.best_solution.vector,
                    'objective': portfolio_result.best_solution.objective_value,
                    'is_valid': portfolio_result.best_solution.is_valid,
                    'iterations': sum(m.get('iterations', 0) for m in
                                     portfolio_result.performance_metrics.values()),
                    'convergence': {'method': portfolio_result.best_method},
                    'metadata': {
                        'mode': 'adaptive_portfolio',
                        'all_results': {
                            k: v.objective_value
                            for k, v in portfolio_result.all_results.items()
                        },
                        'runtime_allocation': portfolio_result.runtime_allocation
                    }
                }

            elif learning_mode == 'ensemble':
                # Use ensemble optimization
                from Librex.adaptive import EnsembleOptimizer

                ensemble = EnsembleOptimizer(
                    algorithms=['simulated_annealing', 'genetic_algorithm', 'tabu_search'],
                    voting_method=learning_config.get('voting_method', 'weighted')
                )

                # Run ensemble optimization
                ensemble_solution = ensemble.optimize(
                    standardized_problem,
                    config,
                    n_rounds=learning_config.get('n_rounds', 3)
                )

                return {
                    'solution': ensemble_solution.vector,
                    'objective': ensemble_solution.objective_value,
                    'is_valid': ensemble_solution.is_valid,
                    'iterations': ensemble_solution.metadata.get('iterations', 0),
                    'convergence': {},
                    'metadata': {
                        'mode': 'adaptive_ensemble',
                        'ensemble_summary': ensemble.get_ensemble_summary()
                    }
                }

        except Exception as e:
            logger.warning(f"Adaptive learning failed: {e}. Falling back to simulated_annealing")
            method = 'simulated_annealing'

    # Select and run optimization method
    if method == 'random_search':
        from Librex.methods.baselines.random_search import random_search_optimize
        result = random_search_optimize(standardized_problem, config)
    elif method == 'simulated_annealing':
        from Librex.methods.baselines.simulated_annealing import (
            simulated_annealing_optimize,
        )
        result = simulated_annealing_optimize(standardized_problem, config)
    elif method == 'local_search':
        from Librex.methods.baselines.local_search import local_search_optimize
        result = local_search_optimize(standardized_problem, config)
    elif method == 'genetic_algorithm':
        from Librex.methods.baselines.genetic_algorithm import (
            genetic_algorithm_optimize,
        )
        result = genetic_algorithm_optimize(standardized_problem, config)
    elif method == 'tabu_search':
        from Librex.methods.baselines.tabu_search import tabu_search_optimize
        result = tabu_search_optimize(standardized_problem, config)
    # Advanced methods
    elif method == 'ant_colony':
        from Librex.methods.advanced.aco import ant_colony_optimize
        result = ant_colony_optimize(standardized_problem, config)
    elif method == 'particle_swarm':
        from Librex.methods.advanced.pso import particle_swarm_optimize
        result = particle_swarm_optimize(standardized_problem, config)
    elif method == 'variable_neighborhood':
        from Librex.methods.advanced.vns import variable_neighborhood_search_optimize
        result = variable_neighborhood_search_optimize(standardized_problem, config)
    elif method == 'iterated_local_search':
        from Librex.methods.advanced.ils import iterated_local_search_optimize
        result = iterated_local_search_optimize(standardized_problem, config)
    elif method == 'grasp':
        from Librex.methods.advanced.grasp import grasp_optimize
        result = grasp_optimize(standardized_problem, config)
    # Multi-objective methods
    elif method in ['nsga2', 'nsga3', 'moead']:
        from Librex.multi_objective import (
            MultiObjectiveProblem,
            NSGA2Optimizer,
            NSGA3Optimizer,
            MOEADOptimizer,
        )
        from Librex.multi_objective.moead import DecompositionMethod

        # Check if problem is multi-objective
        if not isinstance(standardized_problem, MultiObjectiveProblem):
            # Try to detect if it's a multi-objective problem
            if hasattr(standardized_problem, 'n_objectives') and standardized_problem.n_objectives > 1:
                mo_problem = standardized_problem
            else:
                raise ValueError(
                    f"Method '{method}' requires a MultiObjectiveProblem instance. "
                    "For single-objective problems, use other methods."
                )
        else:
            mo_problem = standardized_problem

        # Run the selected multi-objective algorithm
        if method == 'nsga2':
            optimizer = NSGA2Optimizer(
                mo_problem,
                population_size=config.get('population_size', 100),
                n_generations=config.get('n_generations', 100),
                crossover_prob=config.get('crossover_prob', 0.9),
                mutation_prob=config.get('mutation_prob'),
                eta_crossover=config.get('eta_crossover', 15.0),
                eta_mutation=config.get('eta_mutation', 20.0),
                seed=config.get('seed')
            )
        elif method == 'nsga3':
            optimizer = NSGA3Optimizer(
                mo_problem,
                population_size=config.get('population_size'),
                n_generations=config.get('n_generations', 100),
                n_partitions=config.get('n_partitions', 12),
                crossover_prob=config.get('crossover_prob', 0.9),
                mutation_prob=config.get('mutation_prob'),
                eta_crossover=config.get('eta_crossover', 30.0),
                eta_mutation=config.get('eta_mutation', 20.0),
                seed=config.get('seed')
            )
        else:  # moead
            decomposition_str = config.get('decomposition', 'tchebycheff')
            decomposition_map = {
                'weighted_sum': DecompositionMethod.WEIGHTED_SUM,
                'tchebycheff': DecompositionMethod.TCHEBYCHEFF,
                'pbi': DecompositionMethod.PBI
            }
            decomposition = decomposition_map.get(decomposition_str, DecompositionMethod.TCHEBYCHEFF)

            optimizer = MOEADOptimizer(
                mo_problem,
                population_size=config.get('population_size', 100),
                n_generations=config.get('n_generations', 100),
                decomposition=decomposition,
                n_neighbors=config.get('n_neighbors', 20),
                neighbor_selection_prob=config.get('neighbor_selection_prob', 0.9),
                max_replacements=config.get('max_replacements', 2),
                crossover_prob=config.get('crossover_prob', 1.0),
                mutation_prob=config.get('mutation_prob'),
                eta_crossover=config.get('eta_crossover', 20.0),
                eta_mutation=config.get('eta_mutation', 20.0),
                pbi_theta=config.get('pbi_theta', 5.0),
                seed=config.get('seed')
            )

        # Run optimization
        pareto_front = optimizer.optimize()

        # Return multi-objective result
        result = {
            'pareto_front': pareto_front,  # ParetoFront object
            'solutions': [
                {
                    'variables': sol.variables,
                    'objectives': sol.objectives,
                    'constraint_violation': sol.constraint_violation
                }
                for sol in pareto_front.solutions
            ],
            'n_solutions': pareto_front.size(),
            'objectives_array': pareto_front.get_objectives(),
            'metadata': {
                'method': method,
                'history': optimizer.history,
                'multi_objective': True
            }
        }
    else:
        raise NotImplementedError(
            f"Method '{method}' is not implemented. "
            f"Available methods: auto, adaptive, random_search, simulated_annealing, "
            f"local_search, genetic_algorithm, tabu_search, ant_colony, "
            f"particle_swarm, variable_neighborhood, iterated_local_search, grasp, "
            f"nsga2, nsga3, moead"
        )

    # Decode solution if adapter provided
    if adapter is not None and 'solution' in result:
        solution_obj = StandardizedSolution(
            vector=result['solution'],
            objective_value=result['objective'],
            is_valid=result.get('is_valid', True),
            metadata=result.get('metadata', {})
        )
        result['solution'] = adapter.decode_solution(solution_obj)

    logger.info(
        f"Optimization complete: method={method}, "
        f"objective={result.get('objective', 'N/A')}, "
        f"iterations={result.get('iterations', 'N/A')}"
    )

    return result


__all__ = ['optimize']
