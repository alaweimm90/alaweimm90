"""
Core benchmarking functionality

Provides benchmark execution and comparison utilities.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from Librex.core.interfaces import StandardizedProblem
from Librex.benchmarking.metrics import (
    compute_optimality_gap,
    compute_solution_quality,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""

    method: str
    objective_value: float
    runtime_seconds: float
    iterations: int
    solution: np.ndarray
    is_valid: bool
    optimality_gap: Optional[float] = None
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        gap_str = f"{self.optimality_gap:.2%}" if self.optimality_gap is not None else "N/A"
        return (
            f"BenchmarkResult(method={self.method}, "
            f"objective={self.objective_value:.2f}, "
            f"time={self.runtime_seconds:.4f}s, "
            f"gap={gap_str})"
        )


def benchmark_method(
    problem: Any,
    adapter: Any,
    method: str,
    config: Optional[Dict[str, Any]] = None,
    known_optimal: Optional[float] = None,
    runs: int = 1,
) -> BenchmarkResult:
    """
    Benchmark a single optimization method

    Args:
        problem: Problem instance
        adapter: Domain adapter
        method: Method name ('simulated_annealing', 'genetic_algorithm', etc.)
        config: Method configuration
        known_optimal: Known optimal objective value (for gap calculation)
        runs: Number of runs to average over

    Returns:
        BenchmarkResult with performance metrics

    Example:
        >>> from Librex import optimize
        >>> from Librex.adapters.qap import QAPAdapter
        >>> from Librex.benchmarking import benchmark_method
        >>>
        >>> problem = {'flow_matrix': flow, 'distance_matrix': distance}
        >>> adapter = QAPAdapter()
        >>>
        >>> result = benchmark_method(
        ...     problem, adapter, 'simulated_annealing',
        ...     config={'iterations': 1000, 'seed': 42},
        ...     known_optimal=578.0  # Known QAPLIB optimal
        ... )
        >>> print(f"Gap: {result.optimality_gap:.2%}")
    """
    from Librex import optimize

    if config is None:
        config = {}

    results = []
    runtimes = []

    for run_idx in range(runs):
        # Set seed for reproducibility if not already set
        run_config = config.copy()
        if 'seed' not in run_config:
            run_config['seed'] = 42 + run_idx

        # Time the optimization
        start_time = time.perf_counter()
        result = optimize(problem, adapter, method=method, config=run_config)
        end_time = time.perf_counter()

        runtime = end_time - start_time
        runtimes.append(runtime)
        results.append(result)

    # Use best result across runs
    best_idx = min(range(runs), key=lambda i: results[i]['objective'])
    best_result = results[best_idx]
    avg_runtime = np.mean(runtimes)

    # Compute optimality gap if known optimal provided
    gap = None
    if known_optimal is not None:
        gap = compute_optimality_gap(best_result['objective'], known_optimal)

    # Compute quality score
    all_objectives = [r['objective'] for r in results]
    quality = compute_solution_quality(best_result['objective'], all_objectives)

    return BenchmarkResult(
        method=method,
        objective_value=best_result['objective'],
        runtime_seconds=avg_runtime,
        iterations=best_result.get('iterations', 0),
        solution=best_result['solution'],
        is_valid=best_result.get('is_valid', True),
        optimality_gap=gap,
        quality_score=quality,
        metadata={
            'runs': runs,
            'all_objectives': all_objectives,
            'all_runtimes': runtimes,
            'config': config,
        }
    )


def compare_methods(
    problem: Any,
    adapter: Any,
    methods: List[str],
    configs: Optional[Dict[str, Dict[str, Any]]] = None,
    known_optimal: Optional[float] = None,
    runs: int = 1,
) -> List[BenchmarkResult]:
    """
    Compare multiple optimization methods on the same problem

    Args:
        problem: Problem instance
        adapter: Domain adapter
        methods: List of method names to compare
        configs: Optional dict mapping method name to config
        known_optimal: Known optimal objective value
        runs: Number of runs per method

    Returns:
        List of BenchmarkResult objects, sorted by objective value

    Example:
        >>> results = compare_methods(
        ...     problem, adapter,
        ...     methods=['random_search', 'simulated_annealing', 'genetic_algorithm'],
        ...     known_optimal=578.0
        ... )
        >>>
        >>> for r in results:
        ...     print(f"{r.method}: {r.objective_value:.2f} (gap: {r.optimality_gap:.2%})")
    """
    if configs is None:
        configs = {}

    results = []

    for method in methods:
        config = configs.get(method, {})
        result = benchmark_method(
            problem, adapter, method, config,
            known_optimal=known_optimal, runs=runs
        )
        results.append(result)

    # Sort by objective value (lower is better)
    results.sort(key=lambda r: r.objective_value)

    return results


def benchmark_suite(
    problems: List[Dict[str, Any]],
    adapter: Any,
    methods: List[str],
    configs: Optional[Dict[str, Dict[str, Any]]] = None,
    runs: int = 1,
) -> Dict[str, List[BenchmarkResult]]:
    """
    Run a benchmark suite across multiple problems

    Args:
        problems: List of problem instances, each with 'instance' and optional 'optimal' keys
        adapter: Domain adapter
        methods: List of method names
        configs: Optional dict mapping method name to config
        runs: Number of runs per method per problem

    Returns:
        Dict mapping problem name to list of BenchmarkResults

    Example:
        >>> problems = [
        ...     {'name': 'chr12a', 'instance': qap_instance_1, 'optimal': 9552},
        ...     {'name': 'chr15a', 'instance': qap_instance_2, 'optimal': 9896},
        ... ]
        >>>
        >>> results = benchmark_suite(
        ...     problems, QAPAdapter(),
        ...     methods=['simulated_annealing', 'genetic_algorithm'],
        ...     runs=5
        ... )
        >>>
        >>> # Analyze results
        >>> for problem_name, method_results in results.items():
        ...     print(f"\n{problem_name}:")
        ...     for r in method_results:
        ...         print(f"  {r.method}: {r.optimality_gap:.2%} gap")
    """
    if configs is None:
        configs = {}

    suite_results: Dict[str, List[BenchmarkResult]] = {}

    for problem_dict in problems:
        problem_name = problem_dict.get('name', f"problem_{len(suite_results)}")
        problem_instance = problem_dict['instance']
        known_optimal = problem_dict.get('optimal')

        method_results = compare_methods(
            problem_instance, adapter, methods, configs,
            known_optimal=known_optimal, runs=runs
        )

        suite_results[problem_name] = method_results

    return suite_results


__all__ = [
    'BenchmarkResult',
    'benchmark_method',
    'compare_methods',
    'benchmark_suite',
]
