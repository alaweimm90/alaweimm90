"""
QAPLIB Benchmark Runner

Provides functionality to benchmark optimization methods on QAPLIB instances.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field

import numpy as np

from .loader import load_qaplib_instance
from .registry import QAPLIB_REGISTRY, get_small_instances


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    instance_name: str
    method_name: str
    objective_value: float
    runtime_seconds: float
    iterations: int
    solution: Optional[np.ndarray] = None
    optimal_value: Optional[float] = None
    gap_percent: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate gap if optimal value is known"""
        if self.optimal_value is not None and self.objective_value is not None:
            self.gap_percent = 100 * (self.objective_value - self.optimal_value) / self.optimal_value


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results across multiple instances"""
    method_name: str
    results: List[BenchmarkResult]
    total_runtime: float
    avg_gap: Optional[float] = None
    solved_optimally: int = 0
    failed_instances: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate summary statistics"""
        gaps = [r.gap_percent for r in self.results if r.gap_percent is not None]
        if gaps:
            self.avg_gap = np.mean(gaps)
            self.solved_optimally = sum(1 for g in gaps if g < 0.01)  # Within 0.01% of optimal

        self.failed_instances = [r.instance_name for r in self.results
                                if r.objective_value is None or r.objective_value == float('inf')]


class QAPLIBBenchmark:
    """Main benchmark runner for QAPLIB instances"""

    def __init__(self, verbose: bool = True):
        """
        Initialize benchmark runner

        Args:
            verbose: Print progress during benchmarking
        """
        self.verbose = verbose
        self.results: Dict[str, List[BenchmarkResult]] = {}

    def run_instance(
        self,
        instance_name: str,
        method: Callable,
        method_name: str = "unknown",
        method_config: Optional[Dict] = None
    ) -> BenchmarkResult:
        """
        Run a single benchmark on one instance

        Args:
            instance_name: Name of QAPLIB instance
            method: Optimization method (function that takes problem dict and returns solution)
            method_name: Name of the method for reporting
            method_config: Optional configuration for the method

        Returns:
            BenchmarkResult object
        """
        if self.verbose:
            print(f"Running {method_name} on {instance_name}...", end=" ")

        # Load instance
        try:
            instance_data = load_qaplib_instance(instance_name)
        except Exception as e:
            if self.verbose:
                print(f"Failed to load: {e}")
            return BenchmarkResult(
                instance_name=instance_name,
                method_name=method_name,
                objective_value=float('inf'),
                runtime_seconds=0,
                iterations=0,
                metadata={"error": str(e)}
            )

        # Get optimal value if known
        optimal_value = None
        if instance_name in QAPLIB_REGISTRY:
            optimal_value = QAPLIB_REGISTRY[instance_name].optimal_value

        # Run method
        start_time = time.time()
        try:
            # Call method with instance data
            if method_config:
                result = method(instance_data, **method_config)
            else:
                result = method(instance_data)

            runtime = time.time() - start_time

            # Extract result components
            if isinstance(result, dict):
                solution = result.get("solution")
                objective = result.get("objective", float('inf'))
                iterations = result.get("iterations", 0)
                metadata = result.get("metadata", {})
            else:
                # Assume result is just the solution
                solution = result
                objective = self._compute_objective(instance_data, solution)
                iterations = 0
                metadata = {}

            if self.verbose:
                if optimal_value:
                    gap = 100 * (objective - optimal_value) / optimal_value
                    print(f"obj={objective:.0f}, gap={gap:.2f}%, time={runtime:.2f}s")
                else:
                    print(f"obj={objective:.0f}, time={runtime:.2f}s")

            return BenchmarkResult(
                instance_name=instance_name,
                method_name=method_name,
                objective_value=objective,
                runtime_seconds=runtime,
                iterations=iterations,
                solution=solution,
                optimal_value=optimal_value,
                metadata=metadata
            )

        except Exception as e:
            runtime = time.time() - start_time
            if self.verbose:
                print(f"Failed: {e}")
            return BenchmarkResult(
                instance_name=instance_name,
                method_name=method_name,
                objective_value=float('inf'),
                runtime_seconds=runtime,
                iterations=0,
                metadata={"error": str(e)}
            )

    def run_multiple(
        self,
        instance_names: List[str],
        method: Callable,
        method_name: str = "unknown",
        method_config: Optional[Dict] = None
    ) -> BenchmarkSummary:
        """
        Run benchmark on multiple instances

        Args:
            instance_names: List of instance names
            method: Optimization method
            method_name: Name of the method
            method_config: Optional configuration

        Returns:
            BenchmarkSummary object
        """
        if self.verbose:
            print(f"\nBenchmarking {method_name} on {len(instance_names)} instances:")
            print("-" * 60)

        results = []
        total_start = time.time()

        for instance_name in instance_names:
            result = self.run_instance(instance_name, method, method_name, method_config)
            results.append(result)

            # Store result
            if method_name not in self.results:
                self.results[method_name] = []
            self.results[method_name].append(result)

        total_runtime = time.time() - total_start

        summary = BenchmarkSummary(
            method_name=method_name,
            results=results,
            total_runtime=total_runtime
        )

        if self.verbose:
            self._print_summary(summary)

        return summary

    def compare_methods(
        self,
        methods: Dict[str, Callable],
        instance_names: Optional[List[str]] = None,
        method_configs: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, BenchmarkSummary]:
        """
        Compare multiple methods on the same instances

        Args:
            methods: Dictionary of method_name -> method function
            instance_names: List of instances (default: small instances)
            method_configs: Optional configurations for each method

        Returns:
            Dictionary of method_name -> BenchmarkSummary
        """
        if instance_names is None:
            instance_names = get_small_instances()[:10]  # Default to 10 small instances

        if method_configs is None:
            method_configs = {}

        summaries = {}

        for method_name, method in methods.items():
            config = method_configs.get(method_name)
            summary = self.run_multiple(instance_names, method, method_name, config)
            summaries[method_name] = summary

        if self.verbose:
            self._print_comparison(summaries)

        return summaries

    def _compute_objective(self, instance_data: Dict, solution: np.ndarray) -> float:
        """Compute QAP objective value"""
        flow = instance_data['flow_matrix']
        dist = instance_data['distance_matrix']
        n = len(solution)

        obj = 0.0
        for i in range(n):
            for j in range(n):
                obj += flow[i, j] * dist[solution[i], solution[j]]
        return obj

    def _print_summary(self, summary: BenchmarkSummary):
        """Print summary of results"""
        print("-" * 60)
        print(f"Summary for {summary.method_name}:")
        print(f"  Total runtime: {summary.total_runtime:.2f}s")
        if summary.avg_gap is not None:
            print(f"  Average gap: {summary.avg_gap:.2f}%")
            print(f"  Solved optimally: {summary.solved_optimally}/{len(summary.results)}")
        if summary.failed_instances:
            print(f"  Failed instances: {', '.join(summary.failed_instances)}")

    def _print_comparison(self, summaries: Dict[str, BenchmarkSummary]):
        """Print comparison table"""
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        # Header
        print(f"{'Method':<20} {'Avg Gap %':>12} {'Optimal':>10} {'Failed':>10} {'Runtime':>12}")
        print("-" * 80)

        # Sort by average gap
        sorted_methods = sorted(summaries.items(),
                              key=lambda x: x[1].avg_gap if x[1].avg_gap else float('inf'))

        for method_name, summary in sorted_methods:
            gap_str = f"{summary.avg_gap:.2f}" if summary.avg_gap is not None else "N/A"
            print(f"{method_name:<20} {gap_str:>12} {summary.solved_optimally:>10} "
                  f"{len(summary.failed_instances):>10} {summary.total_runtime:>11.2f}s")

        print("=" * 80)


def run_qaplib_benchmark(
    method: Callable,
    instances: Optional[Union[List[str], str]] = "small",
    method_name: str = "unknown",
    method_config: Optional[Dict] = None,
    verbose: bool = True
) -> BenchmarkSummary:
    """
    Convenience function to run QAPLIB benchmarks

    Args:
        method: Optimization method to benchmark
        instances: "small", "medium", "large", or list of instance names
        method_name: Name of the method
        method_config: Optional configuration
        verbose: Print progress

    Returns:
        BenchmarkSummary object

    Example:
        >>> from Librex.benchmarks.qaplib import run_qaplib_benchmark
        >>> from my_optimizer import optimize_qap
        >>>
        >>> results = run_qaplib_benchmark(
        ...     method=optimize_qap,
        ...     instances="small",
        ...     method_name="MyOptimizer",
        ...     method_config={"max_iterations": 1000}
        ... )
    """
    # Determine instances
    if isinstance(instances, str):
        if instances == "small":
            instance_names = get_small_instances()[:10]
        elif instances == "medium":
            from .registry import get_instance_by_size
            instance_names = get_instance_by_size(21, 50)[:10]
        elif instances == "large":
            from .registry import get_instance_by_size
            instance_names = get_instance_by_size(51, 100)[:5]
        else:
            raise ValueError(f"Unknown instance set: {instances}")
    else:
        instance_names = instances

    # Run benchmark
    benchmark = QAPLIBBenchmark(verbose=verbose)
    return benchmark.run_multiple(instance_names, method, method_name, method_config)