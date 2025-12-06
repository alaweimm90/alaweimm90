"""
Benchmarking Framework for MEZAN and Libria Solvers

Provides comprehensive performance testing and comparison:
- Solver performance profiling
- Problem suite benchmarking
- Statistical analysis
- Visualization-ready results
- CI/CD integration support

Author: MEZAN Research Team
Date: 2025-11-18
Version: 1.0
"""

import time
import json
import statistics
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BenchmarkStatus(Enum):
    """Status of a benchmark run"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run"""
    benchmark_id: str
    solver_name: str
    problem_name: str
    objective_value: float
    time_elapsed: float
    iterations: int
    memory_mb: float
    success: bool
    status: BenchmarkStatus
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results"""
    suite_name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def add_result(self, result: BenchmarkResult):
        """Add a result to the suite"""
        self.results.append(result)

    def finalize(self):
        """Mark suite as complete"""
        self.end_time = time.time()

    def get_statistics(self) -> Dict[str, Any]:
        """Compute statistics across all results"""
        if not self.results:
            return {}

        successful = [r for r in self.results if r.success]
        if not successful:
            return {"total": len(self.results), "successful": 0}

        times = [r.time_elapsed for r in successful]
        objectives = [r.objective_value for r in successful]
        iterations = [r.iterations for r in successful]

        return {
            "total": len(self.results),
            "successful": len(successful),
            "failed": len(self.results) - len(successful),
            "time": {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "stdev": statistics.stdev(times) if len(times) > 1 else 0,
                "min": min(times),
                "max": max(times),
                "total": sum(times),
            },
            "objective": {
                "mean": statistics.mean(objectives),
                "median": statistics.median(objectives),
                "stdev": statistics.stdev(objectives) if len(objectives) > 1 else 0,
                "min": min(objectives),
                "max": max(objectives),
            },
            "iterations": {
                "mean": statistics.mean(iterations),
                "median": int(statistics.median(iterations)),
                "min": min(iterations),
                "max": max(iterations),
            },
        }

    def get_by_solver(self) -> Dict[str, List[BenchmarkResult]]:
        """Group results by solver"""
        by_solver = {}
        for result in self.results:
            if result.solver_name not in by_solver:
                by_solver[result.solver_name] = []
            by_solver[result.solver_name].append(result)
        return by_solver

    def get_by_problem(self) -> Dict[str, List[BenchmarkResult]]:
        """Group results by problem"""
        by_problem = {}
        for result in self.results:
            if result.problem_name not in by_problem:
                by_problem[result.problem_name] = []
            by_problem[result.problem_name].append(result)
        return by_problem

    def to_dict(self) -> Dict[str, Any]:
        """Convert suite to dictionary"""
        return {
            "suite_name": self.suite_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": (self.end_time or time.time()) - self.start_time,
            "results": [asdict(r) for r in self.results],
            "statistics": self.get_statistics(),
        }

    def to_json(self, filepath: str):
        """Save suite to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Benchmark suite saved to {filepath}")


class Benchmarker:
    """
    Benchmarking framework for MEZAN and Libria solvers

    Features:
    - Automated problem suite execution
    - Performance profiling
    - Statistical analysis
    - Comparison across solvers
    - Export to multiple formats
    """

    def __init__(self, suite_name: str = "default"):
        self.suite = BenchmarkSuite(suite_name=suite_name)
        self.problem_generators: Dict[str, Callable] = {}
        self.solvers: Dict[str, Any] = {}

    def register_problem_generator(
        self, name: str, generator: Callable[[], Dict[str, Any]]
    ):
        """
        Register a problem generator

        Args:
            name: Problem name
            generator: Function that generates a problem instance
        """
        self.problem_generators[name] = generator
        logger.info(f"Registered problem generator: {name}")

    def register_solver(self, name: str, solver: Any):
        """
        Register a solver for benchmarking

        Args:
            name: Solver name
            solver: Solver instance with solve() method
        """
        self.solvers[name] = solver
        logger.info(f"Registered solver: {name}")

    def benchmark_single(
        self,
        solver_name: str,
        problem_name: str,
        problem: Dict[str, Any],
        timeout: float = 60.0,
        **solver_kwargs,
    ) -> BenchmarkResult:
        """
        Run a single benchmark

        Args:
            solver_name: Name of solver to use
            problem_name: Name of problem
            problem: Problem instance
            timeout: Maximum time allowed (seconds)
            **solver_kwargs: Additional arguments to pass to solver

        Returns:
            BenchmarkResult with performance metrics
        """
        benchmark_id = f"{solver_name}_{problem_name}_{int(time.time())}"

        if solver_name not in self.solvers:
            return BenchmarkResult(
                benchmark_id=benchmark_id,
                solver_name=solver_name,
                problem_name=problem_name,
                objective_value=0.0,
                time_elapsed=0.0,
                iterations=0,
                memory_mb=0.0,
                success=False,
                status=BenchmarkStatus.FAILED,
                error=f"Solver {solver_name} not registered",
            )

        solver = self.solvers[solver_name]

        try:
            # Run benchmark
            start_time = time.time()
            start_memory = 0.0  # TODO: Add memory tracking

            result = solver.solve(problem, **solver_kwargs)

            end_time = time.time()
            elapsed = end_time - start_time

            # Check timeout
            if elapsed > timeout:
                return BenchmarkResult(
                    benchmark_id=benchmark_id,
                    solver_name=solver_name,
                    problem_name=problem_name,
                    objective_value=0.0,
                    time_elapsed=elapsed,
                    iterations=0,
                    memory_mb=0.0,
                    success=False,
                    status=BenchmarkStatus.FAILED,
                    error=f"Timeout: {elapsed:.2f}s > {timeout}s",
                )

            # Extract metrics from result
            if hasattr(result, "objective_value"):
                obj_value = result.objective_value
                iterations = result.iterations
                success = result.success
            elif isinstance(result, dict):
                obj_value = result.get("objective_value", 0.0)
                iterations = result.get("iterations", 1)
                success = result.get("success", True)
            else:
                obj_value = 0.0
                iterations = 1
                success = True

            return BenchmarkResult(
                benchmark_id=benchmark_id,
                solver_name=solver_name,
                problem_name=problem_name,
                objective_value=float(obj_value),
                time_elapsed=elapsed,
                iterations=int(iterations),
                memory_mb=0.0,  # TODO: Implement memory tracking
                success=success,
                status=BenchmarkStatus.COMPLETED,
                metadata={"problem_size": problem.get("size", "unknown")},
            )

        except Exception as e:
            logger.error(f"Benchmark failed: {solver_name} on {problem_name}: {e}")
            return BenchmarkResult(
                benchmark_id=benchmark_id,
                solver_name=solver_name,
                problem_name=problem_name,
                objective_value=0.0,
                time_elapsed=0.0,
                iterations=0,
                memory_mb=0.0,
                success=False,
                status=BenchmarkStatus.FAILED,
                error=str(e),
            )

    def run_suite(
        self,
        solvers: Optional[List[str]] = None,
        problems: Optional[List[str]] = None,
        repetitions: int = 1,
        timeout: float = 60.0,
    ) -> BenchmarkSuite:
        """
        Run full benchmark suite

        Args:
            solvers: List of solver names (None = all registered)
            problems: List of problem names (None = all registered)
            repetitions: Number of times to repeat each benchmark
            timeout: Timeout per benchmark (seconds)

        Returns:
            BenchmarkSuite with all results
        """
        if solvers is None:
            solvers = list(self.solvers.keys())
        if problems is None:
            problems = list(self.problem_generators.keys())

        logger.info(f"Running benchmark suite: {len(solvers)} solvers x {len(problems)} problems x {repetitions} reps")

        total = len(solvers) * len(problems) * repetitions
        current = 0

        for solver_name in solvers:
            for problem_name in problems:
                # Generate problem
                if problem_name not in self.problem_generators:
                    logger.warning(f"Problem generator not found: {problem_name}")
                    continue

                for rep in range(repetitions):
                    current += 1
                    logger.info(f"[{current}/{total}] {solver_name} on {problem_name} (rep {rep+1})")

                    try:
                        problem = self.problem_generators[problem_name]()
                    except Exception as e:
                        logger.error(f"Problem generation failed: {problem_name}: {e}")
                        continue

                    # Run benchmark
                    result = self.benchmark_single(
                        solver_name, problem_name, problem, timeout=timeout
                    )
                    self.suite.add_result(result)

        self.suite.finalize()
        logger.info(f"Benchmark suite complete: {len(self.suite.results)} results")
        return self.suite

    def compare_solvers(self, metric: str = "time_elapsed") -> Dict[str, float]:
        """
        Compare solvers on a specific metric

        Args:
            metric: Metric to compare ("time_elapsed", "objective_value", "iterations")

        Returns:
            Dictionary mapping solver names to average metric values
        """
        by_solver = self.suite.get_by_solver()
        comparison = {}

        for solver_name, results in by_solver.items():
            successful = [r for r in results if r.success]
            if not successful:
                comparison[solver_name] = float('inf')
                continue

            if metric == "time_elapsed":
                values = [r.time_elapsed for r in successful]
            elif metric == "objective_value":
                values = [r.objective_value for r in successful]
            elif metric == "iterations":
                values = [r.iterations for r in successful]
            else:
                values = [0.0]

            comparison[solver_name] = statistics.mean(values)

        return comparison

    def generate_report(self) -> str:
        """
        Generate a human-readable text report

        Returns:
            Formatted report string
        """
        stats = self.suite.get_statistics()

        report = []
        report.append("=" * 70)
        report.append(f" BENCHMARK REPORT: {self.suite.suite_name}")
        report.append("=" * 70)
        report.append("")

        # Overview
        report.append("OVERVIEW")
        report.append("-" * 70)
        report.append(f"Total Benchmarks: {stats.get('total', 0)}")
        report.append(f"Successful: {stats.get('successful', 0)}")
        report.append(f"Failed: {stats.get('failed', 0)}")
        report.append(f"Duration: {self.suite.end_time - self.suite.start_time if self.suite.end_time else 0:.2f}s")
        report.append("")

        # Time statistics
        if "time" in stats:
            report.append("TIME STATISTICS")
            report.append("-" * 70)
            report.append(f"Mean: {stats['time']['mean']:.4f}s")
            report.append(f"Median: {stats['time']['median']:.4f}s")
            report.append(f"StdDev: {stats['time']['stdev']:.4f}s")
            report.append(f"Range: [{stats['time']['min']:.4f}s, {stats['time']['max']:.4f}s]")
            report.append(f"Total: {stats['time']['total']:.4f}s")
            report.append("")

        # Objective statistics
        if "objective" in stats:
            report.append("OBJECTIVE VALUE STATISTICS")
            report.append("-" * 70)
            report.append(f"Mean: {stats['objective']['mean']:.4f}")
            report.append(f"Median: {stats['objective']['median']:.4f}")
            report.append(f"StdDev: {stats['objective']['stdev']:.4f}")
            report.append(f"Range: [{stats['objective']['min']:.4f}, {stats['objective']['max']:.4f}]")
            report.append("")

        # Solver comparison
        report.append("SOLVER COMPARISON (Average Time)")
        report.append("-" * 70)
        comparison = self.compare_solvers("time_elapsed")
        for solver_name, avg_time in sorted(comparison.items(), key=lambda x: x[1]):
            report.append(f"{solver_name:20s}: {avg_time:.4f}s")
        report.append("")

        report.append("=" * 70)
        return "\n".join(report)


def create_qap_problem_generator(size: int) -> Callable[[], Dict[str, Any]]:
    """Create a QAP problem generator"""
    import numpy as np

    def generator():
        np.random.seed(int(time.time() * 1000) % (2**32))
        cost_matrix = np.random.rand(size, size).tolist()
        return {
            "type": "qap",
            "size": size,
            "fit": cost_matrix,
            "constraints": {},
        }

    return generator


def create_allocation_problem_generator(n_options: int, budget: float) -> Callable[[], Dict[str, Any]]:
    """Create an allocation problem generator"""
    import numpy as np

    def generator():
        np.random.seed(int(time.time() * 1000) % (2**32))
        options = [
            {"id": f"option_{i}", "expected_return": np.random.uniform(0.05, 0.20)}
            for i in range(n_options)
        ]
        return {
            "type": "allocation",
            "options": options,
            "budget": budget,
        }

    return generator


# Example usage helper
def create_example_benchmark() -> Benchmarker:
    """Create an example benchmarker with some problems and solvers"""
    benchmarker = Benchmarker(suite_name="example_suite")

    # Register problem generators
    benchmarker.register_problem_generator("qap_small", create_qap_problem_generator(10))
    benchmarker.register_problem_generator("qap_medium", create_qap_problem_generator(20))
    benchmarker.register_problem_generator("allocation_small", create_allocation_problem_generator(5, 1000.0))

    return benchmarker
