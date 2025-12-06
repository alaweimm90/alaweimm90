"""
Performance Profiling Tools for MEZAN Libria Suite

Provides comprehensive profiling and performance analysis for all 7 Libria solvers.

Features:
- CPU/Memory profiling
- Scalability analysis
- Algorithm comparison
- Performance regression detection
- Detailed timing breakdowns
"""

import time
import tracemalloc
import cProfile
import pstats
import io
from typing import Dict, Any, List, Optional, Callable
import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path

from MEZAN.core import (
    OptimizerInterface,
    OptimizerFactory,
    OptimizationProblem,
    OptimizationResult,
    ProblemType,
)


@dataclass
class ProfileResult:
    """Profiling result for a single run"""

    solver_name: str
    problem_type: str
    problem_size: int
    computation_time: float
    memory_peak_mb: float
    memory_current_mb: float
    iterations: Optional[int]
    objective_value: Optional[float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScalabilityResult:
    """Scalability analysis result"""

    solver_name: str
    problem_type: str
    sizes: List[int]
    times: List[float]
    memory_usage: List[float]
    scaling_exponent: float  # From power law fit: T(n) = a * n^b
    scaling_quality: float  # R^2 of fit

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LibriaProfiler:
    """
    Comprehensive profiler for Libria solvers

    Usage:
        profiler = LibriaProfiler()
        result = profiler.profile_solver(solver, problem)
        print(f"Time: {result.computation_time:.3f}s, Memory: {result.memory_peak_mb:.2f}MB")
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("profiling_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ProfileResult] = []

    def profile_solver(
        self,
        solver: OptimizerInterface,
        problem: OptimizationProblem,
        runs: int = 3,
    ) -> ProfileResult:
        """
        Profile a single solver on a problem

        Args:
            solver: Solver instance
            problem: Problem to solve
            runs: Number of runs (for averaging)

        Returns:
            ProfileResult with timing and memory metrics
        """
        times = []
        memory_peaks = []
        iterations_list = []
        objective_values = []

        for run in range(runs):
            # Start memory tracking
            tracemalloc.start()

            # Time the solve
            start_time = time.perf_counter()
            result = solver.solve(problem)
            end_time = time.perf_counter()

            # Get memory stats
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            times.append(end_time - start_time)
            memory_peaks.append(peak_mem / 1024 / 1024)  # Convert to MB

            if result.iterations is not None:
                iterations_list.append(result.iterations)
            if result.objective_value is not None:
                objective_values.append(result.objective_value)

        # Aggregate results
        profile_result = ProfileResult(
            solver_name=solver.__class__.__name__,
            problem_type=problem.problem_type.value,
            problem_size=self._estimate_problem_size(problem),
            computation_time=np.mean(times),
            memory_peak_mb=np.mean(memory_peaks),
            memory_current_mb=current_mem / 1024 / 1024,
            iterations=int(np.mean(iterations_list)) if iterations_list else None,
            objective_value=np.mean(objective_values) if objective_values else None,
            metadata={
                "runs": runs,
                "time_std": float(np.std(times)),
                "time_min": float(np.min(times)),
                "time_max": float(np.max(times)),
                "solver_config": getattr(solver, "config", {}),
            },
        )

        self.results.append(profile_result)
        return profile_result

    def profile_with_cprofile(
        self,
        solver: OptimizerInterface,
        problem: OptimizationProblem,
        top_n: int = 20,
    ) -> str:
        """
        Profile using cProfile for detailed function-level analysis

        Args:
            solver: Solver to profile
            problem: Problem to solve
            top_n: Number of top functions to show

        Returns:
            String with profiling statistics
        """
        profiler = cProfile.Profile()

        # Profile the solve
        profiler.enable()
        result = solver.solve(problem)
        profiler.disable()

        # Get statistics
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(top_n)

        profile_output = s.getvalue()

        # Save to file
        output_file = (
            self.output_dir
            / f"cprofile_{solver.__class__.__name__}_{int(time.time())}.txt"
        )
        with open(output_file, "w") as f:
            f.write(profile_output)

        return profile_output

    def scalability_analysis(
        self,
        solver_factory: Callable[[Dict], OptimizerInterface],
        problem_generator: Callable[[int], OptimizationProblem],
        sizes: List[int],
        problem_type: ProblemType,
    ) -> ScalabilityResult:
        """
        Analyze how solver scales with problem size

        Args:
            solver_factory: Function that creates solver instance
            problem_generator: Function that generates problem of given size
            sizes: List of problem sizes to test
            problem_type: Type of problem

        Returns:
            ScalabilityResult with scaling analysis
        """
        times = []
        memory_usage = []

        for size in sizes:
            problem = problem_generator(size)
            solver = solver_factory({})

            # Profile this size
            result = self.profile_solver(solver, problem, runs=1)
            times.append(result.computation_time)
            memory_usage.append(result.memory_peak_mb)

        # Fit power law: T(n) = a * n^b
        log_sizes = np.log(sizes)
        log_times = np.log(times)

        # Linear regression in log space
        coeffs = np.polyfit(log_sizes, log_times, 1)
        scaling_exponent = coeffs[0]

        # R^2 for quality of fit
        predicted = coeffs[0] * log_sizes + coeffs[1]
        ss_res = np.sum((log_times - predicted) ** 2)
        ss_tot = np.sum((log_times - np.mean(log_times)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        scalability_result = ScalabilityResult(
            solver_name=solver.__class__.__name__,
            problem_type=problem_type.value,
            sizes=sizes,
            times=times,
            memory_usage=memory_usage,
            scaling_exponent=float(scaling_exponent),
            scaling_quality=float(r_squared),
        )

        return scalability_result

    def compare_solvers(
        self,
        solvers: List[OptimizerInterface],
        problem: OptimizationProblem,
    ) -> Dict[str, ProfileResult]:
        """
        Compare multiple solvers on the same problem

        Args:
            solvers: List of solver instances
            problem: Problem to solve

        Returns:
            Dict mapping solver name to ProfileResult
        """
        results = {}

        for solver in solvers:
            result = self.profile_solver(solver, problem)
            results[solver.__class__.__name__] = result

        # Save comparison
        comparison_file = self.output_dir / f"comparison_{int(time.time())}.json"
        with open(comparison_file, "w") as f:
            json.dump(
                {name: res.to_dict() for name, res in results.items()}, f, indent=2
            )

        return results

    def detect_regression(
        self,
        baseline_file: Path,
        threshold: float = 1.2,
    ) -> Dict[str, Any]:
        """
        Detect performance regressions compared to baseline

        Args:
            baseline_file: Path to baseline profiling results (JSON)
            threshold: Regression threshold (1.2 = 20% slower is regression)

        Returns:
            Dict with regression analysis
        """
        with open(baseline_file, "r") as f:
            baseline = json.load(f)

        regressions = []
        improvements = []

        for current_result in self.results:
            key = f"{current_result.solver_name}_{current_result.problem_type}_{current_result.problem_size}"

            if key in baseline:
                baseline_time = baseline[key]["computation_time"]
                current_time = current_result.computation_time

                ratio = current_time / baseline_time

                if ratio > threshold:
                    regressions.append(
                        {
                            "solver": current_result.solver_name,
                            "problem": current_result.problem_type,
                            "size": current_result.problem_size,
                            "baseline_time": baseline_time,
                            "current_time": current_time,
                            "slowdown": ratio,
                        }
                    )
                elif ratio < 1.0 / threshold:
                    improvements.append(
                        {
                            "solver": current_result.solver_name,
                            "problem": current_result.problem_type,
                            "size": current_result.problem_size,
                            "baseline_time": baseline_time,
                            "current_time": current_time,
                            "speedup": 1.0 / ratio,
                        }
                    )

        return {
            "regressions": regressions,
            "improvements": improvements,
            "threshold": threshold,
        }

    def save_baseline(self, filename: str = "baseline.json"):
        """Save current results as baseline for future comparisons"""
        baseline = {}
        for result in self.results:
            key = f"{result.solver_name}_{result.problem_type}_{result.problem_size}"
            baseline[key] = result.to_dict()

        baseline_file = self.output_dir / filename
        with open(baseline_file, "w") as f:
            json.dump(baseline, f, indent=2)

        return baseline_file

    def generate_report(self, filename: str = "profiling_report.html") -> Path:
        """Generate HTML report with all profiling results"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>MEZAN Libria Profiling Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .metric { font-weight: bold; color: #2196F3; }
    </style>
</head>
<body>
    <h1>MEZAN Libria Performance Profiling Report</h1>
    <p>Generated: {timestamp}</p>

    <h2>Summary Statistics</h2>
    <table>
        <tr>
            <th>Solver</th>
            <th>Problem Type</th>
            <th>Size</th>
            <th>Time (s)</th>
            <th>Memory Peak (MB)</th>
            <th>Iterations</th>
            <th>Objective Value</th>
        </tr>
        {rows}
    </table>

    <h2>Performance Notes</h2>
    <ul>
        <li>All times are averaged over {runs} runs</li>
        <li>Memory measurements include peak usage during solve</li>
        <li>Lower objective values are better for minimization problems</li>
    </ul>
</body>
</html>
"""

        rows = ""
        for result in self.results:
            rows += f"""
        <tr>
            <td>{result.solver_name}</td>
            <td>{result.problem_type}</td>
            <td>{result.problem_size}</td>
            <td class="metric">{result.computation_time:.4f}</td>
            <td class="metric">{result.memory_peak_mb:.2f}</td>
            <td>{result.iterations if result.iterations else 'N/A'}</td>
            <td>{result.objective_value:.2f if result.objective_value else 'N/A'}</td>
        </tr>
"""

        html = html.format(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            rows=rows,
            runs=self.results[0].metadata.get("runs", 1) if self.results else 1,
        )

        report_file = self.output_dir / filename
        with open(report_file, "w") as f:
            f.write(html)

        return report_file

    def _estimate_problem_size(self, problem: OptimizationProblem) -> int:
        """Estimate problem size from data"""
        if problem.problem_type == ProblemType.QAP:
            if "distance_matrix" in problem.data:
                return len(problem.data["distance_matrix"])

        elif problem.problem_type == ProblemType.FLOW:
            if "workflow_graph" in problem.data:
                return len(problem.data["workflow_graph"].get("nodes", []))

        elif problem.problem_type == ProblemType.ALLOC:
            if "resource_demands" in problem.data:
                return len(problem.data["resource_demands"])

        elif problem.problem_type == ProblemType.GRAPH:
            if "communication_matrix" in problem.data:
                return len(problem.data["communication_matrix"])

        elif problem.problem_type == ProblemType.EVO:
            return problem.data.get("num_variables", 0)

        return 0


# Convenience functions

def profile_qap_solver(solver, distance_matrix, flow_matrix):
    """Quick profiling for QAP solver"""
    from MEZAN.core import OptimizationProblem, ProblemType

    problem = OptimizationProblem(
        problem_type=ProblemType.QAP,
        data={"distance_matrix": distance_matrix, "flow_matrix": flow_matrix},
    )

    profiler = LibriaProfiler()
    return profiler.profile_solver(solver, problem)


def profile_atlas_workflow(agents, tasks):
    """Quick profiling for ORCHEX optimization"""
    from MEZAN.ORCHEX.atlas_core.src.ORCHEX.optimization_integration import (
        ATLASOptimizationManager,
    )

    manager = ATLASOptimizationManager()

    profiler = LibriaProfiler()

    tracemalloc.start()
    start_time = time.perf_counter()

    result = manager.optimize_agent_assignment(agents, tasks)

    end_time = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "computation_time": end_time - start_time,
        "memory_peak_mb": peak_mem / 1024 / 1024,
        "assignment": result["assignment"],
        "objective_value": result["total_cost"],
    }
