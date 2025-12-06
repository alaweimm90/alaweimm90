"""
Optimization Utilities and Benchmark Functions
==============================================
This module provides utility functions, benchmark test problems,
and analysis tools for optimization algorithms.
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import time
from typing import Callable, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from .unconstrained import OptimizationResult
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
BERKELEY_LIGHT_BLUE = '#3B7EA1'
@dataclass
class BenchmarkResult:
    """
    Results from benchmark testing.
    Attributes:
        function_name: Name of test function
        algorithm_name: Name of optimization algorithm
        result: OptimizationResult object
        success_rate: Success rate over multiple runs
        avg_iterations: Average number of iterations
        avg_function_evaluations: Average number of function evaluations
        avg_execution_time: Average execution time
        best_value_found: Best function value found
        convergence_data: Convergence history data
    """
    function_name: str
    algorithm_name: str
    result: OptimizationResult
    success_rate: float = 0.0
    avg_iterations: int = 0
    avg_function_evaluations: int = 0
    avg_execution_time: float = 0.0
    best_value_found: float = float('inf')
    convergence_data: Optional[List[float]] = None
class OptimizationProblem:
    """
    Represents an optimization problem with objective, constraints, and bounds.
    This class provides a standardized way to define optimization problems
    and evaluate different algorithms on them.
    """
    def __init__(self, name: str, objective: Callable, gradient: Optional[Callable] = None,
                 hessian: Optional[Callable] = None, bounds: Optional[List[Tuple[float, float]]] = None,
                 constraints: Optional[List] = None, global_minimum: Optional[float] = None,
                 optimal_point: Optional[np.ndarray] = None, dimension: Optional[int] = None):
        """
        Initialize optimization problem.
        Args:
            name: Problem name
            objective: Objective function
            gradient: Gradient function (optional)
            hessian: Hessian function (optional)
            bounds: Variable bounds
            constraints: List of constraints
            global_minimum: Known global minimum value
            optimal_point: Known optimal point
            dimension: Problem dimension
        """
        self.name = name
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        self.bounds = bounds
        self.constraints = constraints or []
        self.global_minimum = global_minimum
        self.optimal_point = optimal_point
        self.dimension = dimension or (len(bounds) if bounds else None)
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate objective function."""
        return self.objective(x)
    def evaluate_gradient(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Evaluate gradient if available."""
        return self.gradient(x) if self.gradient else None
    def evaluate_hessian(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Evaluate Hessian if available."""
        return self.hessian(x) if self.hessian else None
    def is_feasible(self, x: np.ndarray) -> bool:
        """Check if point is feasible."""
        # Check bounds
        if self.bounds:
            for i, (low, high) in enumerate(self.bounds):
                if x[i] < low or x[i] > high:
                    return False
        # Check constraints
        for constraint in self.constraints:
            value = constraint.fun(x)
            if constraint.type == 'eq' and abs(value) > 1e-6:
                return False
            elif constraint.type == 'ineq' and value > 1e-6:
                return False
        return True
    def distance_to_optimum(self, x: np.ndarray) -> float:
        """Calculate distance to known optimum."""
        if self.optimal_point is not None:
            return np.linalg.norm(x - self.optimal_point)
        else:
            return float('inf')
class BenchmarkFunctions:
    """
    Collection of standard benchmark functions for optimization testing.
    Includes unimodal, multimodal, and separable/non-separable test functions
    commonly used in optimization literature.
    """
    @staticmethod
    def sphere(dimension: int = 2) -> OptimizationProblem:
        """
        Sphere function: f(x) = sum(x_i^2)
        Global minimum: f(0) = 0
        Properties: Unimodal, separable, convex
        """
        def objective(x):
            return np.sum(x**2)
        def gradient(x):
            return 2 * x
        def hessian(x):
            return 2 * np.eye(len(x))
        bounds = [(-10, 10)] * dimension
        optimal_point = np.zeros(dimension)
        return OptimizationProblem(
            name="Sphere",
            objective=objective,
            gradient=gradient,
            hessian=hessian,
            bounds=bounds,
            global_minimum=0.0,
            optimal_point=optimal_point,
            dimension=dimension
        )
    @staticmethod
    def rosenbrock(dimension: int = 2) -> OptimizationProblem:
        """
        Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
        Global minimum: f(1) = 0
        Properties: Unimodal, non-separable, valley-shaped
        """
        def objective(x):
            return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        def gradient(x):
            grad = np.zeros_like(x)
            grad[:-1] = -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
            grad[1:] += 200 * (x[1:] - x[:-1]**2)
            return grad
        bounds = [(-10, 10)] * dimension
        optimal_point = np.ones(dimension)
        return OptimizationProblem(
            name="Rosenbrock",
            objective=objective,
            gradient=gradient,
            bounds=bounds,
            global_minimum=0.0,
            optimal_point=optimal_point,
            dimension=dimension
        )
    @staticmethod
    def rastrigin(dimension: int = 2) -> OptimizationProblem:
        """
        Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
        Global minimum: f(0) = 0
        Properties: Multimodal, separable, many local minima
        """
        A = 10
        def objective(x):
            return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
        def gradient(x):
            return 2 * x + 2 * A * np.pi * np.sin(2 * np.pi * x)
        bounds = [(-5.12, 5.12)] * dimension
        optimal_point = np.zeros(dimension)
        return OptimizationProblem(
            name="Rastrigin",
            objective=objective,
            gradient=gradient,
            bounds=bounds,
            global_minimum=0.0,
            optimal_point=optimal_point,
            dimension=dimension
        )
    @staticmethod
    def ackley(dimension: int = 2) -> OptimizationProblem:
        """
        Ackley function: f(x) = -a*exp(-b*sqrt(sum(x_i^2)/n)) - exp(sum(cos(c*x_i))/n) + a + exp(1)
        Global minimum: f(0) = 0
        Properties: Multimodal, non-separable, many local minima
        """
        a, b, c = 20, 0.2, 2 * np.pi
        def objective(x):
            n = len(x)
            sum_sq = np.sum(x**2)
            sum_cos = np.sum(np.cos(c * x))
            return -a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.e
        def gradient(x):
            n = len(x)
            sum_sq = np.sum(x**2)
            sqrt_term = np.sqrt(sum_sq / n)
            # Partial derivatives
            grad = np.zeros_like(x)
            # First term derivative
            if sqrt_term > 0:
                grad += a * b * np.exp(-b * sqrt_term) * x / (n * sqrt_term)
            # Second term derivative
            grad += c * np.exp(np.sum(np.cos(c * x)) / n) * np.sin(c * x) / n
            return grad
        bounds = [(-32.768, 32.768)] * dimension
        optimal_point = np.zeros(dimension)
        return OptimizationProblem(
            name="Ackley",
            objective=objective,
            gradient=gradient,
            bounds=bounds,
            global_minimum=0.0,
            optimal_point=optimal_point,
            dimension=dimension
        )
    @staticmethod
    def griewank(dimension: int = 2) -> OptimizationProblem:
        """
        Griewank function: f(x) = sum(x_i^2)/4000 - prod(cos(x_i/sqrt(i+1))) + 1
        Global minimum: f(0) = 0
        Properties: Multimodal, non-separable
        """
        def objective(x):
            sum_sq = np.sum(x**2) / 4000
            prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
            return sum_sq - prod_cos + 1
        def gradient(x):
            n = len(x)
            indices = np.arange(1, n + 1)
            sqrt_indices = np.sqrt(indices)
            # Sum term gradient
            grad_sum = x / 2000
            # Product term gradient
            cos_terms = np.cos(x / sqrt_indices)
            prod_cos = np.prod(cos_terms)
            grad_prod = np.zeros_like(x)
            for i in range(n):
                if abs(cos_terms[i]) > 1e-10:
                    grad_prod[i] = prod_cos * np.sin(x[i] / sqrt_indices[i]) / (sqrt_indices[i] * cos_terms[i])
            return grad_sum + grad_prod
        bounds = [(-600, 600)] * dimension
        optimal_point = np.zeros(dimension)
        return OptimizationProblem(
            name="Griewank",
            objective=objective,
            gradient=gradient,
            bounds=bounds,
            global_minimum=0.0,
            optimal_point=optimal_point,
            dimension=dimension
        )
    @staticmethod
    def schwefel(dimension: int = 2) -> OptimizationProblem:
        """
        Schwefel function: f(x) = 418.9829*n - sum(x_i * sin(sqrt(|x_i|)))
        Global minimum: f(420.9687) ≈ 0
        Properties: Multimodal, separable
        """
        def objective(x):
            return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
        def gradient(x):
            grad = np.zeros_like(x)
            for i, xi in enumerate(x):
                if abs(xi) > 1e-10:
                    sqrt_abs_xi = np.sqrt(abs(xi))
                    grad[i] = -np.sin(sqrt_abs_xi) - 0.5 * np.cos(sqrt_abs_xi) * np.sign(xi)
            return grad
        bounds = [(-500, 500)] * dimension
        optimal_point = np.full(dimension, 420.9687)
        return OptimizationProblem(
            name="Schwefel",
            objective=objective,
            gradient=gradient,
            bounds=bounds,
            global_minimum=0.0,
            optimal_point=optimal_point,
            dimension=dimension
        )
    @staticmethod
    def get_all_functions(dimension: int = 2) -> List[OptimizationProblem]:
        """Get all benchmark functions."""
        return [
            BenchmarkFunctions.sphere(dimension),
            BenchmarkFunctions.rosenbrock(dimension),
            BenchmarkFunctions.rastrigin(dimension),
            BenchmarkFunctions.ackley(dimension),
            BenchmarkFunctions.griewank(dimension),
            BenchmarkFunctions.schwefel(dimension)
        ]
class ConvergenceAnalysis:
    """
    Tools for analyzing convergence behavior of optimization algorithms.
    Provides methods to track and analyze convergence properties,
    including convergence rate estimation and performance metrics.
    """
    @staticmethod
    def analyze_convergence(fitness_history: List[float],
                          true_optimum: Optional[float] = None,
                          tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Analyze convergence behavior from fitness history.
        Args:
            fitness_history: List of function values over iterations
            true_optimum: Known optimal value (optional)
            tolerance: Convergence tolerance
        Returns:
            Dictionary with convergence analysis results
        """
        if not fitness_history:
            return {}
        fitness_array = np.array(fitness_history)
        n_iterations = len(fitness_array)
        analysis = {
            'n_iterations': n_iterations,
            'initial_value': fitness_array[0],
            'final_value': fitness_array[-1],
            'best_value': np.min(fitness_array),
            'improvement': fitness_array[0] - fitness_array[-1],
            'relative_improvement': abs((fitness_array[0] - fitness_array[-1]) / (abs(fitness_array[0]) + 1e-10))
        }
        # Convergence detection
        converged_iteration = None
        if true_optimum is not None:
            errors = np.abs(fitness_array - true_optimum)
            converged_indices = np.where(errors < tolerance)[0]
            if len(converged_indices) > 0:
                converged_iteration = converged_indices[0]
        analysis['converged_iteration'] = converged_iteration
        # Convergence rate estimation (for last part of optimization)
        if n_iterations > 10:
            last_values = fitness_array[-10:]
            if true_optimum is not None:
                errors = np.abs(last_values - true_optimum)
                if np.all(errors > 0):
                    # Estimate convergence rate
                    ratios = errors[1:] / errors[:-1]
                    avg_ratio = np.mean(ratios[ratios > 0])
                    if avg_ratio < 1:
                        if avg_ratio > 0.1:
                            convergence_type = 'linear'
                        else:
                            convergence_type = 'superlinear'
                    else:
                        convergence_type = 'sublinear'
                    analysis['convergence_rate'] = avg_ratio
                    analysis['convergence_type'] = convergence_type
        # Stagnation detection
        if n_iterations > 5:
            recent_values = fitness_array[-5:]
            stagnation = np.std(recent_values) < tolerance
            analysis['stagnation'] = stagnation
        # Monotonicity check
        improvements = np.diff(fitness_array) < 0
        analysis['monotonic_improvement'] = np.all(improvements)
        analysis['improvement_rate'] = np.mean(improvements)
        return analysis
    @staticmethod
    def compare_algorithms(results: List[BenchmarkResult]) -> Dict[str, Any]:
        """
        Compare multiple algorithm results.
        Args:
            results: List of BenchmarkResult objects
        Returns:
            Comparison analysis
        """
        if not results:
            return {}
        # Group by algorithm
        algorithm_groups = {}
        for result in results:
            if result.algorithm_name not in algorithm_groups:
                algorithm_groups[result.algorithm_name] = []
            algorithm_groups[result.algorithm_name].append(result)
        comparison = {
            'algorithms': list(algorithm_groups.keys()),
            'n_algorithms': len(algorithm_groups),
            'n_results': len(results)
        }
        # Performance metrics per algorithm
        for alg_name, alg_results in algorithm_groups.items():
            metrics = {
                'avg_success_rate': np.mean([r.success_rate for r in alg_results]),
                'avg_iterations': np.mean([r.avg_iterations for r in alg_results]),
                'avg_function_evaluations': np.mean([r.avg_function_evaluations for r in alg_results]),
                'avg_execution_time': np.mean([r.avg_execution_time for r in alg_results]),
                'best_value_found': min([r.best_value_found for r in alg_results])
            }
            comparison[alg_name] = metrics
        # Rankings
        algorithms = list(algorithm_groups.keys())
        # Rank by success rate
        success_rates = [comparison[alg]['avg_success_rate'] for alg in algorithms]
        success_ranking = [algorithms[i] for i in np.argsort(success_rates)[::-1]]
        comparison['success_rate_ranking'] = success_ranking
        # Rank by efficiency (function evaluations)
        efficiency = [comparison[alg]['avg_function_evaluations'] for alg in algorithms]
        efficiency_ranking = [algorithms[i] for i in np.argsort(efficiency)]
        comparison['efficiency_ranking'] = efficiency_ranking
        # Rank by speed
        times = [comparison[alg]['avg_execution_time'] for alg in algorithms]
        speed_ranking = [algorithms[i] for i in np.argsort(times)]
        comparison['speed_ranking'] = speed_ranking
        return comparison
def benchmark_algorithm(algorithm, problems: List[OptimizationProblem],
                       n_runs: int = 10, max_iterations: int = 1000) -> List[BenchmarkResult]:
    """
    Benchmark an optimization algorithm on multiple test problems.
    Args:
        algorithm: Optimization algorithm to test
        problems: List of optimization problems
        n_runs: Number of runs per problem
        max_iterations: Maximum iterations per run
    Returns:
        List of benchmark results
    """
    results = []
    for problem in problems:
        print(f"Testing {algorithm.__class__.__name__} on {problem.name}...")
        run_results = []
        successful_runs = 0
        for run in range(n_runs):
            # Random starting point
            if problem.bounds:
                x0 = np.array([np.random.uniform(low, high) for low, high in problem.bounds])
            else:
                x0 = np.random.randn(problem.dimension or 2)
            try:
                # Set algorithm parameters
                if hasattr(algorithm, 'max_iterations'):
                    algorithm.max_iterations = max_iterations
                # Run optimization
                start_time = time.time()
                if hasattr(algorithm, 'minimize'):
                    if problem.bounds and hasattr(algorithm, 'minimize') and 'bounds' in algorithm.minimize.__code__.co_varnames:
                        result = algorithm.minimize(problem.objective, x0 if not hasattr(algorithm, 'minimize') or 'x0' not in algorithm.minimize.__code__.co_varnames else problem.bounds)
                    else:
                        if problem.gradient:
                            result = algorithm.minimize(problem.objective, x0, gradient=problem.gradient)
                        else:
                            result = algorithm.minimize(problem.objective, x0)
                else:
                    # For global optimization algorithms
                    result = algorithm.minimize(problem.objective, problem.bounds, x0=x0)
                execution_time = time.time() - start_time
                # Check success
                if problem.global_minimum is not None:
                    success = abs(result.fun - problem.global_minimum) < 1e-3
                else:
                    success = result.success if hasattr(result, 'success') else True
                if success:
                    successful_runs += 1
                run_results.append({
                    'result': result,
                    'success': success,
                    'execution_time': execution_time
                })
            except Exception as e:
                print(f"  Run {run+1} failed: {str(e)}")
                continue
        if run_results:
            # Calculate statistics
            success_rate = successful_runs / len(run_results)
            avg_iterations = np.mean([r['result'].nit for r in run_results])
            avg_function_evaluations = np.mean([r['result'].nfev for r in run_results])
            avg_execution_time = np.mean([r['execution_time'] for r in run_results])
            best_value_found = min([r['result'].fun for r in run_results])
            # Create benchmark result
            benchmark_result = BenchmarkResult(
                function_name=problem.name,
                algorithm_name=algorithm.__class__.__name__,
                result=run_results[0]['result'],  # First result as representative
                success_rate=success_rate,
                avg_iterations=int(avg_iterations),
                avg_function_evaluations=int(avg_function_evaluations),
                avg_execution_time=avg_execution_time,
                best_value_found=best_value_found
            )
            results.append(benchmark_result)
        print(f"  Success rate: {successful_runs}/{n_runs} ({100*successful_runs/max(n_runs,1):.1f}%)")
    return results
def demo():
    """Demonstrate optimization utilities and benchmark functions."""
    print("Optimization Utilities Demo")
    print("==========================")
    print()
    # Test benchmark functions
    print("1. Benchmark Functions")
    print("---------------------")
    functions = BenchmarkFunctions.get_all_functions(dimension=2)
    for func in functions[:3]:  # Test first 3 functions
        print(f"{func.name} function:")
        print(f"  Dimension: {func.dimension}")
        print(f"  Global minimum: {func.global_minimum}")
        print(f"  Optimal point: {func.optimal_point}")
        # Evaluate at random point
        x_test = np.array([1.0, 1.0])
        f_val = func.evaluate(x_test)
        print(f"  f([1, 1]) = {f_val:.6f}")
        if func.gradient:
            grad_val = func.evaluate_gradient(x_test)
            print(f"  ∇f([1, 1]) = [{grad_val[0]:.6f}, {grad_val[1]:.6f}]")
        print()
    # Test convergence analysis
    print("2. Convergence Analysis")
    print("----------------------")
    # Simulate convergence history
    np.random.seed(42)
    n_iter = 100
    true_optimum = 0.0
    # Linear convergence simulation
    linear_history = [10 * (0.9)**i + np.random.normal(0, 0.01) for i in range(n_iter)]
    analysis = ConvergenceAnalysis.analyze_convergence(linear_history, true_optimum)
    print("Simulated linear convergence:")
    print(f"  Initial value: {analysis['initial_value']:.6f}")
    print(f"  Final value: {analysis['final_value']:.6f}")
    print(f"  Improvement: {analysis['improvement']:.6f}")
    print(f"  Converged at iteration: {analysis.get('converged_iteration', 'Not converged')}")
    print(f"  Convergence type: {analysis.get('convergence_type', 'Unknown')}")
    print(f"  Monotonic improvement: {analysis.get('monotonic_improvement', False)}")
    print()
    print("Demo completed!")
if __name__ == "__main__":
    demo()