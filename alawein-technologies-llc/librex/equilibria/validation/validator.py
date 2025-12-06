"""
Comprehensive validation framework for optimization methods.

This module validates optimization algorithms against problems with known
analytical solutions to ensure correctness and performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import time
from pathlib import Path
import traceback
from datetime import datetime

from .test_problems import (
    TestProblem, get_test_problems, get_problem_categories,
    TEST_PROBLEM_REGISTRY
)


@dataclass
class ValidationResult:
    """Result from validating a single optimization run."""

    problem_name: str
    problem_dimension: int
    method_name: str
    optimal_solution: np.ndarray
    found_solution: np.ndarray
    optimal_value: float
    found_value: float
    solution_error: float
    objective_error: float
    relative_error: float
    passed: bool
    iterations: int
    evaluations: int
    runtime: float
    success_tolerance: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class OptimizationValidator:
    """
    Comprehensive validator for optimization methods.

    Tests optimization algorithms against analytical problems to verify:
    - Correctness: Finding known global optima
    - Efficiency: Number of iterations/evaluations required
    - Robustness: Performance across problem types
    - Scalability: Performance vs problem dimension
    """

    def __init__(self,
                 objective_tolerance: float = 1e-3,
                 solution_tolerance: float = 1e-2,
                 relative_tolerance: float = 0.01,
                 max_iterations: int = 10000,
                 verbose: bool = True):
        """
        Initialize the validator.

        Args:
            objective_tolerance: Absolute tolerance for objective value
            solution_tolerance: Tolerance for solution distance (L2 norm)
            relative_tolerance: Relative tolerance for objective value
            max_iterations: Maximum iterations allowed
            verbose: Whether to print progress
        """
        self.objective_tolerance = objective_tolerance
        self.solution_tolerance = solution_tolerance
        self.relative_tolerance = relative_tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Import optimization module dynamically
        self._import_optimization_module()

    def _import_optimization_module(self):
        """Import the optimization module and methods."""
        try:
            from Librex.optimize import optimize
            from Librex.adapters import ContinuousAdapter
            from Librex.adapters.qap_adapter import QAPAdapter

            self.optimize = optimize
            self.ContinuousAdapter = ContinuousAdapter
            self.QAPAdapter = QAPAdapter

            # List of available methods
            self.available_methods = [
                'random_search',
                'simulated_annealing',
                'genetic_algorithm',
                'local_search',
                'tabu_search',
                'pso',  # Particle Swarm Optimization
                'aco',  # Ant Colony Optimization
                'ils',  # Iterated Local Search
                'vns',  # Variable Neighborhood Search
                'grasp',  # GRASP
            ]

        except ImportError as e:
            print(f"Warning: Could not import optimization module: {e}")
            self.optimize = None
            self.available_methods = []

    def validate_method(self,
                       method_name: str,
                       dimensions: List[int] = None,
                       problem_names: List[str] = None,
                       n_runs: int = 1,
                       seed: int = 42) -> Dict[str, Any]:
        """
        Validate a single optimization method.

        Args:
            method_name: Name of the optimization method
            dimensions: List of dimensions to test
            problem_names: List of problem names to test (None for all)
            n_runs: Number of runs per problem
            seed: Random seed for reproducibility

        Returns:
            Dictionary with validation results and statistics
        """
        if self.optimize is None:
            raise RuntimeError("Optimization module not available")

        if dimensions is None:
            dimensions = [2, 5, 10]

        # Get test problems
        test_problems = get_test_problems(dimensions)

        # Filter by problem names if specified
        if problem_names:
            test_problems = [p for p in test_problems
                           if any(name in p.name for name in problem_names)]

        results = []
        np.random.seed(seed)

        for problem in test_problems:
            if self.verbose:
                print(f"Testing {method_name} on {problem.name}...")

            for run in range(n_runs):
                try:
                    result = self._run_single_validation(
                        method_name, problem, seed + run
                    )
                    results.append(result)
                except Exception as e:
                    if self.verbose:
                        print(f"  Error: {e}")
                    # Record failed run
                    results.append(self._create_failed_result(
                        method_name, problem, str(e)
                    ))

        # Compute statistics
        df_results = pd.DataFrame([r.__dict__ for r in results])

        stats = {
            'method': method_name,
            'total_runs': len(results),
            'passed': sum(r.passed for r in results),
            'failed': sum(not r.passed for r in results),
            'success_rate': sum(r.passed for r in results) / len(results),
            'results': results,
            'dataframe': df_results,
            'summary': self._compute_summary_statistics(df_results),
        }

        return stats

    def _run_single_validation(self,
                              method_name: str,
                              problem: TestProblem,
                              seed: int) -> ValidationResult:
        """Run a single validation test."""

        # Create adapter for the problem
        class ProblemAdapter:
            """Adapter to make TestProblem compatible with optimize()."""

            def __init__(self, test_problem):
                self.problem = test_problem
                self.n = test_problem.dimension

                # Set bounds
                if isinstance(test_problem.bounds, tuple):
                    # Same bounds for all dimensions
                    self.bounds = [(test_problem.bounds[0], test_problem.bounds[1])
                                 for _ in range(self.n)]
                else:
                    self.bounds = test_problem.bounds

            def evaluate(self, x):
                """Evaluate the objective function."""
                return self.problem.evaluate(x)

            def get_random_solution(self):
                """Generate a random solution within bounds."""
                return np.array([np.random.uniform(low, high)
                               for low, high in self.bounds])

            def get_neighbor(self, x, step_size=0.1):
                """Get a neighbor solution."""
                neighbor = x + np.random.randn(len(x)) * step_size
                # Clip to bounds
                for i, (low, high) in enumerate(self.bounds):
                    neighbor[i] = np.clip(neighbor[i], low, high)
                return neighbor

        adapter = ProblemAdapter(problem)

        # Run optimization
        start_time = time.time()

        # Set up optimization parameters
        opt_params = {
            'max_iterations': self.max_iterations,
            'seed': seed,
        }

        # Method-specific parameters
        if method_name in ['genetic_algorithm', 'pso', 'aco']:
            opt_params['population_size'] = min(50, 10 * problem.dimension)

        if method_name == 'simulated_annealing':
            opt_params['initial_temperature'] = 100.0
            opt_params['cooling_rate'] = 0.95

        if method_name == 'tabu_search':
            opt_params['tabu_tenure'] = min(20, problem.dimension * 2)

        # Run optimization
        result = self.optimize(
            problem=adapter,
            adapter=adapter,
            method=method_name,
            **opt_params
        )

        runtime = time.time() - start_time

        # Extract results
        found_solution = result['solution']
        found_value = result['objective']

        # Calculate errors
        if problem.optimal_solution is not None:
            solution_error = np.linalg.norm(
                found_solution - problem.optimal_solution
            )
        else:
            solution_error = float('inf')  # Unknown optimal solution

        objective_error = abs(found_value - problem.optimal_value)

        if abs(problem.optimal_value) > 1e-10:
            relative_error = objective_error / abs(problem.optimal_value)
        else:
            relative_error = objective_error

        # Check if passed
        passed = (
            objective_error < self.objective_tolerance or
            relative_error < self.relative_tolerance
        )

        # For problems with unknown exact solution, use threshold
        if problem.optimal_solution is None:
            passed = objective_error < self.objective_tolerance * 10

        return ValidationResult(
            problem_name=problem.name,
            problem_dimension=problem.dimension,
            method_name=method_name,
            optimal_solution=problem.optimal_solution if problem.optimal_solution is not None else np.array([]),
            found_solution=found_solution,
            optimal_value=problem.optimal_value,
            found_value=found_value,
            solution_error=solution_error,
            objective_error=objective_error,
            relative_error=relative_error,
            passed=passed,
            iterations=result.get('metadata', {}).get('iterations', 0),
            evaluations=result.get('metadata', {}).get('evaluations', 0),
            runtime=runtime,
            success_tolerance=self.objective_tolerance,
            metadata=result.get('metadata', {})
        )

    def _create_failed_result(self,
                            method_name: str,
                            problem: TestProblem,
                            error_msg: str) -> ValidationResult:
        """Create a result object for a failed run."""
        return ValidationResult(
            problem_name=problem.name,
            problem_dimension=problem.dimension,
            method_name=method_name,
            optimal_solution=problem.optimal_solution if problem.optimal_solution is not None else np.array([]),
            found_solution=np.array([]),
            optimal_value=problem.optimal_value,
            found_value=float('inf'),
            solution_error=float('inf'),
            objective_error=float('inf'),
            relative_error=float('inf'),
            passed=False,
            iterations=0,
            evaluations=0,
            runtime=0.0,
            success_tolerance=self.objective_tolerance,
            metadata={'error': error_msg}
        )

    def _compute_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics from results dataframe."""
        if df.empty:
            return {}

        stats = {
            'success_rate': df['passed'].mean(),
            'mean_objective_error': df['objective_error'].mean(),
            'median_objective_error': df['objective_error'].median(),
            'mean_iterations': df['iterations'].mean(),
            'mean_evaluations': df['evaluations'].mean(),
            'mean_runtime': df['runtime'].mean(),
        }

        # Group by problem dimension
        if 'problem_dimension' in df.columns:
            dim_stats = df.groupby('problem_dimension').agg({
                'passed': 'mean',
                'objective_error': 'mean',
                'iterations': 'mean',
                'runtime': 'mean',
            }).to_dict()
            stats['by_dimension'] = dim_stats

        # Group by problem name
        if 'problem_name' in df.columns:
            problem_stats = df.groupby('problem_name').agg({
                'passed': 'mean',
                'objective_error': 'mean',
            }).to_dict()
            stats['by_problem'] = problem_stats

        return stats

    def compare_methods(self,
                       methods: List[str] = None,
                       dimensions: List[int] = None,
                       problem_names: List[str] = None,
                       n_runs: int = 1,
                       seed: int = 42) -> pd.DataFrame:
        """
        Compare multiple optimization methods.

        Args:
            methods: List of method names to compare (None for all)
            dimensions: List of dimensions to test
            problem_names: List of problem names to test
            n_runs: Number of runs per problem per method
            seed: Random seed

        Returns:
            DataFrame with comparison results
        """
        if methods is None:
            methods = self.available_methods

        all_results = []

        for method in methods:
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Validating {method}")
                print(f"{'='*50}")

            validation = self.validate_method(
                method, dimensions, problem_names, n_runs, seed
            )

            # Add results to list
            for result in validation['results']:
                all_results.append(result.__dict__)

        # Create dataframe
        df = pd.DataFrame(all_results)

        return df

    def generate_report(self,
                       results: pd.DataFrame,
                       output_path: str,
                       title: str = "Librex Validation Report"):
        """
        Generate a comprehensive validation report.

        Args:
            results: DataFrame with validation results
            output_path: Path to save the report
            title: Report title
        """
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # 1. Success rate by method
        ax1 = plt.subplot(2, 3, 1)
        success_by_method = results.groupby('method_name')['passed'].mean()
        success_by_method.plot(kind='bar', ax=ax1)
        ax1.set_title('Success Rate by Method')
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim([0, 1])
        ax1.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='70% threshold')
        ax1.legend()

        # 2. Objective error distribution (log scale)
        ax2 = plt.subplot(2, 3, 2)
        for method in results['method_name'].unique():
            method_results = results[results['method_name'] == method]
            errors = method_results['objective_error'].values
            errors = errors[errors < float('inf')]  # Remove infinities
            if len(errors) > 0:
                ax2.hist(np.log10(errors + 1e-10), alpha=0.5, label=method, bins=20)
        ax2.set_title('Objective Error Distribution (log scale)')
        ax2.set_xlabel('log10(Error)')
        ax2.set_ylabel('Frequency')
        ax2.legend()

        # 3. Success rate vs dimension
        ax3 = plt.subplot(2, 3, 3)
        for method in results['method_name'].unique():
            method_results = results[results['method_name'] == method]
            success_by_dim = method_results.groupby('problem_dimension')['passed'].mean()
            ax3.plot(success_by_dim.index, success_by_dim.values,
                    marker='o', label=method)
        ax3.set_title('Success Rate vs Problem Dimension')
        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('Success Rate')
        ax3.legend()
        ax3.grid(True)

        # 4. Function evaluations by method
        ax4 = plt.subplot(2, 3, 4)
        evaluations_data = [
            results[results['method_name'] == method]['evaluations'].values
            for method in results['method_name'].unique()
        ]
        ax4.boxplot(evaluations_data, labels=results['method_name'].unique())
        ax4.set_title('Function Evaluations by Method')
        ax4.set_xlabel('Method')
        ax4.set_ylabel('Evaluations')
        ax4.set_yscale('log')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

        # 5. Runtime by method
        ax5 = plt.subplot(2, 3, 5)
        runtime_data = [
            results[results['method_name'] == method]['runtime'].values
            for method in results['method_name'].unique()
        ]
        ax5.boxplot(runtime_data, labels=results['method_name'].unique())
        ax5.set_title('Runtime by Method (seconds)')
        ax5.set_xlabel('Method')
        ax5.set_ylabel('Runtime (s)')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

        # 6. Heatmap of success rates
        ax6 = plt.subplot(2, 3, 6)
        pivot = results.pivot_table(
            values='passed',
            index='problem_name',
            columns='method_name',
            aggfunc='mean'
        )

        # Only show first 10 problems for readability
        if len(pivot) > 10:
            pivot = pivot.iloc[:10]

        im = ax6.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax6.set_xticks(range(len(pivot.columns)))
        ax6.set_yticks(range(len(pivot.index)))
        ax6.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax6.set_yticklabels(pivot.index)
        ax6.set_title('Success Rate Heatmap')
        plt.colorbar(im, ax=ax6)

        plt.tight_layout()

        # Save figure
        fig_path = output_path.replace('.html', '.png')
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')

        # Generate HTML report
        html = self._generate_html_report(results, fig_path, title)

        # Save HTML
        with open(output_path, 'w') as f:
            f.write(html)

        if self.verbose:
            print(f"Report saved to {output_path}")

    def _generate_html_report(self,
                             results: pd.DataFrame,
                             fig_path: str,
                             title: str) -> str:
        """Generate HTML report content."""

        # Compute overall statistics
        total_tests = len(results)
        total_passed = results['passed'].sum()
        overall_success = results['passed'].mean()

        # Method summary
        method_summary = results.groupby('method_name').agg({
            'passed': ['sum', 'count', 'mean'],
            'objective_error': 'mean',
            'runtime': 'mean',
            'evaluations': 'mean',
        })

        # Problem summary
        problem_summary = results.groupby('problem_name').agg({
            'passed': 'mean',
            'objective_error': 'mean',
        }).sort_values(('passed', 'mean'), ascending=False)

        # Dimension summary
        dim_summary = results.groupby('problem_dimension').agg({
            'passed': 'mean',
            'objective_error': 'mean',
            'runtime': 'mean',
        })

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #333;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #555;
                    margin-top: 30px;
                }}
                .summary {{
                    background-color: white;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px 20px;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #4CAF50;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #666;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    background-color: white;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .success {{
                    color: green;
                    font-weight: bold;
                }}
                .failure {{
                    color: red;
                    font-weight: bold;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>

            <div class="summary">
                <h2>Overall Summary</h2>
                <div class="metric">
                    <div class="metric-value">{total_tests}</div>
                    <div class="metric-label">Total Tests</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{total_passed}</div>
                    <div class="metric-label">Tests Passed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{overall_success:.1%}</div>
                    <div class="metric-label">Success Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(results['method_name'].unique())}</div>
                    <div class="metric-label">Methods Tested</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(results['problem_name'].unique())}</div>
                    <div class="metric-label">Problems Tested</div>
                </div>
            </div>

            <h2>Visualization</h2>
            <img src="{Path(fig_path).name}" alt="Validation Results">

            <h2>Method Performance Summary</h2>
            {method_summary.to_html()}

            <h2>Problem Difficulty Ranking</h2>
            {problem_summary.head(20).to_html()}

            <h2>Performance by Dimension</h2>
            {dim_summary.to_html()}

            <h2>Detailed Results</h2>
            <details>
                <summary>Click to expand detailed results table</summary>
                {results.to_html(max_rows=100)}
            </details>

            <hr>
            <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </body>
        </html>
        """

        return html

    def validate_convergence(self,
                           method_name: str,
                           problem: TestProblem,
                           n_runs: int = 10) -> Dict[str, Any]:
        """
        Validate that a method converges consistently.

        Args:
            method_name: Name of the method to test
            problem: Test problem instance
            n_runs: Number of runs

        Returns:
            Convergence statistics
        """
        results = []

        for i in range(n_runs):
            result = self._run_single_validation(method_name, problem, seed=42 + i)
            results.append(result)

        objective_values = [r.found_value for r in results]

        return {
            'mean_objective': np.mean(objective_values),
            'std_objective': np.std(objective_values),
            'min_objective': np.min(objective_values),
            'max_objective': np.max(objective_values),
            'convergence_rate': sum(r.passed for r in results) / n_runs,
            'results': results,
        }

    def export_results(self, results: pd.DataFrame, filepath: str):
        """Export results to various formats."""
        if filepath.endswith('.csv'):
            results.to_csv(filepath, index=False)
        elif filepath.endswith('.json'):
            results.to_json(filepath, orient='records', indent=2)
        elif filepath.endswith('.xlsx'):
            results.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        if self.verbose:
            print(f"Results exported to {filepath}")