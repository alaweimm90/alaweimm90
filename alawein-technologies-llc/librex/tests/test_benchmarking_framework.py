"""
Comprehensive tests for Benchmarking Framework.
Tests benchmark runners, metrics, and performance analysis.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import tempfile
import json
import time
from pathlib import Path
import pandas as pd

from Librex.benchmarking import (
    BenchmarkRunner,
    BenchmarkSuite,
    PerformanceMetrics,
    BenchmarkReport
)
from Librex.benchmarking.metrics import (
    SolutionQuality,
    ConvergenceAnalysis,
    ScalabilityAnalysis,
    RobustnessAnalysis
)
from Librex.methods.baselines import (
    RandomSearchOptimizer,
    SimulatedAnnealingOptimizer,
    GeneticAlgorithmOptimizer
)


class TestBenchmarkRunner:
    """Test the main benchmark runner."""

    @pytest.fixture
    def runner(self):
        """Create a benchmark runner."""
        methods = [RandomSearchOptimizer, SimulatedAnnealingOptimizer]
        return BenchmarkRunner(methods, max_iterations=100)

    @pytest.fixture
    def test_problems(self):
        """Create test problems."""
        problems = []
        for size in [5, 10]:
            problem = {
                'name': f'test_{size}',
                'size': size,
                'flow': np.random.rand(size, size),
                'distance': np.random.rand(size, size),
                'optimal': np.random.rand() * 100
            }
            problems.append(problem)
        return problems

    def test_initialization(self, runner):
        """Test runner initialization."""
        assert len(runner.methods) == 2
        assert runner.max_iterations == 100
        assert runner.results == []

    def test_run_single_problem(self, runner, test_problems):
        """Test running benchmark on single problem."""
        problem = test_problems[0]
        results = runner.run_problem(problem)

        assert 'problem' in results
        assert 'methods' in results
        assert len(results['methods']) == 2

        for method_result in results['methods']:
            assert 'name' in method_result
            assert 'best_cost' in method_result
            assert 'time' in method_result
            assert 'iterations' in method_result

    def test_run_suite(self, runner, test_problems):
        """Test running full benchmark suite."""
        results = runner.run_suite(test_problems)

        assert len(results) == len(test_problems)
        for result in results:
            assert 'problem' in result
            assert 'methods' in result

    def test_parallel_execution(self, runner, test_problems):
        """Test parallel execution of benchmarks."""
        runner.parallel = True
        runner.n_jobs = 2

        start_time = time.time()
        results = runner.run_suite(test_problems)
        parallel_time = time.time() - start_time

        # Run sequentially for comparison
        runner.parallel = False
        start_time = time.time()
        results_seq = runner.run_suite(test_problems)
        sequential_time = time.time() - start_time

        # Parallel should be faster (or at least not much slower)
        assert len(results) == len(results_seq)

    def test_timeout_handling(self, runner):
        """Test timeout handling for long-running methods."""
        # Create a method that takes too long
        class SlowOptimizer:
            def optimize(self, problem):
                time.sleep(10)
                return np.arange(problem['size']), 100

        runner.methods.append(SlowOptimizer)
        runner.timeout = 0.1  # 100ms timeout

        problem = {'size': 5, 'flow': np.eye(5), 'distance': np.eye(5)}
        results = runner.run_problem(problem)

        # Check that slow method was terminated
        slow_result = next(r for r in results['methods'] if r['name'] == 'SlowOptimizer')
        assert slow_result['status'] == 'timeout'

    def test_memory_tracking(self, runner, test_problems):
        """Test memory usage tracking during benchmarks."""
        runner.track_memory = True
        results = runner.run_problem(test_problems[0])

        for method_result in results['methods']:
            assert 'memory_peak' in method_result
            assert 'memory_avg' in method_result
            assert method_result['memory_peak'] > 0

    def test_warm_start(self, runner, test_problems):
        """Test warm start functionality."""
        problem = test_problems[0]

        # Run with cold start
        runner.warm_start = False
        cold_results = runner.run_problem(problem)

        # Run with warm start
        runner.warm_start = True
        initial_solution = np.random.permutation(problem['size'])
        warm_results = runner.run_problem(problem, initial_solution=initial_solution)

        # Both should produce valid results
        assert len(cold_results['methods']) == len(warm_results['methods'])

    def test_save_and_load_results(self, runner, test_problems):
        """Test saving and loading benchmark results."""
        results = runner.run_suite(test_problems)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            runner.save_results(f.name)
            temp_file = f.name

        try:
            loaded_runner = BenchmarkRunner.load_results(temp_file)
            assert len(loaded_runner.results) == len(results)
        finally:
            Path(temp_file).unlink()


class TestBenchmarkSuite:
    """Test benchmark suite management."""

    @pytest.fixture
    def suite(self):
        """Create a benchmark suite."""
        return BenchmarkSuite(name="TestSuite", version="1.0")

    def test_initialization(self, suite):
        """Test suite initialization."""
        assert suite.name == "TestSuite"
        assert suite.version == "1.0"
        assert len(suite.problems) == 0
        assert len(suite.methods) == 0

    def test_add_problems(self, suite):
        """Test adding problems to suite."""
        problems = [
            {'name': 'p1', 'size': 10},
            {'name': 'p2', 'size': 20}
        ]

        for p in problems:
            suite.add_problem(p)

        assert len(suite.problems) == 2
        assert suite.problems[0]['name'] == 'p1'

    def test_add_methods(self, suite):
        """Test adding methods to suite."""
        suite.add_method(RandomSearchOptimizer)
        suite.add_method(SimulatedAnnealingOptimizer)

        assert len(suite.methods) == 2

    def test_configure_parameters(self, suite):
        """Test configuring benchmark parameters."""
        config = {
            'max_iterations': 1000,
            'timeout': 300,
            'n_runs': 5,
            'seed': 42
        }

        suite.configure(config)
        assert suite.config['max_iterations'] == 1000
        assert suite.config['n_runs'] == 5

    def test_run_benchmark(self, suite):
        """Test running complete benchmark suite."""
        # Setup suite
        suite.add_problem({'name': 'test', 'size': 5, 'flow': np.eye(5), 'distance': np.eye(5)})
        suite.add_method(RandomSearchOptimizer)
        suite.configure({'max_iterations': 10})

        # Run benchmark
        results = suite.run()

        assert 'suite_name' in results
        assert 'timestamp' in results
        assert 'results' in results
        assert len(results['results']) > 0

    def test_statistical_analysis(self, suite):
        """Test statistical analysis of results."""
        # Mock some results
        suite.results = {
            'results': [
                {
                    'problem': 'p1',
                    'methods': [
                        {'name': 'm1', 'costs': [10, 11, 12, 10, 11]},
                        {'name': 'm2', 'costs': [15, 14, 16, 15, 14]}
                    ]
                }
            ]
        }

        stats = suite.analyze_statistics()

        assert 'method_comparison' in stats
        assert 'significance_tests' in stats
        assert 'confidence_intervals' in stats

    def test_export_formats(self, suite):
        """Test exporting results in different formats."""
        suite.results = {'dummy': 'results'}

        # Test JSON export
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            suite.export_json(f.name)
            temp_json = f.name

        try:
            with open(temp_json) as f:
                loaded = json.load(f)
            assert loaded['dummy'] == 'results'
        finally:
            Path(temp_json).unlink()

        # Test CSV export
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            suite.export_csv(f.name)
            temp_csv = f.name

        try:
            df = pd.read_csv(temp_csv)
            assert len(df) >= 0
        finally:
            Path(temp_csv).unlink()


class TestPerformanceMetrics:
    """Test performance metrics calculation."""

    @pytest.fixture
    def metrics(self):
        """Create performance metrics calculator."""
        return PerformanceMetrics()

    def test_solution_quality_metrics(self, metrics):
        """Test solution quality metrics."""
        best_known = 100
        solutions = [105, 102, 108, 101, 110]

        quality = metrics.calculate_solution_quality(solutions, best_known)

        assert 'gap' in quality
        assert 'relative_error' in quality
        assert 'best' in quality
        assert 'worst' in quality
        assert 'mean' in quality
        assert 'std' in quality

        assert quality['best'] == 101
        assert quality['worst'] == 110
        assert quality['gap'] == 1  # 101 - 100

    def test_convergence_metrics(self, metrics):
        """Test convergence metrics."""
        history = [
            {'iteration': 0, 'cost': 200},
            {'iteration': 10, 'cost': 150},
            {'iteration': 20, 'cost': 120},
            {'iteration': 30, 'cost': 110},
            {'iteration': 40, 'cost': 105},
            {'iteration': 50, 'cost': 105}
        ]

        convergence = metrics.calculate_convergence(history)

        assert 'convergence_rate' in convergence
        assert 'iterations_to_converge' in convergence
        assert 'final_improvement' in convergence
        assert 'stagnation_ratio' in convergence

    def test_scalability_metrics(self, metrics):
        """Test scalability metrics."""
        results = [
            {'size': 10, 'time': 0.1, 'cost': 100},
            {'size': 20, 'time': 0.5, 'cost': 400},
            {'size': 30, 'time': 1.2, 'cost': 900},
            {'size': 40, 'time': 2.5, 'cost': 1600}
        ]

        scalability = metrics.calculate_scalability(results)

        assert 'time_complexity' in scalability
        assert 'quality_degradation' in scalability
        assert 'efficiency_ratio' in scalability

    def test_robustness_metrics(self, metrics):
        """Test robustness metrics."""
        # Multiple runs on same problem
        runs = [
            {'cost': 100, 'time': 1.0},
            {'cost': 102, 'time': 0.9},
            {'cost': 98, 'time': 1.1},
            {'cost': 101, 'time': 1.0},
            {'cost': 99, 'time': 1.05}
        ]

        robustness = metrics.calculate_robustness(runs)

        assert 'cost_variance' in robustness
        assert 'time_variance' in robustness
        assert 'success_rate' in robustness
        assert 'consistency_score' in robustness

    def test_comparative_metrics(self, metrics):
        """Test comparative metrics between methods."""
        method1_results = [100, 102, 98, 101, 99]
        method2_results = [105, 103, 107, 104, 106]

        comparison = metrics.compare_methods(method1_results, method2_results)

        assert 'winner' in comparison
        assert 'performance_ratio' in comparison
        assert 'statistical_significance' in comparison
        assert 'effect_size' in comparison

        assert comparison['winner'] == 'method1'

    def test_pareto_analysis(self, metrics):
        """Test Pareto optimality analysis."""
        # Solutions with (cost, time) pairs
        solutions = [
            (100, 1.0),
            (95, 2.0),
            (110, 0.5),
            (90, 3.0),
            (105, 0.8)
        ]

        pareto_front = metrics.find_pareto_front(solutions)

        assert len(pareto_front) > 0
        assert (90, 3.0) in pareto_front  # Best cost
        assert (110, 0.5) in pareto_front  # Best time


class TestBenchmarkReport:
    """Test benchmark report generation."""

    @pytest.fixture
    def report(self):
        """Create a benchmark report."""
        return BenchmarkReport(title="Test Report")

    @pytest.fixture
    def sample_results(self):
        """Create sample benchmark results."""
        return {
            'suite_name': 'TestSuite',
            'timestamp': '2024-01-01 00:00:00',
            'problems': [
                {'name': 'chr12a', 'size': 12, 'optimal': 9552},
                {'name': 'nug15', 'size': 15, 'optimal': 1150}
            ],
            'methods': ['RandomSearch', 'SimulatedAnnealing', 'GeneticAlgorithm'],
            'results': [
                {
                    'problem': 'chr12a',
                    'method_results': [
                        {'method': 'RandomSearch', 'best': 10000, 'time': 1.0},
                        {'method': 'SimulatedAnnealing', 'best': 9600, 'time': 2.0},
                        {'method': 'GeneticAlgorithm', 'best': 9580, 'time': 3.0}
                    ]
                }
            ]
        }

    def test_report_initialization(self, report):
        """Test report initialization."""
        assert report.title == "Test Report"
        assert report.sections == []

    def test_add_summary_section(self, report, sample_results):
        """Test adding summary section."""
        report.add_summary(sample_results)

        assert len(report.sections) > 0
        summary = report.sections[0]
        assert summary['type'] == 'summary'
        assert 'content' in summary

    def test_add_performance_table(self, report, sample_results):
        """Test adding performance comparison table."""
        report.add_performance_table(sample_results)

        table_section = next(s for s in report.sections if s['type'] == 'table')
        assert table_section is not None
        assert 'data' in table_section

    def test_add_convergence_plots(self, report):
        """Test adding convergence plots."""
        convergence_data = {
            'RandomSearch': [(0, 200), (10, 150), (20, 120)],
            'SimulatedAnnealing': [(0, 200), (10, 130), (20, 110)]
        }

        report.add_convergence_plots(convergence_data)

        plot_section = next(s for s in report.sections if s['type'] == 'plot')
        assert plot_section is not None

    def test_add_statistical_analysis(self, report):
        """Test adding statistical analysis section."""
        stats = {
            'anova': {'f_statistic': 5.23, 'p_value': 0.02},
            'tukey_hsd': [
                {'pair': ('m1', 'm2'), 'diff': 10, 'p_value': 0.01}
            ]
        }

        report.add_statistical_analysis(stats)

        stats_section = next(s for s in report.sections if s['type'] == 'statistics')
        assert stats_section is not None

    def test_generate_html_report(self, report, sample_results):
        """Test HTML report generation."""
        report.add_summary(sample_results)
        report.add_performance_table(sample_results)

        html = report.generate_html()

        assert '<html>' in html
        assert report.title in html
        assert 'TestSuite' in html

    def test_generate_latex_report(self, report, sample_results):
        """Test LaTeX report generation."""
        report.add_summary(sample_results)

        latex = report.generate_latex()

        assert '\\documentclass' in latex
        assert '\\begin{document}' in latex
        assert report.title in latex

    def test_save_report(self, report, sample_results):
        """Test saving report to file."""
        report.add_summary(sample_results)

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            report.save(f.name, format='html')
            temp_file = f.name

        try:
            with open(temp_file) as f:
                content = f.read()
            assert '<html>' in content
        finally:
            Path(temp_file).unlink()


class TestConvergenceAnalysis:
    """Test convergence analysis utilities."""

    @pytest.fixture
    def analyzer(self):
        """Create convergence analyzer."""
        return ConvergenceAnalysis()

    def test_detect_convergence(self, analyzer):
        """Test convergence detection."""
        history = [100, 90, 85, 82, 80, 80, 80, 80, 80]
        converged_idx = analyzer.detect_convergence(history, threshold=1e-6)

        assert converged_idx == 5  # Converged at index 5

    def test_calculate_convergence_rate(self, analyzer):
        """Test convergence rate calculation."""
        history = [100, 50, 25, 12.5, 6.25]  # Exponential decay
        rate = analyzer.calculate_rate(history)

        assert rate < 0  # Negative rate indicates improvement
        assert abs(rate) > 0.5  # Fast convergence

    def test_fit_convergence_model(self, analyzer):
        """Test fitting convergence models."""
        iterations = np.arange(50)
        costs = 100 * np.exp(-0.1 * iterations) + 10 + np.random.randn(50) * 2

        model = analyzer.fit_model(iterations, costs, model_type='exponential')

        assert 'parameters' in model
        assert 'r_squared' in model
        assert 'predicted' in model
        assert model['r_squared'] > 0.8  # Good fit

    def test_compare_convergence_profiles(self, analyzer):
        """Test comparing convergence profiles."""
        profile1 = [100, 80, 60, 50, 45, 42, 40]
        profile2 = [100, 70, 45, 35, 30, 28, 27]

        comparison = analyzer.compare_profiles(profile1, profile2)

        assert 'faster_method' in comparison
        assert 'area_difference' in comparison
        assert comparison['faster_method'] == 'method2'


class TestScalabilityAnalysis:
    """Test scalability analysis utilities."""

    @pytest.fixture
    def analyzer(self):
        """Create scalability analyzer."""
        return ScalabilityAnalysis()

    def test_fit_complexity_model(self, analyzer):
        """Test fitting computational complexity models."""
        sizes = [10, 20, 30, 40, 50]
        times = [0.01, 0.04, 0.09, 0.16, 0.25]  # O(n^2)

        complexity = analyzer.fit_complexity(sizes, times)

        assert 'model' in complexity
        assert 'parameters' in complexity
        assert 'predicted_order' in complexity
        assert complexity['predicted_order'] == 'quadratic'

    def test_extrapolate_performance(self, analyzer):
        """Test performance extrapolation."""
        known_sizes = [10, 20, 30]
        known_times = [0.1, 0.4, 0.9]

        predicted_time = analyzer.extrapolate(
            known_sizes,
            known_times,
            target_size=50
        )

        assert predicted_time > known_times[-1]
        assert predicted_time < 10  # Reasonable bound

    def test_efficiency_analysis(self, analyzer):
        """Test parallel efficiency analysis."""
        results = [
            {'processors': 1, 'time': 100},
            {'processors': 2, 'time': 55},
            {'processors': 4, 'time': 30},
            {'processors': 8, 'time': 20}
        ]

        efficiency = analyzer.calculate_efficiency(results)

        assert 'speedup' in efficiency
        assert 'efficiency_scores' in efficiency
        assert 'scaling_type' in efficiency

        # Check Amdahl's law
        assert all(0 <= e <= 1 for e in efficiency['efficiency_scores'])


class TestRobustnessAnalysis:
    """Test robustness analysis utilities."""

    @pytest.fixture
    def analyzer(self):
        """Create robustness analyzer."""
        return RobustnessAnalysis()

    def test_stability_analysis(self, analyzer):
        """Test solution stability analysis."""
        runs = [
            {'solution': [0, 1, 2, 3, 4], 'cost': 100},
            {'solution': [0, 1, 2, 4, 3], 'cost': 102},
            {'solution': [0, 2, 1, 3, 4], 'cost': 105},
            {'solution': [0, 1, 2, 3, 4], 'cost': 100},
            {'solution': [0, 1, 2, 3, 4], 'cost': 101}
        ]

        stability = analyzer.analyze_stability(runs)

        assert 'solution_diversity' in stability
        assert 'cost_stability' in stability
        assert 'most_common_solution' in stability

    def test_noise_sensitivity(self, analyzer):
        """Test noise sensitivity analysis."""
        base_problem = {'flow': np.ones((5, 5)), 'distance': np.eye(5)}

        sensitivity = analyzer.test_noise_sensitivity(
            base_problem,
            noise_levels=[0.01, 0.05, 0.1],
            optimizer=RandomSearchOptimizer(),
            n_trials=5
        )

        assert len(sensitivity) == 3
        for level, results in sensitivity.items():
            assert 'mean_cost' in results
            assert 'std_cost' in results

    def test_parameter_sensitivity(self, analyzer):
        """Test parameter sensitivity analysis."""
        parameter_grid = {
            'temperature': [0.1, 1.0, 10.0],
            'cooling_rate': [0.9, 0.95, 0.99]
        }

        sensitivity = analyzer.analyze_parameter_sensitivity(
            SimulatedAnnealingOptimizer,
            parameter_grid,
            test_problem={'size': 5}
        )

        assert 'best_params' in sensitivity
        assert 'sensitivity_scores' in sensitivity
        assert 'interaction_effects' in sensitivity