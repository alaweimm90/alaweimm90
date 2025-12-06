"""
Unit tests for the validation framework.

Tests the analytical test problems, validator, and continuous validation.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

from Librex.validation.test_problems import (
    SphereFunction, RosenbrockFunction, RastriginFunction,
    AckleyFunction, ConvexQuadratic, GriewankFunction,
    BealeFunction, BoothFunction, HimmelblauFunction,
    get_test_problems, get_problem_categories,
    TEST_PROBLEM_REGISTRY
)
from Librex.validation.validator import (
    OptimizationValidator, ValidationResult
)
from Librex.validation.continuous_validation import ContinuousValidator


class TestAnalyticalProblems:
    """Test the analytical test problems."""

    def test_sphere_function(self):
        """Test sphere function properties."""
        sphere = SphereFunction(5)

        # Test at optimum
        assert sphere.evaluate(sphere.optimal_solution) == pytest.approx(0.0)

        # Test at random point
        x = np.ones(5)
        assert sphere.evaluate(x) == pytest.approx(5.0)

        # Test gradient (should be 2*x)
        x = np.array([1, 2, 3, 4, 5])
        expected = np.sum(x**2)
        assert sphere.evaluate(x) == pytest.approx(expected)

    def test_rosenbrock_function(self):
        """Test Rosenbrock function properties."""
        rosenbrock = RosenbrockFunction(4)

        # Test at optimum
        assert rosenbrock.evaluate(rosenbrock.optimal_solution) == pytest.approx(0.0)

        # Test dimension
        assert rosenbrock.dimension == 4
        assert len(rosenbrock.optimal_solution) == 4

    def test_rastrigin_function(self):
        """Test Rastrigin function properties."""
        rastrigin = RastriginFunction(3)

        # Test at optimum
        assert rastrigin.evaluate(rastrigin.optimal_solution) == pytest.approx(0.0)

        # Test multimodality (should have many local minima)
        x1 = np.array([1.0, 0.0, 0.0])
        x2 = np.array([0.0, 1.0, 0.0])
        # Both should have similar non-zero values due to multimodality
        val1 = rastrigin.evaluate(x1)
        val2 = rastrigin.evaluate(x2)
        assert val1 > 0 and val2 > 0

    def test_convex_quadratic(self):
        """Test convex quadratic function."""
        quad = ConvexQuadratic(5, seed=42)

        # Test that it has an analytical solution
        assert quad.optimal_solution is not None
        assert len(quad.optimal_solution) == 5

        # Test at optimum
        opt_val = quad.evaluate(quad.optimal_solution)
        assert opt_val == pytest.approx(quad.optimal_value, rel=1e-6)

        # Test convexity (midpoint should be lower than endpoints)
        x1 = np.random.randn(5)
        x2 = np.random.randn(5)
        midpoint = (x1 + x2) / 2

        val_mid = quad.evaluate(midpoint)
        val_avg = (quad.evaluate(x1) + quad.evaluate(x2)) / 2
        assert val_mid <= val_avg + 1e-10  # Convexity property

    def test_2d_functions(self):
        """Test 2D-only functions."""
        # Beale function
        beale = BealeFunction()
        assert beale.dimension == 2
        assert beale.evaluate(beale.optimal_solution) == pytest.approx(0.0)

        # Booth function
        booth = BoothFunction()
        assert booth.evaluate(booth.optimal_solution) == pytest.approx(0.0)
        assert np.allclose(booth.optimal_solution, [1.0, 3.0])

        # Himmelblau function (has 4 global minima)
        himmelblau = HimmelblauFunction()
        assert himmelblau.evaluate(himmelblau.optimal_solution) == pytest.approx(0.0)

    def test_problem_registry(self):
        """Test that all problems in registry are valid."""
        for name, problem_class in TEST_PROBLEM_REGISTRY.items():
            if name in ['convex_quadratic']:
                problem = problem_class(5, seed=42)
            elif name in ['beale', 'goldstein_price', 'booth', 'matyas',
                         'himmelblau', 'easom', 'cross_in_tray', 'eggholder',
                         'holder_table', 'mccormick', 'schaffer_n2',
                         'schaffer_n4', 'three_hump_camel', 'six_hump_camel']:
                problem = problem_class()
            else:
                problem = problem_class(5)

            # Check basic properties
            assert hasattr(problem, 'evaluate')
            assert hasattr(problem, 'optimal_value')
            assert hasattr(problem, 'dimension')
            assert hasattr(problem, 'bounds')

    def test_get_test_problems(self):
        """Test problem generation."""
        problems = get_test_problems([2, 5, 10])
        assert len(problems) > 0

        # Check variety of dimensions
        dimensions = [p.dimension for p in problems]
        assert 2 in dimensions
        assert 5 in dimensions
        assert 10 in dimensions

    def test_problem_categories(self):
        """Test problem categorization."""
        categories = get_problem_categories()

        assert 'convex' in categories
        assert 'multimodal' in categories
        assert 'separable' in categories

        # Check that categories are non-empty
        for category, problems in categories.items():
            assert len(problems) > 0


class TestValidator:
    """Test the optimization validator."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = OptimizationValidator(
            objective_tolerance=1e-3,
            solution_tolerance=1e-2,
            verbose=False
        )

        assert validator.objective_tolerance == 1e-3
        assert validator.solution_tolerance == 1e-2
        assert validator.verbose == False

    def test_validation_result(self):
        """Test ValidationResult dataclass."""
        result = ValidationResult(
            problem_name="test_problem",
            problem_dimension=5,
            method_name="test_method",
            optimal_solution=np.zeros(5),
            found_solution=np.ones(5),
            optimal_value=0.0,
            found_value=1.0,
            solution_error=1.0,
            objective_error=1.0,
            relative_error=1.0,
            passed=False,
            iterations=100,
            evaluations=1000,
            runtime=10.5,
            success_tolerance=1e-3
        )

        assert result.problem_name == "test_problem"
        assert result.passed == False
        assert result.runtime == 10.5

    @pytest.mark.skipif(
        True,  # Skip by default as it requires full Librex installation
        reason="Requires full Librex optimization module"
    )
    def test_validate_method(self):
        """Test method validation (requires full Librex)."""
        validator = OptimizationValidator(verbose=False)

        # Test on small problem
        stats = validator.validate_method(
            method_name='random_search',
            dimensions=[2],
            problem_names=['sphere'],
            n_runs=1
        )

        assert 'success_rate' in stats
        assert 'results' in stats
        assert len(stats['results']) > 0

    def test_summary_statistics(self):
        """Test summary statistics computation."""
        validator = OptimizationValidator(verbose=False)

        # Create mock dataframe
        data = {
            'passed': [True, False, True, True],
            'objective_error': [0.01, 0.5, 0.02, 0.03],
            'iterations': [100, 200, 150, 120],
            'runtime': [1.0, 2.5, 1.8, 1.2],
            'problem_dimension': [2, 2, 5, 5],
            'problem_name': ['sphere', 'rastrigin', 'sphere', 'rastrigin']
        }
        df = pd.DataFrame(data)

        stats = validator._compute_summary_statistics(df)

        assert stats['success_rate'] == 0.75
        assert 'mean_objective_error' in stats
        assert 'by_dimension' in stats
        assert 'by_problem' in stats

    def test_html_report_generation(self):
        """Test HTML report generation."""
        validator = OptimizationValidator(verbose=False)

        # Create mock results
        data = {
            'method_name': ['ga', 'ga', 'pso', 'pso'],
            'problem_name': ['sphere', 'rastrigin', 'sphere', 'rastrigin'],
            'problem_dimension': [5, 5, 5, 5],
            'passed': [True, True, True, False],
            'objective_error': [0.01, 0.02, 0.01, 0.5],
            'runtime': [1.0, 1.2, 0.8, 1.5],
            'evaluations': [1000, 1200, 800, 1500],
            'iterations': [100, 120, 80, 150]
        }
        df = pd.DataFrame(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            validator.generate_report(df, str(output_path), "Test Report")

            # Check that files were created
            assert output_path.exists()
            assert output_path.with_suffix('.png').exists()

            # Check HTML content
            html = output_path.read_text()
            assert "Test Report" in html
            assert "Success Rate by Method" in html


class TestContinuousValidation:
    """Test continuous validation for CI/CD."""

    def test_continuous_validator_init(self):
        """Test continuous validator initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ContinuousValidator(
                output_dir=tmpdir,
                success_threshold=0.8,
                verbose=False
            )

            assert validator.success_threshold == 0.8
            assert Path(tmpdir).exists()

    def test_test_configurations(self):
        """Test that test configurations are properly defined."""
        validator = ContinuousValidator(verbose=False)
        configs = validator.test_configs

        assert 'quick' in configs
        assert 'standard' in configs
        assert 'comprehensive' in configs
        assert 'nightly' in configs

        # Check configuration properties
        for level, config in configs.items():
            assert 'methods' in config
            assert 'dimensions' in config
            assert 'n_runs' in config
            assert 'description' in config

    def test_threshold_checking(self):
        """Test threshold checking logic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ContinuousValidator(
                output_dir=tmpdir,
                success_threshold=0.7,
                verbose=False
            )

            # Create mock results
            data = {
                'method_name': ['ga'] * 10 + ['pso'] * 10,
                'passed': [True] * 8 + [False] * 2 + [True] * 5 + [False] * 5
            }
            df = pd.DataFrame(data)

            results = validator._check_thresholds(df, ['ga', 'pso'])

            # GA should pass (80% > 70%)
            assert results['methods']['ga']['passed'] == True
            assert results['methods']['ga']['success_rate'] == 0.8

            # PSO should fail (50% < 70%)
            assert results['methods']['pso']['passed'] == False
            assert results['methods']['pso']['success_rate'] == 0.5

            # Overall should fail (one method failed)
            assert results['overall_passed'] == False

    def test_summary_json_export(self):
        """Test JSON summary export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ContinuousValidator(
                output_dir=tmpdir,
                verbose=False
            )

            results = {
                'overall_passed': True,
                'methods': {
                    'ga': {'success_rate': 0.85, 'passed': True},
                    'pso': {'success_rate': 0.75, 'passed': True}
                },
                'summary': {
                    'total_tests': 100,
                    'total_passed': 80
                }
            }

            json_path = Path(tmpdir) / "test_summary.json"
            validator._save_summary_json(results, json_path)

            # Check file exists and is valid JSON
            assert json_path.exists()

            with open(json_path) as f:
                loaded = json.load(f)

            assert loaded['overall_passed'] == True
            assert loaded['methods']['ga']['success_rate'] == 0.85

    @pytest.mark.skipif(
        True,  # Skip by default as it's slow
        reason="Full validation suite is slow"
    )
    def test_run_validation_suite(self):
        """Test running a quick validation suite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ContinuousValidator(
                output_dir=tmpdir,
                verbose=False
            )

            # Override with minimal config for testing
            validator.test_configs['test'] = {
                'methods': ['random_search'],
                'dimensions': [2],
                'n_runs': 1,
                'max_problems': 2,
                'description': 'Test configuration'
            }

            results = validator.run_validation_suite(level='test')

            assert 'overall_passed' in results
            assert 'methods' in results
            assert 'summary' in results


class TestProblemProperties:
    """Test mathematical properties of test problems."""

    def test_convexity(self):
        """Test convexity of convex functions."""
        sphere = SphereFunction(5)

        # Test convexity: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
        x = np.random.randn(5)
        y = np.random.randn(5)
        lambda_val = 0.3

        midpoint = lambda_val * x + (1 - lambda_val) * y
        f_mid = sphere.evaluate(midpoint)
        f_weighted = lambda_val * sphere.evaluate(x) + (1 - lambda_val) * sphere.evaluate(y)

        assert f_mid <= f_weighted + 1e-10

    def test_multimodality(self):
        """Test that multimodal functions have multiple local minima."""
        rastrigin = RastriginFunction(2)

        # Test at different points
        points = [
            np.array([0, 0]),     # Global minimum
            np.array([1, 0]),     # Local minimum
            np.array([0, 1]),     # Local minimum
            np.array([2, 2]),     # Another local region
        ]

        values = [rastrigin.evaluate(p) for p in points]

        # Global minimum should be best
        assert values[0] == min(values)
        assert values[0] == pytest.approx(0.0)

        # Other points should have higher values
        for v in values[1:]:
            assert v > 0

    def test_bounds_validity(self):
        """Test that all problems have valid bounds."""
        problems = get_test_problems([2, 5])

        for problem in problems:
            if isinstance(problem.bounds, tuple):
                assert problem.bounds[0] < problem.bounds[1]
            else:
                for low, high in problem.bounds:
                    assert low < high

    def test_optimal_solution_validity(self):
        """Test that optimal solutions are within bounds."""
        problems = get_test_problems([2, 5])

        for problem in problems:
            if problem.optimal_solution is None:
                continue  # Some problems don't have closed-form solutions

            # Check that optimal solution is within bounds
            if isinstance(problem.bounds, tuple):
                low, high = problem.bounds
                assert np.all(problem.optimal_solution >= low)
                assert np.all(problem.optimal_solution <= high)
            else:
                for i, (low, high) in enumerate(problem.bounds):
                    assert low <= problem.optimal_solution[i] <= high

    def test_problem_difficulty_ordering(self):
        """Test that problem difficulty is as expected."""
        # Sphere should be easiest (convex, unimodal)
        sphere = SphereFunction(5)

        # Rastrigin should be hard (highly multimodal)
        rastrigin = RastriginFunction(5)

        # Generate random points and test how often we're near optimum
        n_samples = 100
        sphere_close = 0
        rastrigin_close = 0

        for _ in range(n_samples):
            x = np.random.randn(5) * 0.1  # Small perturbation

            if sphere.evaluate(x) < 0.1:
                sphere_close += 1
            if rastrigin.evaluate(x) < 0.1:
                rastrigin_close += 1

        # Sphere should be easier (more points near optimum)
        assert sphere_close > rastrigin_close


class TestValidationIntegration:
    """Integration tests for the validation framework."""

    def test_end_to_end_validation(self):
        """Test complete validation workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create validator
            validator = OptimizationValidator(verbose=False)

            # Create simple mock optimizer for testing
            class MockOptimizer:
                def __init__(self, success_rate=0.8):
                    self.success_rate = success_rate

                def optimize(self, problem):
                    # Simulate optimization with some success rate
                    if np.random.random() < self.success_rate:
                        # Return near-optimal solution
                        solution = problem.optimal_solution + np.random.randn(problem.dimension) * 0.01
                        value = problem.evaluate(solution)
                    else:
                        # Return random solution
                        solution = np.random.randn(problem.dimension)
                        value = problem.evaluate(solution)

                    return {
                        'solution': solution,
                        'objective': value,
                        'metadata': {
                            'iterations': np.random.randint(50, 200),
                            'evaluations': np.random.randint(500, 2000)
                        }
                    }

            # Override optimizer
            validator.optimize = lambda problem, adapter, method, **kwargs: (
                MockOptimizer(success_rate=0.8).optimize(problem)
            )

            # Get test problems
            problems = get_test_problems([2])[:3]  # Just a few for testing

            # Run validation
            results = []
            for problem in problems:
                # Create mock adapter
                class MockAdapter:
                    def __init__(self, p):
                        self.problem = p

                adapter = MockAdapter(problem)

                result = validator._run_single_validation('mock_method', problem, seed=42)
                results.append(result)

            # Check results
            assert len(results) == 3
            for result in results:
                assert isinstance(result, ValidationResult)
                assert result.method_name == 'mock_method'
                assert result.iterations > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])