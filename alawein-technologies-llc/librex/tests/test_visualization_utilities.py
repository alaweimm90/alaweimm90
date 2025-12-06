"""
Comprehensive tests for Visualization Tools and Utilities.
Tests plotting functions, matrix operations, and validation utilities.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
import warnings

from Librex.visualization import (
    plot_convergence,
    plot_solution_quality,
    plot_performance_comparison,
    plot_scalability,
    plot_heatmap,
    create_dashboard
)
from Librex.utils import (
    validate_problem,
    validate_solution,
    compute_cost,
    generate_random_problem,
    symmetrize_matrix,
    normalize_matrix
)
from Librex.utils.matrix_ops import (
    permute_matrix,
    compute_eigenvalues,
    matrix_distance,
    is_positive_semidefinite,
    condition_number,
    spectral_radius
)
from Librex.utils.validation import (
    check_problem_format,
    check_solution_format,
    validate_optimizer_params,
    validate_bounds,
    sanitize_input
)


class TestVisualizationTools:
    """Test visualization and plotting functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for plotting."""
        return {
            'convergence': {
                'iterations': list(range(100)),
                'costs': [1000 * np.exp(-0.05 * i) + 100 + np.random.randn() * 10 for i in range(100)]
            },
            'methods': {
                'RandomSearch': {'costs': [150, 145, 142, 140, 138]},
                'SimulatedAnnealing': {'costs': [150, 130, 115, 105, 100]},
                'GeneticAlgorithm': {'costs': [150, 135, 120, 110, 102]}
            },
            'scalability': [
                {'size': 10, 'time': 0.1},
                {'size': 20, 'time': 0.4},
                {'size': 30, 'time': 0.9},
                {'size': 40, 'time': 1.6},
                {'size': 50, 'time': 2.5}
            ]
        }

    @patch('matplotlib.pyplot.show')
    def test_plot_convergence(self, mock_show, sample_data):
        """Test convergence plotting."""
        fig = plot_convergence(
            sample_data['convergence']['iterations'],
            sample_data['convergence']['costs'],
            title="Test Convergence",
            xlabel="Iteration",
            ylabel="Cost"
        )

        assert fig is not None
        assert len(fig.axes) > 0
        assert fig.axes[0].get_title() == "Test Convergence"

        # Test with multiple runs
        multi_runs = [
            sample_data['convergence']['costs'],
            [c * 1.1 for c in sample_data['convergence']['costs']],
            [c * 0.9 for c in sample_data['convergence']['costs']]
        ]

        fig_multi = plot_convergence(
            sample_data['convergence']['iterations'],
            multi_runs,
            show_confidence=True
        )

        assert fig_multi is not None

    @patch('matplotlib.pyplot.show')
    def test_plot_solution_quality(self, mock_show, sample_data):
        """Test solution quality plotting."""
        methods = list(sample_data['methods'].keys())
        qualities = [np.mean(data['costs']) for data in sample_data['methods'].values()]
        stds = [np.std(data['costs']) for data in sample_data['methods'].values()]

        fig = plot_solution_quality(
            methods,
            qualities,
            errors=stds,
            title="Solution Quality Comparison"
        )

        assert fig is not None
        assert len(fig.axes[0].patches) == len(methods)  # Bar plot

    @patch('matplotlib.pyplot.show')
    def test_plot_performance_comparison(self, mock_show, sample_data):
        """Test performance comparison plotting."""
        # Create performance matrix
        problems = ['chr12a', 'nug15', 'esc16a']
        methods = list(sample_data['methods'].keys())
        performance = np.random.rand(len(problems), len(methods)) * 100 + 100

        fig = plot_performance_comparison(
            performance,
            problems=problems,
            methods=methods,
            metric='cost'
        )

        assert fig is not None

        # Test with multiple metrics
        metrics = {
            'cost': performance,
            'time': np.random.rand(len(problems), len(methods)) * 10,
            'quality': np.random.rand(len(problems), len(methods))
        }

        fig_multi = plot_performance_comparison(
            metrics,
            problems=problems,
            methods=methods,
            subplot=True
        )

        assert fig_multi is not None
        assert len(fig_multi.axes) == len(metrics)

    @patch('matplotlib.pyplot.show')
    def test_plot_scalability(self, mock_show, sample_data):
        """Test scalability plotting."""
        sizes = [d['size'] for d in sample_data['scalability']]
        times = [d['time'] for d in sample_data['scalability']]

        fig = plot_scalability(
            sizes,
            times,
            fit_curve=True,
            complexity_models=['linear', 'quadratic', 'cubic']
        )

        assert fig is not None
        ax = fig.axes[0]
        assert len(ax.lines) > 1  # Data + fitted curves

    @patch('matplotlib.pyplot.show')
    def test_plot_heatmap(self, mock_show):
        """Test heatmap plotting."""
        # Create sample matrix
        matrix = np.random.rand(10, 10)

        fig = plot_heatmap(
            matrix,
            title="Test Heatmap",
            cmap='viridis',
            annotate=True
        )

        assert fig is not None
        assert len(fig.axes) > 0

        # Test with custom labels
        fig_labeled = plot_heatmap(
            matrix,
            row_labels=[f"R{i}" for i in range(10)],
            col_labels=[f"C{i}" for i in range(10)]
        )

        assert fig_labeled is not None

    @patch('matplotlib.pyplot.show')
    def test_create_dashboard(self, mock_show, sample_data):
        """Test dashboard creation."""
        dashboard_data = {
            'convergence': sample_data['convergence'],
            'comparison': sample_data['methods'],
            'scalability': sample_data['scalability'],
            'statistics': {
                'best_method': 'SimulatedAnnealing',
                'average_improvement': 0.35,
                'total_time': 125.3
            }
        }

        fig = create_dashboard(dashboard_data, title="Optimization Dashboard")

        assert fig is not None
        assert len(fig.axes) >= 3  # Multiple subplots
        assert fig._suptitle.get_text() == "Optimization Dashboard"

    def test_save_plots(self, sample_data):
        """Test saving plots to file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_file = f.name

        try:
            fig = plot_convergence(
                sample_data['convergence']['iterations'],
                sample_data['convergence']['costs']
            )
            fig.savefig(temp_file)

            assert Path(temp_file).exists()
            assert Path(temp_file).stat().st_size > 0
        finally:
            plt.close('all')
            Path(temp_file).unlink()

    @patch('matplotlib.pyplot.show')
    def test_interactive_plots(self, mock_show):
        """Test interactive plotting features."""
        # Test with plotly backend if available
        try:
            import plotly.graph_objects as go

            data = np.random.rand(100)
            fig = plot_convergence(
                range(100),
                data,
                backend='plotly',
                interactive=True
            )

            assert fig is not None
        except ImportError:
            # Plotly not installed, skip
            pass


class TestMatrixOperations:
    """Test matrix operation utilities."""

    @pytest.fixture
    def sample_matrices(self):
        """Create sample matrices for testing."""
        return {
            'symmetric': np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]]),
            'asymmetric': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            'positive_definite': np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]]),
            'permutation': np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        }

    def test_permute_matrix(self, sample_matrices):
        """Test matrix permutation."""
        matrix = sample_matrices['symmetric']
        perm = [2, 0, 1]  # Permutation

        permuted = permute_matrix(matrix, perm)

        assert permuted.shape == matrix.shape
        assert permuted[0, 0] == matrix[2, 2]
        assert permuted[1, 1] == matrix[0, 0]

    def test_compute_eigenvalues(self, sample_matrices):
        """Test eigenvalue computation."""
        matrix = sample_matrices['symmetric']
        eigenvalues = compute_eigenvalues(matrix)

        assert len(eigenvalues) == matrix.shape[0]
        assert all(np.isreal(eigenvalues))  # Symmetric matrix has real eigenvalues

        # Test sorting
        sorted_eigs = compute_eigenvalues(matrix, sort=True)
        assert sorted_eigs[0] >= sorted_eigs[-1]

    def test_matrix_distance(self, sample_matrices):
        """Test matrix distance metrics."""
        matrix1 = sample_matrices['symmetric']
        matrix2 = matrix1 + np.random.randn(*matrix1.shape) * 0.1

        # Frobenius norm
        dist_fro = matrix_distance(matrix1, matrix2, metric='frobenius')
        assert dist_fro > 0

        # Spectral norm
        dist_spec = matrix_distance(matrix1, matrix2, metric='spectral')
        assert dist_spec > 0

        # Nuclear norm
        dist_nuc = matrix_distance(matrix1, matrix2, metric='nuclear')
        assert dist_nuc > 0

    def test_is_positive_semidefinite(self, sample_matrices):
        """Test positive semidefinite check."""
        assert is_positive_semidefinite(sample_matrices['positive_definite']) is True
        assert is_positive_semidefinite(-sample_matrices['positive_definite']) is False

        # Identity is positive semidefinite
        assert is_positive_semidefinite(np.eye(5)) is True

    def test_condition_number(self, sample_matrices):
        """Test condition number computation."""
        cond = condition_number(sample_matrices['positive_definite'])
        assert cond >= 1  # Condition number is always >= 1

        # Ill-conditioned matrix
        ill_conditioned = np.array([[1, 1], [1, 1.0001]])
        cond_ill = condition_number(ill_conditioned)
        assert cond_ill > 1000  # Very ill-conditioned

    def test_spectral_radius(self, sample_matrices):
        """Test spectral radius computation."""
        radius = spectral_radius(sample_matrices['symmetric'])
        eigenvalues = compute_eigenvalues(sample_matrices['symmetric'])

        assert radius == pytest.approx(np.max(np.abs(eigenvalues)))

    def test_symmetrize_matrix(self):
        """Test matrix symmetrization."""
        asymmetric = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Average symmetrization
        sym_avg = symmetrize_matrix(asymmetric, method='average')
        assert np.allclose(sym_avg, sym_avg.T)

        # Upper triangular symmetrization
        sym_upper = symmetrize_matrix(asymmetric, method='upper')
        assert np.allclose(sym_upper, sym_upper.T)
        assert sym_upper[1, 0] == asymmetric[0, 1]

    def test_normalize_matrix(self):
        """Test matrix normalization."""
        matrix = np.array([[1, 2], [3, 4]])

        # Max normalization
        norm_max = normalize_matrix(matrix, method='max')
        assert np.max(np.abs(norm_max)) == 1.0

        # Frobenius normalization
        norm_fro = normalize_matrix(matrix, method='frobenius')
        assert np.linalg.norm(norm_fro, 'fro') == pytest.approx(1.0)

        # Row normalization
        norm_row = normalize_matrix(matrix, method='row')
        assert all(np.sum(np.abs(row)) == pytest.approx(1.0) for row in norm_row)


class TestValidationUtilities:
    """Test validation utilities."""

    def test_check_problem_format(self):
        """Test problem format validation."""
        # Valid QAP problem
        valid_problem = {
            'flow': np.random.rand(5, 5),
            'distance': np.random.rand(5, 5),
            'size': 5
        }
        assert check_problem_format(valid_problem, problem_type='QAP') is True

        # Invalid problem (missing fields)
        invalid_problem = {'flow': np.random.rand(5, 5)}
        assert check_problem_format(invalid_problem, problem_type='QAP') is False

        # Invalid problem (mismatched sizes)
        mismatched = {
            'flow': np.random.rand(5, 5),
            'distance': np.random.rand(4, 4)
        }
        assert check_problem_format(mismatched, problem_type='QAP') is False

    def test_check_solution_format(self):
        """Test solution format validation."""
        # Valid permutation solution
        valid_solution = np.array([2, 0, 3, 1, 4])
        assert check_solution_format(valid_solution, problem_size=5) is True

        # Invalid solution (wrong size)
        wrong_size = np.array([0, 1, 2])
        assert check_solution_format(wrong_size, problem_size=5) is False

        # Invalid solution (not a permutation)
        not_permutation = np.array([0, 0, 1, 2, 3])
        assert check_solution_format(not_permutation, problem_size=5) is False

        # Valid binary solution
        binary_solution = np.array([0, 1, 1, 0, 1])
        assert check_solution_format(binary_solution, solution_type='binary') is True

    def test_validate_optimizer_params(self):
        """Test optimizer parameter validation."""
        # Valid parameters
        valid_params = {
            'max_iterations': 1000,
            'population_size': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8
        }

        schema = {
            'max_iterations': {'type': int, 'min': 1},
            'population_size': {'type': int, 'min': 2},
            'mutation_rate': {'type': float, 'min': 0, 'max': 1},
            'crossover_rate': {'type': float, 'min': 0, 'max': 1}
        }

        assert validate_optimizer_params(valid_params, schema) is True

        # Invalid type
        invalid_type = valid_params.copy()
        invalid_type['max_iterations'] = '1000'
        assert validate_optimizer_params(invalid_type, schema) is False

        # Out of range
        out_of_range = valid_params.copy()
        out_of_range['mutation_rate'] = 1.5
        assert validate_optimizer_params(out_of_range, schema) is False

    def test_validate_bounds(self):
        """Test bounds validation."""
        # Valid bounds
        valid_bounds = [(0, 10), (-5, 5), (1, 100)]
        assert validate_bounds(valid_bounds) is True

        # Invalid bounds (lower > upper)
        invalid_bounds = [(10, 0), (5, 5), (1, 100)]
        assert validate_bounds(invalid_bounds) is False

        # Check solution against bounds
        solution = [5, 0, 50]
        assert validate_bounds(valid_bounds, solution=solution) is True

        # Solution out of bounds
        out_solution = [11, 0, 50]
        assert validate_bounds(valid_bounds, solution=out_solution) is False

    def test_sanitize_input(self):
        """Test input sanitization."""
        # Sanitize numeric input
        assert sanitize_input("123", expected_type=int) == 123
        assert sanitize_input("12.5", expected_type=float) == 12.5

        # Sanitize array input
        array_input = [1, 2, "3", 4.5]
        sanitized = sanitize_input(array_input, expected_type='array')
        assert isinstance(sanitized, np.ndarray)
        assert sanitized.dtype == np.float64

        # Sanitize with bounds
        bounded = sanitize_input(150, expected_type=int, min_val=0, max_val=100)
        assert bounded == 100  # Clamped to max

        # Handle invalid input
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sanitize_input("invalid", expected_type=int, default=0)
            assert result == 0


class TestUtilityFunctions:
    """Test general utility functions."""

    def test_validate_problem(self):
        """Test comprehensive problem validation."""
        # Valid problem
        problem = {
            'flow': np.random.rand(10, 10),
            'distance': np.random.rand(10, 10),
            'size': 10,
            'optimal': 1234
        }

        is_valid, message = validate_problem(problem)
        assert is_valid is True
        assert message == "Valid problem"

        # Ensure matrices are non-negative
        problem_negative = problem.copy()
        problem_negative['flow'][0, 0] = -1

        is_valid, message = validate_problem(problem_negative)
        assert is_valid is False
        assert "negative" in message.lower()

    def test_validate_solution(self):
        """Test comprehensive solution validation."""
        problem = {'size': 10}

        # Valid solution
        solution = np.random.permutation(10)
        is_valid, message = validate_solution(solution, problem)
        assert is_valid is True

        # Invalid solution
        invalid_solution = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        is_valid, message = validate_solution(invalid_solution, problem)
        assert is_valid is False

    def test_compute_cost(self):
        """Test cost computation."""
        flow = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
        distance = np.array([[0, 2, 3], [2, 0, 1], [3, 1, 0]])
        solution = [0, 1, 2]

        cost = compute_cost(solution, flow, distance)

        # Manual calculation
        expected = 0
        for i in range(3):
            for j in range(3):
                expected += flow[i, j] * distance[solution[i], solution[j]]

        assert cost == expected

    def test_generate_random_problem(self):
        """Test random problem generation."""
        # Generate QAP problem
        problem = generate_random_problem(
            problem_type='QAP',
            size=15,
            density=0.7,
            seed=42
        )

        assert problem['flow'].shape == (15, 15)
        assert problem['distance'].shape == (15, 15)
        assert problem['size'] == 15

        # Check density
        non_zero = np.sum(problem['flow'] != 0)
        total = 15 * 15
        actual_density = non_zero / total
        assert 0.6 <= actual_density <= 0.8

        # Generate TSP problem
        tsp_problem = generate_random_problem(
            problem_type='TSP',
            size=20,
            bounds=(0, 100)
        )

        assert tsp_problem['distance'].shape == (20, 20)
        assert np.all(tsp_problem['distance'] >= 0)
        assert np.all(tsp_problem['distance'] <= 100)

        # Check symmetry
        assert np.allclose(tsp_problem['distance'], tsp_problem['distance'].T)


class TestAdvancedUtilities:
    """Test advanced utility functions."""

    def test_problem_reduction(self):
        """Test problem size reduction techniques."""
        from Librex.utils import reduce_problem_size

        large_problem = {
            'flow': np.random.rand(100, 100),
            'distance': np.random.rand(100, 100),
            'size': 100
        }

        # Reduce to smaller size
        reduced = reduce_problem_size(large_problem, target_size=20, method='clustering')

        assert reduced['size'] == 20
        assert reduced['flow'].shape == (20, 20)
        assert reduced['distance'].shape == (20, 20)

    def test_solution_repair(self):
        """Test solution repair mechanisms."""
        from Librex.utils import repair_solution

        # Invalid permutation (duplicates)
        invalid_perm = np.array([0, 1, 2, 2, 4])
        repaired = repair_solution(invalid_perm, solution_type='permutation')

        assert len(repaired) == 5
        assert len(set(repaired)) == 5
        assert set(repaired) == set(range(5))

    def test_performance_profiling(self):
        """Test performance profiling utilities."""
        from Librex.utils import profile_function

        @profile_function
        def test_func(n):
            return sum(range(n))

        result = test_func(1000000)

        # Check that profiling info is available
        assert hasattr(test_func, 'profile_stats')
        assert 'time' in test_func.profile_stats
        assert 'memory' in test_func.profile_stats

    def test_parallel_utilities(self):
        """Test parallel processing utilities."""
        from Librex.utils import parallel_map

        def square(x):
            return x ** 2

        inputs = list(range(100))
        results = parallel_map(square, inputs, n_jobs=2)

        assert len(results) == len(inputs)
        assert results[10] == 100  # 10^2

    def test_caching_utilities(self):
        """Test caching utilities."""
        from Librex.utils import cached_function

        call_count = 0

        @cached_function(maxsize=128)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x ** 2

        # First call
        result1 = expensive_func(10)
        assert call_count == 1

        # Second call (cached)
        result2 = expensive_func(10)
        assert call_count == 1  # Not incremented
        assert result1 == result2

        # Different input
        result3 = expensive_func(20)
        assert call_count == 2