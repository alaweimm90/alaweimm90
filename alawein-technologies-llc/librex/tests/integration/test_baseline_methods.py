"""
Integration tests for baseline optimization methods

Tests verify that all 5 baseline methods work correctly with domain adapters.
"""

import numpy as np
import pytest

from Librex import optimize
from Librex.adapters.qap import QAPAdapter
from Librex.adapters.tsp import TSPAdapter


class TestBaselineMethodsWithQAP:
    """Integration tests for baseline methods with QAP adapter"""

    @pytest.fixture
    def small_qap_problem(self):
        """Small QAP problem for testing"""
        return {
            'flow_matrix': np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]),
            'distance_matrix': np.array([[0, 5, 10], [5, 0, 15], [10, 15, 10]])
        }

    def test_random_search_qap(self, small_qap_problem):
        """Test random search with QAP"""
        adapter = QAPAdapter()
        result = optimize(
            small_qap_problem,
            adapter,
            method='random_search',
            config={'iterations': 100, 'seed': 42}
        )

        assert 'solution' in result
        assert 'objective' in result
        assert result['is_valid']
        assert len(result['solution']) == 3
        assert result['iterations'] == 100

    def test_simulated_annealing_qap(self, small_qap_problem):
        """Test simulated annealing with QAP"""
        adapter = QAPAdapter()
        result = optimize(
            small_qap_problem,
            adapter,
            method='simulated_annealing',
            config={'iterations': 200, 'seed': 42}
        )

        assert 'solution' in result
        assert 'objective' in result
        assert result['is_valid']
        assert len(result['solution']) == 3

    def test_local_search_qap(self, small_qap_problem):
        """Test local search with QAP"""
        adapter = QAPAdapter()
        result = optimize(
            small_qap_problem,
            adapter,
            method='local_search',
            config={'max_iterations': 100, 'restarts': 3, 'seed': 42}
        )

        assert 'solution' in result
        assert 'objective' in result
        assert result['is_valid']
        assert result['convergence']['converged']

    def test_genetic_algorithm_qap(self, small_qap_problem):
        """Test genetic algorithm with QAP"""
        adapter = QAPAdapter()
        result = optimize(
            small_qap_problem,
            adapter,
            method='genetic_algorithm',
            config={
                'population_size': 20,
                'generations': 10,
                'seed': 42
            }
        )

        assert 'solution' in result
        assert 'objective' in result
        assert result['is_valid']
        assert len(result['solution']) == 3

    def test_tabu_search_qap(self, small_qap_problem):
        """Test tabu search with QAP"""
        adapter = QAPAdapter()
        result = optimize(
            small_qap_problem,
            adapter,
            method='tabu_search',
            config={'iterations': 100, 'tabu_tenure': 10, 'seed': 42}
        )

        assert 'solution' in result
        assert 'objective' in result
        assert result['is_valid']
        assert len(result['solution']) == 3


class TestBaselineMethodsWithTSP:
    """Integration tests for baseline methods with TSP adapter"""

    @pytest.fixture
    def small_tsp_problem(self):
        """Small TSP problem for testing"""
        return {
            'coordinates': np.array([
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1]
            ])
        }

    def test_random_search_tsp(self, small_tsp_problem):
        """Test random search with TSP"""
        adapter = TSPAdapter()
        result = optimize(
            small_tsp_problem,
            adapter,
            method='random_search',
            config={'iterations': 100, 'seed': 42}
        )

        assert 'solution' in result
        assert 'objective' in result
        assert result['is_valid']
        assert len(result['solution']) == 4

    def test_simulated_annealing_tsp(self, small_tsp_problem):
        """Test simulated annealing with TSP"""
        adapter = TSPAdapter()
        result = optimize(
            small_tsp_problem,
            adapter,
            method='simulated_annealing',
            config={'iterations': 200, 'seed': 42}
        )

        assert 'solution' in result
        assert 'objective' in result
        assert result['is_valid']
        # Tour length should be reasonable for a 2x2 square
        assert result['objective'] < 10  # Max perimeter is 8 + slack

    def test_genetic_algorithm_tsp(self, small_tsp_problem):
        """Test genetic algorithm with TSP"""
        adapter = TSPAdapter()
        result = optimize(
            small_tsp_problem,
            adapter,
            method='genetic_algorithm',
            config={
                'population_size': 20,
                'generations': 15,
                'seed': 42
            }
        )

        assert 'solution' in result
        assert 'objective' in result
        assert result['is_valid']


class TestMethodComparisons:
    """Compare different methods on same problem"""

    def test_all_methods_find_valid_solutions(self):
        """Test that all methods find valid solutions"""
        problem = {
            'flow_matrix': np.array([[0, 1], [1, 0]]),
            'distance_matrix': np.array([[0, 2], [2, 0]])
        }

        adapter = QAPAdapter()
        methods = [
            'random_search',
            'simulated_annealing',
            'local_search',
            'genetic_algorithm',
            'tabu_search'
        ]

        for method in methods:
            result = optimize(
                problem,
                adapter,
                method=method,
                config={'seed': 42}
            )
            assert result['is_valid'], f"{method} produced invalid solution"
            assert len(result['solution']) == 2

    def test_method_reproducibility_with_seed(self):
        """Test that methods are reproducible with same seed"""
        problem = {
            'flow_matrix': np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]),
            'distance_matrix': np.array([[0, 5, 10], [5, 0, 15], [10, 15, 0]])
        }

        adapter = QAPAdapter()

        # Run twice with same seed
        result1 = optimize(
            problem,
            adapter,
            method='simulated_annealing',
            config={'iterations': 100, 'seed': 42}
        )

        result2 = optimize(
            problem,
            adapter,
            method='simulated_annealing',
            config={'iterations': 100, 'seed': 42}
        )

        # Should get same result with same seed
        assert np.array_equal(result1['solution'], result2['solution'])
        assert result1['objective'] == result2['objective']


class TestErrorHandling:
    """Test error handling in optimization methods"""

    def test_invalid_method_raises_error(self):
        """Test that invalid method name raises NotImplementedError"""
        problem = {
            'flow_matrix': np.array([[0, 1], [1, 0]]),
            'distance_matrix': np.array([[0, 2], [2, 0]])
        }

        adapter = QAPAdapter()

        with pytest.raises(NotImplementedError, match="not implemented"):
            optimize(
                problem,
                adapter,
                method='invalid_method_name'
            )

    def test_missing_adapter_raises_error(self):
        """Test that missing adapter raises ValueError"""
        problem = {
            'flow_matrix': np.array([[0, 1], [1, 0]]),
            'distance_matrix': np.array([[0, 2], [2, 0]])
        }

        with pytest.raises(ValueError, match="adapter must be provided"):
            optimize(problem, adapter=None, method='random_search')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
