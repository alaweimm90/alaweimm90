"""
Tests for AI-powered method selection
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from Librex.ai.features import ProblemFeatureExtractor, ProblemFeatures
from Librex.ai.method_selector import MethodRecommendation, MethodSelector, PerformanceRecord
from Librex.ai.models import DataCollector, MethodSelectorModel
from Librex.core.interfaces import StandardizedProblem


class TestProblemFeatureExtractor:
    """Tests for problem feature extraction"""

    def test_extract_features_basic(self):
        """Test basic feature extraction"""
        extractor = ProblemFeatureExtractor()

        # Create a simple problem
        problem = StandardizedProblem(
            dimension=10,
            objective_matrix=np.random.rand(10, 10),
            objective_function=lambda x: np.sum(x),
            constraint_matrix=None,
            problem_metadata={'type': 'test'}
        )

        features = extractor.extract_features(
            problem=None,
            standardized_problem=problem
        )

        assert features.dimension == 10
        assert features.total_elements == 100
        assert features.has_matrix is True
        assert 0 <= features.matrix_sparsity <= 1
        assert 0 <= features.matrix_symmetry <= 1

    def test_extract_features_sparse_matrix(self):
        """Test feature extraction for sparse matrix"""
        extractor = ProblemFeatureExtractor()

        # Create sparse matrix
        matrix = np.zeros((20, 20))
        matrix[0, 1] = 1
        matrix[1, 0] = 1
        matrix[5, 10] = 2

        problem = StandardizedProblem(
            dimension=20,
            objective_matrix=matrix,
            objective_function=None,
            constraint_matrix=None
        )

        features = extractor.extract_features(
            problem=None,
            standardized_problem=problem
        )

        assert features.matrix_sparsity > 0.9  # Very sparse
        assert features.connectivity < 0.1  # Low connectivity

    def test_extract_features_symmetric_matrix(self):
        """Test feature extraction for symmetric matrix"""
        extractor = ProblemFeatureExtractor()

        # Create symmetric matrix
        matrix = np.random.rand(15, 15)
        matrix = (matrix + matrix.T) / 2  # Make symmetric

        problem = StandardizedProblem(
            dimension=15,
            objective_matrix=matrix,
            objective_function=None,
            constraint_matrix=None
        )

        features = extractor.extract_features(
            problem=None,
            standardized_problem=problem
        )

        assert features.matrix_symmetry > 0.99  # Should be perfectly symmetric

    def test_extract_features_with_constraints(self):
        """Test feature extraction with constraints"""
        extractor = ProblemFeatureExtractor()

        problem = StandardizedProblem(
            dimension=10,
            objective_matrix=np.random.rand(10, 10),
            objective_function=None,
            constraint_matrix=np.random.rand(5, 10)
        )

        features = extractor.extract_features(
            problem=None,
            standardized_problem=problem
        )

        assert features.has_constraints is True
        assert features.constraint_count == 5
        assert 0 <= features.constraint_density <= 1

    def test_extract_features_qap_problem(self):
        """Test feature extraction for QAP problem"""
        extractor = ProblemFeatureExtractor()

        # Mock QAP adapter
        adapter = Mock()
        adapter.__class__.__name__ = 'QAPAdapter'

        problem = {
            'flow_matrix': np.array([[0, 5], [5, 0]]),
            'distance_matrix': np.array([[0, 8], [8, 0]])
        }

        standardized = StandardizedProblem(
            dimension=2,
            objective_matrix=np.array([[0, 40], [40, 0]]),
            objective_function=None
        )

        adapter.encode_problem.return_value = standardized

        features = extractor.extract_features(problem, adapter)

        assert features.problem_type == 'QAP'
        assert features.is_permutation is True

    def test_feature_vector_conversion(self):
        """Test conversion to numerical feature vector"""
        extractor = ProblemFeatureExtractor()

        features = ProblemFeatures(
            dimension=10,
            total_elements=100,
            has_matrix=True,
            matrix_sparsity=0.5,
            matrix_symmetry=0.8,
            diagonal_dominance=0.3,
            value_range=10.0,
            value_mean=5.0,
            value_std=2.0,
            value_skewness=0.1,
            value_kurtosis=-0.2,
            has_constraints=False,
            constraint_density=0.0,
            constraint_count=0,
            is_permutation=True,
            is_binary=False,
            is_continuous=False,
            connectivity=0.4,
            clustering_coefficient=0.2,
            problem_type='QAP',
            estimated_difficulty='medium'
        )

        vector = extractor.get_feature_vector(features)

        assert isinstance(vector, np.ndarray)
        assert len(vector) == 20
        assert vector[0] == 10  # dimension
        assert vector[2] == 1.0  # has_matrix (True)


class TestMethodSelector:
    """Tests for method selection"""

    def test_recommend_method_basic(self):
        """Test basic method recommendation"""
        selector = MethodSelector()

        problem = StandardizedProblem(
            dimension=10,
            objective_matrix=np.random.rand(10, 10),
            objective_function=lambda x: np.sum(x)
        )

        method, config, confidence = selector.recommend_method(
            problem=None,
            standardized_problem=problem
        )

        assert method in ['random_search', 'simulated_annealing', 'local_search',
                          'genetic_algorithm', 'tabu_search']
        assert isinstance(config, dict)
        assert 0 <= confidence <= 1

    def test_recommend_method_small_problem(self):
        """Test recommendation for small problem"""
        selector = MethodSelector()

        problem = StandardizedProblem(
            dimension=5,
            objective_matrix=np.random.rand(5, 5),
            objective_function=lambda x: np.sum(x)
        )

        method, config, confidence = selector.recommend_method(
            problem=None,
            standardized_problem=problem,
            quality_requirement='balanced'
        )

        # Small problems often benefit from local search
        # This is a heuristic test - exact method may vary
        assert method in ['local_search', 'simulated_annealing']

    def test_recommend_method_large_problem(self):
        """Test recommendation for large problem"""
        selector = MethodSelector()

        problem = StandardizedProblem(
            dimension=500,
            objective_matrix=None,  # Too large for matrix
            objective_function=lambda x: np.sum(x)
        )

        method, config, confidence = selector.recommend_method(
            problem=None,
            standardized_problem=problem,
            quality_requirement='fast'
        )

        # Large problems with speed requirement often use random search
        assert method in ['random_search', 'simulated_annealing']
        assert config['iterations'] <= 10000  # Should limit iterations for speed

    def test_recommend_top_k(self):
        """Test top-k recommendations"""
        selector = MethodSelector()

        problem = StandardizedProblem(
            dimension=20,
            objective_matrix=np.random.rand(20, 20),
            objective_function=lambda x: np.sum(x)
        )

        recommendations = selector.recommend_top_k(
            problem=None,
            standardized_problem=problem,
            k=3
        )

        assert len(recommendations) == 3
        assert all(isinstance(r, MethodRecommendation) for r in recommendations)
        # Check that recommendations are sorted by confidence
        confidences = [r.confidence for r in recommendations]
        assert confidences == sorted(confidences, reverse=True)

    def test_explain_recommendation(self):
        """Test recommendation explanation"""
        selector = MethodSelector()

        problem = StandardizedProblem(
            dimension=15,
            objective_matrix=np.random.rand(15, 15),
            objective_function=lambda x: np.sum(x),
            constraint_matrix=np.random.rand(5, 15)
        )

        explanation = selector.explain_recommendation(
            problem=None,
            standardized_problem=problem
        )

        assert isinstance(explanation, str)
        assert 'Method:' in explanation
        assert 'Confidence:' in explanation
        assert 'Problem Characteristics:' in explanation
        assert 'Dimension: 15' in explanation
        assert 'Has constraints: Yes' in explanation

    def test_time_budget_affects_recommendation(self):
        """Test that time budget affects recommendations"""
        selector = MethodSelector()

        problem = StandardizedProblem(
            dimension=50,
            objective_matrix=np.random.rand(50, 50),
            objective_function=lambda x: np.sum(x)
        )

        # Fast recommendation
        fast_method, fast_config, _ = selector.recommend_method(
            problem=None,
            standardized_problem=problem,
            time_budget=0.5
        )

        # Slow recommendation
        slow_method, slow_config, _ = selector.recommend_method(
            problem=None,
            standardized_problem=problem,
            time_budget=60.0
        )

        # Different time budgets may lead to different recommendations
        # or at least different configurations
        assert (fast_method != slow_method or
                fast_config.get('iterations', 0) < slow_config.get('iterations', 1))

    def test_record_performance(self):
        """Test performance recording"""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            log_path = f.name

        try:
            selector = MethodSelector(performance_log_path=log_path)

            problem = {'test': 'problem'}
            adapter = Mock()
            adapter.encode_problem.return_value = StandardizedProblem(
                dimension=10,
                objective_matrix=np.random.rand(10, 10)
            )

            result = {
                'objective': 123.45,
                'iterations': 1000,
                'is_valid': True
            }

            selector.record_performance(
                problem=problem,
                adapter=adapter,
                method='simulated_annealing',
                config={'iterations': 1000},
                result=result,
                runtime_seconds=5.2
            )

            # Check that performance was recorded
            with open(log_path, 'r') as f:
                line = f.readline()
                record = json.loads(line)
                assert record['method'] == 'simulated_annealing'
                assert record['objective_value'] == 123.45
                assert record['runtime_seconds'] == 5.2

        finally:
            Path(log_path).unlink(missing_ok=True)


class TestModels:
    """Tests for ML models"""

    def test_data_collector(self):
        """Test data collection for ML training"""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            storage_path = f.name

        try:
            collector = DataCollector(storage_path)

            # Collect some data
            for i in range(5):
                collector.collect(
                    problem_features={'dimension': 10 + i},
                    method='simulated_annealing',
                    config={'iterations': 1000},
                    performance={'objective': 100 - i}
                )

            collector.flush()

            # Load data back
            data = collector.load_data()
            assert len(data) == 5
            assert data[0]['method'] == 'simulated_annealing'

        finally:
            Path(storage_path).unlink(missing_ok=True)

    def test_method_selector_model_basic(self):
        """Test basic ML model structure"""
        model = MethodSelectorModel()

        assert model.model_type == 'ensemble'
        assert not model.is_trained
        assert len(model.feature_names) == 20
        assert len(model.method_names) == 5

    def test_model_training_placeholder(self):
        """Test model training (placeholder implementation)"""
        model = MethodSelectorModel()

        # Create mock training data
        training_data = [
            {
                'problem_features': {
                    'dimension': 10,
                    'matrix_sparsity': 0.5,
                    'estimated_difficulty': 'medium'
                },
                'method': 'simulated_annealing',
                'performance': {'objective': 100}
            }
            for _ in range(10)
        ]

        metrics = model.train(training_data, validation_split=0.2)

        assert model.is_trained
        assert 'train_accuracy' in metrics
        assert 'val_accuracy' in metrics

    def test_get_best_methods(self):
        """Test analysis of best methods from collected data"""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            storage_path = f.name

        try:
            collector = DataCollector(storage_path)

            # Collect performance data for different methods
            for method, performance in [
                ('simulated_annealing', 90),
                ('genetic_algorithm', 100),
                ('local_search', 110)
            ]:
                for _ in range(15):  # Multiple samples per method
                    collector.collect(
                        problem_features={
                            'problem_type': 'QAP',
                            'estimated_difficulty': 'medium',
                            'dimension': 25
                        },
                        method=method,
                        config={},
                        performance={'objective': performance + np.random.randn() * 5}
                    )

            collector.flush()

            # Analyze best methods
            best_methods = collector.get_best_methods(min_samples=10)

            # Check that analysis is correct
            assert len(best_methods) > 0
            qap_methods = best_methods.get('QAP_medium_size0', [])
            if qap_methods:
                # Simulated annealing should be best (lowest objective)
                assert qap_methods[0] == 'simulated_annealing'

        finally:
            Path(storage_path).unlink(missing_ok=True)


class TestIntegration:
    """Integration tests with main optimize function"""

    def test_optimize_with_auto_method(self):
        """Test optimize() with method='auto'"""
        from Librex import optimize

        # Create a simple problem
        problem = StandardizedProblem(
            dimension=10,
            objective_matrix=np.random.rand(10, 10),
            objective_function=lambda x: np.sum(x ** 2)
        )

        # Run optimization with auto method selection
        result = optimize(
            problem=problem,
            method='auto'
        )

        assert 'solution' in result
        assert 'objective' in result
        assert 'iterations' in result

    def test_optimize_auto_fallback(self):
        """Test that auto falls back gracefully on error"""
        from Librex import optimize

        problem = StandardizedProblem(
            dimension=5,
            objective_matrix=np.random.rand(5, 5),
            objective_function=lambda x: np.sum(x)
        )

        # Even if AI selector fails, optimization should still work
        with patch('Librex.ai.MethodSelector', side_effect=Exception("Test error")):
            result = optimize(
                problem=problem,
                method='auto'
            )

            # Should fall back to simulated annealing
            assert 'solution' in result
            assert 'objective' in result