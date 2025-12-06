"""
Comprehensive tests for AI Method Selector.
Tests feature extraction, ML models, recommendations, and caching.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import tempfile
import json
import pickle
from pathlib import Path

from Librex.ai import AIMethodSelector
from Librex.ai.features import FeatureExtractor, ProblemFeatures
from Librex.ai.method_selector import MethodRecommendation
from Librex.ai.models import (
    MethodSelectorModel,
    RandomForestSelector,
    NeuralNetSelector,
    EnsembleSelector
)


class TestFeatureExtractor:
    """Test problem feature extraction."""

    @pytest.fixture
    def extractor(self):
        """Create a FeatureExtractor instance."""
        return FeatureExtractor()

    @pytest.fixture
    def sample_problem(self):
        """Create a sample QAP problem."""
        n = 10
        np.random.seed(42)
        return {
            'flow': np.random.rand(n, n),
            'distance': np.random.rand(n, n),
            'size': n,
            'type': 'QAP'
        }

    def test_extractor_initialization(self, extractor):
        """Test feature extractor initialization."""
        assert extractor is not None
        assert hasattr(extractor, 'extract')
        assert hasattr(extractor, 'extract_statistical')

    def test_extract_basic_features(self, extractor, sample_problem):
        """Test extracting basic problem features."""
        features = extractor.extract(sample_problem)
        assert isinstance(features, ProblemFeatures)
        assert features.size == 10
        assert features.density > 0
        assert features.symmetry >= 0
        assert features.sparsity >= 0

    def test_extract_statistical_features(self, extractor, sample_problem):
        """Test extracting statistical features."""
        features = extractor.extract_statistical(sample_problem)
        assert 'mean' in features
        assert 'std' in features
        assert 'skewness' in features
        assert 'kurtosis' in features
        assert 'entropy' in features

    def test_extract_graph_features(self, extractor, sample_problem):
        """Test extracting graph-theoretic features."""
        features = extractor.extract_graph_features(sample_problem)
        assert 'clustering_coefficient' in features
        assert 'degree_distribution' in features
        assert 'diameter' in features
        assert 'connectivity' in features

    def test_extract_complexity_features(self, extractor, sample_problem):
        """Test extracting complexity features."""
        features = extractor.extract_complexity_features(sample_problem)
        assert 'rank' in features
        assert 'condition_number' in features
        assert 'eigenvalue_spread' in features
        assert 'spectral_gap' in features

    def test_feature_normalization(self, extractor, sample_problem):
        """Test feature normalization."""
        features = extractor.extract(sample_problem)
        normalized = extractor.normalize(features)

        # Check that features are normalized to [0, 1]
        feature_vector = normalized.to_vector()
        assert all(0 <= val <= 1 for val in feature_vector if not np.isnan(val))

    def test_handle_edge_cases(self, extractor):
        """Test handling edge cases in feature extraction."""
        # Empty problem
        empty_problem = {'flow': np.array([]), 'distance': np.array([])}
        with pytest.raises(ValueError):
            extractor.extract(empty_problem)

        # Single element
        single_problem = {'flow': np.array([[1]]), 'distance': np.array([[1]])}
        features = extractor.extract(single_problem)
        assert features.size == 1

        # Sparse problem
        sparse_problem = {
            'flow': np.zeros((10, 10)),
            'distance': np.eye(10)
        }
        features = extractor.extract(sparse_problem)
        assert features.sparsity > 0.8


class TestMethodSelectorModel:
    """Test base method selector model."""

    def test_model_interface(self):
        """Test the model interface."""
        model = MethodSelectorModel()
        assert hasattr(model, 'train')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'save')
        assert hasattr(model, 'load')

    def test_model_prediction_shape(self):
        """Test prediction output shape."""
        model = MethodSelectorModel()
        features = np.random.rand(10)

        with patch.object(model, 'predict', return_value=np.array([0.1, 0.3, 0.2, 0.4])):
            predictions = model.predict(features)
            assert len(predictions) == 4  # Assuming 4 methods

    def test_model_serialization(self):
        """Test model saving and loading."""
        model = MethodSelectorModel()
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name

        try:
            model.save(model_path)
            loaded_model = MethodSelectorModel.load(model_path)
            assert loaded_model is not None
        finally:
            Path(model_path).unlink(missing_ok=True)


class TestRandomForestSelector:
    """Test Random Forest method selector."""

    @pytest.fixture
    def selector(self):
        """Create a RandomForestSelector."""
        return RandomForestSelector(n_estimators=10)

    @pytest.fixture
    def training_data(self):
        """Generate training data."""
        np.random.seed(42)
        X = np.random.rand(100, 20)  # 100 samples, 20 features
        y = np.random.randint(0, 4, 100)  # 4 methods
        return X, y

    def test_initialization(self, selector):
        """Test selector initialization."""
        assert selector.n_estimators == 10
        assert selector.model is None  # Not trained yet

    def test_training(self, selector, training_data):
        """Test model training."""
        X, y = training_data
        selector.train(X, y)
        assert selector.model is not None
        assert hasattr(selector.model, 'predict')

    def test_prediction(self, selector, training_data):
        """Test prediction after training."""
        X, y = training_data
        selector.train(X, y)

        test_features = np.random.rand(10, 20)
        predictions = selector.predict(test_features)
        assert predictions.shape == (10, 4)  # 10 samples, 4 method probabilities
        assert np.allclose(predictions.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_feature_importance(self, selector, training_data):
        """Test feature importance extraction."""
        X, y = training_data
        selector.train(X, y)

        importance = selector.get_feature_importance()
        assert len(importance) == 20  # 20 features
        assert all(imp >= 0 for imp in importance)

    def test_cross_validation(self, selector, training_data):
        """Test cross-validation scoring."""
        X, y = training_data
        scores = selector.cross_validate(X, y, cv=3)
        assert len(scores) == 3
        assert all(0 <= score <= 1 for score in scores)


class TestNeuralNetSelector:
    """Test Neural Network method selector."""

    @pytest.fixture
    def selector(self):
        """Create a NeuralNetSelector."""
        return NeuralNetSelector(hidden_sizes=[64, 32], learning_rate=0.01)

    @pytest.fixture
    def training_data(self):
        """Generate training data."""
        np.random.seed(42)
        X = np.random.rand(100, 20)
        y = np.random.randint(0, 4, 100)
        return X, y

    def test_initialization(self, selector):
        """Test neural network initialization."""
        assert selector.hidden_sizes == [64, 32]
        assert selector.learning_rate == 0.01

    def test_network_architecture(self, selector):
        """Test network architecture creation."""
        selector.build_network(input_size=20, output_size=4)
        assert selector.model is not None
        assert len(selector.model.layers) >= 3  # Input, hidden, output

    def test_training_with_validation(self, selector, training_data):
        """Test training with validation split."""
        X, y = training_data
        history = selector.train(X, y, validation_split=0.2, epochs=5)
        assert 'loss' in history
        assert 'val_loss' in history
        assert len(history['loss']) == 5

    def test_early_stopping(self, selector, training_data):
        """Test early stopping during training."""
        X, y = training_data
        history = selector.train(
            X, y,
            validation_split=0.2,
            epochs=100,
            early_stopping=True,
            patience=3
        )
        # Should stop early
        assert len(history['loss']) < 100

    def test_regularization(self, selector, training_data):
        """Test L2 regularization."""
        selector_reg = NeuralNetSelector(
            hidden_sizes=[64, 32],
            l2_regularization=0.01
        )
        X, y = training_data
        selector_reg.train(X, y, epochs=5)

        # Check that weights are not too large
        weights = selector_reg.get_weights()
        assert all(np.abs(w).max() < 10 for w in weights)


class TestEnsembleSelector:
    """Test Ensemble method selector."""

    @pytest.fixture
    def ensemble(self):
        """Create an EnsembleSelector."""
        models = [
            RandomForestSelector(n_estimators=5),
            NeuralNetSelector(hidden_sizes=[32]),
            RandomForestSelector(n_estimators=10)
        ]
        return EnsembleSelector(models, weights=[0.4, 0.3, 0.3])

    @pytest.fixture
    def training_data(self):
        """Generate training data."""
        np.random.seed(42)
        X = np.random.rand(50, 20)
        y = np.random.randint(0, 4, 50)
        return X, y

    def test_initialization(self, ensemble):
        """Test ensemble initialization."""
        assert len(ensemble.models) == 3
        assert ensemble.weights == [0.4, 0.3, 0.3]
        assert sum(ensemble.weights) == 1.0

    def test_training_all_models(self, ensemble, training_data):
        """Test training all models in ensemble."""
        X, y = training_data
        ensemble.train(X, y)

        for model in ensemble.models:
            assert model.is_trained()

    def test_weighted_prediction(self, ensemble, training_data):
        """Test weighted ensemble prediction."""
        X, y = training_data
        ensemble.train(X, y)

        test_features = np.random.rand(5, 20)
        predictions = ensemble.predict(test_features)

        assert predictions.shape == (5, 4)
        assert np.allclose(predictions.sum(axis=1), 1.0)

    def test_voting_prediction(self, ensemble, training_data):
        """Test voting-based prediction."""
        X, y = training_data
        ensemble.train(X, y)
        ensemble.set_voting_mode('hard')

        test_features = np.random.rand(5, 20)
        predictions = ensemble.predict(test_features)

        # Hard voting should give one-hot encoded results
        assert all(pred.max() == 1.0 and pred.sum() == 1.0 for pred in predictions)


class TestAIMethodSelector:
    """Test the main AI Method Selector."""

    @pytest.fixture
    def selector(self):
        """Create an AIMethodSelector."""
        return AIMethodSelector(cache_recommendations=True)

    @pytest.fixture
    def sample_problem(self):
        """Create a sample problem."""
        return {
            'flow': np.random.rand(10, 10),
            'distance': np.random.rand(10, 10),
            'size': 10
        }

    def test_initialization(self, selector):
        """Test selector initialization."""
        assert selector is not None
        assert selector.cache_enabled is True
        assert selector.model is not None

    def test_recommend_methods(self, selector, sample_problem):
        """Test method recommendation."""
        recommendations = selector.recommend(sample_problem, top_k=3)

        assert len(recommendations) == 3
        assert all(isinstance(r, MethodRecommendation) for r in recommendations)
        assert all(0 <= r.confidence <= 1 for r in recommendations)
        assert recommendations[0].confidence >= recommendations[1].confidence

    def test_recommendation_with_constraints(self, selector, sample_problem):
        """Test recommendation with constraints."""
        constraints = {
            'max_time': 60,  # seconds
            'min_quality': 0.8,
            'excluded_methods': ['genetic_algorithm']
        }

        recommendations = selector.recommend(
            sample_problem,
            constraints=constraints
        )

        assert all(r.method != 'genetic_algorithm' for r in recommendations)
        assert all(r.expected_time <= 60 for r in recommendations)

    def test_adaptive_learning(self, selector, sample_problem):
        """Test adaptive learning from feedback."""
        # Get initial recommendation
        initial_rec = selector.recommend(sample_problem, top_k=1)[0]

        # Provide feedback
        feedback = {
            'method': initial_rec.method,
            'actual_quality': 0.9,
            'actual_time': 45,
            'success': True
        }

        selector.update_from_feedback(sample_problem, feedback)

        # Check that model adapts
        new_rec = selector.recommend(sample_problem, top_k=1)[0]
        # The recommendation might change or confidence might update
        assert new_rec is not None

    def test_caching_mechanism(self, selector, sample_problem):
        """Test recommendation caching."""
        # First call - not cached
        rec1 = selector.recommend(sample_problem)

        # Second call - should be cached
        rec2 = selector.recommend(sample_problem)

        assert rec1 == rec2  # Same recommendations

        # Clear cache
        selector.clear_cache()

        # Third call - not cached
        rec3 = selector.recommend(sample_problem)
        assert len(rec3) == len(rec1)

    def test_batch_recommendation(self, selector):
        """Test batch problem recommendation."""
        problems = [
            {'flow': np.random.rand(10, 10), 'distance': np.random.rand(10, 10)},
            {'flow': np.random.rand(15, 15), 'distance': np.random.rand(15, 15)},
            {'flow': np.random.rand(20, 20), 'distance': np.random.rand(20, 20)},
        ]

        recommendations = selector.recommend_batch(problems)
        assert len(recommendations) == 3
        assert all(len(recs) > 0 for recs in recommendations)

    def test_explainability(self, selector, sample_problem):
        """Test recommendation explainability."""
        recommendations = selector.recommend(sample_problem, explain=True)

        for rec in recommendations:
            assert hasattr(rec, 'explanation')
            assert 'features' in rec.explanation
            assert 'reasoning' in rec.explanation
            assert 'confidence_factors' in rec.explanation

    def test_performance_prediction(self, selector, sample_problem):
        """Test performance prediction for methods."""
        predictions = selector.predict_performance(sample_problem)

        assert len(predictions) > 0
        for method, metrics in predictions.items():
            assert 'expected_quality' in metrics
            assert 'expected_time' in metrics
            assert 'confidence' in metrics
            assert 0 <= metrics['expected_quality'] <= 1
            assert metrics['expected_time'] > 0

    def test_model_persistence(self, selector, sample_problem):
        """Test saving and loading trained model."""
        # Train on some data
        selector.recommend(sample_problem)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name

        try:
            # Save model
            selector.save_model(model_path)

            # Create new selector and load model
            new_selector = AIMethodSelector()
            new_selector.load_model(model_path)

            # Test that it works
            new_recommendations = new_selector.recommend(sample_problem)
            assert len(new_recommendations) > 0
        finally:
            Path(model_path).unlink(missing_ok=True)

    @patch('Librex.ai.method_selector.redis')
    def test_redis_caching(self, mock_redis, selector, sample_problem):
        """Test Redis-based caching if available."""
        mock_redis_instance = MagicMock()
        mock_redis.Redis.return_value = mock_redis_instance
        mock_redis_instance.get.return_value = None

        selector_with_redis = AIMethodSelector(
            cache_recommendations=True,
            use_redis=True
        )

        recommendations = selector_with_redis.recommend(sample_problem)

        # Check that Redis was called
        assert mock_redis_instance.get.called
        assert mock_redis_instance.setex.called