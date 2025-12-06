#!/usr/bin/env python3
"""
Test suite for ML-based solver recommendation engine.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas_core.ml_recommender import (
    ProblemFeatures,
    FeatureExtractor,
    SolverRecommender,
    generate_synthetic_training_data
)


class TestProblemFeatures:
    """Test ProblemFeatures dataclass."""

    def test_default_initialization(self):
        """Test default feature values."""
        features = ProblemFeatures()
        assert features.num_variables == 0
        assert features.num_objectives == 1
        assert features.is_continuous == True
        assert features.is_combinatorial == False

    def test_to_array(self):
        """Test conversion to numpy array."""
        features = ProblemFeatures(
            num_variables=100,
            num_constraints=50,
            constraint_density=0.5
        )
        array = features.to_array()
        assert isinstance(array, np.ndarray)
        assert array.shape == (28,)  # 28 features
        assert array[0] == 100  # num_variables
        assert array[1] == 50   # num_constraints

    def test_feature_names(self):
        """Test feature name retrieval."""
        names = ProblemFeatures.feature_names()
        assert isinstance(names, list)
        assert len(names) == 28
        assert 'num_variables' in names
        assert 'is_combinatorial' in names


class TestFeatureExtractor:
    """Test FeatureExtractor class."""

    def test_initialization(self):
        """Test extractor initialization."""
        extractor = FeatureExtractor()
        assert extractor.feature_cache == {}

    def test_extract_basic_features(self):
        """Test basic feature extraction."""
        extractor = FeatureExtractor()
        problem = {
            'num_variables': 100,
            'num_constraints': 50,
            'num_objectives': 2,
            'type': 'continuous'
        }
        features = extractor.extract_features(problem)

        assert features.num_variables == 100
        assert features.num_constraints == 50
        assert features.num_objectives == 2
        assert features.constraint_density == 0.5

    def test_extract_constraint_features(self):
        """Test constraint feature extraction."""
        extractor = FeatureExtractor()
        problem = {
            'num_constraints': 100,
            'constraints': {
                'linear': 60,
                'nonlinear': 40,
                'equality': 30
            }
        }
        features = extractor.extract_features(problem)

        assert features.linear_constraints_ratio == 0.6
        assert features.nonlinear_constraints_ratio == 0.4
        assert features.equality_constraints_ratio == 0.3

    def test_extract_problem_type_features(self):
        """Test problem type feature extraction."""
        extractor = FeatureExtractor()

        # Combinatorial problem
        problem = {'type': 'combinatorial'}
        features = extractor.extract_features(problem)
        assert features.is_combinatorial == True

        # Continuous problem
        problem = {'type': 'continuous nonlinear'}
        features = extractor.extract_features(problem)
        assert features.is_continuous == True
        assert features.is_nonlinear == True

    def test_extract_batch(self):
        """Test batch feature extraction."""
        extractor = FeatureExtractor()
        problems = [
            {'num_variables': 10, 'type': 'continuous'},
            {'num_variables': 20, 'type': 'combinatorial'},
            {'num_variables': 30, 'type': 'mixed'},
        ]
        batch_features = extractor.extract_batch(problems)

        assert batch_features.shape == (3, 28)
        assert batch_features[0, 0] == 10
        assert batch_features[1, 0] == 20
        assert batch_features[2, 0] == 30


class TestSolverRecommender:
    """Test SolverRecommender class."""

    def test_initialization(self):
        """Test recommender initialization."""
        recommender = SolverRecommender()
        assert recommender.model_type == 'random_forest'
        assert recommender.confidence_threshold == 0.6
        assert recommender.model is not None

    def test_different_model_types(self):
        """Test initialization with different model types."""
        models = ['random_forest', 'extra_trees', 'neural_network', 'svm', 'logistic']

        for model_type in models:
            recommender = SolverRecommender(model_type=model_type)
            assert recommender.model_type == model_type
            assert recommender.model is not None

    def test_train(self):
        """Test model training."""
        recommender = SolverRecommender()
        training_data = generate_synthetic_training_data(n_samples=100)

        metrics = recommender.train(
            training_data,
            perform_hyperopt=False,
            cv_folds=3
        )

        assert 'accuracy' in metrics
        assert 'cv_accuracy_mean' in metrics
        assert metrics['accuracy'] >= 0.0
        assert metrics['accuracy'] <= 1.0
        assert metrics['training_samples'] == 100

    def test_predict_without_training(self):
        """Test prediction without training (should use rule-based)."""
        recommender = SolverRecommender()
        recommender.model = None  # Ensure no model

        problem = {
            'type': 'continuous',
            'objective': {'convexity': 0.9}
        }

        solver = recommender.predict(problem, return_confidence=False)
        assert solver in SolverRecommender.AVAILABLE_SOLVERS

    def test_predict_with_training(self):
        """Test prediction after training."""
        recommender = SolverRecommender()
        training_data = generate_synthetic_training_data(n_samples=100)
        recommender.train(training_data, perform_hyperopt=False)

        problem = training_data[0]['problem']
        solver, confidence = recommender.predict(problem, return_confidence=True)

        assert solver in SolverRecommender.AVAILABLE_SOLVERS
        assert 0.0 <= confidence <= 1.0

    def test_predict_top_k(self):
        """Test top-k predictions."""
        recommender = SolverRecommender()
        training_data = generate_synthetic_training_data(n_samples=100)
        recommender.train(training_data, perform_hyperopt=False)

        problem = training_data[0]['problem']
        top_3 = recommender.predict_top_k(problem, k=3)

        assert len(top_3) <= 3
        assert all(solver in SolverRecommender.AVAILABLE_SOLVERS
                  for solver, _ in top_3)
        assert all(0.0 <= conf <= 1.0 for _, conf in top_3)

        # Check descending order
        confidences = [conf for _, conf in top_3]
        assert confidences == sorted(confidences, reverse=True)

    def test_update_online(self):
        """Test online learning update."""
        recommender = SolverRecommender()
        training_data = generate_synthetic_training_data(n_samples=50)
        recommender.train(training_data, perform_hyperopt=False)

        initial_history_len = len(recommender.training_history)

        problem = training_data[0]['problem']
        recommender.update_online(problem, 'genetic_algorithm', 0.85)

        assert len(recommender.training_history) == initial_history_len + 1

    def test_rule_based_selection(self):
        """Test rule-based fallback selection."""
        recommender = SolverRecommender()

        # Test combinatorial problem
        problem = {'type': 'combinatorial', 'graph': {'density': 0.6}}
        features = recommender.feature_extractor.extract_features(problem)
        features.is_combinatorial = True
        solver = recommender._rule_based_selection(problem)
        assert solver == 'tabu_search'

        # Test convex continuous problem
        problem = {
            'type': 'continuous',
            'objective': {'convexity': 0.9, 'smoothness': 0.9}
        }
        solver = recommender._rule_based_selection(problem)
        assert solver in ['gradient_descent', 'interior_point']

    def test_save_and_load_model(self):
        """Test model persistence."""
        recommender = SolverRecommender()
        training_data = generate_synthetic_training_data(n_samples=100)
        recommender.train(training_data, perform_hyperopt=False)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'test_model.pkl'
            recommender.save_model(model_path)

            assert model_path.exists()

            # Load model
            new_recommender = SolverRecommender()
            new_recommender.load_model(model_path)

            # Check consistency
            problem = training_data[0]['problem']
            solver1 = recommender.predict(problem, return_confidence=False)
            solver2 = new_recommender.predict(problem, return_confidence=False)
            assert solver1 == solver2

    def test_evaluate(self):
        """Test model evaluation."""
        recommender = SolverRecommender()
        data = generate_synthetic_training_data(n_samples=150)
        train_data = data[:100]
        test_data = data[100:]

        recommender.train(train_data, perform_hyperopt=False)
        metrics = recommender.evaluate(test_data)

        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics
        assert 'test_samples' in metrics
        assert metrics['test_samples'] == 50
        assert 0.0 <= metrics['accuracy'] <= 1.0

    def test_feature_importance(self):
        """Test feature importance extraction."""
        recommender = SolverRecommender(model_type='random_forest')
        training_data = generate_synthetic_training_data(n_samples=100)
        metrics = recommender.train(training_data, perform_hyperopt=False)

        if 'feature_importance' in metrics:
            importance = metrics['feature_importance']
            assert isinstance(importance, dict)
            # Check that values sum to approximately 1
            if importance:
                total = sum(importance.values())
                assert 0.9 <= total <= 1.1

    def test_cross_validation(self):
        """Test cross-validation during training."""
        recommender = SolverRecommender()
        training_data = generate_synthetic_training_data(n_samples=100)

        metrics = recommender.train(training_data, cv_folds=5)
        assert 'cv_accuracy_mean' in metrics
        assert 'cv_accuracy_std' in metrics
        assert metrics['cv_accuracy_std'] >= 0.0


class TestSyntheticDataGeneration:
    """Test synthetic data generation."""

    def test_generate_synthetic_data(self):
        """Test synthetic training data generation."""
        data = generate_synthetic_training_data(n_samples=50)

        assert len(data) == 50

        for example in data:
            assert 'problem' in example
            assert 'best_solver' in example
            assert example['best_solver'] in SolverRecommender.AVAILABLE_SOLVERS

            problem = example['problem']
            assert 'num_variables' in problem
            assert 'type' in problem
            assert problem['type'] in ['continuous', 'combinatorial', 'mixed']


def test_end_to_end_workflow():
    """Test complete workflow from training to prediction."""
    # Generate data
    all_data = generate_synthetic_training_data(n_samples=200)
    train_data = all_data[:150]
    test_data = all_data[150:]

    # Initialize and train
    recommender = SolverRecommender(model_type='random_forest')
    train_metrics = recommender.train(train_data, perform_hyperopt=False, cv_folds=3)

    # Evaluate
    test_metrics = recommender.evaluate(test_data)

    # Make predictions
    for example in test_data[:5]:
        problem = example['problem']
        solver, confidence = recommender.predict(problem, return_confidence=True)
        assert solver in SolverRecommender.AVAILABLE_SOLVERS
        assert 0.0 <= confidence <= 1.0

        # Get top-k recommendations
        top_k = recommender.predict_top_k(problem, k=3)
        assert len(top_k) <= 3

    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / 'final_model.pkl'
        recommender.save_model(model_path)

        new_recommender = SolverRecommender()
        new_recommender.load_model(model_path)

        # Verify consistency
        problem = test_data[0]['problem']
        solver1, conf1 = recommender.predict(problem, return_confidence=True)
        solver2, conf2 = new_recommender.predict(problem, return_confidence=True)
        assert solver1 == solver2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])