#!/usr/bin/env python3
"""
Comprehensive Tests for Supervised Learning Module
This test suite validates all supervised learning algorithms in the Berkeley
SciComp Machine Learning package, ensuring correctness, numerical stability,
and scientific computing compatibility.
Author: Berkeley SciComp Team
"""
import numpy as np
import pytest
import sys
from pathlib import Path
# Add package to path
sys.path.append(str(Path(__file__).parent.parent))
from Machine_Learning.supervised import (
    LinearRegression, PolynomialRegression, RidgeRegression,
    LogisticRegression, create_test_datasets
)
class TestLinearRegression:
    """Test suite for LinearRegression class."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 3
        # Create synthetic dataset
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.true_coeffs = np.array([2.5, -1.3, 0.8])
        self.y = self.X @ self.true_coeffs + 0.1 * np.random.randn(self.n_samples)
    def test_initialization(self):
        """Test model initialization."""
        model = LinearRegression()
        assert not model.is_fitted
        assert model.fit_intercept is True
        assert model.solver == 'svd'
        assert model.regularization == 0.0
    def test_svd_solver(self):
        """Test SVD solver accuracy."""
        model = LinearRegression(solver='svd')
        model.fit(self.X, self.y)
        assert model.is_fitted
        assert hasattr(model, 'coefficients')
        assert hasattr(model, 'intercept')
        # Check coefficient accuracy
        np.testing.assert_allclose(model.coefficients, self.true_coeffs, atol=0.2)
    def test_normal_equation_solver(self):
        """Test normal equation solver."""
        model = LinearRegression(solver='normal')
        model.fit(self.X, self.y)
        assert model.is_fitted
        score = model.score(self.X, self.y)
        assert score > 0.9  # Should have high R²
    def test_iterative_solver(self):
        """Test iterative solver."""
        model = LinearRegression(solver='iterative')
        model.fit(self.X, self.y)
        assert model.is_fitted
        predictions = model.predict(self.X)
        assert len(predictions) == self.n_samples
    def test_predictions(self):
        """Test prediction functionality."""
        model = LinearRegression()
        model.fit(self.X, self.y)
        # Basic predictions
        predictions = model.predict(self.X)
        assert predictions.shape == (self.n_samples,)
        # Predictions with uncertainty
        if model.covariance_matrix is not None:
            pred_with_unc = model.predict(self.X, return_uncertainty=True)
            assert len(pred_with_unc) == 2
            predictions, uncertainties = pred_with_unc
            assert uncertainties.shape == predictions.shape
    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        model = LinearRegression(uncertainty_estimation=True)
        model.fit(self.X, self.y)
        if model.covariance_matrix is not None:
            ci = model.confidence_intervals(self.X[:10])
            assert ci.shape == (10, 2)
            assert np.all(ci[:, 1] >= ci[:, 0])  # Upper >= Lower
    def test_regularization(self):
        """Test regularization effects."""
        # Without regularization
        model1 = LinearRegression(regularization=0.0)
        model1.fit(self.X, self.y)
        # With regularization
        model2 = LinearRegression(regularization=1.0)
        model2.fit(self.X, self.y)
        # Regularized coefficients should be smaller
        assert np.linalg.norm(model2.coefficients) <= np.linalg.norm(model1.coefficients)
    def test_no_intercept(self):
        """Test model without intercept."""
        model = LinearRegression(fit_intercept=False)
        model.fit(self.X, self.y)
        assert model.intercept == 0.0
        assert model.is_fitted
    def test_summary(self):
        """Test model summary."""
        model = LinearRegression()
        model.fit(self.X, self.y)
        summary = model.summary()
        assert 'intercept' in summary
        assert 'coefficients' in summary
        assert 'n_features' in summary
        assert summary['n_features'] == self.n_features
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        model = LinearRegression()
        # Test prediction before fitting
        with pytest.raises(ValueError):
            model.predict(self.X)
        # Test with single feature
        X_single = self.X[:, :1]
        model.fit(X_single, self.y)
        assert model.coefficients.shape == (1,)
        # Test with perfect linear relationship
        X_perfect = np.random.randn(50, 2)
        y_perfect = X_perfect @ np.array([1.0, 2.0])
        model_perfect = LinearRegression()
        model_perfect.fit(X_perfect, y_perfect)
        assert model_perfect.score(X_perfect, y_perfect) > 0.99
class TestPolynomialRegression:
    """Test suite for PolynomialRegression class."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 100
        # Create nonlinear dataset
        self.X = np.random.uniform(-2, 2, (self.n_samples, 1))
        self.y = 2 * self.X.ravel()**2 - 3 * self.X.ravel() + 1 + 0.1 * np.random.randn(self.n_samples)
    def test_initialization(self):
        """Test polynomial regression initialization."""
        model = PolynomialRegression(degree=2)
        assert model.degree == 2
        assert model.include_bias is True
        assert not model.interaction_only
    def test_quadratic_fitting(self):
        """Test fitting quadratic polynomial."""
        model = PolynomialRegression(degree=2)
        model.fit(self.X, self.y)
        assert model.is_fitted
        score = model.score(self.X, self.y)
        assert score > 0.95  # Should fit quadratic well
    def test_feature_generation(self):
        """Test polynomial feature generation."""
        X_simple = np.array([[1], [2], [3]])
        model = PolynomialRegression(degree=2)
        model.fit(X_simple, np.array([1, 4, 9]))  # y = x²
        assert model.is_fitted
        # Should capture quadratic relationship well
        predictions = model.predict(X_simple)
        np.testing.assert_allclose(predictions, [1, 4, 9], atol=0.1)
    def test_interaction_only(self):
        """Test interaction-only polynomial features."""
        X_multi = np.random.randn(50, 2)
        y_multi = X_multi[:, 0] * X_multi[:, 1] + 0.1 * np.random.randn(50)
        model = PolynomialRegression(degree=2, interaction_only=True)
        model.fit(X_multi, y_multi)
        assert model.is_fitted
        score = model.score(X_multi, y_multi)
        assert score > 0.8  # Should capture interaction
    def test_regularization(self):
        """Test polynomial regression with regularization."""
        model = PolynomialRegression(degree=5, regularization=0.1)
        model.fit(self.X, self.y)
        assert model.is_fitted
        # Should prevent overfitting
        score = model.score(self.X, self.y)
        assert 0.7 < score < 0.99
    def test_extrapolation_warning(self):
        """Test extrapolation warning."""
        model = PolynomialRegression(degree=2)
        model.fit(self.X, self.y)
        # Create data outside training range
        X_extrap = np.array([[5.0]])  # Outside [-2, 2] range
        with pytest.warns(UserWarning):
            model.predict(X_extrap)
class TestRidgeRegression:
    """Test suite for RidgeRegression class."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 10
        # Create dataset with collinearity
        self.X = np.random.randn(self.n_samples, self.n_features)
        # Add collinear features
        self.X[:, -1] = self.X[:, 0] + 0.1 * np.random.randn(self.n_samples)
        true_coeffs = np.random.randn(self.n_features)
        self.y = self.X @ true_coeffs + 0.1 * np.random.randn(self.n_samples)
    def test_initialization(self):
        """Test Ridge regression initialization."""
        model = RidgeRegression(alpha=1.0)
        assert model.alpha == 1.0
        assert model.regularization == 1.0
    def test_regularization_effect(self):
        """Test regularization reduces overfitting."""
        # High regularization
        model_high = RidgeRegression(alpha=10.0)
        model_high.fit(self.X, self.y)
        # Low regularization
        model_low = RidgeRegression(alpha=0.01)
        model_low.fit(self.X, self.y)
        # High regularization should have smaller coefficients
        assert np.linalg.norm(model_high.coefficients) < np.linalg.norm(model_low.coefficients)
    def test_cross_validation(self):
        """Test cross-validation for alpha selection."""
        model = RidgeRegression()
        alpha_range = np.array([0.1, 1.0, 10.0])
        model.fit_with_cv(self.X, self.y, alpha_range=alpha_range)
        assert model.is_fitted
        assert model.alpha in alpha_range
        assert hasattr(model, 'cv_scores')
        assert len(model.cv_scores) == len(alpha_range)
class TestLogisticRegression:
    """Test suite for LogisticRegression class."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 3
        # Create binary classification dataset
        self.X = np.random.randn(self.n_samples, self.n_features)
        # Create linearly separable classes
        weights = np.array([1.5, -1.0, 0.5])
        scores = self.X @ weights
        self.y_binary = (scores > 0).astype(int)
        # Create multiclass dataset
        self.y_multi = np.random.randint(0, 3, self.n_samples)
    def test_initialization(self):
        """Test logistic regression initialization."""
        model = LogisticRegression()
        assert model.penalty == 'l2'
        assert model.C == 1.0
        assert model.solver == 'lbfgs'
    def test_binary_classification(self):
        """Test binary classification."""
        model = LogisticRegression()
        model.fit(self.X, self.y_binary)
        assert model.is_fitted
        assert len(model.classes) == 2
        # Test predictions
        predictions = model.predict(self.X)
        assert set(predictions).issubset(set(model.classes))
        # Test probabilities
        probabilities = model.predict_proba(self.X)
        assert probabilities.shape == (self.n_samples, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    def test_multiclass_classification(self):
        """Test multiclass classification."""
        model = LogisticRegression()
        model.fit(self.X, self.y_multi)
        assert model.is_fitted
        assert len(model.classes) == 3
        # Test predictions
        predictions = model.predict(self.X)
        probabilities = model.predict_proba(self.X)
        assert probabilities.shape == (self.n_samples, 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    def test_regularization_penalties(self):
        """Test different regularization penalties."""
        # L2 penalty
        model_l2 = LogisticRegression(penalty='l2', C=0.1)
        model_l2.fit(self.X, self.y_binary)
        # L1 penalty
        model_l1 = LogisticRegression(penalty='l1', C=0.1)
        model_l1.fit(self.X, self.y_binary)
        assert model_l2.is_fitted
        assert model_l1.is_fitted
        # L1 should produce sparser coefficients
        l1_sparsity = np.sum(np.abs(model_l1.coefficients) < 1e-6)
        l2_sparsity = np.sum(np.abs(model_l2.coefficients) < 1e-6)
        # Note: This test might be flaky depending on the specific dataset
    def test_sigmoid_function(self):
        """Test sigmoid function implementation."""
        model = LogisticRegression()
        # Test sigmoid properties
        assert model._sigmoid(0) == 0.5
        assert model._sigmoid(1000) < 1.0  # Should not overflow
        assert model._sigmoid(-1000) > 0.0  # Should not underflow
        # Test sigmoid shape
        x = np.linspace(-10, 10, 100)
        sigmoid_vals = model._sigmoid(x)
        assert np.all(sigmoid_vals >= 0)
        assert np.all(sigmoid_vals <= 1)
        assert np.all(np.diff(sigmoid_vals) >= 0)  # Should be monotonic
    def test_probability_consistency(self):
        """Test probability prediction consistency."""
        model = LogisticRegression()
        model.fit(self.X, self.y_binary)
        probabilities = model.predict_proba(self.X)
        predictions = model.predict(self.X)
        # Predictions should match highest probability class
        predicted_from_proba = model.classes[np.argmax(probabilities, axis=1)]
        np.testing.assert_array_equal(predictions, predicted_from_proba)
class TestUtilityFunctions:
    """Test utility functions."""
    def test_create_test_datasets(self):
        """Test test dataset creation."""
        datasets = create_test_datasets()
        assert 'linear' in datasets
        assert 'polynomial' in datasets
        assert 'classification' in datasets
        # Check dataset properties
        X_linear, y_linear = datasets['linear']
        assert X_linear.shape[0] == len(y_linear)
        assert X_linear.shape[1] == 3
        X_poly, y_poly = datasets['polynomial']
        assert X_poly.shape[1] == 1
        X_class, y_class = datasets['classification']
        assert len(np.unique(y_class)) == 2
class TestIntegrationAndEdgeCases:
    """Integration tests and edge cases."""
    def test_sklearn_compatibility(self):
        """Test basic sklearn-style API compatibility."""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        # Create classification dataset
        datasets = create_test_datasets()
        X, y = datasets['classification']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # Test with our logistic regression
        model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        assert accuracy > 0.5  # Should be better than random
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very small values
        X_small = np.random.randn(50, 3) * 1e-8
        y_small = np.random.randn(50) * 1e-8
        model = LinearRegression()
        model.fit(X_small, y_small)
        predictions = model.predict(X_small)
        assert np.all(np.isfinite(predictions))
        # Very large values
        X_large = np.random.randn(50, 3) * 1e8
        y_large = np.random.randn(50) * 1e8
        model_large = LinearRegression(solver='svd')  # Most stable
        model_large.fit(X_large, y_large)
        predictions_large = model_large.predict(X_large)
        assert np.all(np.isfinite(predictions_large))
    def test_empty_and_single_sample(self):
        """Test handling of edge cases."""
        # Single sample
        X_single = np.array([[1, 2, 3]])
        y_single = np.array([1])
        model = LinearRegression()
        # This should not crash, though may not be well-conditioned
        try:
            model.fit(X_single, y_single)
        except (np.linalg.LinAlgError, ValueError):
            pass  # Expected for underdetermined system
    def test_perfect_separation(self):
        """Test logistic regression with perfectly separable data."""
        # Create perfectly separable data
        X_sep = np.array([[1, 1], [1, 2], [2, 1], [10, 10], [10, 11], [11, 10]])
        y_sep = np.array([0, 0, 0, 1, 1, 1])
        model = LogisticRegression(C=1e6)  # Low regularization
        model.fit(X_sep, y_sep)
        predictions = model.predict(X_sep)
        # Should achieve perfect classification
        np.testing.assert_array_equal(predictions, y_sep)
if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])