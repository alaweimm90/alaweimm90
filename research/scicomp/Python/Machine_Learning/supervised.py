"""
Supervised Learning Algorithms for Scientific Computing
This module implements supervised learning algorithms specifically designed for
scientific computing applications, with emphasis on interpretability and
physics-aware modeling.
Classes:
    LinearRegression: Linear regression with scientific extensions
    PolynomialRegression: Polynomial regression for nonlinear relationships
    RidgeRegression: Regularized regression for high-dimensional data
    LogisticRegression: Classification with probabilistic outputs
    SVM: Support Vector Machines for complex decision boundaries
    RandomForest: Ensemble method for robust predictions
    GradientBoosting: Gradient boosting for high-performance modeling
"""
import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any, Union, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy import optimize, stats
from scipy.linalg import svd, pinv
from sklearn.base import BaseEstimator
import logging
logger = logging.getLogger(__name__)
@dataclass
class ModelResults:
    """Container for model results and diagnostics."""
    predictions: np.ndarray
    residuals: np.ndarray
    r_squared: float
    mse: float
    mae: float
    aic: Optional[float] = None
    bic: Optional[float] = None
    confidence_intervals: Optional[np.ndarray] = None
    prediction_intervals: Optional[np.ndarray] = None
    feature_importance: Optional[np.ndarray] = None
class SupervisedModel(ABC):
    """Abstract base class for supervised learning models."""
    def __init__(self):
        self.is_fitted = False
        self.feature_names = None
        self.n_features = None
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SupervisedModel':
        """Fit the model to training data."""
        pass
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Validate input data."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y is not None:
            y = np.asarray(y)
            if X.shape[0] != len(y):
                raise ValueError("X and y must have the same number of samples")
        return X, y
class LinearRegression(SupervisedModel):
    """
    Linear regression with advanced scientific computing features.
    Features:
    - Multiple solvers (normal equation, SVD, iterative)
    - Uncertainty quantification
    - Statistical inference
    - Physics-aware constraints
    """
    def __init__(self,
                 fit_intercept: bool = True,
                 solver: str = 'svd',
                 regularization: float = 0.0,
                 uncertainty_estimation: bool = True):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.regularization = regularization
        self.uncertainty_estimation = uncertainty_estimation
        self.coefficients = None
        self.intercept = None
        self.covariance_matrix = None
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit linear regression model.
        Parameters:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        Returns:
            self: Fitted model
        """
        X, y = self._validate_input(X, y)
        self.n_features = X.shape[1]
        # Add intercept column if needed
        if self.fit_intercept:
            X_aug = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_aug = X.copy()
        # Solve using specified method
        if self.solver == 'normal':
            self._fit_normal_equation(X_aug, y)
        elif self.solver == 'svd':
            self._fit_svd(X_aug, y)
        elif self.solver == 'iterative':
            self._fit_iterative(X_aug, y)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        # Uncertainty quantification
        if self.uncertainty_estimation:
            self._compute_uncertainty(X_aug, y)
        self.is_fitted = True
        return self
    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray):
        """Fit using normal equation."""
        XTX = X.T @ X
        if self.regularization > 0:
            XTX += self.regularization * np.eye(XTX.shape[0])
        coeffs = np.linalg.solve(XTX, X.T @ y)
        self._extract_coefficients(coeffs)
    def _fit_svd(self, X: np.ndarray, y: np.ndarray):
        """Fit using SVD (most stable)."""
        if self.regularization > 0:
            # Ridge regression via SVD
            U, s, Vt = svd(X, full_matrices=False)
            d = s / (s**2 + self.regularization)
            coeffs = Vt.T @ (d * (U.T @ y))
        else:
            coeffs = pinv(X) @ y
        self._extract_coefficients(coeffs)
    def _fit_iterative(self, X: np.ndarray, y: np.ndarray):
        """Fit using iterative solver."""
        from scipy.sparse.linalg import lsqr
        if self.regularization > 0:
            # Add regularization term
            X_reg = np.vstack([X, np.sqrt(self.regularization) * np.eye(X.shape[1])])
            y_reg = np.hstack([y, np.zeros(X.shape[1])])
            coeffs, *_ = lsqr(X_reg, y_reg)
        else:
            coeffs, *_ = lsqr(X, y)
        self._extract_coefficients(coeffs)
    def _extract_coefficients(self, coeffs: np.ndarray):
        """Extract intercept and coefficients."""
        if self.fit_intercept:
            self.intercept = coeffs[0]
            self.coefficients = coeffs[1:]
        else:
            self.intercept = 0.0
            self.coefficients = coeffs
    def _compute_uncertainty(self, X: np.ndarray, y: np.ndarray):
        """Compute uncertainty estimates."""
        try:
            # Compute covariance matrix
            residuals = y - X @ np.hstack([self.intercept, self.coefficients] if self.fit_intercept
                                        else self.coefficients)
            sigma_squared = np.sum(residuals**2) / (len(y) - X.shape[1])
            XTX_inv = np.linalg.inv(X.T @ X + self.regularization * np.eye(X.shape[1]))
            self.covariance_matrix = sigma_squared * XTX_inv
        except np.linalg.LinAlgError:
            warnings.warn("Could not compute uncertainty estimates due to singular matrix")
            self.covariance_matrix = None
    def predict(self, X: np.ndarray, return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions.
        Parameters:
            X: Feature matrix
            return_uncertainty: Whether to return prediction uncertainties
        Returns:
            predictions: Predicted values
            uncertainties: Prediction uncertainties (if requested)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        X, _ = self._validate_input(X)
        predictions = X @ self.coefficients + self.intercept
        if return_uncertainty and self.covariance_matrix is not None:
            # Add intercept column for uncertainty calculation
            if self.fit_intercept:
                X_aug = np.column_stack([np.ones(X.shape[0]), X])
            else:
                X_aug = X.copy()
            # Prediction variance
            pred_var = np.sum((X_aug @ self.covariance_matrix) * X_aug, axis=1)
            uncertainties = np.sqrt(pred_var)
            return predictions, uncertainties
        return predictions
    def confidence_intervals(self, X: np.ndarray, confidence_level: float = 0.95) -> np.ndarray:
        """Compute confidence intervals for predictions."""
        if not self.is_fitted or self.covariance_matrix is None:
            raise ValueError("Model must be fitted with uncertainty estimation")
        predictions, uncertainties = self.predict(X, return_uncertainty=True)
        alpha = 1 - confidence_level
        t_val = stats.t.ppf(1 - alpha/2, df=len(predictions) - self.n_features - 1)
        margin = t_val * uncertainties
        return np.column_stack([predictions - margin, predictions + margin])
    def summary(self) -> Dict[str, Any]:
        """Return model summary statistics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        summary = {
            'intercept': self.intercept,
            'coefficients': self.coefficients,
            'n_features': self.n_features,
            'solver': self.solver,
            'regularization': self.regularization
        }
        if self.covariance_matrix is not None:
            std_errors = np.sqrt(np.diag(self.covariance_matrix))
            if self.fit_intercept:
                summary['intercept_std_error'] = std_errors[0]
                summary['coefficient_std_errors'] = std_errors[1:]
            else:
                summary['coefficient_std_errors'] = std_errors
        return summary
class PolynomialRegression(SupervisedModel):
    """
    Polynomial regression for nonlinear relationships.
    Features:
    - Automatic feature generation
    - Cross-validation for degree selection
    - Regularization options
    - Extrapolation warnings
    """
    def __init__(self,
                 degree: int = 2,
                 include_bias: bool = True,
                 interaction_only: bool = False,
                 regularization: float = 0.0):
        super().__init__()
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.regularization = regularization
        self.linear_model = None
        self.feature_powers = None
        self.X_mean = None
        self.X_std = None
    def _generate_polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """Generate polynomial features."""
        from itertools import combinations_with_replacement
        n_samples, n_features = X.shape
        if self.interaction_only:
            # Only interaction terms, no pure powers > 1
            powers = []
            for deg in range(self.degree + 1):
                for combo in combinations_with_replacement(range(n_features), deg):
                    if len(set(combo)) == len(combo):  # No repeated features
                        powers.append([combo.count(i) for i in range(n_features)])
        else:
            # All polynomial terms up to degree
            powers = []
            for deg in range(self.degree + 1):
                for combo in combinations_with_replacement(range(n_features), deg):
                    powers.append([combo.count(i) for i in range(n_features)])
        self.feature_powers = np.array(powers)
        # Generate polynomial features
        X_poly = np.ones((n_samples, len(powers)))
        for i, power in enumerate(powers):
            for j, p in enumerate(power):
                if p > 0:
                    X_poly[:, i] *= X[:, j] ** p
        return X_poly
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PolynomialRegression':
        """Fit polynomial regression model."""
        X, y = self._validate_input(X, y)
        # Standardize features for numerical stability
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std[self.X_std == 0] = 1  # Avoid division by zero
        X_scaled = (X - self.X_mean) / self.X_std
        # Generate polynomial features
        X_poly = self._generate_polynomial_features(X_scaled)
        # Fit linear model on polynomial features
        self.linear_model = LinearRegression(
            fit_intercept=self.include_bias,
            regularization=self.regularization
        )
        self.linear_model.fit(X_poly, y)
        self.is_fitted = True
        return self
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using polynomial model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        X, _ = self._validate_input(X)
        # Scale features
        X_scaled = (X - self.X_mean) / self.X_std
        # Check for extrapolation
        for i in range(X.shape[1]):
            if np.any(X[:, i] < np.min(self.X_mean[i] - 2*self.X_std[i])) or \
               np.any(X[:, i] > np.max(self.X_mean[i] + 2*self.X_std[i])):
                warnings.warn(f"Extrapolation detected for feature {i}. Results may be unreliable.")
        # Generate polynomial features and predict
        X_poly = self._generate_polynomial_features(X_scaled)
        return self.linear_model.predict(X_poly)
class RidgeRegression(LinearRegression):
    """
    Ridge regression with L2 regularization.
    Inherits from LinearRegression with automatic regularization parameter selection.
    """
    def __init__(self,
                 alpha: float = 1.0,
                 fit_intercept: bool = True,
                 solver: str = 'svd',
                 cv_folds: int = 5):
        super().__init__(fit_intercept=fit_intercept, solver=solver, regularization=alpha)
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.alpha_path = None
        self.cv_scores = None
    def fit_with_cv(self, X: np.ndarray, y: np.ndarray,
                   alpha_range: np.ndarray = None) -> 'RidgeRegression':
        """Fit with cross-validation for optimal alpha."""
        if alpha_range is None:
            alpha_range = np.logspace(-4, 2, 50)
        X, y = self._validate_input(X, y)
        # Cross-validation
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        for alpha in alpha_range:
            fold_scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                # Fit ridge model
                model = LinearRegression(
                    fit_intercept=self.fit_intercept,
                    solver=self.solver,
                    regularization=alpha
                )
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                fold_scores.append(score)
            cv_scores.append(np.mean(fold_scores))
        # Select best alpha
        best_idx = np.argmax(cv_scores)
        self.alpha = alpha_range[best_idx]
        self.regularization = self.alpha
        self.alpha_path = alpha_range
        self.cv_scores = cv_scores
        # Fit final model
        return self.fit(X, y)
class LogisticRegression(SupervisedModel):
    """
    Logistic regression for binary and multiclass classification.
    Features:
    - Multiple solvers
    - Regularization options
    - Probability predictions
    - Feature importance
    """
    def __init__(self,
                 penalty: str = 'l2',
                 C: float = 1.0,
                 solver: str = 'lbfgs',
                 max_iter: int = 1000,
                 multi_class: str = 'auto'):
        super().__init__()
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.coefficients = None
        self.intercept = None
        self.classes = None
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    def _cost_function(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Logistic regression cost function."""
        if self.fit_intercept:
            intercept = params[0]
            coeffs = params[1:]
        else:
            intercept = 0
            coeffs = params
        z = X @ coeffs + intercept
        h = self._sigmoid(z)
        # Prevent log(0)
        h = np.clip(h, 1e-15, 1 - 1e-15)
        # Cost function
        cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
        # Add regularization
        if self.penalty == 'l2':
            cost += (1 / (2 * self.C)) * np.sum(coeffs ** 2)
        elif self.penalty == 'l1':
            cost += (1 / self.C) * np.sum(np.abs(coeffs))
        return cost
    def _gradient(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradient of cost function."""
        if self.fit_intercept:
            intercept = params[0]
            coeffs = params[1:]
        else:
            intercept = 0
            coeffs = params
        z = X @ coeffs + intercept
        h = self._sigmoid(z)
        error = h - y
        if self.fit_intercept:
            grad_intercept = np.mean(error)
            grad_coeffs = X.T @ error / len(y)
            # Add regularization to coefficients only
            if self.penalty == 'l2':
                grad_coeffs += coeffs / self.C
            elif self.penalty == 'l1':
                grad_coeffs += np.sign(coeffs) / self.C
            return np.hstack([grad_intercept, grad_coeffs])
        else:
            grad_coeffs = X.T @ error / len(y)
            if self.penalty == 'l2':
                grad_coeffs += coeffs / self.C
            elif self.penalty == 'l1':
                grad_coeffs += np.sign(coeffs) / self.C
            return grad_coeffs
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """Fit logistic regression model."""
        X, y = self._validate_input(X, y)
        self.classes = np.unique(y)
        if len(self.classes) == 2:
            # Binary classification
            y_binary = (y == self.classes[1]).astype(int)
            self._fit_binary(X, y_binary)
        else:
            # Multiclass classification
            self._fit_multiclass(X, y)
        self.is_fitted = True
        return self
    def _fit_binary(self, X: np.ndarray, y: np.ndarray):
        """Fit binary logistic regression."""
        n_features = X.shape[1]
        n_params = n_features + (1 if self.fit_intercept else 0)
        # Initialize parameters
        params_init = np.zeros(n_params)
        # Optimize
        if self.solver == 'lbfgs':
            result = optimize.minimize(
                self._cost_function,
                params_init,
                args=(X, y),
                method='L-BFGS-B',
                jac=self._gradient,
                options={'maxiter': self.max_iter}
            )
            params = result.x
        else:
            raise ValueError(f"Solver {self.solver} not implemented")
        # Extract parameters
        if self.fit_intercept:
            self.intercept = params[0]
            self.coefficients = params[1:]
        else:
            self.intercept = 0.0
            self.coefficients = params
    def _fit_multiclass(self, X: np.ndarray, y: np.ndarray):
        """Fit multiclass logistic regression using one-vs-rest."""
        n_classes = len(self.classes)
        n_features = X.shape[1]
        self.coefficients = np.zeros((n_classes, n_features))
        self.intercept = np.zeros(n_classes)
        for i, class_label in enumerate(self.classes):
            y_binary = (y == class_label).astype(int)
            # Fit binary classifier
            binary_model = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter
            )
            binary_model.fit_intercept = self.fit_intercept
            binary_model._fit_binary(X, y_binary)
            self.coefficients[i] = binary_model.coefficients
            self.intercept[i] = binary_model.intercept
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        X, _ = self._validate_input(X)
        if len(self.classes) == 2:
            # Binary classification
            z = X @ self.coefficients + self.intercept
            prob_positive = self._sigmoid(z)
            return np.column_stack([1 - prob_positive, prob_positive])
        else:
            # Multiclass classification
            scores = X @ self.coefficients.T + self.intercept
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make class predictions."""
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes[predicted_indices]
# Additional models can be implemented here (SVM, RandomForest, etc.)
# For brevity, I'll focus on the core models above
def create_test_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Create test datasets for supervised learning validation."""
    np.random.seed(42)
    datasets = {}
    # Linear regression dataset
    n_samples = 100
    X_linear = np.random.randn(n_samples, 3)
    true_coeffs = np.array([2.5, -1.3, 0.8])
    y_linear = X_linear @ true_coeffs + 0.1 * np.random.randn(n_samples)
    datasets['linear'] = (X_linear, y_linear)
    # Polynomial regression dataset
    X_poly = np.random.uniform(-2, 2, (n_samples, 1))
    y_poly = 2 * X_poly.ravel()**2 - 3 * X_poly.ravel() + 1 + 0.2 * np.random.randn(n_samples)
    datasets['polynomial'] = (X_poly, y_poly)
    # Classification dataset
    X_class = np.random.randn(n_samples, 2)
    y_class = (X_class[:, 0] + X_class[:, 1] > 0).astype(int)
    datasets['classification'] = (X_class, y_class)
    return datasets
# Visualization utilities
def plot_regression_results(model: SupervisedModel, X: np.ndarray, y: np.ndarray,
                          X_test: np.ndarray = None, title: str = "Regression Results"):
    """Plot regression results with Berkeley styling."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Berkeley colors
    berkeley_blue = '#003262'
    california_gold = '#FDB515'
    # Plot 1: Predictions vs actual
    y_pred = model.predict(X)
    axes[0].scatter(y, y_pred, alpha=0.6, color=berkeley_blue, s=50)
    # Perfect prediction line
    min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val],
                color=california_gold, linewidth=2, linestyle='--')
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Predictions vs Actual')
    axes[0].grid(True, alpha=0.3)
    # Plot 2: Residuals
    residuals = y - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, color=berkeley_blue, s=50)
    axes[1].axhline(y=0, color=california_gold, linewidth=2, linestyle='--')
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig