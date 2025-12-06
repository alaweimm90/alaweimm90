"""
Machine Learning Utilities for Scientific Computing
This module provides utility functions and classes for machine learning
workflows in scientific computing, including data processing, model evaluation,
cross-validation, feature selection, and visualization tools.
Classes:
    DataProcessor: Data preprocessing and transformation utilities
    ModelEvaluator: Comprehensive model evaluation and diagnostics
    CrossValidator: Advanced cross-validation schemes
    FeatureSelector: Feature selection and importance analysis
    Visualizer: Scientific visualization tools for ML
"""
import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any, Union, List, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy import stats, sparse
from scipy.linalg import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import logging
logger = logging.getLogger(__name__)
@dataclass
class EvaluationResults:
    """Container for model evaluation results."""
    # Regression metrics
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    adjusted_r2: Optional[float] = None
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    # Cross-validation results
    cv_scores: Optional[np.ndarray] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    # Additional diagnostics
    residuals: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    feature_importance: Optional[np.ndarray] = None
class DataProcessor:
    """
    Comprehensive data preprocessing for scientific computing.
    Features:
    - Missing value handling
    - Outlier detection and treatment
    - Feature scaling and transformation
    - Dimensionality reduction
    - Time series preprocessing
    """
    def __init__(self):
        self.scalers = {}
        self.transformers = {}
        self.outlier_detectors = {}
        self.is_fitted = False
    def handle_missing_values(self,
                            X: np.ndarray,
                            strategy: str = 'mean',
                            fill_value: Optional[float] = None) -> np.ndarray:
        """
        Handle missing values in dataset.
        Parameters:
            X: Input data
            strategy: Strategy for imputation ('mean', 'median', 'mode', 'constant', 'interpolate')
            fill_value: Value to use for constant strategy
        Returns:
            Data with imputed values
        """
        X = np.asarray(X)
        if not np.any(np.isnan(X)):
            return X
        X_imputed = X.copy()
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if not np.any(mask):
                continue
            if strategy == 'mean':
                fill_val = np.nanmean(X[:, col])
            elif strategy == 'median':
                fill_val = np.nanmedian(X[:, col])
            elif strategy == 'mode':
                values, counts = np.unique(X[~mask, col], return_counts=True)
                fill_val = values[np.argmax(counts)]
            elif strategy == 'constant':
                fill_val = fill_value if fill_value is not None else 0
            elif strategy == 'interpolate':
                # Linear interpolation
                valid_indices = np.where(~mask)[0]
                if len(valid_indices) > 1:
                    X_imputed[mask, col] = np.interp(
                        np.where(mask)[0], valid_indices, X[valid_indices, col]
                    )
                else:
                    fill_val = np.nanmean(X[:, col])
                    X_imputed[mask, col] = fill_val
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            X_imputed[mask, col] = fill_val
        return X_imputed
    def detect_outliers(self,
                       X: np.ndarray,
                       method: str = 'iqr',
                       threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers in dataset.
        Parameters:
            X: Input data
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection
        Returns:
            Boolean mask indicating outliers
        """
        X = np.asarray(X)
        outlier_mask = np.zeros(X.shape[0], dtype=bool)
        if method == 'iqr':
            for col in range(X.shape[1]):
                Q1 = np.percentile(X[:, col], 25)
                Q3 = np.percentile(X[:, col], 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask |= (X[:, col] < lower_bound) | (X[:, col] > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(X, axis=0, nan_policy='omit'))
            outlier_mask = np.any(z_scores > threshold, axis=1)
        elif method == 'isolation_forest':
            # Simplified isolation forest implementation
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            outlier_mask = outlier_labels == -1
        else:
            raise ValueError(f"Unknown method: {method}")
        return outlier_mask
    def scale_features(self,
                      X: np.ndarray,
                      method: str = 'standard',
                      feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        Scale features using various methods.
        Parameters:
            X: Input data
            method: Scaling method ('standard', 'minmax', 'robust', 'unit_vector')
            feature_range: Range for minmax scaling
        Returns:
            Scaled data
        """
        X = np.asarray(X)
        if method == 'standard':
            # Z-score normalization
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            X_scaled = (X - mean) / std
            self.scalers['standard'] = {'mean': mean, 'std': std}
        elif method == 'minmax':
            # Min-max scaling
            min_val = np.min(X, axis=0)
            max_val = np.max(X, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1  # Avoid division by zero
            X_scaled = (X - min_val) / range_val
            X_scaled = X_scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
            self.scalers['minmax'] = {
                'min': min_val, 'max': max_val, 'feature_range': feature_range
            }
        elif method == 'robust':
            # Robust scaling using median and IQR
            median = np.median(X, axis=0)
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            IQR[IQR == 0] = 1  # Avoid division by zero
            X_scaled = (X - median) / IQR
            self.scalers['robust'] = {'median': median, 'iqr': IQR}
        elif method == 'unit_vector':
            # Unit vector scaling
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            X_scaled = X / norms
        else:
            raise ValueError(f"Unknown method: {method}")
        return X_scaled
    def apply_transformations(self,
                            X: np.ndarray,
                            transformations: List[str]) -> np.ndarray:
        """
        Apply feature transformations.
        Parameters:
            X: Input data
            transformations: List of transformations to apply
        Returns:
            Transformed data
        """
        X_transformed = X.copy()
        for transform in transformations:
            if transform == 'log':
                # Log transformation (add small constant for zero values)
                X_transformed = np.log(np.abs(X_transformed) + 1e-8)
            elif transform == 'sqrt':
                # Square root transformation
                X_transformed = np.sqrt(np.abs(X_transformed))
            elif transform == 'square':
                # Square transformation
                X_transformed = X_transformed ** 2
            elif transform == 'reciprocal':
                # Reciprocal transformation
                X_transformed = 1 / (X_transformed + 1e-8)
            elif transform == 'box_cox':
                # Box-Cox transformation
                X_transformed = self._box_cox_transform(X_transformed)
            else:
                warnings.warn(f"Unknown transformation: {transform}")
        return X_transformed
    def _box_cox_transform(self, X: np.ndarray, lambda_val: float = 0.0) -> np.ndarray:
        """Apply Box-Cox transformation."""
        if lambda_val == 0:
            return np.log(X + 1e-8)
        else:
            return (np.power(X + 1e-8, lambda_val) - 1) / lambda_val
class ModelEvaluator:
    """
    Comprehensive model evaluation and diagnostics.
    Features:
    - Multiple evaluation metrics
    - Statistical significance testing
    - Residual analysis
    - Model comparison
    - Bootstrap confidence intervals
    """
    def __init__(self):
        self.evaluation_history = []
    def evaluate_regression(self,
                          model,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          compute_intervals: bool = True) -> EvaluationResults:
        """
        Comprehensive regression model evaluation.
        Parameters:
            model: Trained regression model
            X_test: Test features
            y_test: Test targets
            compute_intervals: Whether to compute confidence intervals
        Returns:
            Evaluation results
        """
        # Make predictions
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        # Basic metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # Adjusted R²
        n = len(y_test)
        p = X_test.shape[1] if hasattr(X_test, 'shape') else 1
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        # Feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_)
        results = EvaluationResults(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2,
            adjusted_r2=adjusted_r2,
            residuals=residuals,
            predictions=y_pred,
            feature_importance=feature_importance
        )
        self.evaluation_history.append(results)
        return results
    def evaluate_classification(self,
                              model,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              average: str = 'weighted') -> EvaluationResults:
        """
        Comprehensive classification model evaluation.
        Parameters:
            model: Trained classification model
            X_test: Test features
            y_test: Test targets
            average: Averaging strategy for multiclass metrics
        Returns:
            Evaluation results
        """
        # Make predictions
        y_pred = model.predict(X_test)
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average, zero_division=0)
        recall = recall_score(y_test, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # Feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        results = EvaluationResults(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix=cm,
            predictions=y_pred,
            feature_importance=feature_importance
        )
        self.evaluation_history.append(results)
        return results
    def statistical_tests(self,
                         y_true: np.ndarray,
                         y_pred1: np.ndarray,
                         y_pred2: np.ndarray) -> Dict[str, float]:
        """
        Statistical significance tests for model comparison.
        Parameters:
            y_true: True values
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
        Returns:
            Dictionary of test statistics and p-values
        """
        # Compute errors
        errors1 = y_true - y_pred1
        errors2 = y_true - y_pred2
        # Paired t-test for difference in errors
        t_stat, t_pvalue = stats.ttest_rel(np.abs(errors1), np.abs(errors2))
        # Wilcoxon signed-rank test (non-parametric alternative)
        w_stat, w_pvalue = stats.wilcoxon(np.abs(errors1), np.abs(errors2),
                                         alternative='two-sided')
        # F-test for variance comparison
        var1, var2 = np.var(errors1), np.var(errors2)
        f_stat = var1 / var2 if var2 != 0 else np.inf
        f_pvalue = 2 * min(stats.f.cdf(f_stat, len(errors1)-1, len(errors2)-1),
                          1 - stats.f.cdf(f_stat, len(errors1)-1, len(errors2)-1))
        return {
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_pvalue': w_pvalue,
            'f_statistic': f_stat,
            'f_pvalue': f_pvalue
        }
    def bootstrap_confidence_intervals(self,
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     metric: str = 'r2',
                                     n_bootstrap: int = 1000,
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Compute bootstrap confidence intervals for metrics.
        Parameters:
            y_true: True values
            y_pred: Predicted values
            metric: Metric to compute ('r2', 'mse', 'mae')
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level
        Returns:
            Lower and upper confidence bounds
        """
        n_samples = len(y_true)
        bootstrap_scores = []
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            # Compute metric
            if metric == 'r2':
                score = r2_score(y_true_boot, y_pred_boot)
            elif metric == 'mse':
                score = mean_squared_error(y_true_boot, y_pred_boot)
            elif metric == 'mae':
                score = mean_absolute_error(y_true_boot, y_pred_boot)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            bootstrap_scores.append(score)
        # Compute confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        lower_bound = np.percentile(bootstrap_scores, lower_percentile)
        upper_bound = np.percentile(bootstrap_scores, upper_percentile)
        return lower_bound, upper_bound
class CrossValidator:
    """
    Advanced cross-validation schemes for scientific computing.
    Features:
    - Multiple CV strategies
    - Nested cross-validation
    - Time series aware splitting
    - Stratified sampling
    - Custom splitting functions
    """
    def __init__(self,
                 cv_type: str = 'kfold',
                 n_splits: int = 5,
                 test_size: float = 0.2,
                 random_state: Optional[int] = None):
        self.cv_type = cv_type
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
    def cross_validate(self,
                      model,
                      X: np.ndarray,
                      y: np.ndarray,
                      scoring: Union[str, List[str]] = 'neg_mean_squared_error',
                      return_train_score: bool = True) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation with specified strategy.
        Parameters:
            model: Model to validate
            X: Features
            y: Targets
            scoring: Scoring metric(s)
            return_train_score: Whether to return training scores
        Returns:
            Cross-validation results
        """
        # Get cross-validation splitter
        cv_splitter = self._get_cv_splitter(X, y)
        # Initialize results
        if isinstance(scoring, str):
            scoring = [scoring]
        results = {}
        for score_name in scoring:
            results[f'test_{score_name}'] = []
            if return_train_score:
                results[f'train_{score_name}'] = []
        # Perform cross-validation
        for train_idx, test_idx in cv_splitter.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            # Fit model
            model_copy = self._clone_model(model)
            model_copy.fit(X_train, y_train)
            # Evaluate on test and train sets
            for score_name in scoring:
                test_score = self._compute_score(model_copy, X_test, y_test, score_name)
                results[f'test_{score_name}'].append(test_score)
                if return_train_score:
                    train_score = self._compute_score(model_copy, X_train, y_train, score_name)
                    results[f'train_{score_name}'].append(train_score)
        # Convert to numpy arrays
        for key in results:
            results[key] = np.array(results[key])
        return results
    def nested_cross_validation(self,
                              model,
                              X: np.ndarray,
                              y: np.ndarray,
                              param_grid: Dict[str, List],
                              inner_cv: int = 3,
                              scoring: str = 'neg_mean_squared_error') -> Dict[str, Any]:
        """
        Perform nested cross-validation for unbiased performance estimation.
        Parameters:
            model: Model to validate
            X: Features
            y: Targets
            param_grid: Parameter grid for hyperparameter tuning
            inner_cv: Number of inner CV folds
            scoring: Scoring metric
        Returns:
            Nested CV results
        """
        from sklearn.model_selection import GridSearchCV
        outer_cv = self._get_cv_splitter(X, y)
        outer_scores = []
        best_params_list = []
        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            # Inner loop: hyperparameter tuning
            inner_cv_splitter = KFold(n_splits=inner_cv, shuffle=True,
                                    random_state=self.random_state)
            grid_search = GridSearchCV(
                model, param_grid, cv=inner_cv_splitter,
                scoring=scoring, n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            # Outer loop: performance estimation
            best_model = grid_search.best_estimator_
            outer_score = self._compute_score(best_model, X_test, y_test, scoring)
            outer_scores.append(outer_score)
            best_params_list.append(grid_search.best_params_)
        return {
            'outer_scores': np.array(outer_scores),
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
            'best_params': best_params_list
        }
    def _get_cv_splitter(self, X: np.ndarray, y: np.ndarray):
        """Get appropriate cross-validation splitter."""
        if self.cv_type == 'kfold':
            return KFold(n_splits=self.n_splits, shuffle=True,
                        random_state=self.random_state)
        elif self.cv_type == 'stratified':
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                 random_state=self.random_state)
        elif self.cv_type == 'timeseries':
            return TimeSeriesSplit(n_splits=self.n_splits)
        else:
            raise ValueError(f"Unknown CV type: {self.cv_type}")
    def _clone_model(self, model):
        """Create a copy of the model."""
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            # Fallback for custom models
            import copy
            return copy.deepcopy(model)
    def _compute_score(self, model, X: np.ndarray, y: np.ndarray, scoring: str) -> float:
        """Compute score for given metric."""
        y_pred = model.predict(X)
        if scoring == 'neg_mean_squared_error':
            return -mean_squared_error(y, y_pred)
        elif scoring == 'neg_mean_absolute_error':
            return -mean_absolute_error(y, y_pred)
        elif scoring == 'r2':
            return r2_score(y, y_pred)
        elif scoring == 'accuracy':
            return accuracy_score(y, y_pred)
        elif scoring == 'f1':
            return f1_score(y, y_pred, average='weighted')
        else:
            raise ValueError(f"Unknown scoring metric: {scoring}")
class FeatureSelector:
    """
    Feature selection and importance analysis.
    Features:
    - Multiple selection methods
    - Recursive feature elimination
    - Stability selection
    - Permutation importance
    - Correlation analysis
    """
    def __init__(self):
        self.selected_features = None
        self.feature_scores = None
        self.feature_names = None
    def select_features(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       method: str = 'univariate',
                       k: int = 10,
                       model=None) -> np.ndarray:
        """
        Select features using specified method.
        Parameters:
            X: Feature matrix
            y: Target vector
            method: Selection method
            k: Number of features to select
            model: Model for model-based selection
        Returns:
            Indices of selected features
        """
        if method == 'univariate':
            return self._univariate_selection(X, y, k)
        elif method == 'mutual_info':
            return self._mutual_info_selection(X, y, k)
        elif method == 'rfe':
            if model is None:
                raise ValueError("Model required for RFE")
            return self._recursive_feature_elimination(X, y, model, k)
        elif method == 'permutation':
            if model is None:
                raise ValueError("Model required for permutation importance")
            return self._permutation_importance_selection(X, y, model, k)
        elif method == 'correlation':
            return self._correlation_selection(X, y, k)
        else:
            raise ValueError(f"Unknown method: {method}")
    def _univariate_selection(self, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """Univariate feature selection using F-statistics."""
        from sklearn.feature_selection import f_regression, f_classif
        # Determine if regression or classification
        if len(np.unique(y)) > 10:  # Assume regression
            scores, p_values = f_regression(X, y)
        else:  # Assume classification
            scores, p_values = f_classif(X, y)
        # Select top k features
        top_indices = np.argsort(scores)[-k:]
        self.feature_scores = scores
        self.selected_features = top_indices
        return top_indices
    def _mutual_info_selection(self, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """Mutual information based feature selection."""
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        # Determine if regression or classification
        if len(np.unique(y)) > 10:  # Assume regression
            scores = mutual_info_regression(X, y)
        else:  # Assume classification
            scores = mutual_info_classif(X, y)
        # Select top k features
        top_indices = np.argsort(scores)[-k:]
        self.feature_scores = scores
        self.selected_features = top_indices
        return top_indices
    def _recursive_feature_elimination(self, X: np.ndarray, y: np.ndarray,
                                     model, k: int) -> np.ndarray:
        """Recursive feature elimination."""
        from sklearn.feature_selection import RFE
        rfe = RFE(estimator=model, n_features_to_select=k)
        rfe.fit(X, y)
        selected_indices = np.where(rfe.support_)[0]
        self.feature_scores = rfe.ranking_
        self.selected_features = selected_indices
        return selected_indices
    def _permutation_importance_selection(self, X: np.ndarray, y: np.ndarray,
                                        model, k: int) -> np.ndarray:
        """Feature selection based on permutation importance."""
        from sklearn.inspection import permutation_importance
        # Fit model
        model.fit(X, y)
        # Compute permutation importance
        perm_importance = permutation_importance(model, X, y, n_repeats=10)
        scores = perm_importance.importances_mean
        # Select top k features
        top_indices = np.argsort(scores)[-k:]
        self.feature_scores = scores
        self.selected_features = top_indices
        return top_indices
    def _correlation_selection(self, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """Feature selection based on correlation with target."""
        correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        # Handle NaN correlations
        correlations = np.nan_to_num(correlations)
        # Select top k features
        top_indices = np.argsort(correlations)[-k:]
        self.feature_scores = correlations
        self.selected_features = top_indices
        return top_indices
class Visualizer:
    """
    Scientific visualization tools for machine learning.
    Features:
    - Berkeley-themed plotting
    - Model diagnostics plots
    - Feature importance visualization
    - Learning curves
    - Residual analysis
    """
    def __init__(self):
        self.berkeley_blue = '#003262'
        self.california_gold = '#FDB515'
        self.colors = [self.berkeley_blue, self.california_gold, '#859438', '#00B2A9']
    def plot_learning_curves(self,
                            model,
                            X: np.ndarray,
                            y: np.ndarray,
                            train_sizes: np.ndarray = None,
                            cv: int = 5,
                            title: str = "Learning Curves") -> plt.Figure:
        """Plot learning curves to diagnose bias/variance."""
        from sklearn.model_selection import learning_curve
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes, n_jobs=-1
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        ax.plot(train_sizes, train_mean, 'o-', color=self.berkeley_blue,
                linewidth=2, label='Training Score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                       alpha=0.3, color=self.berkeley_blue)
        ax.plot(train_sizes, val_mean, 'o-', color=self.california_gold,
                linewidth=2, label='Validation Score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                       alpha=0.3, color=self.california_gold)
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig
    def plot_feature_importance(self,
                              feature_scores: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              title: str = "Feature Importance") -> plt.Figure:
        """Plot feature importance scores."""
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(feature_scores))]
        # Sort by importance
        sorted_indices = np.argsort(feature_scores)
        sorted_scores = feature_scores[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_scores) * 0.3)))
        y_pos = np.arange(len(sorted_scores))
        bars = ax.barh(y_pos, sorted_scores, color=self.berkeley_blue, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax.text(score + 0.01 * max(sorted_scores), bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', va='center', ha='left', fontsize=9)
        plt.tight_layout()
        return fig
    def plot_residuals(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      title: str = "Residual Analysis") -> plt.Figure:
        """Plot residual analysis for regression models."""
        residuals = y_true - y_pred
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, color=self.berkeley_blue, s=30)
        axes[0, 0].axhline(y=0, color=self.california_gold, linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].get_lines()[0].set_markerfacecolor(self.berkeley_blue)
        axes[0, 1].get_lines()[1].set_color(self.california_gold)
        axes[0, 1].set_title('Q-Q Plot')
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color=self.berkeley_blue,
                       density=True, edgecolor='black')
        # Overlay normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        y = stats.norm.pdf(x, mu, sigma)
        axes[1, 0].plot(x, y, color=self.california_gold, linewidth=2, label='Normal Distribution')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        # Scale-Location plot
        sqrt_abs_residuals = np.sqrt(np.abs(residuals))
        axes[1, 1].scatter(y_pred, sqrt_abs_residuals, alpha=0.6,
                          color=self.berkeley_blue, s=30)
        # Add trend line
        z = np.polyfit(y_pred, sqrt_abs_residuals, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(sorted(y_pred), p(sorted(y_pred)),
                       color=self.california_gold, linewidth=2)
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('√|Residuals|')
        axes[1, 1].set_title('Scale-Location Plot')
        axes[1, 1].grid(True, alpha=0.3)
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    def plot_confusion_matrix(self,
                            cm: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            title: str = "Confusion Matrix") -> plt.Figure:
        """Plot confusion matrix with Berkeley styling."""
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title=title,
               ylabel='True Label',
               xlabel='Predicted Label')
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        return fig
# Utility functions
def create_synthetic_dataset(dataset_type: str = 'regression',
                           n_samples: int = 1000,
                           n_features: int = 10,
                           noise: float = 0.1,
                           random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic datasets for testing."""
    np.random.seed(random_state)
    if dataset_type == 'regression':
        X = np.random.randn(n_samples, n_features)
        true_coeffs = np.random.randn(n_features)
        y = X @ true_coeffs + noise * np.random.randn(n_samples)
    elif dataset_type == 'classification':
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                 n_informative=n_features//2, n_redundant=0,
                                 random_state=random_state)
    elif dataset_type == 'clustering':
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n_samples, centers=3, n_features=n_features,
                         random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    return X, y
def calculate_model_complexity(model) -> Dict[str, Any]:
    """Calculate model complexity metrics."""
    complexity = {}
    # Number of parameters
    n_params = 0
    if hasattr(model, 'coef_'):
        n_params += np.prod(model.coef_.shape)
    if hasattr(model, 'intercept_'):
        n_params += np.prod(model.intercept_.shape) if hasattr(model.intercept_, 'shape') else 1
    complexity['n_parameters'] = n_params
    # Model type
    complexity['model_type'] = type(model).__name__
    # Memory usage (approximate)
    import sys
    complexity['memory_usage_bytes'] = sys.getsizeof(model)
    return complexity