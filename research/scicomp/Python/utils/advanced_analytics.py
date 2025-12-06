"""
Advanced analytics and machine learning utilities for SciComp.
This module provides sophisticated analytics including:
- Automated machine learning pipelines
- Statistical analysis and hypothesis testing
- Time series analysis and forecasting
- Dimensionality reduction techniques
- Anomaly detection algorithms
"""
import numpy as np
import warnings
from typing import Union, List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
# Optional ML libraries
try:
    import sklearn
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA, ICA, FastICA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. ML features will be limited.")
try:
    from scipy import stats
    from scipy.signal import find_peaks, periodogram
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Advanced analytics limited.")
class AnalysisType(Enum):
    """Types of analyses available."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
@dataclass
class AnalysisResult:
    """Container for analysis results."""
    analysis_type: AnalysisType
    model: Any
    metrics: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    transformed_data: Optional[np.ndarray] = None
    feature_importance: Optional[np.ndarray] = None
    parameters: Optional[Dict[str, Any]] = None
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'analysis_type': self.analysis_type.value,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'parameters': self.parameters
        }
    def save(self, filename: str):
        """Save analysis result to file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
class AdvancedAnalytics:
    """Main advanced analytics class."""
    def __init__(self):
        """Initialize advanced analytics engine."""
        self.models = {}
        self.scalers = {}
        self.results_history = []
    def auto_analyze(self, data: np.ndarray, target: Optional[np.ndarray] = None,
                    analysis_type: Optional[AnalysisType] = None) -> AnalysisResult:
        """
        Automatically determine and perform the best analysis.
        Args:
            data: Input data
            target: Target values (for supervised learning)
            analysis_type: Force specific analysis type
        Returns:
            Analysis result
        """
        if analysis_type is None:
            analysis_type = self._determine_analysis_type(data, target)
        print(f"Performing {analysis_type.value} analysis...")
        if analysis_type == AnalysisType.CLASSIFICATION:
            return self.classify(data, target)
        elif analysis_type == AnalysisType.REGRESSION:
            return self.regress(data, target)
        elif analysis_type == AnalysisType.CLUSTERING:
            return self.cluster(data)
        elif analysis_type == AnalysisType.ANOMALY_DETECTION:
            return self.detect_anomalies(data)
        elif analysis_type == AnalysisType.TIME_SERIES:
            return self.analyze_time_series(data)
        elif analysis_type == AnalysisType.DIMENSIONALITY_REDUCTION:
            return self.reduce_dimensions(data)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
    def _determine_analysis_type(self, data: np.ndarray,
                                target: Optional[np.ndarray]) -> AnalysisType:
        """Automatically determine the best analysis type."""
        if target is not None:
            # Supervised learning
            unique_targets = len(np.unique(target))
            if unique_targets < 10:  # Likely classification
                return AnalysisType.CLASSIFICATION
            else:  # Likely regression
                return AnalysisType.REGRESSION
        else:
            # Unsupervised learning
            if data.shape[1] > 10:  # High dimensional
                return AnalysisType.DIMENSIONALITY_REDUCTION
            elif len(data) > 1000:  # Large dataset
                return AnalysisType.CLUSTERING
            else:
                return AnalysisType.ANOMALY_DETECTION
    def classify(self, X: np.ndarray, y: np.ndarray,
                test_size: float = 0.2) -> AnalysisResult:
        """
        Perform classification analysis.
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Test set proportion
        Returns:
            Classification results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn required for classification")
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Try multiple classifiers
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        classifiers = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        best_model = None
        best_score = 0
        results = {}
        for name, classifier in classifiers.items():
            try:
                classifier.fit(X_train_scaled, y_train)
                predictions = classifier.predict(X_test_scaled)
                score = accuracy_score(y_test, predictions)
                results[name] = score
                if score > best_score:
                    best_score = score
                    best_model = classifier
            except Exception as e:
                warnings.warn(f"Classifier {name} failed: {e}")
        # Get predictions and feature importance
        predictions = best_model.predict(X_test_scaled)
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
        # Store model and scaler
        self.models['classification'] = best_model
        self.scalers['classification'] = scaler
        result = AnalysisResult(
            analysis_type=AnalysisType.CLASSIFICATION,
            model=best_model,
            metrics={'accuracy': best_score, 'all_scores': results},
            predictions=predictions,
            feature_importance=feature_importance
        )
        self.results_history.append(result)
        return result
    def regress(self, X: np.ndarray, y: np.ndarray,
               test_size: float = 0.2) -> AnalysisResult:
        """
        Perform regression analysis.
        Args:
            X: Feature matrix
            y: Target values
            test_size: Test set proportion
        Returns:
            Regression results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn required for regression")
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Try multiple regressors
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.svm import SVR
        regressors = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'Ridge': Ridge(alpha=1.0),
            'SVR': SVR(kernel='rbf')
        }
        best_model = None
        best_score = -float('inf')
        results = {}
        for name, regressor in regressors.items():
            try:
                regressor.fit(X_train_scaled, y_train)
                predictions = regressor.predict(X_test_scaled)
                score = r2_score(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                results[name] = {'r2': score, 'mse': mse}
                if score > best_score:
                    best_score = score
                    best_model = regressor
            except Exception as e:
                warnings.warn(f"Regressor {name} failed: {e}")
        # Get predictions and feature importance
        predictions = best_model.predict(X_test_scaled)
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
        # Store model and scaler
        self.models['regression'] = best_model
        self.scalers['regression'] = scaler
        result = AnalysisResult(
            analysis_type=AnalysisType.REGRESSION,
            model=best_model,
            metrics={'r2_score': best_score, 'all_scores': results},
            predictions=predictions,
            feature_importance=feature_importance
        )
        self.results_history.append(result)
        return result
    def cluster(self, X: np.ndarray, n_clusters: Optional[int] = None) -> AnalysisResult:
        """
        Perform clustering analysis.
        Args:
            X: Data to cluster
            n_clusters: Number of clusters (auto-determined if None)
        Returns:
            Clustering results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn required for clustering")
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X_scaled)
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        # Calculate clustering metrics
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        metrics = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_score(X_scaled, cluster_labels),
            'calinski_harabasz_score': calinski_harabasz_score(X_scaled, cluster_labels),
            'inertia': kmeans.inertia_
        }
        # Store model and scaler
        self.models['clustering'] = kmeans
        self.scalers['clustering'] = scaler
        result = AnalysisResult(
            analysis_type=AnalysisType.CLUSTERING,
            model=kmeans,
            metrics=metrics,
            predictions=cluster_labels,
            transformed_data=X_scaled
        )
        self.results_history.append(result)
        return result
    def _find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        inertias = []
        K_range = range(2, min(max_clusters + 1, len(X)))
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        # Find elbow point
        if len(inertias) < 2:
            return 2
        # Simple elbow detection
        diffs = np.diff(inertias)
        diff_ratios = diffs[:-1] / diffs[1:]
        optimal_idx = np.argmax(diff_ratios)
        return K_range[optimal_idx]
    def detect_anomalies(self, X: np.ndarray, contamination: float = 0.1) -> AnalysisResult:
        """
        Detect anomalies in data.
        Args:
            X: Data to analyze for anomalies
            contamination: Expected proportion of anomalies
        Returns:
            Anomaly detection results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn required for anomaly detection")
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Apply Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        # Convert to boolean array (True for anomalies)
        is_anomaly = anomaly_labels == -1
        metrics = {
            'n_anomalies': np.sum(is_anomaly),
            'anomaly_rate': np.mean(is_anomaly),
            'contamination': contamination
        }
        # Store model and scaler
        self.models['anomaly_detection'] = iso_forest
        self.scalers['anomaly_detection'] = scaler
        result = AnalysisResult(
            analysis_type=AnalysisType.ANOMALY_DETECTION,
            model=iso_forest,
            metrics=metrics,
            predictions=is_anomaly,
            transformed_data=X_scaled
        )
        self.results_history.append(result)
        return result
    def reduce_dimensions(self, X: np.ndarray, n_components: Optional[int] = None,
                         method: str = 'pca') -> AnalysisResult:
        """
        Perform dimensionality reduction.
        Args:
            X: High-dimensional data
            n_components: Number of components (auto-determined if None)
            method: Reduction method ('pca', 'ica')
        Returns:
            Dimensionality reduction results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn required for dimensionality reduction")
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Determine number of components
        if n_components is None:
            # Use 95% of variance for PCA
            if method.lower() == 'pca':
                pca_temp = PCA()
                pca_temp.fit(X_scaled)
                cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components = np.argmax(cumsum >= 0.95) + 1
            else:
                n_components = min(X.shape[1] // 2, 10)
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
        elif method.lower() == 'ica':
            reducer = FastICA(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        X_reduced = reducer.fit_transform(X_scaled)
        # Calculate metrics
        metrics = {'n_components': n_components, 'method': method}
        if method.lower() == 'pca':
            metrics['explained_variance_ratio'] = reducer.explained_variance_ratio_.tolist()
            metrics['total_explained_variance'] = float(np.sum(reducer.explained_variance_ratio_))
        # Store model and scaler
        self.models['dimensionality_reduction'] = reducer
        self.scalers['dimensionality_reduction'] = scaler
        result = AnalysisResult(
            analysis_type=AnalysisType.DIMENSIONALITY_REDUCTION,
            model=reducer,
            metrics=metrics,
            transformed_data=X_reduced
        )
        self.results_history.append(result)
        return result
    def analyze_time_series(self, data: np.ndarray,
                           forecast_steps: int = 10) -> AnalysisResult:
        """
        Analyze time series data.
        Args:
            data: Time series data
            forecast_steps: Number of steps to forecast
        Returns:
            Time series analysis results
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for time series analysis")
        # Basic time series statistics
        trend = np.polyfit(range(len(data)), data, 1)[0]
        # Detect seasonality using FFT
        if len(data) > 10:
            freqs, power = periodogram(data)
            dominant_freq_idx = np.argmax(power[1:]) + 1  # Skip DC component
            period = len(data) / freqs[dominant_freq_idx] if freqs[dominant_freq_idx] > 0 else None
        else:
            period = None
        # Simple forecasting using linear trend
        if forecast_steps > 0:
            x_future = np.arange(len(data), len(data) + forecast_steps)
            trend_coeff = np.polyfit(range(len(data)), data, 1)
            forecast = np.polyval(trend_coeff, x_future)
        else:
            forecast = None
        # Detect outliers
        if len(data) > 3:
            q25, q75 = np.percentile(data, [25, 75])
            iqr = q75 - q25
            outlier_mask = (data < q25 - 1.5 * iqr) | (data > q75 + 1.5 * iqr)
            n_outliers = np.sum(outlier_mask)
        else:
            n_outliers = 0
        metrics = {
            'trend': float(trend),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'period': float(period) if period else None,
            'n_outliers': int(n_outliers),
            'forecast_steps': forecast_steps
        }
        result = AnalysisResult(
            analysis_type=AnalysisType.TIME_SERIES,
            model=None,  # Simple analysis doesn't require a model
            metrics=metrics,
            predictions=forecast
        )
        self.results_history.append(result)
        return result
class StatisticalTesting:
    """Statistical hypothesis testing utilities."""
    def __init__(self):
        """Initialize statistical testing."""
        if not SCIPY_AVAILABLE:
            warnings.warn("SciPy not available. Statistical testing limited.")
    def test_normality(self, data: np.ndarray) -> Dict[str, Any]:
        """Test if data follows normal distribution."""
        if not SCIPY_AVAILABLE:
            return {'error': 'SciPy not available'}
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(data)
        # Kolmogorov-Smirnov test against normal distribution
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        return {
            'shapiro_statistic': float(shapiro_stat),
            'shapiro_p_value': float(shapiro_p),
            'ks_statistic': float(ks_stat),
            'ks_p_value': float(ks_p),
            'is_normal': bool(shapiro_p > 0.05 and ks_p > 0.05)
        }
    def compare_groups(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
        """Compare two groups statistically."""
        if not SCIPY_AVAILABLE:
            return {'error': 'SciPy not available'}
        # T-test
        t_stat, t_p = stats.ttest_ind(group1, group2)
        # Mann-Whitney U test (non-parametric)
        u_stat, u_p = stats.mannwhitneyu(group1, group2)
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1) +
                             (len(group2) - 1) * np.var(group2)) /
                            (len(group1) + len(group2) - 2))
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return {
            'group1_mean': float(np.mean(group1)),
            'group2_mean': float(np.mean(group2)),
            't_statistic': float(t_stat),
            't_p_value': float(t_p),
            'mannwhitney_statistic': float(u_stat),
            'mannwhitney_p_value': float(u_p),
            'cohens_d': float(cohens_d),
            'significant_difference': bool(t_p < 0.05)
        }
    def correlation_analysis(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze correlation between two variables."""
        if not SCIPY_AVAILABLE:
            return {'error': 'SciPy not available'}
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(x, y)
        # Spearman correlation (non-parametric)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        # Kendall's tau
        kendall_tau, kendall_p = stats.kendalltau(x, y)
        return {
            'pearson_r': float(pearson_r),
            'pearson_p_value': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p_value': float(spearman_p),
            'kendall_tau': float(kendall_tau),
            'kendall_p_value': float(kendall_p),
            'strong_correlation': bool(abs(pearson_r) > 0.7 and pearson_p < 0.05)
        }
# Convenience functions
def quick_analysis(data: np.ndarray, target: Optional[np.ndarray] = None) -> AnalysisResult:
    """Quickly analyze data with automatic method selection."""
    analyzer = AdvancedAnalytics()
    return analyzer.auto_analyze(data, target)
def compare_models(X: np.ndarray, y: np.ndarray, models: List[Any]) -> Dict[str, Dict]:
    """Compare multiple models on the same dataset."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("Scikit-learn required for model comparison")
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for model in models:
        try:
            model_name = model.__class__.__name__
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            if hasattr(model, "predict_proba"):  # Classification
                score = accuracy_score(y_test, predictions)
                metric_name = 'accuracy'
            else:  # Regression
                score = r2_score(y_test, predictions)
                metric_name = 'r2_score'
            results[model_name] = {
                metric_name: float(score),
                'model': model,
                'predictions': predictions
            }
        except Exception as e:
            results[model_name] = {'error': str(e)}
    return results