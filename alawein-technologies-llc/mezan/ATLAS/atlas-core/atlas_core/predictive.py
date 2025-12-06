"""
Predictive Analytics System for ORCHEX
Provides ML-based predictions for workflow execution, resource requirements,
anomaly detection, and optimization suggestions.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import math
import statistics
from enum import Enum
import pickle
import warnings

# Suppress sklearn warnings if not available
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Provide fallback implementations


class PredictionType(Enum):
    """Types of predictions available."""
    EXECUTION_TIME = "execution_time"
    RESOURCE_USAGE = "resource_usage"
    SUCCESS_PROBABILITY = "success_probability"
    BOTTLENECK = "bottleneck"
    ANOMALY = "anomaly"
    OPTIMIZATION = "optimization"


@dataclass
class WorkflowMetrics:
    """Historical metrics for a workflow."""
    workflow_id: str
    execution_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    cpu_usage: List[float] = field(default_factory=list)
    success_rate: float = 0.0
    failure_reasons: List[str] = field(default_factory=list)
    node_metrics: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    timestamps: List[datetime] = field(default_factory=list)


@dataclass
class PredictionResult:
    """Result of a prediction."""
    prediction_type: PredictionType
    value: Any
    confidence: float
    explanation: str
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationSuggestion:
    """Optimization suggestion for workflow."""
    suggestion_type: str
    impact: str  # high, medium, low
    estimated_improvement: float
    implementation: str
    complexity: str  # simple, moderate, complex
    prerequisites: List[str] = field(default_factory=list)


class PredictiveAnalyticsEngine:
    """Main predictive analytics engine for ORCHEX workflows."""

    def __init__(self, history_size: int = 1000):
        """Initialize the predictive analytics engine.

        Args:
            history_size: Maximum number of historical records to keep
        """
        self.history_size = history_size
        self.workflow_history: Dict[str, WorkflowMetrics] = {}
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        self.anomaly_detector = None
        self.is_trained = False

        # Initialize models if sklearn is available
        if SKLEARN_AVAILABLE:
            self._initialize_models()
        else:
            self._initialize_fallback_models()

    def _initialize_models(self):
        """Initialize ML models."""
        self.models = {
            'execution_time': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'resource_usage': RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            ),
            'success_probability': RandomForestRegressor(
                n_estimators=50,
                max_depth=6,
                random_state=42
            )
        }

        self.scalers = {
            'features': StandardScaler(),
            'targets': StandardScaler()
        }

        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )

    def _initialize_fallback_models(self):
        """Initialize fallback models when sklearn is not available."""
        self.models = {
            'execution_time': SimplePredictor('mean'),
            'resource_usage': SimplePredictor('mean'),
            'success_probability': SimplePredictor('ratio')
        }

        self.scalers = {
            'features': SimpleScaler(),
            'targets': SimpleScaler()
        }

        self.anomaly_detector = SimpleAnomalyDetector()

    def add_workflow_execution(self, workflow_id: str, metrics: Dict[str, Any]) -> None:
        """Add a workflow execution to the history.

        Args:
            workflow_id: ID of the workflow
            metrics: Execution metrics
        """
        if workflow_id not in self.workflow_history:
            self.workflow_history[workflow_id] = WorkflowMetrics(workflow_id)

        history = self.workflow_history[workflow_id]

        # Add metrics
        history.execution_times.append(metrics.get('execution_time', 0))
        history.memory_usage.append(metrics.get('memory_usage', 0))
        history.cpu_usage.append(metrics.get('cpu_usage', 0))
        history.timestamps.append(datetime.now())

        # Update success rate
        if metrics.get('success', True):
            history.success_rate = (history.success_rate * (len(history.timestamps) - 1) + 1) / len(history.timestamps)
        else:
            history.success_rate = (history.success_rate * (len(history.timestamps) - 1)) / len(history.timestamps)
            history.failure_reasons.append(metrics.get('failure_reason', 'Unknown'))

        # Store node-level metrics
        if 'node_metrics' in metrics:
            for node_id, node_data in metrics['node_metrics'].items():
                if node_id not in history.node_metrics:
                    history.node_metrics[node_id] = defaultdict(list)

                for metric_name, value in node_data.items():
                    history.node_metrics[node_id][metric_name].append(value)

        # Limit history size
        if len(history.timestamps) > self.history_size:
            history.execution_times.pop(0)
            history.memory_usage.pop(0)
            history.cpu_usage.pop(0)
            history.timestamps.pop(0)

        # Retrain models periodically
        if len(history.timestamps) % 10 == 0:
            self.train_models(workflow_id)

    def train_models(self, workflow_id: Optional[str] = None) -> None:
        """Train predictive models on historical data.

        Args:
            workflow_id: Optional specific workflow to train for
        """
        if not self.workflow_history:
            return

        # Prepare training data
        X, y_time, y_memory, y_cpu = self._prepare_training_data(workflow_id)

        if X.shape[0] < 10:  # Need minimum samples
            return

        if SKLEARN_AVAILABLE:
            # Scale features
            X_scaled = self.scalers['features'].fit_transform(X)

            # Train execution time model
            self.models['execution_time'].fit(X_scaled, y_time)

            # Train resource usage model (combined memory and CPU)
            y_resource = (y_memory + y_cpu) / 2
            self.models['resource_usage'].fit(X_scaled, y_resource)

            # Train anomaly detector
            self.anomaly_detector.fit(X_scaled)

            # Calculate feature importance
            self._calculate_feature_importance(workflow_id)
        else:
            # Use fallback training
            self.models['execution_time'].fit(X, y_time)
            self.models['resource_usage'].fit(X, (y_memory + y_cpu) / 2)
            self.anomaly_detector.fit(X)

        self.is_trained = True

    def _prepare_training_data(self, workflow_id: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from workflow history.

        Returns:
            Tuple of (features, execution_times, memory_usage, cpu_usage)
        """
        features = []
        execution_times = []
        memory_usage = []
        cpu_usage = []

        workflows = [workflow_id] if workflow_id else self.workflow_history.keys()

        for wf_id in workflows:
            if wf_id not in self.workflow_history:
                continue

            history = self.workflow_history[wf_id]

            for i in range(len(history.timestamps)):
                # Create feature vector
                feature_vector = [
                    len(history.node_metrics),  # Number of nodes
                    i,  # Execution index (for trend detection)
                    history.timestamps[i].hour,  # Hour of day
                    history.timestamps[i].weekday(),  # Day of week
                    history.success_rate,  # Historical success rate
                ]

                # Add rolling statistics if enough history
                if i > 0:
                    recent_times = history.execution_times[max(0, i-5):i]
                    feature_vector.extend([
                        np.mean(recent_times),
                        np.std(recent_times) if len(recent_times) > 1 else 0,
                        np.max(recent_times) if recent_times else 0,
                        np.min(recent_times) if recent_times else 0
                    ])
                else:
                    feature_vector.extend([0, 0, 0, 0])

                features.append(feature_vector)
                execution_times.append(history.execution_times[i])
                memory_usage.append(history.memory_usage[i])
                cpu_usage.append(history.cpu_usage[i])

        return (
            np.array(features),
            np.array(execution_times),
            np.array(memory_usage),
            np.array(cpu_usage)
        )

    def predict_execution_time(self, workflow_id: str,
                              context: Optional[Dict[str, Any]] = None) -> PredictionResult:
        """Predict workflow execution time.

        Args:
            workflow_id: ID of the workflow
            context: Optional context information

        Returns:
            Prediction result with estimated execution time
        """
        if not self.is_trained:
            return self._fallback_prediction(
                PredictionType.EXECUTION_TIME,
                workflow_id,
                "execution_times"
            )

        # Prepare features
        features = self._extract_features(workflow_id, context)

        if SKLEARN_AVAILABLE:
            features_scaled = self.scalers['features'].transform([features])
            prediction = self.models['execution_time'].predict(features_scaled)[0]

            # Calculate confidence based on prediction variance
            if hasattr(self.models['execution_time'], 'predict_proba'):
                confidence = self._calculate_confidence(workflow_id, 'execution_time')
            else:
                confidence = 0.75  # Default confidence
        else:
            prediction = self.models['execution_time'].predict([features])[0]
            confidence = 0.5

        # Generate explanation
        explanation = self._generate_time_explanation(workflow_id, prediction)

        # Generate recommendations
        recommendations = self._generate_time_recommendations(workflow_id, prediction)

        return PredictionResult(
            prediction_type=PredictionType.EXECUTION_TIME,
            value=prediction,
            confidence=confidence,
            explanation=explanation,
            recommendations=recommendations,
            metadata={'workflow_id': workflow_id, 'unit': 'seconds'}
        )

    def predict_resource_requirements(self, workflow_id: str,
                                     context: Optional[Dict[str, Any]] = None) -> PredictionResult:
        """Predict resource requirements for workflow.

        Args:
            workflow_id: ID of the workflow
            context: Optional context information

        Returns:
            Prediction result with resource requirements
        """
        if not self.is_trained:
            return self._fallback_prediction(
                PredictionType.RESOURCE_USAGE,
                workflow_id,
                "memory_usage"
            )

        features = self._extract_features(workflow_id, context)

        if SKLEARN_AVAILABLE:
            features_scaled = self.scalers['features'].transform([features])
            resource_score = self.models['resource_usage'].predict(features_scaled)[0]
        else:
            resource_score = self.models['resource_usage'].predict([features])[0]

        # Convert score to actual resource requirements
        resources = {
            'memory_mb': resource_score * 100,  # Scale to MB
            'cpu_cores': max(1, int(resource_score / 25)),
            'disk_gb': resource_score * 0.5,
            'network_mbps': resource_score * 10
        }

        explanation = f"Based on historical data, workflow '{workflow_id}' is expected to require " \
                     f"{resources['memory_mb']:.0f}MB memory and {resources['cpu_cores']} CPU cores."

        recommendations = []
        if resources['memory_mb'] > 1000:
            recommendations.append("Consider using high-memory instances")
        if resources['cpu_cores'] > 4:
            recommendations.append("Enable parallel processing for better performance")

        return PredictionResult(
            prediction_type=PredictionType.RESOURCE_USAGE,
            value=resources,
            confidence=0.7,
            explanation=explanation,
            recommendations=recommendations
        )

    def detect_anomalies(self, workflow_id: str,
                        current_metrics: Dict[str, Any]) -> PredictionResult:
        """Detect anomalies in workflow execution.

        Args:
            workflow_id: ID of the workflow
            current_metrics: Current execution metrics

        Returns:
            Prediction result indicating anomalies
        """
        if not self.is_trained:
            return PredictionResult(
                prediction_type=PredictionType.ANOMALY,
                value=False,
                confidence=0.5,
                explanation="Insufficient data for anomaly detection"
            )

        # Extract features from current metrics
        features = self._extract_features_from_metrics(current_metrics)

        if SKLEARN_AVAILABLE:
            features_scaled = self.scalers['features'].transform([features])
            is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
            anomaly_score = self.anomaly_detector.score_samples(features_scaled)[0]
        else:
            is_anomaly = self.anomaly_detector.predict([features])[0]
            anomaly_score = 0

        anomalies = []
        if is_anomaly:
            # Identify specific anomalies
            anomalies = self._identify_specific_anomalies(workflow_id, current_metrics)

        explanation = f"{'Anomalies detected' if is_anomaly else 'No anomalies detected'} in workflow execution."
        if anomalies:
            explanation += f" Issues: {', '.join(anomalies)}"

        recommendations = []
        if is_anomaly:
            recommendations.append("Investigate the root cause of anomalies")
            recommendations.append("Check system resources and dependencies")
            recommendations.append("Review recent changes to workflow configuration")

        return PredictionResult(
            prediction_type=PredictionType.ANOMALY,
            value=is_anomaly,
            confidence=abs(anomaly_score),
            explanation=explanation,
            recommendations=recommendations,
            metadata={'anomalies': anomalies, 'score': anomaly_score}
        )

    def identify_bottlenecks(self, workflow_id: str) -> PredictionResult:
        """Identify bottlenecks in workflow execution.

        Args:
            workflow_id: ID of the workflow

        Returns:
            Prediction result with identified bottlenecks
        """
        if workflow_id not in self.workflow_history:
            return PredictionResult(
                prediction_type=PredictionType.BOTTLENECK,
                value=[],
                confidence=0,
                explanation="No historical data available for bottleneck analysis"
            )

        history = self.workflow_history[workflow_id]
        bottlenecks = []

        # Analyze node-level metrics
        for node_id, metrics in history.node_metrics.items():
            if 'execution_time' in metrics and metrics['execution_time']:
                avg_time = np.mean(metrics['execution_time'])
                std_time = np.std(metrics['execution_time'])

                # Calculate bottleneck score
                total_time = sum(
                    np.mean(m.get('execution_time', [0]))
                    for m in history.node_metrics.values()
                    if 'execution_time' in m
                )

                if total_time > 0:
                    time_percentage = (avg_time / total_time) * 100

                    if time_percentage > 30:  # Node takes >30% of total time
                        bottlenecks.append({
                            'node_id': node_id,
                            'avg_time': avg_time,
                            'std_time': std_time,
                            'percentage': time_percentage,
                            'severity': 'high' if time_percentage > 50 else 'medium'
                        })

        # Sort by severity
        bottlenecks.sort(key=lambda x: x['percentage'], reverse=True)

        explanation = f"Found {len(bottlenecks)} bottleneck(s) in workflow."
        if bottlenecks:
            top_bottleneck = bottlenecks[0]
            explanation += f" Node '{top_bottleneck['node_id']}' accounts for " \
                          f"{top_bottleneck['percentage']:.1f}% of execution time."

        recommendations = []
        for bottleneck in bottlenecks[:3]:  # Top 3 bottlenecks
            recommendations.append(
                f"Optimize node '{bottleneck['node_id']}' - "
                f"potential {bottleneck['percentage']/2:.1f}% improvement"
            )

        return PredictionResult(
            prediction_type=PredictionType.BOTTLENECK,
            value=bottlenecks,
            confidence=0.8,
            explanation=explanation,
            recommendations=recommendations
        )

    def suggest_optimizations(self, workflow_id: str) -> PredictionResult:
        """Suggest optimizations for workflow.

        Args:
            workflow_id: ID of the workflow

        Returns:
            Prediction result with optimization suggestions
        """
        suggestions = []

        # Analyze historical patterns
        if workflow_id in self.workflow_history:
            history = self.workflow_history[workflow_id]

            # Check for high variability
            if history.execution_times:
                cv = np.std(history.execution_times) / np.mean(history.execution_times)
                if cv > 0.3:  # High coefficient of variation
                    suggestions.append(OptimizationSuggestion(
                        suggestion_type="stabilization",
                        impact="high",
                        estimated_improvement=20,
                        implementation="Implement caching and connection pooling",
                        complexity="moderate",
                        prerequisites=["Identify variable components"]
                    ))

            # Check for resource inefficiency
            if history.memory_usage and history.cpu_usage:
                memory_utilization = np.mean(history.memory_usage)
                cpu_utilization = np.mean(history.cpu_usage)

                if memory_utilization > 80:
                    suggestions.append(OptimizationSuggestion(
                        suggestion_type="memory_optimization",
                        impact="high",
                        estimated_improvement=30,
                        implementation="Optimize data structures and implement streaming",
                        complexity="complex",
                        prerequisites=["Memory profiling"]
                    ))

                if cpu_utilization < 30:
                    suggestions.append(OptimizationSuggestion(
                        suggestion_type="parallelization",
                        impact="medium",
                        estimated_improvement=40,
                        implementation="Increase parallelism and batch processing",
                        complexity="moderate",
                        prerequisites=["Identify parallelizable tasks"]
                    ))

            # Check success rate
            if history.success_rate < 0.95:
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="reliability",
                    impact="high",
                    estimated_improvement=15,
                    implementation="Add retry logic and error handling",
                    complexity="simple",
                    prerequisites=["Error analysis"]
                ))

        # Analyze bottlenecks
        bottleneck_result = self.identify_bottlenecks(workflow_id)
        if bottleneck_result.value:
            for bottleneck in bottleneck_result.value[:2]:
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="bottleneck_removal",
                    impact="high",
                    estimated_improvement=bottleneck['percentage'] * 0.5,
                    implementation=f"Optimize node {bottleneck['node_id']}",
                    complexity="moderate",
                    prerequisites=[f"Profile node {bottleneck['node_id']}"]
                ))

        # Sort by impact and estimated improvement
        suggestions.sort(key=lambda x: (
            {'high': 3, 'medium': 2, 'low': 1}[x.impact],
            x.estimated_improvement
        ), reverse=True)

        explanation = f"Generated {len(suggestions)} optimization suggestions for workflow '{workflow_id}'."
        if suggestions:
            total_improvement = sum(s.estimated_improvement for s in suggestions[:3])
            explanation += f" Top suggestions could improve performance by up to {total_improvement:.0f}%."

        recommendations = [
            f"{s.suggestion_type}: {s.implementation} ({s.estimated_improvement:.0f}% improvement)"
            for s in suggestions[:3]
        ]

        return PredictionResult(
            prediction_type=PredictionType.OPTIMIZATION,
            value=suggestions,
            confidence=0.75,
            explanation=explanation,
            recommendations=recommendations
        )

    def _extract_features(self, workflow_id: str,
                         context: Optional[Dict[str, Any]] = None) -> List[float]:
        """Extract features for prediction.

        Args:
            workflow_id: ID of the workflow
            context: Optional context information

        Returns:
            Feature vector
        """
        features = []

        if workflow_id in self.workflow_history:
            history = self.workflow_history[workflow_id]

            features.extend([
                len(history.node_metrics),  # Number of nodes
                len(history.timestamps),  # Number of executions
                datetime.now().hour,  # Current hour
                datetime.now().weekday(),  # Current day
                history.success_rate  # Historical success rate
            ])

            # Add recent statistics
            if history.execution_times:
                recent = history.execution_times[-5:]
                features.extend([
                    np.mean(recent),
                    np.std(recent) if len(recent) > 1 else 0,
                    np.max(recent),
                    np.min(recent)
                ])
            else:
                features.extend([0, 0, 0, 0])
        else:
            # Default features for unknown workflow
            features = [0] * 9

        # Add context features if provided
        if context:
            features.append(context.get('data_size', 0))
            features.append(context.get('complexity', 1))
            features.append(context.get('priority', 1))

        return features

    def _extract_features_from_metrics(self, metrics: Dict[str, Any]) -> List[float]:
        """Extract features from current metrics.

        Args:
            metrics: Current execution metrics

        Returns:
            Feature vector
        """
        return [
            metrics.get('execution_time', 0),
            metrics.get('memory_usage', 0),
            metrics.get('cpu_usage', 0),
            metrics.get('error_count', 0),
            metrics.get('node_count', 0),
            datetime.now().hour,
            datetime.now().weekday()
        ]

    def _calculate_confidence(self, workflow_id: str, model_name: str) -> float:
        """Calculate prediction confidence.

        Args:
            workflow_id: ID of the workflow
            model_name: Name of the model

        Returns:
            Confidence score between 0 and 1
        """
        base_confidence = 0.5

        if workflow_id in self.workflow_history:
            history = self.workflow_history[workflow_id]
            n_samples = len(history.timestamps)

            # More data = higher confidence
            data_confidence = min(1.0, n_samples / 100)

            # Lower variance = higher confidence
            if history.execution_times:
                cv = np.std(history.execution_times) / (np.mean(history.execution_times) + 1e-6)
                variance_confidence = max(0, 1 - cv)
            else:
                variance_confidence = 0.5

            base_confidence = (data_confidence + variance_confidence) / 2

        return base_confidence

    def _calculate_feature_importance(self, workflow_id: Optional[str] = None) -> None:
        """Calculate feature importance for interpretability.

        Args:
            workflow_id: Optional specific workflow
        """
        if not SKLEARN_AVAILABLE:
            return

        feature_names = [
            'node_count', 'execution_index', 'hour', 'weekday', 'success_rate',
            'recent_mean', 'recent_std', 'recent_max', 'recent_min'
        ]

        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                self.feature_importance[model_name] = {
                    feature_names[i]: importance
                    for i, importance in enumerate(importances)
                    if i < len(feature_names)
                }

    def _identify_specific_anomalies(self, workflow_id: str,
                                    metrics: Dict[str, Any]) -> List[str]:
        """Identify specific anomalies in metrics.

        Args:
            workflow_id: ID of the workflow
            metrics: Current metrics

        Returns:
            List of identified anomalies
        """
        anomalies = []

        if workflow_id in self.workflow_history:
            history = self.workflow_history[workflow_id]

            # Check execution time
            if history.execution_times and 'execution_time' in metrics:
                mean_time = np.mean(history.execution_times)
                std_time = np.std(history.execution_times)
                current_time = metrics['execution_time']

                if abs(current_time - mean_time) > 3 * std_time:
                    anomalies.append(f"Execution time ({current_time:.1f}s) is abnormal")

            # Check memory usage
            if history.memory_usage and 'memory_usage' in metrics:
                mean_memory = np.mean(history.memory_usage)
                current_memory = metrics['memory_usage']

                if current_memory > mean_memory * 2:
                    anomalies.append(f"Memory usage ({current_memory:.0f}MB) is unusually high")

            # Check error rate
            if 'error_count' in metrics and metrics['error_count'] > 0:
                if history.success_rate > 0.95:
                    anomalies.append(f"Errors detected in normally stable workflow")

        return anomalies

    def _generate_time_explanation(self, workflow_id: str, prediction: float) -> str:
        """Generate explanation for execution time prediction.

        Args:
            workflow_id: ID of the workflow
            prediction: Predicted execution time

        Returns:
            Human-readable explanation
        """
        explanation = f"Predicted execution time for '{workflow_id}' is {prediction:.1f} seconds"

        if workflow_id in self.workflow_history:
            history = self.workflow_history[workflow_id]
            if history.execution_times:
                avg_historical = np.mean(history.execution_times)
                diff_percent = ((prediction - avg_historical) / avg_historical) * 100

                if abs(diff_percent) > 10:
                    trend = "higher" if diff_percent > 0 else "lower"
                    explanation += f", {abs(diff_percent):.0f}% {trend} than average"

        return explanation

    def _generate_time_recommendations(self, workflow_id: str,
                                      prediction: float) -> List[str]:
        """Generate recommendations based on time prediction.

        Args:
            workflow_id: ID of the workflow
            prediction: Predicted execution time

        Returns:
            List of recommendations
        """
        recommendations = []

        if prediction > 300:  # More than 5 minutes
            recommendations.append("Consider breaking down into smaller workflows")
            recommendations.append("Enable checkpointing for long-running tasks")

        if workflow_id in self.workflow_history:
            history = self.workflow_history[workflow_id]
            if history.execution_times and len(history.execution_times) > 10:
                cv = np.std(history.execution_times) / np.mean(history.execution_times)
                if cv > 0.3:
                    recommendations.append("High variability detected - investigate causes")

        return recommendations

    def _fallback_prediction(self, prediction_type: PredictionType,
                            workflow_id: str, metric_key: str) -> PredictionResult:
        """Fallback prediction when models are not trained.

        Args:
            prediction_type: Type of prediction
            workflow_id: ID of the workflow
            metric_key: Key for the metric in history

        Returns:
            Simple prediction based on historical average
        """
        if workflow_id in self.workflow_history:
            history = self.workflow_history[workflow_id]
            metric_values = getattr(history, metric_key, [])

            if metric_values:
                value = np.mean(metric_values)
                explanation = f"Based on average of {len(metric_values)} historical executions"
            else:
                value = 0
                explanation = "No historical data available"
        else:
            value = 0
            explanation = "Unknown workflow - no historical data"

        return PredictionResult(
            prediction_type=prediction_type,
            value=value,
            confidence=0.3,
            explanation=explanation
        )

    def export_models(self, filepath: str) -> None:
        """Export trained models to file.

        Args:
            filepath: Path to save models
        """
        export_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'workflow_history': self.workflow_history,
            'is_trained': self.is_trained
        }

        with open(filepath, 'wb') as f:
            pickle.dump(export_data, f)

    def import_models(self, filepath: str) -> None:
        """Import trained models from file.

        Args:
            filepath: Path to load models from
        """
        with open(filepath, 'rb') as f:
            export_data = pickle.load(f)

        self.models = export_data['models']
        self.scalers = export_data['scalers']
        self.feature_importance = export_data['feature_importance']
        self.workflow_history = export_data['workflow_history']
        self.is_trained = export_data['is_trained']


# Fallback implementations when sklearn is not available
class SimplePredictor:
    """Simple predictor for when sklearn is not available."""

    def __init__(self, method: str = 'mean'):
        self.method = method
        self.data = []

    def fit(self, X, y):
        self.data = y

    def predict(self, X):
        if self.method == 'mean' and self.data:
            return [np.mean(self.data)]
        elif self.method == 'ratio' and self.data:
            return [sum(1 for x in self.data if x > 0) / len(self.data)]
        return [0]


class SimpleScaler:
    """Simple scaler for when sklearn is not available."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-6
        return (X - self.mean) / self.std

    def transform(self, X):
        if self.mean is not None and self.std is not None:
            return (X - self.mean) / self.std
        return X


class SimpleAnomalyDetector:
    """Simple anomaly detector for when sklearn is not available."""

    def __init__(self):
        self.threshold = None

    def fit(self, X):
        if len(X) > 0:
            distances = [np.linalg.norm(x - np.mean(X, axis=0)) for x in X]
            self.threshold = np.percentile(distances, 90)

    def predict(self, X):
        if self.threshold is None:
            return [False] * len(X)

        results = []
        for x in X:
            distance = np.linalg.norm(x - np.mean(X, axis=0))
            results.append(distance > self.threshold)
        return results

    def score_samples(self, X):
        return [0.5] * len(X)