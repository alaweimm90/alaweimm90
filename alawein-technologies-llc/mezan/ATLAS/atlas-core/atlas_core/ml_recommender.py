"""
ML-Based Solver Recommendation Engine for MEZAN
------------------------------------------------
Production-grade machine learning system for intelligent optimization solver selection.
Provides feature extraction, model training, prediction with confidence scores,
online learning, and comprehensive model management.

Author: Meshal Alawein
Date: November 2025
"""

import json
import logging
import pickle
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProblemFeatures:
    """Container for optimization problem features."""

    # Problem size features
    num_variables: int = 0
    num_constraints: int = 0
    num_objectives: int = 1
    problem_dimension: int = 0

    # Constraint features
    constraint_density: float = 0.0
    linear_constraints_ratio: float = 0.0
    nonlinear_constraints_ratio: float = 0.0
    equality_constraints_ratio: float = 0.0

    # Variable features
    continuous_vars_ratio: float = 1.0
    discrete_vars_ratio: float = 0.0
    binary_vars_ratio: float = 0.0
    bounded_vars_ratio: float = 0.0

    # Objective features
    objective_linearity: float = 0.0
    objective_convexity: float = 0.0
    objective_smoothness: float = 0.0
    objective_multimodality: float = 0.0

    # Graph/network features (for combinatorial problems)
    graph_density: float = 0.0
    graph_connectivity: float = 0.0
    graph_clustering_coefficient: float = 0.0
    average_degree: float = 0.0

    # Statistical features
    coefficient_variance: float = 0.0
    coefficient_sparsity: float = 0.0
    condition_number: float = 0.0

    # Domain-specific features
    is_combinatorial: bool = False
    is_continuous: bool = True
    is_mixed_integer: bool = False
    is_quadratic: bool = False
    is_nonlinear: bool = False

    def to_array(self) -> np.ndarray:
        """Convert features to numpy array."""
        return np.array([
            self.num_variables,
            self.num_constraints,
            self.num_objectives,
            self.problem_dimension,
            self.constraint_density,
            self.linear_constraints_ratio,
            self.nonlinear_constraints_ratio,
            self.equality_constraints_ratio,
            self.continuous_vars_ratio,
            self.discrete_vars_ratio,
            self.binary_vars_ratio,
            self.bounded_vars_ratio,
            self.objective_linearity,
            self.objective_convexity,
            self.objective_smoothness,
            self.objective_multimodality,
            self.graph_density,
            self.graph_connectivity,
            self.graph_clustering_coefficient,
            self.average_degree,
            self.coefficient_variance,
            self.coefficient_sparsity,
            self.condition_number,
            float(self.is_combinatorial),
            float(self.is_continuous),
            float(self.is_mixed_integer),
            float(self.is_quadratic),
            float(self.is_nonlinear),
        ])

    @classmethod
    def feature_names(cls) -> List[str]:
        """Get feature names in order."""
        return [
            'num_variables', 'num_constraints', 'num_objectives', 'problem_dimension',
            'constraint_density', 'linear_constraints_ratio', 'nonlinear_constraints_ratio',
            'equality_constraints_ratio', 'continuous_vars_ratio', 'discrete_vars_ratio',
            'binary_vars_ratio', 'bounded_vars_ratio', 'objective_linearity',
            'objective_convexity', 'objective_smoothness', 'objective_multimodality',
            'graph_density', 'graph_connectivity', 'graph_clustering_coefficient',
            'average_degree', 'coefficient_variance', 'coefficient_sparsity',
            'condition_number', 'is_combinatorial', 'is_continuous',
            'is_mixed_integer', 'is_quadratic', 'is_nonlinear'
        ]


class FeatureExtractor:
    """Extract features from optimization problems."""

    def __init__(self):
        """Initialize feature extractor."""
        self.feature_cache = {}

    def extract_features(self, problem: Dict[str, Any]) -> ProblemFeatures:
        """
        Extract features from an optimization problem.

        Args:
            problem: Problem dictionary containing problem definition

        Returns:
            ProblemFeatures object
        """
        features = ProblemFeatures()

        # Extract basic dimensions
        features.num_variables = problem.get('num_variables', 0)
        features.num_constraints = problem.get('num_constraints', 0)
        features.num_objectives = problem.get('num_objectives', 1)
        features.problem_dimension = max(features.num_variables,
                                        problem.get('dimension', 0))

        # Calculate constraint features
        if features.num_variables > 0 and features.num_constraints > 0:
            features.constraint_density = features.num_constraints / features.num_variables

        constraints = problem.get('constraints', {})
        total_constraints = max(features.num_constraints, 1)
        features.linear_constraints_ratio = constraints.get('linear', 0) / total_constraints
        features.nonlinear_constraints_ratio = constraints.get('nonlinear', 0) / total_constraints
        features.equality_constraints_ratio = constraints.get('equality', 0) / total_constraints

        # Variable type features
        variables = problem.get('variables', {})
        total_vars = max(features.num_variables, 1)
        features.continuous_vars_ratio = variables.get('continuous', total_vars) / total_vars
        features.discrete_vars_ratio = variables.get('discrete', 0) / total_vars
        features.binary_vars_ratio = variables.get('binary', 0) / total_vars
        features.bounded_vars_ratio = variables.get('bounded', 0) / total_vars

        # Objective function features
        objective = problem.get('objective', {})
        features.objective_linearity = objective.get('linearity', 0.0)
        features.objective_convexity = objective.get('convexity', 0.0)
        features.objective_smoothness = objective.get('smoothness', 1.0)
        features.objective_multimodality = objective.get('multimodality', 0.0)

        # Graph features (for combinatorial problems)
        graph = problem.get('graph', {})
        features.graph_density = graph.get('density', 0.0)
        features.graph_connectivity = graph.get('connectivity', 0.0)
        features.graph_clustering_coefficient = graph.get('clustering', 0.0)
        features.average_degree = graph.get('avg_degree', 0.0)

        # Statistical features
        stats = problem.get('statistics', {})
        features.coefficient_variance = stats.get('variance', 0.0)
        features.coefficient_sparsity = stats.get('sparsity', 0.0)
        features.condition_number = min(stats.get('condition_number', 1.0), 1e6)

        # Problem type features
        problem_type = problem.get('type', '')
        features.is_combinatorial = 'combinatorial' in problem_type.lower()
        features.is_continuous = 'continuous' in problem_type.lower()
        features.is_mixed_integer = 'mixed' in problem_type.lower()
        features.is_quadratic = 'quadratic' in problem_type.lower()
        features.is_nonlinear = 'nonlinear' in problem_type.lower()

        return features

    def extract_batch(self, problems: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from multiple problems."""
        features_list = []
        for problem in problems:
            features = self.extract_features(problem)
            features_list.append(features.to_array())
        return np.array(features_list)


class SolverRecommender:
    """ML-based solver recommendation system."""

    AVAILABLE_SOLVERS = [
        'genetic_algorithm',
        'simulated_annealing',
        'particle_swarm',
        'differential_evolution',
        'tabu_search',
        'ant_colony',
        'gradient_descent',
        'newton_method',
        'interior_point',
        'simplex',
        'branch_and_bound',
        'dynamic_programming'
    ]

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize solver recommender.

        Args:
            model_type: Type of ML model to use
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_extractor = FeatureExtractor()
        self.feature_selector = None
        self.training_history = []
        self.performance_metrics = {}
        self.confidence_threshold = 0.6
        self.model_version = '1.0.0'
        self.last_training_date = None

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the ML model based on type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'extra_trees':
            self.model = ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'neural_network':
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        else:
            # Default to logistic regression
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )

    def train(self, training_data: List[Dict[str, Any]],
              perform_hyperopt: bool = True,
              cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train the recommendation model.

        Args:
            training_data: List of training examples
            perform_hyperopt: Whether to perform hyperparameter optimization
            cv_folds: Number of cross-validation folds

        Returns:
            Training metrics
        """
        logger.info(f"Training solver recommender with {len(training_data)} examples")

        # Prepare training data
        X = []
        y = []
        for example in training_data:
            features = self.feature_extractor.extract_features(example['problem'])
            X.append(features.to_array())
            y.append(example['best_solver'])

        X = np.array(X)
        y = np.array(y)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Feature selection
        self.feature_selector = SelectKBest(mutual_info_classif, k=min(20, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X_scaled, y_encoded)

        # Hyperparameter optimization
        if perform_hyperopt and self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_selected, y_encoded)
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            logger.info(f"Best hyperparameters: {best_params}")
        else:
            # Regular training
            self.model.fit(X_selected, y_encoded)

        # Cross-validation
        cv_scores = cross_val_score(
            self.model,
            X_selected,
            y_encoded,
            cv=StratifiedKFold(n_splits=cv_folds),
            scoring='accuracy'
        )

        # Calculate training metrics
        y_pred = self.model.predict(X_selected)

        metrics = {
            'accuracy': accuracy_score(y_encoded, y_pred),
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_encoded, y_pred).tolist(),
            'classification_report': classification_report(
                y_encoded, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            ),
            'feature_importance': self._get_feature_importance(),
            'training_samples': len(X),
            'timestamp': datetime.now().isoformat()
        }

        self.performance_metrics = metrics
        self.last_training_date = datetime.now()
        self.training_history.append(metrics)

        logger.info(f"Training completed. Accuracy: {metrics['accuracy']:.3f}")

        return metrics

    def predict(self, problem: Dict[str, Any],
                return_confidence: bool = True) -> Union[str, Tuple[str, float]]:
        """
        Predict the best solver for a problem.

        Args:
            problem: Problem dictionary
            return_confidence: Whether to return confidence score

        Returns:
            Solver name or (solver_name, confidence) tuple
        """
        if self.model is None:
            logger.warning("Model not trained, using rule-based fallback")
            return self._rule_based_selection(problem)

        # Extract features
        features = self.feature_extractor.extract_features(problem)
        X = features.to_array().reshape(1, -1)

        # Scale and select features
        X_scaled = self.scaler.transform(X)
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled

        # Get prediction and confidence
        try:
            y_pred = self.model.predict(X_selected)[0]
            solver_name = self.label_encoder.inverse_transform([y_pred])[0]

            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_selected)[0]
                confidence = max(probabilities)
            else:
                confidence = 1.0  # Default confidence for non-probabilistic models

            # Fallback if confidence is too low
            if confidence < self.confidence_threshold:
                logger.info(f"Low confidence ({confidence:.2f}), using rule-based fallback")
                solver_name = self._rule_based_selection(problem)
                confidence = self.confidence_threshold

            if return_confidence:
                return solver_name, confidence
            return solver_name

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._rule_based_selection(problem)

    def predict_top_k(self, problem: Dict[str, Any], k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-k solver recommendations with confidence scores.

        Args:
            problem: Problem dictionary
            k: Number of recommendations

        Returns:
            List of (solver_name, confidence) tuples
        """
        if self.model is None or not hasattr(self.model, 'predict_proba'):
            solver = self._rule_based_selection(problem)
            return [(solver, 1.0)]

        # Extract and prepare features
        features = self.feature_extractor.extract_features(problem)
        X = features.to_array().reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled

        # Get probabilities
        probabilities = self.model.predict_proba(X_selected)[0]

        # Get top-k predictions
        top_k_indices = np.argsort(probabilities)[-k:][::-1]
        recommendations = []

        for idx in top_k_indices:
            solver_name = self.label_encoder.inverse_transform([idx])[0]
            confidence = probabilities[idx]
            recommendations.append((solver_name, confidence))

        return recommendations

    def update_online(self, problem: Dict[str, Any],
                      solver: str,
                      performance: float):
        """
        Update model with new data point (online learning).

        Args:
            problem: Problem dictionary
            solver: Solver that was used
            performance: Performance metric (higher is better)
        """
        # Store new training example
        example = {
            'problem': problem,
            'best_solver': solver,
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        }

        self.training_history.append(example)

        # Retrain periodically (every 100 examples)
        if len(self.training_history) % 100 == 0:
            logger.info("Retraining model with online data")
            self.train(self.training_history, perform_hyperopt=False)

    def _rule_based_selection(self, problem: Dict[str, Any]) -> str:
        """
        Fallback rule-based solver selection.

        Args:
            problem: Problem dictionary

        Returns:
            Selected solver name
        """
        features = self.feature_extractor.extract_features(problem)

        # Rule-based selection logic
        if features.is_combinatorial:
            if features.graph_density > 0.5:
                return 'tabu_search'
            elif features.num_variables < 100:
                return 'branch_and_bound'
            else:
                return 'genetic_algorithm'

        if features.is_continuous:
            if features.objective_convexity > 0.8:
                if features.objective_smoothness > 0.8:
                    return 'gradient_descent'
                else:
                    return 'interior_point'
            elif features.objective_multimodality > 0.5:
                return 'particle_swarm'
            else:
                return 'simulated_annealing'

        if features.is_mixed_integer:
            return 'branch_and_bound'

        if features.is_quadratic:
            if features.num_constraints > 0:
                return 'interior_point'
            else:
                return 'newton_method'

        # Default fallback
        return 'genetic_algorithm'

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not hasattr(self.model, 'feature_importances_'):
            return {}

        feature_names = ProblemFeatures.feature_names()
        importances = self.model.feature_importances_

        # Map to selected features
        if self.feature_selector is not None:
            selected_indices = self.feature_selector.get_support(indices=True)
            feature_importance = {}
            for idx, importance in zip(selected_indices, importances):
                if idx < len(feature_names):
                    feature_importance[feature_names[idx]] = float(importance)
        else:
            feature_importance = {
                name: float(imp)
                for name, imp in zip(feature_names, importances)
            }

        # Sort by importance
        return dict(sorted(feature_importance.items(),
                          key=lambda x: x[1],
                          reverse=True))

    def save_model(self, path: Union[str, Path]):
        """Save the trained model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_selector': self.feature_selector,
            'performance_metrics': self.performance_metrics,
            'model_type': self.model_type,
            'model_version': self.model_version,
            'last_training_date': self.last_training_date,
            'training_history': self.training_history[-100:]  # Keep last 100 examples
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: Union[str, Path]):
        """Load a trained model from disk."""
        path = Path(path)

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_selector = model_data.get('feature_selector')
        self.performance_metrics = model_data.get('performance_metrics', {})
        self.model_type = model_data.get('model_type', 'random_forest')
        self.model_version = model_data.get('model_version', '1.0.0')
        self.last_training_date = model_data.get('last_training_date')
        self.training_history = model_data.get('training_history', [])

        logger.info(f"Model loaded from {path}")

    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.

        Args:
            test_data: List of test examples

        Returns:
            Evaluation metrics
        """
        X = []
        y_true = []

        for example in test_data:
            features = self.feature_extractor.extract_features(example['problem'])
            X.append(features.to_array())
            y_true.append(example['best_solver'])

        X = np.array(X)
        X_scaled = self.scaler.transform(X)

        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled

        # Get predictions
        y_pred_encoded = self.model.predict(X_selected)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        # Calculate metrics
        y_true_encoded = self.label_encoder.transform(y_true)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_recall_f1': precision_recall_fscore_support(
                y_true, y_pred,
                average='weighted'
            ),
            'confusion_matrix': confusion_matrix(y_true_encoded, y_pred_encoded).tolist(),
            'per_class_accuracy': {},
            'test_samples': len(X)
        }

        # Per-class accuracy
        for solver in self.AVAILABLE_SOLVERS:
            if solver in y_true:
                mask = np.array(y_true) == solver
                if mask.sum() > 0:
                    class_acc = (np.array(y_pred)[mask] == solver).mean()
                    metrics['per_class_accuracy'][solver] = float(class_acc)

        return metrics


def generate_synthetic_training_data(n_samples: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic training data for testing."""
    np.random.seed(42)
    training_data = []

    solvers = SolverRecommender.AVAILABLE_SOLVERS

    for _ in range(n_samples):
        # Generate random problem characteristics
        problem = {
            'num_variables': np.random.randint(10, 1000),
            'num_constraints': np.random.randint(0, 500),
            'num_objectives': np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1]),
            'type': np.random.choice(['continuous', 'combinatorial', 'mixed']),
            'constraints': {
                'linear': np.random.randint(0, 100),
                'nonlinear': np.random.randint(0, 50),
                'equality': np.random.randint(0, 30),
            },
            'variables': {
                'continuous': np.random.randint(0, 500),
                'discrete': np.random.randint(0, 200),
                'binary': np.random.randint(0, 100),
            },
            'objective': {
                'linearity': np.random.random(),
                'convexity': np.random.random(),
                'smoothness': np.random.random(),
                'multimodality': np.random.random(),
            },
            'graph': {
                'density': np.random.random(),
                'connectivity': np.random.random(),
                'clustering': np.random.random(),
                'avg_degree': np.random.random() * 10,
            }
        }

        # Assign best solver based on rules (simulating ground truth)
        if 'combinatorial' in problem['type']:
            best_solver = np.random.choice(['genetic_algorithm', 'tabu_search', 'ant_colony'])
        elif problem['objective']['convexity'] > 0.7:
            best_solver = np.random.choice(['gradient_descent', 'newton_method'])
        else:
            best_solver = np.random.choice(solvers)

        training_data.append({
            'problem': problem,
            'best_solver': best_solver
        })

    return training_data


# Example usage and training workflow
if __name__ == "__main__":
    # Initialize recommender
    recommender = SolverRecommender(model_type='random_forest')

    # Generate synthetic training data
    print("Generating synthetic training data...")
    training_data = generate_synthetic_training_data(n_samples=1000)

    # Split into train/test
    split_idx = int(0.8 * len(training_data))
    train_data = training_data[:split_idx]
    test_data = training_data[split_idx:]

    # Train the model
    print(f"Training on {len(train_data)} samples...")
    train_metrics = recommender.train(train_data, perform_hyperopt=True)
    print(f"Training accuracy: {train_metrics['accuracy']:.3f}")
    print(f"CV accuracy: {train_metrics['cv_accuracy_mean']:.3f} ± {train_metrics['cv_accuracy_std']:.3f}")

    # Evaluate on test set
    print(f"\nEvaluating on {len(test_data)} test samples...")
    test_metrics = recommender.evaluate(test_data)
    print(f"Test accuracy: {test_metrics['accuracy']:.3f}")

    # Example prediction
    example_problem = test_data[0]['problem']
    solver, confidence = recommender.predict(example_problem, return_confidence=True)
    print(f"\nExample prediction: {solver} (confidence: {confidence:.2f})")

    # Get top-3 recommendations
    top_3 = recommender.predict_top_k(example_problem, k=3)
    print("\nTop-3 recommendations:")
    for solver, conf in top_3:
        print(f"  - {solver}: {conf:.3f}")

    # Feature importance
    print("\nTop feature importances:")
    importance = train_metrics.get('feature_importance', {})
    for feature, score in list(importance.items())[:5]:
        print(f"  - {feature}: {score:.3f}")

    # Save model
    model_path = Path("models/solver_recommender.pkl")
    recommender.save_model(model_path)
    print(f"\nModel saved to {model_path}")