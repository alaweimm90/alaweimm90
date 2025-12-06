"""
Meta-Learning Module for Cross-Problem Transfer Learning

This module implements meta-learning capabilities to transfer knowledge
across different optimization problems and warm-start new optimizations.

Author: Meshal Alawein
Date: 2025-11-18
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PerformancePredictor:
    """
    Predicts algorithm performance based on problem features.

    Uses ensemble learning to predict expected performance metrics
    for different algorithms on new problems.
    """

    def __init__(
        self,
        model_type: str = 'gradient_boosting',
        n_estimators: int = 100
    ):
        """
        Initialize the performance predictor.

        Args:
            model_type: Type of model ('gradient_boosting', 'random_forest', 'ridge')
            n_estimators: Number of estimators for ensemble methods
        """
        self.model_type = model_type
        self.n_estimators = n_estimators

        # Initialize models for each algorithm
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False

    def train(
        self,
        training_data: List[Tuple[np.ndarray, str, float]],
        algorithms: List[str]
    ):
        """
        Train prediction models on historical data.

        Args:
            training_data: List of (problem_features, algorithm, performance) tuples
            algorithms: List of algorithm names to train models for
        """
        # Organize data by algorithm
        algorithm_data = {alg: {'X': [], 'y': []} for alg in algorithms}

        for features, algorithm, performance in training_data:
            if algorithm in algorithm_data:
                algorithm_data[algorithm]['X'].append(features)
                algorithm_data[algorithm]['y'].append(performance)

        # Train model for each algorithm
        for algorithm in algorithms:
            X = algorithm_data[algorithm]['X']
            y = algorithm_data[algorithm]['y']

            if len(X) < 10:  # Need minimum samples
                logger.warning(
                    f"Insufficient training data for {algorithm}: {len(X)} samples"
                )
                continue

            X = np.array(X)
            y = np.array(y)

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[algorithm] = scaler

            # Create and train model
            model = self._create_model()
            model.fit(X_scaled, y)
            self.models[algorithm] = model

            # Extract feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[algorithm] = model.feature_importances_

            # Evaluate model
            scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X) // 2))
            logger.info(
                f"Trained model for {algorithm}: "
                f"CV R² = {np.mean(scores):.3f} (+/- {np.std(scores):.3f})"
            )

        self.is_trained = True

    def predict(
        self,
        problem_features: np.ndarray,
        algorithms: Optional[List[str]] = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        Predict performance for each algorithm on a new problem.

        Args:
            problem_features: Feature vector of the problem
            algorithms: Specific algorithms to predict (None = all)

        Returns:
            Dict mapping algorithm names to (predicted_performance, uncertainty)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        if algorithms is None:
            algorithms = list(self.models.keys())

        predictions = {}

        for algorithm in algorithms:
            if algorithm not in self.models:
                continue

            # Scale features
            scaler = self.scalers[algorithm]
            features_scaled = scaler.transform(problem_features.reshape(1, -1))

            # Make prediction
            model = self.models[algorithm]
            pred = model.predict(features_scaled)[0]

            # Estimate uncertainty
            if hasattr(model, 'predict_std'):
                std = model.predict_std(features_scaled)[0]
            elif hasattr(model, 'estimators_'):
                # For ensemble methods, use prediction variance
                preds = np.array([
                    estimator.predict(features_scaled)[0]
                    for estimator in model.estimators_
                ])
                std = np.std(preds)
            else:
                std = 0.1  # Default uncertainty

            predictions[algorithm] = (pred, std)

        return predictions

    def recommend_algorithm(
        self,
        problem_features: np.ndarray,
        risk_aversion: float = 0.5
    ) -> Tuple[str, float]:
        """
        Recommend the best algorithm for a problem.

        Args:
            problem_features: Feature vector of the problem
            risk_aversion: Balance between expected value and uncertainty (0-1)

        Returns:
            Tuple of (recommended_algorithm, confidence_score)
        """
        predictions = self.predict(problem_features)

        if not predictions:
            raise ValueError("No predictions available")

        # Calculate utility scores considering uncertainty
        scores = {}
        for algorithm, (mean, std) in predictions.items():
            # Lower bound confidence interval
            lower_bound = mean - risk_aversion * std
            scores[algorithm] = lower_bound

        # Select best algorithm
        best_algorithm = max(scores.keys(), key=lambda k: scores[k])
        confidence = 1.0 / (1.0 + predictions[best_algorithm][1])  # Convert std to confidence

        return best_algorithm, confidence

    def _create_model(self):
        """Create the prediction model based on model_type."""
        if self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'ridge':
            return Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def save_models(self, path: str):
        """Save trained models to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save models
        with open(path / 'models.pkl', 'wb') as f:
            pickle.dump(self.models, f)

        # Save scalers
        with open(path / 'scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)

        # Save feature importance
        with open(path / 'feature_importance.pkl', 'wb') as f:
            pickle.dump(self.feature_importance, f)

        logger.info(f"Saved models to {path}")

    def load_models(self, path: str):
        """Load trained models from disk."""
        path = Path(path)

        # Load models
        with open(path / 'models.pkl', 'rb') as f:
            self.models = pickle.load(f)

        # Load scalers
        with open(path / 'scalers.pkl', 'rb') as f:
            self.scalers = pickle.load(f)

        # Load feature importance
        with open(path / 'feature_importance.pkl', 'rb') as f:
            self.feature_importance = pickle.load(f)

        self.is_trained = True
        logger.info(f"Loaded models from {path}")


class MetaLearner:
    """
    Main meta-learning system for cross-problem knowledge transfer.

    Learns from optimization history across different problem domains
    and provides warm-starting and transfer learning capabilities.
    """

    def __init__(
        self,
        problem_categories: Optional[List[str]] = None
    ):
        """
        Initialize the meta-learner.

        Args:
            problem_categories: List of problem categories to learn
        """
        self.problem_categories = problem_categories or [
            'qap', 'tsp', 'knapsack', 'scheduling', 'continuous'
        ]

        # Initialize components
        self.predictor = PerformancePredictor()
        self.problem_embeddings = {}
        self.hyperparameter_memory = {}
        self.transfer_matrix = None

    def learn_from_history(
        self,
        optimization_history: List[Dict[str, Any]]
    ):
        """
        Learn patterns from optimization history.

        Args:
            optimization_history: List of past optimization runs with results
        """
        # Extract training data
        training_data = []
        algorithms = set()

        for run in optimization_history:
            features = run['problem_features']
            algorithm = run['algorithm']
            performance = run['performance']

            training_data.append((features, algorithm, performance))
            algorithms.add(algorithm)

        # Train performance predictor
        self.predictor.train(training_data, list(algorithms))

        # Learn problem embeddings
        self._learn_problem_embeddings(optimization_history)

        # Learn hyperparameter patterns
        self._learn_hyperparameter_patterns(optimization_history)

        # Build transfer learning matrix
        self._build_transfer_matrix(optimization_history)

        logger.info(
            f"Meta-learner trained on {len(optimization_history)} runs "
            f"across {len(algorithms)} algorithms"
        )

    def warm_start(
        self,
        problem_features: np.ndarray,
        algorithm: str,
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate warm-start configuration for a new problem.

        Args:
            problem_features: Features of the new problem
            algorithm: Algorithm to warm-start
            base_config: Base configuration to augment

        Returns:
            Warm-started configuration
        """
        config = base_config.copy()

        # Find similar problems
        similar_configs = self._find_similar_configurations(
            problem_features,
            algorithm
        )

        if similar_configs:
            # Blend configurations from similar problems
            config = self._blend_configurations(
                config,
                similar_configs,
                problem_features
            )

        # Apply learned hyperparameter patterns
        if algorithm in self.hyperparameter_memory:
            patterns = self.hyperparameter_memory[algorithm]
            config = self._apply_hyperparameter_patterns(
                config,
                patterns,
                problem_features
            )

        logger.info(f"Generated warm-start configuration for {algorithm}")

        return config

    def transfer_knowledge(
        self,
        source_problem: np.ndarray,
        target_problem: np.ndarray,
        source_solution: Any
    ) -> Any:
        """
        Transfer solution knowledge from source to target problem.

        Args:
            source_problem: Features of source problem
            target_problem: Features of target problem
            source_solution: Solution from source problem

        Returns:
            Adapted solution for target problem
        """
        if self.transfer_matrix is None:
            # No transfer matrix available, return source solution as-is
            return source_solution

        # Calculate transfer weights
        similarity = self._calculate_problem_similarity(
            source_problem,
            target_problem
        )

        if similarity < 0.5:
            # Problems too different for direct transfer
            logger.warning("Problems too different for transfer (similarity={:.2f})".format(similarity))
            return None

        # Apply transfer transformation
        # This is problem-specific and would need domain adaptation
        transferred = self._apply_transfer(
            source_solution,
            source_problem,
            target_problem,
            similarity
        )

        return transferred

    def _learn_problem_embeddings(
        self,
        optimization_history: List[Dict[str, Any]]
    ):
        """Learn low-dimensional embeddings of problems."""
        from sklearn.decomposition import PCA

        # Collect all problem features
        features = []
        problem_ids = []

        for run in optimization_history:
            features.append(run['problem_features'])
            problem_ids.append(run.get('problem_id', len(features)))

        if not features:
            return

        features = np.array(features)

        # Learn PCA embedding
        pca = PCA(n_components=min(10, features.shape[1]))
        embeddings = pca.fit_transform(features)

        # Store embeddings
        for i, problem_id in enumerate(problem_ids):
            self.problem_embeddings[problem_id] = embeddings[i]

        logger.info(
            f"Learned embeddings for {len(problem_ids)} problems "
            f"(explained variance: {sum(pca.explained_variance_ratio_):.2%})"
        )

    def _learn_hyperparameter_patterns(
        self,
        optimization_history: List[Dict[str, Any]]
    ):
        """Learn successful hyperparameter patterns."""
        # Group by algorithm
        algorithm_runs = {}

        for run in optimization_history:
            algorithm = run['algorithm']
            if algorithm not in algorithm_runs:
                algorithm_runs[algorithm] = []

            algorithm_runs[algorithm].append({
                'features': run['problem_features'],
                'hyperparameters': run.get('hyperparameters', {}),
                'performance': run['performance']
            })

        # Extract patterns for each algorithm
        for algorithm, runs in algorithm_runs.items():
            if len(runs) < 5:
                continue

            # Sort by performance
            runs.sort(key=lambda x: x['performance'], reverse=True)

            # Take top 20% runs
            top_runs = runs[:max(1, len(runs) // 5)]

            # Extract common hyperparameter patterns
            patterns = self._extract_hyperparameter_patterns(top_runs)
            self.hyperparameter_memory[algorithm] = patterns

    def _extract_hyperparameter_patterns(
        self,
        runs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract common patterns from successful runs."""
        patterns = {}

        if not runs:
            return patterns

        # Collect all hyperparameter values
        all_params = {}
        for run in runs:
            for param, value in run['hyperparameters'].items():
                if param not in all_params:
                    all_params[param] = []
                all_params[param].append(value)

        # Calculate statistics for each parameter
        for param, values in all_params.items():
            if all(isinstance(v, (int, float)) for v in values):
                # Numerical parameter
                patterns[param] = {
                    'type': 'numerical',
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            else:
                # Categorical parameter
                from collections import Counter
                counts = Counter(values)
                patterns[param] = {
                    'type': 'categorical',
                    'mode': counts.most_common(1)[0][0],
                    'distribution': dict(counts)
                }

        return patterns

    def _build_transfer_matrix(
        self,
        optimization_history: List[Dict[str, Any]]
    ):
        """Build transfer learning matrix between problem types."""
        # Group runs by problem category
        category_performance = {}

        for run in optimization_history:
            category = run.get('problem_category', 'unknown')
            algorithm = run['algorithm']
            performance = run['performance']

            key = (category, algorithm)
            if key not in category_performance:
                category_performance[key] = []
            category_performance[key].append(performance)

        # Build transfer matrix
        n_categories = len(self.problem_categories)
        self.transfer_matrix = np.zeros((n_categories, n_categories))

        for i, cat1 in enumerate(self.problem_categories):
            for j, cat2 in enumerate(self.problem_categories):
                if i == j:
                    self.transfer_matrix[i, j] = 1.0
                else:
                    # Calculate transfer potential based on performance correlation
                    transfer_score = self._calculate_transfer_potential(
                        cat1, cat2, category_performance
                    )
                    self.transfer_matrix[i, j] = transfer_score

    def _calculate_transfer_potential(
        self,
        category1: str,
        category2: str,
        performance_data: Dict[Tuple[str, str], List[float]]
    ) -> float:
        """Calculate transfer learning potential between categories."""
        # Find common algorithms
        algs1 = {k[1] for k in performance_data if k[0] == category1}
        algs2 = {k[1] for k in performance_data if k[0] == category2}
        common_algs = algs1 & algs2

        if not common_algs:
            return 0.0

        # Calculate performance correlation
        correlations = []
        for alg in common_algs:
            perf1 = performance_data.get((category1, alg), [])
            perf2 = performance_data.get((category2, alg), [])

            if len(perf1) > 1 and len(perf2) > 1:
                # Simplified correlation measure
                corr = np.corrcoef(
                    perf1[:min(len(perf1), len(perf2))],
                    perf2[:min(len(perf1), len(perf2))]
                )[0, 1]
                correlations.append(abs(corr))

        return np.mean(correlations) if correlations else 0.0

    def _find_similar_configurations(
        self,
        problem_features: np.ndarray,
        algorithm: str,
        n_similar: int = 5
    ) -> List[Dict[str, Any]]:
        """Find configurations from similar problems."""
        # This would query the performance database
        # Simplified implementation
        return []

    def _blend_configurations(
        self,
        base_config: Dict[str, Any],
        similar_configs: List[Dict[str, Any]],
        problem_features: np.ndarray
    ) -> Dict[str, Any]:
        """Blend multiple configurations based on similarity."""
        blended = base_config.copy()

        # Average numerical parameters
        for param in base_config:
            if isinstance(base_config[param], (int, float)):
                values = [cfg.get(param, base_config[param]) for cfg in similar_configs]
                blended[param] = np.mean(values)

        return blended

    def _apply_hyperparameter_patterns(
        self,
        config: Dict[str, Any],
        patterns: Dict[str, Any],
        problem_features: np.ndarray
    ) -> Dict[str, Any]:
        """Apply learned hyperparameter patterns."""
        updated = config.copy()

        for param, pattern in patterns.items():
            if param not in config:
                continue

            if pattern['type'] == 'numerical':
                # Adjust numerical parameter based on pattern
                current = config[param]
                suggested = pattern['mean']
                # Blend current with suggested
                updated[param] = 0.7 * current + 0.3 * suggested
            elif pattern['type'] == 'categorical':
                # Use most successful value
                updated[param] = pattern['mode']

        return updated

    def _calculate_problem_similarity(
        self,
        problem1: np.ndarray,
        problem2: np.ndarray
    ) -> float:
        """Calculate similarity between two problems."""
        # Cosine similarity
        dot_product = np.dot(problem1, problem2)
        norm1 = np.linalg.norm(problem1)
        norm2 = np.linalg.norm(problem2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _apply_transfer(
        self,
        solution: Any,
        source_features: np.ndarray,
        target_features: np.ndarray,
        similarity: float
    ) -> Any:
        """Apply transfer transformation to solution."""
        # This is highly problem-specific
        # Simplified implementation - would need domain adaptation
        return solution

    def get_feature_importance(self, algorithm: str) -> Optional[np.ndarray]:
        """Get feature importance for an algorithm."""
        return self.predictor.feature_importance.get(algorithm)