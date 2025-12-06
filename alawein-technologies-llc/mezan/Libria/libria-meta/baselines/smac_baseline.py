"""
SMAC - Sequential Model-based Algorithm Configuration

Reference:
  Hutter, F., Hoos, H. H., & Leyton-Brown, K. (2011).
  Sequential model-based optimization for general algorithm configuration.
  In International Conference on Learning and Intelligent Optimization.

Key Ideas:
- Bayesian optimization with random forest surrogate
- Expected improvement acquisition function
- Adaptive sampling of algorithm configurations
- Sequential evaluation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from libria_meta.feature_extractor import FeatureExtractor


class SMACBaseline:
    """
    SMAC-style Bayesian optimization for algorithm selection

    Uses random forest as surrogate model and expected improvement
    for acquisition function.
    """

    def __init__(
        self,
        solvers: List[Any],
        n_estimators: int = 50,
        n_init_samples: int = 10,
        xi: float = 0.01
    ):
        """
        Initialize SMAC

        Args:
            solvers: List of solver instances
            n_estimators: Number of trees in random forest
            n_init_samples: Number of initial random samples
            xi: Exploration parameter for expected improvement
        """
        self.solvers = solvers
        self.n_estimators = n_estimators
        self.n_init_samples = n_init_samples
        self.xi = xi

        # Feature processing
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()

        # Surrogate model: Random Forest
        self.surrogate = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )

        # History of evaluations
        self.X_history = []  # Features
        self.y_history = []  # Performance scores
        self.solver_history = []  # Solver indices

        self.fitted = False
        self.solver_names = [s.name if hasattr(s, 'name') else str(s) for s in solvers]
        self.solver_to_idx = {name: i for i, name in enumerate(self.solver_names)}

    @property
    def name(self) -> str:
        return "SMAC"

    def fit(self, training_data: List[Dict]):
        """
        Train SMAC on historical data

        Args:
            training_data: List of dicts with:
                - 'instance': problem instance
                - 'features': instance features (or None to extract)
                - 'performances': {solver_name: performance_score}
        """
        print(f"Training SMAC on {len(training_data)} instances...")

        # Extract features and build configuration-performance pairs
        for data in training_data:
            if data.get('features') is not None:
                features = data['features']
            else:
                features = self.feature_extractor.extract(data['instance'])

            performances = data['performances']

            # For each solver, create a (features + solver_id, performance) pair
            for solver_name, performance in performances.items():
                if solver_name in self.solver_to_idx:
                    solver_idx = self.solver_to_idx[solver_name]

                    # Augment features with solver ID
                    augmented_features = np.concatenate([
                        features,
                        [solver_idx]
                    ])

                    self.X_history.append(augmented_features)
                    self.y_history.append(performance)
                    self.solver_history.append(solver_idx)

        X = np.array(self.X_history)
        y = np.array(self.y_history)

        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Train surrogate model
        print("  Training random forest surrogate...")
        self.surrogate.fit(X_scaled, y)

        self.fitted = True
        print(f"✓ SMAC training complete ({len(self.X_history)} evaluations)")

    def select_solver(
        self,
        instance: Any,
        features: Optional[np.ndarray] = None
    ) -> Any:
        """
        Select best solver using expected improvement

        Args:
            instance: Problem instance
            features: Pre-extracted features (optional)

        Returns:
            selected_solver: Best solver for this instance
        """
        if not self.fitted:
            return self.solvers[0]

        # Extract features
        if features is None:
            features = self.feature_extractor.extract(instance)

        # Evaluate expected improvement for each solver
        ei_scores = []
        for solver_idx in range(len(self.solvers)):
            # Augment features with solver ID
            augmented_features = np.concatenate([features, [solver_idx]])
            augmented_scaled = self.scaler.transform([augmented_features])

            # Compute expected improvement
            ei = self._expected_improvement(augmented_scaled)
            ei_scores.append(ei)

        # Select solver with highest expected improvement
        best_idx = np.argmax(ei_scores)
        return self.solvers[best_idx]

    def _expected_improvement(self, X: np.ndarray) -> float:
        """
        Compute expected improvement acquisition function

        Args:
            X: Augmented feature vector (features + solver_id)

        Returns:
            ei: Expected improvement score
        """
        # Predict mean and std from random forest
        # For RF, we use prediction variance across trees
        predictions = np.array([
            tree.predict(X)[0]
            for tree in self.surrogate.estimators_
        ])

        mu = np.mean(predictions)
        sigma = np.std(predictions)

        if sigma == 0:
            return 0.0

        # Current best performance
        y_best = np.max(self.y_history) if self.y_history else 0.0

        # Expected improvement formula
        z = (mu - y_best - self.xi) / sigma
        ei = (mu - y_best - self.xi) * norm.cdf(z) + sigma * norm.pdf(z)

        return ei

    def predict_performance(
        self,
        instance: Any,
        solver_name: str,
        features: Optional[np.ndarray] = None
    ) -> float:
        """
        Predict performance of specific solver

        Args:
            instance: Problem instance
            solver_name: Solver to predict for
            features: Pre-extracted features (optional)

        Returns:
            predicted_performance: Predicted performance score
        """
        if not self.fitted:
            return 0.5

        if features is None:
            features = self.feature_extractor.extract(instance)

        solver_idx = self.solver_to_idx[solver_name]
        augmented_features = np.concatenate([features, [solver_idx]])
        augmented_scaled = self.scaler.transform([augmented_features])

        return self.surrogate.predict(augmented_scaled)[0]

    def predict_all(
        self,
        instance: Any,
        features: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Predict performance for all solvers

        Args:
            instance: Problem instance
            features: Pre-extracted features (optional)

        Returns:
            predictions: {solver_name: predicted_performance}
        """
        predictions = {}
        for solver_name in self.solver_names:
            predictions[solver_name] = self.predict_performance(
                instance, solver_name, features
            )
        return predictions

    def update(
        self,
        instance: Any,
        solver_name: str,
        performance: float,
        features: Optional[np.ndarray] = None
    ):
        """
        Online update with new observation

        Args:
            instance: Problem instance
            solver_name: Solver that was used
            performance: Observed performance
            features: Pre-extracted features (optional)
        """
        if features is None:
            features = self.feature_extractor.extract(instance)

        solver_idx = self.solver_to_idx[solver_name]
        augmented_features = np.concatenate([features, [solver_idx]])

        # Add to history
        self.X_history.append(augmented_features)
        self.y_history.append(performance)
        self.solver_history.append(solver_idx)

        # Retrain surrogate (in practice, would use incremental updates)
        if len(self.X_history) % 10 == 0:  # Retrain every 10 updates
            X = np.array(self.X_history)
            y = np.array(self.y_history)
            X_scaled = self.scaler.transform(X)
            self.surrogate.fit(X_scaled, y)
