"""
SATzilla - Regression-based Algorithm Selection

Reference:
  Xu, L., Hutter, F., Hoos, H. H., & Leyton-Brown, K. (2008).
  SATzilla: Portfolio-based algorithm selection for SAT.
  Journal of Artificial Intelligence Research, 32, 565-606.

Key Ideas:
- Extract instance features
- Train regression models to predict solver performance
- Select solver with best predicted performance
- Use pre-solving for quick decisions
"""

import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from libria_meta.feature_extractor import FeatureExtractor


class SATzilla:
    """
    SATzilla-style regression-based algorithm selection

    For each solver, trains a regression model to predict performance.
    Selects the solver with the best predicted performance.
    """

    def __init__(
        self,
        solvers: List[Any],
        n_estimators: int = 100,
        use_presolving: bool = True,
        presolve_time_limit: float = 2.0
    ):
        """
        Initialize SATzilla

        Args:
            solvers: List of solver instances
            n_estimators: Number of trees in random forest
            use_presolving: Whether to use pre-solving heuristics
            presolve_time_limit: Time limit for pre-solving (seconds)
        """
        self.solvers = solvers
        self.n_estimators = n_estimators
        self.use_presolving = use_presolving
        self.presolve_time_limit = presolve_time_limit

        # Feature extractor
        self.feature_extractor = FeatureExtractor()

        # Regression models: one per solver
        self.models = {
            s.name if hasattr(s, 'name') else str(s): RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
            for s in solvers
        }

        # Feature scaler
        self.scaler = StandardScaler()
        self.fitted = False

        # Solver name mapping
        self.solver_names = [s.name if hasattr(s, 'name') else str(s) for s in solvers]

    @property
    def name(self) -> str:
        return "SATzilla"

    def fit(self, training_data: List[Dict]):
        """
        Train regression models on historical data

        Args:
            training_data: List of dicts with:
                - 'instance': problem instance
                - 'features': instance features (or None to extract)
                - 'performances': {solver_name: performance_score}
        """
        print(f"Training SATzilla on {len(training_data)} instances...")

        # Extract features
        features_list = []
        performance_dict = {name: [] for name in self.solver_names}

        for data in training_data:
            if data.get('features') is not None:
                features = data['features']
            else:
                features = self.feature_extractor.extract(data['instance'])

            features_list.append(features)

            # Collect performance for each solver
            performances = data['performances']
            for solver_name in self.solver_names:
                if solver_name in performances:
                    performance_dict[solver_name].append(performances[solver_name])
                else:
                    # Missing performance - use default
                    performance_dict[solver_name].append(0.5)

        # Convert to arrays
        X = np.array(features_list)

        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Train one regression model per solver
        for solver_name in self.solver_names:
            y = np.array(performance_dict[solver_name])

            print(f"  Training model for {solver_name}...")
            self.models[solver_name].fit(X_scaled, y)

        self.fitted = True
        print(f"✓ SATzilla training complete")

    def select_solver(
        self,
        instance: Any,
        features: Optional[np.ndarray] = None
    ) -> Any:
        """
        Select best solver for instance

        Args:
            instance: Problem instance
            features: Pre-extracted features (optional)

        Returns:
            selected_solver: Best solver for this instance
        """
        if not self.fitted:
            # If not fitted, return first solver
            return self.solvers[0]

        # Extract features
        if features is None:
            features = self.feature_extractor.extract(instance)

        # Scale features
        features_scaled = self.scaler.transform([features])

        # Predict performance for each solver
        predictions = {}
        for solver_name, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            predictions[solver_name] = pred

        # Select solver with best predicted performance
        best_solver_name = max(predictions, key=predictions.get)
        selected_solver = next(
            s for s in self.solvers
            if (s.name if hasattr(s, 'name') else str(s)) == best_solver_name
        )

        return selected_solver

    def predict_performance(
        self,
        instance: Any,
        solver_name: str,
        features: Optional[np.ndarray] = None
    ) -> float:
        """
        Predict performance of a specific solver on instance

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

        features_scaled = self.scaler.transform([features])
        return self.models[solver_name].predict(features_scaled)[0]

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
        if not self.fitted:
            return {name: 0.5 for name in self.solver_names}

        if features is None:
            features = self.feature_extractor.extract(instance)

        features_scaled = self.scaler.transform([features])

        predictions = {}
        for solver_name, model in self.models.items():
            predictions[solver_name] = model.predict(features_scaled)[0]

        return predictions

    def get_feature_importance(self, solver_name: str) -> np.ndarray:
        """
        Get feature importances for a solver's model

        Args:
            solver_name: Solver name

        Returns:
            importances: Feature importance array
        """
        if not self.fitted or solver_name not in self.models:
            return np.array([])

        return self.models[solver_name].feature_importances_

    def update(
        self,
        instance: Any,
        solver_name: str,
        performance: float,
        features: Optional[np.ndarray] = None
    ):
        """
        Online update (simplified - SATzilla typically doesn't do online learning)

        Args:
            instance: Problem instance
            solver_name: Solver that was used
            performance: Observed performance
            features: Pre-extracted features (optional)
        """
        # SATzilla doesn't typically support online learning
        # This is a placeholder for compatibility with Librex.Meta interface
        pass
