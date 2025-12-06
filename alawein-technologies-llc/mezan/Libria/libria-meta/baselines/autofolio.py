"""
AutoFolio - Automated Algorithm Portfolio Configuration

Reference:
  Lindauer, M., Hoos, H. H., Hutter, F., & Schaub, T. (2015).
  AutoFolio: An automatically configured algorithm selector.
  Journal of Artificial Intelligence Research, 53, 745-778.

Key Ideas:
- Automatically configure algorithm selection pipelines
- Feature pre-processing and selection
- Multiple classification/regression methods
- Performance pre-solving for easy instances
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from libria_meta.feature_extractor import FeatureExtractor


class AutoFolio:
    """
    AutoFolio-style automated algorithm portfolio configuration

    Combines:
    - Feature pre-processing
    - Feature selection
    - Multiple predictive models
    - Pre-solving for easy instances
    """

    def __init__(
        self,
        solvers: List[Any],
        n_features: int = 10,
        use_feature_selection: bool = True,
        use_presolving: bool = True,
        presolve_threshold: float = 0.9
    ):
        """
        Initialize AutoFolio

        Args:
            solvers: List of solver instances
            n_features: Number of features to select
            use_feature_selection: Whether to use feature selection
            use_presolving: Whether to use pre-solving
            presolve_threshold: Threshold for pre-solving confidence
        """
        self.solvers = solvers
        self.n_features = n_features
        self.use_feature_selection = use_feature_selection
        self.use_presolving = use_presolving
        self.presolve_threshold = presolve_threshold

        # Feature processing
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_regression, k=min(n_features, 10))

        # Classification model for solver selection
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        # Regression models for performance prediction
        self.performance_models = {
            s.name if hasattr(s, 'name') else str(s): GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )
            for s in solvers
        }

        # Pre-solving data
        self.presolve_rules = []

        self.fitted = False
        self.solver_names = [s.name if hasattr(s, 'name') else str(s) for s in solvers]
        self.solver_to_idx = {name: i for i, name in enumerate(self.solver_names)}

    @property
    def name(self) -> str:
        return "AutoFolio"

    def fit(self, training_data: List[Dict]):
        """
        Train AutoFolio on historical data

        Args:
            training_data: List of dicts with:
                - 'instance': problem instance
                - 'features': instance features (or None to extract)
                - 'performances': {solver_name: performance_score}
        """
        print(f"Training AutoFolio on {len(training_data)} instances...")

        # Extract features
        features_list = []
        best_solver_labels = []
        performance_dict = {name: [] for name in self.solver_names}

        for data in training_data:
            if data.get('features') is not None:
                features = data['features']
            else:
                features = self.feature_extractor.extract(data['instance'])

            features_list.append(features)

            # Find best solver for this instance
            performances = data['performances']
            best_solver = max(performances, key=performances.get)
            best_solver_labels.append(self.solver_to_idx[best_solver])

            # Collect performance for regression models
            for solver_name in self.solver_names:
                if solver_name in performances:
                    performance_dict[solver_name].append(performances[solver_name])
                else:
                    performance_dict[solver_name].append(0.5)

        X = np.array(features_list)
        y_class = np.array(best_solver_labels)

        # Feature pre-processing
        print("  Pre-processing features...")
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Feature selection (if enabled)
        if self.use_feature_selection and X_scaled.shape[1] > self.n_features:
            print(f"  Selecting top {self.n_features} features...")
            # Use first solver's performance for feature selection
            y_temp = np.array(performance_dict[self.solver_names[0]])
            self.feature_selector.fit(X_scaled, y_temp)
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled

        # Train classifier
        print("  Training solver classifier...")
        self.classifier.fit(X_selected, y_class)

        # Train performance models
        for solver_name in self.solver_names:
            print(f"  Training performance model for {solver_name}...")
            y_perf = np.array(performance_dict[solver_name])
            self.performance_models[solver_name].fit(X_selected, y_perf)

        # Build pre-solving rules (simplified)
        if self.use_presolving:
            self._build_presolving_rules(X_selected, performance_dict)

        self.fitted = True
        print("✓ AutoFolio training complete")

    def _build_presolving_rules(
        self,
        X: np.ndarray,
        performance_dict: Dict[str, List[float]]
    ):
        """
        Build simple pre-solving rules for easy instances

        Args:
            X: Feature matrix
            performance_dict: Performance data per solver
        """
        # Simplified: identify instances where one solver dominates
        for i in range(len(X)):
            perfs = [performance_dict[name][i] for name in self.solver_names]
            max_perf = max(perfs)
            if max_perf > self.presolve_threshold:
                # This is an "easy" instance
                best_idx = perfs.index(max_perf)
                self.presolve_rules.append({
                    'features': X[i],
                    'solver': self.solver_names[best_idx],
                    'confidence': max_perf
                })

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
            return self.solvers[0]

        # Extract and process features
        if features is None:
            features = self.feature_extractor.extract(instance)

        features_scaled = self.scaler.transform([features])

        # Apply feature selection if enabled
        if self.use_feature_selection and hasattr(self.feature_selector, 'scores_'):
            features_selected = self.feature_selector.transform(features_scaled)
        else:
            features_selected = features_scaled

        # Check pre-solving rules
        if self.use_presolving and self.presolve_rules:
            for rule in self.presolve_rules:
                # Simple distance check
                distance = np.linalg.norm(features_selected[0] - rule['features'])
                if distance < 0.5:  # Threshold
                    solver_name = rule['solver']
                    return next(
                        s for s in self.solvers
                        if (s.name if hasattr(s, 'name') else str(s)) == solver_name
                    )

        # Predict best solver using classifier
        pred_idx = self.classifier.predict(features_selected)[0]
        solver_name = self.solver_names[pred_idx]

        selected_solver = next(
            s for s in self.solvers
            if (s.name if hasattr(s, 'name') else str(s)) == solver_name
        )

        return selected_solver

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

        features_scaled = self.scaler.transform([features])

        if self.use_feature_selection and hasattr(self.feature_selector, 'scores_'):
            features_selected = self.feature_selector.transform(features_scaled)
        else:
            features_selected = features_scaled

        return self.performance_models[solver_name].predict(features_selected)[0]

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
        Online update (not typically supported by AutoFolio)

        Args:
            instance: Problem instance
            solver_name: Solver that was used
            performance: Observed performance
            features: Pre-extracted features (optional)
        """
        # AutoFolio doesn't typically support online learning
        pass
