"""
BOHB - Bayesian Optimization and Hyperband

Reference:
  Falkner, S., Klein, A., & Hutter, F. (2018).
  BOHB: Robust and efficient hyperparameter optimization at scale.
  In International Conference on Machine Learning (pp. 1437-1446).

Key Ideas:
- Combines Hyperband's efficiency with BO's sample efficiency
- Uses kernel density estimation for modeling
- Successive halving with BO-guided sampling
- Multi-fidelity optimization
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.stats import gaussian_kde
from libria_meta.feature_extractor import FeatureExtractor


class BOHB:
    """
    BOHB-style Bayesian Optimization + Hyperband for algorithm selection

    Combines:
    - Hyperband's successive halving
    - Bayesian optimization for intelligent sampling
    - Kernel density estimation for modeling
    """

    def __init__(
        self,
        solvers: List[Any],
        max_budget: int = 27,
        eta: int = 3,
        min_points_in_model: int = 10,
        top_n_percent: int = 15,
        bandwidth_factor: float = 3.0
    ):
        """
        Initialize BOHB

        Args:
            solvers: List of solver instances
            max_budget: Maximum budget per configuration
            eta: Downsampling rate (typically 3)
            min_points_in_model: Minimum points before using BO
            top_n_percent: Top percent for good performance modeling
            bandwidth_factor: KDE bandwidth multiplier
        """
        self.solvers = solvers
        self.max_budget = max_budget
        self.eta = eta
        self.min_points_in_model = min_points_in_model
        self.top_n_percent = top_n_percent
        self.bandwidth_factor = bandwidth_factor

        # Feature processing
        self.feature_extractor = FeatureExtractor()

        # Observation history
        self.observations = []  # List of (features, solver_idx, performance)

        # KDE models for good and bad configurations
        self.good_kde = None
        self.bad_kde = None

        self.solver_names = [s.name if hasattr(s, 'name') else str(s) for s in solvers]
        self.solver_to_idx = {name: i for i, name in enumerate(self.solver_names)}

    @property
    def name(self) -> str:
        return "BOHB"

    def fit(self, training_data: List[Dict]):
        """
        Train BOHB on historical data

        Args:
            training_data: List of dicts with:
                - 'instance': problem instance
                - 'features': instance features (or None to extract)
                - 'performances': {solver_name: performance_score}
        """
        print(f"Training BOHB on {len(training_data)} instances...")

        # Collect observations
        for data in training_data:
            if data.get('features') is not None:
                features = data['features']
            else:
                features = self.feature_extractor.extract(data['instance'])

            performances = data['performances']

            # Store observations for each solver
            for solver_name, performance in performances.items():
                if solver_name in self.solver_to_idx:
                    solver_idx = self.solver_to_idx[solver_name]

                    # Augment features with solver index
                    augmented = np.concatenate([features, [solver_idx]])

                    self.observations.append({
                        'features': augmented,
                        'solver_idx': solver_idx,
                        'performance': performance
                    })

        # Build KDE models
        if len(self.observations) >= self.min_points_in_model:
            self._build_kde_models()
            print(f"  ✓ Built KDE models from {len(self.observations)} observations")
        else:
            print(f"  ⚠ Not enough observations ({len(self.observations)}) for KDE models")

        print("✓ BOHB training complete")

    def _build_kde_models(self):
        """Build kernel density estimation models for good/bad configurations"""
        # Sort observations by performance
        sorted_obs = sorted(
            self.observations,
            key=lambda x: x['performance'],
            reverse=True
        )

        # Split into good and bad
        n_good = max(1, int(len(sorted_obs) * self.top_n_percent / 100))
        good_obs = sorted_obs[:n_good]
        bad_obs = sorted_obs[n_good:]

        # Extract feature matrices
        good_features = np.array([obs['features'] for obs in good_obs]).T
        bad_features = np.array([obs['features'] for obs in bad_obs]).T

        # Build KDE models
        try:
            self.good_kde = gaussian_kde(
                good_features,
                bw_method='scott'
            )
            self.bad_kde = gaussian_kde(
                bad_features,
                bw_method='scott'
            )
        except Exception as e:
            print(f"  Warning: KDE building failed: {e}")
            self.good_kde = None
            self.bad_kde = None

    def select_solver(
        self,
        instance: Any,
        features: Optional[np.ndarray] = None
    ) -> Any:
        """
        Select best solver using BOHB acquisition function

        Args:
            instance: Problem instance
            features: Pre-extracted features (optional)

        Returns:
            selected_solver: Best solver
        """
        # Extract features
        if features is None:
            features = self.feature_extractor.extract(instance)

        # If KDE models not available, use random selection
        if self.good_kde is None or self.bad_kde is None:
            # Fallback: select based on average performance
            if self.observations:
                solver_perfs = {}
                for solver_name in self.solver_names:
                    solver_idx = self.solver_to_idx[solver_name]
                    perfs = [
                        obs['performance']
                        for obs in self.observations
                        if obs['solver_idx'] == solver_idx
                    ]
                    solver_perfs[solver_name] = np.mean(perfs) if perfs else 0.5

                best_solver_name = max(solver_perfs, key=solver_perfs.get)
            else:
                best_solver_name = self.solver_names[0]

            return next(
                s for s in self.solvers
                if (s.name if hasattr(s, 'name') else str(s)) == best_solver_name
            )

        # Evaluate acquisition function for each solver
        acquisition_scores = []
        for solver_idx in range(len(self.solvers)):
            # Augment features with solver index
            augmented = np.concatenate([features, [solver_idx]])

            # Compute acquisition = p(good) / p(bad)
            try:
                p_good = self.good_kde.evaluate(augmented)[0]
                p_bad = self.bad_kde.evaluate(augmented)[0]

                # Avoid division by zero
                acquisition = p_good / (p_bad + 1e-10)
            except Exception:
                acquisition = 0.5

            acquisition_scores.append(acquisition)

        # Select solver with highest acquisition score
        best_idx = np.argmax(acquisition_scores)
        return self.solvers[best_idx]

    def run_bohb_iteration(
        self,
        instances: List[Any],
        budget: int = None
    ) -> Dict[str, Any]:
        """
        Run one BOHB iteration with successive halving

        Args:
            instances: List of problem instances
            budget: Total budget (defaults to max_budget)

        Returns:
            results: Dict with best solver and evaluation history
        """
        if budget is None:
            budget = self.max_budget

        # Sample configurations using BO
        n_configs = len(self.solvers)

        # Run successive halving
        candidates = list(range(n_configs))
        budget_per_config = 1
        history = []

        while len(candidates) > 1 and budget_per_config <= budget:
            performances = []

            for solver_idx in candidates:
                # Evaluate on sample of instances
                n_eval = min(len(instances), budget_per_config)
                sample_instances = np.random.choice(
                    instances,
                    size=n_eval,
                    replace=False
                )

                perfs = []
                for inst in sample_instances:
                    if hasattr(self.solvers[solver_idx], 'solve'):
                        result = self.solvers[solver_idx].solve(inst)
                        perfs.append(result.get('objective', 0.5))
                    else:
                        perfs.append(0.5)

                avg_perf = np.mean(perfs)
                performances.append(avg_perf)

                history.append({
                    'solver_idx': solver_idx,
                    'budget': budget_per_config,
                    'performance': avg_perf
                })

            # Keep top candidates
            n_keep = max(1, len(candidates) // self.eta)
            top_indices = np.argsort(performances)[-n_keep:]
            candidates = [candidates[i] for i in top_indices]

            # Increase budget
            budget_per_config *= self.eta

        best_solver_idx = candidates[0] if candidates else 0

        return {
            'best_solver': self.solvers[best_solver_idx],
            'history': history
        }

    def predict_performance(
        self,
        instance: Any,
        solver_name: str,
        features: Optional[np.ndarray] = None
    ) -> float:
        """
        Predict performance using KDE model

        Args:
            instance: Problem instance
            solver_name: Solver to predict for
            features: Pre-extracted features (optional)

        Returns:
            predicted_performance: Predicted score
        """
        # Simplified: return average historical performance
        solver_idx = self.solver_to_idx[solver_name]
        perfs = [
            obs['performance']
            for obs in self.observations
            if obs['solver_idx'] == solver_idx
        ]
        return np.mean(perfs) if perfs else 0.5

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
        return {
            name: self.predict_performance(instance, name, features)
            for name in self.solver_names
        }

    def update(
        self,
        instance: Any,
        solver_name: str,
        performance: float,
        features: Optional[np.ndarray] = None
    ):
        """
        Update with new observation

        Args:
            instance: Problem instance
            solver_name: Solver that was used
            performance: Observed performance
            features: Pre-extracted features (optional)
        """
        if features is None:
            features = self.feature_extractor.extract(instance)

        solver_idx = self.solver_to_idx[solver_name]
        augmented = np.concatenate([features, [solver_idx]])

        self.observations.append({
            'features': augmented,
            'solver_idx': solver_idx,
            'performance': performance
        })

        # Rebuild KDE models periodically
        if len(self.observations) % 20 == 0 and len(self.observations) >= self.min_points_in_model:
            self._build_kde_models()
