"""
Librex.Meta - Tournament-based solver selection using Swiss-system framework

Key Components:
1. Feature Extractor: Extracts problem instance features
2. Elo Tracker: Maintains solver ratings (global + per-cluster)
3. Tournament Manager: Runs Swiss-system rounds
4. Selection Policy: UCB-based selection with Elo priors
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
import copy
from .feature_extractor import FeatureExtractor


class Librex.Meta:
    """
    Tournament-based solver selection using Swiss-system framework

    Key Components:
    1. Feature Extractor: Extracts problem instance features
    2. Elo Tracker: Maintains solver ratings (global + per-cluster)
    3. Tournament Manager: Runs Swiss-system rounds
    4. Selection Policy: UCB-based selection with Elo priors
    """

    def __init__(
        self,
        solvers: List[Any],
        n_clusters: int = 5,
        elo_k: float = 32.0,
        ucb_c: float = 1.414,
        n_tournament_rounds: int = 5
    ):
        super().__init__()
        self.solvers = solvers
        self.n_clusters = n_clusters
        self.elo_k = elo_k
        self.ucb_c = ucb_c
        self.n_tournament_rounds = n_tournament_rounds

        # Feature extractor (train on historical data)
        self.feature_extractor = FeatureExtractor()

        # Cluster problem space
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        self.clusterer_fitted = False

        # Elo ratings: global + per-cluster
        self.global_elo = {s.name if hasattr(s, 'name') else str(s): 1500.0 for s in solvers}
        self.cluster_elo = {
            cluster: {s.name if hasattr(s, 'name') else str(s): 1500.0 for s in solvers}
            for cluster in range(n_clusters)
        }

        # Performance history
        self.history = []

        # Solver name mapping
        self.solver_names = [s.name if hasattr(s, 'name') else str(s) for s in solvers]

    @property
    def name(self) -> str:
        return "Librex.Meta"

    def _ensure_clusterer_fitted(self):
        """Ensure clusterer is fitted, using default cluster if not"""
        if not self.clusterer_fitted:
            # If not fitted, use dummy data to fit the clusterer
            # This allows methods to work without requiring fit() to be called first
            dummy_features = np.random.rand(self.n_clusters * 2, 10)
            self.clusterer.fit(dummy_features)
            self.clusterer_fitted = True

    def fit(self, training_data: List[Dict]):
        """
        Train Librex.Meta on historical solver performance data

        Args:
            training_data: List of dicts with keys:
                - 'instance': problem instance
                - 'features': instance features (or None to extract)
                - 'performances': {solver_name: performance_score}
        """
        print(f"Training Librex.Meta on {len(training_data)} historical instances...")

        # Extract features
        features_list = []
        for data in training_data:
            if data.get('features') is not None:
                features = data['features']
            else:
                features = self.feature_extractor.extract(data['instance'])
            features_list.append(features)

        X = np.array(features_list)

        # Cluster problem space
        self.clusterer.fit(X)
        self.clusterer_fitted = True
        print(f"  Clustered problem space into {self.n_clusters} clusters")

        # Run tournament simulation to initialize Elo ratings
        for data, features in zip(training_data, features_list):
            cluster = self.clusterer.predict([features])[0]
            performances = data['performances']

            # Simulate pairwise matches
            solver_names = list(performances.keys())
            for i, s1 in enumerate(solver_names):
                for s2 in solver_names[i+1:]:
                    # Determine winner
                    if performances[s1] > performances[s2]:
                        self._update_elo(s1, s2, cluster, outcome=1.0)
                    elif performances[s1] < performances[s2]:
                        self._update_elo(s1, s2, cluster, outcome=0.0)
                    else:
                        self._update_elo(s1, s2, cluster, outcome=0.5)

        print(f"✓ Elo ratings initialized from {len(training_data)} instances")
        print(f"  Global Elo ratings: {self.global_elo}")

    def select_solver(
        self,
        instance: Any,
        features: Optional[np.ndarray] = None
    ) -> Any:
        """
        Select best solver for given problem instance

        Args:
            instance: Problem instance
            features: Pre-extracted features (optional)

        Returns:
            selected_solver: Best solver for this instance
        """
        # Extract features
        if features is None:
            features = self.feature_extractor.extract(instance)

        # Ensure clusterer is fitted
        self._ensure_clusterer_fitted()

        # Identify cluster
        cluster = self.clusterer.predict([features])[0]

        # Get cluster-specific Elo ratings
        ratings = self.cluster_elo[cluster]

        # Compute UCB scores
        ucb_scores = {}
        for solver in self.solvers:
            name = solver.name if hasattr(solver, 'name') else str(solver)
            elo = ratings[name]
            n_trials = self._count_trials(name, cluster)

            # UCB formula with Elo normalization
            exploitation = (elo - 1500) / 400  # Normalize Elo to ~[-1, 1]
            exploration = self.ucb_c * np.sqrt(
                np.log(sum(self._count_trials(s.name if hasattr(s, 'name') else str(s), cluster) for s in self.solvers) + 1)
                / (n_trials + 1)
            )
            ucb_scores[name] = exploitation + exploration

        # Select solver with highest UCB
        best_solver_name = max(ucb_scores, key=ucb_scores.get)
        selected_solver = next(s for s in self.solvers if (s.name if hasattr(s, 'name') else str(s)) == best_solver_name)

        return selected_solver

    def run_tournament(
        self,
        instance: Any,
        features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Run full Swiss-system tournament to select best solver

        Args:
            instance: Problem instance
            features: Pre-extracted features (optional)

        Returns:
            result: Dict with 'selected_solver', 'rankings', 'performances'
        """
        if features is None:
            features = self.feature_extractor.extract(instance)

        # Ensure clusterer is fitted
        self._ensure_clusterer_fitted()

        cluster = self.clusterer.predict([features])[0]

        # Initialize tournament
        tournament_elo = {name: self.cluster_elo[cluster][name] for name in self.solver_names}
        match_history = []

        # Run Swiss-system rounds
        for round_num in range(self.n_tournament_rounds):
            # Pair solvers by similar Elo
            sorted_solvers = sorted(
                self.solvers,
                key=lambda s: tournament_elo[s.name if hasattr(s, 'name') else str(s)],
                reverse=True
            )

            pairs = []
            used = set()
            for i in range(0, len(sorted_solvers) - 1, 2):
                s1 = sorted_solvers[i]
                s2 = sorted_solvers[i + 1]
                s1_name = s1.name if hasattr(s1, 'name') else str(s1)
                s2_name = s2.name if hasattr(s2, 'name') else str(s2)
                if s1_name not in used and s2_name not in used:
                    pairs.append((s1, s2))
                    used.add(s1_name)
                    used.add(s2_name)

            # Run matches
            for s1, s2 in pairs:
                s1_name = s1.name if hasattr(s1, 'name') else str(s1)
                s2_name = s2.name if hasattr(s2, 'name') else str(s2)

                # Execute both solvers (if they have solve method)
                if hasattr(s1, 'solve'):
                    perf1 = s1.solve(instance).get('objective', 0.5)
                else:
                    perf1 = 0.5  # Mock performance

                if hasattr(s2, 'solve'):
                    perf2 = s2.solve(instance).get('objective', 0.5)
                else:
                    perf2 = 0.5  # Mock performance

                # Determine winner
                if perf1 > perf2:
                    outcome = 1.0
                elif perf1 < perf2:
                    outcome = 0.0
                else:
                    outcome = 0.5

                # Update tournament Elo
                expected = 1 / (1 + 10 ** ((tournament_elo[s2_name] - tournament_elo[s1_name]) / 400))
                tournament_elo[s1_name] += self.elo_k * (outcome - expected)
                tournament_elo[s2_name] += self.elo_k * ((1 - outcome) - (1 - expected))

                match_history.append({
                    'round': round_num,
                    's1': s1_name,
                    's2': s2_name,
                    'perf1': perf1,
                    'perf2': perf2,
                    'outcome': outcome
                })

        # Select winner
        winner_name = max(tournament_elo, key=tournament_elo.get)
        selected_solver = next(s for s in self.solvers if (s.name if hasattr(s, 'name') else str(s)) == winner_name)

        return {
            'selected_solver': selected_solver,
            'rankings': sorted(tournament_elo.items(), key=lambda x: x[1], reverse=True),
            'match_history': match_history
        }

    def update(
        self,
        instance: Any,
        solver_name: str,
        performance: float,
        features: Optional[np.ndarray] = None
    ):
        """
        Update Elo ratings after observing solver performance

        Args:
            instance: Problem instance
            solver_name: Name of solver that was used
            performance: Performance score achieved
            features: Pre-extracted features (optional)
        """
        if features is None:
            features = self.feature_extractor.extract(instance)

        # Ensure clusterer is fitted
        self._ensure_clusterer_fitted()

        cluster = self.clusterer.predict([features])[0]

        # Store in history
        self.history.append({
            'features': features,
            'cluster': cluster,
            'solver': solver_name,
            'performance': performance
        })

        # Update Elo by simulating matches against recent history
        recent_history = [
            h for h in self.history[-100:]
            if h['cluster'] == cluster and h['solver'] != solver_name
        ]

        for h in recent_history:
            if performance > h['performance']:
                outcome = 1.0
            elif performance < h['performance']:
                outcome = 0.0
            else:
                outcome = 0.5

            self._update_elo(solver_name, h['solver'], cluster, outcome)

    def _update_elo(
        self,
        solver1: str,
        solver2: str,
        cluster: int,
        outcome: float
    ):
        """Update Elo ratings for solver1 vs solver2"""
        # Cluster-specific update
        r1 = self.cluster_elo[cluster][solver1]
        r2 = self.cluster_elo[cluster][solver2]
        expected = 1 / (1 + 10 ** ((r2 - r1) / 400))
        self.cluster_elo[cluster][solver1] += self.elo_k * (outcome - expected)
        self.cluster_elo[cluster][solver2] += self.elo_k * ((1 - outcome) - (1 - expected))

        # Global update
        r1_global = self.global_elo[solver1]
        r2_global = self.global_elo[solver2]
        expected_global = 1 / (1 + 10 ** ((r2_global - r1_global) / 400))
        self.global_elo[solver1] += self.elo_k * (outcome - expected_global)
        self.global_elo[solver2] += self.elo_k * ((1 - outcome) - (1 - expected_global))

    def _count_trials(self, solver_name: str, cluster: int) -> int:
        """Count number of trials for solver in cluster"""
        return sum(
            1 for h in self.history
            if h['solver'] == solver_name and h['cluster'] == cluster
        )


## FeatureExtractor moved to libria_meta.feature_extractor
