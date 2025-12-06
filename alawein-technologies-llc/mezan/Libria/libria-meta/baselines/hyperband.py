"""
Hyperband - Bandit-Based Hyperparameter Optimization

Reference:
  Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017).
  Hyperband: A novel bandit-based approach to hyperparameter optimization.
  Journal of Machine Learning Research, 18(185), 1-52.

Key Ideas:
- Successive halving: allocate resources adaptively
- Multiple brackets with different trade-offs
- Efficient exploration of solver configurations
- Eliminates poorly performing solvers early
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from libria_meta.feature_extractor import FeatureExtractor
import math


class Hyperband:
    """
    Hyperband-style successive halving for algorithm selection

    Adapts Hyperband to algorithm selection by treating solvers
    as configurations and using successive halving to allocate
    evaluation budget.
    """

    def __init__(
        self,
        solvers: List[Any],
        max_budget: int = 27,
        eta: int = 3,
        n_brackets: int = 4
    ):
        """
        Initialize Hyperband

        Args:
            solvers: List of solver instances
            max_budget: Maximum budget per solver
            eta: Downsampling rate (typically 3 or 4)
            n_brackets: Number of brackets (different exploration strategies)
        """
        self.solvers = solvers
        self.max_budget = max_budget
        self.eta = eta
        self.n_brackets = n_brackets

        # Feature processing
        self.feature_extractor = FeatureExtractor()

        # Performance tracking
        self.performance_history = {
            s.name if hasattr(s, 'name') else str(s): []
            for s in solvers
        }

        # Current best solver per instance cluster
        self.best_solver_per_cluster = {}

        self.solver_names = [s.name if hasattr(s, 'name') else str(s) for s in solvers]

    @property
    def name(self) -> str:
        return "Hyperband"

    def fit(self, training_data: List[Dict]):
        """
        Train Hyperband on historical data

        Args:
            training_data: List of dicts with:
                - 'instance': problem instance
                - 'features': instance features (or None to extract)
                - 'performances': {solver_name: performance_score}
        """
        print(f"Training Hyperband on {len(training_data)} instances...")

        # Run successive halving for each bracket
        for bracket in range(self.n_brackets):
            print(f"\n  Bracket {bracket + 1}/{self.n_brackets}:")

            # Calculate bracket parameters
            n_configs = len(self.solvers)
            r = self.max_budget // (self.eta ** bracket)  # Initial budget
            n_iterations = bracket + 1

            # Successive halving
            survivors = list(range(n_configs))

            for iteration in range(n_iterations):
                # Budget for this iteration
                budget = r * (self.eta ** iteration)

                # Evaluate survivors
                performances = []
                for solver_idx in survivors:
                    solver_name = self.solver_names[solver_idx]

                    # Collect performance from training data
                    perf_scores = []
                    for data in training_data[:int(budget)]:
                        if solver_name in data['performances']:
                            perf_scores.append(data['performances'][solver_name])

                    avg_perf = np.mean(perf_scores) if perf_scores else 0.5
                    performances.append(avg_perf)

                # Keep top solvers
                n_keep = max(1, len(survivors) // self.eta)
                top_indices = np.argsort(performances)[-n_keep:]
                survivors = [survivors[i] for i in top_indices]

                print(f"    Iteration {iteration + 1}: {len(survivors)} survivors")

        # Store final performance statistics
        for data in training_data:
            for solver_name, perf in data['performances'].items():
                if solver_name in self.performance_history:
                    self.performance_history[solver_name].append(perf)

        print("\n✓ Hyperband training complete")

    def select_solver(
        self,
        instance: Any,
        features: Optional[np.ndarray] = None
    ) -> Any:
        """
        Select best solver using Hyperband strategy

        For selection, we use a simplified greedy approach
        based on average historical performance.

        Args:
            instance: Problem instance
            features: Pre-extracted features (optional)

        Returns:
            selected_solver: Best solver
        """
        # Calculate average performance per solver
        avg_performances = {}
        for solver_name, perfs in self.performance_history.items():
            if perfs:
                avg_performances[solver_name] = np.mean(perfs)
            else:
                avg_performances[solver_name] = 0.5

        # Select solver with best average performance
        best_solver_name = max(avg_performances, key=avg_performances.get)

        selected_solver = next(
            s for s in self.solvers
            if (s.name if hasattr(s, 'name') else str(s)) == best_solver_name
        )

        return selected_solver

    def run_successive_halving(
        self,
        instance: Any,
        budget: int = None
    ) -> Tuple[Any, List[Dict]]:
        """
        Run successive halving on instance to find best solver

        Args:
            instance: Problem instance
            budget: Total budget (defaults to max_budget)

        Returns:
            best_solver: Selected solver
            history: Evaluation history
        """
        if budget is None:
            budget = self.max_budget

        history = []
        candidates = list(self.solvers)

        # Initial budget per solver
        budget_per_solver = 1
        total_spent = 0

        while len(candidates) > 1 and total_spent < budget:
            # Evaluate each candidate
            performances = []

            for solver in candidates:
                # Simulate evaluation (in practice, would run solver)
                if hasattr(solver, 'solve'):
                    result = solver.solve(instance)
                    perf = result.get('objective', 0.5)
                else:
                    perf = 0.5

                performances.append(perf)

                history.append({
                    'solver': solver.name if hasattr(solver, 'name') else str(solver),
                    'budget': budget_per_solver,
                    'performance': perf
                })

                total_spent += budget_per_solver

                if total_spent >= budget:
                    break

            # Keep top half
            n_keep = max(1, len(candidates) // self.eta)
            top_indices = np.argsort(performances)[-n_keep:]
            candidates = [candidates[i] for i in top_indices]

            # Increase budget for next round
            budget_per_solver *= self.eta

        # Return best candidate
        best_solver = candidates[0] if candidates else self.solvers[0]

        return best_solver, history

    def predict_performance(
        self,
        instance: Any,
        solver_name: str,
        features: Optional[np.ndarray] = None
    ) -> float:
        """
        Predict performance (returns average historical performance)

        Args:
            instance: Problem instance
            solver_name: Solver to predict for
            features: Pre-extracted features (optional)

        Returns:
            predicted_performance: Average historical performance
        """
        if solver_name in self.performance_history:
            perfs = self.performance_history[solver_name]
            return np.mean(perfs) if perfs else 0.5
        return 0.5

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
        Update performance history

        Args:
            instance: Problem instance
            solver_name: Solver that was used
            performance: Observed performance
            features: Pre-extracted features (optional)
        """
        if solver_name in self.performance_history:
            self.performance_history[solver_name].append(performance)
