"""
Active Learning Experiment Designer for TalAI
==============================================

Advanced system for optimal experimental design using Bayesian optimization,
information theory, and multi-armed bandit algorithms.

References:
- Settles, B. (2009). Active Learning Literature Survey. Computer Sciences Technical Report.
- Srinivas, N., et al. (2010). Gaussian Process Optimization in the Bandit Setting. ICML.
- Shahriari, B., et al. (2015). Taking the Human Out of the Loop: A Review of Bayesian Optimization.
- Hernández-Lobato, J. M., et al. (2014). Predictive Entropy Search for Efficient Global Optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats, optimize, special
from scipy.stats import norm, multivariate_normal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import warnings
import logging
from collections import defaultdict, deque
import json
from abc import ABC, abstractmethod
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AcquisitionFunction(Enum):
    """Available acquisition functions for Bayesian optimization."""
    EI = "expected_improvement"
    UCB = "upper_confidence_bound"
    PI = "probability_improvement"
    ENTROPY = "entropy_search"
    KNOWLEDGE_GRADIENT = "knowledge_gradient"
    MULTI_TASK = "multi_task_acquisition"
    COST_AWARE_EI = "cost_aware_expected_improvement"


class ExperimentType(Enum):
    """Types of experiments that can be designed."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    OPTIMIZATION = "optimization"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    CAUSAL_DISCOVERY = "causal_discovery"
    DOSE_RESPONSE = "dose_response"


class SamplingStrategy(Enum):
    """Strategies for selecting experiments."""
    UNCERTAINTY = "uncertainty_sampling"
    DIVERSITY = "diversity_sampling"
    REPRESENTATIVE = "representative_sampling"
    ADVERSARIAL = "adversarial_sampling"
    HYBRID = "hybrid_sampling"


@dataclass
class Experiment:
    """Represents a single experiment with conditions and outcomes."""
    exp_id: str
    parameters: Dict[str, float]  # Experimental conditions
    outcome: Optional[float] = None  # Measured result
    uncertainty: Optional[float] = None  # Measurement uncertainty
    cost: float = 1.0  # Cost to run experiment
    duration_hours: float = 1.0  # Time to complete
    batch_id: Optional[int] = None  # For batch experiments
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None

    def is_completed(self) -> bool:
        """Check if experiment has been completed."""
        return self.outcome is not None

    def information_gain(self) -> float:
        """Calculate information gained from this experiment."""
        if self.uncertainty is None or self.outcome is None:
            return 0.0
        # Information gain approximated by reduction in uncertainty
        return -np.log(self.uncertainty + 1e-10)


@dataclass
class ExperimentBatch:
    """Represents a batch of experiments that can be run in parallel."""
    batch_id: int
    experiments: List[Experiment]
    total_cost: float
    max_duration: float
    expected_information_gain: float

    def add_experiment(self, exp: Experiment):
        """Add experiment to batch."""
        exp.batch_id = self.batch_id
        self.experiments.append(exp)
        self.total_cost += exp.cost
        self.max_duration = max(self.max_duration, exp.duration_hours)


@dataclass
class OptimizationObjective:
    """Defines the objective for optimization experiments."""
    name: str
    target_function: Optional[Callable] = None
    minimize: bool = True
    constraints: List[Callable] = field(default_factory=list)
    parameter_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    noise_level: float = 0.01
    optimal_value: Optional[float] = None


class BayesianOptimizer:
    """
    Bayesian optimization for experimental design.

    Uses Gaussian Processes to model the objective function and
    acquisition functions to select next experiments.
    """

    def __init__(self,
                 objective: OptimizationObjective,
                 acquisition: AcquisitionFunction = AcquisitionFunction.EI,
                 kernel=None,
                 random_state: int = 42):
        """Initialize Bayesian optimizer."""
        self.objective = objective
        self.acquisition = acquisition
        self.random_state = random_state
        np.random.seed(random_state)

        # Setup Gaussian Process
        if kernel is None:
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=objective.noise_level,
            n_restarts_optimizer=10,
            random_state=random_state
        )

        self.X_observed = []
        self.y_observed = []
        self.best_value = float('inf') if objective.minimize else float('-inf')
        self.best_params = None

    def suggest_next_experiment(self,
                               n_suggestions: int = 1,
                               pending_experiments: List[Dict[str, float]] = None) -> List[Dict[str, float]]:
        """
        Suggest next experiment(s) to run using acquisition function.

        Implements parallel Bayesian optimization for batch suggestions.
        """
        if len(self.X_observed) < 2:
            # Initial random exploration
            return self._random_sampling(n_suggestions)

        # Fit GP to observed data
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp.fit(X, y)

        suggestions = []

        for _ in range(n_suggestions):
            # Account for pending experiments (fantasy samples)
            if pending_experiments:
                X_pending = self._dict_to_array(pending_experiments)
                # Use kriging believer strategy
                y_pending_mean, _ = self.gp.predict(X_pending, return_std=True)
                X_aug = np.vstack([X, X_pending])
                y_aug = np.hstack([y, y_pending_mean])
                self.gp.fit(X_aug, y_aug)

            # Optimize acquisition function
            best_x = self._optimize_acquisition()
            suggestions.append(self._array_to_dict(best_x))

            # Add as pending for next iteration
            if pending_experiments is None:
                pending_experiments = []
            pending_experiments.append(self._array_to_dict(best_x))

        return suggestions

    def _optimize_acquisition(self) -> np.ndarray:
        """Optimize the acquisition function to find next point."""
        bounds = self._get_bounds_array()

        # Multi-start optimization
        best_value = float('-inf')
        best_x = None

        for _ in range(10):  # 10 random starts
            x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])

            result = optimize.minimize(
                lambda x: -self._acquisition_function(x.reshape(1, -1)),
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )

            if -result.fun > best_value:
                best_value = -result.fun
                best_x = result.x

        return best_x

    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """Calculate acquisition function value."""
        mean, std = self.gp.predict(X, return_std=True)

        if self.acquisition == AcquisitionFunction.EI:
            return self._expected_improvement(mean, std)
        elif self.acquisition == AcquisitionFunction.UCB:
            return self._upper_confidence_bound(mean, std)
        elif self.acquisition == AcquisitionFunction.PI:
            return self._probability_improvement(mean, std)
        else:
            return self._expected_improvement(mean, std)

    def _expected_improvement(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Expected Improvement acquisition function."""
        if self.objective.minimize:
            incumbent = np.min(self.y_observed)
            improvement = incumbent - mean
        else:
            incumbent = np.max(self.y_observed)
            improvement = mean - incumbent

        Z = improvement / (std + 1e-10)
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)

        return ei

    def _upper_confidence_bound(self, mean: np.ndarray, std: np.ndarray, beta: float = 2.0) -> np.ndarray:
        """Upper Confidence Bound acquisition function."""
        if self.objective.minimize:
            return -(mean - beta * std)
        else:
            return mean + beta * std

    def _probability_improvement(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Probability of Improvement acquisition function."""
        if self.objective.minimize:
            incumbent = np.min(self.y_observed)
            Z = (incumbent - mean) / (std + 1e-10)
        else:
            incumbent = np.max(self.y_observed)
            Z = (mean - incumbent) / (std + 1e-10)

        return norm.cdf(Z)

    def update_observations(self, experiments: List[Experiment]):
        """Update model with new experimental results."""
        for exp in experiments:
            if exp.is_completed():
                x = self._dict_to_array([exp.parameters])
                self.X_observed.append(x[0])
                self.y_observed.append(exp.outcome)

                # Update best observed value
                if self.objective.minimize:
                    if exp.outcome < self.best_value:
                        self.best_value = exp.outcome
                        self.best_params = exp.parameters
                else:
                    if exp.outcome > self.best_value:
                        self.best_value = exp.outcome
                        self.best_params = exp.parameters

    def _dict_to_array(self, param_dicts: List[Dict[str, float]]) -> np.ndarray:
        """Convert parameter dictionaries to numpy array."""
        param_names = sorted(self.objective.parameter_bounds.keys())
        array = np.zeros((len(param_dicts), len(param_names)))

        for i, params in enumerate(param_dicts):
            for j, name in enumerate(param_names):
                array[i, j] = params.get(name, 0.0)

        return array

    def _array_to_dict(self, array: np.ndarray) -> Dict[str, float]:
        """Convert numpy array to parameter dictionary."""
        param_names = sorted(self.objective.parameter_bounds.keys())
        return {name: float(array[i]) for i, name in enumerate(param_names)}

    def _get_bounds_array(self) -> np.ndarray:
        """Get parameter bounds as numpy array."""
        param_names = sorted(self.objective.parameter_bounds.keys())
        bounds = np.array([self.objective.parameter_bounds[name] for name in param_names])
        return bounds

    def _random_sampling(self, n_samples: int) -> List[Dict[str, float]]:
        """Random sampling within parameter bounds."""
        suggestions = []
        bounds = self._get_bounds_array()

        for _ in range(n_samples):
            x = np.random.uniform(bounds[:, 0], bounds[:, 1])
            suggestions.append(self._array_to_dict(x))

        return suggestions


class ThompsonSampling:
    """
    Thompson Sampling for multi-armed bandit experimental selection.

    Particularly useful for hypothesis selection and A/B testing.
    """

    def __init__(self, n_arms: int, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """Initialize Thompson Sampling."""
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms) * prior_alpha  # Success counts
        self.beta = np.ones(n_arms) * prior_beta    # Failure counts
        self.arm_counts = np.zeros(n_arms)
        self.arm_rewards = np.zeros(n_arms)
        self.history = []

    def select_arm(self, n_selections: int = 1) -> List[int]:
        """Select arm(s) to pull using Thompson Sampling."""
        # Sample from Beta distributions
        theta = np.random.beta(self.alpha, self.beta)

        # Select top n_selections arms
        selected = np.argsort(theta)[-n_selections:].tolist()

        return selected

    def update(self, arm: int, reward: float):
        """Update posterior after observing reward."""
        # Update counts
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward

        # Update Beta parameters (assuming Bernoulli rewards)
        if reward > 0:
            self.alpha[arm] += reward
        else:
            self.beta[arm] += (1 - reward)

        self.history.append({'arm': arm, 'reward': reward})

    def get_arm_statistics(self) -> Dict[int, Dict[str, float]]:
        """Get statistics for each arm."""
        stats = {}

        for arm in range(self.n_arms):
            if self.arm_counts[arm] > 0:
                mean_reward = self.arm_rewards[arm] / self.arm_counts[arm]
            else:
                mean_reward = 0.0

            # Posterior mean and variance
            posterior_mean = self.alpha[arm] / (self.alpha[arm] + self.beta[arm])
            posterior_var = (self.alpha[arm] * self.beta[arm]) / \
                          ((self.alpha[arm] + self.beta[arm])**2 * (self.alpha[arm] + self.beta[arm] + 1))

            stats[arm] = {
                'pulls': int(self.arm_counts[arm]),
                'empirical_mean': mean_reward,
                'posterior_mean': posterior_mean,
                'posterior_std': np.sqrt(posterior_var),
                'confidence_interval': (
                    np.percentile(np.random.beta(self.alpha[arm], self.beta[arm], 10000), 2.5),
                    np.percentile(np.random.beta(self.alpha[arm], self.beta[arm], 10000), 97.5)
                )
            }

        return stats


class ActiveLearningEngine:
    """
    Main engine for active learning and adaptive experimental design.

    Implements multiple strategies for optimal experiment selection to
    maximize information gain while minimizing costs.
    """

    def __init__(self,
                 experiment_type: ExperimentType = ExperimentType.OPTIMIZATION,
                 random_state: int = 42):
        """Initialize active learning engine."""
        self.experiment_type = experiment_type
        self.random_state = random_state
        np.random.seed(random_state)

        self.completed_experiments = []
        self.pending_experiments = []
        self.failed_experiments = []
        self.total_cost = 0.0
        self.total_time = 0.0

        # Initialize optimizers
        self.bayesian_optimizer = None
        self.thompson_sampler = None

        # Information tracking
        self.information_history = []
        self.model_uncertainty = float('inf')

    def initialize_optimization(self, objective: OptimizationObjective,
                              acquisition: AcquisitionFunction = AcquisitionFunction.EI):
        """Initialize Bayesian optimization for experiment selection."""
        self.bayesian_optimizer = BayesianOptimizer(
            objective=objective,
            acquisition=acquisition,
            random_state=self.random_state
        )

    def initialize_hypothesis_testing(self, n_hypotheses: int):
        """Initialize Thompson Sampling for hypothesis testing."""
        self.thompson_sampler = ThompsonSampling(
            n_arms=n_hypotheses,
            prior_alpha=1.0,
            prior_beta=1.0
        )

    def suggest_experiments(self,
                          budget: float,
                          batch_size: int = 1,
                          strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY,
                          cost_function: Optional[Callable] = None) -> ExperimentBatch:
        """
        Suggest next batch of experiments within budget constraints.

        Implements cost-aware experimental design.
        """
        if self.experiment_type == ExperimentType.OPTIMIZATION and self.bayesian_optimizer:
            return self._suggest_optimization_experiments(budget, batch_size, cost_function)
        elif self.experiment_type == ExperimentType.HYPOTHESIS_TESTING and self.thompson_sampler:
            return self._suggest_hypothesis_experiments(budget, batch_size)
        else:
            return self._suggest_general_experiments(budget, batch_size, strategy)

    def _suggest_optimization_experiments(self,
                                         budget: float,
                                         batch_size: int,
                                         cost_function: Optional[Callable]) -> ExperimentBatch:
        """Suggest experiments for optimization using Bayesian optimization."""
        batch = ExperimentBatch(
            batch_id=len(self.information_history),
            experiments=[],
            total_cost=0.0,
            max_duration=0.0,
            expected_information_gain=0.0
        )

        # Get parameter suggestions
        pending_params = [exp.parameters for exp in self.pending_experiments]
        suggestions = self.bayesian_optimizer.suggest_next_experiment(
            n_suggestions=batch_size,
            pending_experiments=pending_params
        )

        for i, params in enumerate(suggestions):
            # Calculate cost
            if cost_function:
                cost = cost_function(params)
            else:
                cost = 1.0

            if batch.total_cost + cost > budget:
                break

            # Create experiment
            exp = Experiment(
                exp_id=f"exp_{len(self.completed_experiments) + len(self.pending_experiments) + i}",
                parameters=params,
                cost=cost,
                duration_hours=self._estimate_duration(params)
            )

            batch.add_experiment(exp)

            # Estimate information gain
            if len(self.completed_experiments) > 0:
                batch.expected_information_gain += self._estimate_information_gain(params)

        return batch

    def _suggest_hypothesis_experiments(self,
                                       budget: float,
                                       batch_size: int) -> ExperimentBatch:
        """Suggest experiments for hypothesis testing using Thompson Sampling."""
        batch = ExperimentBatch(
            batch_id=len(self.information_history),
            experiments=[],
            total_cost=0.0,
            max_duration=0.0,
            expected_information_gain=0.0
        )

        # Select hypotheses to test
        selected_arms = self.thompson_sampler.select_arm(n_selections=batch_size)

        for arm in selected_arms:
            if batch.total_cost + 1.0 > budget:
                break

            exp = Experiment(
                exp_id=f"hypothesis_test_{arm}_{self.thompson_sampler.arm_counts[arm]}",
                parameters={'hypothesis_id': arm},
                cost=1.0,
                duration_hours=1.0
            )

            batch.add_experiment(exp)

            # Estimate information gain from reducing uncertainty
            stats = self.thompson_sampler.get_arm_statistics()
            if arm in stats:
                batch.expected_information_gain += stats[arm]['posterior_std']

        return batch

    def _suggest_general_experiments(self,
                                    budget: float,
                                    batch_size: int,
                                    strategy: SamplingStrategy) -> ExperimentBatch:
        """Suggest experiments using general active learning strategies."""
        batch = ExperimentBatch(
            batch_id=len(self.information_history),
            experiments=[],
            total_cost=0.0,
            max_duration=0.0,
            expected_information_gain=0.0
        )

        if strategy == SamplingStrategy.UNCERTAINTY:
            candidates = self._uncertainty_sampling(batch_size * 3)
        elif strategy == SamplingStrategy.DIVERSITY:
            candidates = self._diversity_sampling(batch_size * 3)
        elif strategy == SamplingStrategy.REPRESENTATIVE:
            candidates = self._representative_sampling(batch_size * 3)
        else:
            candidates = self._hybrid_sampling(batch_size * 3)

        # Select experiments within budget
        for params in candidates[:batch_size]:
            if batch.total_cost + 1.0 > budget:
                break

            exp = Experiment(
                exp_id=f"exp_{len(self.completed_experiments) + len(batch.experiments)}",
                parameters=params,
                cost=1.0,
                duration_hours=1.0
            )

            batch.add_experiment(exp)

        return batch

    def _uncertainty_sampling(self, n_candidates: int) -> List[Dict[str, float]]:
        """
        Select experiments with maximum uncertainty.

        Most informative where model is least certain.
        """
        if not self.bayesian_optimizer or len(self.completed_experiments) < 2:
            # Random sampling initially
            return self._random_sampling(n_candidates)

        # Generate candidate pool
        candidates = self._random_sampling(n_candidates * 10)

        # Predict with uncertainty
        X = self.bayesian_optimizer._dict_to_array(candidates)
        _, std = self.bayesian_optimizer.gp.predict(X, return_std=True)

        # Select highest uncertainty
        selected_idx = np.argsort(std)[-n_candidates:]

        return [candidates[i] for i in selected_idx]

    def _diversity_sampling(self, n_candidates: int) -> List[Dict[str, float]]:
        """
        Select diverse experiments to explore parameter space.

        Uses furthest-first traversal algorithm.
        """
        if len(self.completed_experiments) == 0:
            return self._random_sampling(n_candidates)

        # Get completed experiment parameters
        completed_params = [exp.parameters for exp in self.completed_experiments]

        # Generate large candidate pool
        candidate_pool = self._random_sampling(n_candidates * 20)

        selected = []
        remaining = candidate_pool.copy()

        # Start with furthest from all completed
        if completed_params:
            X_completed = self._params_to_array(completed_params)
            X_candidates = self._params_to_array(remaining)

            distances = pairwise_distances(X_candidates, X_completed).min(axis=1)
            furthest_idx = np.argmax(distances)

            selected.append(remaining[furthest_idx])
            remaining.pop(furthest_idx)

        # Iteratively select furthest from selected set
        while len(selected) < n_candidates and remaining:
            X_selected = self._params_to_array(selected)
            X_remaining = self._params_to_array(remaining)

            distances = pairwise_distances(X_remaining, X_selected).min(axis=1)
            furthest_idx = np.argmax(distances)

            selected.append(remaining[furthest_idx])
            remaining.pop(furthest_idx)

        return selected

    def _representative_sampling(self, n_candidates: int) -> List[Dict[str, float]]:
        """
        Select representative experiments using clustering.

        Ensures coverage of parameter space regions.
        """
        # Generate candidate pool
        candidate_pool = self._random_sampling(n_candidates * 20)
        X_candidates = self._params_to_array(candidate_pool)

        # Cluster candidates
        kmeans = KMeans(n_clusters=min(n_candidates, len(candidate_pool)), random_state=self.random_state)
        kmeans.fit(X_candidates)

        # Select closest to each cluster center
        selected = []
        for center in kmeans.cluster_centers_:
            distances = pairwise_distances(X_candidates, center.reshape(1, -1)).flatten()
            closest_idx = np.argmin(distances)
            selected.append(candidate_pool[closest_idx])

        return selected[:n_candidates]

    def _hybrid_sampling(self, n_candidates: int) -> List[Dict[str, float]]:
        """
        Hybrid strategy combining uncertainty and diversity.

        Balances exploration and exploitation.
        """
        n_uncertain = n_candidates // 2
        n_diverse = n_candidates - n_uncertain

        uncertain = self._uncertainty_sampling(n_uncertain)
        diverse = self._diversity_sampling(n_diverse)

        return uncertain + diverse

    def _random_sampling(self, n_samples: int) -> List[Dict[str, float]]:
        """Random sampling for initial exploration."""
        samples = []

        # Simple random sampling in unit hypercube
        for _ in range(n_samples):
            params = {}
            n_dims = 5  # Default dimensionality

            for i in range(n_dims):
                params[f'param_{i}'] = np.random.uniform(0, 1)

            samples.append(params)

        return samples

    def _params_to_array(self, param_list: List[Dict[str, float]]) -> np.ndarray:
        """Convert list of parameter dictionaries to numpy array."""
        if not param_list:
            return np.array([])

        param_names = sorted(param_list[0].keys())
        array = np.zeros((len(param_list), len(param_names)))

        for i, params in enumerate(param_list):
            for j, name in enumerate(param_names):
                array[i, j] = params.get(name, 0.0)

        return array

    def _estimate_duration(self, parameters: Dict[str, float]) -> float:
        """Estimate experiment duration based on parameters."""
        # Simple heuristic: complexity increases with parameter values
        complexity = sum(abs(v) for v in parameters.values()) / len(parameters)
        base_time = 1.0
        return base_time * (1 + complexity)

    def _estimate_information_gain(self, parameters: Dict[str, float]) -> float:
        """Estimate expected information gain from experiment."""
        if not self.bayesian_optimizer:
            return 1.0

        # Information gain approximated by predictive variance
        X = self.bayesian_optimizer._dict_to_array([parameters])
        _, std = self.bayesian_optimizer.gp.predict(X, return_std=True)

        # Convert variance to information (bits)
        information = 0.5 * np.log2(2 * np.pi * np.e * (std[0]**2 + 1e-10))

        return max(0, information)

    def update_results(self, experiments: List[Experiment]):
        """Update engine with experimental results."""
        for exp in experiments:
            if exp.is_completed():
                self.completed_experiments.append(exp)
                self.total_cost += exp.cost
                self.total_time += exp.duration_hours

                # Update optimizers
                if self.bayesian_optimizer:
                    self.bayesian_optimizer.update_observations([exp])

                if self.thompson_sampler and 'hypothesis_id' in exp.parameters:
                    arm = int(exp.parameters['hypothesis_id'])
                    # Convert outcome to reward (0-1 scale)
                    reward = 1.0 if exp.outcome > 0 else 0.0
                    self.thompson_sampler.update(arm, reward)

                # Track information gain
                info_gain = exp.information_gain()
                self.information_history.append({
                    'experiment': exp.exp_id,
                    'information_gain': info_gain,
                    'cumulative_cost': self.total_cost,
                    'cumulative_time': self.total_time
                })

                # Remove from pending
                self.pending_experiments = [
                    e for e in self.pending_experiments if e.exp_id != exp.exp_id
                ]

    def get_convergence_analysis(self) -> Dict[str, Any]:
        """Analyze convergence of the experimental campaign."""
        if len(self.completed_experiments) < 2:
            return {'status': 'Insufficient data for convergence analysis'}

        analysis = {
            'n_experiments': len(self.completed_experiments),
            'total_cost': self.total_cost,
            'total_time': self.total_time,
            'convergence_metrics': {}
        }

        if self.bayesian_optimizer and self.bayesian_optimizer.best_value is not None:
            # Track best value over time
            best_values = []
            cumulative_best = float('inf') if self.bayesian_optimizer.objective.minimize else float('-inf')

            for exp in self.completed_experiments:
                if self.bayesian_optimizer.objective.minimize:
                    cumulative_best = min(cumulative_best, exp.outcome)
                else:
                    cumulative_best = max(cumulative_best, exp.outcome)
                best_values.append(cumulative_best)

            # Calculate simple regret
            if self.bayesian_optimizer.objective.optimal_value is not None:
                simple_regret = abs(cumulative_best - self.bayesian_optimizer.objective.optimal_value)
            else:
                simple_regret = None

            # Calculate convergence rate (improvement per experiment)
            if len(best_values) > 10:
                recent_improvement = abs(best_values[-1] - best_values[-10]) / 10
            else:
                recent_improvement = abs(best_values[-1] - best_values[0]) / len(best_values)

            analysis['convergence_metrics'] = {
                'best_value_found': cumulative_best,
                'simple_regret': simple_regret,
                'improvement_rate': recent_improvement,
                'likely_converged': recent_improvement < 0.01
            }

        # Information gain analysis
        if self.information_history:
            total_info = sum(h['information_gain'] for h in self.information_history)
            recent_info = np.mean([h['information_gain'] for h in self.information_history[-5:]])

            analysis['information_metrics'] = {
                'total_information_gain': total_info,
                'average_information_per_experiment': total_info / len(self.completed_experiments),
                'recent_information_rate': recent_info,
                'information_saturation': recent_info < 0.1
            }

        return analysis

    def generate_experimental_protocol(self, batch: ExperimentBatch) -> str:
        """Generate human-readable experimental protocol."""
        protocol = "# Experimental Protocol\n\n"
        protocol += f"## Batch ID: {batch.batch_id}\n"
        protocol += f"**Total Experiments**: {len(batch.experiments)}\n"
        protocol += f"**Total Cost**: ${batch.total_cost:.2f}\n"
        protocol += f"**Estimated Duration**: {batch.max_duration:.1f} hours\n"
        protocol += f"**Expected Information Gain**: {batch.expected_information_gain:.3f} bits\n\n"

        protocol += "## Experimental Conditions\n\n"

        for i, exp in enumerate(batch.experiments, 1):
            protocol += f"### Experiment {i} (ID: {exp.exp_id})\n"
            protocol += f"**Cost**: ${exp.cost:.2f}\n"
            protocol += f"**Duration**: {exp.duration_hours:.1f} hours\n\n"

            protocol += "**Parameters**:\n"
            for param, value in exp.parameters.items():
                protocol += f"- {param}: {value:.4f}\n"

            protocol += "\n**Procedure**:\n"
            protocol += self._generate_procedure(exp.parameters)
            protocol += "\n"

            protocol += "**Data Collection**:\n"
            protocol += "1. Record all observations with timestamps\n"
            protocol += "2. Note any deviations from protocol\n"
            protocol += "3. Estimate measurement uncertainty\n"
            protocol += "4. Document environmental conditions\n\n"

        protocol += "## Safety Considerations\n"
        protocol += "- Follow all laboratory safety protocols\n"
        protocol += "- Ensure proper PPE is worn\n"
        protocol += "- Have emergency procedures readily available\n\n"

        protocol += "## Quality Control\n"
        protocol += "- Run control experiments at start and end\n"
        protocol += "- Calibrate all instruments before use\n"
        protocol += "- Perform replicate measurements when possible\n"
        protocol += "- Document any equipment issues\n"

        return protocol

    def _generate_procedure(self, parameters: Dict[str, float]) -> str:
        """Generate specific procedure based on parameters."""
        procedure = ""

        # Generic procedure template
        procedure += "1. Set up experimental apparatus\n"
        procedure += "2. Configure parameters:\n"

        for param, value in parameters.items():
            procedure += f"   - Set {param} to {value:.4f}\n"

        procedure += "3. Allow system to equilibrate (5 minutes)\n"
        procedure += "4. Begin data collection\n"
        procedure += "5. Monitor for steady-state conditions\n"
        procedure += "6. Record primary outcome measure\n"
        procedure += "7. Perform replicate measurements (n=3)\n"
        procedure += "8. Calculate mean and standard deviation\n"

        return procedure

    def optimize_batch_design(self,
                            budget: float,
                            max_batch_size: int,
                            parallel_capacity: int) -> List[ExperimentBatch]:
        """
        Optimize design of multiple experimental batches.

        Accounts for parallel execution capacity and sequential learning.
        """
        batches = []
        remaining_budget = budget
        total_expected_gain = 0.0

        while remaining_budget > 0 and len(batches) < 10:  # Max 10 batches
            # Determine batch size (limited by parallel capacity)
            batch_size = min(max_batch_size, parallel_capacity)

            # Design batch
            batch = self.suggest_experiments(
                budget=remaining_budget,
                batch_size=batch_size,
                strategy=SamplingStrategy.HYBRID
            )

            if len(batch.experiments) == 0:
                break

            batches.append(batch)
            remaining_budget -= batch.total_cost
            total_expected_gain += batch.expected_information_gain

            # Simulate completing this batch for next iteration
            for exp in batch.experiments:
                # Simulate outcome (would be actual results in practice)
                exp.outcome = np.random.randn()
                exp.uncertainty = 0.1

            self.update_results(batch.experiments)

        logger.info(f"Designed {len(batches)} batches with total expected gain: {total_expected_gain:.3f}")

        return batches

    def recommend_stopping(self) -> Dict[str, Any]:
        """
        Recommend whether to stop experimentation.

        Based on convergence criteria and diminishing returns.
        """
        recommendation = {
            'should_stop': False,
            'confidence': 0.0,
            'reasons': [],
            'suggested_actions': []
        }

        if len(self.completed_experiments) < 5:
            recommendation['suggested_actions'].append("Continue experimentation - insufficient data")
            return recommendation

        convergence = self.get_convergence_analysis()

        # Check convergence
        if convergence.get('convergence_metrics', {}).get('likely_converged', False):
            recommendation['reasons'].append("Optimization has likely converged")
            recommendation['confidence'] += 0.3

        # Check information saturation
        if convergence.get('information_metrics', {}).get('information_saturation', False):
            recommendation['reasons'].append("Information gain is saturating")
            recommendation['confidence'] += 0.3

        # Check cost efficiency
        if self.information_history:
            recent_efficiency = np.mean([
                h['information_gain'] / self.completed_experiments[i].cost
                for i, h in enumerate(self.information_history[-5:])
            ])

            if recent_efficiency < 0.1:
                recommendation['reasons'].append("Low information per unit cost")
                recommendation['confidence'] += 0.2

        # Check hypothesis testing completion
        if self.thompson_sampler:
            stats = self.thompson_sampler.get_arm_statistics()
            high_confidence_arms = sum(
                1 for s in stats.values()
                if s['posterior_std'] < 0.1
            )

            if high_confidence_arms >= len(stats) * 0.8:
                recommendation['reasons'].append("High confidence in hypothesis rankings")
                recommendation['confidence'] += 0.2

        # Make recommendation
        if recommendation['confidence'] >= 0.5:
            recommendation['should_stop'] = True
            recommendation['suggested_actions'].append("Consider stopping - objectives likely achieved")
        else:
            recommendation['suggested_actions'].append("Continue with focused experiments on high-uncertainty regions")

        return recommendation