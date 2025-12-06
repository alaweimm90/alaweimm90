"""
Online Learning System for Adaptive Method Selection

This module implements various multi-armed bandit algorithms for online
learning and adaptive method selection based on contextual information.

Author: Meshal Alawein
Date: 2025-11-18
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import beta

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmArm:
    """Represents an algorithm as an arm in the multi-armed bandit."""

    name: str
    n_pulls: int = 0
    total_reward: float = 0.0
    successes: int = 0
    failures: int = 0
    context_rewards: Dict[str, float] = None
    last_reward: float = 0.0

    def __post_init__(self):
        if self.context_rewards is None:
            self.context_rewards = {}

    @property
    def mean_reward(self) -> float:
        """Calculate mean reward for this arm."""
        if self.n_pulls == 0:
            return 0.5  # Prior assumption
        return self.total_reward / self.n_pulls


class BanditSelector(ABC):
    """Abstract base class for bandit-based algorithm selectors."""

    def __init__(self, algorithms: List[str], exploration_rate: float = 0.1):
        """
        Initialize the bandit selector.

        Args:
            algorithms: List of algorithm names
            exploration_rate: Rate of exploration vs exploitation
        """
        self.algorithms = algorithms
        self.exploration_rate = exploration_rate
        self.arms = {alg: AlgorithmArm(name=alg) for alg in algorithms}
        self.t = 0  # Time step

    @abstractmethod
    def select_algorithm(
        self,
        context: Optional[np.ndarray] = None
    ) -> str:
        """Select an algorithm based on the bandit strategy."""
        pass

    def update_reward(
        self,
        algorithm: str,
        reward: float,
        context: Optional[np.ndarray] = None
    ):
        """
        Update the reward for the selected algorithm.

        Args:
            algorithm: Name of the algorithm
            reward: Reward value (higher is better)
            context: Optional context vector
        """
        arm = self.arms[algorithm]
        arm.n_pulls += 1
        arm.total_reward += reward
        arm.last_reward = reward

        # Update success/failure counts (assuming reward in [0, 1])
        if reward > 0.5:
            arm.successes += 1
        else:
            arm.failures += 1

        # Store context-specific rewards
        if context is not None:
            context_key = self._hash_context(context)
            if context_key not in arm.context_rewards:
                arm.context_rewards[context_key] = 0.0
            arm.context_rewards[context_key] = (
                0.9 * arm.context_rewards[context_key] + 0.1 * reward
            )

        self.t += 1

    def _hash_context(self, context: np.ndarray) -> str:
        """Create a hash key for context vector."""
        # Discretize context for storage
        discretized = np.round(context, decimals=2)
        return str(discretized.tobytes())


class UCB1Selector(BanditSelector):
    """
    Upper Confidence Bound (UCB1) algorithm for method selection.

    This algorithm balances exploration and exploitation by selecting
    algorithms based on their mean reward plus an exploration bonus.
    """

    def __init__(self, algorithms: List[str], c: float = 2.0):
        """
        Initialize UCB1 selector.

        Args:
            algorithms: List of algorithm names
            c: Exploration constant (higher = more exploration)
        """
        super().__init__(algorithms)
        self.c = c

    def select_algorithm(
        self,
        context: Optional[np.ndarray] = None
    ) -> str:
        """
        Select algorithm using UCB1 strategy.

        UCB = mean_reward + c * sqrt(ln(t) / n_pulls)
        """
        if self.t < len(self.algorithms):
            # Initial exploration: try each algorithm once
            return self.algorithms[self.t]

        ucb_scores = {}
        for alg_name, arm in self.arms.items():
            # Exploitation term
            exploitation = arm.mean_reward

            # Exploration term
            if arm.n_pulls > 0:
                exploration = self.c * np.sqrt(np.log(self.t) / arm.n_pulls)
            else:
                exploration = float('inf')  # Ensure unexplored arms are tried

            ucb_scores[alg_name] = exploitation + exploration

        # Select algorithm with highest UCB score
        selected = max(ucb_scores.keys(), key=lambda k: ucb_scores[k])

        logger.debug(
            f"UCB1 selected {selected} with score {ucb_scores[selected]:.4f}"
        )

        return selected


class ThompsonSampler(BanditSelector):
    """
    Thompson Sampling algorithm for method selection.

    This Bayesian approach samples from the posterior distribution
    of each algorithm's performance.
    """

    def __init__(
        self,
        algorithms: List[str],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ):
        """
        Initialize Thompson Sampler.

        Args:
            algorithms: List of algorithm names
            prior_alpha: Alpha parameter for Beta prior
            prior_beta: Beta parameter for Beta prior
        """
        super().__init__(algorithms)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def select_algorithm(
        self,
        context: Optional[np.ndarray] = None
    ) -> str:
        """
        Select algorithm using Thompson Sampling.

        Samples from Beta(successes + α, failures + β) for each arm.
        """
        samples = {}

        for alg_name, arm in self.arms.items():
            # Sample from posterior Beta distribution
            alpha = arm.successes + self.prior_alpha
            beta_param = arm.failures + self.prior_beta

            sample = np.random.beta(alpha, beta_param)
            samples[alg_name] = sample

        # Select algorithm with highest sample
        selected = max(samples.keys(), key=lambda k: samples[k])

        logger.debug(
            f"Thompson Sampling selected {selected} with sample {samples[selected]:.4f}"
        )

        return selected


class EXP3Selector(BanditSelector):
    """
    Exponential-weight algorithm for Exploration and Exploitation (EXP3).

    This algorithm is designed for adversarial bandits where rewards
    may change over time.
    """

    def __init__(
        self,
        algorithms: List[str],
        gamma: float = 0.1
    ):
        """
        Initialize EXP3 selector.

        Args:
            algorithms: List of algorithm names
            gamma: Exploration parameter (0 = pure exploitation, 1 = uniform random)
        """
        super().__init__(algorithms)
        self.gamma = gamma
        self.weights = {alg: 1.0 for alg in algorithms}

    def select_algorithm(
        self,
        context: Optional[np.ndarray] = None
    ) -> str:
        """
        Select algorithm using EXP3 strategy.

        Maintains weights for each arm and selects probabilistically.
        """
        # Calculate probabilities
        total_weight = sum(self.weights.values())
        probabilities = {}

        for alg_name in self.algorithms:
            # Mix uniform distribution with weight-based distribution
            prob = ((1 - self.gamma) * self.weights[alg_name] / total_weight +
                    self.gamma / len(self.algorithms))
            probabilities[alg_name] = prob

        # Sample from probability distribution
        algorithms = list(probabilities.keys())
        probs = list(probabilities.values())
        selected = np.random.choice(algorithms, p=probs)

        logger.debug(
            f"EXP3 selected {selected} with probability {probabilities[selected]:.4f}"
        )

        return selected

    def update_reward(
        self,
        algorithm: str,
        reward: float,
        context: Optional[np.ndarray] = None
    ):
        """Update weights based on reward."""
        super().update_reward(algorithm, reward, context)

        # Calculate probability of selection
        total_weight = sum(self.weights.values())
        prob = ((1 - self.gamma) * self.weights[algorithm] / total_weight +
                self.gamma / len(self.algorithms))

        # Update weight with importance-weighted reward
        estimated_reward = reward / prob
        self.weights[algorithm] *= np.exp(self.gamma * estimated_reward / len(self.algorithms))

        # Prevent numerical overflow
        max_weight = max(self.weights.values())
        if max_weight > 1e10:
            for alg in self.weights:
                self.weights[alg] /= max_weight


class OnlineLearner:
    """
    Main online learning system that coordinates multiple bandit algorithms
    and provides contextual method selection.
    """

    def __init__(
        self,
        algorithms: List[str],
        strategy: str = 'ucb1',
        contextual: bool = True
    ):
        """
        Initialize the online learner.

        Args:
            algorithms: List of available algorithms
            strategy: Bandit strategy ('ucb1', 'thompson', 'exp3')
            contextual: Whether to use contextual features
        """
        self.algorithms = algorithms
        self.strategy = strategy
        self.contextual = contextual

        # Initialize bandit selector
        if strategy == 'ucb1':
            self.selector = UCB1Selector(algorithms)
        elif strategy == 'thompson':
            self.selector = ThompsonSampler(algorithms)
        elif strategy == 'exp3':
            self.selector = EXP3Selector(algorithms)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Track performance over time
        self.selection_history = []
        self.reward_history = []
        self.regret_history = []
        self.best_known_reward = 0.0

    def select_method(
        self,
        problem_features: Optional[np.ndarray] = None,
        time_budget: Optional[float] = None,
        evaluation_budget: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select an optimization method based on current knowledge.

        Args:
            problem_features: Features of the optimization problem
            time_budget: Available time budget
            evaluation_budget: Available evaluation budget

        Returns:
            Tuple of (method_name, configuration)
        """
        # Create context vector if contextual
        context = None
        if self.contextual and problem_features is not None:
            context = self._create_context(
                problem_features,
                time_budget,
                evaluation_budget
            )

        # Select algorithm using bandit
        selected_algorithm = self.selector.select_algorithm(context)

        # Generate configuration
        config = self._generate_config(
            selected_algorithm,
            problem_features,
            time_budget,
            evaluation_budget
        )

        # Record selection
        self.selection_history.append(selected_algorithm)

        logger.info(f"Online learner selected: {selected_algorithm}")

        return selected_algorithm, config

    def update_performance(
        self,
        algorithm: str,
        objective_value: float,
        runtime: float,
        n_evaluations: int,
        problem_features: Optional[np.ndarray] = None
    ):
        """
        Update the learner with performance feedback.

        Args:
            algorithm: Algorithm that was used
            objective_value: Achieved objective value (lower is better)
            runtime: Runtime in seconds
            n_evaluations: Number of evaluations used
            problem_features: Problem features for context
        """
        # Calculate reward (normalized to [0, 1])
        reward = self._calculate_reward(
            objective_value,
            runtime,
            n_evaluations
        )

        # Create context if needed
        context = None
        if self.contextual and problem_features is not None:
            context = self._create_context(problem_features, runtime, n_evaluations)

        # Update selector
        self.selector.update_reward(algorithm, reward, context)

        # Track performance
        self.reward_history.append(reward)

        # Update best known reward
        self.best_known_reward = max(self.best_known_reward, reward)

        # Calculate and track regret
        regret = self.best_known_reward - reward
        if self.regret_history:
            cumulative_regret = self.regret_history[-1] + regret
        else:
            cumulative_regret = regret
        self.regret_history.append(cumulative_regret)

        logger.info(
            f"Updated {algorithm}: reward={reward:.4f}, "
            f"cumulative_regret={cumulative_regret:.4f}"
        )

    def _create_context(
        self,
        problem_features: np.ndarray,
        time_budget: Optional[float] = None,
        evaluation_budget: Optional[int] = None
    ) -> np.ndarray:
        """Create context vector from problem features and budgets."""
        context = problem_features.copy()

        # Add budget features
        if time_budget is not None:
            context = np.append(context, np.log1p(time_budget))
        if evaluation_budget is not None:
            context = np.append(context, np.log1p(evaluation_budget))

        # Normalize context
        context = (context - np.mean(context)) / (np.std(context) + 1e-8)

        return context

    def _calculate_reward(
        self,
        objective_value: float,
        runtime: float,
        n_evaluations: int
    ) -> float:
        """
        Calculate reward from performance metrics.

        Combines solution quality, runtime efficiency, and evaluation efficiency.
        """
        # Solution quality component (assuming minimization)
        # Map objective to [0, 1] using sigmoid-like transformation
        quality_reward = 1.0 / (1.0 + objective_value)

        # Runtime efficiency component
        # Prefer faster algorithms
        runtime_reward = np.exp(-runtime / 60.0)  # Decay over minutes

        # Evaluation efficiency component
        # Prefer algorithms that use fewer evaluations
        eval_reward = np.exp(-n_evaluations / 10000.0)

        # Weighted combination
        reward = (
            0.6 * quality_reward +
            0.2 * runtime_reward +
            0.2 * eval_reward
        )

        return np.clip(reward, 0.0, 1.0)

    def _generate_config(
        self,
        algorithm: str,
        problem_features: Optional[np.ndarray],
        time_budget: Optional[float],
        evaluation_budget: Optional[int]
    ) -> Dict[str, Any]:
        """Generate configuration for the selected algorithm."""
        config = {}

        # Set budgets
        if time_budget is not None:
            config['max_time'] = time_budget
        if evaluation_budget is not None:
            config['max_evaluations'] = evaluation_budget

        # Algorithm-specific configuration based on learned performance
        arm = self.selector.arms[algorithm]

        if algorithm == 'simulated_annealing':
            # Adjust temperature based on performance
            if arm.n_pulls > 0:
                # Higher temperature if struggling, lower if performing well
                temp_factor = 2.0 - arm.mean_reward
                config['initial_temperature'] = 100.0 * temp_factor
            else:
                config['initial_temperature'] = 100.0

        elif algorithm == 'genetic_algorithm':
            # Adjust population size based on problem size
            if problem_features is not None and len(problem_features) > 0:
                problem_size = problem_features[0]  # Assuming first feature is size
                config['population_size'] = min(200, int(10 * problem_size))

        elif algorithm == 'particle_swarm':
            # Adjust exploration/exploitation balance
            if arm.n_pulls > 0:
                # More exploration if performing poorly
                config['w'] = 0.4 + 0.5 * (1.0 - arm.mean_reward)

        return config

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of online learning performance."""
        summary = {
            'strategy': self.strategy,
            'n_selections': len(self.selection_history),
            'algorithms_used': list(set(self.selection_history)),
            'average_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'best_reward': self.best_known_reward,
            'cumulative_regret': self.regret_history[-1] if self.regret_history else 0.0,
            'algorithm_stats': {}
        }

        # Add per-algorithm statistics
        for alg_name, arm in self.selector.arms.items():
            if arm.n_pulls > 0:
                summary['algorithm_stats'][alg_name] = {
                    'n_pulls': arm.n_pulls,
                    'mean_reward': arm.mean_reward,
                    'success_rate': arm.successes / arm.n_pulls,
                    'last_reward': arm.last_reward
                }

        return summary