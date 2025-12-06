"""
ORCHEX Advanced Learning Algorithms

Multiple bandit algorithms for agent selection and optimization.
Goes beyond UCB1 with Thompson Sampling, contextual bandits, and more.

Cycle 25-26: Custom Learning Algorithms
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import math


class BanditAlgorithm(str, Enum):
    """Available bandit algorithms"""

    UCB1 = "ucb1"  # Upper Confidence Bound (original)
    THOMPSON_SAMPLING = "thompson_sampling"  # Bayesian approach
    EPSILON_GREEDY = "epsilon_greedy"  # Simple exploration
    SOFTMAX = "softmax"  # Temperature-based selection
    EXP3 = "exp3"  # Exponential-weight algorithm for adversarial bandits
    CONTEXTUAL_UCB = "contextual_ucb"  # Context-aware UCB


@dataclass
class ArmStats:
    """Statistics for a bandit arm (agent)"""

    arm_id: str
    pulls: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0

    # Thompson Sampling (Beta distribution)
    successes: float = 1.0  # Prior: Beta(1, 1)
    failures: float = 1.0

    # Contextual bandit
    feature_weights: Optional[np.ndarray] = None
    context_history: List[np.ndarray] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)


class AdvancedBanditSelector:
    """
    Advanced multi-armed bandit algorithms for agent selection.

    Supports multiple algorithms:
    - UCB1: Original upper confidence bound
    - Thompson Sampling: Bayesian posterior sampling
    - Epsilon-Greedy: Simple exploration strategy
    - Softmax: Temperature-based probability
    - EXP3: Adversarial bandit algorithm
    - Contextual UCB: Context-aware selection
    """

    def __init__(
        self,
        arms: List[str],
        algorithm: BanditAlgorithm = BanditAlgorithm.THOMPSON_SAMPLING,
        epsilon: float = 0.1,  # For epsilon-greedy
        temperature: float = 0.3,  # For softmax
        exploration_bonus: float = 2.0,  # For UCB
    ):
        """
        Initialize bandit selector

        Args:
            arms: List of arm IDs (agent names)
            algorithm: Which algorithm to use
            epsilon: Exploration rate for epsilon-greedy
            temperature: Temperature parameter for softmax
            exploration_bonus: Exploration bonus for UCB variants
        """
        self.arms = arms
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.temperature = temperature
        self.exploration_bonus = exploration_bonus

        # Initialize arm statistics
        self.arm_stats: Dict[str, ArmStats] = {
            arm: ArmStats(arm_id=arm)
            for arm in arms
        }

        self.total_pulls = 0

        # EXP3 weights
        self.exp3_weights = {arm: 1.0 for arm in arms}

    def select_arm(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Select an arm using the configured algorithm

        Args:
            context: Optional context for contextual bandits

        Returns:
            Selected arm ID
        """
        self.total_pulls += 1

        if self.algorithm == BanditAlgorithm.UCB1:
            return self._select_ucb1()
        elif self.algorithm == BanditAlgorithm.THOMPSON_SAMPLING:
            return self._select_thompson()
        elif self.algorithm == BanditAlgorithm.EPSILON_GREEDY:
            return self._select_epsilon_greedy()
        elif self.algorithm == BanditAlgorithm.SOFTMAX:
            return self._select_softmax()
        elif self.algorithm == BanditAlgorithm.EXP3:
            return self._select_exp3()
        elif self.algorithm == BanditAlgorithm.CONTEXTUAL_UCB:
            return self._select_contextual_ucb(context)
        else:
            # Fallback to UCB1
            return self._select_ucb1()

    def update(
        self,
        arm: str,
        reward: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Update statistics after observing reward

        Args:
            arm: Selected arm ID
            reward: Observed reward (0-100)
            context: Optional context that was used
        """
        stats = self.arm_stats[arm]

        # Update basic stats
        stats.pulls += 1
        stats.total_reward += reward
        stats.mean_reward = stats.total_reward / stats.pulls

        # Normalize reward to 0-1 for Thompson Sampling
        normalized_reward = reward / 100.0

        # Update Thompson Sampling stats (Beta distribution)
        if normalized_reward >= 0.7:  # Success threshold
            stats.successes += 1
        else:
            stats.failures += 1

        # Update EXP3 weights
        if self.algorithm == BanditAlgorithm.EXP3:
            self._update_exp3_weights(arm, reward)

        # Update contextual bandit
        if self.algorithm == BanditAlgorithm.CONTEXTUAL_UCB and context:
            context_vec = self._context_to_vector(context)
            stats.context_history.append(context_vec)
            stats.reward_history.append(normalized_reward)
            # Simple linear regression (could be improved)
            if len(stats.context_history) > 10:
                X = np.array(stats.context_history)
                y = np.array(stats.reward_history)
                # Least squares fit
                try:
                    stats.feature_weights = np.linalg.lstsq(X, y, rcond=None)[0]
                except:
                    pass

    def _select_ucb1(self) -> str:
        """UCB1 algorithm"""
        best_arm = None
        best_score = float('-inf')

        for arm in self.arms:
            stats = self.arm_stats[arm]

            if stats.pulls == 0:
                # Always try arms that haven't been pulled
                return arm

            # UCB1 formula
            exploitation = stats.mean_reward / 100.0
            exploration = math.sqrt(
                (self.exploration_bonus * math.log(self.total_pulls)) / stats.pulls
            )
            ucb_score = exploitation + exploration

            if ucb_score > best_score:
                best_score = ucb_score
                best_arm = arm

        return best_arm if best_arm else random.choice(self.arms)

    def _select_thompson(self) -> str:
        """Thompson Sampling (Bayesian)"""
        samples = {}

        for arm in self.arms:
            stats = self.arm_stats[arm]

            # Sample from Beta(successes, failures)
            sample = np.random.beta(stats.successes, stats.failures)
            samples[arm] = sample

        # Select arm with highest sample
        return max(samples, key=samples.get)

    def _select_epsilon_greedy(self) -> str:
        """Epsilon-greedy: explore with probability epsilon"""

        # Explore
        if random.random() < self.epsilon:
            return random.choice(self.arms)

        # Exploit: select best arm
        best_arm = max(
            self.arms,
            key=lambda a: self.arm_stats[a].mean_reward
            if self.arm_stats[a].pulls > 0
            else 0
        )
        return best_arm

    def _select_softmax(self) -> str:
        """Softmax (Boltzmann) selection"""

        # Compute softmax probabilities
        scores = {}
        for arm in self.arms:
            stats = self.arm_stats[arm]
            if stats.pulls > 0:
                scores[arm] = stats.mean_reward / 100.0
            else:
                scores[arm] = 0.5  # Neutral for unexplored

        # Apply softmax with temperature
        exp_scores = {
            arm: math.exp(score / self.temperature)
            for arm, score in scores.items()
        }
        total = sum(exp_scores.values())
        probs = {arm: exp / total for arm, exp in exp_scores.items()}

        # Sample from distribution
        r = random.random()
        cumulative = 0.0
        for arm, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                return arm

        return random.choice(self.arms)  # Fallback

    def _select_exp3(self) -> str:
        """EXP3: Exponential-weight algorithm for adversarial bandits"""

        # Normalize weights to probabilities
        total_weight = sum(self.exp3_weights.values())
        probs = {
            arm: weight / total_weight
            for arm, weight in self.exp3_weights.items()
        }

        # Add exploration
        gamma = min(1.0, math.sqrt(len(self.arms) * math.log(len(self.arms)) / self.total_pulls))
        for arm in self.arms:
            probs[arm] = (1 - gamma) * probs[arm] + gamma / len(self.arms)

        # Sample from distribution
        r = random.random()
        cumulative = 0.0
        for arm, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                return arm

        return random.choice(self.arms)

    def _update_exp3_weights(self, arm: str, reward: float):
        """Update EXP3 weights"""
        # Normalize reward to 0-1
        normalized_reward = reward / 100.0

        # Compute probability
        total_weight = sum(self.exp3_weights.values())
        prob = self.exp3_weights[arm] / total_weight

        # Estimated reward
        estimated_reward = normalized_reward / prob if prob > 0 else 0

        # Update weight
        gamma = min(1.0, math.sqrt(len(self.arms) * math.log(len(self.arms)) / self.total_pulls))
        self.exp3_weights[arm] *= math.exp(gamma * estimated_reward / len(self.arms))

    def _select_contextual_ucb(self, context: Optional[Dict[str, Any]]) -> str:
        """Contextual UCB (LinUCB)"""

        if context is None:
            # Fallback to regular UCB1
            return self._select_ucb1()

        context_vec = self._context_to_vector(context)

        best_arm = None
        best_score = float('-inf')

        for arm in self.arms:
            stats = self.arm_stats[arm]

            if stats.pulls == 0 or stats.feature_weights is None:
                # Always try arms without enough data
                if stats.pulls < 5:
                    return arm
                # Use mean reward for arms without model
                score = stats.mean_reward / 100.0
            else:
                # Predict reward using linear model
                predicted_reward = np.dot(stats.feature_weights, context_vec)

                # Add exploration bonus (simplified LinUCB)
                exploration = math.sqrt(
                    (self.exploration_bonus * math.log(self.total_pulls)) / stats.pulls
                )

                score = predicted_reward + exploration

            if score > best_score:
                best_score = score
                best_arm = arm

        return best_arm if best_arm else random.choice(self.arms)

    def _context_to_vector(self, context: Dict[str, Any]) -> np.ndarray:
        """Convert context dict to feature vector"""

        # Simple feature extraction
        features = []

        # Domain features (one-hot encoding for common domains)
        domains = ["machine_learning", "optimization", "nlp", "computer_vision", "other"]
        domain = context.get("domain", "other")
        features.extend([1.0 if d == domain else 0.0 for d in domains])

        # Problem type features (if available)
        problem_types = ["novelty_check", "validation", "optimization", "other"]
        problem_type = context.get("problem_type", "other")
        features.extend([1.0 if pt == problem_type else 0.0 for pt in problem_types])

        # Numeric features
        features.append(context.get("hypothesis_count", 5) / 10.0)  # Normalize
        features.append(context.get("validation_strictness", 0.5))

        return np.array(features)

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics for all arms"""

        stats_summary = {
            "algorithm": self.algorithm,
            "total_pulls": self.total_pulls,
            "arms": {}
        }

        for arm in self.arms:
            stats = self.arm_stats[arm]
            stats_summary["arms"][arm] = {
                "pulls": stats.pulls,
                "mean_reward": stats.mean_reward,
                "total_reward": stats.total_reward,
                "thompson_alpha": stats.successes,
                "thompson_beta": stats.failures,
            }

        return stats_summary

    def get_best_arm(self) -> str:
        """Get arm with highest mean reward (pure exploitation)"""
        return max(
            self.arms,
            key=lambda a: self.arm_stats[a].mean_reward
            if self.arm_stats[a].pulls > 0
            else 0
        )

    def reset(self):
        """Reset all statistics"""
        self.arm_stats = {
            arm: ArmStats(arm_id=arm)
            for arm in self.arms
        }
        self.total_pulls = 0
        self.exp3_weights = {arm: 1.0 for arm in self.arms}


class EnsembleBandit:
    """
    Ensemble of multiple bandit algorithms.

    Uses a meta-bandit to select which bandit algorithm to use!
    """

    def __init__(self, arms: List[str]):
        """
        Initialize ensemble bandit

        Args:
            arms: List of arm IDs
        """
        self.arms = arms

        # Create multiple bandit selectors
        self.bandits = {
            BanditAlgorithm.UCB1: AdvancedBanditSelector(arms, BanditAlgorithm.UCB1),
            BanditAlgorithm.THOMPSON_SAMPLING: AdvancedBanditSelector(arms, BanditAlgorithm.THOMPSON_SAMPLING),
            BanditAlgorithm.EPSILON_GREEDY: AdvancedBanditSelector(arms, BanditAlgorithm.EPSILON_GREEDY),
            BanditAlgorithm.SOFTMAX: AdvancedBanditSelector(arms, BanditAlgorithm.SOFTMAX),
        }

        # Meta-bandit to select which algorithm to use
        self.meta_bandit = AdvancedBanditSelector(
            list(self.bandits.keys()),
            algorithm=BanditAlgorithm.THOMPSON_SAMPLING
        )

        # Track which algorithm was used for each selection
        self.last_algorithm = None
        self.last_arm = None

    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Select arm using ensemble"""

        # Meta-bandit selects which algorithm to use
        selected_algorithm = self.meta_bandit.select_arm()
        self.last_algorithm = selected_algorithm

        # Use selected algorithm to choose arm
        bandit = self.bandits[selected_algorithm]
        self.last_arm = bandit.select_arm(context)

        return self.last_arm

    def update(self, arm: str, reward: float, context: Optional[Dict[str, Any]] = None):
        """Update after observing reward"""

        # Update the bandit that was used
        if self.last_algorithm:
            bandit = self.bandits[self.last_algorithm]
            bandit.update(arm, reward, context)

            # Update meta-bandit (reward is how well the algorithm performed)
            self.meta_bandit.update(self.last_algorithm, reward)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all bandits"""
        return {
            "meta_bandit": self.meta_bandit.get_statistics(),
            "bandits": {
                algo: bandit.get_statistics()
                for algo, bandit in self.bandits.items()
            }
        }
