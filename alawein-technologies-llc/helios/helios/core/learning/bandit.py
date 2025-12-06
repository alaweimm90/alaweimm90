"""
UCB1 Multi-Armed Bandit for Agent Selection

Learns which agents work best for which tasks.
"""

import math
from typing import Dict, List, Optional
import numpy as np


class UCB1Bandit:
    """
    Upper Confidence Bound (UCB1) algorithm for agent selection

    Balances exploration (trying untested agents) with exploitation
    (using agents known to work well).

    Formula: UCB1(agent) = avg_reward + sqrt(2 * ln(total_pulls) / agent_pulls)
    """

    def __init__(self, agent_ids: List[str], c: float = 2.0):
        """
        Initialize bandit

        Args:
            agent_ids: List of available agent IDs
            c: Exploration parameter (higher = more exploration)
        """
        self.agent_ids = agent_ids
        self.c = c

        # Tracking
        self.total_pulls = 0
        self.agent_pulls: Dict[str, int] = {agent_id: 0 for agent_id in agent_ids}
        self.agent_rewards: Dict[str, List[float]] = {agent_id: [] for agent_id in agent_ids}

    def select_agent(self) -> str:
        """
        Select next agent using UCB1

        Returns:
            agent_id to use
        """
        # First, pull all agents at least once (pure exploration)
        for agent_id in self.agent_ids:
            if self.agent_pulls[agent_id] == 0:
                return agent_id

        # UCB1 selection
        ucb_scores = {}

        for agent_id in self.agent_ids:
            avg_reward = np.mean(self.agent_rewards[agent_id])
            exploration_bonus = math.sqrt(
                (self.c * math.log(self.total_pulls)) / self.agent_pulls[agent_id]
            )
            ucb_scores[agent_id] = avg_reward + exploration_bonus

        # Select agent with highest UCB score
        best_agent = max(ucb_scores.items(), key=lambda x: x[1])[0]
        return best_agent

    def update(self, agent_id: str, reward: float):
        """
        Update after observing reward

        Args:
            agent_id: Agent that was used
            reward: Reward received (0-1 typically)
        """
        self.agent_pulls[agent_id] += 1
        self.agent_rewards[agent_id].append(reward)
        self.total_pulls += 1

    def get_best_agent(self) -> str:
        """Get agent with highest average reward"""
        avg_rewards = {
            agent_id: np.mean(rewards) if rewards else 0.0
            for agent_id, rewards in self.agent_rewards.items()
        }
        return max(avg_rewards.items(), key=lambda x: x[1])[0]

    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics for all agents"""
        stats = {}

        for agent_id in self.agent_ids:
            rewards = self.agent_rewards[agent_id]
            pulls = self.agent_pulls[agent_id]

            stats[agent_id] = {
                "pulls": pulls,
                "avg_reward": np.mean(rewards) if rewards else 0.0,
                "std_reward": np.std(rewards) if rewards else 0.0,
                "total_reward": sum(rewards),
                "success_rate": pulls / self.total_pulls if self.total_pulls > 0 else 0.0,
            }

        return stats

    def add_agent(self, agent_id: str):
        """Add a new agent to the pool"""
        if agent_id not in self.agent_ids:
            self.agent_ids.append(agent_id)
            self.agent_pulls[agent_id] = 0
            self.agent_rewards[agent_id] = []

    def remove_agent(self, agent_id: str):
        """Remove an agent from the pool"""
        if agent_id in self.agent_ids:
            self.agent_ids.remove(agent_id)
            del self.agent_pulls[agent_id]
            del self.agent_rewards[agent_id]


class ContextualBandit(UCB1Bandit):
    """
    Contextual bandit that considers task context

    Different agents may be better for different contexts (e.g., domains).
    """

    def __init__(self, agent_ids: List[str], contexts: List[str], c: float = 2.0):
        """
        Initialize contextual bandit

        Args:
            agent_ids: List of agent IDs
            contexts: List of possible contexts (e.g., domains)
            c: Exploration parameter
        """
        super().__init__(agent_ids, c)
        self.contexts = contexts

        # Per-context tracking
        self.context_pulls: Dict[str, Dict[str, int]] = {
            ctx: {agent_id: 0 for agent_id in agent_ids}
            for ctx in contexts
        }

        self.context_rewards: Dict[str, Dict[str, List[float]]] = {
            ctx: {agent_id: [] for agent_id in agent_ids}
            for ctx in contexts
        }

    def select_agent(self, context: str) -> str:
        """
        Select agent for specific context

        Args:
            context: Current context (e.g., "optimization", "machine_learning")

        Returns:
            agent_id to use
        """
        if context not in self.contexts:
            # Fallback to global selection
            return super().select_agent()

        # Context-specific UCB1
        context_total = sum(self.context_pulls[context].values())

        # First pull all agents once in this context
        for agent_id in self.agent_ids:
            if self.context_pulls[context][agent_id] == 0:
                return agent_id

        # UCB1 for this context
        ucb_scores = {}

        for agent_id in self.agent_ids:
            rewards = self.context_rewards[context][agent_id]
            if not rewards:
                continue

            avg_reward = np.mean(rewards)
            pulls = self.context_pulls[context][agent_id]

            exploration_bonus = math.sqrt(
                (self.c * math.log(context_total)) / pulls
            )

            ucb_scores[agent_id] = avg_reward + exploration_bonus

        if not ucb_scores:
            # All agents untried in this context
            return self.agent_ids[0]

        best_agent = max(ucb_scores.items(), key=lambda x: x[1])[0]
        return best_agent

    def update(self, agent_id: str, reward: float, context: Optional[str] = None):
        """
        Update with context

        Args:
            agent_id: Agent used
            reward: Reward received
            context: Context this happened in
        """
        # Update global
        super().update(agent_id, reward)

        # Update context-specific
        if context and context in self.contexts:
            self.context_pulls[context][agent_id] += 1
            self.context_rewards[context][agent_id].append(reward)

    def get_best_agent(self, context: Optional[str] = None) -> str:
        """Get best agent for context"""
        if context and context in self.contexts:
            avg_rewards = {
                agent_id: np.mean(rewards) if rewards else 0.0
                for agent_id, rewards in self.context_rewards[context].items()
            }
            return max(avg_rewards.items(), key=lambda x: x[1])[0]

        return super().get_best_agent()

    def get_context_stats(self, context: str) -> Dict[str, Dict]:
        """Get stats for a specific context"""
        if context not in self.contexts:
            return {}

        stats = {}
        context_total = sum(self.context_pulls[context].values())

        for agent_id in self.agent_ids:
            rewards = self.context_rewards[context][agent_id]
            pulls = self.context_pulls[context][agent_id]

            stats[agent_id] = {
                "pulls": pulls,
                "avg_reward": np.mean(rewards) if rewards else 0.0,
                "std_reward": np.std(rewards) if rewards else 0.0,
                "success_rate": pulls / context_total if context_total > 0 else 0.0,
            }

        return stats
