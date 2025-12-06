"""
Librex.Alloc Solver - Constrained Thompson Sampling for Resource Allocation

Solves the problem of allocating limited resources (API credits, compute budget)
across competing agents to minimize cumulative regret.

Mathematical Formulation:
    Maximize: Expected reward over horizon
    Subject to: Budget constraint (total allocation <= budget)

Uses Thompson Sampling (Bayesian bandit) with budget constraints.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from MEZAN.core import (
    OptimizerInterface,
    OptimizationProblem,
    OptimizationResult,
    ProblemType,
    SolverStatus,
)

logger = logging.getLogger(__name__)


class Librex.AllocSolver(OptimizerInterface):
    """
    Thompson Sampling with budget constraints for resource allocation

    Each agent is modeled as a Beta-Bernoulli bandit.
    We sample from posterior, allocate to highest sample, update beliefs.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_gpu: bool = False,
        timeout: Optional[float] = None,
    ):
        super().__init__(config, enable_gpu, timeout)

        self.horizon = config.get("thompson_sampling_horizon", 100) if config else 100
        self.exploration_rate = config.get("exploration_rate", 0.1) if config else 0.1
        self.budget_buffer = config.get("budget_buffer", 0.05) if config else 0.05

    def initialize(self) -> None:
        self._is_initialized = True
        logger.info("Librex.AllocSolver initialized")

    def get_problem_types(self) -> List[ProblemType]:
        return [ProblemType.ALLOC]

    def estimate_complexity(self, problem: OptimizationProblem) -> str:
        demands = problem.data.get("resource_demands", [])
        if len(demands) <= 10:
            return "low"
        elif len(demands) <= 50:
            return "medium"
        else:
            return "high"

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        if not self._is_initialized:
            self.initialize()

        start_time = time.time()

        # Validate
        is_valid, error = self.validate_problem(problem)
        if not is_valid:
            return OptimizationResult(
                status=SolverStatus.FAILED,
                solution=None,
                objective_value=None,
                metadata={"error": error},
                computation_time=time.time() - start_time,
            )

        # Extract data
        demands = problem.data["resource_demands"]  # List of (agent_id, demand_per_round)
        budget = problem.data["budget_constraint"]  # Total budget available
        rewards = problem.data.get("historical_rewards", None)  # Optional: historical performance

        num_agents = len(demands)

        logger.info(f"Librex.Alloc solving for {num_agents} agents with budget={budget}")

        # Run Thompson Sampling
        allocation, total_reward, rounds = self._thompson_sampling(
            demands, budget, rewards
        )

        computation_time = time.time() - start_time

        # Baseline: equal allocation
        equal_alloc = self._equal_allocation(demands, budget)
        baseline_reward = sum(equal_alloc)  # Simplified reward model

        improvement = ((total_reward - baseline_reward) / baseline_reward * 100) if baseline_reward > 0 else 0.0

        return OptimizationResult(
            status=SolverStatus.SUCCESS,
            solution={
                "allocation": allocation,
                "total_reward": total_reward,
                "method": "thompson_sampling",
            },
            objective_value=-total_reward,  # Negative (we maximize reward)
            metadata={
                "solver": "Librex.AllocSolver",
                "num_agents": num_agents,
                "budget": budget,
                "rounds": rounds,
                "baseline_reward": baseline_reward,
            },
            computation_time=computation_time,
            iterations=rounds,
            improvement_over_baseline=improvement,
        )

    def _thompson_sampling(
        self,
        demands: List[tuple],
        budget: float,
        historical_rewards: Optional[Dict] = None,
    ) -> tuple:
        """
        Thompson Sampling for budget-constrained allocation

        Returns:
            (final_allocation, total_reward, num_rounds)
        """
        num_agents = len(demands)

        # Initialize Beta priors (alpha, beta) for each agent
        # Start with Beta(1, 1) = Uniform prior
        alpha = np.ones(num_agents)
        beta = np.ones(num_agents)

        # If historical rewards provided, use as prior
        if historical_rewards:
            for i, (agent_id, _) in enumerate(demands):
                if agent_id in historical_rewards:
                    successes = historical_rewards[agent_id].get("successes", 0)
                    failures = historical_rewards[agent_id].get("failures", 0)
                    alpha[i] += successes
                    beta[i] += failures

        # Allocation tracker
        allocation = np.zeros(num_agents)
        total_reward = 0.0
        remaining_budget = budget

        for round_idx in range(self.horizon):
            if remaining_budget <= 0:
                break

            # Thompson Sampling: Sample from each agent's posterior
            samples = np.random.beta(alpha, beta)

            # Select agent with highest sample (exploration via randomness in sample)
            selected_agent = np.argmax(samples)

            # Check if we can afford this agent
            agent_id, demand_per_round = demands[selected_agent]
            if demand_per_round > remaining_budget:
                # Can't afford, skip or reduce
                continue

            # Allocate
            allocation[selected_agent] += demand_per_round
            remaining_budget -= demand_per_round

            # Simulate reward (in real system, this would be observed)
            # For now, use sampled probability as reward
            reward = samples[selected_agent]
            total_reward += reward

            # Update posterior (Bayesian update)
            if reward > 0.5:  # Success
                alpha[selected_agent] += 1
            else:  # Failure
                beta[selected_agent] += 1

        return allocation.tolist(), total_reward, round_idx + 1

    def _equal_allocation(self, demands: List[tuple], budget: float) -> List[float]:
        """Baseline: equal allocation across all agents"""
        num_agents = len(demands)
        per_agent_budget = budget / num_agents
        return [per_agent_budget] * num_agents
