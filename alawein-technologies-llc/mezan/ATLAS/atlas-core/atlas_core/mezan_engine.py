"""
MEZAN Engine - Meta-Equilibrium Zero-regret Assignment Network

Implements the dual-solver balancing pattern:
[SOLVER_L] → [ENGINE] ← [SOLVER_R]

The MEZAN engine balances two heterogeneous solvers (e.g., continuous flow +
discrete combinatorial) with adaptive trust/weight allocation.

Core Concepts:
- Meta-Equilibrium: Equilibrium over methods/solvers, not just actions
- Zero-regret: Bandit-style learning to minimize long-run regret
- Assignment: Optimal agent-task assignment and routing
- Network: Multi-agent, multi-solver architecture

Author: MEZAN Research Team
Date: 2025-11-18
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SolverType(Enum):
    """Types of solvers in the MEZAN architecture"""
    CONTINUOUS = "continuous"  # Flow-based, relaxation solvers
    DISCRETE = "discrete"      # Combinatorial, heuristic solvers
    HYBRID = "hybrid"          # Mix of both


@dataclass
class SolverConfig:
    """Configuration for a solver in MEZAN"""
    solver_id: str
    solver_type: SolverType
    weight: float = 0.5  # Initial trust/weight
    max_iterations: int = 1000
    timeout_seconds: float = 10.0
    temperature: float = 1.0  # For exploration/exploitation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverResult:
    """Result from a solver"""
    solver_id: str
    solution: Any
    objective_value: float
    iterations: int
    time_seconds: float
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MezanState:
    """State of the MEZAN engine"""
    iteration: int = 0
    weight_left: float = 0.5
    weight_right: float = 0.5
    cumulative_regret: float = 0.0
    best_objective: float = float('inf')
    best_solution: Any = None
    history: List[Dict[str, Any]] = field(default_factory=list)


class BaseSolver(ABC):
    """Abstract base class for MEZAN solvers"""

    def __init__(self, config: SolverConfig):
        self.config = config

    @abstractmethod
    def solve(self, problem: Dict[str, Any]) -> SolverResult:
        """Solve the given problem"""
        pass

    @abstractmethod
    def warm_start(self, hint: Any) -> None:
        """Provide a warm start hint from another solver"""
        pass


class MezanEngine:
    """
    MEZAN (Meta-Equilibrium Zero-regret Assignment Network) Engine

    Balances two heterogeneous solvers with adaptive trust allocation:

    ```
           Solver_L                  Solver_R
            (left)                    (right)
              \\                          /
               \\                        /
                    [   MEZAN   ]
                    [  ENGINE   ]
               /                        \\
              /                          \\
        feedback_L                  feedback_R
    ```

    The engine:
    1. Runs both solvers in parallel (or sequentially)
    2. Evaluates their solutions
    3. Updates trust/weights based on performance
    4. Exchanges information between solvers
    5. Learns optimal allocation policy over time
    """

    def __init__(
        self,
        solver_left: BaseSolver,
        solver_right: BaseSolver,
        balance_strategy: str = "ucb",  # ucb, thompson, epsilon_greedy
        learning_rate: float = 0.1,
        exploration_param: float = 2.0,
    ):
        """
        Initialize MEZAN engine

        Args:
            solver_left: Left solver (e.g., continuous/flow-based)
            solver_right: Right solver (e.g., discrete/combinatorial)
            balance_strategy: Strategy for balancing solvers
            learning_rate: Learning rate for weight updates
            exploration_param: Exploration parameter for bandits
        """
        self.solver_left = solver_left
        self.solver_right = solver_right
        self.balance_strategy = balance_strategy
        self.learning_rate = learning_rate
        self.exploration_param = exploration_param

        self.state = MezanState(
            weight_left=solver_left.config.weight,
            weight_right=solver_right.config.weight,
        )

        # Normalize weights
        self._normalize_weights()

        logger.info(
            f"MEZAN Engine initialized with {solver_left.config.solver_id} "
            f"<-> {solver_right.config.solver_id}"
        )

    def _normalize_weights(self):
        """Normalize weights to sum to 1"""
        total = self.state.weight_left + self.state.weight_right
        if total > 0:
            self.state.weight_left /= total
            self.state.weight_right /= total

    def balance_step(
        self,
        problem: Dict[str, Any],
        parallel: bool = False,
    ) -> Tuple[SolverResult, SolverResult, SolverResult]:
        """
        Execute one balancing step

        Args:
            problem: Problem specification
            parallel: Whether to run solvers in parallel

        Returns:
            Tuple of (left_result, right_result, balanced_result)
        """
        start_time = time.time()

        # Step 1: Run both solvers
        if parallel:
            # TODO: Implement true parallel execution
            result_left = self.solver_left.solve(problem)
            result_right = self.solver_right.solve(problem)
        else:
            result_left = self.solver_left.solve(problem)
            # Optionally provide warm start from left to right
            self.solver_right.warm_start(result_left.solution)
            result_right = self.solver_right.solve(problem)

        # Step 2: Evaluate and compare
        delta_objective = result_left.objective_value - result_right.objective_value

        # Step 3: Select best solution (or blend)
        if result_left.objective_value < result_right.objective_value:
            best_result = result_left
            best_solver = "left"
        else:
            best_result = result_right
            best_solver = "right"

        # Step 4: Update weights using bandit algorithm
        self._update_weights(result_left, result_right, delta_objective)

        # Step 5: Update state
        self.state.iteration += 1
        if best_result.objective_value < self.state.best_objective:
            self.state.best_objective = best_result.objective_value
            self.state.best_solution = best_result.solution

        # Step 6: Record history
        self.state.history.append({
            "iteration": self.state.iteration,
            "weight_left": self.state.weight_left,
            "weight_right": self.state.weight_right,
            "objective_left": result_left.objective_value,
            "objective_right": result_right.objective_value,
            "delta": delta_objective,
            "best_solver": best_solver,
            "time": time.time() - start_time,
        })

        # Step 7: Create balanced result
        balanced_result = SolverResult(
            solver_id="mezan_balanced",
            solution=best_result.solution,
            objective_value=best_result.objective_value,
            iterations=result_left.iterations + result_right.iterations,
            time_seconds=time.time() - start_time,
            confidence=(
                self.state.weight_left * result_left.confidence +
                self.state.weight_right * result_right.confidence
            ),
            metadata={
                "left_objective": result_left.objective_value,
                "right_objective": result_right.objective_value,
                "weight_left": self.state.weight_left,
                "weight_right": self.state.weight_right,
                "best_solver": best_solver,
            }
        )

        logger.info(
            f"MEZAN step {self.state.iteration}: "
            f"L={result_left.objective_value:.4f} (w={self.state.weight_left:.3f}), "
            f"R={result_right.objective_value:.4f} (w={self.state.weight_right:.3f}), "
            f"Best={best_solver}"
        )

        return result_left, result_right, balanced_result

    def _update_weights(
        self,
        result_left: SolverResult,
        result_right: SolverResult,
        delta: float,
    ):
        """
        Update solver weights based on performance

        Uses bandit-style learning (UCB, Thompson Sampling, or ε-greedy)
        """
        if self.balance_strategy == "ucb":
            self._update_weights_ucb(result_left, result_right)
        elif self.balance_strategy == "thompson":
            self._update_weights_thompson(result_left, result_right)
        elif self.balance_strategy == "epsilon_greedy":
            self._update_weights_epsilon_greedy(result_left, result_right)
        else:
            # Simple performance-based update
            self._update_weights_simple(result_left, result_right, delta)

        self._normalize_weights()

    def _update_weights_simple(
        self,
        result_left: SolverResult,
        result_right: SolverResult,
        delta: float,
    ):
        """Simple gradient-like weight update"""
        # If left is better (delta < 0), increase left weight
        # If right is better (delta > 0), increase right weight
        shift = -delta * self.learning_rate

        self.state.weight_left += shift
        self.state.weight_right -= shift

        # Clip to [0.1, 0.9] to maintain exploration
        self.state.weight_left = np.clip(self.state.weight_left, 0.1, 0.9)
        self.state.weight_right = np.clip(self.state.weight_right, 0.1, 0.9)

    def _update_weights_ucb(
        self,
        result_left: SolverResult,
        result_right: SolverResult,
    ):
        """Update weights using Upper Confidence Bound (UCB)"""
        # Track solver performance for UCB calculation
        n = self.state.iteration + 1

        # Rewards (negative objective for minimization)
        reward_left = -result_left.objective_value
        reward_right = -result_right.objective_value

        # Simple UCB update (simplified for demonstration)
        bonus = self.exploration_param * np.sqrt(np.log(n) / n)

        ucb_left = reward_left + bonus
        ucb_right = reward_right + bonus

        # Softmax to convert to weights
        temp = 0.1
        exp_left = np.exp(ucb_left / temp)
        exp_right = np.exp(ucb_right / temp)

        self.state.weight_left = exp_left / (exp_left + exp_right)
        self.state.weight_right = exp_right / (exp_left + exp_right)

    def _update_weights_thompson(
        self,
        result_left: SolverResult,
        result_right: SolverResult,
    ):
        """Update weights using Thompson Sampling (Bayesian)"""
        # Use solver confidence as prior
        self.state.weight_left = result_left.confidence
        self.state.weight_right = result_right.confidence

    def _update_weights_epsilon_greedy(
        self,
        result_left: SolverResult,
        result_right: SolverResult,
    ):
        """Update weights using ε-greedy strategy"""
        epsilon = 0.1

        if np.random.random() < epsilon:
            # Explore: random weights
            self.state.weight_left = np.random.random()
            self.state.weight_right = 1 - self.state.weight_left
        else:
            # Exploit: choose best
            if result_left.objective_value < result_right.objective_value:
                self.state.weight_left = 0.9
                self.state.weight_right = 0.1
            else:
                self.state.weight_left = 0.1
                self.state.weight_right = 0.9

    def solve_with_balance(
        self,
        problem: Dict[str, Any],
        max_iterations: int = 10,
        convergence_threshold: float = 1e-4,
    ) -> SolverResult:
        """
        Solve problem using MEZAN balancing over multiple iterations

        Args:
            problem: Problem specification
            max_iterations: Maximum balancing iterations
            convergence_threshold: Stop if improvement < threshold

        Returns:
            Best solution found
        """
        logger.info(f"MEZAN solving with up to {max_iterations} balance iterations")

        prev_objective = float('inf')

        for i in range(max_iterations):
            _, _, balanced = self.balance_step(problem)

            # Check convergence
            improvement = prev_objective - balanced.objective_value
            if improvement < convergence_threshold and i > 0:
                logger.info(
                    f"MEZAN converged after {i+1} iterations "
                    f"(improvement={improvement:.6f})"
                )
                break

            prev_objective = balanced.objective_value

        # Return best result found
        best_result = SolverResult(
            solver_id="mezan_final",
            solution=self.state.best_solution,
            objective_value=self.state.best_objective,
            iterations=self.state.iteration,
            time_seconds=sum(h["time"] for h in self.state.history),
            confidence=max(self.state.weight_left, self.state.weight_right),
            metadata={
                "total_iterations": self.state.iteration,
                "final_weight_left": self.state.weight_left,
                "final_weight_right": self.state.weight_right,
                "convergence_history": self.state.history,
            }
        )

        logger.info(
            f"MEZAN final: objective={best_result.objective_value:.4f}, "
            f"iterations={best_result.iterations}"
        )

        return best_result

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get engine diagnostics and statistics"""
        return {
            "iteration": self.state.iteration,
            "weight_left": self.state.weight_left,
            "weight_right": self.state.weight_right,
            "best_objective": self.state.best_objective,
            "cumulative_regret": self.state.cumulative_regret,
            "history_length": len(self.state.history),
        }


# Example solver implementations for testing
class MockContinuousSolver(BaseSolver):
    """Mock continuous/relaxation solver"""

    def solve(self, problem: Dict[str, Any]) -> SolverResult:
        # Simulate continuous relaxation solving
        time.sleep(0.01)
        return SolverResult(
            solver_id=self.config.solver_id,
            solution={"type": "continuous", "value": np.random.rand(10)},
            objective_value=np.random.uniform(0.4, 0.6),
            iterations=100,
            time_seconds=0.01,
            confidence=0.7,
        )

    def warm_start(self, hint: Any) -> None:
        pass  # Mock implementation


class MockDiscreteSolver(BaseSolver):
    """Mock discrete/combinatorial solver"""

    def solve(self, problem: Dict[str, Any]) -> SolverResult:
        # Simulate discrete combinatorial solving
        time.sleep(0.01)
        return SolverResult(
            solver_id=self.config.solver_id,
            solution={"type": "discrete", "permutation": list(range(10))},
            objective_value=np.random.uniform(0.3, 0.7),
            iterations=50,
            time_seconds=0.01,
            confidence=0.8,
        )

    def warm_start(self, hint: Any) -> None:
        # Could use hint to initialize search
        pass
