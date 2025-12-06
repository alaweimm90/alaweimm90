"""
Librex.Meta Solver - Meta-Learning for Automatic Algorithm Selection

The "solver of solvers" that uses UCB (Upper Confidence Bound) and problem
features to automatically select the best Libria solver for each problem.

Mathematical Formulation:
    Select solver s that maximizes: UCB(s) = μ(s) + sqrt(2*ln(t)/n(s))

where:
    - μ(s) = average performance of solver s
    - n(s) = number of times solver s was selected
    - t = total number of selections
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


class Librex.MetaSolver(OptimizerInterface):
    """
    Meta-learning solver using UCB for algorithm selection

    Maintains performance statistics for each available solver and uses
    UCB to balance exploration (trying underused solvers) and exploitation
    (using best-performing solvers).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_gpu: bool = False,
        timeout: Optional[float] = None,
    ):
        super().__init__(config, enable_gpu, timeout)

        self.feature_extraction = config.get("feature_extraction", "problem_signature") if config else "problem_signature"
        self.selection_algorithm = config.get("selection_algorithm", "ucb") if config else "ucb"
        self.online_learning = config.get("online_learning", True) if config else True

        # Solver performance tracking
        self.solver_stats = {
            "Librex.QAP": {"count": 0, "total_reward": 0.0, "avg_reward": 0.0},
            "Librex.Flow": {"count": 0, "total_reward": 0.0, "avg_reward": 0.0},
            "Librex.Alloc": {"count": 0, "total_reward": 0.0, "avg_reward": 0.0},
            "Librex.Graph": {"count": 0, "total_reward": 0.0, "avg_reward": 0.0},
            "Librex.Dual": {"count": 0, "total_reward": 0.0, "avg_reward": 0.0},
            "Librex.Evo": {"count": 0, "total_reward": 0.0, "avg_reward": 0.0},
        }

        self.total_selections = 0

    def initialize(self) -> None:
        self._is_initialized = True
        logger.info("Librex.MetaSolver initialized")

    def get_problem_types(self) -> List[ProblemType]:
        # Meta solver can handle all types by delegating
        return list(ProblemType)

    def estimate_complexity(self, problem: OptimizationProblem) -> str:
        # Meta solver complexity is low (just selection)
        return "low"

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

        # Extract problem features
        features = self._extract_features(problem)

        # Select best solver using UCB
        selected_solver = self._ucb_selection(features)

        logger.info(f"Librex.Meta selected solver: {selected_solver}")

        # In practice, would delegate to selected solver here
        # For now, return the selection
        computation_time = time.time() - start_time

        # Update statistics (simulated feedback)
        if self.online_learning:
            reward = np.random.rand()  # Placeholder (real reward from actual solver)
            self._update_statistics(selected_solver, reward)

        return OptimizationResult(
            status=SolverStatus.SUCCESS,
            solution={
                "selected_solver": selected_solver,
                "features": features,
                "method": self.selection_algorithm,
            },
            objective_value=None,
            metadata={
                "solver": "Librex.MetaSolver",
                "solver_stats": self.solver_stats.copy(),
                "total_selections": self.total_selections,
            },
            computation_time=computation_time,
        )

    def _extract_features(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """
        Extract problem features for meta-learning

        Features might include:
        - Problem type
        - Problem size
        - Data characteristics (sparsity, symmetry, etc.)
        """
        features = {
            "problem_type": problem.problem_type.value,
            "has_constraints": problem.constraints is not None,
            "num_objectives": len(problem.objectives) if problem.objectives else 1,
        }

        # Problem-type-specific features
        if problem.problem_type == ProblemType.QAP:
            dist_matrix = problem.data.get("distance_matrix", [])
            features["problem_size"] = len(dist_matrix)
            features["is_symmetric"] = self._is_symmetric(dist_matrix) if dist_matrix else False

        elif problem.problem_type == ProblemType.FLOW:
            graph = problem.data.get("workflow_graph", {})
            features["num_nodes"] = len(graph.get("nodes", []))
            features["num_edges"] = len(graph.get("edges", []))

        elif problem.problem_type == ProblemType.ALLOC:
            demands = problem.data.get("resource_demands", [])
            features["num_agents"] = len(demands)
            features["budget"] = problem.data.get("budget_constraint", 0.0)

        return features

    def _is_symmetric(self, matrix: List[List]) -> bool:
        """Check if matrix is symmetric"""
        try:
            arr = np.array(matrix)
            return np.allclose(arr, arr.T)
        except:
            return False

    def _ucb_selection(self, features: Dict[str, Any]) -> str:
        """
        UCB (Upper Confidence Bound) algorithm for solver selection

        UCB(s) = μ(s) + c * sqrt(ln(t) / n(s))

        where:
        - μ(s) = average reward for solver s
        - n(s) = number of times solver s was selected
        - t = total number of selections
        - c = exploration constant (typically sqrt(2))
        """
        c = np.sqrt(2)  # Exploration constant

        # Filter solvers by problem type compatibility
        problem_type = features.get("problem_type", "")
        compatible_solvers = self._get_compatible_solvers(problem_type)

        if not compatible_solvers:
            # Default to Librex.QAP
            return "Librex.QAP"

        # Calculate UCB for each compatible solver
        ucb_scores = {}
        for solver in compatible_solvers:
            stats = self.solver_stats[solver]
            count = stats["count"]
            avg_reward = stats["avg_reward"]

            if count == 0:
                # Untriedsolvers get infinite UCB (explore first)
                ucb_scores[solver] = float('inf')
            else:
                # UCB formula
                exploration_bonus = c * np.sqrt(np.log(self.total_selections + 1) / count)
                ucb_scores[solver] = avg_reward + exploration_bonus

        # Select solver with highest UCB
        selected_solver = max(ucb_scores, key=ucb_scores.get)

        self.total_selections += 1

        return selected_solver

    def _get_compatible_solvers(self, problem_type: str) -> List[str]:
        """Get solvers compatible with problem type"""
        compatibility = {
            "quadratic_assignment": ["Librex.QAP"],
            "workflow_routing": ["Librex.Flow"],
            "resource_allocation": ["Librex.Alloc"],
            "network_topology": ["Librex.Graph"],
            "adversarial_robust": ["Librex.Dual"],
            "multi_objective": ["Librex.Evo"],
        }

        return compatibility.get(problem_type, list(self.solver_stats.keys()))

    def _update_statistics(self, solver: str, reward: float) -> None:
        """
        Update solver performance statistics

        Args:
            solver: Solver name
            reward: Observed reward (1.0 = excellent, 0.0 = poor)
        """
        if solver not in self.solver_stats:
            logger.warning(f"Unknown solver: {solver}")
            return

        stats = self.solver_stats[solver]
        stats["count"] += 1
        stats["total_reward"] += reward
        stats["avg_reward"] = stats["total_reward"] / stats["count"]

        logger.debug(
            f"Updated {solver} stats: count={stats['count']}, "
            f"avg_reward={stats['avg_reward']:.3f}"
        )

    def get_solver_rankings(self) -> List[tuple]:
        """
        Get current solver rankings by average reward

        Returns:
            List of (solver_name, avg_reward, count) tuples, sorted by avg_reward
        """
        rankings = [
            (solver, stats["avg_reward"], stats["count"])
            for solver, stats in self.solver_stats.items()
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def reset_statistics(self) -> None:
        """Reset all solver statistics (useful for testing)"""
        for solver in self.solver_stats:
            self.solver_stats[solver] = {
                "count": 0,
                "total_reward": 0.0,
                "avg_reward": 0.0,
            }
        self.total_selections = 0
        logger.info("Librex.Meta statistics reset")
