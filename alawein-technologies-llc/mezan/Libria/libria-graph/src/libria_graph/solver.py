"""
Librex.Graph Solver - Information-Theoretic Network Topology Optimization

Optimizes agent communication network topology using information-theoretic
objectives (minimize entropy, maximize mutual information).

Mathematical Formulation:
    minimize: H(X) = -sum p(x) log p(x)  (entropy)
    subject to: graph connectivity constraints
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


class Librex.GraphSolver(OptimizerInterface):
    """
    Network topology optimization using entropy minimization

    Key idea: Agents that frequently communicate should be closer in the network.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_gpu: bool = False,
        timeout: Optional[float] = None,
    ):
        super().__init__(config, enable_gpu, timeout)

        self.objective = config.get("objective", "minimize_entropy") if config else "minimize_entropy"
        self.max_graph_size = config.get("max_graph_size", 100) if config else 100

    def initialize(self) -> None:
        self._is_initialized = True
        logger.info("Librex.GraphSolver initialized")

    def get_problem_types(self) -> List[ProblemType]:
        return [ProblemType.GRAPH]

    def estimate_complexity(self, problem: OptimizationProblem) -> str:
        nodes = problem.data.get("nodes", [])
        if len(nodes) <= 20:
            return "low"
        elif len(nodes) <= 50:
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
        nodes = problem.data.get("nodes", [])
        communication_matrix = np.array(problem.data.get("communication_matrix", []))

        # Optimize topology
        adjacency, entropy = self._optimize_topology(nodes, communication_matrix)

        computation_time = time.time() - start_time

        # Baseline: fully connected graph
        baseline_adjacency = np.ones((len(nodes), len(nodes))) - np.eye(len(nodes))
        baseline_entropy = self._compute_entropy(baseline_adjacency, communication_matrix)

        improvement = ((baseline_entropy - entropy) / baseline_entropy * 100) if baseline_entropy > 0 else 0.0

        return OptimizationResult(
            status=SolverStatus.SUCCESS,
            solution={
                "adjacency_matrix": adjacency.tolist(),
                "entropy": entropy,
                "method": self.objective,
            },
            objective_value=entropy,
            metadata={
                "solver": "Librex.GraphSolver",
                "num_nodes": len(nodes),
                "baseline_entropy": baseline_entropy,
            },
            computation_time=computation_time,
            improvement_over_baseline=improvement,
        )

    def _optimize_topology(self, nodes: List[str], comm_matrix: np.ndarray) -> tuple:
        """
        Optimize network topology using greedy entropy minimization

        Start with empty graph, add edges that minimize entropy.
        """
        n = len(nodes)
        adjacency = np.zeros((n, n))

        # Greedy edge addition
        max_edges = n * (n - 1) // 2
        for _ in range(max_edges):
            best_i, best_j = -1, -1
            best_entropy = float('inf')

            # Try adding each possible edge
            for i in range(n):
                for j in range(i + 1, n):
                    if adjacency[i, j] == 0:  # Not yet added
                        # Temporarily add edge
                        adjacency[i, j] = 1
                        adjacency[j, i] = 1

                        entropy = self._compute_entropy(adjacency, comm_matrix)

                        if entropy < best_entropy:
                            best_entropy = entropy
                            best_i, best_j = i, j

                        # Remove edge
                        adjacency[i, j] = 0
                        adjacency[j, i] = 0

            if best_i >= 0:
                # Add best edge
                adjacency[best_i, best_j] = 1
                adjacency[best_j, best_i] = 1

        final_entropy = self._compute_entropy(adjacency, comm_matrix)
        return adjacency, final_entropy

    def _compute_entropy(self, adjacency: np.ndarray, comm_matrix: np.ndarray) -> float:
        """
        Compute information entropy of network

        H = -sum p(edge) * log(p(edge))
        where p(edge) = communication_freq / total_communication
        """
        # Communication frequency weighted by adjacency
        weighted_comm = adjacency * comm_matrix

        total_comm = np.sum(weighted_comm)
        if total_comm == 0:
            return float('inf')  # No communication

        probabilities = weighted_comm.flatten() / total_comm
        probabilities = probabilities[probabilities > 0]  # Remove zeros

        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        return entropy
