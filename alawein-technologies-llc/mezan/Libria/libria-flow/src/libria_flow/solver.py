"""
Librex.Flow Solver - Confidence-Aware Workflow Routing

Solves the problem of routing workflows through agent networks while maximizing
end-to-end validation confidence.

Mathematical Formulation:
    maximize confidence_path = product of confidence scores along path
    subject to: valid path constraints

where:
    - confidence[agent] = historical validation accuracy
    - path = sequence of agents from start to goal
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import heapq

from MEZAN.core import (
    OptimizerInterface,
    OptimizationProblem,
    OptimizationResult,
    ProblemType,
    SolverStatus,
)

logger = logging.getLogger(__name__)


class Librex.FlowSolver(OptimizerInterface):
    """
    Confidence-aware workflow routing using modified Dijkstra's algorithm

    Key innovation: Instead of minimizing distance, we maximize cumulative
    confidence (product of confidence scores along path).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_gpu: bool = False,
        timeout: Optional[float] = None,
    ):
        super().__init__(config, enable_gpu, timeout)

        self.confidence_threshold = config.get("confidence_threshold", 0.7) if config else 0.7
        self.recompute_on_failure = config.get("recompute_on_failure", True) if config else True

    def initialize(self) -> None:
        self._is_initialized = True
        logger.info("Librex.FlowSolver initialized")

    def get_problem_types(self) -> List[ProblemType]:
        return [ProblemType.FLOW]

    def estimate_complexity(self, problem: OptimizationProblem) -> str:
        graph = problem.data.get("workflow_graph", {})
        num_nodes = len(graph.get("nodes", []))

        if num_nodes <= 20:
            return "low"
        elif num_nodes <= 50:
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

        # Extract graph and confidence scores
        graph = problem.data["workflow_graph"]
        confidence_scores = problem.data["confidence_scores"]

        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])  # [(from, to, cost), ...]
        start_node = graph.get("start_node", nodes[0] if nodes else None)
        goal_node = graph.get("goal_node", nodes[-1] if len(nodes) > 1 else None)

        if not start_node or not goal_node:
            return OptimizationResult(
                status=SolverStatus.FAILED,
                solution=None,
                objective_value=None,
                metadata={"error": "No start or goal node specified"},
                computation_time=time.time() - start_time,
            )

        # Find path with maximum confidence
        path, confidence = self._confidence_dijkstra(
            nodes, edges, confidence_scores, start_node, goal_node
        )

        computation_time = time.time() - start_time

        if path is None:
            return OptimizationResult(
                status=SolverStatus.FAILED,
                solution=None,
                objective_value=None,
                metadata={"error": "No valid path found"},
                computation_time=computation_time,
            )

        # Baseline: greedy highest confidence single-step
        baseline_path, baseline_conf = self._greedy_path(
            nodes, edges, confidence_scores, start_node, goal_node
        )

        improvement = ((confidence - baseline_conf) / baseline_conf * 100) if baseline_conf > 0 else 0.0

        return OptimizationResult(
            status=SolverStatus.SUCCESS,
            solution={
                "path": path,
                "confidence": confidence,
                "method": "confidence_dijkstra",
            },
            objective_value=-confidence,  # Negative because we maximize (solver expects minimize)
            metadata={
                "solver": "Librex.FlowSolver",
                "num_nodes": len(nodes),
                "path_length": len(path),
                "baseline_confidence": baseline_conf,
            },
            computation_time=computation_time,
            improvement_over_baseline=improvement,
        )

    def _confidence_dijkstra(
        self,
        nodes: List[str],
        edges: List[Tuple],
        confidence: Dict[str, float],
        start: str,
        goal: str,
    ) -> Tuple[Optional[List[str]], float]:
        """
        Modified Dijkstra to maximize cumulative confidence

        Since confidence = product of scores, we use log-confidence to convert
        multiplication to addition, then maximize.
        """
        # Build adjacency list
        graph = {node: [] for node in nodes}
        for edge in edges:
            if len(edge) == 2:
                from_node, to_node = edge
                cost = 1.0  # Default cost
            else:
                from_node, to_node, cost = edge
            graph[from_node].append((to_node, cost))

        # Priority queue: (-log_confidence, node, path)
        # Use negative because heapq is min-heap
        pq = [(0.0, start, [start])]
        visited = set()
        node_log_conf = {node: np.log(max(confidence.get(node, 0.5), 0.01)) for node in nodes}

        while pq:
            neg_log_conf, current, path = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            # Found goal
            if current == goal:
                total_log_conf = -neg_log_conf
                total_conf = np.exp(total_log_conf)
                return path, total_conf

            # Explore neighbors
            for neighbor, edge_cost in graph.get(current, []):
                if neighbor not in visited:
                    new_log_conf = -neg_log_conf + node_log_conf[neighbor]
                    new_path = path + [neighbor]
                    heapq.heappush(pq, (-new_log_conf, neighbor, new_path))

        return None, 0.0

    def _greedy_path(
        self,
        nodes: List[str],
        edges: List[Tuple],
        confidence: Dict[str, float],
        start: str,
        goal: str,
    ) -> Tuple[List[str], float]:
        """Greedy baseline: always pick highest confidence next neighbor"""
        graph = {node: [] for node in nodes}
        for edge in edges:
            from_node = edge[0]
            to_node = edge[1]
            graph[from_node].append(to_node)

        path = [start]
        current = start
        total_conf = confidence.get(start, 0.5)

        visited = {start}
        while current != goal and len(path) < len(nodes):
            neighbors = [n for n in graph.get(current, []) if n not in visited]
            if not neighbors:
                break

            # Pick neighbor with highest confidence
            best_neighbor = max(neighbors, key=lambda n: confidence.get(n, 0.5))
            path.append(best_neighbor)
            total_conf *= confidence.get(best_neighbor, 0.5)
            visited.add(best_neighbor)
            current = best_neighbor

        return path, total_conf
