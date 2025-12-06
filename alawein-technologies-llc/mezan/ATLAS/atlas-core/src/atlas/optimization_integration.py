"""
ORCHEX Integration with MEZAN Libria Suite

This module integrates MEZAN optimization solvers with ORCHEX research orchestration,
replacing simple heuristics with optimal agent assignment.

Key integrations:
1. Agent-Task Assignment (Librex.QAP)
2. Workflow Routing (Librex.Flow)
3. Resource Allocation (Librex.Alloc)
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

# Import MEZAN core
from MEZAN.core import (
    OptimizerFactory,
    OptimizationProblem,
    OptimizationResult,
    ProblemType,
    get_optimizer_factory,
)

logger = logging.getLogger(__name__)


class ATLASOptimizationManager:
    """
    Manages optimization for ORCHEX workflows

    This class provides high-level methods for ORCHEX to use MEZAN solvers
    without needing to understand optimization details.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize optimization manager

        Args:
            config: MEZAN optimizer configuration (feature flags, solver configs, etc.)
        """
        self.factory = get_optimizer_factory(config)
        logger.info("ORCHEX Optimization Manager initialized")

    def optimize_agent_assignment(
        self,
        agents: List[Dict[str, Any]],
        tasks: List[Dict[str, Any]],
        synergy_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Optimize assignment of agents to research tasks

        Uses Librex.QAP to find optimal assignment considering agent synergies.

        Args:
            agents: List of agent dictionaries with capabilities
            tasks: List of task dictionaries with requirements
            synergy_matrix: Optional pre-computed synergy matrix

        Returns:
            Dictionary with assignment and metadata
        """
        n_agents = len(agents)
        n_tasks = len(tasks)

        if n_agents != n_tasks:
            logger.warning(
                f"Agent-task mismatch: {n_agents} agents, {n_tasks} tasks. "
                "Padding to make equal."
            )
            # Pad smaller list
            if n_agents < n_tasks:
                agents = agents + [{"id": f"dummy_{i}", "capabilities": []}
                                  for i in range(n_tasks - n_agents)]
            else:
                tasks = tasks + [{"id": f"dummy_{i}", "requirements": []}
                                for i in range(n_agents - n_tasks)]

        n = max(n_agents, n_tasks)

        # Compute distance matrix (task dissimilarity)
        distance_matrix = self._compute_task_distance_matrix(tasks)

        # Compute flow matrix (agent synergy)
        if synergy_matrix is None:
            flow_matrix = self._compute_agent_synergy_matrix(agents)
        else:
            flow_matrix = synergy_matrix

        # Create QAP problem
        problem = OptimizationProblem(
            problem_type=ProblemType.QAP,
            data={
                "distance_matrix": distance_matrix.tolist(),
                "flow_matrix": flow_matrix.tolist(),
            },
            metadata={
                "source": "ORCHEX agent assignment",
                "n_agents": n_agents,
                "n_tasks": n_tasks,
            },
        )

        # Solve using factory (will use Librex.QAP if enabled, heuristic fallback otherwise)
        optimizer = self.factory.create_optimizer(problem, timeout=30.0)
        result = optimizer.solve(problem)

        # Format result for ORCHEX
        if result.status.value == "success":
            assignment = result.solution["assignment"]

            # Build agent-task mapping
            agent_task_map = {}
            for task_idx, agent_idx in enumerate(assignment):
                if task_idx < len(tasks) and agent_idx < len(agents):
                    task_id = tasks[task_idx]["id"]
                    agent_id = agents[agent_idx]["id"]
                    agent_task_map[task_id] = agent_id

            return {
                "status": "success",
                "assignment": agent_task_map,
                "objective_value": result.objective_value,
                "computation_time": result.computation_time,
                "solver_used": result.metadata.get("solver", "unknown"),
                "improvement": result.improvement_over_baseline,
            }
        else:
            return {
                "status": "failed",
                "error": result.metadata.get("error", "Optimization failed"),
                "fallback_assignment": self._random_assignment(agents, tasks),
            }

    def optimize_workflow_routing(
        self,
        workflow_graph: Dict[str, Any],
        agent_confidence: Dict[str, float],
        start_task: str,
        goal_task: str,
    ) -> Dict[str, Any]:
        """
        Optimize routing of research workflow through agent network

        Uses Librex.Flow to find path with maximum end-to-end confidence.

        Args:
            workflow_graph: Graph structure (nodes, edges)
            agent_confidence: Historical validation accuracy per agent
            start_task: Starting task ID
            goal_task: Goal task ID

        Returns:
            Dictionary with optimal path and metadata
        """
        problem = OptimizationProblem(
            problem_type=ProblemType.FLOW,
            data={
                "workflow_graph": {
                    **workflow_graph,
                    "start_node": start_task,
                    "goal_node": goal_task,
                },
                "confidence_scores": agent_confidence,
            },
            metadata={"source": "ORCHEX workflow routing"},
        )

        optimizer = self.factory.create_optimizer(problem, timeout=10.0)
        result = optimizer.solve(problem)

        if result.status.value == "success":
            return {
                "status": "success",
                "path": result.solution["path"],
                "confidence": result.solution["confidence"],
                "computation_time": result.computation_time,
                "solver_used": result.metadata.get("solver", "unknown"),
            }
        else:
            return {
                "status": "failed",
                "error": result.metadata.get("error", "Routing failed"),
                "fallback_path": [start_task, goal_task],  # Direct path
            }

    def optimize_resource_allocation(
        self,
        agent_demands: List[tuple],
        total_budget: float,
        historical_performance: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize allocation of API budget across agents

        Uses Librex.Alloc with Thompson Sampling to learn optimal allocation.

        Args:
            agent_demands: List of (agent_id, demand_per_round) tuples
            total_budget: Total API credit budget
            historical_performance: Optional historical reward data

        Returns:
            Dictionary with allocation and metadata
        """
        problem = OptimizationProblem(
            problem_type=ProblemType.ALLOC,
            data={
                "resource_demands": agent_demands,
                "budget_constraint": total_budget,
                "historical_rewards": historical_performance,
            },
            metadata={"source": "ORCHEX resource allocation"},
        )

        optimizer = self.factory.create_optimizer(problem, timeout=5.0)
        result = optimizer.solve(problem)

        if result.status.value == "success":
            allocation = result.solution["allocation"]

            # Format as agent_id -> allocation mapping
            allocation_map = {
                agent_id: alloc
                for (agent_id, _), alloc in zip(agent_demands, allocation)
            }

            return {
                "status": "success",
                "allocation": allocation_map,
                "total_reward": result.solution.get("total_reward"),
                "computation_time": result.computation_time,
                "solver_used": result.metadata.get("solver", "unknown"),
            }
        else:
            # Equal allocation fallback
            equal_alloc = total_budget / len(agent_demands)
            return {
                "status": "failed",
                "error": result.metadata.get("error", "Allocation failed"),
                "fallback_allocation": {
                    agent_id: equal_alloc for agent_id, _ in agent_demands
                },
            }

    # Helper methods

    def _compute_task_distance_matrix(self, tasks: List[Dict]) -> np.ndarray:
        """
        Compute distance matrix representing task dissimilarity

        Tasks that are very different should have high distance.
        This encourages assigning similar agents to similar tasks.
        """
        n = len(tasks)
        distance = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Simple heuristic: tasks with different requirements are distant
                task_i_reqs = set(tasks[i].get("requirements", []))
                task_j_reqs = set(tasks[j].get("requirements", []))

                # Jaccard distance
                union = task_i_reqs.union(task_j_reqs)
                intersection = task_i_reqs.intersection(task_j_reqs)

                if len(union) > 0:
                    jaccard_sim = len(intersection) / len(union)
                    jaccard_dist = 1.0 - jaccard_sim
                else:
                    jaccard_dist = 0.0

                distance[i, j] = jaccard_dist * 100  # Scale for QAP
                distance[j, i] = distance[i, j]  # Symmetric

        return distance

    def _compute_agent_synergy_matrix(self, agents: List[Dict]) -> np.ndarray:
        """
        Compute flow matrix representing agent synergies

        Agents with overlapping capabilities have high synergy.
        This encourages assigning synergistic agents to nearby tasks.
        """
        n = len(agents)
        synergy = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                agent_i_caps = set(agents[i].get("capabilities", []))
                agent_j_caps = set(agents[j].get("capabilities", []))

                # Overlap = synergy
                overlap = agent_i_caps.intersection(agent_j_caps)
                synergy_value = len(overlap) * 10  # Scale for QAP

                synergy[i, j] = synergy_value
                synergy[j, i] = synergy_value  # Symmetric

        return synergy

    def _random_assignment(self, agents: List[Dict], tasks: List[Dict]) -> Dict[str, str]:
        """Random fallback assignment"""
        n = min(len(agents), len(tasks))
        agent_ids = [a["id"] for a in agents[:n]]
        task_ids = [t["id"] for t in tasks[:n]]

        np.random.shuffle(agent_ids)

        return {task_id: agent_id for task_id, agent_id in zip(task_ids, agent_ids)}


# Convenience function for ORCHEX to use
def optimize_atlas_agent_assignment(
    agents: List[Dict],
    tasks: List[Dict],
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Convenience function for ORCHEX agent assignment optimization

    Example usage in ORCHEX:
        from ORCHEX.optimization_integration import optimize_atlas_agent_assignment

        agents = [
            {"id": "synthesis_agent", "capabilities": ["synthesis", "reasoning"]},
            {"id": "literature_agent", "capabilities": ["search", "analysis"]},
            ...
        ]

        tasks = [
            {"id": "hypothesis_generation", "requirements": ["reasoning", "creativity"]},
            {"id": "literature_review", "requirements": ["search", "analysis"]},
            ...
        ]

        result = optimize_atlas_agent_assignment(agents, tasks)
        assignment = result["assignment"]  # {"hypothesis_generation": "synthesis_agent", ...}
    """
    manager = ATLASOptimizationManager(config)
    return manager.optimize_agent_assignment(agents, tasks)
