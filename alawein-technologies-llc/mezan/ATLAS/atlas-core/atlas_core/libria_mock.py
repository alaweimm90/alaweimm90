"""
Mock LibriaRouter for ORCHEX Development

This mock router allows ORCHEX to be developed and tested before
the real Libria integration is complete. It implements the same
interface as the real LibriaRouter but uses simple heuristics.

Replace with real LibriaRouter once Libria integration is ready:
    from libria_integration import LibriaRouter
"""

from typing import Dict, List, Any
import numpy as np
import logging
import random

logger = logging.getLogger(__name__)


class MockLibriaRouter:
    """
    Mock implementation of LibriaRouter for development

    Provides same interface as real LibriaRouter but uses
    simple heuristics instead of optimization solvers.
    """

    def __init__(self):
        logger.info("Initialized MockLibriaRouter (development mode)")

    def solve_assignment(
        self, agents: List[Any], tasks: List[Dict]
    ) -> Dict:
        """
        Mock agent-task assignment

        Args:
            agents: List of ResearchAgent instances
            tasks: List of task dictionaries

        Returns:
            Assignment result with agent_id
        """
        if not agents or not tasks:
            return {"agent_id": None, "assignment": [], "cost": float("inf")}

        # Simple mock: assign to agent with lowest workload
        best_agent = min(
            agents, key=lambda a: a.current_workload / a.config.max_tasks
        )

        logger.debug(
            f"MockLibria assigned task to {best_agent.config.agent_id} "
            f"(workload: {best_agent.current_workload}/{best_agent.config.max_tasks})"
        )

        return {
            "agent_id": best_agent.config.agent_id,
            "assignment": [0],  # First task to first agent
            "cost": random.uniform(0.3, 0.7),  # Mock cost
        }

    def route_workflow_step(self, workflow_state: Dict) -> str:
        """
        Mock workflow routing

        Args:
            workflow_state: Current workflow state with available agents

        Returns:
            Selected agent_id
        """
        available_agents = workflow_state.get("available_agents", [])

        if not available_agents:
            raise ValueError("No available agents for workflow routing")

        # Simple mock: select agent with highest skill level
        best_agent = max(available_agents, key=lambda a: a.config.skill_level)

        logger.debug(
            f"MockLibria routed workflow to {best_agent.config.agent_id} "
            f"(skill: {best_agent.config.skill_level:.2f})"
        )

        return best_agent.config.agent_id

    def allocate_resources(
        self, agents: List[Any], total_budget: Dict[str, float]
    ) -> Dict[int, np.ndarray]:
        """
        Mock resource allocation

        Args:
            agents: List of agents
            total_budget: Total available resources

        Returns:
            Allocation map: {agent_idx: resource_array}
        """
        n_agents = len(agents)
        allocation = {}

        # Simple mock: equal allocation
        compute_per_agent = total_budget.get("compute", 10.0) / n_agents
        memory_per_agent = total_budget.get("memory", 32.0) / n_agents
        bandwidth_per_agent = total_budget.get("bandwidth", 1000.0) / n_agents

        for i in range(n_agents):
            allocation[i] = np.array(
                [compute_per_agent, memory_per_agent, bandwidth_per_agent]
            )

        logger.debug(
            f"MockLibria allocated resources equally to {n_agents} agents"
        )

        return allocation

    def optimize_topology(self, agent_states: Any) -> np.ndarray:
        """
        Mock topology optimization

        Args:
            agent_states: Agent state time series (T × n × d)

        Returns:
            Adjacency matrix (n × n)
        """
        # Extract number of agents (assuming agent_states is list-like)
        if isinstance(agent_states, np.ndarray):
            n_agents = agent_states.shape[1] if len(agent_states.shape) > 1 else 1
        else:
            n_agents = len(agent_states) if agent_states else 1

        # Simple mock: create random sparse topology
        adjacency = np.random.rand(n_agents, n_agents)
        adjacency = (adjacency > 0.7).astype(float)  # Sparse connectivity
        np.fill_diagonal(adjacency, 0)  # No self-loops

        logger.debug(
            f"MockLibria created topology with "
            f"{int(adjacency.sum())} connections for {n_agents} agents"
        )

        return adjacency

    def validate_workflow(self, workflow_spec: Dict) -> Dict:
        """
        Mock workflow validation

        Args:
            workflow_spec: Workflow specification with test cases

        Returns:
            Validation report
        """
        # Simple mock: random validation results
        test_cases = workflow_spec.get("test_cases", [])

        validation_report = {
            "attack_success_rate": random.uniform(0.05, 0.15),
            "failed_cases": test_cases[:1] if test_cases else [],
            "robustified_workflow": workflow_spec.get("workflow"),
            "certified_radius": random.uniform(0.3, 0.5),
        }

        logger.debug(
            f"MockLibria validated workflow: "
            f"{validation_report['attack_success_rate']:.2%} attack success rate"
        )

        return validation_report

    def evolve_architecture(self, task_distribution: List[Dict]) -> Dict:
        """
        Mock architecture evolution

        Args:
            task_distribution: Sample tasks for evaluation

        Returns:
            Evolution results with archive
        """
        # Simple mock: return dummy archive
        archive = {
            (0, 0): {"architecture": "baseline", "quality": 0.7},
            (1, 1): {"architecture": "specialized", "quality": 0.85},
            (0, 1): {"architecture": "hybrid", "quality": 0.78},
        }

        results = {
            "archive": archive,
            "coverage": 0.6,
            "max_quality": 0.85,
        }

        logger.debug(
            f"MockLibria evolved architecture: "
            f"coverage={results['coverage']:.2%}, max_quality={results['max_quality']:.2f}"
        )

        return results
