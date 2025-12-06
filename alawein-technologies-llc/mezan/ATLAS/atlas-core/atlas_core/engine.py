"""
ORCHEX Engine - Main Orchestration Class

Manages 40+ research agents, dialectical workflows, and integration
with Libria solvers for optimal task assignment and routing.
"""

from typing import Dict, List, Optional, Any
import logging
import time
import uuid
import numpy as np

from atlas_core.agent import ResearchAgent, AgentConfig
from atlas_core.blackboard import ATLASBlackboard

logger = logging.getLogger(__name__)


class ATLASEngine:
    """
    Main ORCHEX orchestration engine

    Manages:
    - 40+ research agents
    - Dialectical workflows (thesis-antithesis-synthesis)
    - Quality gates
    - Integration with Libria solvers
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        libria_enabled: bool = True,
    ):
        """
        Initialize ORCHEX Engine

        Args:
            redis_url: Redis connection URL for blackboard
            libria_enabled: Whether to use Libria solvers for optimization
        """
        self.blackboard = ATLASBlackboard(redis_url)
        self.agents: Dict[str, ResearchAgent] = {}
        self.workflows: Dict[str, Any] = {}
        self.libria_enabled = libria_enabled
        self.libria_router = None

        # Initialize Libria connection if enabled
        if libria_enabled:
            try:
                # Try to import LibriaRouter (will use mock if real one not available)
                import sys
                sys.path.append('../Libria/libria-integration')
                from libria_integration import LibriaRouter
                self.libria_router = LibriaRouter()
                logger.info("Libria integration enabled")
            except ImportError:
                logger.warning(
                    "LibriaRouter not available, using fallback assignment"
                )
                self.libria_enabled = False

        logger.info("ORCHEX Engine initialized")

    def register_agent(self, agent: ResearchAgent):
        """
        Register research agent with ORCHEX

        Args:
            agent: ResearchAgent instance to register
        """
        self.agents[agent.config.agent_id] = agent

        # Store agent state in Redis blackboard
        self.blackboard.store_agent_state(
            agent.config.agent_id,
            {
                "type": agent.config.agent_type,
                "specialization": agent.config.specialization,
                "skill_level": str(agent.config.skill_level),
                "workload": str(agent.current_workload),
                "max_tasks": str(agent.config.max_tasks),
                "available": "True",
            },
        )

        logger.info(
            f"Registered agent {agent.config.agent_id} "
            f"({agent.config.agent_type})"
        )

    def assign_task(self, task: Dict) -> Optional[str]:
        """
        Assign task to best agent

        Uses Libria solvers for optimal assignment:
        - Librex.QAP: Agent-task assignment optimization
        - Librex.Meta: Agent selection meta-learning
        - Librex.Flow: Workflow routing

        Args:
            task: Task dictionary with requirements

        Returns:
            Agent ID if assignment successful, None otherwise
        """
        if self.libria_enabled and self.libria_router:
            # Use Libria Librex.QAP for optimal assignment
            try:
                assignment = self.libria_router.solve_assignment(
                    agents=list(self.agents.values()), tasks=[task]
                )
                agent_id = assignment.get("agent_id")
                logger.info(
                    f"Libria assigned task {task.get('task_id')} to {agent_id}"
                )
                return agent_id
            except Exception as e:
                logger.error(f"Libria assignment failed: {e}, using fallback")

        # Fallback: simple greedy assignment
        for agent_id, agent in self.agents.items():
            if agent.can_accept_task(task):
                logger.info(
                    f"Fallback assigned task {task.get('task_id')} to {agent_id}"
                )
                return agent_id

        logger.warning(f"No agent available for task {task.get('task_id')}")
        return None

    def execute_task(self, task: Dict, agent_id: str) -> Dict:
        """
        Execute task with specified agent

        Args:
            task: Task dictionary
            agent_id: Agent to execute task

        Returns:
            Task result dictionary
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]
        execution_id = str(uuid.uuid4())

        # Execute task
        start_time = time.time()
        try:
            result = agent.execute(task)
            duration = time.time() - start_time
            success = True

            # Record execution
            agent.record_execution(task, result, duration, success)

            # Store in blackboard
            self.blackboard.store_execution_record(
                execution_id=execution_id,
                agent_id=agent_id,
                task_id=task.get("task_id", "unknown"),
                duration=duration,
                success=success,
                quality=result.get("quality", 0.0),
            )

            # Update agent workload
            agent.current_workload += 1

            logger.info(
                f"Task {task.get('task_id')} executed by {agent_id} "
                f"in {duration:.2f}s"
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Task execution failed: {e}")

            # Record failure
            result = {"error": str(e), "quality": 0.0}
            agent.record_execution(task, result, duration, False)

            self.blackboard.store_execution_record(
                execution_id=execution_id,
                agent_id=agent_id,
                task_id=task.get("task_id", "unknown"),
                duration=duration,
                success=False,
                quality=0.0,
            )

            raise

    def execute_workflow(self, workflow_type: str, inputs: Dict) -> Dict:
        """
        Execute dialectical workflow

        Types:
        - thesis_antithesis_synthesis: Dialectical reasoning workflow
        - multi_perspective_analysis: Multiple perspective analysis
        - adversarial_validation: Uses Librex.Dual for validation

        Args:
            workflow_type: Type of workflow to execute
            inputs: Input data for workflow

        Returns:
            Workflow result dictionary
        """
        if workflow_type == "thesis_antithesis_synthesis":
            return self._dialectical_workflow(inputs)
        elif workflow_type == "adversarial_validation":
            # Use Librex.Dual for workflow validation
            if self.libria_enabled and self.libria_router:
                return self.libria_router.validate_workflow(inputs)
            else:
                raise ValueError("Libria not enabled for adversarial validation")

        raise ValueError(f"Unknown workflow: {workflow_type}")

    def _dialectical_workflow(self, inputs: Dict) -> Dict:
        """
        Thesis-Antithesis-Synthesis dialectical reasoning

        Steps:
        1. Thesis: Generate initial hypothesis
        2. Antithesis: Challenge with counter-arguments
        3. Synthesis: Reconcile into refined conclusion

        Args:
            inputs: Workflow inputs

        Returns:
            Dictionary with thesis, antithesis, and synthesis results
        """
        # Step 1: Thesis
        thesis_agent = self._select_agent_by_type("hypothesis_generation")
        thesis_task = {
            "task_id": f"thesis_{uuid.uuid4()}",
            "task_type": "generate_hypothesis",
            "input": inputs,
        }
        thesis_result = self.execute_task(thesis_task, thesis_agent.config.agent_id)

        # Step 2: Antithesis (use Librex.Flow for routing if available)
        if self.libria_enabled and self.libria_router:
            try:
                antithesis_candidates = [
                    a
                    for a in self.agents.values()
                    if a.config.agent_type == "critical_analysis"
                ]
                antithesis_agent_id = self.libria_router.route_workflow_step(
                    workflow_state={
                        "step": "antithesis",
                        "thesis": thesis_result,
                        "available_agents": antithesis_candidates,
                    }
                )
                antithesis_agent = self.agents[antithesis_agent_id]
            except Exception as e:
                logger.warning(f"Librex.Flow routing failed: {e}, using fallback")
                antithesis_agent = self._select_agent_by_type("critical_analysis")
        else:
            antithesis_agent = self._select_agent_by_type("critical_analysis")

        antithesis_task = {
            "task_id": f"antithesis_{uuid.uuid4()}",
            "task_type": "challenge_hypothesis",
            "thesis": thesis_result,
        }
        antithesis_result = self.execute_task(
            antithesis_task, antithesis_agent.config.agent_id
        )

        # Step 3: Synthesis
        synthesis_agent = self._select_agent_by_type("synthesis")
        synthesis_task = {
            "task_id": f"synthesis_{uuid.uuid4()}",
            "task_type": "synthesize",
            "thesis": thesis_result,
            "antithesis": antithesis_result,
        }
        synthesis_result = self.execute_task(
            synthesis_task, synthesis_agent.config.agent_id
        )

        return {
            "thesis": thesis_result,
            "antithesis": antithesis_result,
            "synthesis": synthesis_result,
        }

    def _select_agent_by_type(self, agent_type: str) -> ResearchAgent:
        """
        Select agent by type

        Args:
            agent_type: Type of agent to select

        Returns:
            ResearchAgent instance

        Raises:
            ValueError: If no agent of type available
        """
        for agent in self.agents.values():
            if (
                agent.config.agent_type == agent_type
                and agent.can_accept_task({})
            ):
                return agent

        raise ValueError(f"No available agent of type: {agent_type}")

    def get_agent_state_history(self, window: int = 100) -> np.ndarray:
        """
        Get agent state history for Librex.Graph

        Args:
            window: Number of historical states to retrieve

        Returns:
            (T × n × d) array of agent states over time
        """
        states = []
        for agent_id in self.agents:
            agent_history = self.blackboard.get_agent_history(agent_id, window)
            # Convert to feature vectors
            states.append(agent_history)

        return np.array(states)  # TODO: Process into proper (T × n × d) format

    def optimize_communication_topology(self):
        """
        Use Librex.Graph to optimize agent communication network

        Returns:
            Optimized adjacency matrix for agent communication
        """
        if not self.libria_enabled or not self.libria_router:
            logger.warning("Libria not enabled, skipping topology optimization")
            return None

        # Get agent state history
        agent_states = self.get_agent_state_history(window=100)

        # Use Librex.Graph to optimize topology
        optimal_topology = self.libria_router.optimize_topology(agent_states)

        # Apply to ORCHEX
        self._apply_topology(optimal_topology)

        logger.info("Communication topology optimized via Librex.Graph")
        return optimal_topology

    def _apply_topology(self, adjacency_matrix: np.ndarray):
        """
        Apply communication topology to agents

        Args:
            adjacency_matrix: (n × n) adjacency matrix
        """
        n = len(self.agents)
        agent_ids = list(self.agents.keys())

        for i, agent_id_i in enumerate(agent_ids):
            connected_agents = [
                agent_ids[j]
                for j in range(n)
                if adjacency_matrix[i, j] > 0.5
            ]
            self.blackboard.set_agent_connections(agent_id_i, connected_agents)

        logger.info(f"Applied communication topology to {n} agents")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get ORCHEX Engine statistics

        Returns:
            Dictionary with engine stats
        """
        return {
            "total_agents": len(self.agents),
            "agents_by_type": self._count_agents_by_type(),
            "total_executions": sum(
                len(agent.execution_history) for agent in self.agents.values()
            ),
            "libria_enabled": self.libria_enabled,
        }

    def _count_agents_by_type(self) -> Dict[str, int]:
        """Count agents by type"""
        counts = {}
        for agent in self.agents.values():
            agent_type = agent.config.agent_type
            counts[agent_type] = counts.get(agent_type, 0) + 1
        return counts

    def __repr__(self) -> str:
        return (
            f"ATLASEngine(agents={len(self.agents)}, "
            f"libria_enabled={self.libria_enabled})"
        )
