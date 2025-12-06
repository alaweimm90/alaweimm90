"""
ORCHEX Research Agent Base Classes

Defines base classes for ORCHEX research agents that can execute
various research tasks as part of multi-agent orchestration workflows.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for ORCHEX research agent"""
    agent_id: str
    agent_type: str  # "synthesis", "literature_review", "hypothesis_gen", etc.
    specialization: str
    skill_level: float  # 0.0 - 1.0
    max_tasks: int
    model: str  # "claude-3-opus", "gpt-4-turbo", etc.


class ResearchAgent:
    """
    Base class for ORCHEX research agents

    All research agents inherit from this class and implement
    the execute() method for their specific task type.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.current_workload = 0
        self.execution_history: List[Dict] = []
        logger.info(f"Initialized {config.agent_type} agent: {config.agent_id}")

    def can_accept_task(self, task: Dict) -> bool:
        """
        Check if agent can accept task

        Args:
            task: Task dictionary with requirements

        Returns:
            True if agent can accept task, False otherwise
        """
        return self.current_workload < self.config.max_tasks

    def execute(self, task: Dict) -> Dict:
        """
        Execute research task

        This method must be implemented by all subclasses.

        Args:
            task: Task dictionary with input data

        Returns:
            Result dictionary with output data
        """
        raise NotImplementedError(
            f"Agent {self.config.agent_id} must implement execute() method"
        )

    def to_features(self) -> np.ndarray:
        """
        Convert agent state to feature vector for Libria solvers

        Returns:
            Feature vector as numpy array
        """
        return np.array([
            self.config.skill_level,
            self.current_workload / self.config.max_tasks if self.config.max_tasks > 0 else 0,
            len(self.execution_history),
            hash(self.config.specialization) % 100 / 100.0,  # Normalized hash
        ])

    def record_execution(self, task: Dict, result: Dict, duration: float, success: bool):
        """
        Record task execution in agent's history

        Args:
            task: The task that was executed
            result: The result of execution
            duration: Time taken in seconds
            success: Whether execution was successful
        """
        execution_record = {
            "task_id": task.get("task_id", "unknown"),
            "task_type": task.get("task_type", "unknown"),
            "duration": duration,
            "success": success,
            "quality": result.get("quality", 0.0),
        }
        self.execution_history.append(execution_record)
        logger.info(
            f"Agent {self.config.agent_id} completed task {execution_record['task_id']} "
            f"in {duration:.2f}s with quality {execution_record['quality']:.2f}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics

        Returns:
            Dictionary with agent stats
        """
        if not self.execution_history:
            return {
                "total_tasks": 0,
                "success_rate": 0.0,
                "avg_quality": 0.0,
                "avg_duration": 0.0,
            }

        successful = [e for e in self.execution_history if e["success"]]
        return {
            "total_tasks": len(self.execution_history),
            "success_rate": len(successful) / len(self.execution_history),
            "avg_quality": np.mean([e["quality"] for e in self.execution_history]),
            "avg_duration": np.mean([e["duration"] for e in self.execution_history]),
        }

    def __repr__(self) -> str:
        return (
            f"ResearchAgent(id={self.config.agent_id}, "
            f"type={self.config.agent_type}, "
            f"workload={self.current_workload}/{self.config.max_tasks})"
        )
