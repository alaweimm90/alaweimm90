"""
ORCHEX-compliant agent wrapper for Librex

Exposes Librex as an ORCHEX research agent that can be orchestrated
alongside other agents in multi-agent workflows.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import redis

from Librex.integrations.ORCHEX.adapter import ATLASOptimizationAdapter
from Librex.integrations.ORCHEX.config import ATLASConfig
from Librex.integrations.ORCHEX.task_queue import TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for ORCHEX research agent (Librex variant)"""
    agent_id: str
    agent_type: str
    specialization: str
    skill_level: float
    max_tasks: int
    model: str = "Librex-native"  # Not an LLM, but native optimization


class LibrexAgent:
    """
    ORCHEX-compliant agent that provides optimization services

    Implements the ORCHEX ResearchAgent interface to participate in
    multi-agent orchestration workflows.
    """

    def __init__(self, config: Optional[ATLASConfig] = None):
        """
        Initialize Librex agent

        Args:
            config: ORCHEX configuration
        """
        self.atlas_config = config or ATLASConfig.from_env()

        # Agent configuration
        self.config = AgentConfig(
            agent_id=self.atlas_config.agent_id,
            agent_type=self.atlas_config.agent_type,
            specialization=self.atlas_config.specialization,
            skill_level=self.atlas_config.skill_level,
            max_tasks=self.atlas_config.max_concurrent_tasks
        )

        # Initialize components
        self.adapter = ATLASOptimizationAdapter(self.atlas_config)
        self.redis_client = redis.Redis.from_url(
            self.atlas_config.redis_url,
            decode_responses=True,
            db=self.atlas_config.redis_db
        )

        # Agent state
        self.current_workload = 0
        self.execution_history: List[Dict] = []
        self.active_tasks: Dict[str, Dict] = {}

        # Background task processor
        self._stop_processor = threading.Event()
        self._processor_thread = None

        logger.info(f"Initialized Librex agent: {self.config.agent_id}")

    def can_accept_task(self, task: Dict) -> bool:
        """
        Check if agent can accept task (ORCHEX protocol)

        Args:
            task: Task dictionary with requirements

        Returns:
            True if agent can accept task
        """
        # Check workload capacity
        if self.current_workload >= self.config.max_tasks:
            return False

        # Check if task is optimization-related
        task_type = task.get("task_type", "")
        if task_type not in ["optimization", "solve", "minimize", "maximize", "optimize"]:
            # Also check if it's a problem type we support
            problem_type = task.get("problem_type", "")
            if problem_type not in ["qap", "tsp", "continuous", "discrete"]:
                return False

        return True

    def execute(self, task: Dict) -> Dict:
        """
        Execute research task (ORCHEX protocol)

        This is the main entry point for ORCHEX orchestration.

        Args:
            task: Task dictionary with input data

        Returns:
            Result dictionary with output data
        """
        start_time = time.time()

        try:
            # Extract task parameters
            task_id = task.get("task_id", str(time.time()))
            problem_type = task.get("problem_type", "unknown")
            problem_data = task.get("problem_data", {})
            method = task.get("method", self.atlas_config.default_method)
            config = task.get("config", {})
            priority = task.get("priority", 0)

            # Increment workload
            self.current_workload += 1
            self.active_tasks[task_id] = task

            # Submit optimization task
            submission_result = self.adapter.submit_optimization_request(
                agent_id=self.config.agent_id,
                problem_type=problem_type,
                problem_data=problem_data,
                method=method,
                config=config,
                priority=priority,
                metadata={"atlas_task_id": task_id}
            )

            Librex_task_id = submission_result["task_id"]

            # Wait for completion (with timeout)
            timeout = task.get("timeout", self.atlas_config.task_timeout)
            result = self._wait_for_task_completion(Librex_task_id, timeout)

            # Record execution
            duration = time.time() - start_time
            success = result.get("status") == "completed"
            quality = self._calculate_quality_score(result)

            self.record_execution(task, result, duration, success)

            # Update blackboard with result
            self._update_blackboard(task_id, result, quality)

            # Decrement workload
            self.current_workload -= 1
            del self.active_tasks[task_id]

            return {
                "task_id": task_id,
                "status": result.get("status", "unknown"),
                "result": result,
                "quality": quality,
                "duration": duration,
                "agent_id": self.config.agent_id
            }

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            self.current_workload = max(0, self.current_workload - 1)

            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

            return {
                "task_id": task.get("task_id", "unknown"),
                "status": "failed",
                "error": str(e),
                "duration": time.time() - start_time,
                "agent_id": self.config.agent_id
            }

    def to_features(self) -> np.ndarray:
        """
        Convert agent state to feature vector for Libria solvers (ORCHEX protocol)

        Returns:
            Feature vector as numpy array
        """
        return np.array([
            self.config.skill_level,
            self.current_workload / self.config.max_tasks if self.config.max_tasks > 0 else 0,
            len(self.execution_history),
            hash(self.config.specialization) % 100 / 100.0,  # Normalized hash
            1.0,  # High reliability for native optimization
            min(1.0, len(self.execution_history) / 100.0)  # Experience factor
        ])

    def record_execution(self, task: Dict, result: Dict, duration: float, success: bool):
        """
        Record task execution in agent's history (ORCHEX protocol)

        Args:
            task: The task that was executed
            result: The result of execution
            duration: Time taken in seconds
            success: Whether execution was successful
        """
        execution_record = {
            "task_id": task.get("task_id", "unknown"),
            "task_type": task.get("task_type", "optimization"),
            "problem_type": task.get("problem_type", "unknown"),
            "duration": duration,
            "success": success,
            "quality": self._calculate_quality_score(result),
            "method": task.get("method", "unknown"),
            "objective_value": result.get("objective") if isinstance(result, dict) else None,
            "timestamp": time.time()
        }

        self.execution_history.append(execution_record)

        # Maintain history size limit
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

    def start_background_processor(self):
        """
        Start background task processor

        Continuously processes tasks from the queue.
        """
        if self._processor_thread and self._processor_thread.is_alive():
            logger.warning("Background processor already running")
            return

        self._stop_processor.clear()
        self._processor_thread = threading.Thread(target=self._process_tasks)
        self._processor_thread.daemon = True
        self._processor_thread.start()

        logger.info("Started background task processor")

    def stop_background_processor(self):
        """
        Stop background task processor
        """
        self._stop_processor.set()

        if self._processor_thread:
            self._processor_thread.join(timeout=5)

        logger.info("Stopped background task processor")

    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get current agent status

        Returns:
            Status dictionary
        """
        return {
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type,
            "specialization": self.config.specialization,
            "skill_level": self.config.skill_level,
            "current_workload": self.current_workload,
            "max_tasks": self.config.max_tasks,
            "capacity_available": self.current_workload < self.config.max_tasks,
            "active_tasks": list(self.active_tasks.keys()),
            "total_executions": len(self.execution_history),
            "recent_success_rate": self._calculate_success_rate(),
            "average_quality": self._calculate_average_quality(),
            "status": "active" if self._processor_thread and self._processor_thread.is_alive() else "idle"
        }

    def register_with_atlas(self):
        """
        Register agent with ORCHEX blackboard

        Makes the agent discoverable by ORCHEX orchestration engine.
        """
        agent_key = f"ORCHEX:agent:{self.config.agent_id}"

        # Store agent metadata
        agent_data = {
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type,
            "specialization": self.config.specialization,
            "skill_level": str(self.config.skill_level),
            "max_tasks": str(self.config.max_tasks),
            "current_workload": str(self.current_workload),
            "available": "true" if self.current_workload < self.config.max_tasks else "false",
            "capabilities": json.dumps([
                "optimization",
                "combinatorial_optimization",
                "continuous_optimization",
                "method_selection",
                "constraint_handling"
            ]),
            "supported_problems": json.dumps(["qap", "tsp", "continuous", "discrete"]),
            "performance_metrics": json.dumps({
                "success_rate": self._calculate_success_rate(),
                "average_quality": self._calculate_average_quality(),
                "total_executions": len(self.execution_history)
            })
        }

        self.redis_client.hset(agent_key, mapping=agent_data)

        # Set agent as available
        self.redis_client.sadd("ORCHEX:available_agents", self.config.agent_id)

        # Publish registration event
        event = {
            "event": "agent_registered",
            "agent_id": self.config.agent_id,
            "timestamp": time.time()
        }
        self.redis_client.publish("ORCHEX:events", json.dumps(event))

        logger.info(f"Registered agent {self.config.agent_id} with ORCHEX blackboard")

    def deregister_from_atlas(self):
        """
        Deregister agent from ORCHEX blackboard
        """
        # Remove from available agents
        self.redis_client.srem("ORCHEX:available_agents", self.config.agent_id)

        # Mark as unavailable
        agent_key = f"ORCHEX:agent:{self.config.agent_id}"
        self.redis_client.hset(agent_key, "available", "false")

        # Publish deregistration event
        event = {
            "event": "agent_deregistered",
            "agent_id": self.config.agent_id,
            "timestamp": time.time()
        }
        self.redis_client.publish("ORCHEX:events", json.dumps(event))

        logger.info(f"Deregistered agent {self.config.agent_id} from ORCHEX blackboard")

    def _process_tasks(self):
        """
        Background task processor loop
        """
        logger.info("Starting task processor loop")

        while not self._stop_processor.is_set():
            try:
                # Check if we can accept more tasks
                if self.current_workload >= self.config.max_tasks:
                    time.sleep(1)
                    continue

                # Get next task from queue
                task = self.adapter.task_queue.get_next_task()

                if task:
                    logger.info(f"Processing task {task.task_id}")
                    result = self.adapter.process_task(task)
                    logger.info(f"Completed task {task.task_id}: {result.get('status')}")
                else:
                    # No tasks available, wait
                    time.sleep(2)

            except Exception as e:
                logger.error(f"Error in task processor: {e}")
                time.sleep(5)

        logger.info("Task processor loop stopped")

    def _wait_for_task_completion(self, task_id: str, timeout: int) -> Dict[str, Any]:
        """
        Wait for task completion with timeout

        Args:
            task_id: Librex task ID
            timeout: Timeout in seconds

        Returns:
            Task result
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.adapter.get_task_status(task_id)

            if status["status"] == "completed":
                return status.get("result", {})
            elif status["status"] == "failed":
                return {"status": "failed", "error": status.get("error", "Unknown error")}

            time.sleep(1)

        # Timeout reached
        return {"status": "timeout", "error": f"Task timed out after {timeout} seconds"}

    def _update_blackboard(self, task_id: str, result: Dict, quality: float):
        """
        Update ORCHEX blackboard with task results

        Args:
            task_id: ORCHEX task ID
            result: Task result
            quality: Quality score
        """
        blackboard_key = f"ORCHEX:blackboard:task:{task_id}:result"

        blackboard_data = {
            "task_id": task_id,
            "agent_id": self.config.agent_id,
            "status": result.get("status", "unknown"),
            "quality": str(quality),
            "objective_value": str(result.get("objective", "N/A")),
            "solution": json.dumps(result.get("solution", [])),
            "timestamp": str(time.time())
        }

        self.redis_client.hset(blackboard_key, mapping=blackboard_data)

        # Set TTL for cleanup
        self.redis_client.expire(blackboard_key, 3600)  # 1 hour

        logger.debug(f"Updated blackboard for task {task_id}")

    def _calculate_quality_score(self, result: Dict) -> float:
        """
        Calculate quality score for a result

        Args:
            result: Task result

        Returns:
            Quality score (0.0 to 1.0)
        """
        if not isinstance(result, dict):
            return 0.0

        # Check if valid solution
        if not result.get("is_valid", True):
            return 0.0

        # Base quality on objective value and iterations
        # This is a simplified heuristic
        if "objective" in result:
            # Assume lower objective is better (minimization)
            # Normalize based on problem size
            obj_value = result["objective"]
            iterations = result.get("iterations", 1)

            # Simple quality heuristic
            if obj_value < 0:
                quality = 1.0  # Negative objectives get max quality
            else:
                # Quality decreases with objective value
                quality = 1.0 / (1.0 + obj_value / 1000.0)

            # Bonus for efficiency (fewer iterations)
            if iterations < 1000:
                quality *= 1.1
            elif iterations < 5000:
                quality *= 1.05

            return min(1.0, max(0.0, quality))

        return 0.5  # Default quality

    def _calculate_success_rate(self) -> float:
        """
        Calculate recent success rate

        Returns:
            Success rate (0.0 to 1.0)
        """
        if not self.execution_history:
            return 0.0

        recent = self.execution_history[-100:]  # Last 100 executions
        successes = sum(1 for ex in recent if ex.get("success", False))

        return successes / len(recent)

    def _calculate_average_quality(self) -> float:
        """
        Calculate average quality of recent executions

        Returns:
            Average quality (0.0 to 1.0)
        """
        if not self.execution_history:
            return 0.0

        recent = self.execution_history[-100:]  # Last 100 executions
        qualities = [ex.get("quality", 0.0) for ex in recent]

        return sum(qualities) / len(qualities) if qualities else 0.0