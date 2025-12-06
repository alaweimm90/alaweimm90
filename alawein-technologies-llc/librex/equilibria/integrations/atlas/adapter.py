"""
ORCHEX adapter for Librex optimization services

Provides integration between ORCHEX agents and Librex's optimization capabilities.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Librex import optimize
from Librex.adapters import get_adapter
from Librex.ai.method_selector import MethodSelector
from Librex.core.interfaces import StandardizedProblem, ValidationResult
from Librex.integrations.ORCHEX.config import ATLASConfig
from Librex.integrations.ORCHEX.task_queue import (
    OptimizationTask,
    OptimizationTaskQueue,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class ATLASOptimizationAdapter:
    """
    Adapter that enables ORCHEX agents to request optimization services from Librex

    Provides:
    - Async optimization task submission
    - Method recommendations via AI selector
    - Result retrieval via Redis blackboard
    - Problem domain adaptation
    """

    def __init__(self, config: Optional[ATLASConfig] = None):
        """
        Initialize ORCHEX adapter

        Args:
            config: ORCHEX configuration
        """
        self.config = config or ATLASConfig.from_env()
        self.task_queue = OptimizationTaskQueue(config)
        self.method_selector = MethodSelector()

        # Domain adapters cache
        self._adapters = {}

        logger.info(f"Initialized ORCHEX adapter with agent ID: {self.config.agent_id}")

    def submit_optimization_request(
        self,
        agent_id: str,
        problem_type: str,
        problem_data: Dict[str, Any],
        method: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Submit optimization request from ORCHEX agent

        Args:
            agent_id: Requesting agent's ID
            problem_type: Type of optimization problem (e.g., "qap", "tsp", "continuous")
            problem_data: Problem-specific data
            method: Optimization method (None for auto-selection)
            config: Method configuration
            priority: Task priority
            metadata: Additional metadata

        Returns:
            Response with task ID and estimated completion time
        """
        # Use default method if not specified
        if method is None:
            method = self.config.default_method

        # If auto method selection requested, get recommendation
        if method == "auto":
            recommendation = self._get_method_recommendation(problem_type, problem_data)
            method = recommendation["method"]

            # Merge recommended config with user config
            if config is None:
                config = recommendation["config"]
            else:
                config = {**recommendation["config"], **config}

            logger.info(
                f"Auto-selected method '{method}' with confidence {recommendation['confidence']:.2%}"
            )

        # Submit task to queue
        task_id = self.task_queue.submit_task(
            problem_type=problem_type,
            problem_data=problem_data,
            method=method,
            config=config or {},
            agent_id=agent_id,
            priority=priority,
            metadata=metadata
        )

        # Estimate completion time based on problem size and method
        estimated_time = self._estimate_completion_time(
            problem_type, problem_data, method
        )

        return {
            "task_id": task_id,
            "status": "submitted",
            "method": method,
            "estimated_completion_seconds": estimated_time,
            "redis_blackboard_key": f"ORCHEX:blackboard:{agent_id}:Librex:{task_id}",
            "message": f"Optimization task submitted successfully. Monitor via task_id: {task_id}"
        }

    def get_method_recommendation(
        self,
        agent_id: str,
        problem_type: str,
        problem_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get AI-powered method recommendation for a problem

        Args:
            agent_id: Requesting agent's ID
            problem_type: Type of optimization problem
            problem_data: Problem-specific data

        Returns:
            Method recommendation with explanation
        """
        recommendation = self._get_method_recommendation(problem_type, problem_data)

        return {
            "agent_id": agent_id,
            "problem_type": problem_type,
            "recommendation": {
                "method": recommendation["method"],
                "confidence": recommendation["confidence"],
                "explanation": recommendation["explanation"],
                "supporting_evidence": recommendation["supporting_evidence"],
                "config": recommendation["config"]
            },
            "alternative_methods": recommendation.get("alternatives", [])
        }

    def process_task(self, task: OptimizationTask) -> Dict[str, Any]:
        """
        Process an optimization task

        Args:
            task: Optimization task to process

        Returns:
            Optimization result
        """
        start_time = time.time()

        try:
            # Get appropriate adapter for problem type
            adapter = self._get_adapter(task.problem_type)

            # Run optimization
            result = optimize(
                problem=task.problem_data,
                adapter=adapter,
                method=task.method,
                config=task.config
            )

            # Calculate execution time
            execution_time = time.time() - start_time

            # Prepare result
            optimization_result = {
                "task_id": task.task_id,
                "status": "completed",
                "solution": result.get("solution"),
                "objective": result.get("objective"),
                "is_valid": result.get("is_valid", True),
                "iterations": result.get("iterations", 0),
                "execution_time_seconds": execution_time,
                "method": task.method,
                "metadata": {
                    "problem_type": task.problem_type,
                    "convergence": result.get("convergence", {}),
                    "method_metadata": result.get("metadata", {})
                }
            }

            # Update task status
            self.task_queue.update_task_status(
                task.task_id,
                TaskStatus.COMPLETED,
                result=optimization_result
            )

            logger.info(
                f"Completed task {task.task_id}: objective={result.get('objective')}, "
                f"time={execution_time:.2f}s"
            )

            return optimization_result

        except Exception as e:
            error_msg = f"Optimization failed: {str(e)}"
            logger.error(f"Task {task.task_id} failed: {error_msg}")

            # Update task status
            self.task_queue.update_task_status(
                task.task_id,
                TaskStatus.FAILED,
                error=error_msg
            )

            # Retry if possible
            if task.retries < self.config.max_retries:
                self.task_queue.retry_task(task.task_id)
                return {
                    "task_id": task.task_id,
                    "status": "retrying",
                    "error": error_msg,
                    "retry_attempt": task.retries + 1
                }

            return {
                "task_id": task.task_id,
                "status": "failed",
                "error": error_msg,
                "execution_time_seconds": time.time() - start_time
            }

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of an optimization task

        Args:
            task_id: Task identifier

        Returns:
            Task status information
        """
        task = self.task_queue.get_task(task_id)

        if not task:
            return {
                "task_id": task_id,
                "status": "not_found",
                "error": f"Task {task_id} not found"
            }

        response = {
            "task_id": task_id,
            "status": task.status.value,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "problem_type": task.problem_type,
            "method": task.method,
            "agent_id": task.agent_id,
            "retries": task.retries
        }

        if task.status == TaskStatus.COMPLETED and task.result:
            response["result"] = task.result
        elif task.status == TaskStatus.FAILED and task.error:
            response["error"] = task.error

        return response

    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        Cancel an optimization task

        Args:
            task_id: Task identifier

        Returns:
            Cancellation result
        """
        success = self.task_queue.cancel_task(task_id)

        if success:
            return {
                "task_id": task_id,
                "status": "cancelled",
                "message": f"Task {task_id} cancelled successfully"
            }
        else:
            return {
                "task_id": task_id,
                "status": "error",
                "error": f"Failed to cancel task {task_id}"
            }

    def get_agent_tasks(
        self,
        agent_id: str,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all tasks for a specific ORCHEX agent

        Args:
            agent_id: Agent identifier
            status_filter: Optional status filter

        Returns:
            List of task summaries
        """
        status_enum = TaskStatus(status_filter) if status_filter else None
        tasks = self.task_queue.get_agent_tasks(agent_id, status_enum)

        return [
            {
                "task_id": task.task_id,
                "status": task.status.value,
                "problem_type": task.problem_type,
                "method": task.method,
                "created_at": task.created_at,
                "priority": task.priority,
                "has_result": bool(task.result)
            }
            for task in tasks
        ]

    def batch_submit(
        self,
        agent_id: str,
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Submit multiple optimization tasks in batch

        Args:
            agent_id: Requesting agent's ID
            tasks: List of task specifications

        Returns:
            List of submission results
        """
        results = []

        for task_spec in tasks:
            try:
                result = self.submit_optimization_request(
                    agent_id=agent_id,
                    problem_type=task_spec["problem_type"],
                    problem_data=task_spec["problem_data"],
                    method=task_spec.get("method"),
                    config=task_spec.get("config"),
                    priority=task_spec.get("priority", 0),
                    metadata=task_spec.get("metadata")
                )
                results.append(result)
            except Exception as e:
                results.append({
                    "status": "error",
                    "error": str(e)
                })

        return results

    def _get_adapter(self, problem_type: str):
        """
        Get or create adapter for problem type

        Args:
            problem_type: Type of optimization problem

        Returns:
            Domain adapter instance
        """
        if problem_type not in self._adapters:
            try:
                self._adapters[problem_type] = get_adapter(problem_type)
            except Exception as e:
                logger.error(f"Failed to get adapter for {problem_type}: {e}")
                # Try to import directly
                if problem_type == "qap":
                    from Librex.adapters.qap import QAPAdapter
                    self._adapters[problem_type] = QAPAdapter()
                elif problem_type == "tsp":
                    from Librex.adapters.tsp import TSPAdapter
                    self._adapters[problem_type] = TSPAdapter()
                else:
                    raise ValueError(f"Unknown problem type: {problem_type}")

        return self._adapters[problem_type]

    def _get_method_recommendation(
        self,
        problem_type: str,
        problem_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get method recommendation from AI selector

        Args:
            problem_type: Type of optimization problem
            problem_data: Problem-specific data

        Returns:
            Recommendation dictionary
        """
        try:
            # Get adapter and encode problem
            adapter = self._get_adapter(problem_type)
            standardized_problem = adapter.encode_problem(problem_data)

            # Get recommendation
            recommendation = self.method_selector.recommend_method(
                problem=problem_data,
                adapter=adapter,
                standardized_problem=standardized_problem
            )

            return {
                "method": recommendation[0],
                "config": recommendation[1],
                "confidence": recommendation[2],
                "explanation": f"Selected {recommendation[0]} based on problem characteristics",
                "supporting_evidence": [
                    f"Problem type: {problem_type}",
                    f"Problem size: {self._estimate_problem_size(problem_data)}",
                    f"Confidence: {recommendation[2]:.2%}"
                ]
            }

        except Exception as e:
            logger.warning(f"Method recommendation failed: {e}. Using default.")
            return {
                "method": "simulated_annealing",
                "config": self.config.method_configs.get("simulated_annealing", {}),
                "confidence": 0.5,
                "explanation": "Default method due to recommendation error",
                "supporting_evidence": [f"Error: {str(e)}"]
            }

    def _estimate_problem_size(self, problem_data: Dict[str, Any]) -> str:
        """
        Estimate problem size category

        Args:
            problem_data: Problem data

        Returns:
            Size category string
        """
        # Check for common matrix types
        for key in ["flow_matrix", "distance_matrix", "cost_matrix", "matrix"]:
            if key in problem_data:
                matrix = problem_data[key]
                if hasattr(matrix, "shape"):
                    size = matrix.shape[0]
                elif isinstance(matrix, list):
                    size = len(matrix)
                else:
                    continue

                if size < 50:
                    return "small"
                elif size < 200:
                    return "medium"
                elif size < 1000:
                    return "large"
                else:
                    return "extra_large"

        # Check for coordinates (TSP-like)
        if "coordinates" in problem_data:
            size = len(problem_data["coordinates"])
            if size < 100:
                return "small"
            elif size < 500:
                return "medium"
            else:
                return "large"

        return "unknown"

    def _estimate_completion_time(
        self,
        problem_type: str,
        problem_data: Dict[str, Any],
        method: str
    ) -> float:
        """
        Estimate task completion time

        Args:
            problem_type: Type of optimization problem
            problem_data: Problem data
            method: Optimization method

        Returns:
            Estimated seconds to completion
        """
        size = self._estimate_problem_size(problem_data)

        # Base estimates (in seconds)
        base_times = {
            "small": {"random_search": 1, "simulated_annealing": 5, "genetic_algorithm": 10},
            "medium": {"random_search": 5, "simulated_annealing": 30, "genetic_algorithm": 60},
            "large": {"random_search": 20, "simulated_annealing": 120, "genetic_algorithm": 300},
            "extra_large": {"random_search": 60, "simulated_annealing": 600, "genetic_algorithm": 1200},
            "unknown": {"random_search": 10, "simulated_annealing": 60, "genetic_algorithm": 120}
        }

        # Get base time or use default
        size_times = base_times.get(size, base_times["unknown"])
        base_time = size_times.get(method, 60)  # Default to 60 seconds

        # Adjust for GPU acceleration
        if self.config.enable_gpu and method in ["genetic_algorithm", "particle_swarm"]:
            base_time *= 0.3  # GPU can be 3x faster

        return base_time