"""
Redis-based asynchronous task queue for Librex optimization tasks
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import redis

from Librex.integrations.ORCHEX.config import ATLASConfig

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Optimization task status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class OptimizationTask:
    """Represents an optimization task in the queue"""

    task_id: str
    problem_type: str  # e.g., "qap", "tsp", "continuous"
    problem_data: Dict[str, Any]
    method: str
    config: Dict[str, Any]
    status: TaskStatus
    created_at: str
    updated_at: str
    agent_id: Optional[str] = None
    priority: int = 0
    retries: int = 0
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for Redis storage"""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationTask":
        """Create task from dictionary"""
        data["status"] = TaskStatus(data["status"])
        return cls(**data)


class OptimizationTaskQueue:
    """
    Redis-based task queue for managing optimization tasks

    Provides async submission, retrieval, and result callback mechanisms.
    """

    def __init__(self, config: Optional[ATLASConfig] = None):
        """
        Initialize task queue

        Args:
            config: ORCHEX configuration
        """
        self.config = config or ATLASConfig.from_env()
        self.redis_client = redis.Redis.from_url(
            self.config.redis_url,
            decode_responses=True,
            db=self.config.redis_db
        )
        self.task_queue = self.config.task_queue_name
        self.result_queue = self.config.result_queue_name

        # Task tracking keys
        self.task_key_prefix = "Librex:task:"
        self.agent_tasks_key = "Librex:agent_tasks:"
        self.blackboard_key_prefix = "ORCHEX:blackboard:"

        logger.info(f"Initialized Librex task queue with Redis at {self.config.redis_url}")

    def submit_task(
        self,
        problem_type: str,
        problem_data: Dict[str, Any],
        method: str = "auto",
        config: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit an optimization task to the queue

        Args:
            problem_type: Type of optimization problem
            problem_data: Problem-specific data
            method: Optimization method to use
            config: Method configuration
            agent_id: ID of requesting ORCHEX agent
            priority: Task priority (higher = more urgent)
            metadata: Additional task metadata

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        task = OptimizationTask(
            task_id=task_id,
            problem_type=problem_type,
            problem_data=problem_data,
            method=method,
            config=config or {},
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            agent_id=agent_id,
            priority=priority,
            metadata=metadata or {}
        )

        # Store task in Redis
        task_key = f"{self.task_key_prefix}{task_id}"
        self.redis_client.hset(task_key, mapping=task.to_dict())

        # Add to priority queue (negative priority for ZPOPMAX behavior)
        self.redis_client.zadd(self.task_queue, {task_id: -priority})

        # Track agent tasks
        if agent_id:
            agent_key = f"{self.agent_tasks_key}{agent_id}"
            self.redis_client.sadd(agent_key, task_id)

        # Publish task submission event
        self._publish_event("task_submitted", {
            "task_id": task_id,
            "problem_type": problem_type,
            "method": method,
            "agent_id": agent_id
        })

        logger.info(f"Submitted task {task_id} for {problem_type} using {method}")
        return task_id

    def get_next_task(self) -> Optional[OptimizationTask]:
        """
        Get the next task from the queue

        Returns:
            Next task or None if queue is empty
        """
        # Pop highest priority task
        result = self.redis_client.zpopmax(self.task_queue, count=1)
        if not result:
            return None

        task_id, _ = result[0]
        task = self.get_task(task_id)

        if task:
            # Update task status
            self.update_task_status(task_id, TaskStatus.IN_PROGRESS)
            logger.debug(f"Retrieved task {task_id} from queue")

        return task

    def get_task(self, task_id: str) -> Optional[OptimizationTask]:
        """
        Get a specific task by ID

        Args:
            task_id: Task identifier

        Returns:
            Task or None if not found
        """
        task_key = f"{self.task_key_prefix}{task_id}"
        task_data = self.redis_client.hgetall(task_key)

        if not task_data:
            return None

        # Convert string values to appropriate types
        task_data["priority"] = int(task_data.get("priority", 0))
        task_data["retries"] = int(task_data.get("retries", 0))

        # Parse JSON fields
        for field in ["problem_data", "config", "result", "metadata"]:
            if field in task_data and task_data[field]:
                try:
                    task_data[field] = json.loads(task_data[field])
                except json.JSONDecodeError:
                    pass

        return OptimizationTask.from_dict(task_data)

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update task status

        Args:
            task_id: Task identifier
            status: New status
            error: Error message if failed
            result: Task result if completed

        Returns:
            True if updated successfully
        """
        task_key = f"{self.task_key_prefix}{task_id}"

        updates = {
            "status": status.value,
            "updated_at": datetime.utcnow().isoformat()
        }

        if error:
            updates["error"] = error
        if result:
            updates["result"] = json.dumps(result)

        success = self.redis_client.hset(task_key, mapping=updates)

        if success:
            # Publish status update event
            self._publish_event("task_status_updated", {
                "task_id": task_id,
                "status": status.value,
                "has_error": bool(error),
                "has_result": bool(result)
            })

            # If completed, add to result queue
            if status == TaskStatus.COMPLETED:
                self.redis_client.lpush(self.result_queue, task_id)
                self._notify_blackboard(task_id, result)

        return bool(success)

    def retry_task(self, task_id: str) -> bool:
        """
        Retry a failed task

        Args:
            task_id: Task identifier

        Returns:
            True if task was retried
        """
        task = self.get_task(task_id)
        if not task:
            return False

        if task.retries >= self.config.max_retries:
            logger.warning(f"Task {task_id} exceeded max retries ({self.config.max_retries})")
            self.update_task_status(task_id, TaskStatus.FAILED,
                                   error=f"Exceeded max retries: {self.config.max_retries}")
            return False

        # Increment retry count
        task_key = f"{self.task_key_prefix}{task_id}"
        self.redis_client.hincrby(task_key, "retries", 1)

        # Re-add to queue with delay
        time.sleep(self.config.retry_delay)
        self.redis_client.zadd(self.task_queue, {task_id: -task.priority})

        self.update_task_status(task_id, TaskStatus.RETRYING)
        logger.info(f"Retrying task {task_id} (attempt {task.retries + 1}/{self.config.max_retries})")

        return True

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or in-progress task

        Args:
            task_id: Task identifier

        Returns:
            True if cancelled successfully
        """
        task = self.get_task(task_id)
        if not task:
            return False

        if task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
            return False

        # Remove from queue if pending
        if task.status == TaskStatus.PENDING:
            self.redis_client.zrem(self.task_queue, task_id)

        self.update_task_status(task_id, TaskStatus.CANCELLED)
        logger.info(f"Cancelled task {task_id}")

        return True

    def get_agent_tasks(
        self,
        agent_id: str,
        status_filter: Optional[TaskStatus] = None
    ) -> List[OptimizationTask]:
        """
        Get all tasks for a specific agent

        Args:
            agent_id: Agent identifier
            status_filter: Optional status filter

        Returns:
            List of tasks
        """
        agent_key = f"{self.agent_tasks_key}{agent_id}"
        task_ids = self.redis_client.smembers(agent_key)

        tasks = []
        for task_id in task_ids:
            task = self.get_task(task_id)
            if task and (not status_filter or task.status == status_filter):
                tasks.append(task)

        return tasks

    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics

        Returns:
            Queue statistics dictionary
        """
        pending_count = self.redis_client.zcard(self.task_queue)
        result_count = self.redis_client.llen(self.result_queue)

        # Count tasks by status
        status_counts = {status.value: 0 for status in TaskStatus}

        # Scan all task keys
        cursor = 0
        while True:
            cursor, keys = self.redis_client.scan(
                cursor,
                match=f"{self.task_key_prefix}*",
                count=100
            )

            for key in keys:
                status = self.redis_client.hget(key, "status")
                if status:
                    status_counts[status] = status_counts.get(status, 0) + 1

            if cursor == 0:
                break

        return {
            "pending_tasks": pending_count,
            "results_available": result_count,
            "status_breakdown": status_counts,
            "redis_connected": self.redis_client.ping(),
            "config": self.config.to_dict()
        }

    def _notify_blackboard(self, task_id: str, result: Dict[str, Any]):
        """
        Notify ORCHEX blackboard of task completion

        Args:
            task_id: Task identifier
            result: Task result
        """
        task = self.get_task(task_id)
        if not task or not task.agent_id:
            return

        # Write result to blackboard
        blackboard_key = f"{self.blackboard_key_prefix}{task.agent_id}:Librex:{task_id}"
        self.redis_client.hset(blackboard_key, mapping={
            "task_id": task_id,
            "status": "completed",
            "objective_value": result.get("objective", "N/A"),
            "solution": json.dumps(result.get("solution", [])),
            "timestamp": datetime.utcnow().isoformat()
        })

        # Set TTL for cleanup
        self.redis_client.expire(blackboard_key, self.config.cache_ttl)

        logger.debug(f"Notified blackboard for agent {task.agent_id} about task {task_id}")

    def _publish_event(self, event_type: str, data: Dict[str, Any]):
        """
        Publish event to Redis pub/sub

        Args:
            event_type: Type of event
            data: Event data
        """
        channel = f"Librex:events:{event_type}"
        message = json.dumps({
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        })
        self.redis_client.publish(channel, message)

    def cleanup_old_tasks(self, days: int = 7):
        """
        Clean up old completed tasks

        Args:
            days: Number of days to retain tasks
        """
        cutoff = datetime.utcnow().timestamp() - (days * 24 * 3600)

        cursor = 0
        deleted = 0

        while True:
            cursor, keys = self.redis_client.scan(
                cursor,
                match=f"{self.task_key_prefix}*",
                count=100
            )

            for key in keys:
                created_at = self.redis_client.hget(key, "created_at")
                status = self.redis_client.hget(key, "status")

                if created_at and status in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value]:
                    task_time = datetime.fromisoformat(created_at).timestamp()
                    if task_time < cutoff:
                        self.redis_client.delete(key)
                        deleted += 1

            if cursor == 0:
                break

        logger.info(f"Cleaned up {deleted} old tasks")
        return deleted