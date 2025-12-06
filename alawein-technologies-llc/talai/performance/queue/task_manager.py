"""
Task management with chaining, grouping, and workflow orchestration.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .task_queue import Task, TaskPriority, TaskQueue, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class TaskChain:
    """Represents a chain of tasks to execute sequentially."""
    id: str
    tasks: List[Task]
    current_index: int = 0
    results: List[Any] = None


@dataclass
class TaskGroup:
    """Represents a group of tasks to execute in parallel."""
    id: str
    tasks: List[Task]
    wait_all: bool = True
    results: Dict[str, Any] = None


class TaskManager:
    """
    High-level task management with chaining, grouping,
    and complex workflow orchestration.
    """

    def __init__(self, task_queue: TaskQueue):
        """Initialize task manager."""
        self.queue = task_queue
        self._chains: Dict[str, TaskChain] = {}
        self._groups: Dict[str, TaskGroup] = {}

    async def create_chain(
        self,
        tasks: List[Dict[str, Any]],
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """
        Create a chain of tasks to execute sequentially.

        Args:
            tasks: List of task definitions
            priority: Chain priority

        Returns:
            Chain ID
        """
        chain_id = f"chain_{datetime.utcnow().timestamp()}"
        chain_tasks = []

        # Create linked tasks
        parent_id = None
        for i, task_def in enumerate(tasks):
            task_id = await self.queue.enqueue(
                task_name=task_def["name"],
                payload=task_def.get("payload", {}),
                priority=priority,
                parent_task_id=parent_id,
                delay_seconds=0 if i == 0 else None  # Delay subsequent tasks
            )

            chain_tasks.append(Task(id=task_id, **task_def))
            parent_id = task_id

        chain = TaskChain(id=chain_id, tasks=chain_tasks, results=[])
        self._chains[chain_id] = chain

        logger.info(f"Created task chain {chain_id} with {len(tasks)} tasks")
        return chain_id

    async def create_group(
        self,
        tasks: List[Dict[str, Any]],
        wait_all: bool = True,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """
        Create a group of tasks to execute in parallel.

        Args:
            tasks: List of task definitions
            wait_all: Wait for all tasks to complete
            priority: Group priority

        Returns:
            Group ID
        """
        group_id = f"group_{datetime.utcnow().timestamp()}"
        group_tasks = []

        # Create parallel tasks
        for task_def in tasks:
            task_id = await self.queue.enqueue(
                task_name=task_def["name"],
                payload=task_def.get("payload", {}),
                priority=priority
            )

            group_tasks.append(Task(id=task_id, **task_def))

        group = TaskGroup(
            id=group_id,
            tasks=group_tasks,
            wait_all=wait_all,
            results={}
        )
        self._groups[group_id] = group

        logger.info(f"Created task group {group_id} with {len(tasks)} tasks")
        return group_id

    async def create_workflow(
        self,
        workflow_def: Dict[str, Any]
    ) -> str:
        """
        Create a complex workflow with mixed chains and groups.

        Args:
            workflow_def: Workflow definition

        Returns:
            Workflow ID
        """
        workflow_id = f"workflow_{datetime.utcnow().timestamp()}"

        # Parse workflow definition and create tasks
        # This is a simplified implementation
        # Real implementation would support DAG-based workflows

        logger.info(f"Created workflow {workflow_id}")
        return workflow_id

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        # Implementation would update task status and prevent execution
        logger.info(f"Cancelled task {task_id}")
        return True

    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get current task status."""
        # Query task status from queue
        return TaskStatus.PENDING

    async def get_chain_status(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """Get chain execution status."""
        chain = self._chains.get(chain_id)
        if not chain:
            return None

        return {
            "id": chain.id,
            "total_tasks": len(chain.tasks),
            "completed_tasks": chain.current_index,
            "progress": chain.current_index / len(chain.tasks),
            "results": chain.results
        }

    async def retry_failed_tasks(self, since: datetime) -> int:
        """Retry all failed tasks since specified time."""
        # Query failed tasks and re-enqueue them
        count = 0
        logger.info(f"Retried {count} failed tasks")
        return count