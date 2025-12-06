"""
Core task queue implementation with priority management and routing.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Set

import redis
from redis import asyncio as aioredis

logger = logging.getLogger(__name__)


class TaskPriority(IntEnum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    DEAD = "dead"


@dataclass
class Task:
    """Represents a queued task."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    result: Optional[Any] = None
    error: Optional[str] = None
    parent_task_id: Optional[str] = None
    child_task_ids: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueueConfig:
    """Task queue configuration."""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1
    redis_password: Optional[str] = None
    namespace: str = "talai:queue"
    max_workers: int = 10
    batch_size: int = 100
    poll_interval_ms: int = 100
    task_timeout_seconds: int = 300
    enable_dead_letter_queue: bool = True
    enable_rate_limiting: bool = True
    rate_limit_per_second: int = 100
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: float = 0.5
    circuit_breaker_timeout: int = 60


class TaskQueue:
    """
    High-performance distributed task queue with priority management,
    rate limiting, circuit breaking, and comprehensive monitoring.
    """

    def __init__(self, config: Optional[QueueConfig] = None):
        """Initialize task queue."""
        self.config = config or QueueConfig()
        self._initialize_redis()
        self._initialize_queues()
        self._workers: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._handlers: Dict[str, Callable] = {}
        self._running_tasks: Dict[str, Task] = {}
        self._circuit_breaker_open = False
        self._last_circuit_check = time.time()

        # Statistics
        self._stats = {
            "enqueued": 0,
            "processed": 0,
            "failed": 0,
            "retried": 0,
            "cancelled": 0,
            "timed_out": 0
        }

        logger.info(f"TaskQueue initialized with namespace: {self.config.namespace}")

    def _initialize_redis(self) -> None:
        """Initialize Redis connection."""
        self.redis = redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            password=self.config.redis_password,
            decode_responses=True,
            connection_pool=redis.ConnectionPool(
                max_connections=self.config.max_workers * 2,
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db
            )
        )

    def _initialize_queues(self) -> None:
        """Initialize priority queues."""
        self.queue_keys = {
            priority: f"{self.config.namespace}:priority:{priority.value}"
            for priority in TaskPriority
        }
        self.processing_key = f"{self.config.namespace}:processing"
        self.completed_key = f"{self.config.namespace}:completed"
        self.failed_key = f"{self.config.namespace}:failed"
        self.dead_letter_key = f"{self.config.namespace}:dlq"
        self.scheduled_key = f"{self.config.namespace}:scheduled"

    def register_handler(self, task_name: str, handler: Callable) -> None:
        """
        Register task handler.

        Args:
            task_name: Name of the task
            handler: Async function to handle task
        """
        self._handlers[task_name] = handler
        logger.info(f"Registered handler for task: {task_name}")

    async def enqueue(
        self,
        task_name: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        delay_seconds: Optional[int] = None,
        parent_task_id: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        timeout_seconds: Optional[int] = None
    ) -> str:
        """
        Enqueue a task for processing.

        Args:
            task_name: Name of the task to execute
            payload: Task payload data
            priority: Task priority
            delay_seconds: Delay before execution
            parent_task_id: Parent task for chaining
            tags: Task tags for grouping
            timeout_seconds: Task timeout

        Returns:
            Task ID
        """
        task = Task(
            name=task_name,
            payload=payload or {},
            priority=priority,
            parent_task_id=parent_task_id,
            tags=tags or set(),
            timeout_seconds=timeout_seconds or self.config.task_timeout_seconds
        )

        # Handle delayed execution
        if delay_seconds:
            task.scheduled_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
            await self._schedule_task(task)
        else:
            await self._enqueue_task(task)

        self._stats["enqueued"] += 1
        logger.info(f"Enqueued task {task.id} ({task_name}) with priority {priority.name}")

        return task.id

    async def _enqueue_task(self, task: Task) -> None:
        """Internal method to enqueue task."""
        task_data = self._serialize_task(task)
        queue_key = self.queue_keys[task.priority]

        # Apply rate limiting if enabled
        if self.config.enable_rate_limiting:
            await self._apply_rate_limiting()

        # Add to priority queue
        await self.redis.lpush(queue_key, task_data)

        # Store task metadata
        task_key = f"{self.config.namespace}:task:{task.id}"
        await self.redis.hset(task_key, mapping={
            "status": TaskStatus.QUEUED.value,
            "queued_at": datetime.utcnow().isoformat()
        })

    async def _schedule_task(self, task: Task) -> None:
        """Schedule task for delayed execution."""
        task_data = self._serialize_task(task)
        score = task.scheduled_at.timestamp()

        # Add to scheduled set
        await self.redis.zadd(self.scheduled_key, {task_data: score})

        # Store task metadata
        task_key = f"{self.config.namespace}:task:{task.id}"
        await self.redis.hset(task_key, mapping={
            "status": TaskStatus.PENDING.value,
            "scheduled_at": task.scheduled_at.isoformat()
        })

    async def process_tasks(self, worker_id: int) -> None:
        """
        Main worker loop for processing tasks.

        Args:
            worker_id: Worker identifier
        """
        logger.info(f"Worker {worker_id} started")

        while not self._shutdown_event.is_set():
            try:
                # Check scheduled tasks first
                await self._process_scheduled_tasks()

                # Check circuit breaker
                if self._check_circuit_breaker():
                    await asyncio.sleep(1)
                    continue

                # Get next task from priority queues
                task = await self._get_next_task()
                if task:
                    await self._execute_task(task, worker_id)
                else:
                    # No tasks available, wait
                    await asyncio.sleep(self.config.poll_interval_ms / 1000)

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)

        logger.info(f"Worker {worker_id} stopped")

    async def _get_next_task(self) -> Optional[Task]:
        """Get next task from priority queues."""
        # Check queues in priority order
        for priority in TaskPriority:
            queue_key = self.queue_keys[priority]
            task_data = await self.redis.rpop(queue_key)

            if task_data:
                task = self._deserialize_task(task_data)
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()

                # Move to processing set
                await self.redis.hset(
                    self.processing_key,
                    task.id,
                    self._serialize_task(task)
                )

                return task

        return None

    async def _execute_task(self, task: Task, worker_id: int) -> None:
        """Execute a single task."""
        logger.info(f"Worker {worker_id} executing task {task.id} ({task.name})")

        try:
            # Get handler
            handler = self._handlers.get(task.name)
            if not handler:
                raise ValueError(f"No handler registered for task: {task.name}")

            # Execute with timeout
            result = await asyncio.wait_for(
                handler(task.payload),
                timeout=task.timeout_seconds
            )

            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.result = result

            await self._complete_task(task)
            self._stats["processed"] += 1

        except asyncio.TimeoutError:
            logger.error(f"Task {task.id} timed out")
            task.error = f"Task timed out after {task.timeout_seconds} seconds"
            await self._handle_task_failure(task, retry=True)
            self._stats["timed_out"] += 1

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.error = str(e)
            await self._handle_task_failure(task, retry=True)
            self._stats["failed"] += 1

        finally:
            # Remove from processing
            await self.redis.hdel(self.processing_key, task.id)
            self._running_tasks.pop(task.id, None)

    async def _complete_task(self, task: Task) -> None:
        """Handle task completion."""
        # Store completed task
        completed_data = self._serialize_task(task)
        await self.redis.hset(self.completed_key, task.id, completed_data)

        # Update task metadata
        task_key = f"{self.config.namespace}:task:{task.id}"
        await self.redis.hset(task_key, mapping={
            "status": TaskStatus.COMPLETED.value,
            "completed_at": task.completed_at.isoformat(),
            "duration_ms": (task.completed_at - task.started_at).total_seconds() * 1000
        })

        # Process child tasks if any
        if task.child_task_ids:
            for child_id in task.child_task_ids:
                await self._trigger_child_task(child_id)

        logger.info(f"Task {task.id} completed successfully")

    async def _handle_task_failure(self, task: Task, retry: bool = True) -> None:
        """Handle task failure with retry logic."""
        if retry and task.retry_count < task.max_retries:
            # Retry task
            task.retry_count += 1
            task.status = TaskStatus.RETRYING
            delay = min(2 ** task.retry_count * 10, 300)  # Exponential backoff

            logger.info(f"Retrying task {task.id} (attempt {task.retry_count}/{task.max_retries})")

            # Re-enqueue with delay
            task.scheduled_at = datetime.utcnow() + timedelta(seconds=delay)
            await self._schedule_task(task)
            self._stats["retried"] += 1

        else:
            # Move to dead letter queue
            task.status = TaskStatus.DEAD

            if self.config.enable_dead_letter_queue:
                dead_data = self._serialize_task(task)
                await self.redis.hset(self.dead_letter_key, task.id, dead_data)

            logger.error(f"Task {task.id} failed permanently: {task.error}")

    async def _process_scheduled_tasks(self) -> None:
        """Process tasks scheduled for execution."""
        current_time = datetime.utcnow().timestamp()

        # Get due tasks
        due_tasks = await self.redis.zrangebyscore(
            self.scheduled_key,
            0,
            current_time,
            withscores=False,
            num=10
        )

        for task_data in due_tasks:
            task = self._deserialize_task(task_data)

            # Remove from scheduled set
            await self.redis.zrem(self.scheduled_key, task_data)

            # Enqueue for immediate execution
            await self._enqueue_task(task)

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should be open."""
        if not self.config.enable_circuit_breaker:
            return False

        current_time = time.time()

        # Check if we should evaluate circuit state
        if current_time - self._last_circuit_check > 10:
            self._last_circuit_check = current_time

            total = self._stats["processed"] + self._stats["failed"]
            if total > 100:  # Minimum sample size
                failure_rate = self._stats["failed"] / total

                if failure_rate > self.config.circuit_breaker_threshold:
                    if not self._circuit_breaker_open:
                        logger.warning(f"Circuit breaker opened (failure rate: {failure_rate:.2%})")
                        self._circuit_breaker_open = True
                else:
                    if self._circuit_breaker_open:
                        logger.info("Circuit breaker closed")
                        self._circuit_breaker_open = False

        return self._circuit_breaker_open

    async def _apply_rate_limiting(self) -> None:
        """Apply rate limiting to task processing."""
        if not self.config.enable_rate_limiting:
            return

        key = f"{self.config.namespace}:rate_limit"
        current_second = int(time.time())

        # Increment counter
        pipe = self.redis.pipeline()
        pipe.incr(f"{key}:{current_second}")
        pipe.expire(f"{key}:{current_second}", 2)
        count, _ = await pipe.execute()

        # Wait if rate limit exceeded
        if count > self.config.rate_limit_per_second:
            wait_time = 1.0 / self.config.rate_limit_per_second
            await asyncio.sleep(wait_time)

    def _serialize_task(self, task: Task) -> str:
        """Serialize task to JSON."""
        return json.dumps({
            "id": task.id,
            "name": task.name,
            "payload": task.payload,
            "priority": task.priority.value,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "scheduled_at": task.scheduled_at.isoformat() if task.scheduled_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
            "timeout_seconds": task.timeout_seconds,
            "result": task.result,
            "error": task.error,
            "parent_task_id": task.parent_task_id,
            "child_task_ids": task.child_task_ids,
            "tags": list(task.tags),
            "metadata": task.metadata
        })

    def _deserialize_task(self, data: str) -> Task:
        """Deserialize task from JSON."""
        task_dict = json.loads(data)
        return Task(
            id=task_dict["id"],
            name=task_dict["name"],
            payload=task_dict["payload"],
            priority=TaskPriority(task_dict["priority"]),
            status=TaskStatus(task_dict["status"]),
            created_at=datetime.fromisoformat(task_dict["created_at"]),
            scheduled_at=datetime.fromisoformat(task_dict["scheduled_at"]) if task_dict["scheduled_at"] else None,
            started_at=datetime.fromisoformat(task_dict["started_at"]) if task_dict["started_at"] else None,
            completed_at=datetime.fromisoformat(task_dict["completed_at"]) if task_dict["completed_at"] else None,
            retry_count=task_dict["retry_count"],
            max_retries=task_dict["max_retries"],
            timeout_seconds=task_dict["timeout_seconds"],
            result=task_dict["result"],
            error=task_dict["error"],
            parent_task_id=task_dict["parent_task_id"],
            child_task_ids=task_dict["child_task_ids"],
            tags=set(task_dict["tags"]),
            metadata=task_dict["metadata"]
        )

    async def start_workers(self) -> None:
        """Start worker tasks."""
        for i in range(self.config.max_workers):
            worker = asyncio.create_task(self.process_tasks(i))
            self._workers.append(worker)

        logger.info(f"Started {self.config.max_workers} workers")

    async def shutdown(self) -> None:
        """Gracefully shutdown task queue."""
        logger.info("Shutting down TaskQueue")
        self._shutdown_event.set()

        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

        logger.info("TaskQueue shutdown complete")

    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        queue_sizes = {}
        for priority in TaskPriority:
            size = await self.redis.llen(self.queue_keys[priority])
            queue_sizes[priority.name] = size

        processing_count = await self.redis.hlen(self.processing_key)
        completed_count = await self.redis.hlen(self.completed_key)
        failed_count = await self.redis.hlen(self.failed_key)
        dlq_count = await self.redis.hlen(self.dead_letter_key)
        scheduled_count = await self.redis.zcard(self.scheduled_key)

        return {
            "queue_sizes": queue_sizes,
            "processing": processing_count,
            "completed": completed_count,
            "failed": failed_count,
            "dead_letter": dlq_count,
            "scheduled": scheduled_count,
            "stats": self._stats,
            "circuit_breaker_open": self._circuit_breaker_open,
            "workers": len(self._workers)
        }