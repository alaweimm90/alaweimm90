"""
Job Queue - Queue management for validation jobs

Manages job queuing, prioritization, and lifecycle.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import heapq
import uuid
import logging


logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job lifecycle status"""
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    PAUSED = "paused"


class JobPriority(Enum):
    """Job priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

    def __lt__(self, other):
        return self.value < other.value


@dataclass
class JobMetrics:
    """Metrics for job execution"""
    queue_time: float = 0.0
    execution_time: float = 0.0
    total_time: float = 0.0
    retry_count: int = 0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    cost: float = 0.0


@dataclass
class Job:
    """Job definition for scheduling"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    priority: JobPriority = JobPriority.NORMAL
    handler: Optional[Callable] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Optional[Dict[str, Any]] = None
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: JobMetrics = field(default_factory=JobMetrics)
    max_retries: int = 3
    timeout_seconds: Optional[int] = None
    resource_requirements: Dict[str, float] = field(default_factory=dict)

    def __lt__(self, other):
        """Compare jobs for priority queue"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at

    def is_ready(self, completed_jobs: set) -> bool:
        """Check if job is ready to execute"""
        return all(dep in completed_jobs for dep in self.dependencies)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize job to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority.name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
            "metrics": {
                "queue_time": self.metrics.queue_time,
                "execution_time": self.metrics.execution_time,
                "total_time": self.metrics.total_time,
                "retry_count": self.metrics.retry_count,
                "cost": self.metrics.cost
            }
        }


class JobQueue:
    """Priority-based job queue with multiple queues"""

    def __init__(self, max_queue_size: Optional[int] = None):
        self.max_queue_size = max_queue_size
        self.queues: Dict[JobPriority, List[Job]] = {
            priority: [] for priority in JobPriority
        }
        self.jobs: Dict[str, Job] = {}
        self.completed_jobs: Set[str] = set()
        self.failed_jobs: Set[str] = set()
        self._lock = asyncio.Lock()
        self._job_added = asyncio.Event()

    async def add_job(self, job: Job) -> str:
        """Add job to queue"""
        async with self._lock:
            # Check queue size limit
            if self.max_queue_size and len(self.jobs) >= self.max_queue_size:
                raise RuntimeError(f"Queue full: {len(self.jobs)}/{self.max_queue_size}")

            # Add to job registry
            self.jobs[job.id] = job

            # Add to priority queue
            heapq.heappush(self.queues[job.priority], job)

            # Signal new job
            self._job_added.set()

            logger.info(f"Added job {job.id} with priority {job.priority.name}")
            return job.id

    async def get_next_job(self, timeout: Optional[float] = None) -> Optional[Job]:
        """Get next job from queue"""
        start_time = datetime.now()

        while True:
            async with self._lock:
                # Check each priority level
                for priority in JobPriority:
                    queue = self.queues[priority]
                    if not queue:
                        continue

                    # Find ready job
                    for i, job in enumerate(queue):
                        if job.status != JobStatus.QUEUED:
                            continue

                        if job.is_ready(self.completed_jobs):
                            # Remove from queue
                            queue.pop(i)
                            heapq.heapify(queue)

                            # Update status
                            job.status = JobStatus.SCHEDULED
                            job.scheduled_at = datetime.now()

                            # Calculate queue time
                            job.metrics.queue_time = (
                                job.scheduled_at - job.created_at
                            ).total_seconds()

                            logger.info(f"Scheduled job {job.id}")
                            return job

            # Check timeout
            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= timeout:
                    return None

            # Wait for new jobs
            self._job_added.clear()
            try:
                await asyncio.wait_for(self._job_added.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    async def mark_completed(self, job_id: str, outputs: Optional[Dict[str, Any]] = None):
        """Mark job as completed"""
        async with self._lock:
            if job_id not in self.jobs:
                return

            job = self.jobs[job_id]
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.outputs = outputs

            # Calculate metrics
            if job.started_at:
                job.metrics.execution_time = (
                    job.completed_at - job.started_at
                ).total_seconds()
            job.metrics.total_time = (
                job.completed_at - job.created_at
            ).total_seconds()

            # Add to completed set
            self.completed_jobs.add(job_id)

            logger.info(f"Job {job_id} completed")

    async def mark_failed(self, job_id: str, error: str):
        """Mark job as failed"""
        async with self._lock:
            if job_id not in self.jobs:
                return

            job = self.jobs[job_id]
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            job.error = error

            # Add to failed set
            self.failed_jobs.add(job_id)

            logger.error(f"Job {job_id} failed: {error}")

    async def mark_running(self, job_id: str):
        """Mark job as running"""
        async with self._lock:
            if job_id not in self.jobs:
                return

            job = self.jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()

            logger.info(f"Job {job_id} started")

    async def cancel_job(self, job_id: str):
        """Cancel job"""
        async with self._lock:
            if job_id not in self.jobs:
                return

            job = self.jobs[job_id]

            # Remove from queue if still queued
            if job.status == JobStatus.QUEUED:
                queue = self.queues[job.priority]
                if job in queue:
                    queue.remove(job)
                    heapq.heapify(queue)

            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()

            logger.info(f"Job {job_id} cancelled")

    async def requeue_job(self, job_id: str, new_priority: Optional[JobPriority] = None):
        """Requeue failed or cancelled job"""
        async with self._lock:
            if job_id not in self.jobs:
                return

            job = self.jobs[job_id]

            # Reset job state
            job.status = JobStatus.QUEUED
            job.scheduled_at = None
            job.started_at = None
            job.completed_at = None
            job.error = None
            job.metrics.retry_count += 1

            # Update priority if specified
            if new_priority:
                job.priority = new_priority

            # Add back to queue
            heapq.heappush(self.queues[job.priority], job)

            # Remove from completed/failed sets
            self.completed_jobs.discard(job_id)
            self.failed_jobs.discard(job_id)

            # Signal new job
            self._job_added.set()

            logger.info(f"Requeued job {job_id}")

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self.jobs.get(job_id)

    def list_jobs(self, status: Optional[JobStatus] = None,
                 priority: Optional[JobPriority] = None) -> List[Job]:
        """List jobs with optional filters"""
        jobs = list(self.jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        if priority:
            jobs = [j for j in jobs if j.priority == priority]

        return jobs

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        stats = {
            "total_jobs": len(self.jobs),
            "completed": len(self.completed_jobs),
            "failed": len(self.failed_jobs),
            "by_status": {},
            "by_priority": {},
            "queue_depths": {}
        }

        # Count by status
        for status in JobStatus:
            count = sum(1 for j in self.jobs.values() if j.status == status)
            stats["by_status"][status.value] = count

        # Count by priority
        for priority in JobPriority:
            count = sum(1 for j in self.jobs.values() if j.priority == priority)
            stats["by_priority"][priority.name] = count
            stats["queue_depths"][priority.name] = len(self.queues[priority])

        return stats

    async def clear_completed(self, before: Optional[datetime] = None):
        """Clear completed jobs from history"""
        async with self._lock:
            jobs_to_remove = []

            for job_id, job in self.jobs.items():
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    if not before or job.completed_at < before:
                        jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                del self.jobs[job_id]
                self.completed_jobs.discard(job_id)
                self.failed_jobs.discard(job_id)

            logger.info(f"Cleared {len(jobs_to_remove)} completed jobs")