"""
Smart Scheduler - Intelligent scheduling with multiple strategies

Implements various scheduling algorithms and policies for optimal job execution.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import statistics

from .job_queue import JobQueue, Job, JobStatus, JobPriority
from .resource_allocator import ResourceAllocator
from .batch_processor import BatchProcessor
from .cost_optimizer import CostOptimizer


logger = logging.getLogger(__name__)


class SchedulingPolicy(Enum):
    """Scheduling policy types"""
    FIFO = "fifo"  # First In First Out
    LIFO = "lifo"  # Last In First Out
    PRIORITY = "priority"  # Priority-based
    ROUND_ROBIN = "round_robin"  # Round-robin
    SHORTEST_JOB_FIRST = "sjf"  # Shortest Job First
    EARLIEST_DEADLINE_FIRST = "edf"  # Earliest Deadline First
    FAIR_SHARE = "fair_share"  # Fair share among users
    ADAPTIVE = "adaptive"  # Adaptive based on metrics


@dataclass
class SchedulerConfig:
    """Configuration for smart scheduler"""
    policy: SchedulingPolicy = SchedulingPolicy.PRIORITY
    max_concurrent_jobs: int = 10
    enable_batching: bool = True
    batch_size: int = 5
    batch_timeout: float = 5.0
    enable_cost_optimization: bool = True
    cost_budget: Optional[float] = None
    enable_preemption: bool = False
    enable_backfilling: bool = True
    starvation_threshold: int = 100  # Max queue time before priority boost


class SmartScheduler:
    """Intelligent job scheduler with multiple strategies"""

    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self.job_queue = JobQueue()
        self.resource_allocator = ResourceAllocator()
        self.batch_processor = BatchProcessor()
        self.cost_optimizer = CostOptimizer()

        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.worker_pool: List[asyncio.Task] = []
        self._shutdown = False
        self._metrics: Dict[str, Any] = {
            "jobs_scheduled": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "avg_queue_time": 0,
            "avg_execution_time": 0,
            "total_cost": 0,
            "resource_utilization": {}
        }

    async def start(self, num_workers: Optional[int] = None):
        """Start scheduler with worker pool"""
        num_workers = num_workers or self.config.max_concurrent_jobs

        # Start worker tasks
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(i))
            self.worker_pool.append(worker)

        # Start monitoring task
        asyncio.create_task(self._monitor())

        logger.info(f"Started scheduler with {num_workers} workers")

    async def stop(self):
        """Stop scheduler and workers"""
        self._shutdown = True

        # Cancel running jobs
        for task in self.running_jobs.values():
            task.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.worker_pool, return_exceptions=True)

        logger.info("Scheduler stopped")

    async def schedule_job(self, job: Job) -> str:
        """Schedule a job for execution"""
        # Apply scheduling policy adjustments
        job = self._apply_policy_adjustments(job)

        # Check cost constraints
        if self.config.enable_cost_optimization:
            if not self.cost_optimizer.can_afford_job(job, self.config.cost_budget):
                raise RuntimeError(f"Job {job.id} exceeds cost budget")

        # Add to queue
        job_id = await self.job_queue.add_job(job)
        self._metrics["jobs_scheduled"] += 1

        logger.info(f"Scheduled job {job_id}")
        return job_id

    async def schedule_batch(self, jobs: List[Job]) -> List[str]:
        """Schedule multiple jobs as a batch"""
        if self.config.enable_batching:
            # Optimize batch for execution
            optimized_batch = await self.batch_processor.optimize_batch(jobs)
            job_ids = []

            for job in optimized_batch:
                job_id = await self.schedule_job(job)
                job_ids.append(job_id)

            return job_ids
        else:
            # Schedule individually
            return [await self.schedule_job(job) for job in jobs]

    async def _worker(self, worker_id: int):
        """Worker task that executes jobs"""
        logger.info(f"Worker {worker_id} started")

        while not self._shutdown:
            try:
                # Get next job based on policy
                job = await self._get_next_job()

                if not job:
                    await asyncio.sleep(0.1)
                    continue

                # Check resource availability
                if not await self._check_resources(job):
                    # Requeue job
                    await self.job_queue.requeue_job(job.id)
                    await asyncio.sleep(1)
                    continue

                # Allocate resources
                resources = await self.resource_allocator.allocate(
                    job.id, job.resource_requirements
                )

                # Execute job
                await self._execute_job(job, worker_id)

                # Release resources
                await self.resource_allocator.release(job.id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.info(f"Worker {worker_id} stopped")

    async def _get_next_job(self) -> Optional[Job]:
        """Get next job based on scheduling policy"""
        if self.config.policy == SchedulingPolicy.FIFO:
            return await self.job_queue.get_next_job(timeout=1.0)

        elif self.config.policy == SchedulingPolicy.PRIORITY:
            # Check for starvation
            jobs = self.job_queue.list_jobs(status=JobStatus.QUEUED)
            for job in jobs:
                queue_time = (datetime.now() - job.created_at).total_seconds()
                if queue_time > self.config.starvation_threshold:
                    # Boost priority
                    job.priority = JobPriority.HIGH
                    logger.info(f"Boosted priority for starving job {job.id}")

            return await self.job_queue.get_next_job(timeout=1.0)

        elif self.config.policy == SchedulingPolicy.SHORTEST_JOB_FIRST:
            # Get all ready jobs and sort by estimated time
            jobs = self.job_queue.list_jobs(status=JobStatus.QUEUED)
            ready_jobs = [j for j in jobs if j.is_ready(self.job_queue.completed_jobs)]

            if ready_jobs:
                # Sort by estimated execution time
                ready_jobs.sort(key=lambda j: j.metadata.get("estimated_time", float('inf')))
                job = ready_jobs[0]
                await self.job_queue.mark_running(job.id)
                return job

        elif self.config.policy == SchedulingPolicy.ADAPTIVE:
            return await self._adaptive_scheduling()

        else:
            return await self.job_queue.get_next_job(timeout=1.0)

    async def _adaptive_scheduling(self) -> Optional[Job]:
        """Adaptive scheduling based on system metrics"""
        # Analyze current metrics
        utilization = self._calculate_utilization()

        # High utilization: prefer short jobs
        if utilization > 0.8:
            jobs = self.job_queue.list_jobs(status=JobStatus.QUEUED)
            ready_jobs = [j for j in jobs if j.is_ready(self.job_queue.completed_jobs)]
            if ready_jobs:
                ready_jobs.sort(key=lambda j: j.metadata.get("estimated_time", float('inf')))
                job = ready_jobs[0]
                await self.job_queue.mark_running(job.id)
                return job

        # Low utilization: prefer high priority
        else:
            return await self.job_queue.get_next_job(timeout=1.0)

    async def _check_resources(self, job: Job) -> bool:
        """Check if resources are available for job"""
        if not job.resource_requirements:
            return True

        return await self.resource_allocator.check_availability(job.resource_requirements)

    async def _execute_job(self, job: Job, worker_id: int):
        """Execute a job"""
        logger.info(f"Worker {worker_id} executing job {job.id}")

        # Mark as running
        await self.job_queue.mark_running(job.id)

        # Create execution task
        if job.handler:
            if asyncio.iscoroutinefunction(job.handler):
                task = asyncio.create_task(job.handler(job.inputs))
            else:
                # Run sync handler in executor
                loop = asyncio.get_event_loop()
                task = loop.run_in_executor(None, job.handler, job.inputs)
        else:
            # Dummy execution
            task = asyncio.create_task(self._dummy_execution(job))

        self.running_jobs[job.id] = task

        try:
            # Execute with timeout
            if job.timeout_seconds:
                result = await asyncio.wait_for(task, timeout=job.timeout_seconds)
            else:
                result = await task

            # Mark completed
            await self.job_queue.mark_completed(job.id, result)
            self._metrics["jobs_completed"] += 1

            # Update cost
            if self.config.enable_cost_optimization:
                cost = self.cost_optimizer.calculate_job_cost(job)
                job.metrics.cost = cost
                self._metrics["total_cost"] += cost

            logger.info(f"Job {job.id} completed successfully")

        except asyncio.TimeoutError:
            job.status = JobStatus.TIMEOUT
            await self.job_queue.mark_failed(job.id, "Timeout")
            self._metrics["jobs_failed"] += 1
            logger.error(f"Job {job.id} timed out")

        except Exception as e:
            await self.job_queue.mark_failed(job.id, str(e))
            self._metrics["jobs_failed"] += 1
            logger.error(f"Job {job.id} failed: {e}")

            # Retry logic
            if job.metrics.retry_count < job.max_retries:
                await self.job_queue.requeue_job(job.id)
                logger.info(f"Requeued job {job.id} for retry")

        finally:
            # Clean up
            del self.running_jobs[job.id]

    async def _dummy_execution(self, job: Job) -> Dict[str, Any]:
        """Dummy execution for testing"""
        execution_time = job.metadata.get("estimated_time", 1.0)
        await asyncio.sleep(execution_time)
        return {"status": "success", "job_id": job.id}

    def _apply_policy_adjustments(self, job: Job) -> Job:
        """Apply scheduling policy adjustments to job"""
        if self.config.policy == SchedulingPolicy.FAIR_SHARE:
            # Adjust priority based on user's resource usage
            user_id = job.metadata.get("user_id")
            if user_id:
                usage = self._get_user_usage(user_id)
                if usage > 0.5:  # High usage
                    job.priority = JobPriority.LOW

        return job

    def _calculate_utilization(self) -> float:
        """Calculate current resource utilization"""
        active_jobs = len(self.running_jobs)
        max_jobs = self.config.max_concurrent_jobs
        return active_jobs / max_jobs if max_jobs > 0 else 0

    def _get_user_usage(self, user_id: str) -> float:
        """Get resource usage for a user"""
        # Calculate user's share of running jobs
        user_jobs = sum(
            1 for job_id in self.running_jobs
            if self.job_queue.get_job(job_id).metadata.get("user_id") == user_id
        )
        total_jobs = len(self.running_jobs)
        return user_jobs / total_jobs if total_jobs > 0 else 0

    async def _monitor(self):
        """Monitor scheduler performance"""
        while not self._shutdown:
            try:
                # Update metrics
                self._update_metrics()

                # Log metrics periodically
                if self._metrics["jobs_scheduled"] % 100 == 0:
                    logger.info(f"Scheduler metrics: {self._metrics}")

                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")

    def _update_metrics(self):
        """Update scheduler metrics"""
        jobs = self.job_queue.list_jobs()

        # Calculate average times
        queue_times = [j.metrics.queue_time for j in jobs if j.metrics.queue_time > 0]
        exec_times = [j.metrics.execution_time for j in jobs if j.metrics.execution_time > 0]

        if queue_times:
            self._metrics["avg_queue_time"] = statistics.mean(queue_times)
        if exec_times:
            self._metrics["avg_execution_time"] = statistics.mean(exec_times)

        # Resource utilization
        self._metrics["resource_utilization"] = {
            "jobs": self._calculate_utilization(),
            "memory": self.resource_allocator.get_utilization("memory"),
            "cpu": self.resource_allocator.get_utilization("cpu")
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler metrics"""
        self._update_metrics()
        return self._metrics.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            "running": not self._shutdown,
            "policy": self.config.policy.value,
            "workers": len(self.worker_pool),
            "active_jobs": len(self.running_jobs),
            "queue_stats": self.job_queue.get_queue_stats(),
            "metrics": self.get_metrics()
        }