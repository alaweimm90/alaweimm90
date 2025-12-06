"""
Advanced task scheduling with cron-like expressions and recurrence.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from croniter import croniter

from .task_queue import TaskPriority, TaskQueue

logger = logging.getLogger(__name__)


class TaskScheduler:
    """
    Task scheduler with cron expressions, recurrence patterns,
    and advanced scheduling capabilities.
    """

    def __init__(self, task_queue: TaskQueue):
        """Initialize task scheduler."""
        self.queue = task_queue
        self._scheduled_jobs: Dict[str, Dict[str, Any]] = {}
        self._scheduler_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def schedule_cron(
        self,
        job_id: str,
        task_name: str,
        cron_expression: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timezone: str = "UTC"
    ) -> bool:
        """
        Schedule task with cron expression.

        Args:
            job_id: Unique job identifier
            task_name: Task to execute
            cron_expression: Cron expression (e.g., "0 0 * * *")
            payload: Task payload
            priority: Task priority
            timezone: Timezone for scheduling

        Returns:
            Success status
        """
        try:
            # Validate cron expression
            cron = croniter(cron_expression, datetime.utcnow())

            self._scheduled_jobs[job_id] = {
                "task_name": task_name,
                "cron_expression": cron_expression,
                "cron": cron,
                "payload": payload or {},
                "priority": priority,
                "timezone": timezone,
                "enabled": True,
                "last_run": None,
                "next_run": cron.get_next(datetime)
            }

            logger.info(f"Scheduled cron job {job_id}: {cron_expression}")
            return True

        except Exception as e:
            logger.error(f"Failed to schedule cron job {job_id}: {e}")
            return False

    async def schedule_recurring(
        self,
        job_id: str,
        task_name: str,
        interval_seconds: int,
        payload: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_runs: Optional[int] = None
    ) -> bool:
        """
        Schedule recurring task with fixed interval.

        Args:
            job_id: Unique job identifier
            task_name: Task to execute
            interval_seconds: Interval between executions
            payload: Task payload
            priority: Task priority
            max_runs: Maximum number of runs

        Returns:
            Success status
        """
        self._scheduled_jobs[job_id] = {
            "task_name": task_name,
            "interval_seconds": interval_seconds,
            "payload": payload or {},
            "priority": priority,
            "enabled": True,
            "max_runs": max_runs,
            "run_count": 0,
            "last_run": None,
            "next_run": datetime.utcnow() + timedelta(seconds=interval_seconds)
        }

        logger.info(f"Scheduled recurring job {job_id}: every {interval_seconds}s")
        return True

    async def schedule_once(
        self,
        job_id: str,
        task_name: str,
        run_at: datetime,
        payload: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> bool:
        """
        Schedule one-time task execution.

        Args:
            job_id: Unique job identifier
            task_name: Task to execute
            run_at: When to execute
            payload: Task payload
            priority: Task priority

        Returns:
            Success status
        """
        delay_seconds = (run_at - datetime.utcnow()).total_seconds()

        if delay_seconds <= 0:
            logger.warning(f"Scheduled time for job {job_id} is in the past")
            return False

        task_id = await self.queue.enqueue(
            task_name=task_name,
            payload=payload,
            priority=priority,
            delay_seconds=int(delay_seconds)
        )

        self._scheduled_jobs[job_id] = {
            "task_name": task_name,
            "run_at": run_at,
            "task_id": task_id,
            "one_time": True,
            "enabled": False  # Will be removed after execution
        }

        logger.info(f"Scheduled one-time job {job_id} at {run_at}")
        return True

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel scheduled job."""
        if job_id in self._scheduled_jobs:
            self._scheduled_jobs[job_id]["enabled"] = False
            logger.info(f"Cancelled scheduled job {job_id}")
            return True
        return False

    async def pause_job(self, job_id: str) -> bool:
        """Pause scheduled job."""
        if job_id in self._scheduled_jobs:
            self._scheduled_jobs[job_id]["enabled"] = False
            logger.info(f"Paused scheduled job {job_id}")
            return True
        return False

    async def resume_job(self, job_id: str) -> bool:
        """Resume paused job."""
        if job_id in self._scheduled_jobs:
            job = self._scheduled_jobs[job_id]
            job["enabled"] = True

            # Recalculate next run
            if "cron" in job:
                job["next_run"] = job["cron"].get_next(datetime)
            elif "interval_seconds" in job:
                job["next_run"] = datetime.utcnow() + timedelta(seconds=job["interval_seconds"])

            logger.info(f"Resumed scheduled job {job_id}")
            return True
        return False

    async def run_scheduler(self) -> None:
        """Main scheduler loop."""
        logger.info("Task scheduler started")

        while not self._shutdown:
            try:
                current_time = datetime.utcnow()

                # Check all scheduled jobs
                for job_id, job in list(self._scheduled_jobs.items()):
                    if not job.get("enabled", False):
                        continue

                    # Check if job should run
                    if job.get("next_run") and current_time >= job["next_run"]:
                        await self._execute_scheduled_job(job_id, job)

                # Sleep for a short interval
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(5)

        logger.info("Task scheduler stopped")

    async def _execute_scheduled_job(self, job_id: str, job: Dict[str, Any]) -> None:
        """Execute a scheduled job."""
        try:
            # Enqueue task
            task_id = await self.queue.enqueue(
                task_name=job["task_name"],
                payload=job["payload"],
                priority=job.get("priority", TaskPriority.NORMAL)
            )

            # Update job metadata
            job["last_run"] = datetime.utcnow()

            # Handle different job types
            if "cron" in job:
                # Cron job - calculate next run
                job["next_run"] = job["cron"].get_next(datetime)

            elif "interval_seconds" in job:
                # Recurring job
                job["run_count"] = job.get("run_count", 0) + 1

                # Check max runs
                if job.get("max_runs") and job["run_count"] >= job["max_runs"]:
                    job["enabled"] = False
                    logger.info(f"Job {job_id} completed {job['max_runs']} runs")
                else:
                    job["next_run"] = datetime.utcnow() + timedelta(seconds=job["interval_seconds"])

            elif job.get("one_time"):
                # One-time job - disable after execution
                job["enabled"] = False
                del self._scheduled_jobs[job_id]

            logger.info(f"Executed scheduled job {job_id} (task {task_id})")

        except Exception as e:
            logger.error(f"Failed to execute scheduled job {job_id}: {e}")

    async def start(self) -> None:
        """Start the scheduler."""
        if not self._scheduler_task:
            self._scheduler_task = asyncio.create_task(self.run_scheduler())
            logger.info("Scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._shutdown = True
        if self._scheduler_task:
            await self._scheduler_task
            self._scheduler_task = None
        logger.info("Scheduler stopped")

    def get_scheduled_jobs(self) -> List[Dict[str, Any]]:
        """Get list of all scheduled jobs."""
        jobs = []
        for job_id, job in self._scheduled_jobs.items():
            jobs.append({
                "id": job_id,
                "task_name": job["task_name"],
                "enabled": job.get("enabled", False),
                "last_run": job.get("last_run"),
                "next_run": job.get("next_run"),
                "run_count": job.get("run_count", 0)
            })
        return jobs