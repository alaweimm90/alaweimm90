"""
Real-time task monitoring and progress tracking.
"""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TaskMetrics:
    """Task execution metrics."""
    task_name: str
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration_ms: float = 0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0
    avg_duration_ms: float = 0
    p95_duration_ms: float = 0
    p99_duration_ms: float = 0


class TaskMonitor:
    """
    Real-time monitoring of task execution with metrics,
    alerts, and progress tracking.
    """

    def __init__(self, task_queue):
        """Initialize task monitor."""
        self.queue = task_queue
        self._metrics: Dict[str, TaskMetrics] = {}
        self._recent_tasks = deque(maxlen=1000)
        self._alerts: List[Dict[str, Any]] = []
        self._thresholds = {
            "failure_rate": 0.1,
            "slow_task_ms": 5000,
            "queue_size": 1000
        }

    async def update_metrics(
        self,
        task_name: str,
        duration_ms: float,
        success: bool
    ) -> None:
        """Update metrics for task execution."""
        if task_name not in self._metrics:
            self._metrics[task_name] = TaskMetrics(task_name=task_name)

        metrics = self._metrics[task_name]
        metrics.execution_count += 1

        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1

        metrics.total_duration_ms += duration_ms
        metrics.min_duration_ms = min(metrics.min_duration_ms, duration_ms)
        metrics.max_duration_ms = max(metrics.max_duration_ms, duration_ms)
        metrics.avg_duration_ms = metrics.total_duration_ms / metrics.execution_count

        # Check for alerts
        await self._check_alerts(task_name, duration_ms, success)

    async def _check_alerts(
        self,
        task_name: str,
        duration_ms: float,
        success: bool
    ) -> None:
        """Check for alert conditions."""
        # Slow task alert
        if duration_ms > self._thresholds["slow_task_ms"]:
            alert = {
                "type": "slow_task",
                "task_name": task_name,
                "duration_ms": duration_ms,
                "timestamp": datetime.utcnow()
            }
            self._alerts.append(alert)
            logger.warning(f"Slow task alert: {task_name} took {duration_ms}ms")

        # Failure rate alert
        metrics = self._metrics[task_name]
        if metrics.execution_count > 10:
            failure_rate = metrics.failure_count / metrics.execution_count
            if failure_rate > self._thresholds["failure_rate"]:
                alert = {
                    "type": "high_failure_rate",
                    "task_name": task_name,
                    "failure_rate": failure_rate,
                    "timestamp": datetime.utcnow()
                }
                self._alerts.append(alert)
                logger.warning(f"High failure rate alert: {task_name} at {failure_rate:.2%}")

    async def get_task_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get progress for a specific task."""
        # Query task status from queue
        status = await self.queue.get_task_status(task_id)

        if not status:
            return None

        return {
            "task_id": task_id,
            "status": status.value,
            "progress": 0.5 if status == "running" else 1.0 if status == "completed" else 0.0,
            "timestamp": datetime.utcnow()
        }

    async def get_queue_health(self) -> Dict[str, Any]:
        """Get overall queue health metrics."""
        stats = await self.queue.get_stats()

        total_queued = sum(stats["queue_sizes"].values())
        processing = stats["processing"]
        completed = stats["completed"]
        failed = stats["failed"]

        health_score = 100.0

        # Deduct for high queue size
        if total_queued > self._thresholds["queue_size"]:
            health_score -= 20

        # Deduct for high failure rate
        if completed > 0:
            failure_rate = failed / (completed + failed)
            if failure_rate > self._thresholds["failure_rate"]:
                health_score -= 30

        # Deduct for circuit breaker
        if stats.get("circuit_breaker_open"):
            health_score -= 40

        return {
            "health_score": max(0, health_score),
            "status": "healthy" if health_score > 70 else "degraded" if health_score > 40 else "unhealthy",
            "total_queued": total_queued,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "workers": stats["workers"],
            "timestamp": datetime.utcnow()
        }

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all task metrics."""
        summary = {
            "tasks": {},
            "totals": {
                "execution_count": 0,
                "success_count": 0,
                "failure_count": 0
            }
        }

        for task_name, metrics in self._metrics.items():
            summary["tasks"][task_name] = {
                "execution_count": metrics.execution_count,
                "success_count": metrics.success_count,
                "failure_count": metrics.failure_count,
                "success_rate": metrics.success_count / metrics.execution_count if metrics.execution_count > 0 else 0,
                "avg_duration_ms": metrics.avg_duration_ms,
                "min_duration_ms": metrics.min_duration_ms,
                "max_duration_ms": metrics.max_duration_ms
            }

            summary["totals"]["execution_count"] += metrics.execution_count
            summary["totals"]["success_count"] += metrics.success_count
            summary["totals"]["failure_count"] += metrics.failure_count

        return summary

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return self._alerts[-limit:]

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts.clear()