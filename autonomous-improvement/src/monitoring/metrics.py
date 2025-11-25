"""
Metrics Collector for REPZ Workflow System

Collects, stores, and provides access to system and workflow performance metrics.
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and manages system and workflow metrics"""

    def __init__(self):
        self.settings = get_settings()

        # Metrics storage
        self.system_metrics: Dict[str, Any] = {}
        self.workflow_metrics: Dict[str, Any] = {}
        self.step_metrics: Dict[int, Dict[str, Any]] = {}
        self.time_series_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Collection state
        self._collection_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_collection = datetime.utcnow()

        # Performance tracking
        self.collection_count = 0
        self.error_count = 0

    async def start_collection(self):
        """Start metrics collection"""
        if self._running:
            logger.warning("Metrics collection already running")
            return

        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())

        logger.info("Metrics collection started")

    async def stop_collection(self):
        """Stop metrics collection"""
        self._running = False

        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        logger.info("Metrics collection stopped")

    async def _collection_loop(self):
        """Main metrics collection loop"""
        while self._running:
            try:
                await self._collect_system_metrics()
                await self._collect_workflow_metrics()
                await self._cleanup_old_data()

                self.collection_count += 1
                self._last_collection = datetime.utcnow()

            except Exception as e:
                self.error_count += 1
                logger.error(f"Metrics collection error: {e}")

            await asyncio.sleep(self.settings.metrics_collection_interval)

    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used
            memory_total = memory.total

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used = disk.used
            disk_total = disk.total

            # Network metrics
            network = psutil.net_io_counters()
            bytes_sent = network.bytes_sent
            bytes_recv = network.bytes_recv

            # System load
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)

            self.system_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count
                },
                "memory": {
                    "percent": memory_percent,
                    "used": memory_used,
                    "total": memory_total
                },
                "disk": {
                    "percent": disk_percent,
                    "used": disk_used,
                    "total": disk_total
                },
                "network": {
                    "bytes_sent": bytes_sent,
                    "bytes_recv": bytes_recv
                },
                "load_average": load_avg
            }

            # Store time series data
            self._store_time_series("cpu_percent", cpu_percent)
            self._store_time_series("memory_percent", memory_percent)
            self._store_time_series("disk_percent", disk_percent)

        except Exception as e:
            logger.error(f"System metrics collection error: {e}")

    async def _collect_workflow_metrics(self):
        """Collect workflow-specific metrics"""
        try:
            # Workflow progress metrics
            self.workflow_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "collection_count": self.collection_count,
                "error_count": self.error_count,
                "uptime_seconds": (datetime.utcnow() - self._last_collection).total_seconds()
                if self._last_collection else 0,
                "active_steps": len(self.step_metrics),
                "total_step_executions": sum(
                    metrics.get("execution_count", 0)
                    for metrics in self.step_metrics.values()
                )
            }

        except Exception as e:
            logger.error(f"Workflow metrics collection error: {e}")

    def _store_time_series(self, metric_name: str, value: float):
        """Store time series data"""
        timestamp = datetime.utcnow().timestamp()
        self.time_series_data[metric_name].append((timestamp, value))

    async def _cleanup_old_data(self):
        """Clean up old metrics data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)

        # Clean time series data older than 24 hours
        for metric_name, data in self.time_series_data.items():
            while data and data[0][0] < cutoff_time.timestamp():
                data.popleft()

    async def record_step_execution(self, step_id: int, start_time: datetime,
                                  end_time: datetime, success: bool, error: Optional[str] = None):
        """Record step execution metrics"""
        if step_id not in self.step_metrics:
            self.step_metrics[step_id] = {
                "execution_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "total_duration": 0,
                "average_duration": 0,
                "last_execution": None,
                "errors": []
            }

        metrics = self.step_metrics[step_id]
        duration = (end_time - start_time).total_seconds()

        metrics["execution_count"] += 1
        metrics["total_duration"] += duration
        metrics["average_duration"] = metrics["total_duration"] / metrics["execution_count"]
        metrics["last_execution"] = end_time.isoformat()

        if success:
            metrics["success_count"] += 1
        else:
            metrics["failure_count"] += 1
            if error:
                metrics["errors"].append({
                    "timestamp": end_time.isoformat(),
                    "error": error
                })
                # Keep only last 10 errors
                metrics["errors"] = metrics["errors"][-10:]

        # Store time series data
        self._store_time_series(f"step_{step_id}_duration", duration)
        self._store_time_series(f"step_{step_id}_success", 1.0 if success else 0.0)

    async def record_step_completion(self, step_id: int, success: bool):
        """Record step completion (simplified)"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=30)  # Mock duration

        await self.record_step_execution(step_id, start_time, end_time, success)

    async def record_workflow_error(self, error: str):
        """Record workflow-level error"""
        self.error_count += 1

        # Store error in time series
        self._store_time_series("workflow_errors", 1.0)

        logger.error(f"Workflow error recorded: {error}")

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return self.system_metrics.copy()

    async def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get current workflow metrics"""
        return self.workflow_metrics.copy()

    async def get_step_metrics(self, step_id: Optional[int] = None) -> Dict[str, Any]:
        """Get step execution metrics"""
        if step_id is not None:
            return self.step_metrics.get(step_id, {}).copy()
        else:
            return self.step_metrics.copy()

    async def get_time_series_data(self, metric_name: str, hours: int = 1) -> List[tuple]:
        """Get time series data for a metric"""
        data = self.time_series_data.get(metric_name, deque())
        cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).timestamp()

        # Filter data within time range
        filtered_data = [(ts, val) for ts, val in data if ts >= cutoff_time]

        return filtered_data

    async def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        return {
            "system": self.system_metrics,
            "workflow": self.workflow_metrics,
            "steps": self.step_metrics,
            "time_series_available": list(self.time_series_data.keys()),
            "collection_stats": {
                "total_collections": self.collection_count,
                "total_errors": self.error_count,
                "last_collection": self._last_collection.isoformat() if self._last_collection else None
            }
        }

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            # Calculate success rates
            total_executions = sum(
                metrics.get("execution_count", 0)
                for metrics in self.step_metrics.values()
            )

            successful_executions = sum(
                metrics.get("success_count", 0)
                for metrics in self.step_metrics.values()
            )

            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0

            # Calculate average step duration
            durations = [
                metrics.get("average_duration", 0)
                for metrics in self.step_metrics.values()
                if metrics.get("execution_count", 0) > 0
            ]

            avg_duration = sum(durations) / len(durations) if durations else 0

            return {
                "total_step_executions": total_executions,
                "successful_executions": successful_executions,
                "success_rate_percent": success_rate,
                "average_step_duration_seconds": avg_duration,
                "total_steps_tracked": len(self.step_metrics),
                "system_health": self._assess_system_health()
            }

        except Exception as e:
            logger.error(f"Performance summary error: {e}")
            return {"error": str(e)}

    def _assess_system_health(self) -> str:
        """Assess overall system health"""
        try:
            cpu_percent = self.system_metrics.get("cpu", {}).get("percent", 0)
            memory_percent = self.system_metrics.get("memory", {}).get("percent", 0)
            disk_percent = self.system_metrics.get("disk", {}).get("percent", 0)

            # Health thresholds
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 95:
                return "critical"
            elif cpu_percent > 70 or memory_percent > 80 or disk_percent > 85:
                return "warning"
            else:
                return "healthy"

        except Exception:
            return "unknown"

    async def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        all_metrics = await self.get_all_metrics()

        if format == "json":
            import json
            return json.dumps(all_metrics, indent=2, default=str)
        elif format == "csv":
            # Simple CSV export for time series
            csv_lines = ["timestamp,metric,value"]
            for metric_name, data in self.time_series_data.items():
                for timestamp, value in data:
                    csv_lines.append(f"{timestamp},{metric_name},{value}")
            return "\n".join(csv_lines)
        else:
            return str(all_metrics)