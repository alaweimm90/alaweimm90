"""
Database performance monitoring and alerting.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class DatabaseMetrics:
    """Database performance metrics."""
    timestamp: datetime
    queries_per_second: float
    avg_query_time_ms: float
    connection_count: int
    deadlock_count: int
    cache_hit_ratio: float
    disk_usage_mb: float
    cpu_usage_percent: float
    memory_usage_mb: float


class DatabaseMonitor:
    """
    Real-time database performance monitoring
    with alerting and trend analysis.
    """

    def __init__(self, connection_pool, alert_thresholds: Optional[Dict] = None):
        """Initialize database monitor."""
        self.pool = connection_pool
        self.metrics_history = deque(maxlen=1000)
        self.alerts = []

        self.thresholds = alert_thresholds or {
            "slow_query_ms": 1000,
            "high_connection_count": 80,
            "low_cache_hit_ratio": 0.8,
            "high_cpu_percent": 80,
            "high_disk_usage_mb": 10000
        }

    async def collect_metrics(self) -> DatabaseMetrics:
        """Collect current database metrics."""
        # Query database for metrics
        # This is database-specific (PostgreSQL example)

        metrics_query = """
        SELECT
            (SELECT count(*) FROM pg_stat_activity) as connection_count,
            (SELECT sum(deadlocks) FROM pg_stat_database) as deadlock_count,
            (SELECT sum(blks_hit)::float / (sum(blks_hit) + sum(blks_read))
             FROM pg_stat_database WHERE blks_hit + blks_read > 0) as cache_hit_ratio
        """

        async with self.pool.get_connection() as conn:
            result = await conn.execute(metrics_query)
            row = result.fetchone()

        metrics = DatabaseMetrics(
            timestamp=datetime.utcnow(),
            queries_per_second=0,  # Would calculate from pg_stat_statements
            avg_query_time_ms=0,  # Would calculate from query stats
            connection_count=row[0],
            deadlock_count=row[1] or 0,
            cache_hit_ratio=row[2] or 0,
            disk_usage_mb=0,  # Would query from pg_database_size
            cpu_usage_percent=0,  # Would get from system metrics
            memory_usage_mb=0  # Would get from system metrics
        )

        self.metrics_history.append(metrics)
        await self._check_alerts(metrics)

        return metrics

    async def _check_alerts(self, metrics: DatabaseMetrics) -> None:
        """Check metrics against alert thresholds."""
        if metrics.connection_count > self.thresholds["high_connection_count"]:
            self._add_alert(
                "high_connections",
                f"Connection count ({metrics.connection_count}) exceeds threshold"
            )

        if metrics.cache_hit_ratio < self.thresholds["low_cache_hit_ratio"]:
            self._add_alert(
                "low_cache_hits",
                f"Cache hit ratio ({metrics.cache_hit_ratio:.2%}) below threshold"
            )

    def _add_alert(self, alert_type: str, message: str) -> None:
        """Add an alert."""
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.utcnow()
        }
        self.alerts.append(alert)
        logger.warning(f"Database alert: {message}")

    async def analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends."""
        if len(self.metrics_history) < 10:
            return {"status": "insufficient_data"}

        recent = list(self.metrics_history)[-100:]

        return {
            "avg_connections": sum(m.connection_count for m in recent) / len(recent),
            "avg_cache_hit_ratio": sum(m.cache_hit_ratio for m in recent) / len(recent),
            "deadlock_trend": "increasing" if recent[-1].deadlock_count > recent[0].deadlock_count else "stable",
            "performance_score": self._calculate_performance_score(recent)
        }

    def _calculate_performance_score(self, metrics: List[DatabaseMetrics]) -> float:
        """Calculate overall performance score."""
        if not metrics:
            return 0

        latest = metrics[-1]
        score = 100.0

        # Deduct for high connection usage
        if latest.connection_count > self.thresholds["high_connection_count"]:
            score -= 20

        # Deduct for low cache hits
        if latest.cache_hit_ratio < self.thresholds["low_cache_hit_ratio"]:
            score -= 30

        # Deduct for deadlocks
        if latest.deadlock_count > 0:
            score -= 10 * min(latest.deadlock_count, 5)

        return max(0, score)

    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts."""
        return self.alerts[-limit:]

    async def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        if self.metrics_history:
            latest = self.metrics_history[-1]
            return {
                "current_metrics": {
                    "connection_count": latest.connection_count,
                    "cache_hit_ratio": latest.cache_hit_ratio,
                    "deadlock_count": latest.deadlock_count
                },
                "trends": await self.analyze_trends(),
                "alert_count": len(self.alerts),
                "metrics_collected": len(self.metrics_history)
            }

        return {"status": "no_metrics"}