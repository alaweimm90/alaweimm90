"""
Cache analytics and monitoring module.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class CacheMetric:
    """Cache performance metric."""
    timestamp: float
    operation: str
    latency_ms: float
    cache_level: str
    hit: bool


class CacheAnalytics:
    """Real-time cache analytics and monitoring."""

    def __init__(self, namespace: str, window_size: int = 1000):
        """Initialize cache analytics."""
        self.namespace = namespace
        self.window_size = window_size

        self._metrics = deque(maxlen=window_size)
        self._hit_count = defaultdict(int)
        self._miss_count = 0
        self._operation_count = defaultdict(int)
        self._latency_sum = defaultdict(float)
        self._start_time = time.time()

    def record_hit(self, cache_level: str, latency: float) -> None:
        """Record cache hit."""
        metric = CacheMetric(
            timestamp=time.time(),
            operation="get",
            latency_ms=latency * 1000,
            cache_level=cache_level,
            hit=True
        )
        self._metrics.append(metric)
        self._hit_count[cache_level] += 1
        self._operation_count["get"] += 1
        self._latency_sum["get"] += latency

    def record_miss(self, latency: float) -> None:
        """Record cache miss."""
        metric = CacheMetric(
            timestamp=time.time(),
            operation="get",
            latency_ms=latency * 1000,
            cache_level="none",
            hit=False
        )
        self._metrics.append(metric)
        self._miss_count += 1
        self._operation_count["get"] += 1
        self._latency_sum["get"] += latency

    def record_set(self) -> None:
        """Record cache set operation."""
        self._operation_count["set"] += 1

    def record_delete(self) -> None:
        """Record cache delete operation."""
        self._operation_count["delete"] += 1

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        total_hits = sum(self._hit_count.values())
        total_requests = total_hits + self._miss_count
        hit_rate = total_hits / total_requests if total_requests > 0 else 0

        # Calculate average latencies
        avg_latencies = {}
        for op, count in self._operation_count.items():
            if count > 0:
                avg_latencies[op] = self._latency_sum.get(op, 0) / count * 1000

        # Calculate hit rate by level
        hit_rates_by_level = {}
        for level, hits in self._hit_count.items():
            hit_rates_by_level[level] = hits / total_requests if total_requests > 0 else 0

        uptime_seconds = time.time() - self._start_time

        return {
            "namespace": self.namespace,
            "uptime_seconds": uptime_seconds,
            "total_requests": total_requests,
            "total_hits": total_hits,
            "total_misses": self._miss_count,
            "hit_rate": hit_rate,
            "hit_rates_by_level": hit_rates_by_level,
            "operations": dict(self._operation_count),
            "average_latencies_ms": avg_latencies,
            "recent_metrics": len(self._metrics)
        }