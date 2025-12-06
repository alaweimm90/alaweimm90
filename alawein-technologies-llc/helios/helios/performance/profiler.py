"""
Dynamic Performance Profiler

Tracks:
- Function execution time
- Memory allocation
- Resource utilization
- Performance trends
"""

import cProfile
import pstats
import time
import psutil
from functools import wraps
from typing import Dict, Any, Callable, List, Optional
from datetime import datetime
import io
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

class PerformanceMetrics:
    """Track performance metrics for a function."""

    def __init__(self, name: str):
        self.name = name
        self.executions: List[Dict[str, float]] = []
        self.total_time = 0.0
        self.total_calls = 0
        self.peak_memory = 0
        self.avg_time = 0.0

    def record(self, duration: float, memory_used: float):
        """Record execution metrics."""
        self.executions.append({
            "duration": duration,
            "memory_mb": memory_used / (1024 * 1024),
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.total_time += duration
        self.total_calls += 1
        self.peak_memory = max(self.peak_memory, memory_used / (1024 * 1024))
        self.avg_time = self.total_time / self.total_calls

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.executions:
            return {}

        times = [e["duration"] for e in self.executions]
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "total_time_seconds": round(self.total_time, 3),
            "average_time_ms": round(self.avg_time * 1000, 2),
            "min_time_ms": round(min(times) * 1000, 2),
            "max_time_ms": round(max(times) * 1000, 2),
            "peak_memory_mb": round(self.peak_memory, 2),
        }


# ============================================================================
# DYNAMIC PROFILER
# ============================================================================

class DynamicProfiler:
    """Profile application performance dynamically."""

    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss

    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a function."""
        func_name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self.process.memory_info().rss

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                memory_used = self.process.memory_info().rss - start_memory

                # Track metrics
                if func_name not in self.metrics:
                    self.metrics[func_name] = PerformanceMetrics(func_name)

                self.metrics[func_name].record(duration, abs(memory_used))

                if duration > 1.0:  # Log slow functions
                    logger.warning(
                        f"Slow function: {func_name} took {duration:.2f}s"
                    )

        return wrapper

    async def profile_async_function(self, func: Callable) -> Callable:
        """Decorator to profile async function."""
        func_name = func.__name__

        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self.process.memory_info().rss

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                memory_used = self.process.memory_info().rss - start_memory

                if func_name not in self.metrics:
                    self.metrics[func_name] = PerformanceMetrics(func_name)

                self.metrics[func_name].record(duration, abs(memory_used))

                if duration > 1.0:
                    logger.warning(
                        f"Slow async function: {func_name} took {duration:.2f}s"
                    )

        return wrapper

    def get_metrics(self, func_name: Optional[str] = None) -> Dict[str, Any]:
        """Get collected metrics."""
        if func_name:
            if func_name in self.metrics:
                return self.metrics[func_name].get_stats()
            return {}

        return {
            name: metrics.get_stats()
            for name, metrics in self.metrics.items()
        }

    def get_slowest_functions(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get slowest functions by total time."""
        stats = [
            {
                **metrics.get_stats(),
                "name": name,
            }
            for name, metrics in self.metrics.items()
        ]

        # Sort by total time
        stats.sort(key=lambda x: x.get("total_time_seconds", 0), reverse=True)
        return stats[:top_n]

    def get_memory_hotspots(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get functions with highest memory usage."""
        stats = [
            {
                **metrics.get_stats(),
                "name": name,
            }
            for name, metrics in self.metrics.items()
        ]

        stats.sort(key=lambda x: x.get("peak_memory_mb", 0), reverse=True)
        return stats[:top_n]

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {"message": "No profiling data collected"}

        slowest = self.get_slowest_functions(5)
        memory_intensive = self.get_memory_hotspots(5)
        total_calls = sum(m.total_calls for m in self.metrics.values())
        total_time = sum(m.total_time for m in self.metrics.values())

        return {
            "total_functions": len(self.metrics),
            "total_calls": total_calls,
            "total_time_seconds": round(total_time, 2),
            "slowest_functions": slowest,
            "memory_intensive": memory_intensive,
            "current_memory_mb": round(
                (self.process.memory_info().rss - self.initial_memory) / (1024 * 1024),
                2
            ),
        }

    def reset(self):
        """Reset collected metrics."""
        self.metrics.clear()
        self.initial_memory = self.process.memory_info().rss


# ============================================================================
# GLOBAL PROFILER
# ============================================================================

_global_profiler: Optional[DynamicProfiler] = None


def initialize_profiler() -> DynamicProfiler:
    """Initialize global profiler."""
    global _global_profiler
    _global_profiler = DynamicProfiler()
    return _global_profiler


def get_profiler() -> DynamicProfiler:
    """Get global profiler."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = DynamicProfiler()
    return _global_profiler
