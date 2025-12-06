"""
Metrics Collection for TalAI Turing Challenge System.

Provides Prometheus-compatible metrics collection for monitoring system performance.
"""

import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from threading import Lock
from functools import wraps


@dataclass
class Metric:
    """Base metric data structure."""

    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    help_text: str = ""


@dataclass
class Counter(Metric):
    """Counter metric - monotonically increasing value."""

    def increment(self, amount: float = 1.0) -> None:
        """Increment counter by amount."""
        self.value += amount
        self.timestamp = datetime.utcnow()


@dataclass
class Gauge(Metric):
    """Gauge metric - value that can go up or down."""

    def set(self, value: float) -> None:
        """Set gauge to specific value."""
        self.value = value
        self.timestamp = datetime.utcnow()

    def increment(self, amount: float = 1.0) -> None:
        """Increment gauge by amount."""
        self.value += amount
        self.timestamp = datetime.utcnow()

    def decrement(self, amount: float = 1.0) -> None:
        """Decrement gauge by amount."""
        self.value -= amount
        self.timestamp = datetime.utcnow()


@dataclass
class Histogram:
    """Histogram metric - track distribution of values."""

    name: str
    buckets: List[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    help_text: str = ""

    def __post_init__(self):
        self.observations: List[float] = []
        self.bucket_counts: Dict[float, int] = defaultdict(int)
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        """Record an observation."""
        self.observations.append(value)
        self.sum += value
        self.count += 1

        # Update bucket counts
        for bucket in self.buckets:
            if value <= bucket:
                self.bucket_counts[bucket] += 1

    def quantile(self, q: float) -> float:
        """Calculate quantile (e.g., 0.95 for 95th percentile)."""
        if not self.observations:
            return 0.0

        sorted_obs = sorted(self.observations)
        index = int(len(sorted_obs) * q)
        return sorted_obs[min(index, len(sorted_obs) - 1)]

    @property
    def mean(self) -> float:
        """Calculate mean value."""
        return self.sum / self.count if self.count > 0 else 0.0


class MetricsCollector:
    """
    Centralized metrics collector for TalAI.

    Thread-safe singleton for collecting and exposing metrics.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self._metrics_lock = Lock()

        # Initialize standard TalAI metrics
        self._init_standard_metrics()

    def _init_standard_metrics(self) -> None:
        """Initialize standard metrics for TalAI components."""
        # Turing Challenge System Metrics
        self.counter(
            "talair_validations_total",
            help_text="Total number of hypothesis validations",
        )
        self.counter(
            "talair_validations_failed",
            help_text="Number of failed validations",
        )

        # Agent Tournament Metrics
        self.counter(
            "talair_tournaments_total",
            help_text="Total number of tournaments run",
        )
        self.histogram(
            "talair_tournament_duration_seconds",
            help_text="Tournament execution time",
        )

        # Devil's Advocate Metrics
        self.counter(
            "talair_attacks_total",
            help_text="Total number of adversarial attacks",
        )
        self.counter(
            "talair_flaws_found_total",
            help_text="Total flaws identified",
        )

        # Swarm Voting Metrics
        self.counter(
            "talair_votes_total",
            help_text="Total votes cast",
        )
        self.gauge(
            "talair_swarm_size",
            help_text="Current swarm size",
        )

        # Emergent Behavior Metrics
        self.counter(
            "talair_patterns_detected_total",
            help_text="Total emergent patterns detected",
        )
        self.gauge(
            "talair_system_health",
            help_text="Current system health score (0-100)",
        )

        # Performance Metrics
        self.histogram(
            "talair_llm_latency_seconds",
            help_text="LLM API call latency",
        )
        self.histogram(
            "talair_validation_latency_seconds",
            help_text="Total validation latency",
        )

    def counter(self, name: str, labels: Optional[Dict[str, str]] = None, help_text: str = "") -> Counter:
        """Get or create a counter metric."""
        key = self._make_key(name, labels)

        with self._metrics_lock:
            if key not in self.counters:
                self.counters[key] = Counter(
                    name=name,
                    value=0.0,
                    labels=labels or {},
                    help_text=help_text,
                )
            return self.counters[key]

    def gauge(self, name: str, labels: Optional[Dict[str, str]] = None, help_text: str = "") -> Gauge:
        """Get or create a gauge metric."""
        key = self._make_key(name, labels)

        with self._metrics_lock:
            if key not in self.gauges:
                self.gauges[key] = Gauge(
                    name=name,
                    value=0.0,
                    labels=labels or {},
                    help_text=help_text,
                )
            return self.gauges[key]

    def histogram(self, name: str, help_text: str = "") -> Histogram:
        """Get or create a histogram metric."""
        with self._metrics_lock:
            if name not in self.histograms:
                self.histograms[name] = Histogram(
                    name=name,
                    help_text=help_text,
                )
            return self.histograms[name]

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create unique key for metric with labels."""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        # Export counters
        for counter in self.counters.values():
            lines.append(f"# HELP {counter.name} {counter.help_text}")
            lines.append(f"# TYPE {counter.name} counter")

            label_str = self._format_labels(counter.labels)
            lines.append(f"{counter.name}{label_str} {counter.value}")

        # Export gauges
        for gauge in self.gauges.values():
            lines.append(f"# HELP {gauge.name} {gauge.help_text}")
            lines.append(f"# TYPE {gauge.name} gauge")

            label_str = self._format_labels(gauge.labels)
            lines.append(f"{gauge.name}{label_str} {gauge.value}")

        # Export histograms
        for histogram in self.histograms.values():
            lines.append(f"# HELP {histogram.name} {histogram.help_text}")
            lines.append(f"# TYPE {histogram.name} histogram")

            # Buckets
            for bucket in histogram.buckets:
                count = histogram.bucket_counts.get(bucket, 0)
                lines.append(f'{histogram.name}_bucket{{le="{bucket}"}} {count}')

            # +Inf bucket
            lines.append(f'{histogram.name}_bucket{{le="+Inf"}} {histogram.count}')

            # Sum and count
            lines.append(f"{histogram.name}_sum {histogram.sum}")
            lines.append(f"{histogram.name}_count {histogram.count}")

        return "\n".join(lines) + "\n"

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus output."""
        if not labels:
            return ""

        label_pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(label_pairs) + "}"

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._metrics_lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self._init_standard_metrics()


# Global metrics collector instance
_collector = MetricsCollector()


def record_metric(name: str, value: float, labels: Optional[Dict[str, str]] = None, metric_type: str = "counter") -> None:
    """
    Convenience function to record a metric.

    Args:
        name: Metric name
        value: Metric value
        labels: Optional labels
        metric_type: Type of metric ("counter" or "gauge")
    """
    if metric_type == "counter":
        counter = _collector.counter(name, labels)
        counter.increment(value)
    elif metric_type == "gauge":
        gauge = _collector.gauge(name, labels)
        gauge.set(value)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def record_latency(name: str, duration_seconds: float) -> None:
    """
    Record latency in a histogram.

    Args:
        name: Metric name
        duration_seconds: Duration in seconds
    """
    histogram = _collector.histogram(f"{name}_seconds")
    histogram.observe(duration_seconds)


def time_operation(metric_name: str) -> Callable:
    """
    Decorator to time an operation and record latency.

    Args:
        metric_name: Name of the metric to record

    Example:
        @time_operation("tournament_execution")
        async def run_tournament():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                record_latency(metric_name, duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                record_latency(metric_name, duration)

        # Detect if function is async
        if hasattr(func, "__await__"):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Example usage
if __name__ == "__main__":
    collector = MetricsCollector()

    # Record some metrics
    validations = collector.counter("talair_validations_total")
    validations.increment()
    validations.increment()

    health = collector.gauge("talair_system_health")
    health.set(85.5)

    latency = collector.histogram("talair_validation_latency_seconds")
    latency.observe(1.234)
    latency.observe(0.567)
    latency.observe(2.345)

    # Export Prometheus format
    print(collector.export_prometheus())

    # Calculate stats
    print(f"\nHistogram stats:")
    print(f"  Mean: {latency.mean:.3f}s")
    print(f"  P50: {latency.quantile(0.50):.3f}s")
    print(f"  P95: {latency.quantile(0.95):.3f}s")
    print(f"  P99: {latency.quantile(0.99):.3f}s")
