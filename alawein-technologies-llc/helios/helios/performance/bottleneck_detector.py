"""
Bottleneck Detection

Identifies:
- Slow functions
- Memory leaks
- Resource contention
- Performance degradation trends
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# BOTTLENECK
# ============================================================================

class Bottleneck:
    """Represents a performance bottleneck."""

    def __init__(
        self,
        name: str,
        bottleneck_type: str,  # "slow", "memory", "cpu", "io"
        severity: str,  # "low", "medium", "high", "critical"
        metric_value: float,
        threshold: float,
        recommendations: List[str],
    ):
        self.name = name
        self.type = bottleneck_type
        self.severity = severity
        self.metric_value = metric_value
        self.threshold = threshold
        self.recommendations = recommendations
        self.detected_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "severity": self.severity,
            "metric_value": round(self.metric_value, 2),
            "threshold": round(self.threshold, 2),
            "exceeds_by_percent": round(
                ((self.metric_value - self.threshold) / self.threshold * 100), 1
            ),
            "recommendations": self.recommendations,
            "detected_at": self.detected_at.isoformat(),
        }


# ============================================================================
# BOTTLENECK DETECTOR
# ============================================================================

class BottleneckDetector:
    """Detect performance bottlenecks."""

    # Thresholds
    SLOW_FUNCTION_MS = 1000  # Functions taking >1s
    SLOW_API_MS = 5000  # API endpoints taking >5s
    MEMORY_THRESHOLD_MB = 500  # >500MB per function
    CPU_THRESHOLD_PERCENT = 80  # >80% CPU usage

    def __init__(self):
        self.bottlenecks: List[Bottleneck] = []
        self.history: Dict[str, List[Dict[str, Any]]] = {}

    def detect_from_metrics(self, metrics: Dict[str, Dict[str, Any]]) -> List[Bottleneck]:
        """
        Detect bottlenecks from profiler metrics.

        Args:
            metrics: Metrics from DynamicProfiler.get_metrics()

        Returns:
            List of detected bottlenecks
        """
        self.bottlenecks.clear()

        for func_name, stats in metrics.items():
            if not stats:
                continue

            # Check execution time
            avg_time_ms = stats.get("average_time_ms", 0)
            if avg_time_ms > self.SLOW_FUNCTION_MS:
                severity = "critical" if avg_time_ms > 5000 else "high"
                self.bottlenecks.append(Bottleneck(
                    name=func_name,
                    bottleneck_type="slow",
                    severity=severity,
                    metric_value=avg_time_ms,
                    threshold=self.SLOW_FUNCTION_MS,
                    recommendations=[
                        f"Profile {func_name} to identify slow operations",
                        "Consider caching results or using memoization",
                        "Look for inefficient algorithms or I/O operations",
                        "Consider async/parallel processing",
                    ]
                ))

            # Check memory usage
            peak_memory = stats.get("peak_memory_mb", 0)
            if peak_memory > self.MEMORY_THRESHOLD_MB:
                severity = "high" if peak_memory > 1000 else "medium"
                self.bottlenecks.append(Bottleneck(
                    name=func_name,
                    bottleneck_type="memory",
                    severity=severity,
                    metric_value=peak_memory,
                    threshold=self.MEMORY_THRESHOLD_MB,
                    recommendations=[
                        f"Optimize {func_name} memory usage",
                        "Check for memory leaks or large data structures",
                        "Consider streaming or chunking large datasets",
                        "Profile memory allocations with tracemalloc",
                    ]
                ))

        return self.bottlenecks

    def detect_degradation(
        self,
        func_name: str,
        current_time_ms: float,
        historical_data: List[float],
    ) -> Optional[Bottleneck]:
        """
        Detect performance degradation trend.

        Args:
            func_name: Function name
            current_time_ms: Current execution time
            historical_data: Previous execution times

        Returns:
            Bottleneck if degradation detected
        """
        if len(historical_data) < 3:
            return None

        # Calculate trend
        recent_avg = sum(historical_data[-3:]) / 3
        older_avg = sum(historical_data[:-3]) / max(1, len(historical_data) - 3)

        # Check if degradation > 20%
        degradation_percent = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0

        if degradation_percent > 20:
            return Bottleneck(
                name=func_name,
                bottleneck_type="degradation",
                severity="medium",
                metric_value=recent_avg,
                threshold=older_avg,
                recommendations=[
                    f"Performance of {func_name} has degraded by {degradation_percent:.1f}%",
                    "Check recent code changes",
                    "Monitor data volume and complexity",
                    "Review resource allocation",
                ]
            )

        return None

    def get_critical_bottlenecks(self) -> List[Bottleneck]:
        """Get only critical severity bottlenecks."""
        return [b for b in self.bottlenecks if b.severity == "critical"]

    def get_bottlenecks_by_type(self, bottleneck_type: str) -> List[Bottleneck]:
        """Get bottlenecks by type."""
        return [b for b in self.bottlenecks if b.type == bottleneck_type]

    def get_summary(self) -> Dict[str, Any]:
        """Get bottleneck summary."""
        severities = {
            "critical": len([b for b in self.bottlenecks if b.severity == "critical"]),
            "high": len([b for b in self.bottlenecks if b.severity == "high"]),
            "medium": len([b for b in self.bottlenecks if b.severity == "medium"]),
            "low": len([b for b in self.bottlenecks if b.severity == "low"]),
        }

        types = {}
        for bottleneck in self.bottlenecks:
            types[bottleneck.type] = types.get(bottleneck.type, 0) + 1

        return {
            "total_bottlenecks": len(self.bottlenecks),
            "by_severity": severities,
            "by_type": types,
            "critical_bottlenecks": [b.to_dict() for b in self.get_critical_bottlenecks()],
        }


# ============================================================================
# GLOBAL DETECTOR
# ============================================================================

_global_detector: Optional[BottleneckDetector] = None


def detect_bottlenecks(metrics: Dict[str, Dict[str, Any]]) -> List[Bottleneck]:
    """Detect bottlenecks from metrics."""
    global _global_detector
    if _global_detector is None:
        _global_detector = BottleneckDetector()

    return _global_detector.detect_from_metrics(metrics)
