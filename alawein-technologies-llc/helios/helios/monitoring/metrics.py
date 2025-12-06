"""
Prometheus Metrics Collection

Tracks:
- API request metrics (count, latency, status codes)
- Algorithm execution metrics (runtime, quality, speedup)
- Validation metrics (count, quality scores, completion time)
- System metrics (memory, CPU, uptime)
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Registry,
    CollectorRegistry,
    REGISTRY,
    generate_latest,
    start_http_server,
)
import psutil
import time
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# METRICS REGISTRY
# ============================================================================

metrics_registry = CollectorRegistry()


# ============================================================================
# API METRICS
# ============================================================================

class APIMetrics:
    """Track API request metrics."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY

        # Request counters
        self.request_count = Counter(
            'helios_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.request_latency = Histogram(
            'helios_api_request_duration_seconds',
            'API request latency',
            ['method', 'endpoint'],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry
        )

        self.request_size = Histogram(
            'helios_api_request_size_bytes',
            'API request size',
            ['method', 'endpoint'],
            registry=self.registry
        )

        self.response_size = Histogram(
            'helios_api_response_size_bytes',
            'API response size',
            ['method', 'endpoint'],
            registry=self.registry
        )

    def record_request(self, method: str, endpoint: str, status: int, duration: float,
                      request_size: int = 0, response_size: int = 0):
        """Record API request metrics."""
        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_latency.labels(method=method, endpoint=endpoint).observe(duration)
        if request_size > 0:
            self.request_size.labels(method=method, endpoint=endpoint).observe(request_size)
        if response_size > 0:
            self.response_size.labels(method=method, endpoint=endpoint).observe(response_size)


# ============================================================================
# ALGORITHM METRICS
# ============================================================================

class AlgorithmMetrics:
    """Track algorithm execution metrics."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY

        # Execution metrics
        self.executions_total = Counter(
            'helios_algorithm_executions_total',
            'Total algorithm executions',
            ['algorithm', 'domain', 'status'],
            registry=self.registry
        )

        self.execution_duration = Histogram(
            'helios_algorithm_execution_duration_seconds',
            'Algorithm execution time',
            ['algorithm', 'domain'],
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry
        )

        self.quality_score = Histogram(
            'helios_algorithm_quality_score',
            'Algorithm quality scores',
            ['algorithm', 'domain'],
            buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            registry=self.registry
        )

        self.speedup_factor = Histogram(
            'helios_algorithm_speedup_factor',
            'Algorithm speedup vs baseline',
            ['algorithm', 'domain'],
            buckets=(1.0, 1.5, 2.0, 2.5, 3.0, 5.0),
            registry=self.registry
        )

    def record_execution(self, algorithm: str, domain: str, duration: float,
                        quality: float, speedup: float, status: str = "success"):
        """Record algorithm execution metrics."""
        self.executions_total.labels(algorithm=algorithm, domain=domain, status=status).inc()
        self.execution_duration.labels(algorithm=algorithm, domain=domain).observe(duration)
        self.quality_score.labels(algorithm=algorithm, domain=domain).observe(quality)
        self.speedup_factor.labels(algorithm=algorithm, domain=domain).observe(speedup)


# ============================================================================
# VALIDATION METRICS
# ============================================================================

class ValidationMetrics:
    """Track hypothesis validation metrics."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY

        # Validation counters
        self.validations_total = Counter(
            'helios_validations_total',
            'Total hypotheses validated',
            ['domain', 'result'],
            registry=self.registry
        )

        self.validation_duration = Histogram(
            'helios_validation_duration_seconds',
            'Validation execution time',
            ['domain'],
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0),
            registry=self.registry
        )

        self.validation_score = Histogram(
            'helios_validation_score',
            'Hypothesis validation scores',
            ['domain'],
            buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            registry=self.registry
        )

    def record_validation(self, domain: str, duration: float, score: float,
                         result: str = "valid"):
        """Record validation metrics."""
        self.validations_total.labels(domain=domain, result=result).inc()
        self.validation_duration.labels(domain=domain).observe(duration)
        self.validation_score.labels(domain=domain).observe(score)


# ============================================================================
# SYSTEM METRICS
# ============================================================================

class SystemMetrics:
    """Track system resource metrics."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        self.process = psutil.Process()
        self.start_time = time.time()

        # System gauges
        self.cpu_usage = Gauge(
            'helios_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )

        self.memory_usage = Gauge(
            'helios_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )

        self.memory_available = Gauge(
            'helios_memory_available_bytes',
            'Available memory in bytes',
            registry=self.registry
        )

        self.uptime_seconds = Gauge(
            'helios_uptime_seconds',
            'Application uptime in seconds',
            registry=self.registry
        )

        self.open_files = Gauge(
            'helios_open_files_count',
            'Number of open files',
            registry=self.registry
        )

        self.threads = Gauge(
            'helios_thread_count',
            'Number of threads',
            registry=self.registry
        )

    def update_metrics(self):
        """Update system metrics."""
        try:
            # CPU and memory
            self.cpu_usage.set(self.process.cpu_percent(interval=0.1))
            memory_info = self.process.memory_info()
            self.memory_usage.set(memory_info.rss)

            # Available memory
            available = psutil.virtual_memory().available
            self.memory_available.set(available)

            # Uptime
            uptime = time.time() - self.start_time
            self.uptime_seconds.set(uptime)

            # Open files and threads
            self.open_files.set(self.process.num_fds() if hasattr(self.process, 'num_fds') else 0)
            self.threads.set(self.process.num_threads())

        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")


# ============================================================================
# METRICS COLLECTOR
# ============================================================================

class MetricsCollector:
    """Unified metrics collector."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        self.api_metrics = APIMetrics(self.registry)
        self.algorithm_metrics = AlgorithmMetrics(self.registry)
        self.validation_metrics = ValidationMetrics(self.registry)
        self.system_metrics = SystemMetrics(self.registry)

    def get_metrics_string(self) -> str:
        """Get Prometheus metrics in text format."""
        self.system_metrics.update_metrics()
        return generate_latest(self.registry).decode('utf-8')

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary."""
        self.system_metrics.update_metrics()
        return {
            "api_metrics": "enabled",
            "algorithm_metrics": "enabled",
            "validation_metrics": "enabled",
            "system_metrics": "enabled",
            "timestamp": datetime.utcnow().isoformat(),
        }


# ============================================================================
# GLOBAL METRICS INSTANCE
# ============================================================================

_global_collector: Optional[MetricsCollector] = None


def initialize_metrics(registry: Optional[CollectorRegistry] = None) -> MetricsCollector:
    """Initialize global metrics collector."""
    global _global_collector
    _global_collector = MetricsCollector(registry)
    return _global_collector


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def get_metrics_registry() -> CollectorRegistry:
    """Get Prometheus registry."""
    return REGISTRY


def start_metrics_server(port: int = 8001):
    """Start Prometheus metrics HTTP server."""
    try:
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
