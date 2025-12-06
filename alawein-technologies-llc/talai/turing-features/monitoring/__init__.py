"""
TalAI Monitoring & Observability System

Provides comprehensive logging, metrics, and tracing for all Turing Challenge components.
"""

from monitoring.logger import get_logger, setup_logging
from monitoring.metrics import MetricsCollector, record_metric, record_latency
from monitoring.tracer import trace_operation, get_tracer

__all__ = [
    "get_logger",
    "setup_logging",
    "MetricsCollector",
    "record_metric",
    "record_latency",
    "trace_operation",
    "get_tracer",
]

__version__ = "1.0.0"
