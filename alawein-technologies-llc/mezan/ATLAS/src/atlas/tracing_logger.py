#!/usr/bin/env python3
"""
Distributed Tracing and Structured Logging for ORCHEX
Priority [059] implementation with OpenTelemetry-compatible tracing.
"""

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class LogLevel(Enum):
    """Log severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SpanStatus(Enum):
    """Span execution status."""

    OK = "OK"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


@dataclass
class TraceContext:
    """Distributed trace context."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, Any] = None

    def __post_init__(self):
        if self.baggage is None:
            self.baggage = {}


@dataclass
class Span:
    """Trace span representation."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    start_time: float
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    attributes: Dict[str, Any] = None
    events: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
        if self.events is None:
            self.events = []

    def duration_ms(self) -> Optional[float]:
        """Calculate span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None


class StructuredLogger:
    """Structured logging with trace correlation."""

    def __init__(self, name: str, output_path: Optional[str] = None):
        self.name = name
        self.output_path = output_path
        self.current_context: Optional[TraceContext] = None

        # Configure Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Add structured handler
        handler = logging.StreamHandler()
        handler.setFormatter(self._get_formatter())
        self.logger.addHandler(handler)

        if output_path:
            file_handler = logging.FileHandler(output_path)
            file_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(file_handler)

    def _get_formatter(self) -> logging.Formatter:
        """Get JSON formatter for structured logs."""
        return logging.Formatter("%(message)s")

    def _create_log_entry(self, level: LogLevel, message: str, **kwargs) -> Dict[str, Any]:
        """Create structured log entry."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level.value,
            "logger": self.name,
            "message": message,
            "attributes": kwargs,
        }

        # Add trace context if available
        if self.current_context:
            entry["trace"] = {
                "trace_id": self.current_context.trace_id,
                "span_id": self.current_context.span_id,
                "parent_span_id": self.current_context.parent_span_id,
            }

        return entry

    def log(self, level: LogLevel, message: str, **kwargs):
        """Log structured message."""
        entry = self._create_log_entry(level, message, **kwargs)
        self.logger.log(getattr(logging, level.value), json.dumps(entry))

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, message, **kwargs)


class DistributedTracer:
    """Distributed tracing system for ORCHEX."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.spans: List[Span] = []
        self.current_span: Optional[Span] = None
        self.logger = StructuredLogger(f"tracer.{service_name}")

    def create_trace_id(self) -> str:
        """Generate new trace ID."""
        return uuid.uuid4().hex

    def create_span_id(self) -> str:
        """Generate new span ID."""
        return uuid.uuid4().hex[:16]

    @contextmanager
    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Start a new trace span."""
        # Create or inherit trace ID
        if trace_id is None:
            if self.current_span:
                trace_id = self.current_span.trace_id
            else:
                trace_id = self.create_trace_id()

        # Set parent span
        if parent_span_id is None and self.current_span:
            parent_span_id = self.current_span.span_id

        # Create new span
        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=self.create_span_id(),
            parent_span_id=parent_span_id,
            start_time=time.time(),
            attributes=attributes or {},
        )

        # Add service name
        span.attributes["service.name"] = self.service_name

        # Set as current span
        previous_span = self.current_span
        self.current_span = span

        # Update logger context
        self.logger.current_context = TraceContext(
            trace_id=span.trace_id, span_id=span.span_id, parent_span_id=span.parent_span_id
        )

        # Log span start
        self.logger.info(
            f"Span started: {name}",
            span_id=span.span_id,
            trace_id=span.trace_id,
            parent_span_id=span.parent_span_id,
            attributes=span.attributes,
        )

        try:
            yield span
            span.status = SpanStatus.OK
        except Exception as e:
            span.status = SpanStatus.ERROR
            span.attributes["error"] = str(e)
            span.attributes["error.type"] = type(e).__name__
            self.logger.error(f"Span error: {name}", span_id=span.span_id, error=str(e))
            raise
        finally:
            # End span
            span.end_time = time.time()
            self.spans.append(span)

            # Log span end
            self.logger.info(
                f"Span ended: {name}",
                span_id=span.span_id,
                duration_ms=span.duration_ms(),
                status=span.status.value,
            )

            # Restore previous span
            self.current_span = previous_span
            if previous_span:
                self.logger.current_context = TraceContext(
                    trace_id=previous_span.trace_id,
                    span_id=previous_span.span_id,
                    parent_span_id=previous_span.parent_span_id,
                )
            else:
                self.logger.current_context = None

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to current span."""
        if self.current_span:
            event = {"name": name, "timestamp": time.time(), "attributes": attributes or {}}
            self.current_span.events.append(event)

            self.logger.info(
                f"Span event: {name}",
                span_id=self.current_span.span_id,
                event_attributes=attributes,
            )

    def set_attribute(self, key: str, value: Any):
        """Set attribute on current span."""
        if self.current_span:
            self.current_span.attributes[key] = value

    def export_traces(self) -> List[Dict[str, Any]]:
        """Export all traces in OpenTelemetry format."""
        traces = []
        for span in self.spans:
            trace_data = {
                "name": span.name,
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "parent_span_id": span.parent_span_id,
                "start_time_unix_nano": int(span.start_time * 1e9),
                "end_time_unix_nano": int(span.end_time * 1e9) if span.end_time else None,
                "status": {"code": span.status.value},
                "attributes": [
                    {"key": k, "value": {"string_value": str(v)}}
                    for k, v in span.attributes.items()
                ],
                "events": [
                    {
                        "time_unix_nano": int(e["timestamp"] * 1e9),
                        "name": e["name"],
                        "attributes": [
                            {"key": k, "value": {"string_value": str(v)}}
                            for k, v in e.get("attributes", {}).items()
                        ],
                    }
                    for e in span.events
                ],
            }
            traces.append(trace_data)

        return traces


class ATLASTracer:
    """ORCHEX-specific tracing with feature-aware spans."""

    def __init__(self):
        self.tracer = DistributedTracer("ORCHEX")
        self.feature_tracers = {}

    def get_feature_tracer(self, feature: str) -> DistributedTracer:
        """Get or create feature-specific tracer."""
        if feature not in self.feature_tracers:
            self.feature_tracers[feature] = DistributedTracer(f"ORCHEX.{feature}")
        return self.feature_tracers[feature]

    @contextmanager
    def trace_feature_execution(self, feature: str, input_data: Dict[str, Any]):
        """Trace complete feature execution."""
        tracer = self.get_feature_tracer(feature)

        with tracer.start_span(
            f"{feature}_execution",
            attributes={
                "feature": feature,
                "input.title": input_data.get("title"),
                "input.hypothesis": input_data.get("hypothesis")[:100] + "...",
                "parameters": json.dumps(input_data.get("parameters", {})),
            },
        ) as span:
            # Add input validation span
            with tracer.start_span("input_validation"):
                tracer.add_event("schema_check")
                tracer.add_event("pii_scan")
                tracer.add_event("safety_check")

            # Add main execution span
            with tracer.start_span("main_execution"):
                tracer.add_event("context_loading")
                tracer.add_event("processing")
                tracer.add_event("output_generation")
                yield span

            # Add output validation span
            with tracer.start_span("output_validation"):
                tracer.add_event("result_check")
                tracer.add_event("manifest_generation")

    def export_all_traces(self) -> Dict[str, List[Dict[str, Any]]]:
        """Export all feature traces."""
        all_traces = {"main": self.tracer.export_traces()}

        for feature, tracer in self.feature_tracers.items():
            all_traces[feature] = tracer.export_traces()

        return all_traces


# Example usage
if __name__ == "__main__":
    # Initialize ORCHEX tracer
    atlas_tracer = ATLASTracer()

    # Simulate feature execution with tracing
    test_input = {
        "title": "Test Hypothesis",
        "hypothesis": "Testing the tracing system for ORCHEX",
        "feature": "nightmare",
        "parameters": {"ensemble_size": 3},
    }

    with atlas_tracer.trace_feature_execution("nightmare", test_input) as span:
        # Simulate processing
        time.sleep(0.1)
        span.attributes["result"] = "success"
        span.attributes["score"] = 75.5

    # Export traces
    traces = atlas_tracer.export_all_traces()
    print("Exported traces:", json.dumps(traces, indent=2))
