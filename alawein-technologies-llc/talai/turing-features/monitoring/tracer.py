"""
Distributed Tracing for TalAI Turing Challenge System.

Simple tracing implementation for debugging and performance analysis.
"""

import time
import uuid
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from contextlib import contextmanager


@dataclass
class Span:
    """Represents a single span in a trace."""

    trace_id: str
    span_id: str
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    parent_span_id: Optional[str] = None


class Tracer:
    """Simple tracer for distributed tracing."""

    def __init__(self):
        self.spans: Dict[str, Span] = {}
        self.active_spans: Dict[str, str] = {}  # thread_id -> span_id

    @contextmanager
    def start_span(
        self,
        operation_name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ):
        """Start a new span."""
        span_id = str(uuid.uuid4())[:8]
        trace_id = trace_id or str(uuid.uuid4())[:16]

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            operation_name=operation_name,
            start_time=datetime.utcnow(),
            tags=tags or {},
            parent_span_id=parent_span_id,
        )

        self.spans[span_id] = span

        try:
            yield span
        finally:
            span.end_time = datetime.utcnow()
            span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000


# Global tracer instance
_tracer = Tracer()


def get_tracer() -> Tracer:
    """Get global tracer instance."""
    return _tracer


def trace_operation(operation_name: str, tags: Optional[Dict[str, Any]] = None) -> Callable:
    """
    Decorator to trace an operation.

    Args:
        operation_name: Name of the operation
        tags: Optional tags to attach to span

    Example:
        @trace_operation("run_tournament", tags={"format": "elimination"})
        async def run_tournament():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with _tracer.start_span(operation_name, tags=tags):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with _tracer.start_span(operation_name, tags=tags):
                return func(*args, **kwargs)

        if hasattr(func, "__await__"):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Example usage
if __name__ == "__main__":
    tracer = get_tracer()

    # Create a trace
    with tracer.start_span("hypothesis_validation") as root_span:
        print(f"Started validation trace: {root_span.trace_id}")

        with tracer.start_span(
            "run_interrogation",
            trace_id=root_span.trace_id,
            parent_span_id=root_span.span_id,
            tags={"questions": 200},
        ) as span:
            time.sleep(0.1)  # Simulate work

        with tracer.start_span(
            "run_tournament",
            trace_id=root_span.trace_id,
            parent_span_id=root_span.span_id,
            tags={"agents": 8},
        ) as span:
            time.sleep(0.2)  # Simulate work

    # Print spans
    for span in tracer.spans.values():
        print(f"\n{span.operation_name}:")
        print(f"  Span ID: {span.span_id}")
        print(f"  Duration: {span.duration_ms:.2f}ms")
        if span.tags:
            print(f"  Tags: {span.tags}")
