"""
ORCHEX Event Bus Module

Provides event-driven architecture with pub/sub system,
event handlers, async processing, replay capability, and dead letter queue.

Features:
- Event-driven pub/sub system with topics
- Strong typing for events
- Event handler registry with priority
- Async event processing with thread pools
- Event replay and audit log
- Dead letter queue for failed events
- Event filtering and routing
- Metrics and monitoring

Author: MEZAN Research Team
Version: 1.0.0
"""

import json
import logging
import time
import threading
import uuid
from typing import Any, Dict, List, Optional, Callable, Type, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict, deque
import traceback
import inspect

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standard event types in ORCHEX"""
    # Workflow events
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_PAUSED = "workflow.paused"
    WORKFLOW_RESUMED = "workflow.resumed"

    # Task events
    TASK_CREATED = "task.created"
    TASK_ASSIGNED = "task.assigned"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_RETRIED = "task.retried"

    # Agent events
    AGENT_REGISTERED = "agent.registered"
    AGENT_UNREGISTERED = "agent.unregistered"
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_HEARTBEAT = "agent.heartbeat"

    # System events
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"

    # Resource events
    RESOURCE_ALLOCATED = "resource.allocated"
    RESOURCE_RELEASED = "resource.released"
    RESOURCE_EXHAUSTED = "resource.exhausted"

    # Optimization events
    OPTIMIZATION_STARTED = "optimization.started"
    OPTIMIZATION_COMPLETED = "optimization.completed"
    OPTIMIZATION_IMPROVED = "optimization.improved"

    # Custom events
    CUSTOM = "custom"


@dataclass
class Event:
    """Base event class with metadata"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.CUSTOM
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else self.event_type,
            "timestamp": self.timestamp,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "payload": self.payload,
            "metadata": self.metadata,
            "version": self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary"""
        event_type = data.get("event_type", "custom")

        # Convert string to EventType if needed
        if isinstance(event_type, str):
            try:
                event_type = EventType(event_type)
            except ValueError:
                event_type = EventType.CUSTOM

        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=event_type,
            timestamp=data.get("timestamp", time.time()),
            source=data.get("source", ""),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            version=data.get("version", "1.0.0")
        )


# Specific event classes for strong typing
@dataclass
class WorkflowStartedEvent(Event):
    """Event fired when workflow starts"""
    def __init__(self, workflow_id: str, workflow_type: str, **kwargs):
        super().__init__(
            event_type=EventType.WORKFLOW_STARTED,
            payload={
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                **kwargs
            }
        )


@dataclass
class TaskCompletedEvent(Event):
    """Event fired when task completes"""
    def __init__(self, task_id: str, result: Any, duration: float, **kwargs):
        super().__init__(
            event_type=EventType.TASK_COMPLETED,
            payload={
                "task_id": task_id,
                "result": result,
                "duration": duration,
                **kwargs
            }
        )


@dataclass
class AgentHeartbeatEvent(Event):
    """Event fired for agent heartbeat"""
    def __init__(self, agent_id: str, status: str, metrics: Dict[str, Any], **kwargs):
        super().__init__(
            event_type=EventType.AGENT_HEARTBEAT,
            payload={
                "agent_id": agent_id,
                "status": status,
                "metrics": metrics,
                **kwargs
            }
        )


class EventHandler:
    """Wrapper for event handler functions"""

    def __init__(
        self,
        handler: Callable,
        event_types: List[EventType],
        priority: int = 0,
        filter_func: Optional[Callable[[Event], bool]] = None,
        async_execution: bool = False
    ):
        """
        Initialize event handler

        Args:
            handler: Callable to handle events
            event_types: List of event types to handle
            priority: Handler priority (higher = earlier execution)
            filter_func: Optional filter function
            async_execution: Whether to execute asynchronously
        """
        self.handler = handler
        self.event_types = event_types
        self.priority = priority
        self.filter_func = filter_func
        self.async_execution = async_execution
        self.execution_count = 0
        self.error_count = 0
        self.last_execution = None
        self.last_error = None

    def can_handle(self, event: Event) -> bool:
        """Check if handler can process event"""
        # Check event type
        if event.event_type not in self.event_types and EventType.CUSTOM not in self.event_types:
            return False

        # Apply filter if present
        if self.filter_func:
            try:
                return self.filter_func(event)
            except Exception as e:
                logger.error(f"Filter function error: {e}")
                return False

        return True

    def execute(self, event: Event) -> Any:
        """Execute handler with event"""
        try:
            self.execution_count += 1
            self.last_execution = time.time()

            # Check handler signature
            sig = inspect.signature(self.handler)

            if len(sig.parameters) == 1:
                result = self.handler(event)
            else:
                # Legacy support for handlers without arguments
                result = self.handler()

            return result

        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Handler execution error: {e}\n{traceback.format_exc()}")
            raise


class EventBus:
    """
    Central event bus for publish/subscribe pattern

    Features:
    - Topic-based routing
    - Handler priority
    - Async processing
    - Event replay
    - Dead letter queue
    - Metrics collection
    """

    def __init__(
        self,
        max_workers: int = 10,
        event_history_size: int = 1000,
        enable_replay: bool = True
    ):
        """Initialize event bus"""
        self.handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self.event_history: deque = deque(maxlen=event_history_size)
        self.dead_letter_queue: deque = deque(maxlen=100)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.enable_replay = enable_replay
        self.running = True

        # Metrics
        self.metrics = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "handlers_registered": 0,
            "average_processing_time": 0
        }

        # Event interceptors
        self.before_publish: List[Callable] = []
        self.after_publish: List[Callable] = []

        logger.info(f"Event bus initialized with {max_workers} workers")

    def register_handler(
        self,
        handler: Callable,
        event_types: List[EventType],
        priority: int = 0,
        filter_func: Optional[Callable[[Event], bool]] = None,
        async_execution: bool = False
    ) -> EventHandler:
        """
        Register event handler

        Args:
            handler: Function to handle events
            event_types: Event types to subscribe to
            priority: Handler priority
            filter_func: Optional event filter
            async_execution: Execute asynchronously

        Returns:
            EventHandler instance
        """
        event_handler = EventHandler(
            handler=handler,
            event_types=event_types,
            priority=priority,
            filter_func=filter_func,
            async_execution=async_execution
        )

        for event_type in event_types:
            self.handlers[event_type].append(event_handler)
            # Sort by priority (descending)
            self.handlers[event_type].sort(key=lambda h: h.priority, reverse=True)

        self.metrics["handlers_registered"] += 1

        logger.debug(f"Registered handler for {[t.value for t in event_types]} with priority {priority}")

        return event_handler

    def unregister_handler(self, handler: EventHandler):
        """Unregister event handler"""
        for event_type in handler.event_types:
            if handler in self.handlers[event_type]:
                self.handlers[event_type].remove(handler)
                self.metrics["handlers_registered"] -= 1

    def publish(self, event: Event) -> Future:
        """
        Publish event to bus

        Args:
            event: Event to publish

        Returns:
            Future for async processing result
        """
        start_time = time.time()

        # Run before-publish interceptors
        for interceptor in self.before_publish:
            try:
                interceptor(event)
            except Exception as e:
                logger.error(f"Before-publish interceptor error: {e}")

        # Store in history if replay enabled
        if self.enable_replay:
            self.event_history.append(event)

        # Update metrics
        self.metrics["events_published"] += 1

        # Submit for processing
        future = self.executor.submit(self._process_event, event)

        # Schedule after-publish interceptors
        def run_after_publish(f):
            try:
                for interceptor in self.after_publish:
                    interceptor(event)
            except Exception as e:
                logger.error(f"After-publish interceptor error: {e}")

        future.add_done_callback(run_after_publish)

        # Update processing time
        processing_time = time.time() - start_time
        self.metrics["average_processing_time"] = (
            self.metrics["average_processing_time"] * 0.9 + processing_time * 0.1
        )

        logger.debug(f"Published event {event.event_id} of type {event.event_type.value}")

        return future

    def publish_batch(self, events: List[Event]) -> List[Future]:
        """Publish multiple events"""
        futures = []

        for event in events:
            futures.append(self.publish(event))

        return futures

    def _process_event(self, event: Event):
        """Process single event with all handlers"""
        results = []
        errors = []

        # Get handlers for event type
        handlers = self.handlers.get(event.event_type, [])

        # Add handlers subscribed to CUSTOM events
        if event.event_type != EventType.CUSTOM:
            handlers.extend(self.handlers.get(EventType.CUSTOM, []))

        # Filter and sort handlers
        applicable_handlers = [h for h in handlers if h.can_handle(event)]

        for handler in applicable_handlers:
            try:
                if handler.async_execution:
                    # Submit to executor for async processing
                    future = self.executor.submit(handler.execute, event)
                    results.append(future)
                else:
                    # Execute synchronously
                    result = handler.execute(event)
                    results.append(result)

                logger.debug(f"Handler {handler.handler.__name__} processed event {event.event_id}")

            except Exception as e:
                error_msg = f"Handler {handler.handler.__name__} failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # If all handlers failed, add to DLQ
        if errors and not results:
            self._add_to_dlq(event, errors)
            self.metrics["events_failed"] += 1
        else:
            self.metrics["events_processed"] += 1

        return results

    def _add_to_dlq(self, event: Event, errors: List[str]):
        """Add failed event to dead letter queue"""
        dlq_entry = {
            "event": event.to_dict(),
            "errors": errors,
            "timestamp": time.time()
        }

        self.dead_letter_queue.append(dlq_entry)

        logger.warning(f"Event {event.event_id} added to DLQ with {len(errors)} errors")

    def replay_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_types: Optional[List[EventType]] = None,
        correlation_id: Optional[str] = None
    ) -> List[Event]:
        """
        Replay historical events

        Args:
            start_time: Start timestamp for replay
            end_time: End timestamp for replay
            event_types: Filter by event types
            correlation_id: Filter by correlation ID

        Returns:
            List of replayed events
        """
        if not self.enable_replay:
            logger.warning("Event replay is disabled")
            return []

        replayed = []

        for event in self.event_history:
            # Apply filters
            if start_time and event.timestamp < start_time:
                continue

            if end_time and event.timestamp > end_time:
                continue

            if event_types and event.event_type not in event_types:
                continue

            if correlation_id and event.correlation_id != correlation_id:
                continue

            # Replay event
            logger.info(f"Replaying event {event.event_id}")
            self.publish(event)
            replayed.append(event)

        logger.info(f"Replayed {len(replayed)} events")

        return replayed

    def process_dlq(self, max_retries: int = 1) -> int:
        """
        Process dead letter queue

        Args:
            max_retries: Maximum retry attempts

        Returns:
            Number of successfully processed events
        """
        processed = 0
        dlq_copy = list(self.dead_letter_queue)
        self.dead_letter_queue.clear()

        for entry in dlq_copy:
            event = Event.from_dict(entry["event"])

            retry_count = entry.get("retry_count", 0)

            if retry_count >= max_retries:
                # Re-add to DLQ
                entry["retry_count"] = retry_count + 1
                self.dead_letter_queue.append(entry)
                continue

            # Retry processing
            try:
                self._process_event(event)
                processed += 1
                logger.info(f"Successfully processed DLQ event {event.event_id}")
            except Exception as e:
                # Update retry count and re-add
                entry["retry_count"] = retry_count + 1
                entry["errors"].append(str(e))
                self.dead_letter_queue.append(entry)

        logger.info(f"Processed {processed} events from DLQ")

        return processed

    def wait_for_event(
        self,
        event_type: EventType,
        timeout: float = 30,
        filter_func: Optional[Callable[[Event], bool]] = None
    ) -> Optional[Event]:
        """
        Wait for specific event

        Args:
            event_type: Event type to wait for
            timeout: Maximum wait time
            filter_func: Optional filter

        Returns:
            Event if received, None if timeout
        """
        received_event = None
        event_received = threading.Event()

        def handler(event: Event):
            nonlocal received_event
            if not filter_func or filter_func(event):
                received_event = event
                event_received.set()

        # Register temporary handler
        temp_handler = self.register_handler(
            handler=handler,
            event_types=[event_type],
            priority=1000  # High priority
        )

        # Wait for event
        event_received.wait(timeout)

        # Unregister handler
        self.unregister_handler(temp_handler)

        return received_event

    def add_interceptor(
        self,
        interceptor: Callable,
        before: bool = True
    ):
        """
        Add event interceptor

        Args:
            interceptor: Function to intercept events
            before: True for before-publish, False for after-publish
        """
        if before:
            self.before_publish.append(interceptor)
        else:
            self.after_publish.append(interceptor)

    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics"""
        metrics = self.metrics.copy()

        # Add handler metrics
        handler_metrics = {}
        for event_type, handlers in self.handlers.items():
            handler_metrics[event_type.value] = [
                {
                    "handler": h.handler.__name__,
                    "executions": h.execution_count,
                    "errors": h.error_count,
                    "last_execution": h.last_execution
                }
                for h in handlers
            ]

        metrics["handlers"] = handler_metrics
        metrics["event_history_size"] = len(self.event_history)
        metrics["dlq_size"] = len(self.dead_letter_queue)

        return metrics

    def clear_history(self):
        """Clear event history"""
        self.event_history.clear()
        logger.info("Event history cleared")

    def shutdown(self):
        """Shutdown event bus"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Event bus shutdown complete")


class EventBusWithPersistence(EventBus):
    """
    Event bus with persistent storage

    Extends EventBus with:
    - Event persistence to disk
    - Event sourcing support
    - Audit logging
    """

    def __init__(
        self,
        storage_path: str = "/tmp/atlas_events",
        **kwargs
    ):
        """Initialize persistent event bus"""
        super().__init__(**kwargs)
        self.storage_path = storage_path
        self.event_log = []

        # Ensure storage directory exists
        import os
        os.makedirs(storage_path, exist_ok=True)

    def publish(self, event: Event) -> Future:
        """Publish event with persistence"""
        # Persist event before processing
        self._persist_event(event)

        # Continue with normal processing
        return super().publish(event)

    def _persist_event(self, event: Event):
        """Persist event to disk"""
        import os

        try:
            # Create filename based on date
            date_str = datetime.fromtimestamp(event.timestamp).strftime("%Y%m%d")
            filename = os.path.join(self.storage_path, f"events_{date_str}.jsonl")

            # Append to JSONL file
            with open(filename, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")

            # Add to event log
            self.event_log.append(event)

            logger.debug(f"Persisted event {event.event_id} to {filename}")

        except Exception as e:
            logger.error(f"Failed to persist event: {e}")

    def load_events(
        self,
        date: Optional[datetime] = None,
        days_back: int = 7
    ) -> List[Event]:
        """Load historical events from disk"""
        import os
        from datetime import timedelta

        events = []

        if not date:
            date = datetime.now()

        for i in range(days_back):
            check_date = date - timedelta(days=i)
            date_str = check_date.strftime("%Y%m%d")
            filename = os.path.join(self.storage_path, f"events_{date_str}.jsonl")

            if not os.path.exists(filename):
                continue

            try:
                with open(filename, "r") as f:
                    for line in f:
                        event_data = json.loads(line)
                        events.append(Event.from_dict(event_data))

                logger.info(f"Loaded {len(events)} events from {filename}")

            except Exception as e:
                logger.error(f"Failed to load events from {filename}: {e}")

        return events

    def rebuild_state(self, events: List[Event]) -> Dict[str, Any]:
        """
        Rebuild application state from events

        Args:
            events: List of events to replay

        Returns:
            Reconstructed state
        """
        state = {}

        for event in sorted(events, key=lambda e: e.timestamp):
            # Apply event to state based on type
            if event.event_type == EventType.WORKFLOW_STARTED:
                workflow_id = event.payload.get("workflow_id")
                state[f"workflow_{workflow_id}"] = "started"

            elif event.event_type == EventType.WORKFLOW_COMPLETED:
                workflow_id = event.payload.get("workflow_id")
                state[f"workflow_{workflow_id}"] = "completed"

            elif event.event_type == EventType.TASK_COMPLETED:
                task_id = event.payload.get("task_id")
                state[f"task_{task_id}"] = event.payload.get("result")

            # Add more state transitions as needed

        logger.info(f"Rebuilt state from {len(events)} events")

        return state


# Convenience function for global event bus
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create global event bus instance"""
    global _global_event_bus

    if _global_event_bus is None:
        _global_event_bus = EventBusWithPersistence()

    return _global_event_bus


def publish_event(event: Event):
    """Publish event to global bus"""
    get_event_bus().publish(event)


def subscribe(
    event_types: List[EventType],
    handler: Callable,
    priority: int = 0
):
    """Subscribe to events on global bus"""
    get_event_bus().register_handler(
        handler=handler,
        event_types=event_types,
        priority=priority
    )