"""
Event Bus Integration Tests for ORCHEX

Tests event-driven architecture, pub/sub, and event streaming.
"""

import pytest
import asyncio
import json
import time
import uuid
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from datetime import datetime

from atlas_core.event_bus import (
    EventBus,
    EventStore,
    EventPublisher,
    EventSubscriber,
    EventProcessor,
    EventSchema,
    EventRouter,
    EventReplay,
)
from atlas_core.engine import ATLASEngine


@dataclass
class WorkflowEvent:
    """Base workflow event structure."""

    event_id: str
    event_type: str
    timestamp: datetime
    workflow_id: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any]


class TestEventBusIntegration:
    """Test event-driven architecture integration."""

    @pytest.fixture
    def event_bus(self):
        """Create event bus instance."""
        return EventBus(
            brokers=["kafka", "rabbitmq", "redis"],
            persistence=True,
            replay_enabled=True,
            ordering_guarantee="total",
        )

    @pytest.fixture
    def event_store(self):
        """Create event store for event sourcing."""
        return EventStore(
            backend="postgresql",
            retention_days=30,
            snapshot_interval=100,
            compression=True,
        )

    @pytest.mark.integration
    async def test_event_publishing_and_subscription(self, event_bus):
        """Test basic pub/sub functionality."""
        received_events = []

        # Create subscriber
        async def event_handler(event):
            received_events.append(event)

        subscriber = await event_bus.subscribe(
            topics=["workflow.started", "workflow.completed"], handler=event_handler
        )

        # Publish events
        events = [
            {
                "type": "workflow.started",
                "workflow_id": "test_123",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "type": "workflow.completed",
                "workflow_id": "test_123",
                "duration": 45.2,
                "timestamp": datetime.now().isoformat(),
            },
        ]

        for event in events:
            await event_bus.publish(event["type"], event)

        # Allow processing
        await asyncio.sleep(0.5)

        # Verify events received
        assert len(received_events) == 2
        assert received_events[0]["type"] == "workflow.started"
        assert received_events[1]["type"] == "workflow.completed"

        # Unsubscribe
        await subscriber.unsubscribe()

    @pytest.mark.integration
    async def test_event_routing_rules(self, event_bus):
        """Test complex event routing based on rules."""
        router = EventRouter(event_bus)

        # Define routing rules
        rules = [
            {
                "name": "critical_errors",
                "condition": lambda e: e.get("severity") == "critical",
                "routes": ["alerts.critical", "logging.errors"],
            },
            {
                "name": "performance_degradation",
                "condition": lambda e: e.get("latency", 0) > 5.0,
                "routes": ["monitoring.performance", "alerts.warning"],
            },
            {
                "name": "workflow_completion",
                "condition": lambda e: e.get("type") == "workflow.completed",
                "routes": ["analytics.completions", "notifications.success"],
            },
        ]

        await router.register_rules(rules)

        # Track routed events
        routed_events = {route: [] for rule in rules for route in rule["routes"]}

        # Subscribe to routes
        for route in routed_events.keys():

            async def make_handler(r):
                async def handler(event):
                    routed_events[r].append(event)

                return handler

            await event_bus.subscribe([route], await make_handler(route))

        # Publish events that match different rules
        test_events = [
            {"type": "error", "severity": "critical", "message": "Database down"},
            {"type": "api.request", "latency": 7.5, "endpoint": "/api/process"},
            {"type": "workflow.completed", "workflow_id": "abc", "duration": 30},
        ]

        for event in test_events:
            await router.route_event(event)

        await asyncio.sleep(0.5)

        # Verify routing
        assert len(routed_events["alerts.critical"]) == 1
        assert len(routed_events["monitoring.performance"]) == 1
        assert len(routed_events["analytics.completions"]) == 1

    @pytest.mark.integration
    async def test_event_sourcing_and_replay(self, event_store):
        """Test event sourcing pattern with replay capability."""
        # Create aggregate
        workflow_id = str(uuid.uuid4())

        # Store events
        events = [
            {
                "aggregate_id": workflow_id,
                "type": "WorkflowCreated",
                "version": 1,
                "data": {"name": "Research", "created_by": "user1"},
            },
            {
                "aggregate_id": workflow_id,
                "type": "TaskAdded",
                "version": 2,
                "data": {"task_id": "task1", "name": "Literature Review"},
            },
            {
                "aggregate_id": workflow_id,
                "type": "TaskCompleted",
                "version": 3,
                "data": {"task_id": "task1", "result": "success"},
            },
            {
                "aggregate_id": workflow_id,
                "type": "WorkflowCompleted",
                "version": 4,
                "data": {"duration": 120, "status": "success"},
            },
        ]

        for event in events:
            await event_store.append(event)

        # Replay events to rebuild state
        replayed_events = await event_store.replay(aggregate_id=workflow_id, from_version=1)

        assert len(replayed_events) == 4
        assert replayed_events[0]["type"] == "WorkflowCreated"
        assert replayed_events[-1]["type"] == "WorkflowCompleted"

        # Test partial replay
        partial_replay = await event_store.replay(
            aggregate_id=workflow_id, from_version=2, to_version=3
        )
        assert len(partial_replay) == 2

    @pytest.mark.integration
    async def test_event_streaming_and_processing(self, event_bus):
        """Test event stream processing with windowing and aggregation."""
        processor = EventProcessor(event_bus)

        # Define stream processing pipeline
        pipeline = processor.create_pipeline("workflow_analytics")

        # Add processing stages
        (
            pipeline.filter(lambda e: e.get("type").startswith("workflow."))
            .map(lambda e: {**e, "processed_at": datetime.now().isoformat()})
            .window(size=10, slide=5)  # Sliding window
            .aggregate(
                lambda window: {
                    "count": len(window),
                    "avg_duration": sum(e.get("duration", 0) for e in window) / len(window),
                    "success_rate": sum(1 for e in window if e.get("status") == "success")
                    / len(window),
                }
            )
            .sink("analytics.aggregated")
        )

        # Start processing
        await pipeline.start()

        # Generate stream of events
        for i in range(50):
            event = {
                "type": "workflow.completed",
                "workflow_id": f"wf_{i}",
                "duration": 10 + i % 20,
                "status": "success" if i % 3 != 0 else "failure",
            }
            await event_bus.publish(event["type"], event)
            await asyncio.sleep(0.01)

        # Allow processing
        await asyncio.sleep(1)

        # Verify aggregated results
        results = await pipeline.get_results()
        assert len(results) > 0
        assert all("count" in r for r in results)
        assert all("avg_duration" in r for r in results)
        assert all("success_rate" in r for r in results)

        await pipeline.stop()

    @pytest.mark.integration
    async def test_event_transactions(self, event_bus, event_store):
        """Test transactional event publishing."""
        # Begin transaction
        tx = await event_store.begin_transaction()

        try:
            # Publish multiple events as part of transaction
            events = [
                {"type": "order.created", "order_id": "123", "amount": 100},
                {"type": "payment.processed", "order_id": "123", "status": "success"},
                {"type": "inventory.updated", "items": [{"id": "item1", "quantity": -1}]},
            ]

            for event in events:
                await tx.append(event)

            # Commit transaction
            await tx.commit()

            # Verify all events persisted
            stored_events = await event_store.get_events(
                filters={"order_id": "123"}, limit=10
            )
            assert len(stored_events) >= 2

        except Exception as e:
            # Rollback on error
            await tx.rollback()
            raise e

    @pytest.mark.integration
    async def test_event_schema_validation(self, event_bus):
        """Test event schema validation and evolution."""
        # Define event schemas
        schemas = {
            "workflow.started": {
                "type": "object",
                "required": ["workflow_id", "timestamp", "user"],
                "properties": {
                    "workflow_id": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "user": {"type": "string"},
                    "metadata": {"type": "object"},
                },
            }
        }

        validator = EventSchema(schemas)
        event_bus.set_validator(validator)

        # Test valid event
        valid_event = {
            "type": "workflow.started",
            "workflow_id": "test_123",
            "timestamp": datetime.now().isoformat(),
            "user": "researcher1",
        }

        result = await event_bus.publish("workflow.started", valid_event)
        assert result["success"] is True

        # Test invalid event (missing required field)
        invalid_event = {
            "type": "workflow.started",
            "workflow_id": "test_456",
            # Missing timestamp and user
        }

        with pytest.raises(ValidationError):
            await event_bus.publish("workflow.started", invalid_event)

    @pytest.mark.integration
    async def test_event_deduplication(self, event_bus):
        """Test event deduplication to prevent duplicate processing."""
        processed_events = []

        async def handler(event):
            processed_events.append(event)

        await event_bus.subscribe(["test.event"], handler, deduplication=True)

        # Publish same event multiple times
        event = {
            "id": "event_123",  # Deduplication key
            "type": "test.event",
            "data": "test_data",
        }

        for _ in range(5):
            await event_bus.publish("test.event", event)

        await asyncio.sleep(0.5)

        # Should only process once due to deduplication
        assert len(processed_events) == 1

    @pytest.mark.integration
    async def test_dead_letter_queue(self, event_bus):
        """Test dead letter queue for failed event processing."""
        # Configure DLQ
        dlq_config = {
            "max_retries": 3,
            "retry_delay": 0.1,
            "dlq_topic": "events.failed",
        }

        event_bus.configure_dlq(dlq_config)

        # Create failing handler
        attempt_count = 0

        async def failing_handler(event):
            nonlocal attempt_count
            attempt_count += 1
            raise Exception("Processing failed")

        await event_bus.subscribe(["test.failing"], failing_handler)

        # Subscribe to DLQ
        dlq_events = []

        async def dlq_handler(event):
            dlq_events.append(event)

        await event_bus.subscribe(["events.failed"], dlq_handler)

        # Publish event that will fail
        await event_bus.publish("test.failing", {"data": "test"})

        # Allow retries and DLQ processing
        await asyncio.sleep(1)

        # Verify retries occurred
        assert attempt_count == dlq_config["max_retries"]

        # Verify event moved to DLQ
        assert len(dlq_events) == 1
        assert dlq_events[0]["original_topic"] == "test.failing"
        assert dlq_events[0]["retry_count"] == dlq_config["max_retries"]

    @pytest.mark.integration
    async def test_event_priorities(self, event_bus):
        """Test event priority handling."""
        processed_order = []

        async def handler(event):
            processed_order.append(event["id"])
            await asyncio.sleep(0.01)  # Simulate processing

        await event_bus.subscribe(["priority.test"], handler, prioritized=True)

        # Publish events with different priorities
        events = [
            {"id": "low_1", "priority": 0, "data": "low priority"},
            {"id": "high_1", "priority": 10, "data": "high priority"},
            {"id": "medium_1", "priority": 5, "data": "medium priority"},
            {"id": "high_2", "priority": 10, "data": "high priority"},
            {"id": "low_2", "priority": 0, "data": "low priority"},
        ]

        for event in events:
            await event_bus.publish("priority.test", event, priority=event["priority"])

        # Allow processing
        await asyncio.sleep(0.5)

        # Verify high priority events processed first
        assert processed_order[0] in ["high_1", "high_2"]
        assert processed_order[1] in ["high_1", "high_2"]
        assert processed_order[-1] in ["low_1", "low_2"]

    @pytest.mark.integration
    async def test_event_batching(self, event_bus):
        """Test event batching for efficient processing."""
        batches_received = []

        async def batch_handler(batch):
            batches_received.append(batch)

        # Subscribe with batching
        await event_bus.subscribe(
            ["batch.test"], batch_handler, batch_size=10, batch_timeout=0.5
        )

        # Publish many events
        for i in range(25):
            await event_bus.publish("batch.test", {"id": i, "data": f"event_{i}"})
            await asyncio.sleep(0.01)

        # Allow batch processing
        await asyncio.sleep(1)

        # Verify batching occurred
        assert len(batches_received) == 3  # 10 + 10 + 5
        assert len(batches_received[0]) == 10
        assert len(batches_received[1]) == 10
        assert len(batches_received[2]) == 5

    @pytest.mark.integration
    async def test_event_filtering_and_transformation(self, event_bus):
        """Test event filtering and transformation pipelines."""
        # Create transformation pipeline
        pipeline = event_bus.create_pipeline("transformation")

        results = []

        async def sink(event):
            results.append(event)

        (
            pipeline.source(["raw.events"])
            .filter(lambda e: e.get("value", 0) > 10)
            .map(lambda e: {**e, "squared": e.get("value", 0) ** 2})
            .filter(lambda e: e.get("squared", 0) < 10000)
            .sink(sink)
        )

        await pipeline.start()

        # Publish test events
        for i in range(20):
            await event_bus.publish("raw.events", {"id": i, "value": i * 5})

        await asyncio.sleep(0.5)

        # Verify filtering and transformation
        assert all(r["value"] > 10 for r in results)
        assert all("squared" in r for r in results)
        assert all(r["squared"] < 10000 for r in results)

        await pipeline.stop()

    @pytest.mark.integration
    async def test_event_correlation(self, event_bus):
        """Test event correlation across multiple streams."""
        correlator = event_bus.create_correlator()

        # Define correlation rules
        correlator.correlate(
            streams=["orders", "payments", "shipping"],
            key="order_id",
            window=60,  # 60 second window
            output="correlated.orders",
        )

        correlated_results = []

        async def correlation_handler(correlated):
            correlated_results.append(correlated)

        await event_bus.subscribe(["correlated.orders"], correlation_handler)

        # Publish related events
        order_id = "order_789"

        await event_bus.publish("orders", {"order_id": order_id, "type": "order", "amount": 100})
        await event_bus.publish(
            "payments", {"order_id": order_id, "type": "payment", "status": "success"}
        )
        await event_bus.publish(
            "shipping", {"order_id": order_id, "type": "shipping", "carrier": "UPS"}
        )

        await asyncio.sleep(0.5)

        # Verify correlation
        assert len(correlated_results) == 1
        assert correlated_results[0]["order_id"] == order_id
        assert len(correlated_results[0]["events"]) == 3
        assert any(e["type"] == "order" for e in correlated_results[0]["events"])
        assert any(e["type"] == "payment" for e in correlated_results[0]["events"])
        assert any(e["type"] == "shipping" for e in correlated_results[0]["events"])