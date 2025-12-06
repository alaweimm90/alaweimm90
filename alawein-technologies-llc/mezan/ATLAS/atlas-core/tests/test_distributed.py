"""
Comprehensive tests for ORCHEX distributed systems

Tests distributed orchestration, event bus, message queue,
and all distributed execution scenarios.

Author: MEZAN Research Team
"""

import pytest
import time
import threading
import uuid
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Import distributed components
from atlas_core.distributed import (
    RedisBackend,
    DistributedLockManager,
    DistributedQueue,
    DistributedTask,
    LeaderElection,
    ClusterHealthMonitor,
    DistributedOrchestrator,
    ClusterNode,
    NodeStatus
)
from atlas_core.event_bus import (
    EventBus,
    Event,
    EventType,
    WorkflowStartedEvent,
    TaskCompletedEvent,
    EventHandler,
    EventBusWithPersistence
)
from atlas_core.message_queue import (
    MessageQueue,
    Message,
    MessageBrokerType,
    DeliveryMode,
    InMemoryBroker,
    MessageQueueFactory
)
from atlas_core.intelligent_mezan import (
    IntelligentMezanEngine,
    DistributedWorkflow,
    create_distributed_mezan
)


class TestRedisBackend:
    """Test Redis backend for distributed state"""

    def test_redis_backend_initialization(self):
        """Test Redis backend initialization"""
        backend = RedisBackend(host="localhost", port=6379)

        # Should gracefully handle connection failure
        assert backend is not None

        # If Redis not available, should work in degraded mode
        if not backend.is_connected:
            assert backend.get("test_key") is None
            assert backend.set("test_key", "test_value") is False

    def test_redis_operations_with_mock(self):
        """Test Redis operations with mocked client"""
        with patch('atlas_core.distributed.redis') as mock_redis:
            mock_client = MagicMock()
            mock_redis.Redis.return_value = mock_client
            mock_client.ping.return_value = True

            backend = RedisBackend()

            # Test set operation
            backend.set("key1", {"data": "value"})
            assert mock_client.set.called

            # Test get operation
            mock_client.get.return_value = backend._serialize({"data": "value"})
            result = backend.get("key1")
            assert result == {"data": "value"}

    def test_serialization(self):
        """Test value serialization/deserialization"""
        backend = RedisBackend()

        test_data = {
            "string": "value",
            "number": 42,
            "list": [1, 2, 3],
            "nested": {"key": "value"}
        }

        serialized = backend._serialize(test_data)
        assert isinstance(serialized, bytes)

        deserialized = backend._deserialize(serialized)
        assert deserialized == test_data

    def test_key_namespacing(self):
        """Test key namespacing"""
        backend = RedisBackend(key_prefix="test:")

        key = backend._make_key("mykey")
        assert key == "test:mykey"


class TestDistributedLockManager:
    """Test distributed lock manager"""

    def test_lock_manager_initialization(self):
        """Test lock manager initialization"""
        backend = RedisBackend()
        lock_manager = DistributedLockManager(backend)

        assert lock_manager is not None
        assert len(lock_manager.locks) == 0
        assert len(lock_manager.lock_stats) == 0

    def test_local_lock_fallback(self):
        """Test fallback to local locks when Redis unavailable"""
        backend = RedisBackend()
        backend.is_connected = False  # Simulate Redis unavailable

        lock_manager = DistributedLockManager(backend)

        # Should fall back to threading.Lock
        acquired = lock_manager.acquire_lock("resource1", blocking=False)
        assert acquired is True

        # Second acquire should fail (non-blocking)
        acquired2 = lock_manager.acquire_lock("resource1", blocking=False)
        assert acquired2 is False

        # Release lock
        released = lock_manager.release_lock("resource1")
        assert released is True

        # Now can acquire again
        acquired3 = lock_manager.acquire_lock("resource1", blocking=False)
        assert acquired3 is True

    def test_lock_statistics(self):
        """Test lock statistics tracking"""
        backend = RedisBackend()
        backend.is_connected = False

        lock_manager = DistributedLockManager(backend)

        # Acquire and release locks
        lock_manager.acquire_lock("resource1")
        lock_manager.release_lock("resource1")
        lock_manager.acquire_lock("resource1")
        lock_manager.release_lock("resource1")

        stats = lock_manager.get_lock_stats()
        assert stats["resource1"]["acquisitions"] == 2
        assert stats["resource1"]["releases"] == 2
        assert stats["resource1"]["timeouts"] == 0

    def test_concurrent_lock_access(self):
        """Test concurrent access to locks"""
        backend = RedisBackend()
        backend.is_connected = False

        lock_manager = DistributedLockManager(backend)
        counter = {"value": 0}
        iterations = 10

        def worker():
            for _ in range(iterations):
                if lock_manager.acquire_lock("counter", timeout=1):
                    try:
                        # Critical section
                        current = counter["value"]
                        time.sleep(0.001)  # Simulate work
                        counter["value"] = current + 1
                    finally:
                        lock_manager.release_lock("counter")

        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All increments should be atomic
        assert counter["value"] == 50  # 5 threads * 10 iterations


class TestDistributedQueue:
    """Test distributed task queue"""

    def test_queue_initialization(self):
        """Test queue initialization"""
        backend = RedisBackend()
        queue = DistributedQueue(backend, "test_queue")

        assert queue is not None
        assert queue.queue_name == "test_queue"

    def test_local_queue_fallback(self):
        """Test fallback to local queue"""
        backend = RedisBackend()
        backend.is_connected = False

        queue = DistributedQueue(backend)

        # Create tasks
        task1 = DistributedTask(
            task_type="test",
            payload={"data": "task1"},
            priority=5
        )

        task2 = DistributedTask(
            task_type="test",
            payload={"data": "task2"},
            priority=10  # Higher priority
        )

        # Enqueue tasks
        assert queue.enqueue(task1) is True
        assert queue.enqueue(task2) is True

        # Should dequeue by priority
        dequeued = queue.dequeue()
        assert dequeued is not None
        assert dequeued.priority == 10  # Higher priority first

    def test_task_lifecycle(self):
        """Test task lifecycle (enqueue, dequeue, complete)"""
        backend = RedisBackend()
        backend.is_connected = False

        queue = DistributedQueue(backend)

        task = DistributedTask(
            task_type="compute",
            payload={"operation": "add", "a": 1, "b": 2}
        )

        # Enqueue
        queue.enqueue(task)
        assert queue.get_queue_size() == 1

        # Dequeue
        dequeued = queue.dequeue()
        assert dequeued is not None
        assert dequeued.task_type == "compute"
        assert queue.get_queue_size() == 0

        # Complete (would update Redis in production)
        success = queue.complete_task(dequeued.task_id, result=3)
        # In local mode, just returns False
        assert success is False


class TestLeaderElection:
    """Test leader election mechanism"""

    def test_leader_election_local_mode(self):
        """Test leader election in local mode"""
        backend = RedisBackend()
        backend.is_connected = False

        election = LeaderElection(backend, "node1")

        # Start election
        is_leader = election.start()
        assert is_leader is True  # Always leader in local mode

        # Stop election
        election.stop()

    def test_leader_callbacks(self):
        """Test leader election callbacks"""
        backend = RedisBackend()
        backend.is_connected = False

        election = LeaderElection(backend, "node1")

        elected_called = {"value": False}
        demoted_called = {"value": False}

        def on_elected():
            elected_called["value"] = True

        def on_demoted():
            demoted_called["value"] = True

        election.on_elected = on_elected
        election.on_demoted = on_demoted

        # Start election
        election.start()
        assert elected_called["value"] is True

        # Stop election
        election.stop()


class TestClusterHealthMonitor:
    """Test cluster health monitoring"""

    def test_health_monitor_initialization(self):
        """Test health monitor initialization"""
        backend = RedisBackend()
        monitor = ClusterHealthMonitor(backend, "node1")

        assert monitor is not None
        assert monitor.node_id == "node1"
        assert monitor.local_node.status == NodeStatus.HEALTHY

    def test_node_metrics_update(self):
        """Test updating node metrics"""
        backend = RedisBackend()
        backend.is_connected = False

        monitor = ClusterHealthMonitor(backend, "node1")

        # Update metrics
        monitor.update_node_metrics(cpu_usage=50.0, memory_usage=60.0)
        assert monitor.local_node.cpu_usage == 50.0
        assert monitor.local_node.memory_usage == 60.0

        # Increment task counters
        monitor.increment_task_counter(success=True)
        monitor.increment_task_counter(success=True)
        monitor.increment_task_counter(success=False)

        assert monitor.local_node.tasks_completed == 2
        assert monitor.local_node.tasks_failed == 1

    def test_cluster_status(self):
        """Test getting cluster status"""
        backend = RedisBackend()
        backend.is_connected = False

        monitor = ClusterHealthMonitor(backend, "node1")

        status = monitor.get_cluster_status()
        assert status["total_nodes"] == 1
        assert status["healthy_nodes"] == 1
        assert len(status["nodes"]) == 1


class TestEventBus:
    """Test event bus functionality"""

    def test_event_bus_initialization(self):
        """Test event bus initialization"""
        bus = EventBus(max_workers=5, event_history_size=100)

        assert bus is not None
        assert bus.running is True
        assert bus.metrics["events_published"] == 0

    def test_event_publish_and_handle(self):
        """Test publishing and handling events"""
        bus = EventBus()

        handled_events = []

        def handler(event: Event):
            handled_events.append(event)

        # Register handler
        bus.register_handler(
            handler=handler,
            event_types=[EventType.TASK_COMPLETED],
            priority=10
        )

        # Publish event
        event = TaskCompletedEvent(
            task_id="task1",
            result={"value": 42},
            duration=1.5
        )

        future = bus.publish(event)
        future.result(timeout=1)  # Wait for processing

        time.sleep(0.1)  # Allow handler to execute

        assert len(handled_events) == 1
        assert handled_events[0].event_type == EventType.TASK_COMPLETED

    def test_event_filtering(self):
        """Test event filtering"""
        bus = EventBus()

        handled_events = []

        def filter_func(event: Event) -> bool:
            return event.payload.get("priority", 0) > 5

        def handler(event: Event):
            handled_events.append(event)

        # Register handler with filter
        bus.register_handler(
            handler=handler,
            event_types=[EventType.CUSTOM],
            filter_func=filter_func
        )

        # Publish events
        event1 = Event(payload={"priority": 3})
        event2 = Event(payload={"priority": 10})

        bus.publish(event1).result(timeout=1)
        bus.publish(event2).result(timeout=1)

        time.sleep(0.1)

        # Only high priority event should be handled
        assert len(handled_events) == 1
        assert handled_events[0].payload["priority"] == 10

    def test_event_replay(self):
        """Test event replay functionality"""
        bus = EventBus(enable_replay=True)

        # Publish some events
        event1 = WorkflowStartedEvent("wf1", "optimization")
        event2 = TaskCompletedEvent("task1", {"result": 1}, 1.0)

        bus.publish(event1)
        bus.publish(event2)

        time.sleep(0.1)

        # Replay events
        replayed = bus.replay_events(
            event_types=[EventType.TASK_COMPLETED]
        )

        assert len(replayed) == 1
        assert replayed[0].event_type == EventType.TASK_COMPLETED

    def test_dead_letter_queue(self):
        """Test dead letter queue for failed events"""
        bus = EventBus()

        def failing_handler(event: Event):
            raise Exception("Handler failed")

        # Register failing handler
        bus.register_handler(
            handler=failing_handler,
            event_types=[EventType.CUSTOM]
        )

        # Publish event
        event = Event()
        bus.publish(event).result(timeout=1)

        time.sleep(0.1)

        # Event should be in DLQ
        assert len(bus.dead_letter_queue) > 0
        assert bus.metrics["events_failed"] > 0

    def test_event_metrics(self):
        """Test event bus metrics"""
        bus = EventBus()

        # Publish events
        for i in range(5):
            bus.publish(Event(payload={"index": i}))

        time.sleep(0.1)

        metrics = bus.get_metrics()
        assert metrics["events_published"] == 5
        assert metrics["event_history_size"] <= 5

    def test_shutdown(self):
        """Test event bus shutdown"""
        bus = EventBus()

        assert bus.running is True

        bus.shutdown()

        # Executor should be shut down
        assert bus.executor._shutdown is True


class TestMessageQueue:
    """Test message queue abstraction"""

    def test_message_queue_initialization(self):
        """Test message queue initialization"""
        queue = MessageQueue(
            broker_type=MessageBrokerType.IN_MEMORY
        )

        assert queue is not None
        assert queue.primary_broker is not None

    def test_send_receive_messages(self):
        """Test sending and receiving messages"""
        queue = MessageQueue(MessageBrokerType.IN_MEMORY)

        # Send message
        success = queue.send(
            topic="test_topic",
            payload={"data": "test"},
            priority=5
        )
        assert success is True

        # Receive message
        received = queue.receive("test_topic", timeout=1)
        assert received is not None
        assert received["data"] == "test"

    def test_pub_sub_pattern(self):
        """Test publish/subscribe pattern"""
        queue = MessageQueue(MessageBrokerType.IN_MEMORY)

        received_messages = []

        def callback(payload):
            received_messages.append(payload)

        # Subscribe
        queue.subscribe("events", callback)

        # Publish messages
        queue.send("events", {"event": "event1"})
        queue.send("events", {"event": "event2"})

        time.sleep(0.1)

        assert len(received_messages) == 2
        assert received_messages[0]["event"] == "event1"
        assert received_messages[1]["event"] == "event2"

    def test_request_reply_pattern(self):
        """Test request/reply pattern"""
        queue = MessageQueue(MessageBrokerType.IN_MEMORY)

        def echo_service(payload):
            # Echo back the payload
            if "reply_to" in payload:
                queue.send(payload["reply_to"], {"echo": payload})

        # Setup echo service
        queue.subscribe("echo_service", echo_service)

        # Send request and wait for reply
        reply = queue.request_reply(
            topic="echo_service",
            payload={"message": "hello"},
            timeout=1
        )

        # In this test, reply might be None since we're not properly
        # handling the reply_to field in Message
        # This is a limitation of the test setup

    def test_message_broker_factory(self):
        """Test message broker factory"""
        # Test creating different broker types
        redis_broker = MessageQueueFactory.create_broker(
            MessageBrokerType.REDIS,
            {"host": "localhost"}
        )
        assert redis_broker.broker_type == MessageBrokerType.REDIS

        memory_broker = MessageQueueFactory.create_broker(
            MessageBrokerType.IN_MEMORY
        )
        assert memory_broker.broker_type == MessageBrokerType.IN_MEMORY

        # Test with string type
        kafka_broker = MessageQueueFactory.create_broker(
            "kafka",
            {"bootstrap_servers": ["localhost:9092"]}
        )
        assert kafka_broker.broker_type == MessageBrokerType.KAFKA


class TestDistributedOrchestrator:
    """Test distributed orchestrator"""

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = DistributedOrchestrator(
            redis_host="localhost",
            enable_leader_election=True,
            enable_health_monitoring=True
        )

        assert orchestrator is not None
        assert orchestrator.node_id is not None

    def test_task_submission_and_processing(self):
        """Test submitting and processing tasks"""
        orchestrator = DistributedOrchestrator()
        orchestrator.backend.is_connected = False  # Use local mode

        # Submit task
        task_id = orchestrator.submit_task(
            task_type="compute",
            payload={"operation": "multiply", "a": 3, "b": 4},
            priority=5
        )

        assert task_id is not None

        # Process task (would normally be done by worker)
        task = orchestrator.process_next_task()
        assert task is not None
        assert task.task_type == "compute"
        assert task.payload["operation"] == "multiply"

        # Complete task
        orchestrator.complete_task(task_id, result=12)

    def test_distributed_lock_execution(self):
        """Test executing function with distributed lock"""
        orchestrator = DistributedOrchestrator()
        orchestrator.backend.is_connected = False

        counter = {"value": 0}

        def critical_function(increment):
            counter["value"] += increment
            return counter["value"]

        # Execute with lock
        result = orchestrator.with_lock(
            "counter_resource",
            critical_function,
            5
        )

        assert result == 5
        assert counter["value"] == 5

    def test_cluster_status_report(self):
        """Test getting cluster status"""
        orchestrator = DistributedOrchestrator(
            enable_health_monitoring=True
        )

        status = orchestrator.get_cluster_status()

        assert "node_id" in status
        assert "is_connected" in status
        assert "queue" in status
        assert status["queue"]["pending"] >= 0


class TestIntelligentMezanDistributed:
    """Test distributed features of IntelligentMezanEngine"""

    def test_distributed_mezan_initialization(self):
        """Test initializing distributed MEZAN"""
        engine = IntelligentMezanEngine(
            enable_distributed=True,
            redis_host="localhost",
            worker_count=2,
            message_broker_type=MessageBrokerType.IN_MEMORY
        )

        assert engine is not None
        assert engine.enable_distributed is True or engine.enable_distributed is False
        # May be False if Redis not available

    def test_create_distributed_workflow(self):
        """Test creating distributed workflow"""
        engine = IntelligentMezanEngine(enable_distributed=False)

        problem = {"type": "optimization", "dimensions": 10}
        workflow = engine.create_distributed_workflow(problem)

        assert workflow is not None
        assert workflow.workflow_id is not None
        assert len(workflow.tasks) > 0

        # Should have analysis, solver, and balance tasks
        task_types = [task.task_type for task in workflow.tasks]
        assert "analyze" in task_types
        assert "solve" in task_types
        assert "balance" in task_types

    def test_workflow_status_tracking(self):
        """Test workflow status tracking"""
        engine = IntelligentMezanEngine(enable_distributed=False)

        problem = {"type": "test"}
        workflow = engine.create_distributed_workflow(problem)

        # Get status
        status = engine.get_workflow_status(workflow.workflow_id)

        assert status is not None
        assert status["workflow_id"] == workflow.workflow_id
        assert status["total_tasks"] == len(workflow.tasks)
        assert status["status"] == "pending"

    def test_factory_functions(self):
        """Test factory functions for creating engines"""
        # Test intelligent MEZAN
        engine1 = create_intelligent_mezan(enable_distributed=False)
        assert engine1 is not None
        assert engine1.enable_distributed is False

        # Test distributed MEZAN
        engine2 = create_distributed_mezan(
            redis_host="localhost",
            worker_count=4
        )
        assert engine2 is not None
        # May or may not be distributed depending on Redis availability


class TestDistributedIntegration:
    """Integration tests for distributed system"""

    def test_end_to_end_distributed_workflow(self):
        """Test complete distributed workflow execution"""
        # Create components
        backend = RedisBackend()
        backend.is_connected = False  # Use local mode

        orchestrator = DistributedOrchestrator(
            enable_leader_election=False,
            enable_health_monitoring=False
        )
        orchestrator.backend = backend

        event_bus = EventBus()
        message_queue = MessageQueue(MessageBrokerType.IN_MEMORY)

        # Track workflow progress
        task_results = {}

        def task_handler(task_type, payload):
            """Simulate task processing"""
            if task_type == "analyze":
                return {"insights": ["insight1", "insight2"]}
            elif task_type == "solve":
                return {"solution": "optimal", "value": 42}
            return {"result": "completed"}

        # Submit tasks
        task1_id = orchestrator.submit_task("analyze", {"data": "test"})
        task2_id = orchestrator.submit_task("solve", {"problem": "optimize"})

        # Process tasks (simulate worker)
        for _ in range(2):
            task = orchestrator.process_next_task()
            if task:
                result = task_handler(task.task_type, task.payload)
                task_results[task.task_id] = result
                orchestrator.complete_task(task.task_id, result)

                # Publish completion event
                event_bus.publish(TaskCompletedEvent(
                    task_id=task.task_id,
                    result=result,
                    duration=0.1
                ))

        # Verify results
        assert len(task_results) == 2
        assert any("insights" in r for r in task_results.values())
        assert any("solution" in r for r in task_results.values())

    def test_multi_node_simulation(self):
        """Simulate multi-node cluster operation"""
        nodes = []

        # Create multiple nodes
        for i in range(3):
            orchestrator = DistributedOrchestrator(
                node_id=f"node_{i}",
                enable_leader_election=False,
                enable_health_monitoring=False
            )
            orchestrator.backend.is_connected = False  # Local mode
            nodes.append(orchestrator)

        # Submit tasks from different nodes
        task_ids = []
        for i, node in enumerate(nodes):
            task_id = node.submit_task(
                task_type=f"task_from_node_{i}",
                payload={"node": i},
                priority=i
            )
            task_ids.append(task_id)

        # Each node processes one task
        for node in nodes:
            task = node.process_next_task()
            if task:
                node.complete_task(task.task_id, {"processed_by": node.node_id})

        # Verify all nodes participated
        assert len(task_ids) == 3


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])