"""
Failure Scenarios Integration Tests for ORCHEX

Tests system resilience, error handling, and recovery from various failure conditions.
"""

import pytest
import asyncio
import time
import random
import uuid
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import psutil
import signal

from atlas_core.engine import ATLASEngine
from atlas_core.resilience import (
    CircuitBreaker,
    RetryManager,
    FallbackHandler,
    BulkheadIsolation,
    TimeoutManager,
    GracefulDegradation,
)
from atlas_core.monitoring import HealthChecker, AlertManager


class TestFailureScenarios:
    """Test various failure scenarios and recovery mechanisms."""

    @pytest.fixture
    def engine(self):
        """Create engine with resilience features enabled."""
        return ATLASEngine(
            redis_url="redis://localhost:6379/1",
            resilience_config={
                "circuit_breaker_enabled": True,
                "retry_enabled": True,
                "bulkhead_enabled": True,
                "timeout_enabled": True,
                "graceful_degradation": True,
            },
        )

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker instance."""
        return CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=Exception,
            half_open_requests=2,
        )

    @pytest.mark.integration
    async def test_agent_failure_and_recovery(self, engine):
        """Test handling of agent failures and automatic recovery."""
        # Configure agent to fail
        failing_agent_config = {
            "agent": "DataAnalysisAgent",
            "failure_mode": "exception",
            "failure_rate": 0.5,  # 50% failure rate
            "failure_message": "Simulated analysis failure",
        }

        engine.inject_failure(failing_agent_config)

        # Execute workflow with potential failures
        workflow = {
            "id": "test_workflow",
            "steps": [
                {"agent": "DataCollector", "task": "collect"},
                {"agent": "DataAnalysisAgent", "task": "analyze"},
                {"agent": "Reporter", "task": "report"},
            ],
            "retry_policy": {
                "max_attempts": 3,
                "backoff": "exponential",
                "initial_delay": 1,
            },
        }

        results = []
        for i in range(10):
            result = await engine.execute_workflow(workflow)
            results.append(result)

        # Verify some succeeded despite failures
        successful = [r for r in results if r["status"] == "completed"]
        assert len(successful) > 0

        # Verify retry attempts were made
        retried = [r for r in results if r.get("retry_count", 0) > 0]
        assert len(retried) > 0

    @pytest.mark.integration
    async def test_database_connection_failure(self, engine):
        """Test handling of database connection failures."""
        # Simulate database outage
        with patch("atlas_core.database.Database.connect") as mock_connect:
            mock_connect.side_effect = ConnectionError("Database unavailable")

            # Try to execute database-dependent operation
            workflow = {
                "id": "db_workflow",
                "requires_persistence": True,
                "task": "Store results in database",
            }

            result = await engine.execute_workflow(workflow)

            # Should use fallback (e.g., local file storage)
            assert result["status"] == "completed_with_fallback"
            assert result["fallback_used"] == "local_storage"
            assert "warning" in result
            assert "database unavailable" in result["warning"].lower()

    @pytest.mark.integration
    async def test_redis_blackboard_failure(self, engine):
        """Test handling of Redis blackboard failures."""
        # Simulate Redis failure
        original_write = engine.blackboard.write

        async def failing_write(*args, **kwargs):
            if random.random() < 0.3:  # 30% failure rate
                raise ConnectionError("Redis connection lost")
            return await original_write(*args, **kwargs)

        engine.blackboard.write = failing_write

        # Execute workflow that heavily uses blackboard
        workflow = {
            "id": "blackboard_workflow",
            "agents": ["Agent1", "Agent2", "Agent3"],
            "blackboard_operations": 100,
        }

        result = await engine.execute_workflow(workflow)

        # Should complete using in-memory fallback
        assert result["status"] in ["completed", "completed_with_degradation"]
        assert result.get("blackboard_fallback") is True
        assert result.get("degraded_features", []) == ["distributed_state"]

    @pytest.mark.integration
    async def test_network_partition(self, engine):
        """Test handling of network partitions in distributed setup."""
        # Simulate network partition
        partition_config = {
            "type": "network_partition",
            "affected_nodes": ["node2", "node3"],
            "duration": 10,  # seconds
        }

        engine.simulate_failure(partition_config)

        # Execute distributed workflow
        workflow = {
            "id": "distributed_workflow",
            "distribution": {"nodes": ["node1", "node2", "node3", "node4"]},
            "tasks": [f"task_{i}" for i in range(20)],
        }

        start_time = time.time()
        result = await engine.execute_workflow(workflow)
        duration = time.time() - start_time

        # Should complete but with reduced capacity
        assert result["status"] == "completed"
        assert result["nodes_used"] == 2  # Only unaffected nodes
        assert duration > 10  # Should wait for partition to heal
        assert result.get("partition_detected") is True

    @pytest.mark.integration
    async def test_memory_exhaustion(self, engine):
        """Test handling of memory exhaustion scenarios."""
        # Monitor memory usage
        initial_memory = psutil.Process().memory_info().rss

        # Configure memory limits
        engine.set_resource_limits({"max_memory_mb": 512, "gc_threshold": 0.8})

        # Create memory-intensive workflow
        workflow = {
            "id": "memory_intensive",
            "task": "process_large_dataset",
            "data_size": "10GB",
            "processing_strategy": "streaming",
        }

        # Execute with memory pressure
        result = await engine.execute_workflow(workflow)

        # Should adapt to memory constraints
        assert result["status"] == "completed"
        assert result["memory_optimization_applied"] is True
        assert result["processing_mode"] == "streaming"  # Instead of in-memory

        # Verify memory didn't exceed limit
        peak_memory = psutil.Process().memory_info().rss
        assert (peak_memory - initial_memory) / 1024 / 1024 < 512

    @pytest.mark.integration
    async def test_cpu_throttling(self, engine):
        """Test handling of CPU resource constraints."""
        # Set CPU limits
        engine.set_resource_limits({"max_cpu_percent": 50, "worker_threads": 2})

        # Create CPU-intensive workflow
        workflow = {
            "id": "cpu_intensive",
            "tasks": [
                {"type": "compute", "complexity": "high", "iterations": 1000000}
                for _ in range(10)
            ],
        }

        # Monitor CPU usage during execution
        cpu_samples = []

        async def monitor_cpu():
            while True:
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
                await asyncio.sleep(0.1)

        monitor_task = asyncio.create_task(monitor_cpu())

        # Execute workflow
        result = await engine.execute_workflow(workflow)

        monitor_task.cancel()

        # Verify CPU throttling worked
        assert result["status"] == "completed"
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
        assert avg_cpu < 60  # Should stay near 50% limit

    @pytest.mark.integration
    async def test_cascading_failures(self, engine):
        """Test prevention of cascading failures."""
        # Setup dependencies between services
        services = {
            "service_a": {"depends_on": []},
            "service_b": {"depends_on": ["service_a"]},
            "service_c": {"depends_on": ["service_b"]},
            "service_d": {"depends_on": ["service_b", "service_c"]},
        }

        # Inject failure in service_a
        engine.inject_failure({"service": "service_a", "type": "crash"})

        # Execute workflow using all services
        workflow = {"id": "cascading_test", "services": list(services.keys())}

        result = await engine.execute_workflow(workflow)

        # Should isolate failure and use fallbacks
        assert result["status"] == "partial_success"
        assert result["failed_services"] == ["service_a"]
        assert result["isolated_services"] == ["service_b", "service_c", "service_d"]
        assert result["fallback_services"]["service_a"] == "cache"

    @pytest.mark.integration
    async def test_circuit_breaker_pattern(self, circuit_breaker):
        """Test circuit breaker implementation."""
        failure_count = 0
        success_count = 0

        async def flaky_service():
            nonlocal failure_count
            if failure_count < 5:
                failure_count += 1
                raise Exception("Service temporarily unavailable")
            return "success"

        # Test circuit breaker states
        results = []
        for i in range(15):
            try:
                result = await circuit_breaker.call(flaky_service)
                results.append(("success", result))
                success_count += 1
            except Exception as e:
                results.append(("failure", str(e)))
            await asyncio.sleep(0.5)

        # Verify circuit breaker behavior
        assert failure_count == 5  # Should stop calling after threshold
        assert any("Circuit breaker is OPEN" in str(r[1]) for r in results)
        assert circuit_breaker.state in ["OPEN", "HALF_OPEN", "CLOSED"]

    @pytest.mark.integration
    async def test_retry_with_backoff(self, engine):
        """Test retry mechanisms with various backoff strategies."""
        retry_manager = RetryManager(
            strategies={
                "exponential": {"base": 2, "max_delay": 60},
                "linear": {"increment": 5, "max_delay": 30},
                "fibonacci": {"max_delay": 55},
                "jitter": {"base": 2, "jitter_range": 0.3},
            }
        )

        # Test each strategy
        for strategy_name in ["exponential", "linear", "fibonacci", "jitter"]:
            attempts = []

            async def failing_task():
                attempts.append(time.time())
                if len(attempts) < 3:
                    raise Exception(f"Attempt {len(attempts)} failed")
                return "success"

            result = await retry_manager.execute_with_retry(
                failing_task, strategy=strategy_name, max_attempts=5
            )

            assert result == "success"
            assert len(attempts) == 3

            # Verify backoff timing
            if len(attempts) > 1:
                delays = [attempts[i] - attempts[i - 1] for i in range(1, len(attempts))]
                if strategy_name == "exponential":
                    assert delays[1] > delays[0]  # Exponential growth

    @pytest.mark.integration
    async def test_bulkhead_isolation(self, engine):
        """Test bulkhead pattern for failure isolation."""
        bulkhead = BulkheadIsolation(
            compartments={
                "critical": {"max_concurrent": 2, "queue_size": 5},
                "normal": {"max_concurrent": 5, "queue_size": 10},
                "batch": {"max_concurrent": 1, "queue_size": 20},
            }
        )

        # Submit tasks to different compartments
        tasks = []
        for priority in ["critical", "normal", "batch"]:
            for i in range(10):
                task = {
                    "id": f"{priority}_{i}",
                    "compartment": priority,
                    "duration": random.uniform(0.1, 0.5),
                }
                tasks.append(bulkhead.submit(task))

        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify isolation
        critical_results = results[:10]
        normal_results = results[10:20]
        batch_results = results[20:30]

        # Critical should complete despite others
        critical_completed = sum(1 for r in critical_results if not isinstance(r, Exception))
        assert critical_completed >= 7  # Most should complete

    @pytest.mark.integration
    async def test_timeout_management(self, engine):
        """Test timeout handling for long-running operations."""
        timeout_manager = TimeoutManager(
            default_timeout=5,
            operation_timeouts={
                "api_call": 2,
                "database_query": 3,
                "agent_execution": 10,
            },
        )

        # Test operation that exceeds timeout
        async def slow_operation():
            await asyncio.sleep(10)
            return "completed"

        with pytest.raises(asyncio.TimeoutError):
            await timeout_manager.execute_with_timeout(
                slow_operation(), timeout=2
            )

        # Test operation that completes within timeout
        async def fast_operation():
            await asyncio.sleep(0.5)
            return "completed"

        result = await timeout_manager.execute_with_timeout(
            fast_operation(), timeout=2
        )
        assert result == "completed"

    @pytest.mark.integration
    async def test_graceful_degradation(self, engine):
        """Test graceful degradation of features under stress."""
        degradation_manager = GracefulDegradation(
            features={
                "real_time_analytics": {"priority": 3, "fallback": "batch_analytics"},
                "ai_recommendations": {"priority": 2, "fallback": "rule_based"},
                "core_processing": {"priority": 1, "fallback": None},  # Cannot degrade
            },
            thresholds={
                "cpu": 80,
                "memory": 85,
                "error_rate": 0.05,
            },
        )

        # Simulate high load
        load_metrics = {
            "cpu": 90,
            "memory": 70,
            "error_rate": 0.02,
        }

        # Apply degradation
        active_features = await degradation_manager.evaluate_features(load_metrics)

        # Verify degradation decisions
        assert active_features["core_processing"] == "active"
        assert active_features["ai_recommendations"] == "active"
        assert active_features["real_time_analytics"] == "degraded"  # Due to high CPU

    @pytest.mark.integration
    async def test_poison_pill_handling(self, engine):
        """Test handling of poison pill messages."""
        # Create workflow with poison pill detection
        workflow = {
            "id": "poison_pill_test",
            "poison_pill_detection": True,
            "max_processing_attempts": 3,
        }

        # Inject poison pill message
        poison_message = {
            "id": "poison_123",
            "content": None,  # Malformed content
            "corrupt": True,
        }

        engine.blackboard.write("task_queue", poison_message)

        # Execute workflow
        result = await engine.execute_workflow(workflow)

        # Should detect and quarantine poison pill
        assert result["status"] == "completed_with_errors"
        assert "poison_pill_detected" in result
        assert result["poison_pill_detected"] == ["poison_123"]
        assert result["quarantined_messages"] == 1

    @pytest.mark.integration
    async def test_deadlock_detection_and_resolution(self, engine):
        """Test detection and resolution of deadlocks."""
        # Create workflow with potential deadlock
        workflow = {
            "id": "deadlock_test",
            "agents": [
                {"name": "agent_a", "requires": ["resource_1"], "provides": ["resource_2"]},
                {"name": "agent_b", "requires": ["resource_2"], "provides": ["resource_3"]},
                {"name": "agent_c", "requires": ["resource_3"], "provides": ["resource_1"]},
            ],
            "deadlock_detection": True,
            "resolution_strategy": "victim_selection",
        }

        # Execute workflow
        result = await engine.execute_workflow(workflow)

        # Should detect and resolve deadlock
        assert result["status"] == "completed"
        assert result.get("deadlock_detected") is True
        assert result.get("deadlock_resolved") is True
        assert "victim_agent" in result
        assert result["resolution_strategy"] == "victim_selection"

    @pytest.mark.integration
    async def test_chaos_engineering_resilience(self, engine):
        """Test system resilience using chaos engineering."""
        # Configure chaos scenarios
        chaos_scenarios = [
            {"type": "latency_injection", "target": "api_calls", "delay": 5},
            {"type": "error_injection", "target": "database", "error_rate": 0.3},
            {"type": "resource_exhaustion", "target": "memory", "usage": 90},
            {"type": "network_packet_loss", "target": "redis", "loss_rate": 0.1},
        ]

        # Run workflow under chaos conditions
        results = []
        for scenario in chaos_scenarios:
            engine.enable_chaos(scenario)

            workflow = {
                "id": f"chaos_{scenario['type']}",
                "tasks": ["collect", "process", "analyze", "store"],
            }

            result = await engine.execute_workflow(workflow)
            results.append({
                "scenario": scenario["type"],
                "result": result,
            })

            engine.disable_chaos()

        # Verify system remained operational
        successful = [r for r in results if r["result"]["status"] in ["completed", "completed_with_degradation"]]
        assert len(successful) >= 3  # Most scenarios should be handled

        # Verify adaptive responses
        for result in results:
            if result["result"]["status"] == "completed_with_degradation":
                assert "adaptations" in result["result"]
                assert len(result["result"]["adaptations"]) > 0