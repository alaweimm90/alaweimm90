"""
Integration tests for ORCHEX-Librex integration

Tests the integration between MEZAN/ORCHEX and Librex optimization framework.
"""

import json
import time
import uuid
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
import redis

from Librex.integrations.ORCHEX import (
    ATLASConfig,
    ATLASOptimizationAdapter,
    LibrexAgent,
    OptimizationTaskQueue,
)
from Librex.integrations.ORCHEX.task_queue import OptimizationTask, TaskStatus


@pytest.fixture
def redis_client():
    """Mock Redis client for testing"""
    mock_redis = Mock(spec=redis.Redis)
    mock_redis.ping.return_value = True
    mock_redis.hgetall.return_value = {}
    mock_redis.hset.return_value = True
    mock_redis.zadd.return_value = 1
    mock_redis.zpopmax.return_value = []
    mock_redis.sadd.return_value = 1
    mock_redis.srem.return_value = 1
    mock_redis.smembers.return_value = set()
    mock_redis.llen.return_value = 0
    mock_redis.zcard.return_value = 0
    mock_redis.scan.return_value = (0, [])
    return mock_redis


@pytest.fixture
def config():
    """Test configuration"""
    return ATLASConfig(
        redis_url="redis://localhost:6379/0",
        agent_id="test-agent",
        max_concurrent_tasks=5,
        default_method="simulated_annealing",
        enable_gpu=False
    )


@pytest.fixture
def task_queue(config, redis_client):
    """Task queue with mocked Redis"""
    with patch('redis.Redis.from_url', return_value=redis_client):
        queue = OptimizationTaskQueue(config)
        return queue


@pytest.fixture
def adapter(config):
    """ORCHEX adapter with mocked components"""
    with patch('redis.Redis.from_url'):
        adapter = ATLASOptimizationAdapter(config)
        return adapter


@pytest.fixture
def agent(config, redis_client):
    """Librex agent with mocked Redis"""
    with patch('redis.Redis.from_url', return_value=redis_client):
        agent = LibrexAgent(config)
        return agent


class TestATLASConfig:
    """Test configuration management"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ATLASConfig()

        assert config.redis_url == "redis://localhost:6379/0"
        assert config.agent_id == "Librex-agent"
        assert config.default_method == "auto"
        assert config.max_concurrent_tasks == 10
        assert config.enable_gpu is True

    def test_config_from_env(self, monkeypatch):
        """Test configuration from environment variables"""
        monkeypatch.setenv("ATLAS_REDIS_URL", "redis://test:6379/1")
        monkeypatch.setenv("Librex_AGENT_ID", "test-agent-env")
        monkeypatch.setenv("Librex_DEFAULT_METHOD", "genetic_algorithm")
        monkeypatch.setenv("Librex_ENABLE_GPU", "false")

        config = ATLASConfig.from_env()

        assert config.redis_url == "redis://test:6379/1"
        assert config.agent_id == "test-agent-env"
        assert config.default_method == "genetic_algorithm"
        assert config.enable_gpu is False

    def test_config_to_dict(self, config):
        """Test configuration serialization"""
        config_dict = config.to_dict()

        assert config_dict["agent_id"] == "test-agent"
        assert config_dict["default_method"] == "simulated_annealing"
        assert "method_configs" not in config_dict  # Not included in to_dict


class TestOptimizationTaskQueue:
    """Test Redis-based task queue"""

    def test_submit_task(self, task_queue, redis_client):
        """Test task submission"""
        task_id = task_queue.submit_task(
            problem_type="qap",
            problem_data={"test": "data"},
            method="simulated_annealing",
            agent_id="test-agent"
        )

        assert task_id is not None
        assert redis_client.hset.called
        assert redis_client.zadd.called

    def test_get_task_not_found(self, task_queue, redis_client):
        """Test getting non-existent task"""
        redis_client.hgetall.return_value = {}

        task = task_queue.get_task("non-existent")
        assert task is None

    def test_get_task_success(self, task_queue, redis_client):
        """Test getting existing task"""
        task_data = {
            "task_id": "test-123",
            "problem_type": "qap",
            "problem_data": '{"test": "data"}',
            "method": "simulated_annealing",
            "config": "{}",
            "status": "pending",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "priority": "0",
            "retries": "0"
        }
        redis_client.hgetall.return_value = task_data

        task = task_queue.get_task("test-123")

        assert task is not None
        assert task.task_id == "test-123"
        assert task.problem_type == "qap"
        assert task.status == TaskStatus.PENDING

    def test_update_task_status(self, task_queue, redis_client):
        """Test updating task status"""
        success = task_queue.update_task_status(
            "test-123",
            TaskStatus.COMPLETED,
            result={"objective": 100}
        )

        assert success is True
        assert redis_client.hset.called

    def test_cancel_task(self, task_queue, redis_client):
        """Test task cancellation"""
        # Mock existing task
        task_data = {
            "task_id": "test-123",
            "status": "pending",
            "problem_type": "qap",
            "problem_data": "{}",
            "method": "auto",
            "config": "{}",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
            "priority": "0",
            "retries": "0"
        }
        redis_client.hgetall.return_value = task_data

        success = task_queue.cancel_task("test-123")
        assert success is True

    def test_get_queue_stats(self, task_queue, redis_client):
        """Test queue statistics"""
        redis_client.zcard.return_value = 5
        redis_client.llen.return_value = 3

        stats = task_queue.get_queue_stats()

        assert stats["pending_tasks"] == 5
        assert stats["results_available"] == 3
        assert stats["redis_connected"] is True

    def test_retry_task(self, task_queue, redis_client):
        """Test task retry logic"""
        # Mock task with retries < max
        task_data = {
            "task_id": "test-123",
            "status": "failed",
            "retries": "1",
            "priority": "1",
            "problem_type": "qap",
            "problem_data": "{}",
            "method": "auto",
            "config": "{}",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01"
        }
        redis_client.hgetall.return_value = task_data

        with patch('time.sleep'):  # Skip actual sleep
            success = task_queue.retry_task("test-123")

        assert success is True
        assert redis_client.hincrby.called


class TestATLASOptimizationAdapter:
    """Test ORCHEX optimization adapter"""

    def test_submit_optimization_request(self, adapter):
        """Test optimization request submission"""
        with patch.object(adapter.task_queue, 'submit_task', return_value="task-123"):
            result = adapter.submit_optimization_request(
                agent_id="test-agent",
                problem_type="qap",
                problem_data={"test": "data"},
                method="simulated_annealing"
            )

        assert result["task_id"] == "task-123"
        assert result["status"] == "submitted"
        assert "redis_blackboard_key" in result

    def test_get_method_recommendation(self, adapter):
        """Test method recommendation"""
        with patch.object(
            adapter.method_selector,
            'recommend_method',
            return_value=("genetic_algorithm", {"population_size": 100}, 0.85)
        ):
            recommendation = adapter.get_method_recommendation(
                agent_id="test-agent",
                problem_type="qap",
                problem_data={"flow_matrix": [[0, 1], [1, 0]]}
            )

        assert recommendation["recommendation"]["method"] == "genetic_algorithm"
        assert recommendation["recommendation"]["confidence"] == 0.85

    @patch('Librex.optimize')
    def test_process_task_success(self, mock_optimize, adapter):
        """Test successful task processing"""
        mock_optimize.return_value = {
            "solution": [0, 1, 2],
            "objective": 100,
            "is_valid": True,
            "iterations": 1000
        }

        task = OptimizationTask(
            task_id="test-123",
            problem_type="qap",
            problem_data={"test": "data"},
            method="simulated_annealing",
            config={},
            status=TaskStatus.IN_PROGRESS,
            created_at="2024-01-01",
            updated_at="2024-01-01"
        )

        with patch.object(adapter.task_queue, 'update_task_status'):
            result = adapter.process_task(task)

        assert result["status"] == "completed"
        assert result["objective"] == 100
        assert result["is_valid"] is True

    def test_process_task_failure(self, adapter):
        """Test task processing failure"""
        task = OptimizationTask(
            task_id="test-123",
            problem_type="invalid",
            problem_data={},
            method="invalid_method",
            config={},
            status=TaskStatus.IN_PROGRESS,
            created_at="2024-01-01",
            updated_at="2024-01-01",
            retries=0
        )

        with patch.object(adapter.task_queue, 'update_task_status'):
            with patch.object(adapter.task_queue, 'retry_task'):
                result = adapter.process_task(task)

        assert result["status"] == "retrying"
        assert "error" in result

    def test_batch_submit(self, adapter):
        """Test batch task submission"""
        tasks = [
            {
                "problem_type": "qap",
                "problem_data": {"test": 1}
            },
            {
                "problem_type": "tsp",
                "problem_data": {"test": 2}
            }
        ]

        with patch.object(
            adapter,
            'submit_optimization_request',
            side_effect=[
                {"task_id": "task-1", "status": "submitted"},
                {"task_id": "task-2", "status": "submitted"}
            ]
        ):
            results = adapter.batch_submit("test-agent", tasks)

        assert len(results) == 2
        assert results[0]["task_id"] == "task-1"
        assert results[1]["task_id"] == "task-2"


class TestLibrexAgent:
    """Test ORCHEX-compliant agent"""

    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent.config.agent_id == "test-agent"
        assert agent.config.max_tasks == 5
        assert agent.current_workload == 0

    def test_can_accept_task(self, agent):
        """Test task acceptance logic"""
        # Should accept optimization tasks
        task = {"task_type": "optimization", "problem_type": "qap"}
        assert agent.can_accept_task(task) is True

        # Should reject non-optimization tasks
        task = {"task_type": "research", "problem_type": "unknown"}
        assert agent.can_accept_task(task) is False

        # Should reject when at capacity
        agent.current_workload = 5
        task = {"task_type": "optimization"}
        assert agent.can_accept_task(task) is False

    def test_execute_task(self, agent):
        """Test task execution"""
        task = {
            "task_id": "test-task-123",
            "task_type": "optimization",
            "problem_type": "qap",
            "problem_data": {"test": "data"},
            "method": "simulated_annealing"
        }

        with patch.object(
            agent.adapter,
            'submit_optimization_request',
            return_value={"task_id": "opt-123", "status": "submitted"}
        ):
            with patch.object(
                agent,
                '_wait_for_task_completion',
                return_value={"status": "completed", "objective": 100}
            ):
                result = agent.execute(task)

        assert result["status"] == "completed"
        assert result["agent_id"] == "test-agent"

    def test_to_features(self, agent):
        """Test feature vector generation"""
        features = agent.to_features()

        assert isinstance(features, np.ndarray)
        assert len(features) == 6
        assert features[0] == agent.config.skill_level

    def test_register_with_atlas(self, agent, redis_client):
        """Test ORCHEX registration"""
        agent.register_with_atlas()

        assert redis_client.hset.called
        assert redis_client.sadd.called
        assert redis_client.publish.called

    def test_deregister_from_atlas(self, agent, redis_client):
        """Test ORCHEX deregistration"""
        agent.deregister_from_atlas()

        assert redis_client.srem.called
        assert redis_client.hset.called
        assert redis_client.publish.called

    def test_get_agent_status(self, agent):
        """Test agent status reporting"""
        status = agent.get_agent_status()

        assert status["agent_id"] == "test-agent"
        assert status["current_workload"] == 0
        assert status["max_tasks"] == 5
        assert status["capacity_available"] is True


class TestIntegrationScenarios:
    """Test complete integration scenarios"""

    @pytest.mark.integration
    def test_end_to_end_optimization(self, config):
        """Test complete optimization workflow"""
        # This test requires Redis to be running
        try:
            r = redis.Redis.from_url(config.redis_url)
            r.ping()
        except:
            pytest.skip("Redis not available")

        # Initialize components
        adapter = ATLASOptimizationAdapter(config)
        agent = LibrexAgent(config)

        # Register agent
        agent.register_with_atlas()

        # Submit task
        result = adapter.submit_optimization_request(
            agent_id="integration-test",
            problem_type="qap",
            problem_data={
                "flow_matrix": [[0, 5], [5, 0]],
                "distance_matrix": [[0, 8], [8, 0]]
            },
            method="random_search"  # Fast method for testing
        )

        task_id = result["task_id"]
        assert task_id is not None

        # Process task
        task = adapter.task_queue.get_task(task_id)
        if task:
            optimization_result = adapter.process_task(task)
            assert optimization_result["status"] in ["completed", "failed"]

        # Cleanup
        agent.deregister_from_atlas()

    @pytest.mark.integration
    def test_concurrent_task_processing(self, config):
        """Test concurrent task processing"""
        # This test requires Redis
        try:
            r = redis.Redis.from_url(config.redis_url)
            r.ping()
        except:
            pytest.skip("Redis not available")

        adapter = ATLASOptimizationAdapter(config)

        # Submit multiple tasks
        task_ids = []
        for i in range(3):
            result = adapter.submit_optimization_request(
                agent_id=f"agent-{i}",
                problem_type="tsp",
                problem_data={
                    "coordinates": [[0, 0], [1, 1], [2, 0]]
                },
                method="random_search",
                priority=i
            )
            task_ids.append(result["task_id"])

        # Check all tasks are queued
        for task_id in task_ids:
            task = adapter.task_queue.get_task(task_id)
            assert task is not None


@pytest.mark.parametrize("problem_type,expected_method", [
    ("qap", "simulated_annealing"),
    ("tsp", "simulated_annealing"),
])
def test_method_selection_by_problem_type(adapter, problem_type, expected_method):
    """Test method selection based on problem type"""
    with patch.object(
        adapter.method_selector,
        'recommend_method',
        return_value=(expected_method, {}, 0.9)
    ):
        recommendation = adapter.get_method_recommendation(
            agent_id="test",
            problem_type=problem_type,
            problem_data={}
        )

    assert recommendation["recommendation"]["method"] == expected_method