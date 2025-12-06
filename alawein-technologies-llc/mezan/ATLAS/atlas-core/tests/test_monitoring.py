"""
Tests for MEZAN Monitoring System.

Comprehensive test coverage for monitoring, service discovery,
health checks, and distributed tracing.
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import aiohttp
from aiohttp import web

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from atlas_core.monitoring import (
    ServiceStatus, MetricAggregation, ServiceInfo, HealthCheckResult,
    TraceSpan, MetricsAggregator, ServiceRegistry, HealthChecker,
    DistributedTracer, MonitoringServer, create_monitoring_stack
)


class TestMetricsAggregator:
    """Test metrics aggregation functionality."""

    def test_add_metric(self):
        """Test adding metrics."""
        aggregator = MetricsAggregator()

        # Add metrics
        aggregator.add_metric("cpu_usage", 75.0)
        aggregator.add_metric("cpu_usage", 80.0)
        aggregator.add_metric("cpu_usage", 85.0)

        # Add with tags
        aggregator.add_metric("request_count", 100, {"endpoint": "/api/v1"})
        aggregator.add_metric("request_count", 200, {"endpoint": "/api/v2"})

        # Verify metrics stored
        assert len(aggregator._metrics) == 3

    def test_aggregation_functions(self):
        """Test various aggregation functions."""
        aggregator = MetricsAggregator()

        # Add test data
        values = [10, 20, 30, 40, 50]
        for v in values:
            aggregator.add_metric("test_metric", v)

        # Test aggregations
        assert aggregator.aggregate("test_metric", MetricAggregation.AVG) == 30.0
        assert aggregator.aggregate("test_metric", MetricAggregation.SUM) == 150.0
        assert aggregator.aggregate("test_metric", MetricAggregation.MIN) == 10.0
        assert aggregator.aggregate("test_metric", MetricAggregation.MAX) == 50.0
        assert aggregator.aggregate("test_metric", MetricAggregation.COUNT) == 5

    def test_percentile_calculation(self):
        """Test percentile calculations."""
        aggregator = MetricsAggregator()

        # Add test data (1-100)
        for i in range(1, 101):
            aggregator.add_metric("percentile_test", i)

        # Test percentiles
        assert aggregator.aggregate("percentile_test", MetricAggregation.P50) == 50
        assert aggregator.aggregate("percentile_test", MetricAggregation.P95) == 95
        assert aggregator.aggregate("percentile_test", MetricAggregation.P99) == 99

    def test_time_series_retrieval(self):
        """Test time series data retrieval."""
        aggregator = MetricsAggregator()

        # Add metrics with timestamps
        base_time = time.time()
        for i in range(10):
            aggregator.add_metric("time_series", i * 10)
            time.sleep(0.01)  # Small delay

        # Get time series
        series = aggregator.get_time_series("time_series", duration=1)
        assert len(series) == 10
        assert all(t >= base_time for t, _ in series)

    def test_tagged_metrics(self):
        """Test metrics with tags."""
        aggregator = MetricsAggregator()

        # Add metrics with different tags
        aggregator.add_metric("api_latency", 0.1, {"endpoint": "/users", "method": "GET"})
        aggregator.add_metric("api_latency", 0.2, {"endpoint": "/users", "method": "POST"})
        aggregator.add_metric("api_latency", 0.15, {"endpoint": "/posts", "method": "GET"})

        # Aggregate by tags
        avg1 = aggregator.aggregate("api_latency", MetricAggregation.AVG,
                                   {"endpoint": "/users", "method": "GET"})
        avg2 = aggregator.aggregate("api_latency", MetricAggregation.AVG,
                                   {"endpoint": "/users", "method": "POST"})

        assert avg1 == 0.1
        assert avg2 == 0.2

    def test_window_size_limit(self):
        """Test that window size is respected."""
        aggregator = MetricsAggregator(window_size=5)

        # Add more metrics than window size
        for i in range(10):
            aggregator.add_metric("windowed", i)

        # Should only keep last 5
        series = aggregator.get_time_series("windowed")
        assert len(series) == 5
        assert [v for _, v in series] == [5, 6, 7, 8, 9]


class TestServiceRegistry:
    """Test service discovery and registration."""

    def test_service_registration(self):
        """Test registering services."""
        registry = ServiceRegistry()

        service = ServiceInfo(
            name="test-service",
            host="localhost",
            port=8080,
            version="1.0.0",
            status=ServiceStatus.HEALTHY,
            metadata={"region": "us-west"},
            last_heartbeat=time.time()
        )

        registry.register(service)
        assert registry.get_service("test-service") == service

    def test_service_unregistration(self):
        """Test unregistering services."""
        registry = ServiceRegistry()

        service = ServiceInfo(
            name="test-service",
            host="localhost",
            port=8080,
            version="1.0.0",
            status=ServiceStatus.HEALTHY,
            metadata={},
            last_heartbeat=time.time()
        )

        registry.register(service)
        registry.unregister("test-service")
        assert registry.get_service("test-service") is None

    def test_heartbeat_update(self):
        """Test service heartbeat updates."""
        registry = ServiceRegistry()

        service = ServiceInfo(
            name="test-service",
            host="localhost",
            port=8080,
            version="1.0.0",
            status=ServiceStatus.HEALTHY,
            metadata={},
            last_heartbeat=time.time() - 100
        )

        registry.register(service)
        old_heartbeat = service.last_heartbeat

        time.sleep(0.1)
        registry.heartbeat("test-service")

        updated_service = registry.get_service("test-service")
        assert updated_service.last_heartbeat > old_heartbeat

    def test_get_healthy_services(self):
        """Test filtering healthy services."""
        registry = ServiceRegistry()

        # Register services with different statuses
        for i, status in enumerate([ServiceStatus.HEALTHY, ServiceStatus.DEGRADED,
                                   ServiceStatus.UNHEALTHY, ServiceStatus.UNKNOWN]):
            service = ServiceInfo(
                name=f"service-{i}",
                host="localhost",
                port=8080 + i,
                version="1.0.0",
                status=status,
                metadata={},
                last_heartbeat=time.time()
            )
            registry.register(service)

        healthy = registry.get_healthy_services()
        assert len(healthy) == 2  # HEALTHY and DEGRADED

    def test_stale_service_cleanup(self):
        """Test that stale services are removed."""
        registry = ServiceRegistry(ttl_seconds=1)

        service = ServiceInfo(
            name="stale-service",
            host="localhost",
            port=8080,
            version="1.0.0",
            status=ServiceStatus.HEALTHY,
            metadata={},
            last_heartbeat=time.time() - 10  # Old heartbeat
        )

        registry.register(service)

        # Wait for cleanup
        time.sleep(2)

        assert registry.get_service("stale-service") is None


class TestHealthChecker:
    """Test health checking functionality."""

    @pytest.mark.asyncio
    async def test_basic_health_check(self):
        """Test basic service health check."""
        checker = HealthChecker()

        service = ServiceInfo(
            name="test-service",
            host="localhost",
            port=12345,  # Non-existent port
            version="1.0.0",
            status=ServiceStatus.HEALTHY,
            metadata={},
            last_heartbeat=time.time()
        )

        result = await checker.check_service(service)

        assert result.service == "test-service"
        assert result.status == ServiceStatus.UNHEALTHY
        assert result.latency_ms > 0
        assert "connectivity" in result.checks

    @pytest.mark.asyncio
    async def test_http_health_check(self):
        """Test HTTP endpoint health check."""
        checker = HealthChecker()

        # Mock HTTP response
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = asyncio.coroutine(lambda: {"checks": {"database": True}})

            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            service = ServiceInfo(
                name="test-service",
                host="localhost",
                port=8080,
                version="1.0.0",
                status=ServiceStatus.HEALTHY,
                metadata={},
                last_heartbeat=time.time(),
                health_check_url="http://localhost:8080/health"
            )

            result = await checker.check_service(service)

            assert result.status == ServiceStatus.HEALTHY
            assert result.checks["http_status"] == True
            assert result.checks["database"] == True

    @pytest.mark.asyncio
    async def test_check_all_services(self):
        """Test checking multiple services."""
        checker = HealthChecker()

        services = [
            ServiceInfo(
                name=f"service-{i}",
                host="localhost",
                port=8080 + i,
                version="1.0.0",
                status=ServiceStatus.HEALTHY,
                metadata={},
                last_heartbeat=time.time()
            )
            for i in range(3)
        ]

        results = await checker.check_all(services)
        assert len(results) == 3
        assert all(isinstance(r, HealthCheckResult) for r in results)


class TestDistributedTracer:
    """Test distributed tracing functionality."""

    def test_start_span(self):
        """Test starting a trace span."""
        tracer = DistributedTracer("test-service")

        span = tracer.start_span("test-operation", attributes={"user": "test"})

        assert span.operation == "test-operation"
        assert span.service == "test-service"
        assert span.status == "in_progress"
        assert span.attributes["user"] == "test"
        assert span.end_time is None

    def test_end_span(self):
        """Test ending a trace span."""
        tracer = DistributedTracer("test-service")

        span = tracer.start_span("test-operation")
        time.sleep(0.1)
        tracer.end_span(span, status="ok")

        assert span.status == "ok"
        assert span.end_time is not None
        assert span.end_time > span.start_time

    def test_add_span_event(self):
        """Test adding events to spans."""
        tracer = DistributedTracer("test-service")

        span = tracer.start_span("test-operation")
        tracer.add_event(span, "checkpoint", {"progress": 50})

        assert len(span.events) == 1
        assert span.events[0]["name"] == "checkpoint"
        assert span.events[0]["attributes"]["progress"] == 50

    def test_trace_retrieval(self):
        """Test retrieving trace spans."""
        tracer = DistributedTracer("test-service")

        # Create parent span
        parent = tracer.start_span("parent-operation")
        trace_id = parent.trace_id

        # Create child spans
        child1 = tracer.start_span("child-1", parent_span_id=parent.span_id)
        child1.trace_id = trace_id  # Set same trace ID
        tracer.traces[trace_id].append(child1)

        child2 = tracer.start_span("child-2", parent_span_id=parent.span_id)
        child2.trace_id = trace_id
        tracer.traces[trace_id].append(child2)

        # Get trace
        spans = tracer.get_trace(trace_id)
        assert len(spans) == 3
        assert spans[0] == parent

    def test_recent_traces(self):
        """Test getting recent trace IDs."""
        tracer = DistributedTracer("test-service")

        # Create multiple traces
        trace_ids = []
        for i in range(5):
            span = tracer.start_span(f"operation-{i}")
            trace_ids.append(span.trace_id)

        recent = tracer.get_recent_traces(limit=3)
        assert len(recent) == 3
        assert recent == trace_ids[-3:]


class TestMonitoringServer:
    """Test monitoring HTTP server."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check endpoint."""
        server = MonitoringServer()

        # Create mock request
        request = MagicMock()

        with patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:

            mock_disk.return_value.percent = 50.0
            mock_memory.return_value.percent = 60.0
            mock_cpu.return_value = 40.0

            response = await server.health_endpoint(request)

            data = json.loads(response.text)
            assert data["status"] == "healthy"
            assert data["checks"]["disk_space"] == True
            assert data["checks"]["memory"] == True
            assert data["checks"]["cpu"] == True

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        server = MonitoringServer()

        # Add some metrics
        server.metrics_aggregator.add_metric("test_metric", 42.0)

        request = MagicMock()
        response = await server.metrics_endpoint(request)

        assert "test_metric 42.0" in response.text
        assert "mezan_cpu_usage_percent" in response.text

    @pytest.mark.asyncio
    async def test_service_registration_endpoint(self):
        """Test service registration via HTTP."""
        server = MonitoringServer()

        # Create mock request
        request = MagicMock()
        request.json = asyncio.coroutine(lambda: {
            "name": "test-service",
            "host": "localhost",
            "port": 8080,
            "version": "1.0.0",
            "status": "healthy"
        })

        response = await server.register_service(request)

        data = json.loads(response.text)
        assert data["status"] == "registered"
        assert data["service"] == "test-service"

        # Verify service registered
        service = server.service_registry.get_service("test-service")
        assert service is not None

    @pytest.mark.asyncio
    async def test_heartbeat_endpoint(self):
        """Test service heartbeat endpoint."""
        server = MonitoringServer()

        # Register a service first
        service = ServiceInfo(
            name="test-service",
            host="localhost",
            port=8080,
            version="1.0.0",
            status=ServiceStatus.HEALTHY,
            metadata={},
            last_heartbeat=time.time() - 10
        )
        server.service_registry.register(service)

        # Send heartbeat
        request = MagicMock()
        request.json = asyncio.coroutine(lambda: {"service": "test-service"})

        response = await server.heartbeat_endpoint(request)

        data = json.loads(response.text)
        assert data["status"] == "ok"

        # Verify heartbeat updated
        updated = server.service_registry.get_service("test-service")
        assert updated.last_heartbeat > service.last_heartbeat


class TestIntegration:
    """Integration tests for monitoring stack."""

    def test_create_monitoring_stack(self):
        """Test creating complete monitoring stack."""
        stack = create_monitoring_stack("test-service")

        assert "server" in stack
        assert "metrics" in stack
        assert "registry" in stack
        assert "health" in stack
        assert "tracer" in stack

        # Verify components are connected
        assert stack["server"].metrics_aggregator == stack["metrics"]
        assert stack["server"].service_registry == stack["registry"]
        assert stack["server"].health_checker == stack["health"]

    def test_end_to_end_metrics_flow(self):
        """Test end-to-end metrics collection and aggregation."""
        stack = create_monitoring_stack("test-service")

        # Add metrics
        for i in range(10):
            stack["metrics"].add_metric("request_latency", i * 0.1)
            stack["metrics"].add_metric("error_count", 1 if i % 3 == 0 else 0)

        # Calculate aggregations
        avg_latency = stack["metrics"].aggregate("request_latency", MetricAggregation.AVG)
        total_errors = stack["metrics"].aggregate("error_count", MetricAggregation.SUM)
        p95_latency = stack["metrics"].aggregate("request_latency", MetricAggregation.P95)

        assert avg_latency == pytest.approx(0.45, 0.01)
        assert total_errors == 4
        assert p95_latency >= 0.8

    @pytest.mark.asyncio
    async def test_service_discovery_with_health_checks(self):
        """Test service discovery integrated with health checking."""
        stack = create_monitoring_stack("test-service")

        # Register services
        for i in range(3):
            service = ServiceInfo(
                name=f"service-{i}",
                host="localhost",
                port=8080 + i,
                version="1.0.0",
                status=ServiceStatus.HEALTHY,
                metadata={"zone": f"zone-{i}"},
                last_heartbeat=time.time()
            )
            stack["registry"].register(service)

        # Run health checks
        services = stack["registry"].get_all_services()
        results = await stack["health"].check_all(services)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, HealthCheckResult)
            assert result.latency_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])