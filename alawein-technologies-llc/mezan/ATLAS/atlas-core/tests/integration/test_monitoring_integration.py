"""
Monitoring Integration Tests for ORCHEX

Tests monitoring stack integration including metrics, logging, tracing, and alerting.
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
import prometheus_client
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
import structlog

from atlas_core.monitoring import (
    MetricsCollector,
    TracingManager,
    LogAggregator,
    AlertManager,
    HealthChecker,
    DashboardAPI,
    AnomalyDetector,
)
from atlas_core.engine import ATLASEngine


class TestMonitoringIntegration:
    """Test comprehensive monitoring integration."""

    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector instance."""
        return MetricsCollector(
            backends=["prometheus", "statsd", "cloudwatch"],
            collection_interval=1,
            aggregation_window=60,
        )

    @pytest.fixture
    def tracing_manager(self):
        """Create distributed tracing manager."""
        return TracingManager(
            backends=["jaeger", "zipkin", "datadog"],
            sample_rate=1.0,
            trace_all_requests=True,
        )

    @pytest.fixture
    def alert_manager(self):
        """Create alert manager instance."""
        return AlertManager(
            backends=["pagerduty", "slack", "email"],
            alert_rules_path="config/alerts.yaml",
            escalation_enabled=True,
        )

    @pytest.mark.integration
    async def test_metrics_collection(self, metrics_collector):
        """Test comprehensive metrics collection."""
        # Start metrics collection
        await metrics_collector.start()

        # Record various metrics
        metrics_collector.increment("requests.total", tags={"endpoint": "/api/workflow"})
        metrics_collector.histogram("request.latency", 0.125, tags={"endpoint": "/api/workflow"})
        metrics_collector.gauge("active.workflows", 42)
        metrics_collector.timer("task.duration", 5.5, tags={"task": "analysis"})

        # Allow collection
        await asyncio.sleep(2)

        # Verify metrics collected
        metrics = await metrics_collector.get_metrics()
        assert metrics["requests.total"] > 0
        assert metrics["request.latency"]["p95"] is not None
        assert metrics["active.workflows"] == 42
        assert metrics["task.duration"]["mean"] > 0

        # Test aggregation
        aggregated = await metrics_collector.get_aggregated_metrics(window="1m")
        assert "requests.total.rate" in aggregated
        assert aggregated["requests.total.rate"] > 0

    @pytest.mark.integration
    async def test_distributed_tracing(self, tracing_manager):
        """Test distributed tracing across services."""
        # Start tracing
        await tracing_manager.start()

        # Create root span
        with tracing_manager.start_span("workflow.execute") as root_span:
            root_span.set_attribute("workflow.id", "test_123")
            root_span.set_attribute("workflow.type", "research")

            # Create child spans
            with tracing_manager.start_span("agent.execute", parent=root_span) as agent_span:
                agent_span.set_attribute("agent.name", "LiteratureReview")
                agent_span.add_event("Processing started")

                # Simulate processing
                await asyncio.sleep(0.1)

                agent_span.add_event("Processing completed")
                agent_span.set_status("OK")

            # Create another child span
            with tracing_manager.start_span("database.query", parent=root_span) as db_span:
                db_span.set_attribute("db.statement", "SELECT * FROM workflows")
                db_span.set_attribute("db.rows_affected", 10)

        # Verify trace collection
        traces = await tracing_manager.get_traces(trace_id=root_span.trace_id)
        assert len(traces) > 0
        assert traces[0]["operation_name"] == "workflow.execute"
        assert len(traces[0]["spans"]) >= 3

    @pytest.mark.integration
    async def test_log_aggregation(self):
        """Test centralized log aggregation."""
        # Configure structured logging
        log_aggregator = LogAggregator(
            backends=["elasticsearch", "splunk", "cloudwatch"],
            buffer_size=1000,
            flush_interval=5,
        )

        await log_aggregator.start()

        # Create structured logger
        logger = structlog.get_logger()

        # Log various events
        logger.info("workflow.started", workflow_id="test_123", user="researcher")
        logger.warning("agent.slow_response", agent="DataAnalysis", latency=5.2)
        logger.error("database.connection_failed", error="Connection refused", retries=3)

        # Allow aggregation
        await asyncio.sleep(1)

        # Query logs
        logs = await log_aggregator.query(
            filters={"level": "error", "timerange": "last_1h"}, limit=100
        )
        assert len(logs) > 0
        assert logs[0]["level"] == "error"

        # Test log correlation
        correlated_logs = await log_aggregator.correlate_by_trace_id("trace_123")
        assert all(log.get("trace_id") == "trace_123" for log in correlated_logs)

    @pytest.mark.integration
    async def test_health_checking(self):
        """Test comprehensive health checking system."""
        health_checker = HealthChecker(
            check_interval=5,
            timeout=2,
            checks=[
                {"name": "database", "type": "tcp", "host": "localhost", "port": 5432},
                {"name": "redis", "type": "redis", "url": "redis://localhost:6379"},
                {"name": "api", "type": "http", "url": "http://localhost:8080/health"},
                {"name": "disk", "type": "disk", "path": "/", "threshold": 80},
                {"name": "memory", "type": "memory", "threshold": 90},
            ],
        )

        # Perform health checks
        health_status = await health_checker.check_all()

        # Verify health status structure
        assert "overall" in health_status
        assert "checks" in health_status
        assert health_status["overall"] in ["healthy", "degraded", "unhealthy"]

        # Verify individual checks
        for check in health_status["checks"]:
            assert "name" in check
            assert "status" in check
            assert "latency" in check
            assert check["status"] in ["healthy", "unhealthy"]

    @pytest.mark.integration
    async def test_alerting_system(self, alert_manager):
        """Test alerting and notification system."""
        # Define alert rules
        rules = [
            {
                "name": "high_error_rate",
                "condition": "error_rate > 0.05",
                "severity": "critical",
                "channels": ["pagerduty", "slack"],
            },
            {
                "name": "high_latency",
                "condition": "p95_latency > 1.0",
                "severity": "warning",
                "channels": ["slack"],
            },
            {
                "name": "low_disk_space",
                "condition": "disk_usage > 90",
                "severity": "critical",
                "channels": ["pagerduty", "email"],
            },
        ]

        await alert_manager.register_rules(rules)

        # Trigger alerts
        await alert_manager.evaluate_metrics(
            {"error_rate": 0.08, "p95_latency": 1.5, "disk_usage": 92}
        )

        # Verify alerts fired
        active_alerts = await alert_manager.get_active_alerts()
        assert len(active_alerts) == 3
        assert any(a["name"] == "high_error_rate" for a in active_alerts)

        # Test alert suppression
        await alert_manager.suppress_alert("high_latency", duration=300)
        suppressed = await alert_manager.get_suppressed_alerts()
        assert "high_latency" in [a["name"] for a in suppressed]

    @pytest.mark.integration
    async def test_performance_profiling(self):
        """Test performance profiling integration."""
        from atlas_core.monitoring import PerformanceProfiler

        profiler = PerformanceProfiler(
            enabled=True, sample_rate=0.1, profile_types=["cpu", "memory", "io", "network"]
        )

        # Start profiling
        profile_id = await profiler.start_profile("workflow_execution")

        # Simulate workload
        async def workload():
            data = []
            for i in range(1000):
                data.append(i * 2)
                await asyncio.sleep(0.001)
            return sum(data)

        result = await workload()

        # Stop profiling
        profile_data = await profiler.stop_profile(profile_id)

        # Verify profile data
        assert "cpu" in profile_data
        assert "memory" in profile_data
        assert profile_data["duration"] > 0
        assert profile_data["samples_collected"] > 0

    @pytest.mark.integration
    async def test_custom_metrics_and_dashboards(self):
        """Test custom metrics and dashboard creation."""
        dashboard_api = DashboardAPI(backend="grafana", api_key="test_key")

        # Create custom dashboard
        dashboard_config = {
            "title": "ORCHEX Workflow Monitoring",
            "panels": [
                {
                    "title": "Request Rate",
                    "type": "graph",
                    "metrics": ["requests.per.second"],
                    "aggregation": "sum",
                },
                {
                    "title": "Error Rate",
                    "type": "graph",
                    "metrics": ["errors.per.minute"],
                    "threshold": 10,
                },
                {
                    "title": "Active Workflows",
                    "type": "stat",
                    "metric": "workflows.active",
                    "color_thresholds": [10, 50, 100],
                },
                {
                    "title": "Agent Performance",
                    "type": "heatmap",
                    "metrics": ["agent.latency.by.type"],
                },
            ],
        }

        dashboard_id = await dashboard_api.create_dashboard(dashboard_config)
        assert dashboard_id is not None

        # Update dashboard
        await dashboard_api.add_panel(
            dashboard_id,
            {
                "title": "Resource Usage",
                "type": "gauge",
                "metrics": ["cpu.usage", "memory.usage"],
            },
        )

        # Verify dashboard
        dashboard = await dashboard_api.get_dashboard(dashboard_id)
        assert len(dashboard["panels"]) == 5

    @pytest.mark.integration
    async def test_anomaly_detection(self):
        """Test anomaly detection in metrics."""
        anomaly_detector = AnomalyDetector(
            algorithms=["isolation_forest", "lstm", "prophet"],
            sensitivity=0.95,
            training_window=7 * 24 * 60,  # 7 days in minutes
        )

        # Train on historical data
        historical_data = {
            "request_rate": [100 + i % 20 for i in range(1000)],
            "error_rate": [0.01 + (i % 100) * 0.001 for i in range(1000)],
            "latency": [0.1 + (i % 50) * 0.01 for i in range(1000)],
        }

        await anomaly_detector.train(historical_data)

        # Detect anomalies in new data
        new_data = {
            "request_rate": [100, 105, 500, 110, 108],  # 500 is anomalous
            "error_rate": [0.01, 0.02, 0.15, 0.01, 0.02],  # 0.15 is anomalous
            "latency": [0.1, 0.12, 0.11, 2.5, 0.13],  # 2.5 is anomalous
        }

        anomalies = await anomaly_detector.detect(new_data)

        assert len(anomalies) > 0
        assert any(a["metric"] == "request_rate" and a["index"] == 2 for a in anomalies)
        assert any(a["metric"] == "error_rate" and a["index"] == 2 for a in anomalies)

    @pytest.mark.integration
    async def test_sla_monitoring(self):
        """Test SLA monitoring and reporting."""
        from atlas_core.monitoring import SLAMonitor

        sla_monitor = SLAMonitor(
            sla_configs=[
                {
                    "name": "API Availability",
                    "metric": "uptime",
                    "target": 99.9,
                    "window": "30d",
                },
                {
                    "name": "Response Time",
                    "metric": "p95_latency",
                    "target": 1.0,  # 1 second
                    "window": "24h",
                },
                {
                    "name": "Error Rate",
                    "metric": "error_percentage",
                    "target": 1.0,  # 1%
                    "window": "1h",
                },
            ]
        )

        # Check SLA status
        sla_status = await sla_monitor.check_all()

        for sla in sla_status:
            assert "name" in sla
            assert "current_value" in sla
            assert "target" in sla
            assert "compliance" in sla
            assert sla["compliance"] in [True, False]
            assert "error_budget_remaining" in sla

        # Generate SLA report
        report = await sla_monitor.generate_report(period="last_30d")
        assert "summary" in report
        assert "detailed_metrics" in report
        assert "violations" in report

    @pytest.mark.integration
    async def test_distributed_logging_correlation(self):
        """Test log correlation across distributed components."""
        from atlas_core.monitoring import DistributedLogger

        logger = DistributedLogger(
            service_name="ORCHEX-core", correlation_enabled=True, sampling_rate=1.0
        )

        # Create correlated log entries
        correlation_id = "workflow_abc123"
        trace_id = "trace_xyz789"

        with logger.correlation_context(correlation_id=correlation_id, trace_id=trace_id):
            logger.info("Workflow started", workflow_type="research")
            logger.info("Agent invoked", agent_name="LiteratureReview")
            logger.warning("Slow response", latency=3.5)
            logger.info("Workflow completed", duration=45.2)

        # Query correlated logs
        correlated_logs = await logger.get_correlated_logs(correlation_id)
        assert len(correlated_logs) == 4
        assert all(log["correlation_id"] == correlation_id for log in correlated_logs)
        assert all(log["trace_id"] == trace_id for log in correlated_logs)

    @pytest.mark.integration
    async def test_monitoring_data_export(self, metrics_collector):
        """Test exporting monitoring data for analysis."""
        # Collect metrics for a period
        for i in range(100):
            metrics_collector.increment("test.counter", tags={"iteration": i})
            metrics_collector.histogram("test.histogram", i * 0.1, tags={"iteration": i})
            await asyncio.sleep(0.01)

        # Export data in various formats
        exports = {}

        # Export as Prometheus format
        exports["prometheus"] = await metrics_collector.export(format="prometheus")
        assert "test_counter" in exports["prometheus"]
        assert "test_histogram" in exports["prometheus"]

        # Export as JSON
        exports["json"] = await metrics_collector.export(format="json")
        assert isinstance(exports["json"], dict)
        assert "metrics" in exports["json"]

        # Export as CSV
        exports["csv"] = await metrics_collector.export(format="csv")
        assert "test.counter" in exports["csv"]

        # Export to time series database
        export_result = await metrics_collector.export_to_tsdb(
            backend="influxdb", database="atlas_metrics", retention="30d"
        )
        assert export_result["success"] is True
        assert export_result["points_written"] > 0