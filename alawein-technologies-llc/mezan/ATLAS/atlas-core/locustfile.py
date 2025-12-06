"""
Locust configuration file for ORCHEX load testing.

Usage:
    # Web UI mode
    locust --host=http://localhost:8080

    # Headless mode
    locust --host=http://localhost:8080 --headless --users 100 --spawn-rate 10 --run-time 5m

    # Distributed mode (master)
    locust --host=http://localhost:8080 --master

    # Distributed mode (worker)
    locust --host=http://localhost:8080 --worker --master-host=localhost
"""

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import json
import random
import uuid
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ATLASUser(HttpUser):
    """Standard ORCHEX user behavior."""

    wait_time = between(1, 3)
    weight = 10  # 10x more likely than admin users

    def on_start(self):
        """Initialize user session."""
        # Authenticate
        response = self.client.post("/api/v1/auth/login", json={
            "username": f"user_{random.randint(1, 100)}",
            "password": "password123"
        }, catch_response=True)

        if response.status_code == 200:
            self.token = response.json().get("token")
            self.headers = {"Authorization": f"Bearer {self.token}"}
            response.success()
        else:
            response.failure(f"Login failed: {response.status_code}")
            self.headers = {}

        self.workflow_ids = []

    @task(30)
    def create_and_monitor_workflow(self):
        """Create a workflow and monitor its progress."""
        # Create workflow
        workflow_data = {
            "type": random.choice(["research", "analysis", "synthesis"]),
            "name": f"workflow_{uuid.uuid4().hex[:8]}",
            "priority": random.choice(["low", "medium", "high"]),
            "agents": random.sample([
                "LiteratureReview",
                "DataAnalysis",
                "Synthesis",
                "HypothesisGenerator",
                "ExperimentDesigner"
            ], k=3)
        }

        with self.client.post(
            "/api/v1/workflow/create",
            json=workflow_data,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 201:
                workflow_id = response.json().get("workflow_id")
                if workflow_id:
                    self.workflow_ids.append(workflow_id)
                    response.success()

                    # Monitor workflow
                    for _ in range(random.randint(2, 5)):
                        self.client.get(
                            f"/api/v1/workflow/{workflow_id}/status",
                            headers=self.headers,
                            name="/api/v1/workflow/[id]/status"
                        )
                        time.sleep(random.uniform(0.5, 2))
            else:
                response.failure(f"Failed to create workflow: {response.status_code}")

    @task(20)
    def list_workflows(self):
        """List user's workflows."""
        params = {
            "limit": random.choice([10, 25, 50, 100]),
            "offset": random.randint(0, 100),
            "status": random.choice(["all", "pending", "running", "completed"])
        }

        self.client.get(
            "/api/v1/workflow/list",
            headers=self.headers,
            params=params
        )

    @task(15)
    def get_workflow_details(self):
        """Get details of a specific workflow."""
        if self.workflow_ids:
            workflow_id = random.choice(self.workflow_ids)
            self.client.get(
                f"/api/v1/workflow/{workflow_id}",
                headers=self.headers,
                name="/api/v1/workflow/[id]"
            )

    @task(10)
    def search_workflows(self):
        """Search for workflows."""
        search_params = {
            "query": random.choice(["research", "analysis", "test", "production"]),
            "date_from": "2024-01-01",
            "limit": 50
        }

        self.client.get(
            "/api/v1/workflow/search",
            headers=self.headers,
            params=search_params
        )

    @task(10)
    def get_agents_list(self):
        """Get list of available agents."""
        self.client.get(
            "/api/v1/agents/list",
            headers=self.headers
        )

    @task(8)
    def invoke_agent(self):
        """Invoke a specific agent."""
        agent = random.choice([
            "LiteratureReview",
            "DataAnalysis",
            "Synthesis"
        ])

        task_data = {
            "input": {
                "data": f"test_data_{random.randint(1, 1000)}",
                "size": random.choice(["small", "medium", "large"])
            }
        }

        self.client.post(
            f"/api/v1/agents/{agent}/invoke",
            json=task_data,
            headers=self.headers,
            name="/api/v1/agents/[name]/invoke"
        )

    @task(5)
    def get_metrics(self):
        """Get system metrics."""
        self.client.get(
            "/api/v1/metrics",
            headers=self.headers
        )

    @task(2)
    def delete_workflow(self):
        """Delete old workflows."""
        if len(self.workflow_ids) > 20:
            workflow_id = self.workflow_ids.pop(0)
            self.client.delete(
                f"/api/v1/workflow/{workflow_id}",
                headers=self.headers,
                name="/api/v1/workflow/[id]"
            )


class AdminUser(HttpUser):
    """Admin user behavior - less frequent but important operations."""

    wait_time = between(5, 10)
    weight = 1  # Rare compared to regular users

    def on_start(self):
        """Admin login."""
        response = self.client.post("/api/v1/auth/admin/login", json={
            "username": "admin",
            "password": "admin_password",
            "mfa_code": "123456"
        })

        if response.status_code == 200:
            self.token = response.json().get("token")
            self.headers = {"Authorization": f"Bearer {self.token}"}

    @task(5)
    def view_system_health(self):
        """Check system health."""
        self.client.get("/api/v1/admin/health", headers=self.headers)

    @task(3)
    def view_all_workflows(self):
        """View all system workflows."""
        self.client.get(
            "/api/v1/admin/workflows",
            headers=self.headers,
            params={"limit": 100}
        )

    @task(2)
    def view_audit_logs(self):
        """Check audit logs."""
        self.client.get(
            "/api/v1/admin/audit",
            headers=self.headers,
            params={"hours": 24}
        )

    @task(1)
    def update_configuration(self):
        """Update system configuration."""
        config = {
            "max_workers": random.randint(8, 32),
            "cache_ttl": random.randint(60, 600)
        }

        self.client.put(
            "/api/v1/admin/config",
            json=config,
            headers=self.headers
        )


class HighThroughputUser(HttpUser):
    """User that generates high request rate for stress testing."""

    wait_time = between(0.1, 0.5)
    weight = 2

    def on_start(self):
        """Quick auth."""
        self.headers = {"X-API-Key": "test_api_key"}

    @task
    def rapid_status_checks(self):
        """Rapid status polling."""
        workflow_id = f"test_{random.randint(1, 1000)}"
        self.client.get(
            f"/api/v1/workflow/{workflow_id}/status",
            headers=self.headers,
            name="/api/v1/workflow/[id]/status"
        )


# Event handlers for custom metrics and reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test."""
    logger.info(f"Starting load test against {environment.host}")
    logger.info(f"Users: {environment.parsed_options.users if hasattr(environment, 'parsed_options') else 'N/A'}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Cleanup and report."""
    logger.info("Load test completed")

    # Print summary statistics
    stats = environment.stats
    logger.info(f"Total Requests: {stats.total.num_requests}")
    logger.info(f"Failure Rate: {stats.total.fail_ratio:.2%}")
    logger.info(f"Average Response Time: {stats.total.avg_response_time:.0f}ms")
    logger.info(f"Median Response Time: {stats.total.median_response_time:.0f}ms")

    if hasattr(stats.total, "get_response_time_percentile"):
        logger.info(f"95th Percentile: {stats.total.get_response_time_percentile(0.95):.0f}ms")
        logger.info(f"99th Percentile: {stats.total.get_response_time_percentile(0.99):.0f}ms")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Track individual requests."""
    # Log slow requests
    if response_time > 2000:
        logger.warning(f"Slow request: {name} took {response_time}ms")

    # Log errors
    if exception:
        logger.error(f"Request failed: {name} - {exception}")


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize Locust environment."""
    if isinstance(environment.runner, MasterRunner):
        logger.info("Running as Locust master node")

        # Custom reporting on master
        @environment.runner.greenlet.spawn_later(60)
        def periodic_report():
            """Print periodic reports."""
            while environment.runner.state == "running":
                stats = environment.runner.stats
                logger.info(f"Current RPS: {stats.total.current_rps:.0f}")
                logger.info(f"Current Users: {environment.runner.user_count}")
                time.sleep(60)


# Custom test shapes can be imported from load_test.py
try:
    from tests.performance.load_test import (
        StressTestShape,
        SpikeTestShape,
        EnduranceTestShape
    )

    # Uncomment to use a specific test shape
    # LoadTestShape = StressTestShape

except ImportError:
    pass


if __name__ == "__main__":
    import os
    import sys

    # Provide usage instructions when run directly
    print("ORCHEX Load Testing with Locust")
    print("=" * 50)
    print("\nUsage:")
    print("  locust --host=http://localhost:8080")
    print("\nOptions:")
    print("  --users NUMBER      Number of concurrent users")
    print("  --spawn-rate RATE   User spawn rate")
    print("  --run-time TIME     Test duration (e.g., 10m, 1h)")
    print("  --headless         Run without web UI")
    print("  --master           Run as master node")
    print("  --worker           Run as worker node")
    print("\nExample:")
    print("  locust --host=http://localhost:8080 --users 100 --spawn-rate 10")
    print("=" * 50)