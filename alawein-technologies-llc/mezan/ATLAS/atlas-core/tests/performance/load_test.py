"""
Load Testing for ORCHEX using Locust

Comprehensive load testing scenarios for the ORCHEX system.
"""

from locust import HttpUser, task, between, events, LoadTestShape
from locust.runners import MasterRunner, WorkerRunner
import json
import random
import time
import uuid
from datetime import datetime
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ATLASUser(HttpUser):
    """Simulated ORCHEX user for load testing."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Called when a user starts."""
        # Authenticate user
        response = self.client.post("/api/v1/auth/login", json={
            "username": f"user_{random.randint(1, 100)}",
            "password": "test_password"
        })

        if response.status_code == 200:
            self.token = response.json().get("token")
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.headers = {}

        self.workflow_ids = []
        self.agent_names = [
            "LiteratureReviewAgent",
            "DataAnalysisAgent",
            "SynthesisAgent",
            "HypothesisGeneratorAgent",
            "ExperimentDesignerAgent"
        ]

    @task(10)
    def create_workflow(self):
        """Create a new workflow."""
        workflow_data = {
            "type": random.choice(["research", "analysis", "synthesis"]),
            "name": f"load_test_workflow_{uuid.uuid4().hex[:8]}",
            "priority": random.choice(["low", "medium", "high"]),
            "tasks": [
                {"agent": random.choice(self.agent_names), "task": "process"},
                {"agent": random.choice(self.agent_names), "task": "analyze"},
                {"agent": random.choice(self.agent_names), "task": "synthesize"}
            ]
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
            else:
                response.failure(f"Failed to create workflow: {response.status_code}")

    @task(20)
    def get_workflow_status(self):
        """Get workflow status."""
        if not self.workflow_ids:
            return

        workflow_id = random.choice(self.workflow_ids)
        self.client.get(
            f"/api/v1/workflow/{workflow_id}/status",
            headers=self.headers,
            name="/api/v1/workflow/[id]/status"
        )

    @task(15)
    def list_workflows(self):
        """List all workflows."""
        self.client.get(
            "/api/v1/workflow/list",
            headers=self.headers,
            params={"limit": 100, "offset": 0}
        )

    @task(5)
    def execute_agent_task(self):
        """Execute a single agent task."""
        agent_name = random.choice(self.agent_names)
        task_data = {
            "input": {
                "data": f"test_data_{random.randint(1, 1000)}",
                "size": random.choice(["small", "medium", "large"])
            }
        }

        self.client.post(
            f"/api/v1/agents/{agent_name}/invoke",
            json=task_data,
            headers=self.headers,
            name="/api/v1/agents/[name]/invoke"
        )

    @task(10)
    def get_metrics(self):
        """Get system metrics."""
        self.client.get(
            "/api/v1/metrics",
            headers=self.headers
        )

    @task(8)
    def search_workflows(self):
        """Search workflows."""
        search_params = {
            "query": random.choice(["research", "analysis", "synthesis", "test"]),
            "status": random.choice(["pending", "running", "completed", "all"]),
            "date_from": "2024-01-01",
            "date_to": datetime.now().isoformat()
        }

        self.client.get(
            "/api/v1/workflow/search",
            headers=self.headers,
            params=search_params
        )

    @task(3)
    def update_workflow(self):
        """Update workflow configuration."""
        if not self.workflow_ids:
            return

        workflow_id = random.choice(self.workflow_ids)
        update_data = {
            "priority": random.choice(["low", "medium", "high"]),
            "tags": [f"tag_{i}" for i in range(random.randint(1, 5))]
        }

        self.client.patch(
            f"/api/v1/workflow/{workflow_id}",
            json=update_data,
            headers=self.headers,
            name="/api/v1/workflow/[id]"
        )

    @task(2)
    def delete_workflow(self):
        """Delete a workflow."""
        if len(self.workflow_ids) > 10:  # Keep some workflows
            workflow_id = self.workflow_ids.pop(0)
            self.client.delete(
                f"/api/v1/workflow/{workflow_id}",
                headers=self.headers,
                name="/api/v1/workflow/[id]"
            )

    @task(5)
    def batch_operation(self):
        """Perform batch operations."""
        batch_request = {
            "operations": [
                {
                    "method": "GET",
                    "path": "/api/v1/agents/list"
                },
                {
                    "method": "GET",
                    "path": "/api/v1/workflow/list"
                },
                {
                    "method": "GET",
                    "path": "/api/v1/metrics"
                }
            ]
        }

        self.client.post(
            "/api/v1/batch",
            json=batch_request,
            headers=self.headers
        )


class AdminUser(HttpUser):
    """Simulated admin user with different behavior patterns."""

    wait_time = between(2, 5)
    weight = 1  # Lower weight than regular users

    def on_start(self):
        """Admin authentication."""
        response = self.client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin_password",
            "mfa_code": "123456"
        })

        if response.status_code == 200:
            self.token = response.json().get("token")
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.headers = {}

    @task(5)
    def view_system_health(self):
        """Check system health."""
        self.client.get("/api/v1/admin/health", headers=self.headers)

    @task(3)
    def view_audit_logs(self):
        """View audit logs."""
        self.client.get(
            "/api/v1/admin/audit",
            headers=self.headers,
            params={"limit": 100, "offset": 0}
        )

    @task(2)
    def manage_users(self):
        """User management operations."""
        self.client.get("/api/v1/admin/users", headers=self.headers)

    @task(1)
    def system_configuration(self):
        """Update system configuration."""
        config_data = {
            "max_workers": random.randint(4, 16),
            "cache_ttl": random.randint(60, 600),
            "log_level": random.choice(["DEBUG", "INFO", "WARNING"])
        }

        self.client.put(
            "/api/v1/admin/config",
            json=config_data,
            headers=self.headers
        )


class WebSocketUser(HttpUser):
    """User that primarily uses WebSocket connections."""

    wait_time = between(0.5, 2)

    def on_start(self):
        """Establish WebSocket connection."""
        # Note: Locust doesn't natively support WebSocket
        # This is a placeholder for WebSocket-like behavior
        self.workflow_subscriptions = []

    @task
    def subscribe_to_updates(self):
        """Subscribe to workflow updates via polling (simulating WebSocket)."""
        if self.workflow_subscriptions:
            workflow_id = random.choice(self.workflow_subscriptions)
            self.client.get(
                f"/api/v1/workflow/{workflow_id}/updates",
                name="/api/v1/workflow/[id]/updates"
            )


class StressTestShape(LoadTestShape):
    """Custom load shape for stress testing."""

    time_limit = 600  # 10 minutes
    spawn_rate = 10

    stages = [
        {"duration": 60, "users": 10, "spawn_rate": 2},    # Warm-up
        {"duration": 120, "users": 50, "spawn_rate": 5},   # Ramp-up
        {"duration": 180, "users": 100, "spawn_rate": 10}, # Sustained load
        {"duration": 60, "users": 200, "spawn_rate": 20},  # Peak load
        {"duration": 60, "users": 100, "spawn_rate": 10},  # Recovery
        {"duration": 60, "users": 50, "spawn_rate": 5},    # Ramp-down
        {"duration": 60, "users": 10, "spawn_rate": 2},    # Cool-down
    ]

    def tick(self):
        """Return tick data for custom load shape."""
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            return None

        current_time = 0
        for stage in self.stages:
            if current_time + stage["duration"] > run_time:
                return (stage["users"], stage["spawn_rate"])
            current_time += stage["duration"]

        return None


class SpikeTestShape(LoadTestShape):
    """Load shape for spike testing."""

    time_limit = 300  # 5 minutes

    def tick(self):
        """Generate spike load pattern."""
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            return None

        # Normal load with periodic spikes
        if run_time % 60 < 10:  # 10-second spike every minute
            return (500, 100)  # Spike to 500 users
        else:
            return (50, 10)  # Normal load of 50 users


class EnduranceTestShape(LoadTestShape):
    """Load shape for endurance/soak testing."""

    time_limit = 3600  # 1 hour
    target_users = 100

    def tick(self):
        """Maintain steady load for extended period."""
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            return None

        if run_time < 60:
            # Ramp up in first minute
            users = int((run_time / 60) * self.target_users)
            return (users, 10)
        else:
            # Maintain steady load
            return (self.target_users, 1)


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Handler for test start event."""
    logger.info("Load test started")
    logger.info(f"Target host: {environment.host}")

    # Initialize test metrics storage
    environment.stats.reset_all()


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Handler for test stop event."""
    logger.info("Load test completed")

    # Generate summary report
    stats = environment.stats
    logger.info(f"Total requests: {stats.total.num_requests}")
    logger.info(f"Failure rate: {stats.total.fail_ratio * 100:.2f}%")
    logger.info(f"Median response time: {stats.total.median_response_time}ms")
    logger.info(f"95th percentile: {stats.total.get_response_time_percentile(0.95)}ms")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Handler for each request."""
    if exception:
        logger.error(f"Request failed: {name} - {exception}")

    # Custom metrics collection
    if response_time > 5000:  # Log slow requests
        logger.warning(f"Slow request: {name} took {response_time}ms")


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Handler for Locust initialization."""
    if isinstance(environment.runner, MasterRunner):
        logger.info("Running as master node")
    elif isinstance(environment.runner, WorkerRunner):
        logger.info("Running as worker node")
    else:
        logger.info("Running as standalone")


# Custom failure criteria
def check_failure_criteria(environment):
    """Check if test should fail based on criteria."""
    stats = environment.stats

    # Fail if error rate > 5%
    if stats.total.fail_ratio > 0.05:
        logger.error("Test failed: Error rate exceeded 5%")
        environment.runner.quit()
        return False

    # Fail if 95th percentile > 2000ms
    p95 = stats.total.get_response_time_percentile(0.95)
    if p95 and p95 > 2000:
        logger.error("Test failed: 95th percentile exceeded 2000ms")
        environment.runner.quit()
        return False

    return True


# Register failure check
@events.heartbeat.add_listener
def on_heartbeat(**kwargs):
    """Periodic check of failure criteria."""
    # Note: In real implementation, you'd pass the environment
    pass


if __name__ == "__main__":
    # This file is typically run using the locust command
    # locust -f load_test.py --host=http://localhost:8080
    logger.info("Load test module loaded")
    logger.info("Run with: locust -f load_test.py --host=http://localhost:8080")