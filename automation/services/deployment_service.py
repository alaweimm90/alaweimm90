#!/usr/bin/env python3
"""
Deployment Service - Async deployment with Docker integration.

This module provides:
- Docker container management and orchestration
- Multi-environment deployment pipelines
- Asynchronous deployment with progress monitoring
- Rollback capabilities and health checks
- Integration with parallel workflow executor
- Blue-green and canary deployment strategies
"""

import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import subprocess
import threading
import requests
import yaml
import os
import signal

from workflow_types import ParallelTask, TaskResult, TaskStatus


class DeploymentEnvironment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"


class ContainerStatus(Enum):
    """Docker container status."""
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    PENDING = "pending"
    UNKNOWN = "unknown"


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""
    name: str
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    docker_image: str
    container_port: int
    host_port: int
    environment_vars: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=list)
    networks: List[str] = field(default_factory=list)
    health_check_url: Optional[str] = None
    health_check_timeout: int = 60
    rollback_enabled: bool = True
    replicas: int = 1


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""
    deployment_id: str
    success: bool
    container_ids: List[str]
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    deployment_time_ms: int
    health_status: str
    endpoints: List[str]
    rollback_available: bool
    error_message: Optional[str] = None


@dataclass
class ContainerInfo:
    """Information about a Docker container."""
    container_id: str
    name: str
    image: str
    status: ContainerStatus
    ports: Dict[str, str]
    created_at: datetime
    health_status: str


class DockerManager:
    """Manages Docker containers and operations."""

    def __init__(self):
        self.active_containers: Dict[str, ContainerInfo] = {}
        self._lock = threading.Lock()

    async def pull_image(self, image: str, tag: str = "latest") -> bool:
        """Pull Docker image from registry."""
        cmd = ["docker", "pull", f"{image}:{tag}"]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return True
            else:
                print(f"Failed to pull image {image}:{tag}: {stderr.decode()}")
                return False

        except Exception as e:
            print(f"Error pulling image {image}:{tag}: {e}")
            return False

    async def create_container(self, config: DeploymentConfig) -> Optional[str]:
        """Create and start a Docker container."""
        container_name = f"{config.name}-{config.environment.value}-{uuid.uuid4().hex[:8]}"

        # Build docker run command
        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{config.host_port}:{config.container_port}"
        ]

        # Add environment variables
        for key, value in config.environment_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Add volumes
        for volume in config.volumes:
            cmd.extend(["-v", volume])

        # Add networks
        for network in config.networks:
            cmd.extend(["--network", network])

        # Add health check if specified
        if config.health_check_url:
            cmd.extend([
                "--health-cmd", f"curl -f {config.health_check_url} || exit 1",
                "--health-interval", "30s",
                "--health-timeout", "10s",
                "--health-retries", "3"
            ])

        # Add restart policy
        cmd.extend(["--restart", "unless-stopped"])

        # Add image
        cmd.append(config.docker_image)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                container_id = stdout.decode().strip()

                # Store container info
                container_info = ContainerInfo(
                    container_id=container_id,
                    name=container_name,
                    image=config.docker_image,
                    status=ContainerStatus.PENDING,
                    ports={f"{config.container_port}/tcp": f"0.0.0.0:{config.host_port}"},
                    created_at=datetime.now(),
                    health_status="starting"
                )

                with self._lock:
                    self.active_containers[container_id] = container_info

                return container_id
            else:
                print(f"Failed to create container: {stderr.decode()}")
                return None

        except Exception as e:
            print(f"Error creating container: {e}")
            return None

    async def start_container(self, container_id: str) -> bool:
        """Start a stopped container."""
        cmd = ["docker", "start", container_id]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await process.communicate()

            if process.returncode == 0:
                with self._lock:
                    if container_id in self.active_containers:
                        self.active_containers[container_id].status = ContainerStatus.RUNNING
                return True
            else:
                return False

        except Exception as e:
            print(f"Error starting container {container_id}: {e}")
            return False

    async def stop_container(self, container_id: str, timeout: int = 30) -> bool:
        """Stop a running container."""
        cmd = ["docker", "stop", "-t", str(timeout), container_id]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await process.communicate()

            if process.returncode == 0:
                with self._lock:
                    if container_id in self.active_containers:
                        self.active_containers[container_id].status = ContainerStatus.STOPPED
                return True
            else:
                return False

        except Exception as e:
            print(f"Error stopping container {container_id}: {e}")
            return False

    async def remove_container(self, container_id: str, force: bool = False) -> bool:
        """Remove a container."""
        cmd = ["docker", "rm"]
        if force:
            cmd.append("-f")
        cmd.append(container_id)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await process.communicate()

            if process.returncode == 0:
                with self._lock:
                    if container_id in self.active_containers:
                        del self.active_containers[container_id]
                return True
            else:
                return False

        except Exception as e:
            print(f"Error removing container {container_id}: {e}")
            return False

    async def get_container_status(self, container_id: str) -> Optional[ContainerStatus]:
        """Get container status."""
        cmd = ["docker", "inspect", "--format", "{{.State.Status}}", container_id]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                status_str = stdout.decode().strip()
                status_map = {
                    "running": ContainerStatus.RUNNING,
                    "exited": ContainerStatus.STOPPED,
                    "failed": ContainerStatus.FAILED,
                    "created": ContainerStatus.PENDING
                }

                status = status_map.get(status_str, ContainerStatus.UNKNOWN)

                with self._lock:
                    if container_id in self.active_containers:
                        self.active_containers[container_id].status = status

                return status
            else:
                return ContainerStatus.UNKNOWN

        except Exception as e:
            print(f"Error getting container status {container_id}: {e}")
            return ContainerStatus.UNKNOWN

    async def get_container_logs(self, container_id: str, lines: int = 100) -> str:
        """Get container logs."""
        cmd = ["docker", "logs", "--tail", str(lines), container_id]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return stdout.decode()
            else:
                return stderr.decode()

        except Exception as e:
            return f"Error getting logs: {e}"

    async def execute_in_container(self, container_id: str, command: List[str]) -> Tuple[str, str, int]:
        """Execute command in container."""
        cmd = ["docker", "exec"] + [container_id] + command

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            return stdout.decode(), stderr.decode(), process.returncode

        except Exception as e:
            return "", str(e), 1


class DeploymentService:
    """Main deployment service with async capabilities."""

    def __init__(self):
        self.docker_manager = DockerManager()
        self.active_deployments: Dict[str, DeploymentResult] = {}
        self.deployment_history: List[DeploymentResult] = []
        self._lock = threading.Lock()

    async def deploy(
        self,
        config: DeploymentConfig,
        progress_callback: Optional[callable] = None
    ) -> DeploymentResult:
        """
        Deploy application using specified strategy.

        Args:
            config: Deployment configuration
            progress_callback: Optional progress callback

        Returns:
            Deployment result
        """
        deployment_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            if progress_callback:
                progress_callback("deployment_start", {
                    "deployment_id": deployment_id,
                    "environment": config.environment.value,
                    "strategy": config.strategy.value
                })

            # Pull Docker image
            if progress_callback:
                progress_callback("pulling_image", {"image": config.docker_image})

            image_success = await self.docker_manager.pull_image(config.docker_image)
            if not image_success:
                raise Exception(f"Failed to pull image: {config.docker_image}")

            # Deploy based on strategy
            if config.strategy == DeploymentStrategy.ROLLING:
                container_ids = await self._rolling_deployment(config, progress_callback)
            elif config.strategy == DeploymentStrategy.BLUE_GREEN:
                container_ids = await self._blue_green_deployment(config, progress_callback)
            elif config.strategy == DeploymentStrategy.CANARY:
                container_ids = await self._canary_deployment(config, progress_callback)
            elif config.strategy == DeploymentStrategy.RECREATE:
                container_ids = await self._recreate_deployment(config, progress_callback)
            else:
                raise Exception(f"Unsupported deployment strategy: {config.strategy}")

            # Wait for health check
            health_status = "unknown"
            if config.health_check_url:
                if progress_callback:
                    progress_callback("health_check", {"timeout": config.health_check_timeout})

                health_status = await self._wait_for_health_check(
                    container_ids, config.health_check_url, config.health_check_timeout
                )

            # Create endpoints
            endpoints = []
            for container_id in container_ids:
                container_info = self.docker_manager.active_containers.get(container_id)
                if container_info:
                    for port_mapping in container_info.ports.values():
                        if ":" in port_mapping:
                            host_port = port_mapping.split(":")[1]
                            endpoints.append(f"http://localhost:{host_port}")

            deployment_time_ms = int((time.time() - start_time) * 1000)

            result = DeploymentResult(
                deployment_id=deployment_id,
                success=len(container_ids) > 0,
                container_ids=container_ids,
                environment=config.environment,
                strategy=config.strategy,
                deployment_time_ms=deployment_time_ms,
                health_status=health_status,
                endpoints=endpoints,
                rollback_available=config.rollback_enabled and len(container_ids) > 0
            )

            with self._lock:
                self.active_deployments[deployment_id] = result
                self.deployment_history.append(result)

            if progress_callback:
                progress_callback("deployment_complete", result)

            return result

        except Exception as e:
            deployment_time_ms = int((time.time() - start_time) * 1000)

            error_result = DeploymentResult(
                deployment_id=deployment_id,
                success=False,
                container_ids=[],
                environment=config.environment,
                strategy=config.strategy,
                deployment_time_ms=deployment_time_ms,
                health_status="failed",
                endpoints=[],
                rollback_available=False,
                error_message=str(e)
            )

            with self._lock:
                self.active_deployments[deployment_id] = error_result
                self.deployment_history.append(error_result)

            if progress_callback:
                progress_callback("deployment_failed", {"error": str(e)})

            return error_result

    async def rollback(self, deployment_id: str, progress_callback: Optional[callable] = None) -> bool:
        """
        Rollback deployment to previous version.

        Args:
            deployment_id: Deployment ID to rollback
            progress_callback: Optional progress callback

        Returns:
            True if rollback successful
        """
        with self._lock:
            if deployment_id not in self.active_deployments:
                return False

            deployment = self.active_deployments[deployment_id]

            if not deployment.rollback_available:
                return False

        try:
            if progress_callback:
                progress_callback("rollback_start", {"deployment_id": deployment_id})

            # Stop current containers
            for container_id in deployment.container_ids:
                await self.docker_manager.stop_container(container_id)

            # Find previous deployment for same environment
            previous_deployment = self._find_previous_deployment(deployment.environment)

            if previous_deployment:
                # Restart previous containers
                for container_id in previous_deployment.container_ids:
                    await self.docker_manager.start_container(container_id)

                if progress_callback:
                    progress_callback("rollback_complete", {
                        "deployment_id": deployment_id,
                        "previous_deployment": previous_deployment.deployment_id
                    })

                return True
            else:
                if progress_callback:
                    progress_callback("rollback_failed", {"reason": "No previous deployment found"})
                return False

        except Exception as e:
            if progress_callback:
                progress_callback("rollback_failed", {"error": str(e)})
            return False

    async def _rolling_deployment(self, config: DeploymentConfig, progress_callback: Optional[callable]) -> List[str]:
        """Execute rolling deployment strategy."""
        container_ids = []

        for i in range(config.replicas):
            if progress_callback:
                progress_callback("rolling_update", {"replica": i + 1, "total": config.replicas})

            container_id = await self.docker_manager.create_container(config)
            if container_id:
                container_ids.append(container_id)

                # Wait for container to be healthy before creating next one
                if config.health_check_url:
                    await asyncio.sleep(5)  # Brief wait for startup
                    health = await self._wait_for_health_check([container_id], config.health_check_url, 30)
                    if health != "healthy":
                        print(f"Container {container_id} failed health check")

        return container_ids

    async def _blue_green_deployment(self, config: DeploymentConfig, progress_callback: Optional[callable]) -> List[str]:
        """Execute blue-green deployment strategy."""
        if progress_callback:
            progress_callback("blue_green_setup", {"environment": config.environment.value})

        # Create new containers (green)
        green_containers = []
        for i in range(config.replicas):
            container_id = await self.docker_manager.create_container(config)
            if container_id:
                green_containers.append(container_id)

        # Wait for green containers to be healthy
        if config.health_check_url and green_containers:
            if progress_callback:
                progress_callback("blue_green_health_check", {"containers": len(green_containers)})

            health = await self._wait_for_health_check(green_containers, config.health_check_url, config.health_check_timeout)

            if health == "healthy":
                # Switch traffic to green (in real implementation, this would update load balancer)
                if progress_callback:
                    progress_callback("blue_green_switch", {"active_containers": len(green_containers)})

                # Stop old containers (blue) - would be identified by label or naming convention
                # For now, just return green containers
                return green_containers
            else:
                # Health check failed, clean up green containers
                for container_id in green_containers:
                    await self.docker_manager.remove_container(container_id, force=True)

                raise Exception("Blue-green deployment failed health check")

        return green_containers

    async def _canary_deployment(self, config: DeploymentConfig, progress_callback: Optional[callable]) -> List[str]:
        """Execute canary deployment strategy."""
        # Start with small number of canary instances
        canary_count = max(1, config.replicas // 4)

        if progress_callback:
            progress_callback("canary_start", {"canary_instances": canary_count})

        canary_containers = []
        for i in range(canary_count):
            container_id = await self.docker_manager.create_container(config)
            if container_id:
                canary_containers.append(container_id)

        # Wait for canary health check
        if config.health_check_url and canary_containers:
            if progress_callback:
                progress_callback("canary_health_check", {"containers": len(canary_containers)})

            health = await self._wait_for_health_check(canary_containers, config.health_check_url, config.health_check_timeout)

            if health == "healthy":
                # Gradually increase canary instances
                remaining_containers = []
                for i in range(canary_count, config.replicas):
                    if progress_callback:
                        progress_callback("canary_scale", {"instance": i + 1, "total": config.replicas})

                    container_id = await self.docker_manager.create_container(config)
                    if container_id:
                        remaining_containers.append(container_id)

                canary_containers.extend(remaining_containers)
                return canary_containers
            else:
                # Canary failed, clean up
                for container_id in canary_containers:
                    await self.docker_manager.remove_container(container_id, force=True)

                raise Exception("Canary deployment failed health check")

        return canary_containers

    async def _recreate_deployment(self, config: DeploymentConfig, progress_callback: Optional[callable]) -> List[str]:
        """Execute recreate deployment strategy."""
        if progress_callback:
            progress_callback("recreate_start", {"replicas": config.replicas})

        # Stop all existing containers for this deployment
        # In real implementation, would identify by labels/naming

        # Create new containers
        container_ids = []
        for i in range(config.replicas):
            if progress_callback:
                progress_callback("recreate_replica", {"replica": i + 1, "total": config.replicas})

            container_id = await self.docker_manager.create_container(config)
            if container_id:
                container_ids.append(container_id)

        return container_ids

    async def _wait_for_health_check(self, container_ids: List[str], health_url: str, timeout: int) -> str:
        """Wait for containers to pass health check."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_healthy = True

            for container_id in container_ids:
                status = await self.docker_manager.get_container_status(container_id)
                if status != ContainerStatus.RUNNING:
                    all_healthy = False
                    break

            if all_healthy:
                # Check HTTP health endpoint
                try:
                    response = requests.get(health_url, timeout=5)
                    if response.status_code == 200:
                        return "healthy"
                except:
                    pass

            await asyncio.sleep(2)

        return "unhealthy"

    def _find_previous_deployment(self, environment: DeploymentEnvironment) -> Optional[DeploymentResult]:
        """Find previous successful deployment for environment."""
        with self._lock:
            for deployment in reversed(self.deployment_history[:-1]):  # Exclude current
                if (deployment.environment == environment and
                    deployment.success and
                    deployment.container_ids):
                    return deployment
        return None

    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get deployment status."""
        with self._lock:
            return self.active_deployments.get(deployment_id)

    def get_active_deployments(self) -> List[DeploymentResult]:
        """Get all active deployments."""
        with self._lock:
            return list(self.active_deployments.values())

    def get_deployment_history(self, limit: int = 50) -> List[DeploymentResult]:
        """Get deployment history."""
        with self._lock:
            return self.deployment_history[-limit:]


# Integration with ParallelWorkflowExecutor
class DeploymentHandler:
    """Handler for integrating deployment service with parallel workflows."""

    def __init__(self):
        self.deployment_service = DeploymentService()

    async def handle_deployment_task(self, task: ParallelTask) -> TaskResult:
        """Handle a deployment task from the parallel executor."""
        start_time = time.time()

        try:
            stage = task.stage
            context = task.context

            # Extract deployment configuration
            config = DeploymentConfig(
                name=stage.get("name", "app"),
                environment=DeploymentEnvironment(stage.get("environment", "staging")),
                strategy=DeploymentStrategy(stage.get("strategy", "rolling")),
                docker_image=stage.get("docker_image", context.inputs.get("docker_image", "nginx:latest")),
                container_port=stage.get("container_port", 80),
                host_port=stage.get("host_port", 8080),
                environment_vars=stage.get("environment_vars", {}),
                volumes=stage.get("volumes", []),
                networks=stage.get("networks", []),
                health_check_url=stage.get("health_check_url"),
                health_check_timeout=stage.get("health_check_timeout", 60),
                rollback_enabled=stage.get("rollback_enabled", True),
                replicas=stage.get("replicas", 1)
            )

            # Execute deployment
            result = await self.deployment_service.deploy(config)

            output = {
                "deployment_success": result.success,
                "deployment_id": result.deployment_id,
                "container_ids": result.container_ids,
                "environment": result.environment.value,
                "strategy": result.strategy.value,
                "deployment_time_ms": result.deployment_time_ms,
                "health_status": result.health_status,
                "endpoints": result.endpoints,
                "rollback_available": result.rollback_available
            }

            if result.error_message:
                output["error_message"] = result.error_message

            status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            duration_ms = int((time.time() - start_time) * 1000)

            return TaskResult(
                task_id=task.task_id,
                status=status,
                output=output,
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms
            )


# Global handler instance
_deployment_handler = DeploymentHandler()


async def handle_deployment_task(task: ParallelTask) -> TaskResult:
    """Global handler function for deployment tasks."""
    return await _deployment_handler.handle_deployment_task(task)
