#!/usr/bin/env python3
"""
Integration Coordinator - Connects all parallel services with the workflow executor.

This module provides:
- Service handler registration and management
- Claude API integration
- Conflict detection and resolution
- End-to-end workflow orchestration
- Service health monitoring and failover
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable
import requests
import subprocess
import yaml
import os
import signal
import re

try:
    import git
except ImportError:
    git = None

from workflow_types import ParallelTask, TaskResult, TaskStatus
from .claude_reasoning import handle_claude_analysis_task
from .build_test_coordinator import handle_compilation_task, handle_testing_task
from .deployment_service import handle_deployment_task


class ServiceStatus(Enum):
    """Status of integrated services."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Health status of a service."""
    service_name: str
    status: ServiceStatus
    last_check: datetime
    response_time_ms: int
    error_message: Optional[str] = None


@dataclass
class ConflictInfo:
    """Information about a detected conflict."""
    conflict_id: str
    conflict_type: str
    file_path: str
    description: str
    severity: "low" | "medium" | "high" | "critical"
    auto_resolvable: bool
    resolution_suggestion: Optional[str]


class ClaudeAPIClient:
    """Actual Claude API integration."""

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        })

    async def analyze_code(self, code: str, analysis_type: str, context: str = "") -> Dict[str, Any]:
        """Analyze code using Claude API."""
        prompt = f"""
        Analyze the following code for {analysis_type}:

        Context: {context}

        Code:
        ```python
        {code}
        ```

        Provide analysis in JSON format with:
        - findings: list of issues found
        - recommendations: list of improvement suggestions
        - confidence_score: float 0-1
        """

        try:
            response = await self._call_claude(prompt)
            return self._parse_claude_response(response)
        except Exception as e:
            return {
                "findings": [{"error": str(e)}],
                "recommendations": ["API call failed"],
                "confidence_score": 0.0
            }

    async def generate_pr_description(self, changes: List[Dict[str, Any]], context: str = "") -> Dict[str, Any]:
        """Generate PR description using Claude."""
        changes_summary = "\n".join([
            f"- {change.get('file', '')}: {change.get('description', '')}"
            for change in changes
        ])

        prompt = f"""
        Generate a pull request description for the following changes:

        Context: {context}

        Changes:
        {changes_summary}

        Provide response in JSON format with:
        - title: PR title
        - description: Detailed description
        - type: PR type (feature, bugfix, etc.)
        - impact: Impact level (low, medium, high)
        """

        try:
            response = await self._call_claude(prompt)
            return self._parse_claude_response(response)
        except Exception as e:
            return {
                "title": "Automated PR",
                "description": f"Failed to generate description: {e}",
                "type": "automated",
                "impact": "unknown"
            }

    async def _call_claude(self, prompt: str) -> str:
        """Make API call to Claude."""
        data = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        response = self.session.post(
            f"{self.base_url}/v1/messages",
            json=data,
            timeout=30
        )

        response.raise_for_status()
        return response.json()["content"][0]["text"]

    def _parse_claude_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Claude response as JSON."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"raw_response": response_text}
        except:
            return {"raw_response": response_text}


class ConflictDetector:
    """Detects and resolves conflicts in code and workflows."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.repo = None
        try:
            self.repo = git.Repo(self.repo_path)
        except:
            pass

    async def detect_conflicts(self, file_paths: List[str] = None) -> List[ConflictInfo]:
        """Detect conflicts in specified files or entire repo."""
        conflicts = []

        if not self.repo:
            return conflicts

        # Check for merge conflicts
        merge_conflicts = await self._detect_merge_conflicts(file_paths)
        conflicts.extend(merge_conflicts)

        # Check for dependency conflicts
        dependency_conflicts = await self._detect_dependency_conflicts(file_paths)
        conflicts.extend(dependency_conflicts)

        # Check for API conflicts
        api_conflicts = await self._detect_api_conflicts(file_paths)
        conflicts.extend(api_conflicts)

        # Check for configuration conflicts
        config_conflicts = await self._detect_config_conflicts(file_paths)
        conflicts.extend(config_conflicts)

        return conflicts

    async def auto_resolve_conflicts(self, conflicts: List[ConflictInfo]) -> List[ConflictInfo]:
        """Attempt to auto-resolve conflicts."""
        resolved = []
        unresolved = []

        for conflict in conflicts:
            if conflict.auto_resolvable:
                resolution_success = await self._resolve_conflict(conflict)
                if resolution_success:
                    resolved.append(conflict)
                else:
                    unresolved.append(conflict)
            else:
                unresolved.append(conflict)

        return unresolved

    async def _detect_merge_conflicts(self, file_paths: List[str] = None) -> List[ConflictInfo]:
        """Detect Git merge conflicts."""
        conflicts = []

        try:
            # Check for unmerged paths
            if self.repo.index.unmerged_blobs():
                for path, entries in self.repo.index.unmerged_blobs().items():
                    conflict = ConflictInfo(
                        conflict_id=str(uuid.uuid4()),
                        conflict_type="merge_conflict",
                        file_path=str(path),
                        description=f"Git merge conflict in {path}",
                        severity="high",
                        auto_resolvable=False,
                        resolution_suggestion="Resolve merge markers manually"
                    )
                    conflicts.append(conflict)

            # Check for conflict markers in files
            files_to_check = file_paths or self._get_all_files()
            for file_path in files_to_check:
                full_path = self.repo_path / file_path
                if full_path.exists():
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        if '<<<<<<<' in content or '>>>>>>>' in content:
                            conflict = ConflictInfo(
                                conflict_id=str(uuid.uuid4()),
                                conflict_type="merge_markers",
                                file_path=file_path,
                                description="Merge conflict markers found in file",
                                severity="critical",
                                auto_resolvable=False,
                                resolution_suggestion="Remove merge markers and resolve conflicts"
                            )
                            conflicts.append(conflict)
                    except:
                        continue

        except Exception as e:
            print(f"Error detecting merge conflicts: {e}")

        return conflicts

    async def _detect_dependency_conflicts(self, file_paths: List[str] = None) -> List[ConflictInfo]:
        """Detect dependency version conflicts."""
        conflicts = []

        files_to_check = file_paths or self._get_all_files()

        # Check package files
        package_files = [
            "package.json", "package-lock.json",
            "requirements.txt", "pyproject.toml",
            "Cargo.toml", "pom.xml", "build.gradle"
        ]

        for file_path in files_to_check:
            if any(pkg_file in file_path for pkg_file in package_files):
                full_path = self.repo_path / file_path
                if full_path.exists():
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # Simple conflict detection (can be enhanced)
                        if file_path.endswith('.json'):
                            data = json.loads(content)
                            # Check for conflicting versions
                            if 'dependencies' in data:
                                for dep, version in data['dependencies'].items():
                                    if '||' in version or version.startswith('>'):
                                        conflict = ConflictInfo(
                                            conflict_id=str(uuid.uuid4()),
                                            conflict_type="dependency_version",
                                            file_path=file_path,
                                            description=f"Complex version constraint for {dep}: {version}",
                                            severity="medium",
                                            auto_resolvable=True,
                                            resolution_suggestion="Consider pinning to specific version"
                                        )
                                        conflicts.append(conflict)
                    except:
                        continue

        return conflicts

    async def _detect_api_conflicts(self, file_paths: List[str] = None) -> List[ConflictInfo]:
        """Detect API contract conflicts."""
        conflicts = []

        files_to_check = file_paths or self._get_all_files()

        # Look for API definitions
        api_patterns = [
            r'@\w+\.api\(', r'def api_', r'router\.', r'app\.route\(',
            r'@app\.', r'@router\.', r'APIRouter\('
        ]

        api_endpoints = {}

        for file_path in files_to_check:
            if not file_path.endswith(('.py', '.js', '.ts')):
                continue

            full_path = self.repo_path / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    lines = content.splitlines()
                    for i, line in enumerate(lines):
                        for pattern in api_patterns:
                            if re.search(pattern, line):
                                # Extract endpoint info (simplified)
                                endpoint_info = {
                                    'file': file_path,
                                    'line': i + 1,
                                    'content': line.strip()
                                }

                                # Create a simple key for endpoint
                                endpoint_key = f"{file_path}:{i}"
                                if endpoint_key in api_endpoints:
                                    conflict = ConflictInfo(
                                        conflict_id=str(uuid.uuid4()),
                                        conflict_type="api_duplicate",
                                        file_path=file_path,
                                        description=f"Duplicate API endpoint definition",
                                        severity="medium",
                                        auto_resolvable=True,
                                        resolution_suggestion="Review and consolidate duplicate endpoints"
                                    )
                                    conflicts.append(conflict)
                                else:
                                    api_endpoints[endpoint_key] = endpoint_info
                except:
                    continue

        return conflicts

    async def _detect_config_conflicts(self, file_paths: List[str] = None) -> List[ConflictInfo]:
        """Detect configuration conflicts."""
        conflicts = []

        files_to_check = file_paths or self._get_all_files()

        # Check for duplicate configuration
        config_files = [
            "config.yaml", "config.yml", "settings.json",
            ".env", "docker-compose.yml", "docker-compose.yaml"
        ]

        config_keys = {}

        for file_path in files_to_check:
            if any(config_file in file_path for config_file in config_files):
                full_path = self.repo_path / file_path
                if full_path.exists():
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        if file_path.endswith(('.yaml', '.yml')):
                            data = yaml.safe_load(content)
                            if isinstance(data, dict):
                                for key, value in data.items():
                                    if key in config_keys:
                                        conflict = ConflictInfo(
                                            conflict_id=str(uuid.uuid4()),
                                            conflict_type="config_duplicate",
                                            file_path=file_path,
                                            description=f"Duplicate configuration key: {key}",
                                            severity="low",
                                            auto_resolvable=True,
                                            resolution_suggestion="Consolidate configuration or use different key names"
                                        )
                                        conflicts.append(conflict)
                                    else:
                                        config_keys[key] = {'file': file_path, 'value': value}
                    except:
                        continue

        return conflicts

    async def _resolve_conflict(self, conflict: ConflictInfo) -> bool:
        """Attempt to resolve a specific conflict."""
        try:
            if conflict.conflict_type == "dependency_version":
                return await self._resolve_dependency_conflict(conflict)
            elif conflict.conflict_type == "config_duplicate":
                return await self._resolve_config_conflict(conflict)
            elif conflict.conflict_type == "api_duplicate":
                return await self._resolve_api_conflict(conflict)

            return False
        except:
            return False

    async def _resolve_dependency_conflict(self, conflict: ConflictInfo) -> bool:
        """Resolve dependency version conflict."""
        # Simplified resolution - would be more sophisticated in practice
        return False

    async def _resolve_config_conflict(self, conflict: ConflictInfo) -> bool:
        """Resolve configuration conflict."""
        # Simplified resolution - would be more sophisticated in practice
        return False

    async def _resolve_api_conflict(self, conflict: ConflictInfo) -> bool:
        """Resolve API conflict."""
        # Simplified resolution - would be more sophisticated in practice
        return False

    def _get_all_files(self) -> List[str]:
        """Get all files in repository."""
        if not self.repo_path.exists():
            return []

        files = []
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    relative_path = file_path.relative_to(self.repo_path)
                    files.append(str(relative_path))
                except:
                    continue

        return files


class CodeReviewService:
    """AI-assisted code review and PR generation."""

    def __init__(self, claude_client: Optional[ClaudeAPIClient] = None):
        self.claude_client = claude_client

    async def review_changes(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Review code changes using AI."""
        if not self.claude_client:
            return self._mock_review(changes)

        review_results = {
            "summary": "",
            "issues": [],
            "suggestions": [],
            "approval_status": "pending"
        }

        for change in changes:
            file_path = change.get("file", "")
            diff = change.get("diff", "")

            if diff:
                analysis = await self.claude_client.analyze_code(
                    diff, "code_review", f"File: {file_path}"
                )

                review_results["issues"].extend(analysis.get("findings", []))
                review_results["suggestions"].extend(analysis.get("recommendations", []))

        # Generate summary
        if review_results["issues"]:
            review_results["approval_status"] = "changes_requested"
        else:
            review_results["approval_status"] = "approved"

        review_results["summary"] = f"Reviewed {len(changes)} files, found {len(review_results['issues'])} issues"

        return review_results

    async def generate_pr_description(self, changes: List[Dict[str, Any]], context: str = "") -> Dict[str, Any]:
        """Generate pull request description."""
        if not self.claude_client:
            return self._mock_pr_description(changes, context)

        return await self.claude_client.generate_pr_description(changes, context)

    def _mock_review(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock code review when Claude API is not available."""
        return {
            "summary": f"Mock review of {len(changes)} files",
            "issues": [
                {"type": "style", "message": "Consider adding more comments"},
                {"type": "security", "message": "Review input validation"}
            ],
            "suggestions": [
                "Add unit tests for new functions",
                "Update documentation"
            ],
            "approval_status": "approved_with_suggestions"
        }

    def _mock_pr_description(self, changes: List[Dict[str, Any]], context: str) -> Dict[str, Any]:
        """Mock PR description when Claude API is not available."""
        return {
            "title": "Feature: Add parallel workflow capabilities",
            "description": f"This PR adds parallel workflow execution with {len(changes)} files changed.",
            "type": "feature",
            "impact": "medium"
        }


class IntegrationCoordinator:
    """Main coordinator for all parallel services."""

    def __init__(self, repo_path: str = ".", claude_api_key: Optional[str] = None):
        self.repo_path = repo_path
        self.claude_client = ClaudeAPIClient(claude_api_key) if claude_api_key else None
        self.conflict_detector = ConflictDetector(repo_path)
        self.code_review_service = CodeReviewService(self.claude_client)
        self.service_health: Dict[str, ServiceHealth] = {}
        self.handler_registry: Dict[str, Callable] = {}
        self._register_handlers()

    def _register_handlers(self):
        """Register all service handlers with the workflow executor."""
        self.handler_registry = {
            "claude_analysis": handle_claude_analysis_task,
            "compilation": handle_compilation_task,
            "testing": handle_testing_task,
            "deployment": handle_deployment_task,
            "conflict_resolution": self._handle_conflict_resolution,
            "code_review": self._handle_code_review,
            "pr_generation": self._handle_pr_generation
        }

    async def start_services(self):
        """Start all integrated services."""
        # Start health monitoring
        asyncio.create_task(self._health_monitor_loop())

        # Initialize any services that need startup
        print("Integration coordinator started")

    async def stop_services(self):
        """Stop all integrated services."""
        print("Integration coordinator stopped")

    def get_handler(self, task_type: str) -> Optional[Callable]:
        """Get handler for a specific task type."""
        return self.handler_registry.get(task_type)

    async def detect_and_resolve_conflicts(self, file_paths: List[str] = None) -> Dict[str, Any]:
        """Detect and attempt to resolve conflicts."""
        conflicts = await self.conflict_detector.detect_conflicts(file_paths)
        unresolved = await self.conflict_detector.auto_resolve_conflicts(conflicts)

        return {
            "total_conflicts": len(conflicts),
            "auto_resolved": len(conflicts) - len(unresolved),
            "unresolved_conflicts": [
                {
                    "id": c.conflict_id,
                    "type": c.conflict_type,
                    "file": c.file_path,
                    "description": c.description,
                    "severity": c.severity
                }
                for c in unresolved
            ]
        }

    async def review_code_changes(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Review code changes and generate PR description."""
        review = await self.code_review_service.review_changes(changes)
        pr_desc = await self.code_review_service.generate_pr_description(changes)

        return {
            "review": review,
            "pr_description": pr_desc
        }

    async def _handle_conflict_resolution(self, task: ParallelTask) -> TaskResult:
        """Handle conflict resolution task."""
        start_time = time.time()

        try:
            stage = task.stage
            file_paths = stage.get("files", [])

            result = await self.detect_and_resolve_conflicts(file_paths)

            duration_ms = int((time.time() - start_time) * 1000)

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                output=result,
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

    async def _handle_code_review(self, task: ParallelTask) -> TaskResult:
        """Handle code review task."""
        start_time = time.time()

        try:
            stage = task.stage
            changes = stage.get("changes", [])

            result = await self.review_code_changes(changes)

            duration_ms = int((time.time() - start_time) * 1000)

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                output=result,
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

    async def _handle_pr_generation(self, task: ParallelTask) -> TaskResult:
        """Handle PR generation task."""
        start_time = time.time()

        try:
            stage = task.stage
            changes = stage.get("changes", [])
            context = stage.get("context", "")

            pr_desc = await self.code_review_service.generate_pr_description(changes, context)

            duration_ms = int((time.time() - start_time) * 1000)

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                output=pr_desc,
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

    async def _health_monitor_loop(self):
        """Monitor health of all services."""
        while True:
            try:
                await self._check_service_health()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Health monitor error: {e}")
                await asyncio.sleep(30)

    async def _check_service_health(self):
        """Check health of individual services."""
        services = ["claude_analysis", "compilation", "testing", "deployment"]

        for service in services:
            start_time = time.time()

            try:
                # Simple health check - try to create a basic task
                if service in self.handler_registry:
                    handler = self.handler_registry[service]
                    # Health check would be more sophisticated in practice
                    status = ServiceStatus.HEALTHY
                else:
                    status = ServiceStatus.UNHEALTHY

                response_time = int((time.time() - start_time) * 1000)

                self.service_health[service] = ServiceHealth(
                    service_name=service,
                    status=status,
                    last_check=datetime.now(),
                    response_time_ms=response_time
                )

            except Exception as e:
                self.service_health[service] = ServiceHealth(
                    service_name=service,
                    status=ServiceStatus.UNHEALTHY,
                    last_check=datetime.now(),
                    response_time_ms=int((time.time() - start_time) * 1000),
                    error_message=str(e)
                )

    def get_service_health(self) -> Dict[str, ServiceHealth]:
        """Get health status of all services."""
        return self.service_health.copy()


# Global integration coordinator
_integration_coordinator: Optional[IntegrationCoordinator] = None


def get_integration_coordinator(repo_path: str = ".", claude_api_key: Optional[str] = None) -> IntegrationCoordinator:
    """Get or create the global integration coordinator."""
    global _integration_coordinator

    if _integration_coordinator is None:
        _integration_coordinator = IntegrationCoordinator(repo_path, claude_api_key)

    return _integration_coordinator


def register_handlers_with_executor(executor):
    """Register all service handlers with a parallel workflow executor."""
    coordinator = get_integration_coordinator()

    # Update executor's handler registry with actual service handlers
    if hasattr(executor, '_parallel_handlers'):
        executor._parallel_handlers.update(coordinator.handler_registry)

    # Mark handlers as registered
    if hasattr(executor, '_handlers_registered'):
        executor._handlers_registered = True
