#!/usr/bin/env python3
"""
Build and Test Coordinator - Parallel compilation and distributed testing.

This module provides:
- Multi-core compilation coordination
- Distributed unit test execution
- Resource-aware test worker scaling
- Integration with Docker for isolated builds
- Real-time progress monitoring and reporting
"""

import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import subprocess
import threading
import queue
import psutil
import os
import signal
import tempfile
import shutil

from workflow_types import ParallelTask, TaskResult, TaskStatus


class BuildType(Enum):
    """Types of builds supported."""
    DEBUG = "debug"
    RELEASE = "release"
    TESTING = "testing"
    PRODUCTION = "production"


class TestFramework(Enum):
    """Supported test frameworks."""
    PYTEST = "pytest"
    JEST = "jest"
    JUNIT = "junit"
    GO_TEST = "go_test"
    CARGO_TEST = "cargo_test"
    MAVEN = "maven"
    GRADLE = "gradle"


@dataclass
class BuildConfig:
    """Configuration for build processes."""
    build_type: BuildType
    target_directory: str
    output_directory: str
    parallel_jobs: int
    use_docker: bool = False
    docker_image: Optional[str] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    build_args: List[str] = field(default_factory=list)


@dataclass
class TestConfig:
    """Configuration for test execution."""
    test_framework: TestFramework
    test_directory: str
    test_pattern: str = "*test*"
    parallel_workers: int = 4
    coverage_enabled: bool = True
    timeout_seconds: int = 300
    test_args: List[str] = field(default_factory=list)


@dataclass
class BuildResult:
    """Result of a build process."""
    build_id: str
    success: bool
    artifacts: List[str]
    warnings: int
    errors: int
    build_time_ms: int
    log_file: Optional[str]
    docker_container_id: Optional[str]


@dataclass
class TestResult:
    """Result of test execution."""
    test_id: str
    framework: TestFramework
    tests_run: int
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    coverage_percent: Optional[float]
    test_time_ms: int
    failure_details: List[Dict[str, Any]]


class BuildTestCoordinator:
    """Coordinates parallel builds and distributed testing."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.build_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.test_executor = ProcessPoolExecutor(max_workers=max_workers)
        self.active_builds: Dict[str, subprocess.Popen] = {}
        self.active_tests: Dict[str, subprocess.Popen] = {}
        self.build_results: Dict[str, BuildResult] = {}
        self.test_results: Dict[str, TestResult] = {}
        self._lock = threading.Lock()

    def __del__(self):
        """Cleanup resources."""
        self.build_executor.shutdown(wait=True)
        self.test_executor.shutdown(wait=True)

        # Kill any running processes
        for process in self.active_builds.values():
            try:
                process.terminate()
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError, ProcessLookupError):
                process.kill()  # Force kill if graceful termination fails

        for process in self.active_tests.values():
            try:
                process.terminate()
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError, ProcessLookupError):
                process.kill()  # Force kill if graceful termination fails

    async def build_parallel(
        self,
        build_configs: List[BuildConfig],
        progress_callback: Optional[callable] = None
    ) -> List[BuildResult]:
        """
        Execute multiple builds in parallel.

        Args:
            build_configs: List of build configurations
            progress_callback: Optional callback for progress updates

        Returns:
            List of build results
        """
        if not build_configs:
            return []

        # Optimize worker count based on system resources
        optimal_workers = min(len(build_configs), self._get_optimal_build_workers())

        # Submit build tasks
        future_to_config = {}
        for config in build_configs:
            future = self.build_executor.submit(
                self._execute_build, config, progress_callback
            )
            future_to_config[future] = config

        # Collect results
        results = []
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)

                if progress_callback:
                    progress_callback("build_complete", result)

            except Exception as e:
                # Create failure result
                error_result = BuildResult(
                    build_id=str(uuid.uuid4()),
                    success=False,
                    artifacts=[],
                    warnings=0,
                    errors=1,
                    build_time_ms=0,
                    log_file=None,
                    docker_container_id=None
                )
                results.append(error_result)

                if progress_callback:
                    progress_callback("build_error", {"config": config, "error": str(e)})

        return results

    async def test_distributed(
        self,
        test_configs: List[TestConfig],
        progress_callback: Optional[callable] = None
    ) -> List[TestResult]:
        """
        Execute tests in distributed manner.

        Args:
            test_configs: List of test configurations
            progress_callback: Optional callback for progress updates

        Returns:
            List of test results
        """
        if not test_configs:
            return []

        # Group tests by framework for optimal distribution
        framework_groups = {}
        for config in test_configs:
            framework = config.test_framework
            if framework not in framework_groups:
                framework_groups[framework] = []
            framework_groups[framework].append(config)

        # Execute tests per framework
        all_results = []
        for framework, configs in framework_groups.items():
            # Optimize worker count for this framework
            optimal_workers = min(len(configs), self._get_optimal_test_workers())

            # Submit test tasks
            future_to_config = {}
            for config in configs:
                future = self.test_executor.submit(
                    self._execute_tests, config, progress_callback
                )
                future_to_config[future] = config

            # Collect results
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    all_results.append(result)

                    if progress_callback:
                        progress_callback("test_complete", result)

                except Exception as e:
                    # Create failure result
                    error_result = TestResult(
                        test_id=str(uuid.uuid4()),
                        framework=config.test_framework,
                        tests_run=0,
                        tests_passed=0,
                        tests_failed=0,
                        tests_skipped=0,
                        coverage_percent=0.0,
                        test_time_ms=0,
                        failure_details=[{"error": str(e)}]
                    )
                    all_results.append(error_result)

                    if progress_callback:
                        progress_callback("test_error", {"config": config, "error": str(e)})

        return all_results

    async def build_and_test_pipeline(
        self,
        build_config: BuildConfig,
        test_configs: List[TestConfig],
        progress_callback: Optional[callable] = None
    ) -> Tuple[BuildResult, List[TestResult]]:
        """
        Execute build followed by tests (CI/CD pipeline).

        Args:
            build_config: Build configuration
            test_configs: List of test configurations
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (build_result, test_results)
        """
        # Execute build first
        if progress_callback:
            progress_callback("pipeline_start", {"stage": "build"})

        build_results = await self.build_parallel([build_config], progress_callback)

        if not build_results or not build_results[0].success:
            # Build failed, skip tests
            if progress_callback:
                progress_callback("pipeline_failed", {"stage": "build"})

            return build_results[0] if build_results else None, []

        # Build succeeded, run tests
        if progress_callback:
            progress_callback("pipeline_stage", {"stage": "test"})

        test_results = await self.test_distributed(test_configs, progress_callback)

        if progress_callback:
            progress_callback("pipeline_complete", {
                "build_success": True,
                "test_results": len(test_results)
            })

        return build_results[0], test_results

    def _execute_build(
        self,
        config: BuildConfig,
        progress_callback: Optional[callable]
    ) -> BuildResult:
        """Execute a single build process."""
        build_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            if config.use_docker:
                result = self._execute_docker_build(config, build_id, start_time)
            else:
                result = self._execute_native_build(config, build_id, start_time)

            with self._lock:
                self.build_results[build_id] = result

            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            error_result = BuildResult(
                build_id=build_id,
                success=False,
                artifacts=[],
                warnings=0,
                errors=1,
                build_time_ms=duration_ms,
                log_file=None,
                docker_container_id=None
            )

            with self._lock:
                self.build_results[build_id] = error_result

            return error_result

    def _execute_docker_build(
        self,
        config: BuildConfig,
        build_id: str,
        start_time: float
    ) -> BuildResult:
        """Execute build inside Docker container."""
        docker_image = config.docker_image or "ubuntu:22.04"

        # Create Docker build command
        docker_cmd = [
            "docker", "run", "--rm",
            "--name", f"build-{build_id}",
            "-v", f"{config.target_directory}:/src",
            "-v", f"{config.output_directory}:/output",
            "-w", "/src"
        ]

        # Add environment variables
        for key, value in config.environment_vars.items():
            docker_cmd.extend(["-e", f"{key}={value}"])

        docker_cmd.extend([docker_image] + config.build_args)

        # Execute build
        log_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log')
        log_file.close()

        try:
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True
            )

            with self._lock:
                self.active_builds[build_id] = process

            # Monitor process and capture output
            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    output_lines.append(line.strip())

            process.wait()

            # Parse results
            success = process.returncode == 0
            artifacts = self._find_build_artifacts(config.output_directory)
            warnings, errors = self._parse_build_output(output_lines)

            duration_ms = int((time.time() - start_time) * 1000)

            # Write log file
            with open(log_file.name, 'w') as f:
                f.write('\n'.join(output_lines))

            result = BuildResult(
                build_id=build_id,
                success=success,
                artifacts=artifacts,
                warnings=warnings,
                errors=errors,
                build_time_ms=duration_ms,
                log_file=log_file.name,
                docker_container_id=f"build-{build_id}"
            )

            return result

        finally:
            with self._lock:
                if build_id in self.active_builds:
                    del self.active_builds[build_id]

    def _execute_native_build(
        self,
        config: BuildConfig,
        build_id: str,
        start_time: float
    ) -> BuildResult:
        """Execute native build process."""
        # Detect build system and create command
        build_cmd = self._create_build_command(config)

        if not build_cmd:
            raise ValueError(f"Unsupported build system for {config.target_directory}")

        # Set environment variables
        env = os.environ.copy()
        env.update(config.environment_vars)

        # Execute build
        log_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log')
        log_file.close()

        try:
            process = subprocess.Popen(
                build_cmd,
                cwd=config.target_directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True,
                env=env
            )

            with self._lock:
                self.active_builds[build_id] = process

            # Monitor process and capture output
            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    output_lines.append(line.strip())

            process.wait()

            # Parse results
            success = process.returncode == 0
            artifacts = self._find_build_artifacts(config.output_directory)
            warnings, errors = self._parse_build_output(output_lines)

            duration_ms = int((time.time() - start_time) * 1000)

            # Write log file
            with open(log_file.name, 'w') as f:
                f.write('\n'.join(output_lines))

            result = BuildResult(
                build_id=build_id,
                success=success,
                artifacts=artifacts,
                warnings=warnings,
                errors=errors,
                build_time_ms=duration_ms,
                log_file=log_file.name,
                docker_container_id=None
            )

            return result

        finally:
            with self._lock:
                if build_id in self.active_builds:
                    del self.active_builds[build_id]

    def _execute_tests(
        self,
        config: TestConfig,
        progress_callback: Optional[callable]
    ) -> TestResult:
        """Execute test suite."""
        test_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            result = self._run_test_framework(config, test_id, start_time)

            with self._lock:
                self.test_results[test_id] = result

            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            error_result = TestResult(
                test_id=test_id,
                framework=config.test_framework,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                coverage_percent=0.0,
                test_time_ms=duration_ms,
                failure_details=[{"error": str(e)}]
            )

            with self._lock:
                self.test_results[test_id] = error_result

            return error_result

    def _run_test_framework(
        self,
        config: TestConfig,
        test_id: str,
        start_time: float
    ) -> TestResult:
        """Run tests using specified framework."""
        test_cmd = self._create_test_command(config)

        if not test_cmd:
            raise ValueError(f"Unsupported test framework: {config.test_framework}")

        # Execute tests
        process = subprocess.Popen(
            test_cmd,
            cwd=config.test_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True
        )

        with self._lock:
            self.active_tests[test_id] = process

        try:
            stdout, stderr = process.communicate(timeout=config.timeout_seconds)

            # Parse test results
            test_stats = self._parse_test_output(config.test_framework, stdout, stderr)
            coverage = self._parse_coverage_output(config.test_framework, stdout, stderr) if config.coverage_enabled else None
            failures = self._parse_test_failures(config.test_framework, stdout, stderr)

            duration_ms = int((time.time() - start_time) * 1000)

            result = TestResult(
                test_id=test_id,
                framework=config.test_framework,
                tests_run=test_stats["run"],
                tests_passed=test_stats["passed"],
                tests_failed=test_stats["failed"],
                tests_skipped=test_stats["skipped"],
                coverage_percent=coverage,
                test_time_ms=duration_ms,
                failure_details=failures
            )

            return result

        finally:
            with self._lock:
                if test_id in self.active_tests:
                    del self.active_tests[test_id]

    def _create_build_command(self, config: BuildConfig) -> Optional[List[str]]:
        """Create build command based on detected build system."""
        target_dir = Path(config.target_directory)

        # Detect build system
        if (target_dir / "Cargo.toml").exists():
            # Rust/Cargo
            cmd = ["cargo", "build"]
            if config.build_type == BuildType.RELEASE:
                cmd.append("--release")
            if config.parallel_jobs > 1:
                cmd.extend(["-j", str(config.parallel_jobs)])
            return cmd

        elif (target_dir / "package.json").exists():
            # Node.js/npm
            if config.build_type == BuildType.RELEASE:
                return ["npm", "run", "build"]
            else:
                return ["npm", "run", "build:dev"]

        elif (target_dir / "Makefile").exists():
            # Make
            cmd = ["make", "-j", str(config.parallel_jobs)]
            if config.build_type == BuildType.RELEASE:
                cmd.append("release")
            return cmd

        elif (target_dir / "pom.xml").exists():
            # Maven
            cmd = ["mvn", "compile"]
            if config.build_type == BuildType.RELEASE:
                cmd.append("package")
            return cmd

        elif (target_dir / "build.gradle").exists() or (target_dir / "build.gradle.kts").exists():
            # Gradle
            cmd = ["./gradlew", "build"]
            if config.parallel_jobs > 1:
                cmd.extend(["--parallel", "--max-workers", str(config.parallel_jobs)])
            return cmd

        elif list(target_dir.glob("*.sln")) or list(target_dir.glob("*.csproj")):
            # .NET
            cmd = ["dotnet", "build"]
            if config.parallel_jobs > 1:
                cmd.extend(["--parallel", str(config.parallel_jobs)])
            if config.build_type == BuildType.RELEASE:
                cmd.extend(["--configuration", "Release"])
            return cmd

        # Python (setup.py or pyproject.toml)
        elif (target_dir / "setup.py").exists() or (target_dir / "pyproject.toml").exists():
            cmd = ["python", "-m", "build"]
            if config.build_type == BuildType.RELEASE:
                pass  # Default is release
            return cmd

        return None

    def _create_test_command(self, config: TestConfig) -> Optional[List[str]]:
        """Create test command based on framework."""
        cmd = []

        if config.test_framework == TestFramework.PYTEST:
            cmd = ["python", "-m", "pytest"]
            if config.coverage_enabled:
                cmd = ["python", "-m", "coverage", "run", "-m", "pytest"]
            if config.parallel_workers > 1:
                cmd.extend(["-n", str(config.parallel_workers)])
            cmd.extend(["-v", config.test_pattern])

        elif config.test_framework == TestFramework.JEST:
            cmd = ["npx", "jest"]
            if config.parallel_workers > 1:
                cmd.extend(["--maxWorkers", str(config.parallel_workers)])
            cmd.extend([config.test_pattern, "--verbose"])

        elif config.test_framework == TestFramework.JUNIT:
            cmd = ["mvn", "test"]
            if config.parallel_workers > 1:
                cmd.extend(["-T", str(config.parallel_workers)])

        elif config.test_framework == TestFramework.GO_TEST:
            cmd = ["go", "test", "./..."]
            if config.parallel_workers > 1:
                cmd.extend(["-parallel", str(config.parallel_workers)])
            if config.coverage_enabled:
                cmd.extend(["-coverprofile=coverage.out"])

        elif config.test_framework == TestFramework.CARGO_TEST:
            cmd = ["cargo", "test"]
            if config.parallel_workers > 1:
                cmd.extend(["-j", str(config.parallel_workers)])

        elif config.test_framework == TestFramework.MAVEN:
            cmd = ["mvn", "test"]
            if config.parallel_workers > 1:
                cmd.extend(["-T", str(config.parallel_workers)])

        elif config.test_framework == TestFramework.GRADLE:
            cmd = ["./gradlew", "test"]
            if config.parallel_workers > 1:
                cmd.extend(["--parallel", "--max-workers", str(config.parallel_workers)])

        # Add custom args
        cmd.extend(config.test_args)

        return cmd if cmd else None

    def _find_build_artifacts(self, output_dir: str) -> List[str]:
        """Find build artifacts in output directory."""
        artifacts = []
        output_path = Path(output_dir)

        if not output_path.exists():
            return artifacts

        # Common artifact patterns
        patterns = [
            "*.exe", "*.dll", "*.so", "*.dylib",  # Libraries/binaries
            "*.jar", "*.war", "*.ear",  # Java artifacts
            "*.whl", "*.egg",  # Python packages
            "*.tar.gz", "*.zip",  # Archives
            "dist/*", "build/*", "target/*"  # Build directories
        ]

        for pattern in patterns:
            artifacts.extend([str(f) for f in output_path.glob(pattern)])

        return artifacts

    def _parse_build_output(self, output_lines: List[str]) -> Tuple[int, int]:
        """Parse build output to count warnings and errors."""
        warnings = 0
        errors = 0

        for line in output_lines:
            line_lower = line.lower()
            if "warning" in line_lower or "warn" in line_lower:
                warnings += 1
            if "error" in line_lower:
                errors += 1

        return warnings, errors

    def _parse_test_output(self, framework: TestFramework, stdout: str, stderr: str) -> Dict[str, int]:
        """Parse test output to extract test statistics."""
        stats = {"run": 0, "passed": 0, "failed": 0, "skipped": 0}

        if framework == TestFramework.PYTEST:
            # Parse pytest summary
            import re
            summary_pattern = r"(\d+) passed, (\d+) failed, (\d+) skipped"
            match = re.search(summary_pattern, stdout)
            if match:
                stats["passed"] = int(match.group(1))
                stats["failed"] = int(match.group(2))
                stats["skipped"] = int(match.group(3))
                stats["run"] = stats["passed"] + stats["failed"] + stats["skipped"]

        elif framework == TestFramework.JEST:
            # Parse Jest output
            import re
            pattern = r"Tests:\s+(\d+) passed, (\d+) failed, (\d+) skipped"
            match = re.search(pattern, stdout)
            if match:
                stats["passed"] = int(match.group(1))
                stats["failed"] = int(match.group(2))
                stats["skipped"] = int(match.group(3))
                stats["run"] = stats["passed"] + stats["failed"] + stats["skipped"]

        # Add more parsers as needed

        return stats

    def _parse_coverage_output(self, framework: TestFramework, stdout: str, stderr: str) -> Optional[float]:
        """Parse coverage percentage from test output."""
        if framework == TestFramework.PYTEST:
            import re
            coverage_pattern = r"TOTAL\s+\d+\s+\d+\s+(\d+)%"
            match = re.search(coverage_pattern, stdout)
            if match:
                return float(match.group(1))

        elif framework == TestFramework.GO_TEST:
            import re
            coverage_pattern = r"coverage:\s+(\d+\.?\d*)%"
            match = re.search(coverage_pattern, stdout)
            if match:
                return float(match.group(1))

        return None

    def _parse_test_failures(self, framework: TestFramework, stdout: str, stderr: str) -> List[Dict[str, Any]]:
        """Parse test failure details."""
        failures = []

        # Simple failure parsing - can be enhanced per framework
        lines = stdout.split('\n')
        for i, line in enumerate(lines):
            if 'FAILED' in line or 'Error' in line:
                failures.append({
                    "line": i + 1,
                    "message": line.strip(),
                    "context": lines[max(0, i-2):i+3]  # Include context
                })

        return failures

    def _get_optimal_build_workers(self) -> int:
        """Calculate optimal number of build workers based on system resources."""
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)

        # Base on CPU cores, but limit by memory (each build worker needs ~1GB)
        cpu_based = cpu_count
        memory_based = int(memory_gb)

        return min(cpu_based, memory_based, self.max_workers)

    def _get_optimal_test_workers(self) -> int:
        """Calculate optimal number of test workers based on system resources."""
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)

        # Tests can be more parallel than builds
        cpu_based = cpu_count * 2
        memory_based = int(memory_gb / 0.5)  # Each test worker needs ~500MB

        return min(cpu_based, memory_based, self.max_workers * 2)

    def get_active_status(self) -> Dict[str, Any]:
        """Get status of active builds and tests."""
        with self._lock:
            return {
                "active_builds": len(self.active_builds),
                "active_tests": len(self.active_tests),
                "completed_builds": len(self.build_results),
                "completed_tests": len(self.test_results),
                "max_workers": self.max_workers
            }


# Integration with ParallelWorkflowExecutor
class BuildTestHandler:
    """Handler for integrating build/test coordination with parallel workflows."""

    def __init__(self, max_workers: int = 4):
        self.coordinator = BuildTestCoordinator(max_workers)

    async def handle_build_task(self, task: ParallelTask) -> TaskResult:
        """Handle a build task from the parallel executor."""
        start_time = time.time()

        try:
            stage = task.stage
            context = task.context

            # Extract build configuration
            build_config = BuildConfig(
                build_type=BuildType(stage.get("build_type", "debug")),
                target_directory=stage.get("target_directory", context.inputs.get("repo_path", ".")),
                output_directory=stage.get("output_directory", "./dist"),
                parallel_jobs=stage.get("parallel_jobs", self.coordinator._get_optimal_build_workers()),
                use_docker=stage.get("use_docker", False),
                docker_image=stage.get("docker_image"),
                environment_vars=stage.get("environment_vars", {}),
                build_args=stage.get("build_args", [])
            )

            # Execute build
            results = await self.coordinator.build_parallel([build_config])

            if results:
                result = results[0]
                output = {
                    "build_success": result.success,
                    "artifacts": result.artifacts,
                    "warnings": result.warnings,
                    "errors": result.errors,
                    "build_time_ms": result.build_time_ms,
                    "log_file": result.log_file
                }

                status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            else:
                output = {"error": "No build results"}
                status = TaskStatus.FAILED

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

    async def handle_test_task(self, task: ParallelTask) -> TaskResult:
        """Handle a test task from the parallel executor."""
        start_time = time.time()

        try:
            stage = task.stage
            context = task.context

            # Extract test configuration
            test_config = TestConfig(
                test_framework=TestFramework(stage.get("test_framework", "pytest")),
                test_directory=stage.get("test_directory", context.inputs.get("test_path", "./tests")),
                test_pattern=stage.get("test_pattern", "*test*"),
                parallel_workers=stage.get("parallel_workers", self.coordinator._get_optimal_test_workers()),
                coverage_enabled=stage.get("coverage_enabled", True),
                timeout_seconds=stage.get("timeout_seconds", 300),
                test_args=stage.get("test_args", [])
            )

            # Execute tests
            results = await self.coordinator.test_distributed([test_config])

            if results:
                result = results[0]
                output = {
                    "tests_run": result.tests_run,
                    "tests_passed": result.tests_passed,
                    "tests_failed": result.tests_failed,
                    "tests_skipped": result.tests_skipped,
                    "coverage_percent": result.coverage_percent,
                    "test_time_ms": result.test_time_ms,
                    "failure_details": result.failure_details[:5]  # Limit details
                }

                status = TaskStatus.COMPLETED if result.tests_failed == 0 else TaskStatus.FAILED
            else:
                output = {"error": "No test results"}
                status = TaskStatus.FAILED

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


# Global handler instances
_build_handler = BuildTestHandler()


async def handle_compilation_task(task: ParallelTask) -> TaskResult:
    """Global handler function for compilation tasks."""
    return await _build_handler.handle_build_task(task)


async def handle_testing_task(task: ParallelTask) -> TaskResult:
    """Global handler function for testing tasks."""
    return await _build_handler.handle_test_task(task)
