#!/usr/bin/env python3
"""
Parallel Workflow Executor - Extends WorkflowExecutor with async parallel capabilities.

This module provides:
- Parallel stage execution with resource monitoring
- Async task queue for background jobs
- Dynamic worker scaling based on system resources
- Integration with existing workflow definitions
- Background Claude reasoning for code analysis
- Parallel compilation and testing coordination
- Async deployment with Docker integration
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
from typing import Any, Callable, Dict, List, Optional, Union
import subprocess
import threading
import queue
import psutil
import yaml

# Import shared types to avoid circular imports
from workflow_types import WorkflowContext, TaskResult, TaskStatus, ParallelTask, TaskPriority
from executor import WorkflowExecutor as BaseWorkflowExecutor

AUTOMATION_PATH = Path(__file__).parent


class BackgroundProcessManager:
    """Manages long-running background processes like watch modes."""

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.process_status: Dict[str, Dict[str, Any]] = {}

    def start_process(self, process_id: str, command: str, cwd: str = None) -> bool:
        """Start a background process with improved security and output handling."""
        try:
            # Security: Validate and sanitize command
            if not self._validate_command(command):
                raise ValueError(f"Command validation failed: {command}")

            # Use shlex for safer command parsing instead of shell=True
            import shlex
            command_parts = shlex.split(command)

            # Start process with proper output handling for watch processes
            # Use subprocess.DEVNULL for watch processes to prevent buffer overflow
            process = subprocess.Popen(
                command_parts,
                cwd=cwd,
                stdout=subprocess.DEVNULL,  # Prevent buffer issues for long-running processes
                stderr=subprocess.DEVNULL,
                text=False
            )

            self.processes[process_id] = process
            self.process_status[process_id] = {
                "started": datetime.now(),
                "command": command,
                "status": "running",
                "pid": process.pid
            }

            return True
        except Exception as e:
            print(f"Failed to start process {process_id}: {e}")
            return False

    def _validate_command(self, command: str) -> bool:
        """Basic command validation for security."""
        # Block dangerous commands and characters
        dangerous_patterns = [
            'rm -rf', 'sudo', 'su', 'chmod 777', 'chown',
            'wget', 'curl', 'nc', 'netcat', '&&', '||', ';',
            '$(', '`', '>', '>>', '<'
        ]

        command_lower = command.lower()
        for pattern in dangerous_patterns:
            if pattern in command_lower:
                return False

        # Allow only specific safe commands for watch mode
        allowed_commands = ['npx', 'nodemon', 'sphinx-build', 'jest', 'tsc', 'vite', 'eslint']
        if not any(command.startswith(cmd) for cmd in allowed_commands):
            return False

        return True

    def stop_process(self, process_id: str) -> bool:
        """Stop a background process."""
        if process_id in self.processes:
            try:
                process = self.processes[process_id]
                process.terminate()
                process.wait(timeout=5)
                self.process_status[process_id]["status"] = "stopped"
                return True
            except subprocess.TimeoutExpired:
                process.kill()
                self.process_status[process_id]["status"] = "killed"
                return True
            except Exception as e:
                print(f"Failed to stop process {process_id}: {e}")
                return False
        return False

    def stop_all_processes(self):
        """Stop all background processes."""
        for process_id in list(self.processes.keys()):
            self.stop_process(process_id)

    def get_process_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific process."""
        if process_id in self.process_status:
            status = self.process_status[process_id].copy()
            if process_id in self.processes:
                process = self.processes[process_id]
                if process.poll() is not None:
                    status["status"] = "completed"
                    status["return_code"] = process.returncode
            return status
        return None

    def health_check(self) -> Dict[str, Any]:
        """Check health of all managed processes."""
        healthy = []
        unhealthy = []

        for process_id, process in self.processes.items():
            if process.poll() is None:
                healthy.append(process_id)
            else:
                unhealthy.append(process_id)

        return {
            "total_processes": len(self.processes),
            "healthy": len(healthy),
            "unhealthy": len(unhealthy),
            "healthy_processes": healthy,
            "unhealthy_processes": unhealthy
        }


@dataclass
class ResourceMetrics:
    """System resource metrics for dynamic scaling."""
    cpu_percent: float
    memory_percent: float
    available_cores: int
    load_average: float
    disk_io_percent: float
    network_io_active: bool


class ResourceMonitor:
    """Monitor system resources and provide scaling recommendations."""

    def __init__(self):
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history = 100
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self, interval: float = 1.0):
        """Start continuous resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            metrics = self.get_current_metrics()
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)
            time.sleep(interval)

    def get_current_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        cores = psutil.cpu_count()

        # Load average (Unix-like systems)
        try:
            load_avg = psutil.getloadavg()[0] / cores  # Normalize by core count
        except (AttributeError, IndexError):
            load_avg = cpu_percent / 100.0

        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            disk_io_percent = min((disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024 * 100), 100.0)  # Normalize to 100MB/s
        except:
            disk_io_percent = 0.0

        # Network I/O
        try:
            net_io = psutil.net_io_counters()
            network_io_active = (net_io.bytes_sent + net_io.bytes_recv) > 1024
        except:
            network_io_active = False

        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            available_cores=cores,
            load_average=load_avg,
            disk_io_percent=disk_io_percent,
            network_io_active=network_io_active
        )

    def get_optimal_worker_count(self, base_workers: int = 4) -> int:
        """Calculate optimal worker count based on current resources."""
        if not self.metrics_history:
            return base_workers

        metrics = self.metrics_history[-1]

        # Scale down if CPU or memory is high
        if metrics.cpu_percent > 80:
            return max(1, base_workers // 2)
        elif metrics.memory_percent > 85:
            return max(1, base_workers // 2)

        # Scale up if resources are available
        if metrics.cpu_percent < 30 and metrics.memory_percent < 50:
            return min(base_workers * 2, metrics.available_cores)

        return base_workers


class AsyncTaskQueue:
    """Priority queue for managing async background tasks."""

    def __init__(self, max_workers: int = 4):
        self.task_queue = queue.PriorityQueue()
        self.results: Dict[str, TaskResult] = {}
        self.max_workers = max_workers
        self.workers: List[threading.Thread] = []
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def start(self):
        """Start the task queue workers."""
        self.running = True
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"AsyncWorker-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def stop(self):
        """Stop the task queue and wait for workers to finish."""
        self.running = False
        self.executor.shutdown(wait=True)
        for worker in self.workers:
            worker.join(timeout=2.0)

    def enqueue_task(self, task: ParallelTask) -> str:
        """Add a task to the queue."""
        # Use negative priority for max-heap behavior
        priority_value = -task.priority.value
        self.task_queue.put((priority_value, time.time(), task))
        return task.task_id

    def get_result(self, task_id: str) -> Optional[TaskResult]:
        """Get the result of a completed task."""
        return self.results.get(task_id)

    def _worker_loop(self):
        """Main worker loop."""
        while self.running:
            try:
                # Get task with timeout
                priority, timestamp, task = self.task_queue.get(timeout=1.0)

                # Execute task in thread pool
                future = self.executor.submit(self._execute_task, task)

                # Store result when complete
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    self.results[task.task_id] = result
                except Exception as e:
                    error_result = TaskResult(
                        task_id=task.task_id,
                        status=TaskStatus.FAILED,
                        error=str(e)
                    )
                    self.results[task.task_id] = error_result

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")

    def _execute_task(self, task: ParallelTask) -> TaskResult:
        """Execute a single parallel task."""
        # This would delegate to the appropriate handler
        # For now, return a mock result
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            output={"parallel_executed": True, "stage": task.stage.get("name")}
        )


class ParallelWorkflowExecutor(BaseWorkflowExecutor):
    """Extended workflow executor with parallel capabilities."""

    def __init__(self, agent_handler: Optional[Callable] = None, max_workers: int = 4):
        super().__init__(agent_handler)
        self.resource_monitor = ResourceMonitor()
        self.task_queue = AsyncTaskQueue(max_workers)
        self.background_manager = BackgroundProcessManager()  # Add background process manager
        self.max_workers = max_workers
        self._parallel_handlers = self._init_parallel_handlers()
        self._handlers_registered = False

    def _ensure_handlers_registered(self):
        """Lazy initialization of handlers to avoid circular imports."""
        if not self._handlers_registered:
            try:
                # Try to register actual service handlers
                from services.integration_coordinator import register_handlers_with_executor
                register_handlers_with_executor(self)
                self._handlers_registered = True
                print("✅ Service handlers registered successfully")
            except ImportError as e:
                print(f"⚠️  Warning: Could not register service handlers: {e}")
                print("   Using fallback mock handlers for demo purposes")
                # Fallback to basic handlers is already set up
                self._handlers_registered = True

    async def _run_async_handler(self, handler, parallel_task):
        """Helper to run async handlers with proper event loop handling."""
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # We're in a running loop, create task and wait for it
            task = loop.create_task(handler(parallel_task))
            return await task
        except RuntimeError:
            # No running loop, use asyncio.run()
            return asyncio.run(handler(parallel_task))

    def _init_parallel_handlers(self) -> Dict[str, Callable]:
        """Initialize handlers for different parallel task types."""
        return {
            "claude_analysis": self._handle_claude_analysis,
            "compilation": self._handle_compilation,
            "testing": self._handle_testing,
            "deployment": self._handle_deployment,
            "code_review": self._handle_code_review,
            "shell_command": self._handle_shell_command,  # Generic handler for tool types
            "linter": self._handle_shell_command,        # Map linter to shell command
            "static_analyzer": self._handle_shell_command,  # Map static analyzer to shell command
            "test_runner": self._handle_shell_command,   # Map test runner to shell command
            "file_system": self._handle_shell_command,   # Map file system to shell command
        }

    def start_parallel_services(self):
        """Start parallel execution services."""
        self.resource_monitor.start_monitoring()
        self.task_queue.start()

    def stop_parallel_services(self):
        """Stop parallel execution services."""
        self.resource_monitor.stop_monitoring()
        self.task_queue.stop()
        self.background_manager.stop_all_processes()  # Clean up background processes

    def execute_workflow_parallel(
        self,
        workflow_name: str,
        inputs: Dict[str, Any],
        dry_run: bool = False
    ) -> WorkflowContext:
        """
        Execute a workflow with parallel stage support.

        Extends the base execute_workflow to handle parallel stages.
        """
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        workflow = self.workflows[workflow_name]
        workflow_id = f"wf_{uuid.uuid4().hex[:8]}"

        context = WorkflowContext(
            workflow_id=workflow_id,
            inputs=inputs
        )

        self._record_telemetry("parallel_workflow_start", {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "dry_run": dry_run
        })

        stages = workflow.get("stages", [])
        completed_stages = set()

        # Dynamic worker scaling based on resources
        optimal_workers = self.resource_monitor.get_optimal_worker_count(self.max_workers)
        if optimal_workers != self.max_workers:
            self.task_queue.max_workers = optimal_workers
            self._record_telemetry("worker_scaling", {
                "old_workers": self.max_workers,
                "new_workers": optimal_workers,
                "reason": "resource_optimization"
            })

        # Execute stages with parallel support
        max_iterations = len(stages) * 2
        iteration = 0

        while len(completed_stages) < len(stages) and iteration < max_iterations:
            iteration += 1
            progress_made = False

            # Find stages that can be executed (dependencies satisfied)
            ready_stages = []
            for stage in stages:
                stage_name = stage.get("name", f"stage_{iteration}")

                if stage_name in completed_stages:
                    continue

                # Check dependencies
                depends_on = stage.get("depends_on", [])
                if not all(dep in completed_stages for dep in depends_on):
                    continue

                ready_stages.append((stage_name, stage))

            # Group ready stages by parallel capability
            parallel_stages = []
            sequential_stages = []

            for stage_name, stage in ready_stages:
                if stage.get("parallel", False) or stage.get("tasks"):
                    parallel_stages.append((stage_name, stage))
                else:
                    sequential_stages.append((stage_name, stage))

            # Execute parallel stages
            if parallel_stages:
                parallel_results = self._execute_parallel_stages(
                    parallel_stages, context, dry_run
                )
                for stage_name, result in parallel_results.items():
                    context.stage_results[stage_name] = result
                    if result.status == TaskStatus.COMPLETED:
                        completed_stages.add(stage_name)
                        context.checkpoint(stage_name)
                        progress_made = True

            # Execute sequential stages
            for stage_name, stage in sequential_stages:
                result = self._execute_stage(stage, context, dry_run)
                context.stage_results[stage_name] = result

                if result.status == TaskStatus.COMPLETED:
                    completed_stages.add(stage_name)
                    context.checkpoint(stage_name)
                    progress_made = True

            if not progress_made:
                break

        # Record completion
        self._record_telemetry("parallel_workflow_complete", {
            "workflow_id": workflow_id,
            "stages_completed": len(completed_stages),
            "total_stages": len(stages),
            "duration_ms": int((datetime.now() - context.start_time).total_seconds() * 1000)
        })

        return context

    def _execute_parallel_stages(
        self,
        parallel_stages: List[tuple],
        context: WorkflowContext,
        dry_run: bool
    ) -> Dict[str, TaskResult]:
        """Execute multiple stages in parallel."""
        results = {}

        # Create parallel tasks
        tasks = []
        for stage_name, stage in parallel_stages:
            if stage.get("tasks"):
                # Handle parallel tasks within a stage
                task_results = self._execute_parallel_tasks(stage, context, dry_run)
                results.update(task_results)
            else:
                # Handle entire stage in parallel
                task = ParallelTask(
                    task_id=f"task_{uuid.uuid4().hex[:8]}",
                    stage=stage,
                    context=context,
                    priority=self._get_stage_priority(stage)
                )
                tasks.append((stage_name, task))

        if tasks:
            # Execute stages in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(tasks), self.max_workers)) as executor:
                future_to_stage = {
                    executor.submit(self._execute_stage, stage, context, dry_run): stage_name
                    for stage_name, stage in parallel_stages if not stage.get("tasks")
                }

                for future in as_completed(future_to_stage):
                    stage_name = future_to_stage[future]
                    try:
                        result = future.result()
                        results[stage_name] = result
                    except Exception as e:
                        results[stage_name] = TaskResult(
                            task_id=f"task_{uuid.uuid4().hex[:8]}",
                            status=TaskStatus.FAILED,
                            error=str(e)
                        )

        return results

    def _execute_parallel_tasks(
        self,
        stage: Dict[str, Any],
        context: WorkflowContext,
        dry_run: bool
    ) -> Dict[str, TaskResult]:
        """Execute parallel tasks within a stage."""
        tasks = stage.get("tasks", [])
        results = {}

        # Create task definitions for parallel execution
        parallel_tasks = []
        for task_name in tasks:
            task_def = {
                "name": task_name,
                "action": f"Execute {task_name}",
                "agent": stage.get("agent", "default_agent")
            }

            task = ParallelTask(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                stage=task_def,
                context=context,
                priority=self._get_stage_priority(stage)
            )
            parallel_tasks.append((task_name, task))

        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=min(len(parallel_tasks), self.max_workers)) as executor:
            future_to_task = {
                executor.submit(self._execute_stage, task.stage, context, dry_run): task_name
                for task_name, task in parallel_tasks
            }

            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result()
                    results[task_name] = result
                except Exception as e:
                    results[task_name] = TaskResult(
                        task_id=f"task_{uuid.uuid4().hex[:8]}",
                        status=TaskStatus.FAILED,
                        error=str(e)
                    )

        return results

    def _get_stage_priority(self, stage: Dict[str, Any]) -> TaskPriority:
        """Determine task priority based on stage characteristics."""
        stage_name = stage.get("name", "").lower()

        if "critical" in stage_name or "security" in stage_name:
            return TaskPriority.CRITICAL
        elif "build" in stage_name or "deploy" in stage_name:
            return TaskPriority.HIGH
        elif "test" in stage_name or "review" in stage_name:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.LOW

    def _handle_claude_analysis(self, task: ParallelTask) -> TaskResult:
        """Handle Claude reasoning for code analysis."""
        start_time = time.time()

        try:
            # Simulate Claude analysis
            analysis_result = {
                "refactoring_opportunities": [
                    "Extract complex method in class A",
                    "Consolidate duplicate utility functions",
                    "Simplify conditional logic in module B"
                ],
                "complexity_score": 0.7,
                "recommendations": [
                    "Consider breaking down large functions",
                    "Add type hints for better maintainability",
                    "Implement proper error handling"
                ]
            }

            duration_ms = int((time.time() - start_time) * 1000)

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                output=analysis_result,
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

    def _handle_testing(self, task: ParallelTask) -> TaskResult:
        """Handle parallel testing."""
        start_time = time.time()

        try:
            # Simulate distributed testing
            testing_result = {
                "tests_run": 150,
                "tests_passed": 145,
                "tests_failed": 5,
                "coverage_percent": 87.5,
                "test_framework": "pytest",
                "parallel_workers": 4
            }

            duration_ms = int((time.time() - start_time) * 1000)

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                output=testing_result,
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

    def _handle_shell_command(self, task: ParallelTask) -> TaskResult:
        """Handle shell command execution for various tool types."""
        start_time = time.time()

        try:
            stage = task.stage
            command = stage.get("command", "")
            timeout_minutes = stage.get("timeout_minutes", 5)
            tool_type = stage.get("tools", ["shell_command"])[0]

            if not command:
                raise ValueError("No command specified for shell command task")

            # Handle infinite timeout (watch mode) with background process manager
            if timeout_minutes == 0:
                process_id = f"{task.task_id}_{tool_type}"
                success = self.background_manager.start_process(
                    process_id,
                    command,
                    cwd="."
                )

                if success:
                    duration_ms = int((time.time() - start_time) * 1000)
                    return TaskResult(
                        task_id=task.task_id,
                        status=TaskStatus.COMPLETED,
                        output={
                            "process_id": process_id,
                            "command": command,
                            "mode": "background_watch",
                            "tool_type": tool_type,
                            "message": f"Background process started with PID: {self.background_manager.processes[process_id].pid}"
                        },
                        duration_ms=duration_ms
                    )
                else:
                    raise RuntimeError("Failed to start background process")

            # Handle regular commands with timeout
            timeout_seconds = timeout_minutes * 60
            result = subprocess.run(
                command,
                shell=True,
                cwd=".",
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )

            duration_ms = int((time.time() - start_time) * 1000)

            if result.returncode == 0:
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    output={
                        "command": command,
                        "tool_type": tool_type,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "return_code": result.returncode
                    },
                    duration_ms=duration_ms
                )
            else:
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Command failed with return code {result.returncode}: {result.stderr}",
                    output={
                        "command": command,
                        "tool_type": tool_type,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "return_code": result.returncode
                    },
                    duration_ms=duration_ms
                )

        except subprocess.TimeoutExpired:
            duration_ms = int((time.time() - start_time) * 1000)
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=f"Command timed out after {timeout_minutes} minutes",
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

    def _handle_deployment(self, task: ParallelTask) -> TaskResult:
        """Handle async deployment."""
        start_time = time.time()

        try:
            # Simulate Docker deployment
            deployment_result = {
                "deployment_success": True,
                "environment": "staging",
                "strategy": "blue_green",
                "containers_deployed": 2,
                "health_check_passed": True,
                "rollback_available": True
            }

            duration_ms = int((time.time() - start_time) * 1000)

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                output=deployment_result,
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

    def _handle_compilation(self, task: ParallelTask) -> TaskResult:
        """Handle parallel compilation."""
        start_time = time.time()

        try:
            # Simulate multi-core compilation
            compilation_result = {
                "build_success": True,
                "artifacts": ["app.exe", "lib.dll"],
                "warnings": 2,
                "errors": 0,
                "compile_time_ms": 1500
            }

            duration_ms = int((time.time() - start_time) * 1000)

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                output=compilation_result,
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

    def _execute_stage(
        self,
        stage: Dict[str, Any],
        context: WorkflowContext,
        dry_run: bool
    ) -> TaskResult:
        """Execute a single workflow stage."""
        stage_name = stage.get("name", "unnamed")
        agent_name = stage.get("agent", "default_agent")
        action = stage.get("action", "")

        task_id = f"task_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        self._record_telemetry("stage_start", {
            "task_id": task_id,
            "stage_name": stage_name,
            "agent": agent_name
        })

        # Ensure parallel handlers are registered (lazy initialization)
        self._ensure_handlers_registered()

        # Check if this is a parallel task that should use our handlers
        if hasattr(self, '_parallel_handlers') and action in self._parallel_handlers:
            # Create parallel task for handler
            parallel_task = ParallelTask(
                task_id=task_id,
                stage=stage,
                context=context,
                priority=self._get_stage_priority(stage)
            )

            try:
                # Use async helper to handle event loop properly
                result = asyncio.run(self._run_async_handler(self._parallel_handlers[action], parallel_task))

                self._record_telemetry("stage_complete", {
                    "task_id": task_id,
                    "stage_name": stage_name,
                    "status": result.status.value,
                    "duration_ms": result.duration_ms
                })

                return result

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                result = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=str(e),
                    duration_ms=duration_ms
                )

                self._record_telemetry("stage_complete", {
                    "task_id": task_id,
                    "stage_name": stage_name,
                    "status": result.status.value,
                    "duration_ms": result.duration_ms
                })

                return result

        # Gather inputs for traditional agent handling
        stage_inputs = {}
        for input_name in stage.get("inputs", []):
            if input_name in context.inputs:
                stage_inputs[input_name] = context.inputs[input_name]
            elif input_name in context.outputs:
                stage_inputs[input_name] = context.outputs[input_name]

        try:
            if dry_run:
                output = {
                    "dry_run": True,
                    "would_execute": action[:200],
                    "agent": agent_name,
                    "inputs": list(stage_inputs.keys())
                }
            else:
                output = self.agent_handler(agent_name, action, stage_inputs)

            duration_ms = int((time.time() - start_time) * 1000)

            result = TaskResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                output=output,
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            result = TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms
            )

        self._record_telemetry("stage_complete", {
            "task_id": task_id,
            "stage_name": stage_name,
            "status": result.status.value,
            "duration_ms": result.duration_ms
        })

        return result

    def _handle_code_review(self, task: ParallelTask) -> TaskResult:
        """Handle AI-assisted code review and PR generation."""
        start_time = time.time()

        try:
            # Simulate AI code review
            review_result = {
                "review_summary": "Code quality is good with minor suggestions",
                "issues_found": 3,
                "suggestions": [
                    "Add documentation for new function",
                    "Consider using more descriptive variable names",
                    "Add error handling for edge cases"
                ],
                "pr_title": "Feature: Add parallel workflow execution",
                "pr_description": "Implements parallel stage execution with resource monitoring",
                "approval_status": "approved_with_suggestions"
            }

            duration_ms = int((time.time() - start_time) * 1000)

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                output=review_result,
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

    def generate_mermaid_workflow(self, workflow_name: str) -> str:
        """Generate Mermaid diagram for workflow visualization."""
        if workflow_name not in self.workflows:
            return f"Unknown workflow: {workflow_name}"

        workflow = self.workflows[workflow_name]
        stages = workflow.get("stages", [])

        mermaid_lines = [
            "graph TD",
            "    %% Parallel Development Workflow",
            "    Start([Start]) --> Init[Initialize Context]"
        ]

        stage_nodes = {}
        for i, stage in enumerate(stages):
            stage_name = stage.get("name", f"stage_{i}")
            stage_id = f"S{i}"

            # Determine node shape based on type
            if stage.get("parallel", False):
                node_shape = f"{stage_id}{{{{ {stage_name} (Parallel) }}}}"
            elif stage.get("tasks"):
                node_shape = f"{stage_id}{{ {stage_name} (Tasks) }}"
            else:
                node_shape = f"{stage_id}[{stage_name}]"

            stage_nodes[stage_name] = stage_id
            mermaid_lines.append(f"    {node_shape}")

        # Add connections
        prev_node = "Init"
        for i, stage in enumerate(stages):
            stage_name = stage.get("name", f"stage_{i}")
            stage_id = stage_nodes[stage_name]

            # Handle dependencies
            depends_on = stage.get("depends_on", [])
            if depends_on:
                for dep in depends_on:
                    if dep in stage_nodes:
                        mermaid_lines.append(f"    {stage_nodes[dep]} --> {stage_id}")
            else:
                mermaid_lines.append(f"    {prev_node} --> {stage_id}")

            # Handle parallel tasks
            if stage.get("tasks"):
                tasks = stage.get("tasks", [])
                for j, task in enumerate(tasks):
                    task_id = f"T{i}_{j}"
                    mermaid_lines.append(f"    {stage_id} --> {task_id}[{task}]")
                    mermaid_lines.append(f"    {task_id} --> {stage_id}_done")
                prev_node = f"{stage_id}_done"
            else:
                prev_node = stage_id

        mermaid_lines.extend([
            f"    {prev_node} --> End([End])",
            "",
            "    %% Styling",
            "    classDef parallel fill:#e1f5fe,stroke:#01579b,stroke-width:2px",
            "    classDef tasks fill:#f3e5f5,stroke:#4a148c,stroke-width:2px",
            "    classDef critical fill:#ffebee,stroke:#b71c1c,stroke-width:2px",
            "",
            "    %% Apply styles to parallel stages"
        ])

        # Apply styles
        for i, stage in enumerate(stages):
            stage_name = stage.get("name", f"stage_{i}")
            stage_id = stage_nodes[stage_name]

            if stage.get("parallel", False):
                mermaid_lines.append(f"    class {stage_id} parallel")
            elif stage.get("tasks"):
                mermaid_lines.append(f"    class {stage_id} tasks")

        return "\n".join(mermaid_lines)


# Convenience functions
def execute_parallel_workflow(workflow_name: str, inputs: Dict[str, Any], dry_run: bool = False) -> WorkflowContext:
    """Execute a workflow with parallel capabilities."""
    executor = ParallelWorkflowExecutor()
    executor.start_parallel_services()
    try:
        return executor.execute_workflow_parallel(workflow_name, inputs, dry_run)
    finally:
        executor.stop_parallel_services()


if __name__ == "__main__":
    # Demo parallel execution
    print("=" * 60)
    print("PARALLEL WORKFLOW EXECUTOR DEMO")
    print("=" * 60)

    executor = ParallelWorkflowExecutor(max_workers=4)
    executor.start_parallel_services()

    try:
        print("\nAvailable workflows:")
        for wf in executor.list_workflows():
            print(f"  - {wf}")

        print("\n" + "-" * 40)
        print("Executing 'ci_cd_pipeline' workflow with parallel stages...")
        print("-" * 40)

        context = executor.execute_workflow_parallel(
            "ci_cd_pipeline",
            inputs={"repo_path": ".", "environment": "staging"},
            dry_run=True
        )

        print(f"\nWorkflow ID: {context.workflow_id}")
        print(f"Stages completed: {len(context.stage_results)}")

        for stage_name, result in context.stage_results.items():
            print(f"  {stage_name}: {result.status.value} ({result.duration_ms}ms)")

        print("\n" + "-" * 40)
        print("Generating Mermaid workflow diagram...")
        print("-" * 40)

        mermaid = executor.generate_mermaid_workflow("ci_cd_pipeline")
        print(mermaid)

    finally:
        executor.stop_parallel_services()
