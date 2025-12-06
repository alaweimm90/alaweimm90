"""
Production-Grade Parallel Execution System for MEZAN (Version 2)

This module provides a robust parallel execution framework using multiprocessing
with proper handling for function serialization and worker pool management.

Features:
- True parallel processing with multiprocessing
- Dynamic worker pool management
- Real-time progress tracking and callbacks
- Comprehensive error handling with retry logic
- Memory-efficient batch processing
- Worker health monitoring
- Graceful shutdown support

Author: MEZAN Team
Version: 2.0.0
"""

import asyncio
import functools
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    worker_pid: Optional[int] = None
    attempts: int = 0

    @property
    def execution_time(self) -> float:
        """Calculate execution time in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0


def worker_execute_task(
    task_id: str,
    func: Callable,
    args: tuple,
    kwargs: dict,
    timeout: Optional[float] = None
) -> TaskResult:
    """
    Worker function that executes a task in a separate process.
    This function must be at module level to be picklable.
    """
    start_time = time.time()
    worker_pid = os.getpid()

    try:
        # Set up signal handler for timeout if specified
        if timeout:
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Task {task_id} exceeded timeout of {timeout}s")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))

        # Execute the function
        result = func(*args, **kwargs)

        # Cancel alarm if timeout was set
        if timeout:
            signal.alarm(0)

        return TaskResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            result=result,
            start_time=start_time,
            end_time=time.time(),
            worker_pid=worker_pid
        )

    except Exception as e:
        # Cancel alarm if timeout was set
        if timeout:
            signal.alarm(0)

        return TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error=str(e),
            error_traceback=traceback.format_exc(),
            start_time=start_time,
            end_time=time.time(),
            worker_pid=worker_pid
        )


class WorkerPoolManager:
    """Manages the worker pool and monitors health."""

    def __init__(self, num_workers: int):
        """Initialize worker pool manager."""
        self.num_workers = num_workers
        self.executor = ProcessPoolExecutor(max_workers=num_workers)
        self.worker_stats = defaultdict(lambda: {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_time': 0.0,
            'last_task_time': None
        })

    def submit(self, func: Callable, *args, **kwargs):
        """Submit a task to the worker pool."""
        return self.executor.submit(func, *args, **kwargs)

    def update_stats(self, result: TaskResult):
        """Update worker statistics based on task result."""
        if result.worker_pid:
            stats = self.worker_stats[result.worker_pid]
            if result.status == TaskStatus.COMPLETED:
                stats['tasks_completed'] += 1
            else:
                stats['tasks_failed'] += 1
            stats['total_time'] += result.execution_time
            stats['last_task_time'] = time.time()

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of worker statistics."""
        total_tasks = sum(s['tasks_completed'] + s['tasks_failed'] for s in self.worker_stats.values())
        total_completed = sum(s['tasks_completed'] for s in self.worker_stats.values())
        total_failed = sum(s['tasks_failed'] for s in self.worker_stats.values())
        avg_time = sum(s['total_time'] for s in self.worker_stats.values()) / max(total_tasks, 1)

        return {
            'num_workers': self.num_workers,
            'total_tasks': total_tasks,
            'completed': total_completed,
            'failed': total_failed,
            'avg_execution_time': avg_time,
            'worker_details': dict(self.worker_stats)
        }

    def shutdown(self, wait: bool = True):
        """Shutdown the worker pool."""
        self.executor.shutdown(wait=wait)


class ParallelExecutor:
    """
    Main parallel execution system for MEZAN.

    Provides high-level interface for parallel task execution with
    automatic retry, progress tracking, and error handling.
    """

    def __init__(
        self,
        num_workers: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_progress: bool = True,
        batch_size: Optional[int] = None
    ):
        """
        Initialize the parallel executor.

        Args:
            num_workers: Number of worker processes (default: CPU count)
            max_retries: Maximum retry attempts for failed tasks
            retry_delay: Delay between retry attempts in seconds
            enable_progress: Enable progress tracking and callbacks
            batch_size: Process tasks in batches of this size
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_progress = enable_progress
        self.batch_size = batch_size

        # Task management
        self.tasks = {}  # task_id -> (func, args, kwargs, priority)
        self.results = {}  # task_id -> TaskResult
        self.pending_tasks = set()
        self.running_tasks = set()
        self.completed_tasks = set()

        # Worker pool
        self.pool_manager = None

        # Progress tracking
        self.progress_callbacks = []
        self.start_time = None

        # Shutdown handling
        self.shutdown_event = mp.Event()
        self._original_sigint = None
        self._original_sigterm = None

        logger.info(f"Initialized ParallelExecutor with {self.num_workers} workers")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()

    def start(self):
        """Start the executor and worker pool."""
        self.pool_manager = WorkerPoolManager(self.num_workers)
        self.start_time = time.time()

        # Set up signal handlers for graceful shutdown
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("ParallelExecutor started")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        self.shutdown()
        sys.exit(0)

    def add_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        priority: int = 0,
        timeout: Optional[float] = None,
        **kwargs
    ):
        """
        Add a task to be executed.

        Args:
            task_id: Unique identifier for the task
            func: Function to execute (must be picklable)
            *args: Positional arguments for the function
            priority: Task priority (lower = higher priority)
            timeout: Execution timeout in seconds
            **kwargs: Keyword arguments for the function
        """
        self.tasks[task_id] = {
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'priority': priority,
            'timeout': timeout,
            'attempts': 0
        }
        self.pending_tasks.add(task_id)
        logger.debug(f"Added task {task_id} with priority {priority}")

    def add_batch(
        self,
        tasks: List[Tuple[str, Callable, tuple, dict]],
        priority: int = 0
    ):
        """
        Add multiple tasks as a batch.

        Args:
            tasks: List of (task_id, func, args, kwargs) tuples
            priority: Default priority for all tasks
        """
        for task_id, func, args, kwargs in tasks:
            self.add_task(task_id, func, *args, priority=priority, **kwargs)
        logger.info(f"Added batch of {len(tasks)} tasks")

    def execute(self, timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """
        Execute all pending tasks.

        Args:
            timeout: Overall execution timeout in seconds

        Returns:
            Dictionary mapping task IDs to results
        """
        if not self.pool_manager:
            self.start()

        execution_start = time.time()
        futures = {}  # future -> task_id mapping

        # Sort tasks by priority
        sorted_tasks = sorted(
            self.pending_tasks,
            key=lambda tid: self.tasks[tid]['priority']
        )

        # Process tasks in batches if specified
        if self.batch_size:
            batches = [sorted_tasks[i:i + self.batch_size]
                      for i in range(0, len(sorted_tasks), self.batch_size)]
        else:
            batches = [sorted_tasks]

        for batch in batches:
            if self.shutdown_event.is_set():
                break

            batch_futures = {}

            # Submit batch tasks
            for task_id in batch:
                if timeout and (time.time() - execution_start) > timeout:
                    logger.warning("Execution timeout reached")
                    break

                task_info = self.tasks[task_id]
                task_info['attempts'] += 1

                self.pending_tasks.discard(task_id)
                self.running_tasks.add(task_id)

                future = self.pool_manager.submit(
                    worker_execute_task,
                    task_id,
                    task_info['func'],
                    task_info['args'],
                    task_info['kwargs'],
                    task_info['timeout']
                )
                batch_futures[future] = task_id

            # Wait for batch completion
            for future in as_completed(batch_futures):
                task_id = batch_futures[future]

                try:
                    result = future.result()
                    result.attempts = self.tasks[task_id]['attempts']

                    self.pool_manager.update_stats(result)
                    self._handle_task_result(task_id, result)

                except Exception as e:
                    logger.error(f"Error getting result for task {task_id}: {e}")
                    result = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error=str(e),
                        attempts=self.tasks[task_id]['attempts']
                    )
                    self._handle_task_result(task_id, result)

        # Handle any remaining retries
        self._process_retries()

        logger.info(f"Execution completed: {len(self.completed_tasks)} tasks in "
                   f"{time.time() - execution_start:.2f}s")

        return self.results

    def _handle_task_result(self, task_id: str, result: TaskResult):
        """Handle task completion or failure."""
        self.running_tasks.discard(task_id)

        if result.status == TaskStatus.FAILED and result.attempts < self.max_retries:
            # Schedule for retry
            logger.info(f"Task {task_id} failed (attempt {result.attempts}), scheduling retry")
            time.sleep(self.retry_delay)
            self.pending_tasks.add(task_id)
        else:
            # Task completed or max retries reached
            self.completed_tasks.add(task_id)
            self.results[task_id] = result

            if result.status == TaskStatus.COMPLETED:
                logger.debug(f"Task {task_id} completed successfully")
            else:
                logger.error(f"Task {task_id} failed after {result.attempts} attempts")

        # Notify progress
        if self.enable_progress:
            self._notify_progress()

    def _process_retries(self):
        """Process any remaining tasks that need retry."""
        retry_tasks = list(self.pending_tasks)

        for task_id in retry_tasks:
            if self.shutdown_event.is_set():
                break

            task_info = self.tasks[task_id]
            task_info['attempts'] += 1

            self.pending_tasks.discard(task_id)
            self.running_tasks.add(task_id)

            future = self.pool_manager.submit(
                worker_execute_task,
                task_id,
                task_info['func'],
                task_info['args'],
                task_info['kwargs'],
                task_info['timeout']
            )

            try:
                result = future.result()
                result.attempts = task_info['attempts']
                self.pool_manager.update_stats(result)
                self._handle_task_result(task_id, result)
            except Exception as e:
                logger.error(f"Error in retry for task {task_id}: {e}")
                result = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=str(e),
                    attempts=task_info['attempts']
                )
                self._handle_task_result(task_id, result)

    def _notify_progress(self):
        """Notify progress callbacks."""
        progress = self.get_progress()
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    def add_progress_callback(self, callback: Callable):
        """Add a progress tracking callback."""
        self.progress_callbacks.append(callback)

    def get_progress(self) -> Dict[str, Any]:
        """Get current execution progress."""
        total_tasks = len(self.tasks)
        completed = len(self.completed_tasks)
        running = len(self.running_tasks)
        pending = len(self.pending_tasks)
        failed = sum(1 for r in self.results.values() if r.status == TaskStatus.FAILED)

        elapsed = time.time() - self.start_time if self.start_time else 0

        return {
            'total': total_tasks,
            'completed': completed,
            'running': running,
            'pending': pending,
            'failed': failed,
            'success_rate': (completed - failed) / max(completed, 1),
            'progress_percent': (completed / max(total_tasks, 1)) * 100,
            'elapsed_time': elapsed,
            'estimated_remaining': (elapsed / max(completed, 1)) * pending if completed > 0 else 0
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        stats = {
            'progress': self.get_progress(),
            'worker_stats': self.pool_manager.get_stats_summary() if self.pool_manager else None,
            'task_results': {
                'successful': sum(1 for r in self.results.values() if r.status == TaskStatus.COMPLETED),
                'failed': sum(1 for r in self.results.values() if r.status == TaskStatus.FAILED),
                'avg_execution_time': sum(r.execution_time for r in self.results.values()) / max(len(self.results), 1),
                'total_execution_time': time.time() - self.start_time if self.start_time else 0
            }
        }
        return stats

    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        logger.info("Shutting down ParallelExecutor...")

        self.shutdown_event.set()

        if self.pool_manager:
            self.pool_manager.shutdown(wait=wait)

        # Restore original signal handlers
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)

        logger.info("ParallelExecutor shutdown complete")


# Example usage functions for testing
def example_cpu_task(n: int) -> int:
    """Example CPU-intensive task."""
    result = sum(i * i for i in range(n))
    time.sleep(0.1)  # Simulate some work
    return result


def example_io_task(duration: float) -> str:
    """Example I/O-bound task."""
    time.sleep(duration)
    return f"Completed after {duration}s"


def example_failing_task():
    """Example task that always fails."""
    raise ValueError("This task always fails")


def progress_reporter(progress: Dict[str, Any]):
    """Example progress callback."""
    print(f"Progress: {progress['completed']}/{progress['total']} "
          f"({progress['progress_percent']:.1f}%) - "
          f"Running: {progress['running']}, Failed: {progress['failed']}")


def main_example():
    """Example usage of ParallelExecutor."""
    print("=" * 60)
    print("ParallelExecutor Example")
    print("=" * 60)

    # Create executor with context manager
    with ParallelExecutor(num_workers=4, max_retries=2, batch_size=10) as executor:
        # Add progress callback
        executor.add_progress_callback(progress_reporter)

        # Add various types of tasks
        for i in range(20):
            executor.add_task(f"cpu_task_{i}", example_cpu_task, 1000000 + i * 10000)

        for i in range(10):
            executor.add_task(f"io_task_{i}", example_io_task, 0.5 + i * 0.1, priority=1)

        # Add some failing tasks to test retry logic
        executor.add_task("failing_task_1", example_failing_task, priority=2)

        # Execute all tasks
        results = executor.execute(timeout=30)

        # Display results
        print("\n" + "=" * 60)
        print("Execution Results")
        print("=" * 60)

        successful = sum(1 for r in results.values() if r.status == TaskStatus.COMPLETED)
        failed = sum(1 for r in results.values() if r.status == TaskStatus.FAILED)

        print(f"Total tasks: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        # Show statistics
        stats = executor.get_statistics()
        print(f"\nExecution time: {stats['task_results']['total_execution_time']:.2f}s")
        print(f"Average task time: {stats['task_results']['avg_execution_time']:.2f}s")

        if stats['worker_stats']:
            print(f"\nWorker statistics:")
            print(f"  Workers used: {stats['worker_stats']['num_workers']}")
            print(f"  Tasks per worker: {stats['worker_stats']['total_tasks'] / stats['worker_stats']['num_workers']:.1f}")


if __name__ == "__main__":
    main_example()