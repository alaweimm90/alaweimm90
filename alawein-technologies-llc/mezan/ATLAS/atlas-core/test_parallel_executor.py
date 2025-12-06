#!/usr/bin/env python
"""
Test script for the ParallelExecutor system.

This script demonstrates the capabilities of the parallel executor
with various use cases and configurations.
"""

import asyncio
import time
import random
from typing import Dict, Any
from atlas_core.parallel_executor import ParallelExecutor, TaskStatus, TaskResult


def cpu_intensive_task(task_id: int, n: int = 1000000) -> Dict[str, Any]:
    """Simulate CPU-intensive optimization task."""
    start = time.time()

    # Simulate some CPU work
    result = sum(i * i for i in range(n))

    # Simulate optimization result
    return {
        'task_id': task_id,
        'result': result,
        'execution_time': time.time() - start,
        'optimal_value': random.random() * 100
    }


def failing_task(task_id: int) -> None:
    """Task that always fails for testing error handling."""
    raise ValueError(f"Task {task_id} intentionally failed")


def random_duration_task(task_id: int, min_duration: float = 0.1, max_duration: float = 1.0) -> Dict[str, Any]:
    """Task with random duration to simulate real-world variance."""
    duration = random.uniform(min_duration, max_duration)
    time.sleep(duration)

    return {
        'task_id': task_id,
        'duration': duration,
        'success': True
    }


def sometimes_failing_task(task_id: int, fail_probability: float = 0.5) -> str:
    """Task that sometimes fails (module-level for pickling)."""
    if random.random() < fail_probability:
        raise Exception(f"Task {task_id} randomly failed")
    return f"Task {task_id} succeeded"


def test_basic_execution():
    """Test basic parallel execution functionality."""
    print("\n=== Testing Basic Parallel Execution ===")

    with ParallelExecutor(num_workers=4) as executor:
        # Submit 10 tasks
        for i in range(10):
            executor.add_task(
                f"basic_task_{i}",
                cpu_intensive_task,
                i,
                500000
            )

        # Execute and get results
        start_time = time.time()
        results = executor.execute()
        total_time = time.time() - start_time

        # Display results
        successful = sum(1 for r in results.values() if r.status == TaskStatus.COMPLETED)
        print(f"Completed {successful}/10 tasks in {total_time:.2f} seconds")
        print(f"Average time per task: {total_time/10:.2f} seconds")


def test_priority_execution():
    """Test priority-based task execution."""
    print("\n=== Testing Priority-Based Execution ===")

    with ParallelExecutor(num_workers=2) as executor:
        # Submit tasks with different priorities
        # Lower priority number = higher priority
        executor.add_task("high_priority_1", random_duration_task, 1, 0.5, 0.5, priority=0)
        executor.add_task("low_priority_1", random_duration_task, 2, 0.5, 0.5, priority=10)
        executor.add_task("high_priority_2", random_duration_task, 3, 0.5, 0.5, priority=0)
        executor.add_task("medium_priority", random_duration_task, 4, 0.5, 0.5, priority=5)

        results = executor.execute()

        # Tasks should be executed roughly in priority order
        for task_id, result in results.items():
            if result.status == TaskStatus.COMPLETED:
                print(f"  {task_id}: Completed")


def test_error_handling_and_retry():
    """Test error handling and retry logic."""
    print("\n=== Testing Error Handling and Retry ===")

    with ParallelExecutor(num_workers=2, max_retries=3) as executor:
        # Submit tasks that might fail
        for i in range(5):
            executor.add_task(
                f"retry_task_{i}",
                sometimes_failing_task,
                i,
                0.7  # 70% chance of failure
            )

        results = executor.execute()

        # Display retry statistics
        total_attempts = sum(r.attempts for r in results.values())
        successful = sum(1 for r in results.values() if r.status == TaskStatus.COMPLETED)
        print(f"  Successful: {successful}/5 tasks")
        print(f"  Total attempts: {total_attempts}")


def test_batch_processing():
    """Test batch task submission."""
    print("\n=== Testing Batch Processing ===")

    with ParallelExecutor(num_workers=4, batch_size=10) as executor:
        # Create a large batch of tasks
        tasks = [
            (f"batch_task_{i}", cpu_intensive_task, (i, 100000), {})
            for i in range(20)
        ]

        # Submit as batch
        start_time = time.time()
        executor.add_batch(tasks)

        # Execute with progress tracking
        executor.add_progress_callback(
            lambda p: print(f"  Progress: {p['completed']}/{p['total']} ({p['progress_percent']:.0f}%)", end='\r')
        )

        results = executor.execute()
        total_time = time.time() - start_time

        print(f"\n  Processed {len(results)} tasks in {total_time:.2f} seconds")
        print(f"  Throughput: {len(results)/total_time:.2f} tasks/second")


def test_worker_monitoring():
    """Test worker health monitoring capabilities."""
    print("\n=== Testing Worker Monitoring ===")

    with ParallelExecutor(num_workers=3, enable_progress=True) as executor:
        # Submit some CPU-intensive tasks
        for i in range(9):
            executor.add_task(
                f"monitor_task_{i}",
                cpu_intensive_task,
                i,
                2000000  # More intensive
            )

        # Execute with monitoring
        results = executor.execute()

        # Get worker statistics
        stats = executor.get_statistics()
        if stats['worker_stats']:
            print("  Worker Statistics:")
            print(f"    Total tasks: {stats['worker_stats']['total_tasks']}")
            print(f"    Completed: {stats['worker_stats']['completed']}")
            print(f"    Failed: {stats['worker_stats']['failed']}")
            print(f"    Avg execution time: {stats['worker_stats']['avg_execution_time']:.2f}s")


def run_all_tests():
    """Run all test scenarios."""
    print("=" * 60)
    print("ParallelExecutor Test Suite")
    print("=" * 60)

    test_basic_execution()
    test_priority_execution()
    test_error_handling_and_retry()
    test_batch_processing()
    test_worker_monitoring()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()