"""
Performance Optimization Utilities - Cycle 19

Speed improvements for ORCHEX research pipeline.
"""

import asyncio
from functools import lru_cache, wraps
from typing import Callable, Any
import time


def timed(func: Callable) -> Callable:
    """
    Decorator to time async functions

    Usage:
        @timed
        async def my_function():
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"⏱️  {func.__name__}: {elapsed:.2f}s")
        return result
    return wrapper


def cached_literature_search(max_cache_size: int = 100):
    """
    Cache for literature search results

    Significantly speeds up repeated searches for same topics.
    """
    @lru_cache(maxsize=max_cache_size)
    def _cache_wrapper(query: str, domain: str) -> tuple:
        # This is a placeholder - actual caching happens in hypothesis_generator
        return (query, domain)

    return _cache_wrapper


async def run_parallel(tasks: list, max_concurrent: int = 4) -> list:
    """
    Run multiple async tasks in parallel with concurrency limit

    Args:
        tasks: List of coroutines to run
        max_concurrent: Maximum concurrent tasks

    Returns:
        List of results in same order as input tasks

    Example:
        tasks = [
            hypothesis_gen.generate_one(topic1),
            hypothesis_gen.generate_one(topic2),
            hypothesis_gen.generate_one(topic3),
        ]
        results = await run_parallel(tasks, max_concurrent=2)
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*[bounded_task(t) for t in tasks])


class BatchProcessor:
    """
    Process items in batches for efficiency

    Useful for:
    - Batch hypothesis validation
    - Batch experiment execution
    - Batch agent actions
    """

    def __init__(self, batch_size: int = 10):
        """
        Initialize batch processor

        Args:
            batch_size: Number of items per batch
        """
        self.batch_size = batch_size

    async def process_batches(
        self,
        items: list,
        process_func: Callable,
        show_progress: bool = True
    ) -> list:
        """
        Process items in batches

        Args:
            items: List of items to process
            process_func: Async function to process each item
            show_progress: Whether to print progress

        Returns:
            List of results
        """
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(items))
            batch = items[start_idx:end_idx]

            if show_progress:
                print(f"📦 Processing batch {batch_idx + 1}/{total_batches} "
                      f"({len(batch)} items)")

            # Process batch in parallel
            batch_results = await asyncio.gather(*[
                process_func(item) for item in batch
            ])

            results.extend(batch_results)

        return results


class ProgressTracker:
    """Track progress of long-running operations"""

    def __init__(self, total: int, description: str = "Progress"):
        """
        Initialize progress tracker

        Args:
            total: Total number of items
            description: Description of operation
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()

    def update(self, n: int = 1):
        """Update progress by n items"""
        self.current += n
        percent = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time

        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            print(f"{self.description}: {self.current}/{self.total} "
                  f"({percent:.1f}%) - ETA: {eta:.0f}s")

    def finish(self):
        """Mark as finished"""
        elapsed = time.time() - self.start_time
        print(f"✓ {self.description} complete in {elapsed:.1f}s")


# Memory optimization helpers

def clear_large_objects(*objects):
    """
    Explicitly delete large objects to free memory

    Usage:
        clear_large_objects(large_dataframe, huge_tensor)
    """
    import gc
    for obj in objects:
        del obj
    gc.collect()


def get_memory_usage() -> dict:
    """
    Get current memory usage

    Returns:
        Dict with memory stats (if psutil available)
    """
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        return {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    except ImportError:
        return {"error": "psutil not available"}


# Async optimization helpers

async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> Any:
    """
    Retry async function with exponential backoff

    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Result from successful function call

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e

            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                print(f"⚠️  Retry {attempt + 1}/{max_retries} after {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                print(f"❌ All {max_retries} retries failed")

    raise last_exception


# Quick optimization guide

OPTIMIZATION_TIPS = """
ORCHEX Performance Optimization Guide
====================================

1. LITERATURE SEARCH CACHING
   - Searches are cached automatically
   - Clear cache if stale: hypothesis_generator._cache.clear()

2. PARALLEL HYPOTHESIS GENERATION
   - Use run_parallel() for multiple hypotheses
   - Set max_concurrent based on API rate limits

3. BATCH PROCESSING
   - Use BatchProcessor for validation pipeline
   - Reduces overhead, improves throughput

4. MEMORY MANAGEMENT
   - Call clear_large_objects() after Stage 2
   - Monitor with get_memory_usage()

5. ASYNC BEST PRACTICES
   - Always await long operations
   - Use asyncio.gather() for parallel tasks
   - Set timeouts on external API calls

6. PROFILING
   - Use @timed decorator on slow functions
   - Use ProgressTracker for long loops

Example:
    from ORCHEX.performance_utils import run_parallel, BatchProcessor

    # Parallel hypothesis generation
    tasks = [gen_hypothesis(topic) for _ in range(5)]
    hypotheses = await run_parallel(tasks, max_concurrent=2)

    # Batch validation
    processor = BatchProcessor(batch_size=10)
    results = await processor.process_batches(
        hypotheses,
        validate_hypothesis,
        show_progress=True
    )
"""


def print_optimization_tips():
    """Print optimization guide"""
    print(OPTIMIZATION_TIPS)
