"""
Parallel Executor - Manages parallel execution of workflow nodes

Provides advanced parallel execution capabilities with resource management,
throttling, and execution pools.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from enum import Enum
import logging
import multiprocessing


logger = logging.getLogger(__name__)


class ExecutorType(Enum):
    """Type of executor for parallel execution"""
    THREAD = "thread"
    PROCESS = "process"
    ASYNC = "async"


@dataclass
class ExecutionPool:
    """Pool configuration for parallel execution"""
    name: str
    executor_type: ExecutorType
    max_workers: int
    queue_size: int = 0  # 0 = unlimited
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParallelExecutor:
    """Manages parallel execution of tasks"""

    def __init__(self, default_max_workers: int = 10):
        self.default_max_workers = default_max_workers
        self.pools: Dict[str, ExecutionPool] = {}
        self.executors: Dict[str, Any] = {}
        self.active_tasks: Dict[str, Set[asyncio.Task]] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}

    def create_pool(self, pool: ExecutionPool):
        """Create execution pool"""
        if pool.name in self.pools:
            raise ValueError(f"Pool {pool.name} already exists")

        self.pools[pool.name] = pool

        # Create executor based on type
        if pool.executor_type == ExecutorType.THREAD:
            self.executors[pool.name] = ThreadPoolExecutor(max_workers=pool.max_workers)
        elif pool.executor_type == ExecutorType.PROCESS:
            self.executors[pool.name] = ProcessPoolExecutor(max_workers=pool.max_workers)
        elif pool.executor_type == ExecutorType.ASYNC:
            self._semaphores[pool.name] = asyncio.Semaphore(pool.max_workers)

        self.active_tasks[pool.name] = set()
        logger.info(f"Created execution pool: {pool.name}")

    async def execute_parallel(self,
                              tasks: List[Callable],
                              pool_name: Optional[str] = None,
                              return_exceptions: bool = True) -> List[Any]:
        """Execute multiple tasks in parallel"""
        if not tasks:
            return []

        # Use default pool if not specified
        if not pool_name:
            pool_name = "default"
            if pool_name not in self.pools:
                self.create_pool(ExecutionPool(
                    name=pool_name,
                    executor_type=ExecutorType.ASYNC,
                    max_workers=self.default_max_workers
                ))

        pool = self.pools[pool_name]

        if pool.executor_type == ExecutorType.ASYNC:
            return await self._execute_async(tasks, pool_name, return_exceptions)
        else:
            return await self._execute_with_executor(tasks, pool_name, return_exceptions)

    async def _execute_async(self,
                           tasks: List[Callable],
                           pool_name: str,
                           return_exceptions: bool) -> List[Any]:
        """Execute tasks asynchronously"""
        semaphore = self._semaphores[pool_name]
        pool = self.pools[pool_name]

        async def run_with_semaphore(task: Callable) -> Any:
            async with semaphore:
                if pool.timeout:
                    return await asyncio.wait_for(task(), timeout=pool.timeout)
                else:
                    return await task()

        # Create coroutines
        coroutines = []
        for task in tasks:
            if asyncio.iscoroutinefunction(task):
                coro = run_with_semaphore(task)
            else:
                # Wrap sync function
                async def wrapper(t=task):
                    return t()
                coro = run_with_semaphore(wrapper)
            coroutines.append(coro)

        # Execute all coroutines
        results = await asyncio.gather(*coroutines, return_exceptions=return_exceptions)
        return results

    async def _execute_with_executor(self,
                                    tasks: List[Callable],
                                    pool_name: str,
                                    return_exceptions: bool) -> List[Any]:
        """Execute tasks using thread/process executor"""
        executor = self.executors[pool_name]
        pool = self.pools[pool_name]
        loop = asyncio.get_event_loop()

        futures = []
        for task in tasks:
            if asyncio.iscoroutinefunction(task):
                # Run async function in thread
                future = loop.run_in_executor(
                    executor,
                    asyncio.run,
                    task()
                )
            else:
                future = loop.run_in_executor(executor, task)
            futures.append(future)

        # Wait for completion
        results = []
        for future in futures:
            try:
                if pool.timeout:
                    result = await asyncio.wait_for(future, timeout=pool.timeout)
                else:
                    result = await future
                results.append(result)
            except Exception as e:
                if return_exceptions:
                    results.append(e)
                else:
                    raise

        return results

    async def map_reduce(self,
                        data: List[Any],
                        mapper: Callable[[Any], Any],
                        reducer: Callable[[List[Any]], Any],
                        pool_name: Optional[str] = None,
                        chunk_size: Optional[int] = None) -> Any:
        """Execute map-reduce pattern"""
        if not data:
            return reducer([])

        # Chunk data if specified
        if chunk_size:
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        else:
            chunks = [[item] for item in data]

        # Map phase
        map_tasks = [lambda c=chunk: mapper(c) for chunk in chunks]
        map_results = await self.execute_parallel(map_tasks, pool_name)

        # Reduce phase
        return reducer(map_results)

    async def execute_batched(self,
                            tasks: List[Callable],
                            batch_size: int,
                            pool_name: Optional[str] = None,
                            delay_between_batches: float = 0) -> List[Any]:
        """Execute tasks in batches"""
        results = []

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await self.execute_parallel(batch, pool_name)
            results.extend(batch_results)

            # Delay between batches
            if delay_between_batches > 0 and i + batch_size < len(tasks):
                await asyncio.sleep(delay_between_batches)

        return results

    async def execute_with_dependencies(self,
                                       tasks: Dict[str, Dict[str, Any]],
                                       pool_name: Optional[str] = None) -> Dict[str, Any]:
        """Execute tasks with dependency resolution"""
        completed = set()
        results = {}
        max_iterations = len(tasks) * 2  # Safety limit

        iteration = 0
        while len(completed) < len(tasks) and iteration < max_iterations:
            iteration += 1

            # Find ready tasks
            ready_tasks = []
            for task_id, task_info in tasks.items():
                if task_id in completed:
                    continue

                dependencies = task_info.get("dependencies", [])
                if all(dep in completed for dep in dependencies):
                    ready_tasks.append((task_id, task_info["function"]))

            if not ready_tasks:
                if len(completed) < len(tasks):
                    raise RuntimeError("Circular dependency detected")
                break

            # Execute ready tasks in parallel
            task_ids = [t[0] for t in ready_tasks]
            task_funcs = [t[1] for t in ready_tasks]
            task_results = await self.execute_parallel(task_funcs, pool_name)

            # Store results
            for task_id, result in zip(task_ids, task_results):
                results[task_id] = result
                completed.add(task_id)

        return results

    def get_pool_stats(self, pool_name: str) -> Dict[str, Any]:
        """Get statistics for execution pool"""
        if pool_name not in self.pools:
            return {}

        pool = self.pools[pool_name]
        active = len(self.active_tasks.get(pool_name, set()))

        stats = {
            "name": pool_name,
            "type": pool.executor_type.value,
            "max_workers": pool.max_workers,
            "active_tasks": active,
            "queue_size": pool.queue_size
        }

        return stats

    def shutdown_pool(self, pool_name: str, wait: bool = True):
        """Shutdown execution pool"""
        if pool_name not in self.pools:
            return

        # Shutdown executor
        if pool_name in self.executors:
            executor = self.executors[pool_name]
            if hasattr(executor, 'shutdown'):
                executor.shutdown(wait=wait)
            del self.executors[pool_name]

        # Clean up
        del self.pools[pool_name]
        self.active_tasks.pop(pool_name, None)
        self._semaphores.pop(pool_name, None)

        logger.info(f"Shutdown execution pool: {pool_name}")

    def shutdown_all(self, wait: bool = True):
        """Shutdown all execution pools"""
        pool_names = list(self.pools.keys())
        for pool_name in pool_names:
            self.shutdown_pool(pool_name, wait=wait)