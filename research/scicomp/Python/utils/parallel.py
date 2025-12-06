#!/usr/bin/env python3
"""
Parallelization Utilities Module
Provides tools for parallel and distributed computing in scientific applications,
including multiprocessing, memory-efficient computation, and distributed computing.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import multiprocessing as mp
import concurrent.futures
import numpy as np
from typing import Callable, Any, List, Optional, Union, Iterator, Tuple
from functools import partial
import psutil
import warnings
from joblib import Parallel, delayed
import time
import threading
from queue import Queue
def get_optimal_workers(task_complexity: str = 'medium',
                       memory_usage: str = 'medium',
                       max_workers: Optional[int] = None) -> int:
    """
    Determine optimal number of worker processes based on system resources.
    Parameters
    ----------
    task_complexity : str, default 'medium'
        Computational complexity ('low', 'medium', 'high')
    memory_usage : str, default 'medium'
        Memory usage per task ('low', 'medium', 'high')
    max_workers : int, optional
        Maximum number of workers to use
    Returns
    -------
    int
        Optimal number of workers
    """
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    # Adjust based on task complexity
    complexity_factors = {'low': 1.0, 'medium': 0.8, 'high': 0.6}
    complexity_factor = complexity_factors.get(task_complexity, 0.8)
    # Adjust based on memory usage
    memory_factors = {'low': 1.0, 'medium': 0.7, 'high': 0.4}
    memory_factor = memory_factors.get(memory_usage, 0.7)
    # Calculate optimal workers
    optimal_workers = int(cpu_count * complexity_factor * memory_factor)
    optimal_workers = max(1, optimal_workers)  # At least 1 worker
    if max_workers is not None:
        optimal_workers = min(optimal_workers, max_workers)
    return optimal_workers
def parallel_map(func: Callable,
                iterable: List[Any],
                n_jobs: Optional[int] = None,
                backend: str = 'threading',
                verbose: int = 0,
                batch_size: Optional[int] = None) -> List[Any]:
    """
    Parallel map function with automatic optimization.
    Parameters
    ----------
    func : callable
        Function to apply to each element
    iterable : list
        Input data to process
    n_jobs : int, optional
        Number of parallel jobs. If None, uses optimal number
    backend : str, default 'threading'
        Backend to use ('threading', 'multiprocessing', 'concurrent')
    verbose : int, default 0
        Verbosity level
    batch_size : int, optional
        Batch size for processing
    Returns
    -------
    list
        Results from parallel execution
    """
    if n_jobs is None:
        n_jobs = get_optimal_workers()
    if len(iterable) < n_jobs:
        n_jobs = len(iterable)
    if backend == 'joblib':
        return Parallel(n_jobs=n_jobs, verbose=verbose, batch_size=batch_size)(
            delayed(func)(item) for item in iterable
        )
    elif backend == 'threading':
        return _parallel_map_threading(func, iterable, n_jobs, verbose)
    elif backend == 'multiprocessing':
        return _parallel_map_multiprocessing(func, iterable, n_jobs, verbose)
    elif backend == 'concurrent':
        return _parallel_map_concurrent(func, iterable, n_jobs, verbose)
    else:
        raise ValueError(f"Unknown backend: {backend}")
def _parallel_map_threading(func: Callable, iterable: List[Any],
                          n_jobs: int, verbose: int) -> List[Any]:
    """Threading-based parallel map."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        if verbose > 0:
            print(f"Processing {len(iterable)} items with {n_jobs} threads...")
        futures = [executor.submit(func, item) for item in iterable]
        results = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if verbose > 0 and (i + 1) % max(1, len(iterable) // 10) == 0:
                print(f"Completed {i + 1}/{len(iterable)} tasks")
            results.append(future.result())
    return results
def _parallel_map_multiprocessing(func: Callable, iterable: List[Any],
                                n_jobs: int, verbose: int) -> List[Any]:
    """Multiprocessing-based parallel map."""
    with mp.Pool(processes=n_jobs) as pool:
        if verbose > 0:
            print(f"Processing {len(iterable)} items with {n_jobs} processes...")
        results = pool.map(func, iterable)
    return results
def _parallel_map_concurrent(func: Callable, iterable: List[Any],
                           n_jobs: int, verbose: int) -> List[Any]:
    """ProcessPoolExecutor-based parallel map."""
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        if verbose > 0:
            print(f"Processing {len(iterable)} items with {n_jobs} processes...")
        results = list(executor.map(func, iterable))
    return results
def parallel_compute(func: Callable,
                    data: np.ndarray,
                    axis: int = 0,
                    n_jobs: Optional[int] = None,
                    chunk_size: Optional[int] = None) -> np.ndarray:
    """
    Parallel computation along specified axis of array.
    Parameters
    ----------
    func : callable
        Function to apply to each chunk
    data : ndarray
        Input data array
    axis : int, default 0
        Axis along which to parallelize
    n_jobs : int, optional
        Number of parallel jobs
    chunk_size : int, optional
        Size of chunks for processing
    Returns
    -------
    ndarray
        Result array
    """
    if n_jobs is None:
        n_jobs = get_optimal_workers()
    # Split data along specified axis
    chunks = np.array_split(data, n_jobs, axis=axis)
    # Process chunks in parallel
    results = parallel_map(func, chunks, n_jobs=n_jobs, backend='threading')
    # Concatenate results
    return np.concatenate(results, axis=axis)
def memory_efficient_compute(func: Callable,
                           data: np.ndarray,
                           chunk_size: int,
                           axis: int = 0,
                           n_jobs: int = 1,
                           progress_callback: Optional[Callable] = None) -> np.ndarray:
    """
    Memory-efficient computation by processing data in chunks.
    Parameters
    ----------
    func : callable
        Function to apply to each chunk
    data : ndarray
        Input data array
    chunk_size : int
        Maximum chunk size
    axis : int, default 0
        Axis along which to chunk
    n_jobs : int, default 1
        Number of parallel jobs per chunk
    progress_callback : callable, optional
        Progress callback function
    Returns
    -------
    ndarray
        Result array
    """
    # Calculate number of chunks
    n_chunks = int(np.ceil(data.shape[axis] / chunk_size))
    results = []
    for i in range(n_chunks):
        # Extract chunk
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, data.shape[axis])
        chunk_slice = tuple(slice(start_idx, end_idx) if j == axis else slice(None)
                           for j in range(data.ndim))
        chunk = data[chunk_slice]
        # Process chunk
        if n_jobs > 1:
            result = parallel_compute(func, chunk, axis=axis, n_jobs=n_jobs)
        else:
            result = func(chunk)
        results.append(result)
        # Progress callback
        if progress_callback is not None:
            progress_callback(i + 1, n_chunks)
    # Concatenate results
    return np.concatenate(results, axis=axis)
class ProgressMonitor:
    """Simple progress monitor for parallel computations."""
    def __init__(self, total_tasks: int, update_interval: float = 1.0):
        """
        Initialize progress monitor.
        Parameters
        ----------
        total_tasks : int
            Total number of tasks
        update_interval : float, default 1.0
            Update interval in seconds
        """
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.start_time = time.time()
        self.last_update = self.start_time
        self.update_interval = update_interval
        self.lock = threading.Lock()
    def update(self, n_completed: int = 1) -> None:
        """Update progress counter."""
        with self.lock:
            self.completed_tasks += n_completed
            current_time = time.time()
            if current_time - self.last_update >= self.update_interval:
                self._print_progress()
                self.last_update = current_time
    def _print_progress(self) -> None:
        """Print current progress."""
        elapsed = time.time() - self.start_time
        percentage = (self.completed_tasks / self.total_tasks) * 100
        if self.completed_tasks > 0:
            rate = self.completed_tasks / elapsed
            eta = (self.total_tasks - self.completed_tasks) / rate
            print(f"Progress: {self.completed_tasks}/{self.total_tasks} "
                  f"({percentage:.1f}%) - ETA: {eta:.1f}s")
        else:
            print(f"Progress: {self.completed_tasks}/{self.total_tasks} "
                  f"({percentage:.1f}%)")
    def finish(self) -> None:
        """Print final progress."""
        elapsed = time.time() - self.start_time
        rate = self.total_tasks / elapsed
        print(f"Completed {self.total_tasks} tasks in {elapsed:.1f}s "
              f"(rate: {rate:.1f} tasks/s)")
def distributed_compute(func: Callable,
                       data_chunks: List[Any],
                       n_workers: Optional[int] = None,
                       timeout: Optional[float] = None) -> List[Any]:
    """
    Distributed computation across multiple processes.
    Parameters
    ----------
    func : callable
        Function to execute on each chunk
    data_chunks : list
        List of data chunks to process
    n_workers : int, optional
        Number of worker processes
    timeout : float, optional
        Timeout for each task in seconds
    Returns
    -------
    list
        Results from all workers
    """
    if n_workers is None:
        n_workers = get_optimal_workers()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(func, chunk): i
            for i, chunk in enumerate(data_chunks)
        }
        results = [None] * len(data_chunks)
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk, timeout=timeout):
            chunk_index = future_to_chunk[future]
            try:
                results[chunk_index] = future.result()
            except Exception as exc:
                print(f'Chunk {chunk_index} generated an exception: {exc}')
                results[chunk_index] = None
    return results
def chunked_operation(operation: Callable,
                     arrays: List[np.ndarray],
                     chunk_size: int,
                     axis: int = 0,
                     combine_func: Optional[Callable] = None) -> np.ndarray:
    """
    Apply operation to arrays in chunks to manage memory usage.
    Parameters
    ----------
    operation : callable
        Operation to apply to array chunks
    arrays : list of ndarray
        Input arrays to process
    chunk_size : int
        Size of each chunk
    axis : int, default 0
        Axis along which to chunk
    combine_func : callable, optional
        Function to combine chunk results. If None, uses np.concatenate
    Returns
    -------
    ndarray
        Combined result
    """
    if combine_func is None:
        combine_func = lambda chunks: np.concatenate(chunks, axis=axis)
    # Determine number of chunks
    n_elements = arrays[0].shape[axis]
    n_chunks = int(np.ceil(n_elements / chunk_size))
    chunk_results = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_elements)
        # Extract chunks from all arrays
        chunk_slice = tuple(slice(start_idx, end_idx) if j == axis else slice(None)
                           for j in range(arrays[0].ndim))
        array_chunks = [arr[chunk_slice] for arr in arrays]
        # Apply operation to chunks
        chunk_result = operation(*array_chunks)
        chunk_results.append(chunk_result)
    # Combine results
    return combine_func(chunk_results)
class ResourceMonitor:
    """Monitor system resources during computation."""
    def __init__(self, monitoring_interval: float = 1.0):
        """
        Initialize resource monitor.
        Parameters
        ----------
        monitoring_interval : float, default 1.0
            Monitoring interval in seconds
        """
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.resources = {
            'cpu_percent': [],
            'memory_percent': [],
            'timestamps': []
        }
        self.monitor_thread = None
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    def stop_monitoring(self) -> Dict[str, List[float]]:
        """Stop monitoring and return collected data."""
        self.monitoring = False
        if self.monitor_thread is not None:
            self.monitor_thread.join()
        return self.resources.copy()
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        start_time = time.time()
        while self.monitoring:
            current_time = time.time()
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            self.resources['timestamps'].append(current_time - start_time)
            self.resources['cpu_percent'].append(cpu_percent)
            self.resources['memory_percent'].append(memory_percent)
            time.sleep(self.monitoring_interval)
    def get_statistics(self) -> Dict[str, float]:
        """Get resource usage statistics."""
        if not self.resources['cpu_percent']:
            return {}
        return {
            'avg_cpu_percent': np.mean(self.resources['cpu_percent']),
            'max_cpu_percent': np.max(self.resources['cpu_percent']),
            'avg_memory_percent': np.mean(self.resources['memory_percent']),
            'max_memory_percent': np.max(self.resources['memory_percent']),
            'monitoring_duration': self.resources['timestamps'][-1] if self.resources['timestamps'] else 0
        }