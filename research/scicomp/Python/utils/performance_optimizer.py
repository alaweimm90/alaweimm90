"""
Performance optimization utilities for SciComp.
This module provides advanced performance optimization including:
- Just-in-time compilation with Numba
- Memory management and profiling
- Parallel processing utilities
- GPU acceleration helpers
- Performance benchmarking tools
"""
import numpy as np
import time
import functools
import warnings
from typing import Union, List, Tuple, Optional, Dict, Callable, Any
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# Optional performance libraries
try:
    import numba
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. JIT compilation disabled.")
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
class PerformanceOptimizer:
    """Main performance optimization class."""
    def __init__(self):
        """Initialize performance optimizer."""
        self.benchmarks = {}
        self.memory_tracker = MemoryTracker() if PSUTIL_AVAILABLE else None
        self.parallel_executor = ParallelExecutor()
    def optimize_function(self, func: Callable, method: str = 'auto') -> Callable:
        """
        Optimize function using specified method.
        Args:
            func: Function to optimize
            method: Optimization method ('numba', 'vectorize', 'parallel', 'auto')
        Returns:
            Optimized function
        """
        if method == 'auto':
            method = self._choose_optimization_method(func)
        if method == 'numba' and NUMBA_AVAILABLE:
            return self._apply_numba_optimization(func)
        elif method == 'vectorize':
            return self._apply_vectorization(func)
        elif method == 'parallel':
            return self._apply_parallelization(func)
        else:
            return func
    def _choose_optimization_method(self, func: Callable) -> str:
        """Automatically choose best optimization method."""
        # Simple heuristics - in practice would analyze function more deeply
        if NUMBA_AVAILABLE:
            return 'numba'
        return 'vectorize'
    def _apply_numba_optimization(self, func: Callable) -> Callable:
        """Apply Numba JIT compilation."""
        try:
            return numba.jit(nopython=True, cache=True)(func)
        except Exception:
            warnings.warn(f"Numba compilation failed for {func.__name__}, using original function")
            return func
    def _apply_vectorization(self, func: Callable) -> Callable:
        """Apply numpy vectorization."""
        return np.vectorize(func)
    def _apply_parallelization(self, func: Callable) -> Callable:
        """Apply parallel execution wrapper."""
        @functools.wraps(func)
        def parallel_wrapper(*args, **kwargs):
            return self.parallel_executor.execute_parallel(func, *args, **kwargs)
        return parallel_wrapper
    def benchmark_function(self, func: Callable, *args, n_runs: int = 100,
                          **kwargs) -> Dict[str, float]:
        """
        Benchmark function performance.
        Args:
            func: Function to benchmark
            args: Function arguments
            n_runs: Number of benchmark runs
            kwargs: Function keyword arguments
        Returns:
            Performance statistics
        """
        times = []
        memory_usage = []
        for _ in range(n_runs):
            if self.memory_tracker:
                mem_before = self.memory_tracker.get_memory_usage()
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            if self.memory_tracker:
                mem_after = self.memory_tracker.get_memory_usage()
                memory_usage.append(mem_after - mem_before)
        stats = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'total_time': np.sum(times)
        }
        if memory_usage:
            stats.update({
                'mean_memory': np.mean(memory_usage),
                'max_memory': np.max(memory_usage)
            })
        return stats
class MemoryTracker:
    """Memory usage tracking utilities."""
    def __init__(self):
        """Initialize memory tracker."""
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil required for memory tracking")
        self.process = psutil.Process()
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    def get_memory_percent(self) -> float:
        """Get memory usage as percentage of system memory."""
        return self.process.memory_percent()
    def memory_profile(self, func: Callable) -> Callable:
        """Decorator to profile memory usage of function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mem_before = self.get_memory_usage()
            result = func(*args, **kwargs)
            mem_after = self.get_memory_usage()
            print(f"Memory usage for {func.__name__}:")
            print(f"  Before: {mem_before:.2f} MB")
            print(f"  After: {mem_after:.2f} MB")
            print(f"  Delta: {mem_after - mem_before:.2f} MB")
            return result
        return wrapper
class ParallelExecutor:
    """Parallel execution utilities."""
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel executor.
        Args:
            max_workers: Maximum number of worker threads/processes
        """
        self.max_workers = max_workers or mp.cpu_count()
    def execute_parallel(self, func: Callable, data: Union[List, np.ndarray],
                        method: str = 'thread', chunk_size: Optional[int] = None) -> List:
        """
        Execute function in parallel over data.
        Args:
            func: Function to execute
            data: Data to process in parallel
            method: Parallelization method ('thread' or 'process')
            chunk_size: Size of chunks for processing
        Returns:
            Results from parallel execution
        """
        if method == 'thread':
            executor_class = ThreadPoolExecutor
        elif method == 'process':
            executor_class = ProcessPoolExecutor
        else:
            raise ValueError(f"Unknown parallel method: {method}")
        with executor_class(max_workers=self.max_workers) as executor:
            if chunk_size:
                chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
                futures = [executor.submit(func, chunk) for chunk in chunks]
                results = []
                for future in futures:
                    results.extend(future.result())
                return results
            else:
                futures = [executor.submit(func, item) for item in data]
                return [future.result() for future in futures]
    def parallel_map(self, func: Callable, data: List, method: str = 'thread') -> List:
        """
        Apply function to data in parallel (like built-in map).
        Args:
            func: Function to apply
            data: Data to map over
            method: Parallelization method
        Returns:
            Mapped results
        """
        return self.execute_parallel(func, data, method)
class GPUAccelerator:
    """GPU acceleration utilities using CuPy."""
    def __init__(self):
        """Initialize GPU accelerator."""
        self.gpu_available = CUPY_AVAILABLE
        if not self.gpu_available:
            warnings.warn("CuPy not available. GPU acceleration disabled.")
    def to_gpu(self, array: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
        """Transfer array to GPU if available."""
        if self.gpu_available:
            return cp.asarray(array)
        return array
    def to_cpu(self, array: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """Transfer array to CPU."""
        if self.gpu_available and hasattr(array, 'get'):
            return array.get()
        return array
    def gpu_accelerated(self, func: Callable) -> Callable:
        """Decorator to automatically use GPU acceleration."""
        if not self.gpu_available:
            return func
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Convert numpy arrays to cupy arrays
            gpu_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    gpu_args.append(self.to_gpu(arg))
                else:
                    gpu_args.append(arg)
            gpu_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    gpu_kwargs[key] = self.to_gpu(value)
                else:
                    gpu_kwargs[key] = value
            # Execute function
            result = func(*gpu_args, **gpu_kwargs)
            # Convert result back to numpy if needed
            if hasattr(result, 'get'):
                result = result.get()
            return result
        return wrapper
class CacheManager:
    """Intelligent caching for expensive computations."""
    def __init__(self, max_size: int = 128):
        """
        Initialize cache manager.
        Args:
            max_size: Maximum cache size
        """
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    def cached_function(self, func: Callable) -> Callable:
        """Decorator to cache function results."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = self._make_cache_key(func.__name__, args, kwargs)
            # Check cache
            if key in self.cache:
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return self.cache[key]
            # Compute result
            result = func(*args, **kwargs)
            # Store in cache
            self._store_in_cache(key, result)
            return result
        return wrapper
    def _make_cache_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Create cache key from function arguments."""
        # Simple key creation - could be more sophisticated
        key_parts = [func_name]
        for arg in args:
            if isinstance(arg, np.ndarray):
                key_parts.append(f"array_{arg.shape}_{arg.dtype}_{hash(arg.data.tobytes())}")
            else:
                key_parts.append(str(hash(arg)))
        for k, v in sorted(kwargs.items()):
            if isinstance(v, np.ndarray):
                key_parts.append(f"{k}_array_{v.shape}_{v.dtype}_{hash(v.data.tobytes())}")
            else:
                key_parts.append(f"{k}_{hash(v)}")
        return "_".join(key_parts)
    def _store_in_cache(self, key: str, result: Any):
        """Store result in cache with LRU eviction."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self.access_count.keys(), key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
        self.cache[key] = result
        self.access_count[key] = 1
    def clear_cache(self):
        """Clear the entire cache."""
        self.cache.clear()
        self.access_count.clear()
    def cache_info(self) -> Dict:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'total_accesses': sum(self.access_count.values()),
            'unique_computations': len(self.cache)
        }
class ProfilerContext:
    """Context manager for performance profiling."""
    def __init__(self, name: str = "Operation"):
        """
        Initialize profiler context.
        Args:
            name: Name of operation being profiled
        """
        self.name = name
        self.start_time = None
        self.memory_tracker = MemoryTracker() if PSUTIL_AVAILABLE else None
        self.start_memory = None
    def __enter__(self):
        """Enter profiling context."""
        self.start_time = time.perf_counter()
        if self.memory_tracker:
            self.start_memory = self.memory_tracker.get_memory_usage()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit profiling context and report results."""
        end_time = time.perf_counter()
        elapsed = end_time - self.start_time
        print(f"\nPerformance Profile: {self.name}")
        print(f"  Execution time: {elapsed:.4f} seconds")
        if self.memory_tracker:
            end_memory = self.memory_tracker.get_memory_usage()
            memory_delta = end_memory - self.start_memory
            print(f"  Memory usage: {memory_delta:+.2f} MB")
            print(f"  Peak memory: {self.memory_tracker.get_memory_percent():.1f}%")
# Convenience functions and decorators
def optimize(method: str = 'auto'):
    """Decorator for automatic function optimization."""
    optimizer = PerformanceOptimizer()
    def decorator(func: Callable) -> Callable:
        return optimizer.optimize_function(func, method)
    return decorator
def profile(func: Callable) -> Callable:
    """Decorator for function profiling."""
    if PSUTIL_AVAILABLE:
        tracker = MemoryTracker()
        return tracker.memory_profile(func)
    else:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
            return result
        return wrapper
def cache(max_size: int = 128):
    """Decorator for function result caching."""
    cache_manager = CacheManager(max_size)
    def decorator(func: Callable) -> Callable:
        return cache_manager.cached_function(func)
    return decorator
def gpu_accelerate(func: Callable) -> Callable:
    """Decorator for GPU acceleration."""
    accelerator = GPUAccelerator()
    return accelerator.gpu_accelerated(func)
def parallel(method: str = 'thread', max_workers: Optional[int] = None):
    """Decorator for parallel execution."""
    executor = ParallelExecutor(max_workers)
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(data, *args, **kwargs):
            if isinstance(data, (list, tuple, np.ndarray)):
                return executor.parallel_map(lambda x: func(x, *args, **kwargs), data, method)
            else:
                return func(data, *args, **kwargs)
        return wrapper
    return decorator
# Example optimized functions for common operations
if NUMBA_AVAILABLE:
    @numba.jit(nopython=True, cache=True)
    def fast_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Optimized matrix multiplication using Numba."""
        return np.dot(A, B)
    @numba.jit(nopython=True, cache=True, parallel=True)
    def fast_element_wise_operation(array: np.ndarray, operation: str) -> np.ndarray:
        """Fast element-wise operations."""
        result = np.empty_like(array)
        for i in numba.prange(array.size):
            if operation == 'square':
                result.flat[i] = array.flat[i] ** 2
            elif operation == 'sqrt':
                result.flat[i] = np.sqrt(array.flat[i])
            elif operation == 'exp':
                result.flat[i] = np.exp(array.flat[i])
            else:
                result.flat[i] = array.flat[i]
        return result
else:
    def fast_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiplication fallback."""
        return np.dot(A, B)
    def fast_element_wise_operation(array: np.ndarray, operation: str) -> np.ndarray:
        """Element-wise operations fallback."""
        if operation == 'square':
            return array ** 2
        elif operation == 'sqrt':
            return np.sqrt(array)
        elif operation == 'exp':
            return np.exp(array)
        else:
            return array
# Performance analysis utilities
def compare_implementations(*implementations, test_data=None, n_runs: int = 10):
    """
    Compare performance of multiple implementations.
    Args:
        implementations: Functions to compare
        test_data: Test data (generated if None)
        n_runs: Number of benchmark runs
    Returns:
        Performance comparison results
    """
    if test_data is None:
        test_data = (np.random.randn(1000, 1000), np.random.randn(1000, 1000))
    optimizer = PerformanceOptimizer()
    results = {}
    for impl in implementations:
        print(f"\nBenchmarking {impl.__name__}...")
        stats = optimizer.benchmark_function(impl, *test_data, n_runs=n_runs)
        results[impl.__name__] = stats
        print(f"  Mean time: {stats['mean_time']:.4f}s")
        print(f"  Std time: {stats['std_time']:.4f}s")
        if 'mean_memory' in stats:
            print(f"  Memory usage: {stats['mean_memory']:.2f}MB")
    # Find fastest implementation
    fastest = min(results.keys(), key=lambda k: results[k]['mean_time'])
    print(f"\nFastest implementation: {fastest}")
    return results