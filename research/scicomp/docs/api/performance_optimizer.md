# performance_optimizer
**Module:** `Python/utils/performance_optimizer.py`
## Overview
Performance optimization utilities for SciComp.
This module provides advanced performance optimization including:
- Just-in-time compilation with Numba
- Memory management and profiling
- Parallel processing utilities
- GPU acceleration helpers
- Performance benchmarking tools
## Functions
### `optimize(method)`
Decorator for automatic function optimization.
**Source:** [Line 417](Python/utils/performance_optimizer.py#L417)
### `profile(func)`
Decorator for function profiling.
**Source:** [Line 427](Python/utils/performance_optimizer.py#L427)
### `cache(max_size)`
Decorator for function result caching.
**Source:** [Line 443](Python/utils/performance_optimizer.py#L443)
### `gpu_accelerate(func)`
Decorator for GPU acceleration.
**Source:** [Line 453](Python/utils/performance_optimizer.py#L453)
### `parallel(method, max_workers)`
Decorator for parallel execution.
**Source:** [Line 459](Python/utils/performance_optimizer.py#L459)
### `compare_implementations()`
Compare performance of multiple implementations.
Args:
implementations: Functions to compare
test_data: Test data (generated if None)
n_runs: Number of benchmark runs
Returns:
Performance comparison results
**Source:** [Line 515](Python/utils/performance_optimizer.py#L515)
## Classes
### `PerformanceOptimizer`
Main performance optimization class.
#### Methods
##### `__init__(self)`
Initialize performance optimizer.
**Source:** [Line 46](Python/utils/performance_optimizer.py#L46)
##### `optimize_function(self, func, method)`
Optimize function using specified method.
Args:
func: Function to optimize
method: Optimization method ('numba', 'vectorize', 'parallel', 'auto')
Returns:
Optimized function
**Source:** [Line 52](Python/utils/performance_optimizer.py#L52)
##### `_choose_optimization_method(self, func)`
Automatically choose best optimization method.
**Source:** [Line 75](Python/utils/performance_optimizer.py#L75)
##### `_apply_numba_optimization(self, func)`
Apply Numba JIT compilation.
**Source:** [Line 82](Python/utils/performance_optimizer.py#L82)
##### `_apply_vectorization(self, func)`
Apply numpy vectorization.
**Source:** [Line 90](Python/utils/performance_optimizer.py#L90)
##### `_apply_parallelization(self, func)`
Apply parallel execution wrapper.
**Source:** [Line 94](Python/utils/performance_optimizer.py#L94)
##### `benchmark_function(self, func)`
Benchmark function performance.
Args:
func: Function to benchmark
args: Function arguments
n_runs: Number of benchmark runs
kwargs: Function keyword arguments
Returns:
Performance statistics
**Source:** [Line 101](Python/utils/performance_optimizer.py#L101)
**Class Source:** [Line 43](Python/utils/performance_optimizer.py#L43)
### `MemoryTracker`
Memory usage tracking utilities.
#### Methods
##### `__init__(self)`
Initialize memory tracker.
**Source:** [Line 152](Python/utils/performance_optimizer.py#L152)
##### `get_memory_usage(self)`
Get current memory usage in MB.
**Source:** [Line 158](Python/utils/performance_optimizer.py#L158)
##### `get_memory_percent(self)`
Get memory usage as percentage of system memory.
**Source:** [Line 162](Python/utils/performance_optimizer.py#L162)
##### `memory_profile(self, func)`
Decorator to profile memory usage of function.
**Source:** [Line 166](Python/utils/performance_optimizer.py#L166)
**Class Source:** [Line 149](Python/utils/performance_optimizer.py#L149)
### `ParallelExecutor`
Parallel execution utilities.
#### Methods
##### `__init__(self, max_workers)`
Initialize parallel executor.
Args:
max_workers: Maximum number of worker threads/processes
**Source:** [Line 186](Python/utils/performance_optimizer.py#L186)
##### `execute_parallel(self, func, data, method, chunk_size)`
Execute function in parallel over data.
Args:
func: Function to execute
data: Data to process in parallel
method: Parallelization method ('thread' or 'process')
chunk_size: Size of chunks for processing
Returns:
Results from parallel execution
**Source:** [Line 195](Python/utils/performance_optimizer.py#L195)
##### `parallel_map(self, func, data, method)`
Apply function to data in parallel (like built-in map).
Args:
func: Function to apply
data: Data to map over
method: Parallelization method
Returns:
Mapped results
**Source:** [Line 228](Python/utils/performance_optimizer.py#L228)
**Class Source:** [Line 183](Python/utils/performance_optimizer.py#L183)
### `GPUAccelerator`
GPU acceleration utilities using CuPy.
#### Methods
##### `__init__(self)`
Initialize GPU accelerator.
**Source:** [Line 246](Python/utils/performance_optimizer.py#L246)
##### `to_gpu(self, array)`
Transfer array to GPU if available.
**Source:** [Line 252](Python/utils/performance_optimizer.py#L252)
##### `to_cpu(self, array)`
Transfer array to CPU.
**Source:** [Line 258](Python/utils/performance_optimizer.py#L258)
##### `gpu_accelerated(self, func)`
Decorator to automatically use GPU acceleration.
**Source:** [Line 264](Python/utils/performance_optimizer.py#L264)
**Class Source:** [Line 243](Python/utils/performance_optimizer.py#L243)
### `CacheManager`
Intelligent caching for expensive computations.
#### Methods
##### `__init__(self, max_size)`
Initialize cache manager.
Args:
max_size: Maximum cache size
**Source:** [Line 301](Python/utils/performance_optimizer.py#L301)
##### `cached_function(self, func)`
Decorator to cache function results.
**Source:** [Line 312](Python/utils/performance_optimizer.py#L312)
##### `_make_cache_key(self, func_name, args, kwargs)`
Create cache key from function arguments.
**Source:** [Line 334](Python/utils/performance_optimizer.py#L334)
##### `_store_in_cache(self, key, result)`
Store result in cache with LRU eviction.
**Source:** [Line 353](Python/utils/performance_optimizer.py#L353)
##### `clear_cache(self)`
Clear the entire cache.
**Source:** [Line 364](Python/utils/performance_optimizer.py#L364)
##### `cache_info(self)`
Get cache statistics.
**Source:** [Line 369](Python/utils/performance_optimizer.py#L369)
**Class Source:** [Line 298](Python/utils/performance_optimizer.py#L298)
### `ProfilerContext`
Context manager for performance profiling.
#### Methods
##### `__init__(self, name)`
Initialize profiler context.
Args:
name: Name of operation being profiled
**Source:** [Line 382](Python/utils/performance_optimizer.py#L382)
##### `__enter__(self)`
Enter profiling context.
**Source:** [Line 394](Python/utils/performance_optimizer.py#L394)
##### `__exit__(self, exc_type, exc_val, exc_tb)`
Exit profiling context and report results.
**Source:** [Line 401](Python/utils/performance_optimizer.py#L401)
**Class Source:** [Line 379](Python/utils/performance_optimizer.py#L379)