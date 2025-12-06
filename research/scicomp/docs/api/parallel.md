# parallel
**Module:** `Python/utils/parallel.py`
## Overview
Parallelization Utilities Module
Provides tools for parallel and distributed computing in scientific applications,
including multiprocessing, memory-efficient computation, and distributed computing.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
## Functions
### `get_optimal_workers(task_complexity, memory_usage, max_workers)`
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
**Source:** [Line 26](Python/utils/parallel.py#L26)
### `parallel_map(func, iterable, n_jobs, backend, verbose, batch_size)`
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
**Source:** [Line 66](Python/utils/parallel.py#L66)
### `_parallel_map_threading(func, iterable, n_jobs, verbose)`
Threading-based parallel map.
**Source:** [Line 114](Python/utils/parallel.py#L114)
### `_parallel_map_multiprocessing(func, iterable, n_jobs, verbose)`
Multiprocessing-based parallel map.
**Source:** [Line 131](Python/utils/parallel.py#L131)
### `_parallel_map_concurrent(func, iterable, n_jobs, verbose)`
ProcessPoolExecutor-based parallel map.
**Source:** [Line 142](Python/utils/parallel.py#L142)
### `parallel_compute(func, data, axis, n_jobs, chunk_size)`
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
**Source:** [Line 153](Python/utils/parallel.py#L153)
### `memory_efficient_compute(func, data, chunk_size, axis, n_jobs, progress_callback)`
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
**Source:** [Line 191](Python/utils/parallel.py#L191)
### `distributed_compute(func, data_chunks, n_workers, timeout)`
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
**Source:** [Line 300](Python/utils/parallel.py#L300)
### `chunked_operation(operation, arrays, chunk_size, axis, combine_func)`
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
**Source:** [Line 346](Python/utils/parallel.py#L346)
## Classes
### `ProgressMonitor`
Simple progress monitor for parallel computations.
#### Methods
##### `__init__(self, total_tasks, update_interval)`
Initialize progress monitor.
Parameters
----------
total_tasks : int
Total number of tasks
update_interval : float, default 1.0
Update interval in seconds
**Source:** [Line 251](Python/utils/parallel.py#L251)
##### `update(self, n_completed)`
Update progress counter.
**Source:** [Line 269](Python/utils/parallel.py#L269)
##### `_print_progress(self)`
Print current progress.
**Source:** [Line 279](Python/utils/parallel.py#L279)
##### `finish(self)`
Print final progress.
**Source:** [Line 293](Python/utils/parallel.py#L293)
**Class Source:** [Line 248](Python/utils/parallel.py#L248)
### `ResourceMonitor`
Monitor system resources during computation.
#### Methods
##### `__init__(self, monitoring_interval)`
Initialize resource monitor.
Parameters
----------
monitoring_interval : float, default 1.0
Monitoring interval in seconds
**Source:** [Line 401](Python/utils/parallel.py#L401)
##### `start_monitoring(self)`
Start resource monitoring.
**Source:** [Line 419](Python/utils/parallel.py#L419)
##### `stop_monitoring(self)`
Stop monitoring and return collected data.
**Source:** [Line 425](Python/utils/parallel.py#L425)
##### `_monitor_loop(self)`
Main monitoring loop.
**Source:** [Line 432](Python/utils/parallel.py#L432)
##### `get_statistics(self)`
Get resource usage statistics.
**Source:** [Line 447](Python/utils/parallel.py#L447)
**Class Source:** [Line 398](Python/utils/parallel.py#L398)