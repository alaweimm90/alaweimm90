"""
Parallel Evaluation Module

High-performance parallel objective function evaluation using multiple backends.
"""

import logging
import multiprocessing
from typing import List, Callable, Any, Dict, Optional, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for parallel evaluation performance"""
    total_time: float
    avg_time_per_eval: float
    throughput: float  # evaluations per second
    speedup: float  # vs sequential
    efficiency: float  # speedup / n_workers
    n_evaluations: int
    n_workers: int


class ParallelEvaluator:
    """
    Parallel evaluation with multiple backends.

    Supports:
    - CPU multiprocessing
    - GPU vectorized evaluation
    - Distributed evaluation with Ray
    - Async evaluation
    - Batch processing
    - Caching
    """

    def __init__(self, backend: str = 'auto', n_workers: Optional[int] = None,
                 cache_size: int = 10000):
        """
        Initialize parallel evaluator.

        Args:
            backend: Evaluation backend ('cpu', 'gpu', 'ray', 'auto')
            n_workers: Number of parallel workers (None = auto-detect)
            cache_size: Size of evaluation cache
        """
        self.backend = self._select_backend(backend)
        self.n_workers = n_workers or self._detect_workers()
        self.cache_size = cache_size
        self._cache = {}
        self._metrics = None

        logger.info(f"ParallelEvaluator initialized: backend={self.backend}, n_workers={self.n_workers}")

    def _select_backend(self, backend: str) -> str:
        """Select evaluation backend"""
        if backend != 'auto':
            return backend

        # Auto-detect best backend
        try:
            import jax
            if len(jax.devices()) > 1 or any(d.platform != 'cpu' for d in jax.devices()):
                return 'gpu_jax'
        except ImportError:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                return 'gpu_torch'
        except ImportError:
            pass

        try:
            import ray
            if ray.is_initialized():
                return 'ray'
        except ImportError:
            pass

        return 'cpu'

    def _detect_workers(self) -> int:
        """Auto-detect optimal number of workers"""
        if self.backend == 'cpu':
            return multiprocessing.cpu_count()
        elif self.backend.startswith('gpu'):
            try:
                import torch
                return torch.cuda.device_count() if torch.cuda.is_available() else 1
            except ImportError:
                return 1
        elif self.backend == 'ray':
            try:
                import ray
                return int(ray.cluster_resources().get('CPU', multiprocessing.cpu_count()))
            except ImportError:
                return multiprocessing.cpu_count()
        else:
            return multiprocessing.cpu_count()

    def evaluate_batch(self, objective_fn: Callable, solutions: Union[List, np.ndarray],
                      use_cache: bool = True, measure_performance: bool = False) -> Union[List[float], np.ndarray]:
        """
        Evaluate batch of solutions in parallel.

        Args:
            objective_fn: Objective function to evaluate
            solutions: List or array of solutions
            use_cache: Whether to use caching
            measure_performance: Whether to measure performance metrics

        Returns:
            Fitness values for all solutions
        """
        start_time = time.time()

        # Convert to appropriate format
        if isinstance(solutions, list):
            solutions_array = np.array(solutions)
        else:
            solutions_array = solutions

        # Check cache
        if use_cache:
            results, uncached_indices = self._check_cache(solutions_array)
            if len(uncached_indices) == 0:
                return results
            uncached_solutions = solutions_array[uncached_indices]
        else:
            uncached_solutions = solutions_array
            uncached_indices = list(range(len(solutions_array)))

        # Dispatch to appropriate backend
        if self.backend == 'cpu':
            uncached_results = self._evaluate_batch_cpu(objective_fn, uncached_solutions)
        elif self.backend == 'gpu_jax':
            uncached_results = self._evaluate_batch_gpu_jax(objective_fn, uncached_solutions)
        elif self.backend == 'gpu_torch':
            uncached_results = self._evaluate_batch_gpu_torch(objective_fn, uncached_solutions)
        elif self.backend == 'ray':
            uncached_results = self._evaluate_batch_ray(objective_fn, uncached_solutions)
        elif self.backend == 'thread':
            uncached_results = self._evaluate_batch_thread(objective_fn, uncached_solutions)
        else:
            # Fallback to sequential
            uncached_results = [objective_fn(sol) for sol in uncached_solutions]

        # Update cache
        if use_cache:
            self._update_cache(uncached_solutions, uncached_results)
            # Merge results
            for idx, result in zip(uncached_indices, uncached_results):
                results[idx] = result
        else:
            results = uncached_results

        # Measure performance
        if measure_performance:
            elapsed_time = time.time() - start_time
            self._metrics = self._calculate_metrics(
                elapsed_time, len(solutions_array), objective_fn, solutions_array[0]
            )

        return results if isinstance(solutions, list) else np.array(results)

    def _evaluate_batch_cpu(self, objective_fn: Callable, solutions: np.ndarray) -> List[float]:
        """CPU multiprocessing evaluation"""
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(objective_fn, sol) for sol in solutions]

            # Collect results in order
            results = []
            for future in futures:
                results.append(future.result())

        return results

    def _evaluate_batch_thread(self, objective_fn: Callable, solutions: np.ndarray) -> List[float]:
        """Thread-based evaluation (good for I/O bound tasks)"""
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(objective_fn, sol) for sol in solutions]
            results = [future.result() for future in futures]

        return results

    def _evaluate_batch_gpu_jax(self, objective_fn: Callable, solutions: np.ndarray) -> np.ndarray:
        """GPU vectorized evaluation using JAX"""
        try:
            import jax
            import jax.numpy as jnp
            from jax import vmap, jit

            # Move to GPU
            solutions_gpu = jax.device_put(solutions)

            # JIT compile and vectorize
            @jit
            def batched_objective(batch):
                return vmap(objective_fn)(batch)

            # Evaluate entire batch on GPU
            results = batched_objective(solutions_gpu)

            return np.array(results)

        except Exception as e:
            logger.warning(f"GPU evaluation failed, falling back to CPU: {e}")
            return self._evaluate_batch_cpu(objective_fn, solutions)

    def _evaluate_batch_gpu_torch(self, objective_fn: Callable, solutions: np.ndarray) -> np.ndarray:
        """GPU evaluation using PyTorch"""
        try:
            import torch

            # Move to GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            solutions_gpu = torch.tensor(solutions, device=device, dtype=torch.float32)

            # Batch evaluation
            results = []
            batch_size = 1000  # Process in chunks to avoid OOM

            for i in range(0, len(solutions_gpu), batch_size):
                batch = solutions_gpu[i:i+batch_size]

                # Evaluate batch
                batch_results = []
                for sol in batch:
                    # Convert to numpy for objective function
                    sol_numpy = sol.cpu().numpy()
                    result = objective_fn(sol_numpy)
                    batch_results.append(result)

                results.extend(batch_results)

            return np.array(results)

        except Exception as e:
            logger.warning(f"GPU evaluation failed, falling back to CPU: {e}")
            return self._evaluate_batch_cpu(objective_fn, solutions)

    def _evaluate_batch_ray(self, objective_fn: Callable, solutions: np.ndarray) -> List[float]:
        """Distributed evaluation using Ray"""
        try:
            import ray

            if not ray.is_initialized():
                ray.init()

            # Create remote function
            @ray.remote
            def evaluate_remote(sol):
                return objective_fn(sol)

            # Submit all evaluations
            futures = [evaluate_remote.remote(sol) for sol in solutions]

            # Get results
            results = ray.get(futures)

            return results

        except Exception as e:
            logger.warning(f"Ray evaluation failed, falling back to CPU: {e}")
            return self._evaluate_batch_cpu(objective_fn, solutions)

    def evaluate_async(self, objective_fn: Callable, solutions: List,
                       callback: Optional[Callable] = None) -> List:
        """
        Asynchronous evaluation with optional callback.

        Args:
            objective_fn: Objective function
            solutions: Solutions to evaluate
            callback: Optional callback for each completed evaluation

        Returns:
            Future-like objects for results
        """
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(objective_fn, sol): i for i, sol in enumerate(solutions)}
            results = [None] * len(solutions)

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results[idx] = result

                    if callback:
                        callback(idx, result)

                except Exception as e:
                    logger.error(f"Evaluation failed for solution {idx}: {e}")
                    results[idx] = float('inf')

        return results

    def evaluate_chunked(self, objective_fn: Callable, solutions: List,
                        chunk_size: int = 1000) -> List[float]:
        """
        Evaluate in chunks to handle very large populations.

        Args:
            objective_fn: Objective function
            solutions: Solutions to evaluate
            chunk_size: Size of each chunk

        Returns:
            Fitness values
        """
        results = []
        n_solutions = len(solutions)

        for i in range(0, n_solutions, chunk_size):
            chunk = solutions[i:min(i+chunk_size, n_solutions)]
            chunk_results = self.evaluate_batch(objective_fn, chunk, use_cache=True)
            results.extend(chunk_results)

            # Clear GPU memory if needed
            if self.backend.startswith('gpu'):
                self._clear_gpu_memory()

        return results

    def _check_cache(self, solutions: np.ndarray) -> tuple:
        """Check cache for already evaluated solutions"""
        results = [None] * len(solutions)
        uncached_indices = []

        for i, sol in enumerate(solutions):
            cache_key = self._make_cache_key(sol)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_indices.append(i)

        return results, uncached_indices

    def _update_cache(self, solutions: np.ndarray, results: List[float]):
        """Update cache with new evaluations"""
        for sol, result in zip(solutions, results):
            cache_key = self._make_cache_key(sol)
            self._cache[cache_key] = result

            # Limit cache size
            if len(self._cache) > self.cache_size:
                # Remove oldest entry (FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

    def _make_cache_key(self, solution: np.ndarray) -> str:
        """Create cache key from solution"""
        return hash(solution.tobytes())

    def _clear_gpu_memory(self):
        """Clear GPU memory"""
        if self.backend == 'gpu_torch':
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        elif self.backend == 'gpu_jax':
            try:
                import jax
                # JAX doesn't have explicit cache clearing
                import gc
                gc.collect()
            except ImportError:
                pass

    def _calculate_metrics(self, elapsed_time: float, n_evaluations: int,
                          objective_fn: Callable, sample_solution: np.ndarray) -> EvaluationMetrics:
        """Calculate performance metrics"""
        # Measure sequential time on small sample
        sample_size = min(10, n_evaluations)
        seq_start = time.time()
        for _ in range(sample_size):
            objective_fn(sample_solution)
        seq_time = (time.time() - seq_start) / sample_size

        # Estimate total sequential time
        estimated_seq_time = seq_time * n_evaluations

        # Calculate metrics
        speedup = estimated_seq_time / elapsed_time if elapsed_time > 0 else 1.0
        efficiency = speedup / self.n_workers if self.n_workers > 0 else 1.0

        return EvaluationMetrics(
            total_time=elapsed_time,
            avg_time_per_eval=elapsed_time / n_evaluations if n_evaluations > 0 else 0,
            throughput=n_evaluations / elapsed_time if elapsed_time > 0 else 0,
            speedup=speedup,
            efficiency=efficiency,
            n_evaluations=n_evaluations,
            n_workers=self.n_workers
        )

    def get_metrics(self) -> Optional[EvaluationMetrics]:
        """Get performance metrics from last evaluation"""
        return self._metrics

    def benchmark(self, objective_fn: Callable, problem_sizes: List[int] = None) -> Dict:
        """
        Benchmark parallel evaluation performance.

        Args:
            objective_fn: Objective function to benchmark
            problem_sizes: List of problem sizes to test

        Returns:
            Benchmark results
        """
        if problem_sizes is None:
            problem_sizes = [10, 100, 1000, 5000]

        results = []

        for size in problem_sizes:
            # Generate random solutions
            solutions = np.random.randn(size, 10)  # Assuming 10-dimensional

            # Benchmark parallel evaluation
            start = time.time()
            self.evaluate_batch(objective_fn, solutions, use_cache=False, measure_performance=True)
            parallel_time = time.time() - start

            # Benchmark sequential evaluation
            start = time.time()
            sequential_results = [objective_fn(sol) for sol in solutions]
            sequential_time = time.time() - start

            # Get metrics
            metrics = self.get_metrics()

            results.append({
                'problem_size': size,
                'parallel_time': parallel_time,
                'sequential_time': sequential_time,
                'speedup': sequential_time / parallel_time if parallel_time > 0 else 0,
                'efficiency': metrics.efficiency if metrics else 0,
                'throughput': metrics.throughput if metrics else 0,
            })

            logger.info(f"Size {size}: Speedup={results[-1]['speedup']:.2f}x, "
                       f"Throughput={results[-1]['throughput']:.0f} evals/sec")

        return {
            'backend': self.backend,
            'n_workers': self.n_workers,
            'results': results,
            'average_speedup': np.mean([r['speedup'] for r in results]),
            'average_efficiency': np.mean([r['efficiency'] for r in results]),
        }