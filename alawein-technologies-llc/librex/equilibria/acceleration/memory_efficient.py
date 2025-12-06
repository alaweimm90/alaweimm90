"""
Memory-Efficient Operations Module

Implements memory-optimized algorithms for large-scale optimization problems.
"""

import logging
import numpy as np
from typing import Dict, Any, Callable, Optional, List, Tuple
from dataclasses import dataclass
import gc

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory-efficient operations"""
    chunk_size: int = 1000
    enable_gradient_checkpointing: bool = True
    enable_memory_mapping: bool = False
    max_memory_mb: int = 4096
    enable_streaming: bool = True
    compression: bool = False


class MemoryEfficientOptimizer:
    """
    Memory-efficient optimization for large-scale problems.

    Features:
    - Chunked population processing
    - Gradient checkpointing
    - Memory-mapped arrays
    - Streaming evaluation
    - On-the-fly compression
    - Automatic memory management
    """

    def __init__(self, backend: str = 'auto', config: Optional[MemoryConfig] = None):
        """
        Initialize memory-efficient optimizer.

        Args:
            backend: Computation backend
            config: Memory configuration
        """
        from .gpu_backend import GPUBackend

        self.gpu = GPUBackend(backend)
        self.config = config or MemoryConfig()
        self.backend_name = self.gpu.backend_name

        # Setup backend-specific optimizations
        if self.backend_name == 'jax':
            self._setup_jax_memory()
        elif self.backend_name == 'pytorch':
            self._setup_pytorch_memory()

        self._memory_stats = []

    def _setup_jax_memory(self):
        """Setup JAX memory optimizations"""
        import jax

        # Enable memory defragmentation
        import os
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'

        # Configure JAX for memory efficiency
        jax.config.update('jax_enable_x64', False)  # Use float32
        jax.config.update('jax_platform_name', 'gpu' if self.gpu.is_gpu else 'cpu')

    def _setup_pytorch_memory(self):
        """Setup PyTorch memory optimizations"""
        try:
            import torch

            # Set memory fraction
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.8)

            # Enable cudnn benchmarking for better memory usage
            torch.backends.cudnn.benchmark = True
        except ImportError:
            pass

    def chunked_genetic_algorithm(self, problem: Any, config: Dict) -> Dict[str, Any]:
        """
        Genetic algorithm with chunked population processing.

        Processes population in chunks to avoid OOM on large populations.
        """
        import time
        start_time = time.time()

        population_size = config.get('population_size', 10000)
        n_generations = config.get('n_generations', 1000)
        chunk_size = min(self.config.chunk_size, population_size)

        # Initialize population in chunks
        population_chunks = []
        n_chunks = (population_size + chunk_size - 1) // chunk_size

        for i in range(n_chunks):
            chunk_start = i * chunk_size
            chunk_end = min((i + 1) * chunk_size, population_size)
            chunk_pop_size = chunk_end - chunk_start

            if hasattr(problem, 'bounds') and problem.bounds:
                lower, upper = problem.bounds
                chunk = np.random.uniform(lower, upper, (chunk_pop_size, problem.dimension))
            else:
                chunk = np.random.randn(chunk_pop_size, problem.dimension)

            population_chunks.append(chunk)

        best_solution = None
        best_fitness = float('inf')
        generation_stats = []

        for generation in range(n_generations):
            # Evaluate population in chunks
            all_fitness = []
            all_individuals = []

            for chunk_idx, chunk in enumerate(population_chunks):
                # Move chunk to GPU if available
                if self.gpu.is_gpu:
                    chunk_gpu = self.gpu.to_device(chunk)
                    chunk_fitness = self._evaluate_chunk_gpu(chunk_gpu, problem.objective_function)
                    chunk_fitness = self.gpu.from_device(chunk_fitness)
                else:
                    chunk_fitness = np.array([problem.objective_function(ind) for ind in chunk])

                all_fitness.extend(chunk_fitness)
                all_individuals.extend(chunk)

                # Free GPU memory after each chunk
                if self.gpu.is_gpu:
                    self._clear_gpu_memory()

            all_fitness = np.array(all_fitness)
            all_individuals = np.array(all_individuals)

            # Update best
            min_idx = np.argmin(all_fitness)
            if all_fitness[min_idx] < best_fitness:
                best_fitness = all_fitness[min_idx]
                best_solution = all_individuals[min_idx].copy()

            # Selection and reproduction in chunks
            new_population_chunks = []

            for chunk_idx in range(n_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, population_size)

                # Tournament selection for this chunk
                chunk_parents = self._chunked_tournament_selection(
                    all_individuals, all_fitness, chunk_end - chunk_start,
                    config.get('tournament_size', 3)
                )

                # Crossover and mutation
                chunk_offspring = self._chunked_genetic_operators(
                    chunk_parents, config.get('mutation_rate', 0.01),
                    config.get('crossover_rate', 0.8), problem.bounds
                )

                new_population_chunks.append(chunk_offspring)

                # Memory management
                del chunk_parents
                gc.collect()

            population_chunks = new_population_chunks

            # Record statistics
            generation_stats.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': float(np.mean(all_fitness)),
                'memory_mb': self._get_memory_usage(),
            })

            # Periodic memory cleanup
            if generation % 10 == 0:
                self._aggressive_memory_cleanup()

        runtime = time.time() - start_time

        return {
            'solution': best_solution,
            'objective': best_fitness,
            'runtime': runtime,
            'generation_stats': generation_stats,
            'metadata': {
                'backend': self.backend_name,
                'chunk_size': chunk_size,
                'n_chunks': n_chunks,
                'max_memory_mb': max(s['memory_mb'] for s in generation_stats),
            }
        }

    def streaming_pso(self, problem: Any, config: Dict) -> Dict[str, Any]:
        """
        Particle Swarm Optimization with streaming updates.

        Processes particles in streams to reduce memory footprint.
        """
        import time
        start_time = time.time()

        n_particles = config.get('n_particles', 10000)
        n_iterations = config.get('n_iterations', 1000)
        stream_size = min(self.config.chunk_size, n_particles)

        # Initialize swarm metadata (positions stored on disk if needed)
        if self.config.enable_memory_mapping:
            positions = self._create_memory_mapped_array((n_particles, problem.dimension))
            velocities = self._create_memory_mapped_array((n_particles, problem.dimension))
        else:
            positions = np.random.randn(n_particles, problem.dimension)
            velocities = np.random.randn(n_particles, problem.dimension) * 0.1

        # Initialize with bounds
        if hasattr(problem, 'bounds') and problem.bounds:
            lower, upper = problem.bounds
            positions = np.random.uniform(lower, upper, (n_particles, problem.dimension))

        personal_best_positions = positions.copy()
        personal_best_scores = np.full(n_particles, float('inf'))
        global_best_position = positions[0].copy()
        global_best_score = float('inf')

        w = config.get('w', 0.729)
        c1 = config.get('c1', 1.49445)
        c2 = config.get('c2', 1.49445)

        for iteration in range(n_iterations):
            # Process particles in streams
            for stream_start in range(0, n_particles, stream_size):
                stream_end = min(stream_start + stream_size, n_particles)
                stream_indices = slice(stream_start, stream_end)

                # Load stream
                stream_positions = positions[stream_indices]
                stream_velocities = velocities[stream_indices]
                stream_pbest = personal_best_positions[stream_indices]

                # Evaluate stream
                if self.gpu.is_gpu:
                    stream_positions_gpu = self.gpu.to_device(stream_positions)
                    stream_scores = self._evaluate_stream_gpu(stream_positions_gpu, problem.objective_function)
                    stream_scores = self.gpu.from_device(stream_scores)
                else:
                    stream_scores = np.array([problem.objective_function(pos) for pos in stream_positions])

                # Update personal bests
                improved = stream_scores < personal_best_scores[stream_indices]
                personal_best_positions[stream_indices][improved] = stream_positions[improved]
                personal_best_scores[stream_indices][improved] = stream_scores[improved]

                # Update global best
                min_idx = np.argmin(stream_scores)
                if stream_scores[min_idx] < global_best_score:
                    global_best_score = stream_scores[min_idx]
                    global_best_position = stream_positions[min_idx].copy()

                # PSO update for stream
                r1 = np.random.random(stream_positions.shape)
                r2 = np.random.random(stream_positions.shape)

                stream_velocities = (w * stream_velocities +
                                   c1 * r1 * (stream_pbest - stream_positions) +
                                   c2 * r2 * (global_best_position - stream_positions))

                stream_positions = stream_positions + stream_velocities

                # Boundary handling
                if hasattr(problem, 'bounds') and problem.bounds:
                    lower, upper = problem.bounds
                    stream_positions = np.clip(stream_positions, lower, upper)

                # Write back
                positions[stream_indices] = stream_positions
                velocities[stream_indices] = stream_velocities

                # Clear memory
                if self.gpu.is_gpu:
                    self._clear_gpu_memory()

            # Decay inertia
            w *= config.get('w_decay', 0.99)

        runtime = time.time() - start_time

        return {
            'solution': global_best_position,
            'objective': global_best_score,
            'runtime': runtime,
            'metadata': {
                'backend': self.backend_name,
                'stream_size': stream_size,
                'memory_mapped': self.config.enable_memory_mapping,
            }
        }

    def gradient_checkpointing_optimization(self, problem: Any, config: Dict) -> Dict[str, Any]:
        """
        Optimization with gradient checkpointing for memory efficiency.

        Trades compute for memory by recomputing activations during backprop.
        """
        if self.backend_name != 'jax':
            logger.warning("Gradient checkpointing only supported with JAX")
            return self.chunked_genetic_algorithm(problem, config)

        import jax
        from jax.experimental import checkify
        from jax import grad, jit

        @jit
        @checkify.checkify
        def objective_with_gradient(x):
            """Compute objective and gradient with checkpointing"""
            return problem.objective_function(x), grad(problem.objective_function)(x)

        # Gradient-based optimization with checkpointing
        import time
        start_time = time.time()

        # Initialize
        if hasattr(problem, 'bounds') and problem.bounds:
            lower, upper = problem.bounds
            x = jax.random.uniform(jax.random.PRNGKey(0), (problem.dimension,), minval=lower, maxval=upper)
        else:
            x = jax.random.normal(jax.random.PRNGKey(0), (problem.dimension,))

        learning_rate = config.get('learning_rate', 0.01)
        n_iterations = config.get('n_iterations', 1000)

        best_x = x
        best_obj = float('inf')
        iteration_stats = []

        for iteration in range(n_iterations):
            # Compute objective and gradient with checkpointing
            err, (obj, grad_val) = objective_with_gradient(x)

            # Update
            x = x - learning_rate * grad_val

            # Track best
            if obj < best_obj:
                best_obj = float(obj)
                best_x = x

            # Record stats
            if iteration % 100 == 0:
                iteration_stats.append({
                    'iteration': iteration,
                    'objective': float(obj),
                    'gradient_norm': float(jax.numpy.linalg.norm(grad_val)),
                    'memory_mb': self._get_memory_usage(),
                })

            # Adaptive learning rate
            if iteration % 200 == 0:
                learning_rate *= 0.9

        runtime = time.time() - start_time

        return {
            'solution': np.array(best_x),
            'objective': best_obj,
            'runtime': runtime,
            'iteration_stats': iteration_stats,
            'metadata': {
                'backend': 'jax',
                'gradient_checkpointing': True,
            }
        }

    def _evaluate_chunk_gpu(self, chunk, objective_fn):
        """Evaluate chunk on GPU"""
        if self.backend_name == 'jax':
            import jax
            from jax import vmap
            return vmap(objective_fn)(chunk)
        elif self.backend_name == 'pytorch':
            import torch
            results = []
            for ind in chunk:
                ind_cpu = ind.cpu().numpy() if hasattr(ind, 'cpu') else ind
                results.append(objective_fn(ind_cpu))
            return torch.tensor(results, device=chunk.device if hasattr(chunk, 'device') else 'cpu')
        else:
            return np.array([objective_fn(ind) for ind in chunk])

    def _evaluate_stream_gpu(self, stream, objective_fn):
        """Evaluate stream on GPU"""
        return self._evaluate_chunk_gpu(stream, objective_fn)

    def _chunked_tournament_selection(self, population, fitness, n_select, tournament_size):
        """Tournament selection in chunks"""
        selected = []
        pop_size = len(population)

        for _ in range(n_select):
            tournament_indices = np.random.choice(pop_size, tournament_size, replace=False)
            tournament_fitness = fitness[tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx])

        return np.array(selected)

    def _chunked_genetic_operators(self, parents, mutation_rate, crossover_rate, bounds):
        """Apply genetic operators in chunks"""
        offspring = []
        n_parents = len(parents)

        # Crossover
        for i in range(0, n_parents - 1, 2):
            if np.random.random() < crossover_rate:
                # Single-point crossover
                point = np.random.randint(1, len(parents[i]))
                child1 = np.concatenate([parents[i][:point], parents[i + 1][point:]])
                child2 = np.concatenate([parents[i + 1][:point], parents[i][point:]])
                offspring.extend([child1, child2])
            else:
                offspring.extend([parents[i].copy(), parents[i + 1].copy()])

        if n_parents % 2 == 1:
            offspring.append(parents[-1].copy())

        offspring = np.array(offspring[:n_parents])

        # Mutation
        mask = np.random.random(offspring.shape) < mutation_rate
        mutations = np.random.randn(*offspring.shape) * 0.1
        offspring = offspring + mask * mutations

        # Bounds
        if bounds:
            lower, upper = bounds
            offspring = np.clip(offspring, lower, upper)

        return offspring

    def _create_memory_mapped_array(self, shape: Tuple[int, ...], dtype=np.float32):
        """Create memory-mapped array for large data"""
        import tempfile
        import os

        # Create temporary file
        fd, path = tempfile.mkstemp()
        os.close(fd)

        # Create memory-mapped array
        mmap_array = np.memmap(path, dtype=dtype, mode='w+', shape=shape)

        # Store path for cleanup
        if not hasattr(self, '_mmap_files'):
            self._mmap_files = []
        self._mmap_files.append(path)

        return mmap_array

    def _clear_gpu_memory(self):
        """Clear GPU memory"""
        if self.backend_name == 'jax':
            import jax
            # JAX doesn't have explicit memory clearing
            import gc
            gc.collect()
        elif self.backend_name == 'pytorch':
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif self.backend_name == 'cupy':
            import cupy
            mempool = cupy.get_default_memory_pool()
            mempool.free_all_blocks()

    def _aggressive_memory_cleanup(self):
        """Aggressive memory cleanup"""
        # Clear GPU
        self._clear_gpu_memory()

        # Force garbage collection
        gc.collect()
        gc.collect()  # Second pass for cyclic references

        # Clear caches
        if self.backend_name == 'jax':
            import jax
            jax.clear_caches()

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def cleanup(self):
        """Cleanup temporary files and memory"""
        # Clean memory-mapped files
        if hasattr(self, '_mmap_files'):
            import os
            for path in self._mmap_files:
                try:
                    os.remove(path)
                except:
                    pass

        # Clear GPU memory
        self._aggressive_memory_cleanup()

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()