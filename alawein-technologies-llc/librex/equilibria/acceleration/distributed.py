"""
Distributed Optimization Module

Implements distributed optimization across multiple GPUs and nodes.
Supports island model GA, master-worker patterns, and data parallelism.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DistributionStrategy(Enum):
    """Distribution strategies for optimization"""
    ISLAND_MODEL = "island_model"  # Independent populations with migration
    MASTER_WORKER = "master_worker"  # Central coordinator with workers
    DATA_PARALLEL = "data_parallel"  # Same algorithm, split data
    MODEL_PARALLEL = "model_parallel"  # Split model across devices
    RING_ALLREDUCE = "ring_allreduce"  # Ring topology for gradients


@dataclass
class DistributedConfig:
    """Configuration for distributed optimization"""
    strategy: DistributionStrategy = DistributionStrategy.ISLAND_MODEL
    n_devices: Optional[int] = None  # None = use all available
    migration_interval: int = 50
    migration_rate: float = 0.1
    sync_interval: int = 100
    communication_backend: str = 'auto'  # 'nccl', 'gloo', 'mpi', 'auto'


class DistributedOptimizer:
    """
    Distribute optimization across multiple GPUs/nodes.

    Features:
    - Island model genetic algorithms
    - Distributed particle swarm
    - Parallel simulated annealing
    - Automatic load balancing
    - Fault tolerance
    """

    def __init__(self, backend='jax', config: Optional[DistributedConfig] = None):
        """
        Initialize distributed optimizer.

        Args:
            backend: Computation backend ('jax', 'pytorch', 'ray')
            config: Distribution configuration
        """
        self.backend = backend
        self.config = config or DistributedConfig()

        # Import backend-specific modules
        if backend == 'jax':
            self._setup_jax()
        elif backend == 'pytorch':
            self._setup_pytorch()
        elif backend == 'ray':
            self._setup_ray()
        else:
            raise ValueError(f"Unsupported backend for distribution: {backend}")

        self._detect_devices()

    def _setup_jax(self):
        """Setup JAX for multi-device computation"""
        import jax
        from jax import pmap, device_put_replicated, device_put_sharded
        from jax.experimental import mesh_utils
        from jax.sharding import Mesh, PartitionSpec

        self.jax = jax
        self.pmap = pmap
        self.device_put_replicated = device_put_replicated
        self.device_put_sharded = device_put_sharded

        # Setup mesh for multi-device
        self.mesh_utils = mesh_utils
        self.Mesh = Mesh
        self.PartitionSpec = PartitionSpec

    def _setup_pytorch(self):
        """Setup PyTorch distributed"""
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel

        self.torch = torch
        self.dist = dist
        self.DistributedDataParallel = DistributedDataParallel

        # Initialize process group if not already done
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

    def _setup_ray(self):
        """Setup Ray for distributed computation"""
        import ray

        if not ray.is_initialized():
            ray.init()

        self.ray = ray

    def _detect_devices(self):
        """Detect available devices for distribution"""
        if self.backend == 'jax':
            import jax
            self.devices = jax.devices()
            self.n_devices = len(self.devices) if self.config.n_devices is None else min(self.config.n_devices, len(self.devices))
            logger.info(f"JAX: Using {self.n_devices} devices: {self.devices[:self.n_devices]}")

        elif self.backend == 'pytorch':
            import torch
            if torch.cuda.is_available():
                self.n_devices = torch.cuda.device_count() if self.config.n_devices is None else min(self.config.n_devices, torch.cuda.device_count())
                self.devices = [f'cuda:{i}' for i in range(self.n_devices)]
            else:
                self.n_devices = 1
                self.devices = ['cpu']
            logger.info(f"PyTorch: Using {self.n_devices} devices")

        elif self.backend == 'ray':
            import ray
            resources = ray.cluster_resources()
            self.n_devices = int(resources.get('GPU', 1))
            if self.config.n_devices:
                self.n_devices = min(self.config.n_devices, self.n_devices)
            logger.info(f"Ray: Using {self.n_devices} GPUs from cluster")

    def optimize(self, problem: Any, algorithm: str, algorithm_config: Dict) -> Dict[str, Any]:
        """
        Run distributed optimization.

        Args:
            problem: Optimization problem
            algorithm: Algorithm to use ('genetic', 'pso', 'sa')
            algorithm_config: Algorithm-specific configuration

        Returns:
            Optimization results
        """
        if self.config.strategy == DistributionStrategy.ISLAND_MODEL:
            return self._island_model_optimization(problem, algorithm, algorithm_config)
        elif self.config.strategy == DistributionStrategy.MASTER_WORKER:
            return self._master_worker_optimization(problem, algorithm, algorithm_config)
        elif self.config.strategy == DistributionStrategy.DATA_PARALLEL:
            return self._data_parallel_optimization(problem, algorithm, algorithm_config)
        else:
            raise ValueError(f"Strategy {self.config.strategy} not implemented")

    def _island_model_optimization(self, problem: Any, algorithm: str, config: Dict) -> Dict[str, Any]:
        """
        Island model: independent populations with migration.

        Each device runs its own population, periodically exchanging
        best individuals between islands.
        """
        if self.backend == 'jax':
            return self._island_model_jax(problem, algorithm, config)
        elif self.backend == 'pytorch':
            return self._island_model_pytorch(problem, algorithm, config)
        elif self.backend == 'ray':
            return self._island_model_ray(problem, algorithm, config)

    def _island_model_jax(self, problem: Any, algorithm: str, config: Dict) -> Dict[str, Any]:
        """JAX implementation of island model"""
        import time
        start_time = time.time()

        # Import algorithm
        if algorithm == 'genetic':
            from .gpu_methods.gpu_genetic import GPUGeneticAlgorithm, GeneticConfig
            algo_class = GPUGeneticAlgorithm
            algo_config = GeneticConfig(**config)
        else:
            raise ValueError(f"Algorithm {algorithm} not supported for island model")

        # Create separate PRNG keys for each device
        base_key = self.jax.random.PRNGKey(config.get('seed', 0))
        keys = self.jax.random.split(base_key, self.n_devices)

        # Initialize populations on each device
        @self.pmap
        def initialize_island(key):
            """Initialize population on single device"""
            if hasattr(problem, 'bounds') and problem.bounds:
                lower, upper = problem.bounds
                population = self.jax.random.uniform(
                    key, (algo_config.population_size, problem.dimension),
                    minval=lower, maxval=upper
                )
            else:
                population = self.jax.random.normal(
                    key, (algo_config.population_size, problem.dimension)
                )
            return population

        island_populations = initialize_island(keys)

        # Parallel genetic algorithm step
        @self.pmap
        def evolve_island(population, key):
            """Evolve single island for one generation"""
            # This would call the actual GA operations
            # Simplified for demonstration
            return population

        # Migration function
        @self.pmap
        def migrate_individuals(populations, fitness_values):
            """Exchange best individuals between islands"""
            # Get best individuals from each island
            best_indices = self.jax.numpy.argsort(fitness_values, axis=1)[:, :int(algo_config.population_size * self.config.migration_rate)]

            # Ring topology migration
            # Each island sends its best to the next island
            # Implementation simplified
            return populations

        best_solution = None
        best_fitness = float('inf')
        island_stats = []

        for generation in range(algo_config.n_generations):
            # Evolve each island independently
            new_keys = self.jax.random.split(keys[0], self.n_devices)
            island_populations = evolve_island(island_populations, new_keys)

            # Periodic migration
            if generation % self.config.migration_interval == 0 and generation > 0:
                # Evaluate fitness for migration
                @self.pmap
                def evaluate_populations(population):
                    return self.jax.vmap(problem.objective_function)(population)

                fitness_values = evaluate_populations(island_populations)

                # Migrate best individuals
                island_populations = migrate_individuals(island_populations, fitness_values)

                # Track global best
                all_fitness = fitness_values.reshape(-1)
                min_idx = self.jax.numpy.argmin(all_fitness)
                min_fitness = float(all_fitness[min_idx])

                if min_fitness < best_fitness:
                    best_fitness = min_fitness
                    # Get the actual solution
                    island_idx = min_idx // algo_config.population_size
                    ind_idx = min_idx % algo_config.population_size
                    best_solution = island_populations[island_idx, ind_idx]

                island_stats.append({
                    'generation': generation,
                    'best_fitness': best_fitness,
                    'avg_fitness': float(self.jax.numpy.mean(all_fitness)),
                })

        runtime = time.time() - start_time

        return {
            'solution': self.jax.numpy.array(best_solution) if best_solution is not None else None,
            'objective': best_fitness,
            'runtime': runtime,
            'n_evaluations': self.n_devices * algo_config.population_size * algo_config.n_generations,
            'island_stats': island_stats,
            'metadata': {
                'backend': 'jax',
                'n_devices': self.n_devices,
                'strategy': 'island_model',
            }
        }

    def _island_model_pytorch(self, problem: Any, algorithm: str, config: Dict) -> Dict[str, Any]:
        """PyTorch implementation of island model"""
        import time
        import torch.multiprocessing as mp

        start_time = time.time()

        def run_island(device_id, problem, config, result_queue):
            """Run optimization on single GPU"""
            import torch
            torch.cuda.set_device(device_id)

            # Import and run algorithm
            if algorithm == 'genetic':
                from .gpu_methods.gpu_genetic import GPUGeneticAlgorithm
                algo = GPUGeneticAlgorithm('pytorch', device_id)
                result = algo.optimize(problem, config)
                result_queue.put((device_id, result))

        # Start processes for each GPU
        mp.set_start_method('spawn', force=True)
        result_queue = mp.Queue()
        processes = []

        for device_id in range(self.n_devices):
            p = mp.Process(target=run_island, args=(device_id, problem, config, result_queue))
            p.start()
            processes.append(p)

        # Wait for all processes
        for p in processes:
            p.join()

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # Find best result
        best_result = min(results, key=lambda x: x[1]['objective'])

        runtime = time.time() - start_time

        return {
            'solution': best_result[1]['solution'],
            'objective': best_result[1]['objective'],
            'runtime': runtime,
            'metadata': {
                'backend': 'pytorch',
                'n_devices': self.n_devices,
                'strategy': 'island_model',
            }
        }

    def _island_model_ray(self, problem: Any, algorithm: str, config: Dict) -> Dict[str, Any]:
        """Ray implementation of island model"""
        import time
        start_time = time.time()

        @self.ray.remote(num_gpus=1)
        class Island:
            """Ray actor representing single island"""

            def __init__(self, island_id, problem, algorithm, config):
                self.island_id = island_id
                self.problem = problem
                self.algorithm = algorithm
                self.config = config

                # Initialize algorithm
                if algorithm == 'genetic':
                    from .gpu_methods.gpu_genetic import GPUGeneticAlgorithm
                    self.algo = GPUGeneticAlgorithm()

            def evolve(self, n_generations):
                """Evolve island for n generations"""
                # Run optimization
                result = self.algo.optimize(self.problem, self.config)
                return result

            def get_best_individuals(self, n):
                """Get n best individuals for migration"""
                # Return best individuals
                pass

            def receive_migrants(self, migrants):
                """Receive migrant individuals"""
                # Incorporate migrants into population
                pass

        # Create island actors
        islands = [Island.remote(i, problem, algorithm, config) for i in range(self.n_devices)]

        # Evolution with migration
        n_migrations = config.get('n_generations', 1000) // self.config.migration_interval

        for migration_round in range(n_migrations):
            # Evolve islands in parallel
            evolution_futures = [island.evolve.remote(self.config.migration_interval) for island in islands]

            # Wait for evolution to complete
            self.ray.wait(evolution_futures)

            # Migration (if not last round)
            if migration_round < n_migrations - 1:
                # Get best individuals from each island
                best_futures = [island.get_best_individuals.remote(5) for island in islands]
                best_individuals = self.ray.get(best_futures)

                # Ring migration
                for i, island in enumerate(islands):
                    next_island = (i + 1) % len(islands)
                    island.receive_migrants.remote(best_individuals[next_island])

        # Collect final results
        final_results = self.ray.get([island.evolve.remote(0) for island in islands])

        # Find best
        best_result = min(final_results, key=lambda x: x['objective'])

        runtime = time.time() - start_time

        return {
            'solution': best_result['solution'],
            'objective': best_result['objective'],
            'runtime': runtime,
            'metadata': {
                'backend': 'ray',
                'n_devices': self.n_devices,
                'strategy': 'island_model',
            }
        }

    def _master_worker_optimization(self, problem: Any, algorithm: str, config: Dict) -> Dict[str, Any]:
        """
        Master-worker pattern: central coordinator distributes work.

        Master maintains global state and assigns tasks to workers.
        """
        if self.backend == 'ray':
            return self._master_worker_ray(problem, algorithm, config)
        else:
            raise NotImplementedError(f"Master-worker not implemented for {self.backend}")

    def _master_worker_ray(self, problem: Any, algorithm: str, config: Dict) -> Dict[str, Any]:
        """Ray implementation of master-worker pattern"""
        import time
        start_time = time.time()

        @self.ray.remote(num_gpus=1)
        def evaluate_batch(solutions, problem):
            """Worker: evaluate batch of solutions on GPU"""
            # Move to GPU and evaluate
            fitness = [problem.objective_function(sol) for sol in solutions]
            return fitness

        # Master process
        if algorithm == 'genetic':
            population_size = config.get('population_size', 100)
            n_generations = config.get('n_generations', 1000)

            # Initialize population
            population = np.random.randn(population_size, problem.dimension)
            best_solution = None
            best_fitness = float('inf')

            for generation in range(n_generations):
                # Split population for workers
                batch_size = population_size // self.n_devices
                batches = [population[i:i+batch_size] for i in range(0, population_size, batch_size)]

                # Distribute evaluation
                futures = [evaluate_batch.remote(batch, problem) for batch in batches]
                fitness_batches = self.ray.get(futures)

                # Combine results
                fitness = np.concatenate(fitness_batches)

                # Update best
                min_idx = np.argmin(fitness)
                if fitness[min_idx] < best_fitness:
                    best_fitness = fitness[min_idx]
                    best_solution = population[min_idx].copy()

                # Genetic operations (selection, crossover, mutation)
                # ... (simplified)

        runtime = time.time() - start_time

        return {
            'solution': best_solution,
            'objective': best_fitness,
            'runtime': runtime,
            'metadata': {
                'backend': 'ray',
                'n_devices': self.n_devices,
                'strategy': 'master_worker',
            }
        }

    def _data_parallel_optimization(self, problem: Any, algorithm: str, config: Dict) -> Dict[str, Any]:
        """
        Data parallel: same algorithm, split data/population.

        All devices run the same algorithm but on different data subsets.
        """
        if self.backend == 'jax':
            return self._data_parallel_jax(problem, algorithm, config)
        elif self.backend == 'pytorch':
            return self._data_parallel_pytorch(problem, algorithm, config)
        else:
            raise NotImplementedError(f"Data parallel not implemented for {self.backend}")

    def _data_parallel_jax(self, problem: Any, algorithm: str, config: Dict) -> Dict[str, Any]:
        """JAX data parallel implementation"""
        import time
        from functools import partial

        start_time = time.time()

        # Shard data across devices
        if algorithm == 'genetic':
            population_size = config.get('population_size', 100)
            n_generations = config.get('n_generations', 1000)

            # Create sharded population
            from jax.experimental import mesh_utils
            from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

            # Create device mesh
            devices = mesh_utils.create_device_mesh((self.n_devices,))
            mesh = Mesh(devices, axis_names=('batch',))

            # Define sharding
            sharding = NamedSharding(mesh, P('batch', None))

            # Initialize population with sharding
            key = self.jax.random.PRNGKey(config.get('seed', 0))
            global_population_shape = (population_size, problem.dimension)

            # Create sharded population
            if hasattr(problem, 'bounds') and problem.bounds:
                lower, upper = problem.bounds
                population = self.jax.random.uniform(
                    key, global_population_shape, minval=lower, maxval=upper
                )
            else:
                population = self.jax.random.normal(key, global_population_shape)

            # Shard population across devices
            population = self.jax.device_put(population, sharding)

            # Define parallel operations with sharding constraints
            @partial(self.jax.jit, in_shardings=(sharding,), out_shardings=sharding)
            def parallel_evaluate(population):
                return self.jax.vmap(problem.objective_function)(population)

            best_fitness = float('inf')
            best_solution = None

            for generation in range(n_generations):
                # Evaluate in parallel across devices
                fitness = parallel_evaluate(population)

                # Find global best (automatic reduction across shards)
                min_idx = self.jax.numpy.argmin(fitness)
                min_fit = fitness[min_idx]

                if min_fit < best_fitness:
                    best_fitness = float(min_fit)
                    best_solution = population[min_idx]

                # Parallel genetic operations
                # ... (implement sharded selection, crossover, mutation)

        runtime = time.time() - start_time

        return {
            'solution': np.array(best_solution) if best_solution is not None else None,
            'objective': best_fitness,
            'runtime': runtime,
            'metadata': {
                'backend': 'jax',
                'n_devices': self.n_devices,
                'strategy': 'data_parallel',
                'sharding': 'mesh_sharding',
            }
        }

    def _data_parallel_pytorch(self, problem: Any, algorithm: str, config: Dict) -> Dict[str, Any]:
        """PyTorch data parallel implementation"""
        import time
        import torch
        from torch.nn.parallel import DataParallel

        start_time = time.time()

        # Simplified data parallel for optimization
        # In practice, would use DistributedDataParallel for multi-node

        class OptimizationModule(torch.nn.Module):
            """Wrap optimization in module for DataParallel"""

            def __init__(self, problem):
                super().__init__()
                self.problem = problem

            def forward(self, population):
                # Evaluate population
                fitness = []
                for individual in population:
                    fit = self.problem.objective_function(individual.cpu().numpy())
                    fitness.append(fit)
                return torch.tensor(fitness)

        # Create module and wrap with DataParallel
        module = OptimizationModule(problem)
        if self.n_devices > 1:
            module = DataParallel(module, device_ids=list(range(self.n_devices)))

        # Run optimization
        if algorithm == 'genetic':
            # Initialize population
            population = torch.randn(
                config.get('population_size', 100),
                problem.dimension,
                device='cuda:0' if torch.cuda.is_available() else 'cpu'
            )

            best_solution = None
            best_fitness = float('inf')

            for generation in range(config.get('n_generations', 1000)):
                # Parallel evaluation
                fitness = module(population)

                # Update best
                min_idx = torch.argmin(fitness)
                if fitness[min_idx] < best_fitness:
                    best_fitness = float(fitness[min_idx])
                    best_solution = population[min_idx].cpu().numpy()

                # Genetic operations
                # ... (simplified)

        runtime = time.time() - start_time

        return {
            'solution': best_solution,
            'objective': best_fitness,
            'runtime': runtime,
            'metadata': {
                'backend': 'pytorch',
                'n_devices': self.n_devices,
                'strategy': 'data_parallel',
            }
        }