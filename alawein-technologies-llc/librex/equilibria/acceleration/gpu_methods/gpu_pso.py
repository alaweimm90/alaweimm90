"""
GPU-Accelerated Particle Swarm Optimization

High-performance PSO implementation with GPU acceleration for massive swarms.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PSOConfig:
    """Configuration for GPU Particle Swarm Optimization"""
    n_particles: int = 100
    n_iterations: int = 1000
    w: float = 0.729  # Inertia weight
    c1: float = 1.49445  # Cognitive coefficient
    c2: float = 1.49445  # Social coefficient
    w_decay: float = 0.99  # Inertia decay rate
    velocity_limit: float = 1.0
    seed: int = 42
    topology: str = 'global'  # 'global', 'ring', 'random'
    async_update: bool = False


class GPUParticleSwarm:
    """
    Particle Swarm Optimization with GPU acceleration.

    Features:
    - Vectorized swarm updates on GPU
    - Multiple topology support (global, ring, random, dynamic)
    - Adaptive parameters
    - Constraint handling
    - Multi-swarm variants
    """

    def __init__(self, backend='jax', device_id: int = 0):
        """Initialize GPU PSO"""
        from ..gpu_backend import GPUBackend
        self.gpu = GPUBackend(backend, device_id)
        self.backend_name = self.gpu.backend_name

        # Setup backend-specific functions
        if self.backend_name == 'jax':
            self._setup_jax()
        elif self.backend_name == 'pytorch':
            self._setup_pytorch()
        else:
            self._setup_numpy()

    def _setup_jax(self):
        """Setup JAX-specific functions"""
        import jax
        import jax.numpy as jnp
        from jax import jit, vmap

        self.jax = jax
        self.jnp = jnp
        self.jit = jit
        self.vmap = vmap

        # JIT compile core PSO update
        self._update_swarm = jit(self._jax_update_swarm)

    def _setup_pytorch(self):
        """Setup PyTorch-specific functions"""
        import torch
        self.torch = torch

    def _setup_numpy(self):
        """Setup NumPy fallback"""
        self.np = np

    def optimize(self, problem: Any, config: Optional[PSOConfig] = None) -> Dict[str, Any]:
        """
        Run GPU-accelerated PSO.

        Args:
            problem: Optimization problem
            config: PSO configuration

        Returns:
            Optimization results
        """
        if config is None:
            config = PSOConfig()

        if self.backend_name == 'jax':
            return self._optimize_jax(problem, config)
        elif self.backend_name == 'pytorch':
            return self._optimize_pytorch(problem, config)
        else:
            return self._optimize_numpy(problem, config)

    def _optimize_jax(self, problem: Any, config: PSOConfig) -> Dict[str, Any]:
        """JAX implementation of PSO"""
        import time
        start_time = time.time()

        key = self.jax.random.PRNGKey(config.seed)

        # Initialize swarm
        key, subkey = self.jax.random.split(key)
        positions, velocities = self._initialize_swarm_jax(
            subkey, config.n_particles, problem.dimension, problem.bounds
        )

        # Initialize personal and global bests
        personal_best_positions = positions.copy()
        personal_best_scores = self.jnp.full(config.n_particles, float('inf'))
        global_best_position = positions[0].copy()
        global_best_score = float('inf')

        # Vectorized objective function
        @self.jit
        def evaluate_swarm(positions):
            return self.vmap(problem.objective_function)(positions)

        # Adaptive parameters
        w = config.w
        iteration_stats = []

        for iteration in range(config.n_iterations):
            # Evaluate entire swarm in parallel
            scores = evaluate_swarm(positions)

            # Update personal bests
            improved_mask = scores < personal_best_scores
            personal_best_positions = self.jnp.where(
                improved_mask[:, None], positions, personal_best_positions
            )
            personal_best_scores = self.jnp.where(improved_mask, scores, personal_best_scores)

            # Update global best
            min_idx = self.jnp.argmin(scores)
            if scores[min_idx] < global_best_score:
                global_best_score = float(scores[min_idx])
                global_best_position = positions[min_idx].copy()

            # Get topology-specific best
            if config.topology == 'global':
                social_best = global_best_position
            elif config.topology == 'ring':
                social_best = self._get_ring_topology_best_jax(
                    positions, scores, config.n_particles
                )
            else:  # random
                key, subkey = self.jax.random.split(key)
                social_best = self._get_random_topology_best_jax(
                    positions, scores, subkey
                )

            # Update swarm (vectorized)
            key, subkey = self.jax.random.split(key)
            positions, velocities = self._update_swarm(
                positions, velocities, personal_best_positions,
                social_best, w, config.c1, config.c2, config.velocity_limit,
                problem.bounds, subkey
            )

            # Decay inertia
            w *= config.w_decay

            # Statistics
            iteration_stats.append({
                'iteration': iteration,
                'global_best': global_best_score,
                'avg_score': float(self.jnp.mean(scores)),
                'std_score': float(self.jnp.std(scores)),
                'inertia': w,
            })

        runtime = time.time() - start_time

        return {
            'solution': self.gpu.from_device(global_best_position),
            'objective': global_best_score,
            'n_evaluations': config.n_particles * config.n_iterations,
            'runtime': runtime,
            'iteration_stats': iteration_stats,
            'final_swarm': {
                'positions': self.gpu.from_device(positions),
                'velocities': self.gpu.from_device(velocities),
                'personal_bests': self.gpu.from_device(personal_best_positions),
            },
            'metadata': {
                'backend': self.backend_name,
                'device': str(self.gpu._device),
                'config': config.__dict__,
            }
        }

    def _initialize_swarm_jax(self, key, n_particles: int, dimension: int,
                              bounds: Optional[Tuple[float, float]] = None):
        """Initialize swarm positions and velocities on GPU"""
        key1, key2 = self.jax.random.split(key)

        if bounds:
            lower, upper = bounds
            positions = self.jax.random.uniform(
                key1, (n_particles, dimension), minval=lower, maxval=upper
            )
            # Initialize velocities as small random values
            velocity_range = (upper - lower) * 0.1
            velocities = self.jax.random.uniform(
                key2, (n_particles, dimension), minval=-velocity_range, maxval=velocity_range
            )
        else:
            positions = self.jax.random.normal(key1, (n_particles, dimension))
            velocities = self.jax.random.normal(key2, (n_particles, dimension)) * 0.1

        return positions, velocities

    def _jax_update_swarm(self, positions, velocities, personal_best, social_best,
                          w, c1, c2, velocity_limit, bounds, key):
        """
        Vectorized PSO update for entire swarm on GPU.

        Standard PSO equations:
        v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        x = x + v
        """
        n_particles, dimension = positions.shape

        # Generate random coefficients
        key1, key2 = self.jax.random.split(key)
        r1 = self.jax.random.uniform(key1, (n_particles, dimension))
        r2 = self.jax.random.uniform(key2, (n_particles, dimension))

        # Velocity update
        cognitive = c1 * r1 * (personal_best - positions)

        # Handle different topology types for social component
        if social_best.ndim == 1:  # Global best (single vector)
            social = c2 * r2 * (social_best[None, :] - positions)
        else:  # Per-particle social best
            social = c2 * r2 * (social_best - positions)

        velocities = w * velocities + cognitive + social

        # Velocity clamping
        velocities = self.jnp.clip(velocities, -velocity_limit, velocity_limit)

        # Position update
        positions = positions + velocities

        # Boundary handling
        if bounds is not None:
            lower, upper = bounds
            # Reflect at boundaries
            mask_lower = positions < lower
            mask_upper = positions > upper

            positions = self.jnp.where(mask_lower, lower + (lower - positions), positions)
            positions = self.jnp.where(mask_upper, upper - (positions - upper), positions)

            # Ensure within bounds
            positions = self.jnp.clip(positions, lower, upper)

            # Reverse velocity at boundaries
            velocities = self.jnp.where(mask_lower | mask_upper, -velocities, velocities)

        return positions, velocities

    def _get_ring_topology_best_jax(self, positions, scores, n_particles):
        """Get best neighbor in ring topology"""
        # In ring topology, each particle is connected to its adjacent neighbors
        best_positions = []

        for i in range(n_particles):
            # Get neighbors (previous, current, next)
            prev_idx = (i - 1) % n_particles
            next_idx = (i + 1) % n_particles

            neighbor_indices = [prev_idx, i, next_idx]
            neighbor_scores = scores[neighbor_indices]

            # Find best neighbor
            best_neighbor_idx = neighbor_indices[self.jnp.argmin(neighbor_scores)]
            best_positions.append(positions[best_neighbor_idx])

        return self.jnp.stack(best_positions)

    def _get_random_topology_best_jax(self, positions, scores, key):
        """Get best in random topology (random neighbors)"""
        n_particles = len(positions)
        n_neighbors = min(5, n_particles // 2)  # Each particle has 5 random neighbors

        best_positions = []

        for i in range(n_particles):
            # Select random neighbors
            neighbor_indices = self.jax.random.choice(
                key, n_particles, (n_neighbors,), replace=False
            )

            # Always include self
            neighbor_indices = neighbor_indices.at[0].set(i)

            neighbor_scores = scores[neighbor_indices]
            best_neighbor_idx = neighbor_indices[self.jnp.argmin(neighbor_scores)]
            best_positions.append(positions[best_neighbor_idx])

            key = self.jax.random.split(key)[0]

        return self.jnp.stack(best_positions)

    def _optimize_pytorch(self, problem: Any, config: PSOConfig) -> Dict[str, Any]:
        """PyTorch implementation of PSO"""
        import time
        start_time = time.time()

        self.torch.manual_seed(config.seed)

        # Initialize swarm
        positions, velocities = self._initialize_swarm_torch(
            config.n_particles, problem.dimension, problem.bounds
        )

        personal_best_positions = positions.clone()
        personal_best_scores = self.torch.full((config.n_particles,), float('inf'),
                                               device=self.gpu._device)
        global_best_position = positions[0].clone()
        global_best_score = float('inf')

        w = config.w

        for iteration in range(config.n_iterations):
            # Evaluate swarm
            scores = self._evaluate_swarm_torch(positions, problem.objective_function)

            # Update personal bests
            improved_mask = scores < personal_best_scores
            personal_best_positions[improved_mask] = positions[improved_mask].clone()
            personal_best_scores[improved_mask] = scores[improved_mask]

            # Update global best
            min_idx = self.torch.argmin(scores)
            if scores[min_idx] < global_best_score:
                global_best_score = float(scores[min_idx])
                global_best_position = positions[min_idx].clone()

            # Update velocities and positions
            r1 = self.torch.rand_like(positions)
            r2 = self.torch.rand_like(positions)

            velocities = (w * velocities +
                         config.c1 * r1 * (personal_best_positions - positions) +
                         config.c2 * r2 * (global_best_position - positions))

            # Velocity clamping
            velocities = self.torch.clamp(velocities, -config.velocity_limit, config.velocity_limit)

            # Update positions
            positions = positions + velocities

            # Boundary handling
            if problem.bounds:
                lower, upper = problem.bounds
                positions = self.torch.clamp(positions, lower, upper)

            # Decay inertia
            w *= config.w_decay

        runtime = time.time() - start_time

        return {
            'solution': global_best_position.cpu().numpy(),
            'objective': global_best_score,
            'runtime': runtime,
            'metadata': {'backend': 'pytorch', 'device': str(self.gpu._device)}
        }

    def _initialize_swarm_torch(self, n_particles: int, dimension: int,
                                bounds: Optional[Tuple[float, float]] = None):
        """Initialize swarm using PyTorch"""
        if bounds:
            lower, upper = bounds
            positions = self.torch.rand(n_particles, dimension, device=self.gpu._device) * (upper - lower) + lower
            velocities = self.torch.randn(n_particles, dimension, device=self.gpu._device) * 0.1 * (upper - lower)
        else:
            positions = self.torch.randn(n_particles, dimension, device=self.gpu._device)
            velocities = self.torch.randn(n_particles, dimension, device=self.gpu._device) * 0.1

        return positions, velocities

    def _evaluate_swarm_torch(self, positions, objective_fn):
        """Evaluate swarm using PyTorch"""
        scores = []
        for pos in positions:
            # Convert to numpy for objective function
            pos_numpy = pos.cpu().numpy()
            score = objective_fn(pos_numpy)
            scores.append(score)

        return self.torch.tensor(scores, device=self.gpu._device)

    def _optimize_numpy(self, problem: Any, config: PSOConfig) -> Dict[str, Any]:
        """NumPy CPU fallback"""
        import time
        start_time = time.time()

        np.random.seed(config.seed)

        # Initialize swarm
        if hasattr(problem, 'bounds') and problem.bounds:
            lower, upper = problem.bounds
            positions = np.random.uniform(lower, upper, (config.n_particles, problem.dimension))
            velocities = np.random.randn(config.n_particles, problem.dimension) * .0 * (upper - lower)
        else:
            positions = np.random.randn(config.n_particles, problem.dimension)
            velocities = np.random.randn(config.n_particles, problem.dimension) * 0.1

        personal_best_positions = positions.copy()
        personal_best_scores = np.full(config.n_particles, float('inf'))
        global_best_position = positions[0].copy()
        global_best_score = float('inf')

        w = config.w

        for iteration in range(config.n_iterations):
            # Evaluate
            scores = np.array([problem.objective_function(pos) for pos in positions])

            # Update bests
            improved = scores < personal_best_scores
            personal_best_positions[improved] = positions[improved]
            personal_best_scores[improved] = scores[improved]

            min_idx = np.argmin(scores)
            if scores[min_idx] < global_best_score:
                global_best_score = scores[min_idx]
                global_best_position = positions[min_idx].copy()

            # PSO update
            r1 = np.random.random(positions.shape)
            r2 = np.random.random(positions.shape)

            velocities = (w * velocities +
                         config.c1 * r1 * (personal_best_positions - positions) +
                         config.c2 * r2 * (global_best_position - positions))

            velocities = np.clip(velocities, -config.velocity_limit, config.velocity_limit)
            positions = positions + velocities

            if hasattr(problem, 'bounds') and problem.bounds:
                lower, upper = problem.bounds
                positions = np.clip(positions, lower, upper)

            w *= config.w_decay

        runtime = time.time() - start_time

        return {
            'solution': global_best_position,
            'objective': global_best_score,
            'runtime': runtime,
            'metadata': {'backend': 'numpy (CPU)'}
        }