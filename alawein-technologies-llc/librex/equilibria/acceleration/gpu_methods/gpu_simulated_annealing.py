"""
GPU-Accelerated Simulated Annealing

Parallel simulated annealing with multiple chains on GPU.
"""

import logging
from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SimulatedAnnealingConfig:
    """Configuration for GPU Simulated Annealing"""
    n_chains: int = 32  # Number of parallel SA chains
    n_iterations: int = 10000
    initial_temp: float = 100.0
    final_temp: float = 0.001
    cooling_schedule: str = 'exponential'  # 'exponential', 'linear', 'adaptive'
    neighbor_scale: float = 0.1
    seed: int = 42
    exchange_interval: int = 100  # Exchange between chains
    restart_threshold: float = 0.1  # Restart if no improvement


class GPUSimulatedAnnealing:
    """
    Simulated Annealing with GPU acceleration.

    Features:
    - Multiple parallel SA chains
    - Vectorized state transitions
    - Adaptive cooling schedules
    - Parallel tempering (replica exchange)
    - Adaptive neighborhood sizes
    """

    def __init__(self, backend='jax', device_id: int = 0):
        """Initialize GPU SA"""
        from ..gpu_backend import GPUBackend
        self.gpu = GPUBackend(backend, device_id)
        self.backend_name = self.gpu.backend_name

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
        from jax import jit, vmap, pmap

        self.jax = jax
        self.jnp = jnp
        self.jit = jit
        self.vmap = vmap
        self.pmap = pmap

        # JIT compile core SA operations
        self._sa_step = jit(self._jax_sa_step)
        self._parallel_sa_step = vmap(self._sa_step, in_axes=(0, None, 0, 0))

    def _setup_pytorch(self):
        """Setup PyTorch-specific functions"""
        import torch
        self.torch = torch

    def _setup_numpy(self):
        """Setup NumPy fallback"""
        self.np = np

    def optimize(self, problem: Any, config: Optional[SimulatedAnnealingConfig] = None) -> Dict[str, Any]:
        """
        Run GPU-accelerated simulated annealing.

        Args:
            problem: Optimization problem
            config: SA configuration

        Returns:
            Optimization results
        """
        if config is None:
            config = SimulatedAnnealingConfig()

        if self.backend_name == 'jax':
            return self._optimize_jax(problem, config)
        elif self.backend_name == 'pytorch':
            return self._optimize_pytorch(problem, config)
        else:
            return self._optimize_numpy(problem, config)

    def _optimize_jax(self, problem: Any, config: SimulatedAnnealingConfig) -> Dict[str, Any]:
        """JAX implementation with parallel SA chains"""
        import time
        start_time = time.time()

        key = self.jax.random.PRNGKey(config.seed)

        # Initialize multiple SA chains
        key, subkey = self.jax.random.split(key)
        states = self._initialize_chains_jax(
            subkey, config.n_chains, problem.dimension, problem.bounds
        )

        # Initialize states dictionary for each chain
        current_states = states
        current_objectives = self.vmap(problem.objective_function)(current_states)
        best_states = current_states.copy()
        best_objectives = current_objectives.copy()

        # Global best tracking
        global_best_idx = self.jnp.argmin(best_objectives)
        global_best_state = best_states[global_best_idx]
        global_best_objective = float(best_objectives[global_best_idx])

        # Temperature schedule
        temperatures = self._get_temperature_schedule_jax(
            config.n_chains, config.n_iterations,
            config.initial_temp, config.final_temp,
            config.cooling_schedule
        )

        iteration_stats = []
        n_accepted = self.jnp.zeros(config.n_chains)

        for iteration in range(config.n_iterations):
            temperature = temperatures[iteration]

            # Different temperatures for each chain (parallel tempering)
            chain_temps = temperature * self.jnp.logspace(0, 1, config.n_chains)

            # Parallel SA step for all chains
            key, subkey = self.jax.random.split(key)
            keys = self.jax.random.split(subkey, config.n_chains)

            new_states, new_objectives, accepted = self._parallel_sa_step_jax(
                current_states, current_objectives, chain_temps,
                config.neighbor_scale, problem, keys
            )

            # Update states
            current_states = new_states
            current_objectives = new_objectives
            n_accepted = n_accepted + accepted

            # Update best states for each chain
            improved = new_objectives < best_objectives
            best_states = self.jnp.where(improved[:, None], new_states, best_states)
            best_objectives = self.jnp.where(improved, new_objectives, best_objectives)

            # Update global best
            chain_best_idx = self.jnp.argmin(best_objectives)
            if best_objectives[chain_best_idx] < global_best_objective:
                global_best_objective = float(best_objectives[chain_best_idx])
                global_best_state = best_states[chain_best_idx]

            # Replica exchange between chains
            if iteration % config.exchange_interval == 0 and iteration > 0:
                key, subkey = self.jax.random.split(key)
                current_states, current_objectives = self._replica_exchange_jax(
                    current_states, current_objectives, chain_temps, subkey
                )

            # Adaptive restart for stuck chains
            if iteration % 1000 == 0 and iteration > 0:
                acceptance_rate = n_accepted / 1000
                stuck_chains = acceptance_rate < config.restart_threshold

                if self.jnp.any(stuck_chains):
                    key, subkey = self.jax.random.split(key)
                    current_states = self._restart_chains_jax(
                        current_states, stuck_chains, best_states[0],
                        config.neighbor_scale * 10, subkey
                    )
                    current_objectives = self.vmap(problem.objective_function)(current_states)

                n_accepted = self.jnp.zeros(config.n_chains)

            # Statistics
            if iteration % 100 == 0:
                iteration_stats.append({
                    'iteration': iteration,
                    'temperature': float(temperature),
                    'global_best': global_best_objective,
                    'avg_objective': float(self.jnp.mean(current_objectives)),
                    'std_objective': float(self.jnp.std(current_objectives)),
                    'acceptance_rate': float(self.jnp.mean(n_accepted / max(1, iteration % 1000))),
                })

        runtime = time.time() - start_time

        return {
            'solution': self.gpu.from_device(global_best_state),
            'objective': global_best_objective,
            'n_evaluations': config.n_chains * config.n_iterations,
            'runtime': runtime,
            'iteration_stats': iteration_stats,
            'chain_results': {
                'final_states': self.gpu.from_device(current_states),
                'final_objectives': self.gpu.from_device(current_objectives),
                'best_states': self.gpu.from_device(best_states),
                'best_objectives': self.gpu.from_device(best_objectives),
            },
            'metadata': {
                'backend': self.backend_name,
                'device': str(self.gpu._device),
                'config': config.__dict__,
            }
        }

    def _initialize_chains_jax(self, key, n_chains: int, dimension: int,
                               bounds: Optional[Tuple[float, float]] = None):
        """Initialize multiple SA chains on GPU"""
        if bounds:
            lower, upper = bounds
            states = self.jax.random.uniform(
                key, (n_chains, dimension), minval=lower, maxval=upper
            )
        else:
            states = self.jax.random.normal(key, (n_chains, dimension))

        return states

    def _jax_sa_step(self, state, objective, temperature, neighbor_scale, problem, key):
        """Single SA step on GPU"""
        # Generate neighbor
        key, subkey = self.jax.random.split(key)
        perturbation = self.jax.random.normal(subkey, state.shape) * neighbor_scale
        neighbor = state + perturbation

        # Clip to bounds if needed
        if hasattr(problem, 'bounds') and problem.bounds:
            lower, upper = problem.bounds
            neighbor = self.jnp.clip(neighbor, lower, upper)

        # Evaluate neighbor
        neighbor_obj = problem.objective_function(neighbor)

        # Metropolis criterion
        delta = neighbor_obj - objective
        key, subkey = self.jax.random.split(key)

        # Accept if better or with probability
        if delta < 0:
            accept_prob = 1.0
        else:
            accept_prob = self.jnp.exp(-delta / temperature)

        accept = self.jax.random.uniform(subkey) < accept_prob

        # Update state
        new_state = self.jnp.where(accept, neighbor, state)
        new_objective = self.jnp.where(accept, neighbor_obj, objective)

        return new_state, new_objective, accept

    def _parallel_sa_step_jax(self, states, objectives, temperatures,
                              neighbor_scale, problem, keys):
        """Parallel SA step for all chains"""
        # Vectorized neighbor generation
        perturbations = self.vmap(
            lambda k: self.jax.random.normal(k, states[0].shape) * neighbor_scale
        )(keys)
        neighbors = states + perturbations

        # Clip to bounds
        if hasattr(problem, 'bounds') and problem.bounds:
            lower, upper = problem.bounds
            neighbors = self.jnp.clip(neighbors, lower, upper)

        # Evaluate all neighbors
        neighbor_objs = self.vmap(problem.objective_function)(neighbors)

        # Metropolis criterion for all chains
        deltas = neighbor_objs - objectives
        accept_probs = self.jnp.where(
            deltas < 0,
            self.jnp.ones_like(deltas),
            self.jnp.exp(-deltas / temperatures)
        )

        # Random acceptance
        uniform_samples = self.vmap(lambda k: self.jax.random.uniform(k))(keys)
        accept = uniform_samples < accept_probs

        # Update states
        new_states = self.jnp.where(accept[:, None], neighbors, states)
        new_objectives = self.jnp.where(accept, neighbor_objs, objectives)

        return new_states, new_objectives, accept

    def _replica_exchange_jax(self, states, objectives, temperatures, key):
        """Replica exchange between adjacent chains"""
        n_chains = len(states)

        for i in range(0, n_chains - 1, 2):
            # Try to exchange chains i and i+1
            delta_beta = 1.0 / temperatures[i] - 1.0 / temperatures[i + 1]
            delta_energy = objectives[i + 1] - objectives[i]
            exchange_prob = self.jnp.minimum(1.0, self.jnp.exp(delta_beta * delta_energy))

            key, subkey = self.jax.random.split(key)
            if self.jax.random.uniform(subkey) < exchange_prob:
                # Swap states
                states = states.at[i].set(states[i + 1])
                states = states.at[i + 1].set(states[i])
                objectives = objectives.at[i].set(objectives[i + 1])
                objectives = objectives.at[i + 1].set(objectives[i])

        return states, objectives

    def _restart_chains_jax(self, states, restart_mask, best_state, noise_scale, key):
        """Restart stuck chains near best solution"""
        keys = self.jax.random.split(key, len(states))

        new_states = []
        for i, (state, should_restart, k) in enumerate(zip(states, restart_mask, keys)):
            if should_restart:
                # Restart near best with noise
                noise = self.jax.random.normal(k, best_state.shape) * noise_scale
                new_state = best_state + noise
            else:
                new_state = state
            new_states.append(new_state)

        return self.jnp.stack(new_states)

    def _get_temperature_schedule_jax(self, n_chains, n_iterations,
                                      initial_temp, final_temp, schedule_type):
        """Generate temperature schedule"""
        if schedule_type == 'exponential':
            alpha = (final_temp / initial_temp) ** (1.0 / n_iterations)
            temps = initial_temp * (alpha ** self.jnp.arange(n_iterations))
        elif schedule_type == 'linear':
            temps = self.jnp.linspace(initial_temp, final_temp, n_iterations)
        elif schedule_type == 'adaptive':
            # Slower cooling at the beginning, faster at the end
            t = self.jnp.arange(n_iterations) / n_iterations
            temps = initial_temp * ((final_temp / initial_temp) ** (t ** 2))
        else:
            # Logarithmic
            temps = initial_temp / (1 + self.jnp.log(1 + self.jnp.arange(n_iterations)))

        return temps

    def _optimize_pytorch(self, problem: Any, config: SimulatedAnnealingConfig) -> Dict[str, Any]:
        """PyTorch implementation"""
        import time
        start_time = time.time()

        self.torch.manual_seed(config.seed)

        # Initialize chains
        states = self._initialize_chains_torch(
            config.n_chains, problem.dimension, problem.bounds
        )

        current_states = states
        current_objectives = self._evaluate_batch_torch(states, problem.objective_function)
        best_states = states.clone()
        best_objectives = current_objectives.clone()

        global_best_idx = self.torch.argmin(best_objectives)
        global_best_state = best_states[global_best_idx].clone()
        global_best_objective = float(best_objectives[global_best_idx])

        # Temperature schedule
        temperatures = self._get_temperature_schedule_torch(
            config.n_iterations, config.initial_temp,
            config.final_temp, config.cooling_schedule
        )

        for iteration in range(config.n_iterations):
            temperature = temperatures[iteration]

            # Generate neighbors
            perturbations = self.torch.randn_like(states) * config.neighbor_scale
            neighbors = states + perturbations

            if problem.bounds:
                lower, upper = problem.bounds
                neighbors = self.torch.clamp(neighbors, lower, upper)

            # Evaluate neighbors
            neighbor_objs = self._evaluate_batch_torch(neighbors, problem.objective_function)

            # Metropolis criterion
            deltas = neighbor_objs - current_objectives
            accept_probs = self.torch.where(
                deltas < 0,
                self.torch.ones_like(deltas),
                self.torch.exp(-deltas / temperature)
            )

            accept = self.torch.rand_like(accept_probs) < accept_probs

            # Update states
            current_states = self.torch.where(
                accept.unsqueeze(1), neighbors, current_states
            )
            current_objectives = self.torch.where(accept, neighbor_objs, current_objectives)

            # Update bests
            improved = current_objectives < best_objectives
            best_states[improved] = current_states[improved].clone()
            best_objectives[improved] = current_objectives[improved]

            # Global best
            min_idx = self.torch.argmin(best_objectives)
            if best_objectives[min_idx] < global_best_objective:
                global_best_objective = float(best_objectives[min_idx])
                global_best_state = best_states[min_idx].clone()

        runtime = time.time() - start_time

        return {
            'solution': global_best_state.cpu().numpy(),
            'objective': global_best_objective,
            'runtime': runtime,
            'metadata': {'backend': 'pytorch', 'device': str(self.gpu._device)}
        }

    def _initialize_chains_torch(self, n_chains: int, dimension: int,
                                 bounds: Optional[Tuple[float, float]] = None):
        """Initialize chains using PyTorch"""
        if bounds:
            lower, upper = bounds
            states = self.torch.rand(n_chains, dimension, device=self.gpu._device) * (upper - lower) + lower
        else:
            states = self.torch.randn(n_chains, dimension, device=self.gpu._device)

        return states

    def _evaluate_batch_torch(self, states, objective_fn):
        """Evaluate batch of states"""
        objectives = []
        for state in states:
            obj = objective_fn(state.cpu().numpy())
            objectives.append(obj)

        return self.torch.tensor(objectives, device=self.gpu._device)

    def _get_temperature_schedule_torch(self, n_iterations, initial_temp,
                                        final_temp, schedule_type):
        """Temperature schedule using PyTorch"""
        if schedule_type == 'exponential':
            alpha = (final_temp / initial_temp) ** (1.0 / n_iterations)
            temps = initial_temp * (alpha ** self.torch.arange(n_iterations))
        elif schedule_type == 'linear':
            temps = self.torch.linspace(initial_temp, final_temp, n_iterations)
        else:
            t = self.torch.arange(n_iterations) / n_iterations
            temps = initial_temp * ((final_temp / initial_temp) ** (t ** 2))

        return temps

    def _optimize_numpy(self, problem: Any, config: SimulatedAnnealingConfig) -> Dict[str, Any]:
        """NumPy CPU fallback"""
        import time
        start_time = time.time()

        np.random.seed(config.seed)

        # Single chain SA (simplified)
        if hasattr(problem, 'bounds') and problem.bounds:
            lower, upper = problem.bounds
            state = np.random.uniform(lower, upper, problem.dimension)
        else:
            state = np.random.randn(problem.dimension)

        current_obj = problem.objective_function(state)
        best_state = state.copy()
        best_obj = current_obj

        for iteration in range(config.n_iterations):
            # Temperature
            progress = iteration / config.n_iterations
            if config.cooling_schedule == 'exponential':
                temp = config.initial_temp * ((config.final_temp / config.initial_temp) ** progress)
            else:
                temp = config.initial_temp * (1 - progress) + config.final_temp * progress

            # Generate neighbor
            neighbor = state + np.random.randn(problem.dimension) * config.neighbor_scale

            if hasattr(problem, 'bounds') and problem.bounds:
                lower, upper = problem.bounds
                neighbor = np.clip(neighbor, lower, upper)

            # Evaluate
            neighbor_obj = problem.objective_function(neighbor)

            # Accept/reject
            delta = neighbor_obj - current_obj
            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                state = neighbor
                current_obj = neighbor_obj

                if current_obj < best_obj:
                    best_state = state.copy()
                    best_obj = current_obj

        runtime = time.time() - start_time

        return {
            'solution': best_state,
            'objective': best_obj,
            'runtime': runtime,
            'metadata': {'backend': 'numpy (CPU)'}
        }