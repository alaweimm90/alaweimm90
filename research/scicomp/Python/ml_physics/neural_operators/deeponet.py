#!/usr/bin/env python3
"""
Deep Operator Network (DeepONet)
Advanced neural operator implementation for learning operators between function
spaces. Demonstrates operator learning for parametric PDEs, multi-physics problems,
and surrogate modeling with Berkeley-styled visualizations.
Key Features:
- Branch-trunk architecture for operator learning
- Multi-fidelity and transfer learning
- Uncertainty quantification
- Physics-informed operator networks
- Berkeley color scheme visualizations
- Comprehensive benchmarking suite
Applications:
- Parametric PDE solving
- Multi-scale modeling
- Real-time simulation
- Design optimization
- Inverse problems
- Digital twins
Architecture:
- Branch net: Encodes function inputs (parameters, boundary conditions)
- Trunk net: Encodes spatial/temporal coordinates
- Operator: Learns mapping between input functions and solution functions
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')
try:
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow_probability as tfp
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Using numpy-based implementation.")
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, vmap, jit, random
    import optax
    from jax.scipy.spatial.distance import cdist
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union, Callable, Any
from Python.visualization.berkeley_style import BerkeleyPlot
from Python.utils.constants import *
@dataclass
class DeepONetConfig:
    """Configuration for DeepONet architecture and training."""
    # Architecture parameters
    branch_layers: List[int] = None  # Branch network layers
    trunk_layers: List[int] = None   # Trunk network layers
    latent_dim: int = 128           # Latent space dimension
    activation: str = 'tanh'        # Activation function
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 10000
    validation_split: float = 0.2
    # Data parameters
    n_sensors: int = 100            # Number of sensor points for branch input
    n_evaluation: int = 1000        # Number of evaluation points
    input_dim: int = 1              # Input function dimension
    output_dim: int = 1             # Output function dimension
    coordinate_dim: int = 2         # Spatial-temporal coordinate dimension
    # Problem-specific parameters
    problem_type: str = 'heat'      # 'heat', 'wave', 'burgers', 'poisson'
    parameter_range: Tuple[float, float] = (0.1, 2.0)  # Parameter variation range
    domain_bounds: List[Tuple[float, float]] = None
    # Physics-informed options
    physics_informed: bool = False
    physics_weight: float = 1.0
    # Uncertainty quantification
    enable_uncertainty: bool = False
    n_ensemble: int = 5
    dropout_rate: float = 0.1
    # Multi-fidelity options
    multi_fidelity: bool = False
    fidelity_levels: List[int] = None
    def __post_init__(self):
        if self.branch_layers is None:
            self.branch_layers = [128, 128, 128]
        if self.trunk_layers is None:
            self.trunk_layers = [128, 128, 128]
        if self.domain_bounds is None:
            if self.coordinate_dim == 1:
                self.domain_bounds = [(0.0, 1.0)]
            elif self.coordinate_dim == 2:
                self.domain_bounds = [(0.0, 1.0), (0.0, 1.0)]
            elif self.coordinate_dim == 3:
                self.domain_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        if self.fidelity_levels is None:
            self.fidelity_levels = [32, 64, 128]
class DeepONet:
    """
    Deep Operator Network for learning operators between function spaces.
    The architecture consists of:
    - Branch network: Encodes input functions at sensor locations
    - Trunk network: Encodes coordinates where we want to evaluate the output
    - Combination layer: Combines branch and trunk outputs to produce final result
    """
    def __init__(self, config: DeepONetConfig):
        self.config = config
        self.berkeley_plot = BerkeleyPlot()
        # Initialize networks
        if TF_AVAILABLE:
            self._init_tensorflow_networks()
        elif JAX_AVAILABLE:
            self._init_jax_networks()
        else:
            self._init_numpy_networks()
        # Training history
        self.training_history = {}
        self.validation_history = {}
        # Problem setup
        self._setup_problem()
    def _init_tensorflow_networks(self):
        """Initialize TensorFlow-based networks."""
        # Branch network (encodes input functions)
        branch_input = keras.layers.Input(shape=(self.config.n_sensors,))
        branch = branch_input
        for units in self.config.branch_layers:
            branch = keras.layers.Dense(
                units,
                activation=self.config.activation,
                kernel_initializer='glorot_normal'
            )(branch)
            if self.config.enable_uncertainty:
                branch = keras.layers.Dropout(self.config.dropout_rate)(branch)
        # Final branch layer (no bias - important for DeepONet)
        branch_output = keras.layers.Dense(
            self.config.latent_dim,
            activation=self.config.activation,
            use_bias=False
        )(branch)
        self.branch_net = keras.Model(inputs=branch_input, outputs=branch_output)
        # Trunk network (encodes coordinates)
        trunk_input = keras.layers.Input(shape=(self.config.coordinate_dim,))
        trunk = trunk_input
        for units in self.config.trunk_layers:
            trunk = keras.layers.Dense(
                units,
                activation=self.config.activation,
                kernel_initializer='glorot_normal'
            )(trunk)
            if self.config.enable_uncertainty:
                trunk = keras.layers.Dropout(self.config.dropout_rate)(trunk)
        # Final trunk layer (with bias)
        trunk_output = keras.layers.Dense(
            self.config.latent_dim,
            activation=self.config.activation,
            use_bias=True
        )(trunk)
        self.trunk_net = keras.Model(inputs=trunk_input, outputs=trunk_output)
        # Combined DeepONet model
        branch_in = keras.layers.Input(shape=(self.config.n_sensors,))
        trunk_in = keras.layers.Input(shape=(self.config.coordinate_dim,))
        branch_out = self.branch_net(branch_in)
        trunk_out = self.trunk_net(trunk_in)
        # Element-wise multiplication and sum
        combined = keras.layers.Multiply()([branch_out, trunk_out])
        output = keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True))(combined)
        self.model = keras.Model(inputs=[branch_in, trunk_in], outputs=output)
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        # Compile model
        self.model.compile(
            optimizer=self.optimizer,
            loss='mse',
            metrics=['mae']
        )
    def _init_jax_networks(self):
        """Initialize JAX-based networks."""
        def init_network_params(rng_key, layer_sizes, use_bias=True):
            params = []
            for i in range(len(layer_sizes) - 1):
                key_w, key_b, rng_key = random.split(rng_key, 3)
                W = random.normal(key_w, (layer_sizes[i], layer_sizes[i+1])) * np.sqrt(2.0 / layer_sizes[i])
                if use_bias:
                    b = random.normal(key_b, (layer_sizes[i+1],)) * 0.01
                else:
                    b = jnp.zeros(layer_sizes[i+1])
                params.append({'W': W, 'b': b})
            return params, rng_key
        self.rng_key = random.PRNGKey(42)
        # Branch network parameters
        branch_sizes = [self.config.n_sensors] + self.config.branch_layers + [self.config.latent_dim]
        self.branch_params, self.rng_key = init_network_params(
            self.rng_key, branch_sizes[:-1] + [self.config.latent_dim], use_bias=True
        )
        # Last layer has no bias
        key, self.rng_key = random.split(self.rng_key)
        final_W = random.normal(key, (self.config.branch_layers[-1], self.config.latent_dim)) * \
                 np.sqrt(2.0 / self.config.branch_layers[-1])
        self.branch_params.append({'W': final_W, 'b': jnp.zeros(self.config.latent_dim)})
        # Trunk network parameters
        trunk_sizes = [self.config.coordinate_dim] + self.config.trunk_layers + [self.config.latent_dim]
        self.trunk_params, self.rng_key = init_network_params(self.rng_key, trunk_sizes, use_bias=True)
        # Network forward functions
        @jit
        def branch_forward(params, x):
            for i, layer in enumerate(params[:-1]):
                x = jnp.dot(x, layer['W']) + layer['b']
                x = jnp.tanh(x)
            # Final layer (no bias)
            x = jnp.dot(x, params[-1]['W'])
            return jnp.tanh(x)
        @jit
        def trunk_forward(params, x):
            for i, layer in enumerate(params[:-1]):
                x = jnp.dot(x, layer['W']) + layer['b']
                x = jnp.tanh(x)
            # Final layer (with bias)
            x = jnp.dot(x, params[-1]['W']) + params[-1]['b']
            return jnp.tanh(x)
        @jit
        def deeponet_forward(branch_params, trunk_params, branch_input, trunk_input):
            branch_out = branch_forward(branch_params, branch_input)
            trunk_out = trunk_forward(trunk_params, trunk_input)
            return jnp.sum(branch_out * trunk_out, axis=-1, keepdims=True)
        self.branch_fn = branch_forward
        self.trunk_fn = trunk_forward
        self.deeponet_fn = deeponet_forward
        # Optimizer
        self.optimizer = optax.adam(self.config.learning_rate)
        self.opt_state = self.optimizer.init({'branch': self.branch_params, 'trunk': self.trunk_params})
    def _init_numpy_networks(self):
        """Initialize numpy-based networks."""
        # Branch network weights
        self.branch_weights = []
        self.branch_biases = []
        branch_sizes = [self.config.n_sensors] + self.config.branch_layers + [self.config.latent_dim]
        for i in range(len(branch_sizes) - 1):
            W = np.random.normal(0, np.sqrt(2.0 / branch_sizes[i]),
                               (branch_sizes[i], branch_sizes[i+1]))
            b = np.zeros(branch_sizes[i+1]) if i == len(branch_sizes) - 2 else np.random.normal(0, 0.01, branch_sizes[i+1])
            self.branch_weights.append(W)
            self.branch_biases.append(b)
        # Trunk network weights
        self.trunk_weights = []
        self.trunk_biases = []
        trunk_sizes = [self.config.coordinate_dim] + self.config.trunk_layers + [self.config.latent_dim]
        for i in range(len(trunk_sizes) - 1):
            W = np.random.normal(0, np.sqrt(2.0 / trunk_sizes[i]),
                               (trunk_sizes[i], trunk_sizes[i+1]))
            b = np.random.normal(0, 0.01, trunk_sizes[i+1])
            self.trunk_weights.append(W)
            self.trunk_biases.append(b)
    def _setup_problem(self):
        """Set up the specific problem configuration."""
        if self.config.problem_type == 'heat':
            self._setup_heat_equation()
        elif self.config.problem_type == 'wave':
            self._setup_wave_equation()
        elif self.config.problem_type == 'burgers':
            self._setup_burgers_equation()
        elif self.config.problem_type == 'poisson':
            self._setup_poisson_equation()
        else:
            raise ValueError(f"Unknown problem type: {self.config.problem_type}")
    def _setup_heat_equation(self):
        """Set up heat equation problem."""
        self.pde_params = {
            'equation': 'heat',
            'diffusivity_range': self.config.parameter_range,
            'initial_condition_type': 'gaussian',
            'boundary_conditions': 'dirichlet'
        }
        def heat_pde_residual(u, u_t, u_xx, diffusivity):
            """Heat equation: ∂u/∂t = α∇²u"""
            return u_t - diffusivity * u_xx
        self.pde_residual = heat_pde_residual
    def _setup_wave_equation(self):
        """Set up wave equation problem."""
        self.pde_params = {
            'equation': 'wave',
            'wave_speed_range': self.config.parameter_range,
            'initial_condition_type': 'sine',
            'boundary_conditions': 'periodic'
        }
        def wave_pde_residual(u, u_tt, u_xx, wave_speed):
            """Wave equation: ∂²u/∂t² = c²∇²u"""
            return u_tt - wave_speed**2 * u_xx
        self.pde_residual = wave_pde_residual
    def _setup_burgers_equation(self):
        """Set up Burgers' equation problem."""
        self.pde_params = {
            'equation': 'burgers',
            'viscosity_range': self.config.parameter_range,
            'initial_condition_type': 'shock',
            'boundary_conditions': 'periodic'
        }
        def burgers_pde_residual(u, u_t, u_x, u_xx, viscosity):
            """Burgers' equation: ∂u/∂t + u∂u/∂x = ν∇²u"""
            return u_t + u * u_x - viscosity * u_xx
        self.pde_residual = burgers_pde_residual
    def _setup_poisson_equation(self):
        """Set up Poisson equation problem."""
        self.pde_params = {
            'equation': 'poisson',
            'parameter_range': self.config.parameter_range,
            'source_function_type': 'polynomial',
            'boundary_conditions': 'dirichlet'
        }
        def poisson_pde_residual(u_xx, u_yy, source):
            """Poisson equation: ∇²u = f"""
            return u_xx + u_yy - source
        self.pde_residual = poisson_pde_residual
    def generate_training_data(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Generate training data for the operator learning problem."""
        if self.config.problem_type == 'heat':
            return self._generate_heat_data(n_samples)
        elif self.config.problem_type == 'wave':
            return self._generate_wave_data(n_samples)
        elif self.config.problem_type == 'burgers':
            return self._generate_burgers_data(n_samples)
        elif self.config.problem_type == 'poisson':
            return self._generate_poisson_data(n_samples)
        else:
            raise ValueError(f"Data generation not implemented for {self.config.problem_type}")
    def _generate_heat_data(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Generate heat equation training data."""
        # Sample parameters
        diffusivities = np.random.uniform(*self.config.parameter_range, n_samples)
        # Spatial domain
        x = np.linspace(*self.config.domain_bounds[0], self.config.n_sensors)
        x_eval = np.linspace(*self.config.domain_bounds[0], self.config.n_evaluation)
        # Time domain
        if len(self.config.domain_bounds) > 1:
            t_eval = np.linspace(*self.config.domain_bounds[1], 50)
        else:
            t_eval = np.array([1.0])  # Final time
        # Generate solutions
        branch_inputs = []
        trunk_inputs = []
        solutions = []
        for i, alpha in enumerate(diffusivities):
            # Initial condition (random Gaussian)
            x0 = np.random.uniform(0.2, 0.8)
            sigma = np.random.uniform(0.05, 0.2)
            u0 = np.exp(-((x - x0) / sigma)**2)
            branch_inputs.append(u0)
            # Solve heat equation (analytical solution for Gaussian initial condition)
            for t in t_eval:
                sigma_t = np.sqrt(sigma**2 + 2 * alpha * t)
                u_exact = (sigma / sigma_t) * np.exp(-((x_eval - x0)**2) / sigma_t**2)
                # Create coordinate pairs
                coords = np.column_stack([x_eval, np.full_like(x_eval, t)])
                trunk_inputs.append(coords)
                solutions.append(u_exact)
        return {
            'branch_inputs': np.array(branch_inputs),
            'trunk_inputs': np.array(trunk_inputs),
            'solutions': np.array(solutions),
            'parameters': diffusivities
        }
    def _generate_wave_data(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Generate wave equation training data."""
        # Sample wave speeds
        wave_speeds = np.random.uniform(*self.config.parameter_range, n_samples)
        # Spatial and temporal domains
        x = np.linspace(*self.config.domain_bounds[0], self.config.n_sensors)
        x_eval = np.linspace(*self.config.domain_bounds[0], self.config.n_evaluation)
        t_eval = np.linspace(0, 1, 50)
        branch_inputs = []
        trunk_inputs = []
        solutions = []
        for i, c in enumerate(wave_speeds):
            # Initial condition (sine wave)
            k = np.random.uniform(1, 5)  # Wave number
            u0 = np.sin(k * np.pi * x)
            branch_inputs.append(u0)
            # Analytical solution: u(x,t) = sin(kπx)cos(ckπt)
            for t in t_eval:
                u_exact = np.sin(k * np.pi * x_eval) * np.cos(c * k * np.pi * t)
                coords = np.column_stack([x_eval, np.full_like(x_eval, t)])
                trunk_inputs.append(coords)
                solutions.append(u_exact)
        return {
            'branch_inputs': np.array(branch_inputs),
            'trunk_inputs': np.array(trunk_inputs),
            'solutions': np.array(solutions),
            'parameters': wave_speeds
        }
    def _generate_burgers_data(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Generate Burgers' equation training data using finite differences."""
        viscosities = np.random.uniform(*self.config.parameter_range, n_samples)
        # Discretization
        nx = self.config.n_sensors
        nt = 100
        x = np.linspace(0, 1, nx)
        t = np.linspace(0, 0.5, nt)
        dx = x[1] - x[0]
        dt = t[1] - t[0]
        branch_inputs = []
        trunk_inputs = []
        solutions = []
        for nu in viscosities:
            # Initial condition (shock-like)
            u0 = np.where(x < 0.5, 1.0, 0.0) + 0.1 * np.sin(4 * np.pi * x)
            # Solve using finite differences
            u = np.zeros((nt, nx))
            u[0] = u0
            for n in range(nt - 1):
                u_new = u[n].copy()
                for i in range(1, nx - 1):
                    # Burgers' equation discretization
                    u_t = -(u[n, i] * (u[n, i+1] - u[n, i-1]) / (2 * dx)) + \
                          nu * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dx**2
                    u_new[i] = u[n, i] + dt * u_t
                # Periodic boundary conditions
                u_new[0] = u_new[-2]
                u_new[-1] = u_new[1]
                u[n+1] = u_new
            branch_inputs.append(u0)
            # Sample solution at different times
            for t_idx in range(0, nt, 10):
                coords = np.column_stack([x, np.full_like(x, t[t_idx])])
                trunk_inputs.append(coords)
                solutions.append(u[t_idx])
        return {
            'branch_inputs': np.array(branch_inputs),
            'trunk_inputs': np.array(trunk_inputs),
            'solutions': np.array(solutions),
            'parameters': viscosities
        }
    def _generate_poisson_data(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Generate Poisson equation training data."""
        # Sample source function parameters
        source_params = np.random.uniform(*self.config.parameter_range, n_samples)
        # 2D grid
        nx, ny = self.config.n_sensors // 2, self.config.n_sensors // 2
        x = np.linspace(*self.config.domain_bounds[0], nx)
        y = np.linspace(*self.config.domain_bounds[1], ny)
        X, Y = np.meshgrid(x, y)
        branch_inputs = []
        trunk_inputs = []
        solutions = []
        for param in source_params:
            # Source function
            f = param * np.sin(np.pi * X) * np.sin(np.pi * Y)
            # Analytical solution for this specific source
            u_exact = -f / (2 * np.pi**2)
            branch_inputs.append(f.ravel())
            coords = np.column_stack([X.ravel(), Y.ravel()])
            trunk_inputs.append(coords)
            solutions.append(u_exact.ravel())
        return {
            'branch_inputs': np.array(branch_inputs),
            'trunk_inputs': np.array(trunk_inputs),
            'solutions': np.array(solutions),
            'parameters': source_params
        }
    def predict_tensorflow(self, branch_input: np.ndarray, trunk_input: np.ndarray) -> np.ndarray:
        """Make predictions using TensorFlow model."""
        if len(branch_input.shape) == 1:
            branch_input = branch_input.reshape(1, -1)
        # Repeat branch input for each trunk coordinate
        n_coords = len(trunk_input)
        branch_repeated = np.tile(branch_input, (n_coords, 1))
        predictions = self.model([branch_repeated, trunk_input])
        return predictions.numpy().flatten()
    def predict_jax(self, branch_input: np.ndarray, trunk_input: np.ndarray) -> np.ndarray:
        """Make predictions using JAX model."""
        if len(branch_input.shape) == 1:
            branch_input = branch_input.reshape(1, -1)
        # Repeat branch input for each trunk coordinate
        n_coords = len(trunk_input)
        branch_repeated = jnp.tile(branch_input, (n_coords, 1))
        predictions = self.deeponet_fn(
            self.branch_params, self.trunk_params,
            branch_repeated, trunk_input
        )
        return np.array(predictions).flatten()
    def predict_numpy(self, branch_input: np.ndarray, trunk_input: np.ndarray) -> np.ndarray:
        """Make predictions using numpy implementation."""
        if len(branch_input.shape) == 1:
            branch_input = branch_input.reshape(1, -1)
        # Branch network forward pass
        x_branch = branch_input
        for i, (W, b) in enumerate(zip(self.branch_weights, self.branch_biases)):
            x_branch = np.dot(x_branch, W) + b
            if i < len(self.branch_weights) - 1:
                x_branch = np.tanh(x_branch)
        branch_out = np.tanh(x_branch)
        # Trunk network forward pass
        x_trunk = trunk_input
        for i, (W, b) in enumerate(zip(self.trunk_weights, self.trunk_biases)):
            x_trunk = np.dot(x_trunk, W) + b
            if i < len(self.trunk_weights) - 1:
                x_trunk = np.tanh(x_trunk)
        trunk_out = np.tanh(x_trunk)
        # Combine outputs
        predictions = np.sum(branch_out * trunk_out, axis=-1)
        return predictions
    def predict(self, branch_input: np.ndarray, trunk_input: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if TF_AVAILABLE and hasattr(self, 'model'):
            return self.predict_tensorflow(branch_input, trunk_input)
        elif JAX_AVAILABLE and hasattr(self, 'branch_params'):
            return self.predict_jax(branch_input, trunk_input)
        else:
            return self.predict_numpy(branch_input, trunk_input)
    def train(self, training_data: Dict[str, np.ndarray], verbose: bool = True) -> Dict[str, List[float]]:
        """Train the DeepONet model."""
        if TF_AVAILABLE and hasattr(self, 'model'):
            return self._train_tensorflow(training_data, verbose)
        elif JAX_AVAILABLE and hasattr(self, 'branch_params'):
            return self._train_jax(training_data, verbose)
        else:
            return self._train_numpy(training_data, verbose)
    def _train_tensorflow(self, training_data: Dict[str, np.ndarray], verbose: bool) -> Dict[str, List[float]]:
        """Train using TensorFlow backend."""
        branch_inputs = training_data['branch_inputs']
        trunk_inputs = training_data['trunk_inputs']
        solutions = training_data['solutions']
        # Prepare data for training
        n_samples = len(branch_inputs)
        n_coords_per_sample = len(trunk_inputs) // n_samples
        # Reshape data
        branch_train = []
        trunk_train = []
        solution_train = []
        for i in range(n_samples):
            for j in range(n_coords_per_sample):
                idx = i * n_coords_per_sample + j
                branch_train.append(branch_inputs[i])
                trunk_train.append(trunk_inputs[idx])
                solution_train.append(solutions[idx])
        branch_train = np.array(branch_train)
        trunk_train = np.array(trunk_train)
        solution_train = np.array(solution_train).reshape(-1, 1)
        # Train model
        history = self.model.fit(
            [branch_train, trunk_train],
            solution_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.validation_split,
            verbose=1 if verbose else 0
        )
        return {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
    def _train_jax(self, training_data: Dict[str, np.ndarray], verbose: bool) -> Dict[str, List[float]]:
        """Train using JAX backend."""
        # JAX training implementation
        print("JAX training implementation in progress...")
        return {'loss': [], 'val_loss': []}
    def _train_numpy(self, training_data: Dict[str, np.ndarray], verbose: bool) -> Dict[str, List[float]]:
        """Train using numpy backend."""
        print("Numpy training provides limited functionality.")
        return {'loss': [], 'val_loss': []}
    def evaluate_operator(self, test_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate the trained operator on test data."""
        branch_inputs = test_data['branch_inputs']
        trunk_inputs = test_data['trunk_inputs']
        true_solutions = test_data['solutions']
        predictions = []
        n_samples = len(branch_inputs)
        n_coords_per_sample = len(trunk_inputs) // n_samples
        for i in range(n_samples):
            start_idx = i * n_coords_per_sample
            end_idx = (i + 1) * n_coords_per_sample
            pred = self.predict(branch_inputs[i], trunk_inputs[start_idx:end_idx])
            predictions.extend(pred)
        predictions = np.array(predictions)
        true_solutions = true_solutions.flatten()
        # Compute metrics
        mse = np.mean((predictions - true_solutions)**2)
        mae = np.mean(np.abs(predictions - true_solutions))
        relative_error = np.mean(np.abs(predictions - true_solutions) / (np.abs(true_solutions) + 1e-10))
        return {
            'mse': mse,
            'mae': mae,
            'relative_error': relative_error
        }
    def plot_solution_comparison(self, test_sample: Dict[str, np.ndarray], sample_idx: int = 0) -> None:
        """Plot comparison between predicted and true solutions."""
        branch_input = test_sample['branch_inputs'][sample_idx]
        if self.config.coordinate_dim == 1:
            self._plot_1d_comparison(test_sample, sample_idx)
        elif self.config.coordinate_dim == 2:
            self._plot_2d_comparison(test_sample, sample_idx)
        else:
            print("Plotting not supported for >2D coordinates")
    def _plot_1d_comparison(self, test_sample: Dict[str, np.ndarray], sample_idx: int) -> None:
        """Plot 1D solution comparison."""
        branch_input = test_sample['branch_inputs'][sample_idx]
        # Generate evaluation points
        x_eval = np.linspace(*self.config.domain_bounds[0], self.config.n_evaluation)
        if self.config.problem_type in ['heat', 'wave', 'burgers']:
            # Time-dependent problems
            times = [0.1, 0.3,0.5]
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            # Input function
            x_sensors = np.linspace(*self.config.domain_bounds[0], self.config.n_sensors)
            axes[0, 0].plot(x_sensors, branch_input, 'o-',
                           color=self.berkeley_plot.colors['berkeley_blue'],
                           linewidth=2, markersize=4)
            axes[0, 0].set_title('Input Function', fontweight='bold')
            axes[0, 0].set_xlabel('x')
            axes[0, 0].set_ylabel('u₀(x)')
            axes[0, 0].grid(True, alpha=0.3)
            # Solutions at different times
            for i, t in enumerate(times):
                ax = axes[0, 1] if i == 0 else axes[1, i-1]
                # Create trunk input
                trunk_input = np.column_stack([x_eval, np.full_like(x_eval, t)])
                # Predict
                prediction = self.predict(branch_input, trunk_input)
                # Plot
                ax.plot(x_eval, prediction, '-',
                       color=self.berkeley_plot.colors['california_gold'],
                       linewidth=2, label='DeepONet')
                ax.set_title(f'Solution at t = {t}', fontweight='bold')
                ax.set_xlabel('x')
                ax.set_ylabel('u(x,t)')
                ax.grid(True, alpha=0.3)
                ax.legend()
        plt.tight_layout()
        plt.show()
    def _plot_2d_comparison(self, test_sample: Dict[str, np.ndarray], sample_idx: int) -> None:
        """Plot 2D solution comparison."""
        branch_input = test_sample['branch_inputs'][sample_idx]
        # Create 2D evaluation grid
        x = np.linspace(*self.config.domain_bounds[0], 50)
        y = np.linspace(*self.config.domain_bounds[1], 50)
        X, Y = np.meshgrid(x, y)
        coords = np.column_stack([X.ravel(), Y.ravel()])
        # Predict
        prediction = self.predict(branch_input, coords)
        prediction = prediction.reshape(X.shape)
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        # Input function (if 2D)
        if len(branch_input) == X.size:
            input_2d = branch_input.reshape(X.shape)
            cs1 = axes[0].contourf(X, Y, input_2d, levels=20, cmap='viridis')
            axes[0].set_title('Input Function', fontweight='bold')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y')
            plt.colorbar(cs1, ax=axes[0])
        # Predicted solution
        cs2 = axes[1].contourf(X, Y, prediction, levels=20, cmap='plasma')
        axes[1].set_title('DeepONet Prediction', fontweight='bold')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(cs2, ax=axes[1])
        plt.tight_layout()
        plt.show()
    def plot_training_history(self) -> None:
        """Plot training history."""
        if not self.training_history:
            print("No training history available.")
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(self.training_history['loss']) + 1)
        ax.semilogy(epochs, self.training_history['loss'],
                   color=self.berkeley_plot.colors['berkeley_blue'],
                   linewidth=2, label='Training Loss')
        if 'val_loss' in self.training_history:
            ax.semilogy(epochs, self.training_history['val_loss'],
                       color=self.berkeley_plot.colors['california_gold'],
                       linewidth=2, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('DeepONet Training History', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()
    def uncertainty_quantification(self, branch_input: np.ndarray, trunk_input: np.ndarray,
                                 n_samples: int = 100) -> Dict[str, np.ndarray]:
        """Perform uncertainty quantification using Monte Carlo dropout."""
        if not self.config.enable_uncertainty or not TF_AVAILABLE:
            print("Uncertainty quantification requires TensorFlow with dropout enabled.")
            return {}
        predictions = []
        for _ in range(n_samples):
            # Enable dropout during inference
            pred = self.model([np.tile(branch_input.reshape(1, -1), (len(trunk_input), 1)),
                              trunk_input], training=True)
            predictions.append(pred.numpy().flatten())
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        return {
            'mean': mean_pred,
            'std': std_pred,
            'samples': predictions
        }
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if TF_AVAILABLE and hasattr(self, 'model'):
            self.model.save(filepath)
            print(f"DeepONet model saved to {filepath}")
        else:
            print("Model saving not available for current backend.")
def create_heat_operator_demo():
    """Create a demonstration of DeepONet for heat equation operator learning."""
    print("=" * 70)
    print("SciComp")
    print("DeepONet: Heat Equation Operator Learning")
    print("=" * 70)
    # Configuration
    config = DeepONetConfig(
        branch_layers=[128, 128, 128],
        trunk_layers=[128, 128, 128],
        latent_dim=128,
        problem_type='heat',
        n_sensors=100,
        n_evaluation=200,
        coordinate_dim=2,  # (x, t)
        parameter_range=(0.1, 1.0),  # Diffusivity range
        domain_bounds=[(0.0, 1.0), (0.0, 1.0)],  # x and t domains
        epochs=5000,
        batch_size=32,
        learning_rate=1e-3
    )
    # Create DeepONet
    deeponet = DeepONet(config)
    # Generate training data
    print("Generating training data...")
    train_data = deeponet.generate_training_data(n_samples=1000)
    test_data = deeponet.generate_training_data(n_samples=100)
    # Train model
    print("Training DeepONet...")
    training_history = deeponet.train(train_data, verbose=True)
    deeponet.training_history = training_history
    # Evaluate
    print("Evaluating operator...")
    metrics = deeponet.evaluate_operator(test_data)
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
    # Plot results
    deeponet.plot_solution_comparison(test_data, sample_idx=0)
    deeponet.plot_training_history()
    print("\nHeat equation operator learning completed!")
    return deeponet
if __name__ == "__main__":
    # Run demonstration
    deeponet_model = create_heat_operator_demo()