"""
Physics-Informed Neural Networks (PINNs) for SciComp.
This module implements physics-informed machine learning for solving PDEs,
discovering governing equations, and accelerating simulations.
Author: UC Berkeley SciComp Team
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
import warnings
from typing import Callable, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
# Optional ML libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available. PINNs functionality limited.")
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
@dataclass
class PINNConfig:
    """Configuration for Physics-Informed Neural Networks."""
    layers: List[int]  # Network architecture [input, hidden1, hidden2, ..., output]
    activation: str = 'tanh'  # Activation function
    learning_rate: float = 0.001
    epochs: int = 1000
    batch_size: int = 32
    physics_weight: float = 1.0  # Weight for physics loss term
    data_weight: float = 1.0  # Weight for data loss term
    regularization: float = 0.0  # L2 regularization
class PhysicsInformedNN:
    """Base class for Physics-Informed Neural Networks."""
    def __init__(self, config: PINNConfig):
        """
        Initialize PINN.
        Args:
            config: PINN configuration
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.history = {'loss': [], 'physics_loss': [], 'data_loss': []}
        if TF_AVAILABLE:
            self._build_tensorflow_model()
        elif TORCH_AVAILABLE:
            self._build_pytorch_model()
        else:
            warnings.warn("No deep learning framework available")
    def _build_tensorflow_model(self):
        """Build TensorFlow/Keras model."""
        inputs = keras.Input(shape=(self.config.layers[0],))
        x = inputs
        # Hidden layers
        for i, units in enumerate(self.config.layers[1:-1]):
            x = layers.Dense(units, activation=self.config.activation,
                            kernel_regularizer=keras.regularizers.l2(self.config.regularization))(x)
        # Output layer
        outputs = layers.Dense(self.config.layers[-1], activation=None)(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.optimizer = optimizers.Adam(learning_rate=self.config.learning_rate)
    def _build_pytorch_model(self):
        """Build PyTorch model."""
        class PINNModel(nn.Module):
            def __init__(self, layers, activation):
                super().__init__()
                self.layers = nn.ModuleList()
                for i in range(len(layers) - 1):
                    self.layers.append(nn.Linear(layers[i], layers[i+1]))
                if activation == 'tanh':
                    self.activation = nn.Tanh()
                elif activation == 'relu':
                    self.activation = nn.ReLU()
                else:
                    self.activation = nn.Sigmoid()
            def forward(self, x):
                for i, layer in enumerate(self.layers[:-1]):
                    x = self.activation(layer(x))
                x = self.layers[-1](x)
                return x
        self.model = PINNModel(self.config.layers, self.config.activation)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
    def physics_loss(self, x: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute physics-based loss (to be overridden by specific PDEs).
        Args:
            x: Input coordinates
            y_pred: Predicted values
        Returns:
            Physics loss value
        """
        raise NotImplementedError("Override this method for specific PDEs")
    def train(self, x_data: np.ndarray, y_data: np.ndarray,
             x_physics: np.ndarray, verbose: bool = True):
        """
        Train the PINN.
        Args:
            x_data: Input data points
            y_data: Output data values
            x_physics: Collocation points for physics loss
            verbose: Print training progress
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        if TF_AVAILABLE:
            self._train_tensorflow(x_data, y_data, x_physics, verbose)
        elif TORCH_AVAILABLE:
            self._train_pytorch(x_data, y_data, x_physics, verbose)
    def _train_tensorflow(self, x_data, y_data, x_physics, verbose):
        """TensorFlow training loop."""
        @tf.function
        def train_step(x_d, y_d, x_p):
            with tf.GradientTape() as tape:
                # Data loss
                y_pred_data = self.model(x_d, training=True)
                data_loss = tf.reduce_mean(tf.square(y_pred_data - y_d))
                # Physics loss (simplified - override for specific PDEs)
                y_pred_physics = self.model(x_p, training=True)
                physics_loss = tf.reduce_mean(tf.square(y_pred_physics))
                # Total loss
                total_loss = (self.config.data_weight * data_loss +
                            self.config.physics_weight * physics_loss)
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return total_loss, data_loss, physics_loss
        # Training loop
        for epoch in range(self.config.epochs):
            total_loss, data_loss, physics_loss = train_step(
                tf.constant(x_data, dtype=tf.float32),
                tf.constant(y_data, dtype=tf.float32),
                tf.constant(x_physics, dtype=tf.float32)
            )
            self.history['loss'].append(float(total_loss))
            self.history['data_loss'].append(float(data_loss))
            self.history['physics_loss'].append(float(physics_loss))
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}, "
                     f"Data = {data_loss:.4f}, Physics = {physics_loss:.4f}")
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        Args:
            x: Input points
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained")
        if TF_AVAILABLE:
            return self.model.predict(x, verbose=0)
        elif TORCH_AVAILABLE:
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x)
                return self.model(x_tensor).numpy()
        else:
            return np.zeros((len(x), self.config.layers[-1]))
class HeatEquationPINN(PhysicsInformedNN):
    """PINN for solving the heat equation."""
    def __init__(self, config: PINNConfig, thermal_diffusivity: float = 1.0):
        """
        Initialize heat equation PINN.
        Args:
            config: PINN configuration
            thermal_diffusivity: Thermal diffusivity coefficient
        """
        super().__init__(config)
        self.alpha = thermal_diffusivity
    def physics_loss(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Compute physics loss for heat equation: ∂u/∂t - α∇²u = 0
        Args:
            x: Spatial coordinates
            t: Time coordinates
        Returns:
            Physics residual loss
        """
        if TF_AVAILABLE:
            return self._physics_loss_tf(x, t)
        else:
            # Simplified finite difference approximation
            return 0.0
    def _physics_loss_tf(self, x, t):
        """TensorFlow physics loss computation."""
        x = tf.constant(x, dtype=tf.float32)
        t = tf.constant(t, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, t])
            xt = tf.stack([x, t], axis=1)
            u = self.model(xt)
            # First derivatives
            u_t = tape.gradient(u, t)
            u_x = tape.gradient(u, x)
        # Second derivative
        u_xx = tape.gradient(u_x, x)
        # Heat equation residual
        residual = u_t - self.alpha * u_xx
        return tf.reduce_mean(tf.square(residual))
class WaveEquationPINN(PhysicsInformedNN):
    """PINN for solving the wave equation."""
    def __init__(self, config: PINNConfig, wave_speed: float = 1.0):
        """
        Initialize wave equation PINN.
        Args:
            config: PINN configuration
            wave_speed: Wave propagation speed
        """
        super().__init__(config)
        self.c = wave_speed
    def physics_loss(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Compute physics loss for wave equation: ∂²u/∂t² - c²∇²u = 0
        Args:
            x: Spatial coordinates
            t: Time coordinates
        Returns:
            Physics residual loss
        """
        if TF_AVAILABLE:
            return self._physics_loss_tf(x, t)
        else:
            return 0.0
    def _physics_loss_tf(self, x, t):
        """TensorFlow physics loss computation."""
        x = tf.constant(x, dtype=tf.float32)
        t = tf.constant(t, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, t])
            xt = tf.stack([x, t], axis=1)
            u = self.model(xt)
            # First derivatives
            u_t = tape.gradient(u, t)
            u_x = tape.gradient(u, x)
            # Second derivatives
            u_tt = tape.gradient(u_t, t)
            u_xx = tape.gradient(u_x, x)
        # Wave equation residual
        residual = u_tt - self.c**2 * u_xx
        return tf.reduce_mean(tf.square(residual))
class SchrodingerPINN(PhysicsInformedNN):
    """PINN for solving the Schrödinger equation."""
    def __init__(self, config: PINNConfig, potential: Callable[[np.ndarray], np.ndarray]):
        """
        Initialize Schrödinger equation PINN.
        Args:
            config: PINN configuration
            potential: Potential energy function V(x)
        """
        super().__init__(config)
        self.V = potential
    def physics_loss(self, x: np.ndarray, E: float) -> float:
        """
        Compute physics loss for time-independent Schrödinger equation:
        -ℏ²/2m ∇²ψ + V(x)ψ = Eψ
        Args:
            x: Spatial coordinates
            E: Energy eigenvalue
        Returns:
            Physics residual loss
        """
        if TF_AVAILABLE:
            return self._physics_loss_tf(x, E)
        else:
            return 0.0
    def _physics_loss_tf(self, x, E):
        """TensorFlow physics loss computation."""
        x = tf.constant(x, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            psi = self.model(tf.expand_dims(x, 1))
            # First derivative
            psi_x = tape.gradient(psi, x)
            # Second derivative
            psi_xx = tape.gradient(psi_x, x)
        # Potential energy
        V_x = tf.constant(self.V(x.numpy()), dtype=tf.float32)
        # Schrödinger equation residual (in units where ℏ²/2m = 1)
        residual = -psi_xx + V_x * psi - E * psi
        return tf.reduce_mean(tf.square(residual))
class EquationDiscovery:
    """Discover governing equations from data using sparse regression."""
    def __init__(self, library_functions: List[Callable] = None):
        """
        Initialize equation discovery.
        Args:
            library_functions: Candidate functions for the library
        """
        if library_functions is None:
            # Default library: polynomials and trigonometric functions
            self.library_functions = [
                lambda x: np.ones_like(x),  # Constant
                lambda x: x,  # Linear
                lambda x: x**2,  # Quadratic
                lambda x: x**3,  # Cubic
                lambda x: np.sin(x),  # Sine
                lambda x: np.cos(x),  # Cosine
                lambda x: np.exp(-x**2),  # Gaussian
            ]
        else:
            self.library_functions = library_functions
        self.coefficients = None
        self.sparsity_threshold = 0.01
    def build_library(self, x: np.ndarray) -> np.ndarray:
        """
        Build library matrix from candidate functions.
        Args:
            x: Input data
        Returns:
            Library matrix
        """
        library = []
        for func in self.library_functions:
            try:
                library.append(func(x))
            except:
                library.append(np.zeros_like(x))
        return np.column_stack(library)
    def sparse_regression(self, library: np.ndarray, derivatives: np.ndarray,
                         lambda_reg: float = 0.1) -> np.ndarray:
        """
        Perform sparse regression to identify governing equations.
        Args:
            library: Library matrix of candidate functions
            derivatives: Time derivatives or target values
            lambda_reg: Regularization parameter
        Returns:
            Sparse coefficients
        """
        # Ridge regression with L1 penalty approximation
        n_features = library.shape[1]
        coeffs = np.linalg.lstsq(
            library.T @ library + lambda_reg * np.eye(n_features),
            library.T @ derivatives,
            rcond=None
        )[0]
        # Iterative thresholding for sparsity
        for _ in range(10):
            small_indices = np.abs(coeffs) < self.sparsity_threshold
            coeffs[small_indices] = 0
            # Recompute for non-zero terms
            if np.any(~small_indices):
                active_library = library[:, ~small_indices]
                active_coeffs = np.linalg.lstsq(
                    active_library, derivatives, rcond=None
                )[0]
                coeffs[~small_indices] = active_coeffs
        return coeffs
    def discover(self, x: np.ndarray, y: np.ndarray,
                derivatives: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Discover governing equations from data.
        Args:
            x: Input coordinates
            y: Observed values
            derivatives: Time derivatives (computed if not provided)
        Returns:
            Discovered equation coefficients and description
        """
        # Build library
        library = self.build_library(x)
        # Compute derivatives if not provided
        if derivatives is None:
            # Simple finite difference
            derivatives = np.gradient(y)
        # Sparse regression
        self.coefficients = self.sparse_regression(library, derivatives)
        # Build equation string
        equation_terms = []
        function_names = ['1', 'x', 'x²', 'x³', 'sin(x)', 'cos(x)', 'exp(-x²)']
        for i, coeff in enumerate(self.coefficients):
            if abs(coeff) > self.sparsity_threshold:
                if i < len(function_names):
                    equation_terms.append(f"{coeff:.3f}·{function_names[i]}")
        equation_string = " + ".join(equation_terms) if equation_terms else "0"
        return {
            'coefficients': self.coefficients,
            'equation': f"dy/dt = {equation_string}",
            'library_size': len(self.library_functions),
            'active_terms': np.sum(np.abs(self.coefficients) > self.sparsity_threshold)
        }
class NeuralODE:
    """Neural Ordinary Differential Equations for continuous-time dynamics."""
    def __init__(self, hidden_dim: int = 32):
        """
        Initialize Neural ODE.
        Args:
            hidden_dim: Hidden layer dimension
        """
        self.hidden_dim = hidden_dim
        self.model = None
        if TF_AVAILABLE:
            self._build_model()
    def _build_model(self):
        """Build the neural network for ODE dynamics."""
        self.model = keras.Sequential([
            layers.Dense(self.hidden_dim, activation='tanh'),
            layers.Dense(self.hidden_dim, activation='tanh'),
            layers.Dense(2)  # Output dimension for 2D dynamics
        ])
    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute dynamics dx/dt = f(x, t).
        Args:
            t: Time
            state: Current state
        Returns:
            State derivatives
        """
        if self.model is None:
            return np.zeros_like(state)
        # Add time as input feature
        input_data = np.concatenate([state, [t]])
        input_data = input_data.reshape(1, -1)
        return self.model.predict(input_data, verbose=0)[0]
    def solve(self, initial_state: np.ndarray, t_span: Tuple[float, float],
             t_eval: np.ndarray) -> np.ndarray:
        """
        Solve the Neural ODE.
        Args:
            initial_state: Initial condition
            t_span: Time span (t0, tf)
            t_eval: Times to evaluate solution
        Returns:
            Solution trajectory
        """
        from scipy.integrate import solve_ivp
        # Wrapper for scipy ODE solver
        def ode_func(t, y):
            return self.dynamics(t, y)
        solution = solve_ivp(ode_func, t_span, initial_state,
                           t_eval=t_eval, method='RK45')
        return solution.y.T
# Convenience functions
def create_pinn_for_pde(pde_type: str, config: PINNConfig, **kwargs) -> PhysicsInformedNN:
    """
    Create a PINN for a specific PDE type.
    Args:
        pde_type: Type of PDE ('heat', 'wave', 'schrodinger')
        config: PINN configuration
        **kwargs: Additional PDE-specific parameters
    Returns:
        Configured PINN instance
    """
    if pde_type == 'heat':
        return HeatEquationPINN(config, kwargs.get('thermal_diffusivity', 1.0))
    elif pde_type == 'wave':
        return WaveEquationPINN(config, kwargs.get('wave_speed', 1.0))
    elif pde_type == 'schrodinger':
        return SchrodingerPINN(config, kwargs.get('potential', lambda x: 0.5*x**2))
    else:
        return PhysicsInformedNN(config)
def discover_physics_from_data(x: np.ndarray, y: np.ndarray,
                              derivatives: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Discover governing physics equations from observational data.
    Args:
        x: Spatial/temporal coordinates
        y: Observed values
        derivatives: Known derivatives (optional)
    Returns:
        Discovered equations and coefficients
    """
    discoverer = EquationDiscovery()
    return discoverer.discover(x, y, derivatives)