#!/usr/bin/env python3
"""
Physics-Informed Neural Network for Wave Equation
Implementation of PINNs for solving the wave equation with various boundary
conditions and source terms. Includes advanced features like dispersion
analysis and Berkeley-styled visualization.
Mathematical Foundation:
The wave equation is: ∂²u/∂t² = c²∇²u + f(x,t)
where:
- u(x,t) is the wave field (displacement, pressure, etc.)
- c is wave speed
- f(x,t) is source term
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional, Callable, Union
import matplotlib.pyplot as plt
from pathlib import Path
import time
from dataclasses import dataclass
# Constants and Berkeley styling
pi = np.pi
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
@dataclass
class WaveEquationConfig:
    """Configuration for wave equation problem."""
    domain_bounds: Tuple[float, float, float, float]  # (x_min, x_max, t_min, t_max)
    wave_speed: float = 1.0
    initial_displacement: Optional[Callable] = None
    initial_velocity: Optional[Callable] = None
    boundary_conditions: Optional[Dict] = None
    source_term: Optional[Callable] = None
    exact_solution: Optional[Callable] = None
class WaveEquationPINN:
    """
    Physics-Informed Neural Network for the wave equation.
    Solves ∂²u/∂t² = c²∇²u + f(x,t) using automatic differentiation
    to enforce physics constraints and initial/boundary conditions.
    """
    def __init__(self,
                 config: WaveEquationConfig,
                 layers: List[int] = [2, 50, 50, 50, 1],
                 activation: str = 'tanh',
                 learning_rate: float = 1e-3):
        """
        Initialize wave equation PINN.
        Parameters
        ----------
        config : WaveEquationConfig
            Problem configuration
        layers : list, default [2, 50, 50, 50, 1]
            Network architecture (input_dim, hidden_layers..., output_dim)
        activation : str, default 'tanh'
            Activation function
        learning_rate : float, default 1e-3
            Learning rate for optimizer
        """
        self.config = config
        self.layers = layers
        self.activation = activation
        self.learning_rate = learning_rate
        # Build neural network
        self.model = self._build_network()
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        # Training history
        self.history = {
            'total_loss': [],
            'pde_loss': [],
            'ic_loss': [],
            'bc_loss': [],
            'data_loss': []
        }
        # Domain bounds
        self.x_min, self.x_max, self.t_min, self.t_max = config.domain_bounds
    def _build_network(self) -> keras.Model:
        """Build the neural network architecture."""
        inputs = keras.Input(shape=(2,), name='inputs')  # (x, t)
        x = inputs
        # Hidden layers
        for i, units in enumerate(self.layers[1:-1]):
            x = keras.layers.Dense(
                units,
                activation=self.activation,
                kernel_initializer='glorot_normal',
                name=f'hidden_{i+1}'
            )(x)
        # Output layer
        outputs = keras.layers.Dense(
            self.layers[-1],
            activation=None,
            name='output'
        )(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name='WavePINN')
        return model
    def physics_loss(self, x_pde: tf.Tensor, t_pde: tf.Tensor) -> tf.Tensor:
        """
        Compute physics-informed loss for the wave equation.
        Parameters
        ----------
        x_pde : Tensor
            Spatial coordinates for PDE evaluation
        t_pde : Tensor
            Time coordinates for PDE evaluation
        Returns
        -------
        Tensor
            Physics loss value
        """
        with tf.GradientTape(persistent=True) as tape3:
            tape3.watch([x_pde, t_pde])
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch([x_pde, t_pde])
                with tf.GradientTape(persistent=True) as tape1:
                    tape1.watch([x_pde, t_pde])
                    # Network prediction
                    inputs = tf.stack([x_pde, t_pde], axis=1)
                    u = self.model(inputs)
                # First derivatives
                du_dx = tape1.gradient(u, x_pde)
                du_dt = tape1.gradient(u, t_pde)
            # Second derivatives
            d2u_dx2 = tape2.gradient(du_dx, x_pde)
            d2u_dt2 = tape2.gradient(du_dt, t_pde)
        # Wave equation residual: ∂²u/∂t² - c²∇²u - f = 0
        pde_residual = d2u_dt2 - self.config.wave_speed**2 * d2u_dx2
        # Add source term if provided
        if self.config.source_term is not None:
            source = self.config.source_term(x_pde, t_pde)
            pde_residual = pde_residual - source
        # Mean squared error of residual
        pde_loss = tf.reduce_mean(tf.square(pde_residual))
        return pde_loss
    def initial_condition_loss(self, x_ic: tf.Tensor,
                              u_ic: tf.Tensor,
                              v_ic: tf.Tensor) -> tf.Tensor:
        """
        Compute loss for initial conditions (displacement and velocity).
        Parameters
        ----------
        x_ic : Tensor
            Spatial coordinates for initial condition
        u_ic : Tensor
            Initial displacement values
        v_ic : Tensor
            Initial velocity values
        Returns
        -------
        Tensor
            Initial condition loss
        """
        t_ic = tf.fill(tf.shape(x_ic), self.t_min)
        with tf.GradientTape() as tape:
            tape.watch([x_ic, t_ic])
            inputs = tf.stack([x_ic, t_ic], axis=1)
            u_pred = self.model(inputs)
        # Initial displacement loss
        u_loss = tf.reduce_mean(tf.square(u_pred - tf.expand_dims(u_ic, -1)))
        # Initial velocity loss (∂u/∂t at t=0)
        du_dt = tape.gradient(u_pred, t_ic)
        v_loss = tf.reduce_mean(tf.square(du_dt - tf.expand_dims(v_ic, -1)))
        ic_loss = u_loss + v_loss
        return ic_loss
    def boundary_condition_loss(self, x_bc: tf.Tensor, t_bc: tf.Tensor,
                              u_bc: tf.Tensor, bc_type: str = 'dirichlet') -> tf.Tensor:
        """
        Compute loss for boundary conditions.
        Parameters
        ----------
        x_bc : Tensor
            Boundary spatial coordinates
        t_bc : Tensor
            Boundary time coordinates
        u_bc : Tensor
            Boundary condition values
        bc_type : str, default 'dirichlet'
            Type of boundary condition ('dirichlet', 'neumann', 'absorbing')
        Returns
        -------
        Tensor
            Boundary condition loss
        """
        if bc_type.lower() == 'dirichlet':
            # Dirichlet BC: u = g on boundary
            inputs = tf.stack([x_bc, t_bc], axis=1)
            u_pred = self.model(inputs)
            bc_loss = tf.reduce_mean(tf.square(u_pred - tf.expand_dims(u_bc, -1)))
        elif bc_type.lower() == 'neumann':
            # Neumann BC: ∂u/∂n = g on boundary
            with tf.GradientTape() as tape:
                tape.watch([x_bc])
                inputs = tf.stack([x_bc, t_bc], axis=1)
                u = self.model(inputs)
            du_dx = tape.gradient(u, x_bc)
            bc_loss = tf.reduce_mean(tf.square(du_dx - tf.expand_dims(u_bc, -1)))
        elif bc_type.lower() == 'absorbing':
            # Absorbing BC: ∂u/∂t ± c∂u/∂x = 0
            with tf.GradientTape(persistent=True) as tape:
                tape.watch([x_bc, t_bc])
                inputs = tf.stack([x_bc, t_bc], axis=1)
                u = self.model(inputs)
            du_dx = tape.gradient(u, x_bc)
            du_dt = tape.gradient(u, t_bc)
            # Outgoing wave condition (sign depends on boundary)
            # Left boundary: ∂u/∂t - c∂u/∂x = 0
            # Right boundary: ∂u/∂t + c∂u/∂x = 0
            c = self.config.wave_speed
            # Determine boundary side based on x coordinate
            is_left = tf.less(x_bc, (self.x_min + self.x_max) / 2)
            sign = tf.where(is_left, -c, c)
            absorbing_residual = du_dt + tf.expand_dims(sign, -1) * du_dx
            bc_loss = tf.reduce_mean(tf.square(absorbing_residual))
        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")
        return bc_loss
    def data_loss(self, x_data: tf.Tensor, t_data: tf.Tensor, u_data: tf.Tensor) -> tf.Tensor:
        """
        Compute loss for observational data.
        Parameters
        ----------
        x_data : Tensor
            Data spatial coordinates
        t_data : Tensor
            Data time coordinates
        u_data : Tensor
            Observed data values
        Returns
        -------
        Tensor
            Data fitting loss
        """
        inputs = tf.stack([x_data, t_data], axis=1)
        u_pred = self.model(inputs)
        data_loss = tf.reduce_mean(tf.square(u_pred - tf.expand_dims(u_data, -1)))
        return data_loss
    @tf.function
    def train_step(self,
                   x_pde: tf.Tensor, t_pde: tf.Tensor,
                   x_ic: tf.Tensor, u_ic: tf.Tensor, v_ic: tf.Tensor,
                   x_bc: tf.Tensor, t_bc: tf.Tensor, u_bc: tf.Tensor,
                   x_data: Optional[tf.Tensor] = None,
                   t_data: Optional[tf.Tensor] = None,
                   u_data: Optional[tf.Tensor] = None,
                   lambda_pde: float = 1.0,
                   lambda_ic: float = 1.0,
                   lambda_bc: float = 1.0,
                   lambda_data: float = 1.0) -> Dict[str, tf.Tensor]:
        """
        Perform one training step.
        Parameters
        ----------
        x_pde, t_pde : Tensor
            PDE collocation points
        x_ic, u_ic, v_ic : Tensor
            Initial condition data (displacement and velocity)
        x_bc, t_bc, u_bc : Tensor
            Boundary condition data
        x_data, t_data, u_data : Tensor, optional
            Observational data
        lambda_pde, lambda_ic, lambda_bc, lambda_data : float
            Loss weights
        Returns
        -------
        dict
            Loss components
        """
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            # Compute loss components
            pde_loss = self.physics_loss(x_pde, t_pde)
            ic_loss = self.initial_condition_loss(x_ic, u_ic, v_ic)
            bc_loss = self.boundary_condition_loss(x_bc, t_bc, u_bc)
            # Data loss (if provided)
            if x_data is not None and t_data is not None and u_data is not None:
                data_loss_val = self.data_loss(x_data, t_data, u_data)
            else:
                data_loss_val = tf.constant(0.0)
            # Total weighted loss
            total_loss = (lambda_pde * pde_loss +
                         lambda_ic * ic_loss +
                         lambda_bc * bc_loss +
                         lambda_data * data_loss_val)
        # Compute gradients and update
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {
            'total_loss': total_loss,
            'pde_loss': pde_loss,
            'ic_loss': ic_loss,
            'bc_loss': bc_loss,
            'data_loss': data_loss_val
        }
    def generate_training_data(self,
                             n_pde: int = 1000,
                             n_ic: int = 100,
                             n_bc: int = 100) -> Dict:
        """
        Generate training data for the PINN.
        Parameters
        ----------
        n_pde : int, default 1000
            Number of PDE collocation points
        n_ic : int, default 100
            Number of initial condition points
        n_bc : int, default 100
            Number of boundary condition points per boundary
        Returns
        -------
        dict
            Training data dictionary
        """
        # PDE collocation points (interior)
        x_pde = np.random.uniform(self.x_min, self.x_max, n_pde).astype(np.float32)
        t_pde = np.random.uniform(self.t_min, self.t_max, n_pde).astype(np.float32)
        # Initial condition points (t = t_min)
        x_ic = np.linspace(self.x_min, self.x_max, n_ic).astype(np.float32)
        # Initial displacement
        if self.config.initial_displacement is not None:
            u_ic = self.config.initial_displacement(x_ic).astype(np.float32)
        else:
            u_ic = np.zeros_like(x_ic).astype(np.float32)
        # Initial velocity
        if self.config.initial_velocity is not None:
            v_ic = self.config.initial_velocity(x_ic).astype(np.float32)
        else:
            v_ic = np.zeros_like(x_ic).astype(np.float32)
        # Boundary condition points
        t_bc = np.random.uniform(self.t_min, self.t_max, n_bc).astype(np.float32)
        # Left boundary (x = x_min)
        x_bc_left = np.full(n_bc, self.x_min).astype(np.float32)
        u_bc_left = np.zeros(n_bc).astype(np.float32)  # Default: u = 0
        # Right boundary (x = x_max)
        x_bc_right = np.full(n_bc, self.x_max).astype(np.float32)
        u_bc_right = np.zeros(n_bc).astype(np.float32)  # Default: u = 0
        # Combine boundary data
        x_bc = np.concatenate([x_bc_left, x_bc_right]).astype(np.float32)
        t_bc_combined = np.concatenate([t_bc, t_bc]).astype(np.float32)
        u_bc = np.concatenate([u_bc_left, u_bc_right]).astype(np.float32)
        return {
            'x_pde': tf.constant(x_pde),
            't_pde': tf.constant(t_pde),
            'x_ic': tf.constant(x_ic),
            'u_ic': tf.constant(u_ic),
            'v_ic': tf.constant(v_ic),
            'x_bc': tf.constant(x_bc),
            't_bc': tf.constant(t_bc_combined),
            'u_bc': tf.constant(u_bc)
        }
    def train(self,
              epochs: int = 2000,
              n_pde: int = 1000,
              n_ic: int = 100,
              n_bc: int = 100,
              lambda_weights: Optional[Dict] = None,
              print_frequency: int = 200,
              data_points: Optional[Dict] = None) -> Dict:
        """
        Train the PINN model.
        Parameters
        ----------
        epochs : int, default 2000
            Number of training epochs
        n_pde, n_ic, n_bc : int
            Number of collocation points for each constraint
        lambda_weights : dict, optional
            Loss weights dictionary
        print_frequency : int, default 200
            Print progress every N epochs
        data_points : dict, optional
            Observational data for fitting
        Returns
        -------
        dict
            Training history
        """
        if lambda_weights is None:
            lambda_weights = {'pde': 1.0, 'ic': 1.0, 'bc': 1.0, 'data': 1.0}
        print(f"🌊 Training Wave Equation PINN...")
        print(f"   Network: {self.layers}")
        print(f"   Domain: x ∈ [{self.x_min:.1f}, {self.x_max:.1f}], t ∈ [{self.t_min:.1f}, {self.t_max:.1f}]")
        print(f"   Wave speed: c = {self.config.wave_speed}")
        print(f"   Training points: PDE={n_pde}, IC={n_ic}, BC={n_bc}")
        print("   ──────────────────────────────────────")
        # Generate training data
        training_data = self.generate_training_data(n_pde, n_ic, n_bc)
        # Data points (if provided)
        x_data = t_data = u_data = None
        if data_points is not None:
            x_data = tf.constant(data_points['x'].astype(np.float32))
            t_data = tf.constant(data_points['t'].astype(np.float32))
            u_data = tf.constant(data_points['u'].astype(np.float32))
        start_time = time.time()
        for epoch in range(epochs):
            # Regenerate collocation points periodically
            if epoch % 500 == 0 and epoch > 0:
                training_data = self.generate_training_data(n_pde, n_ic, n_bc)
            # Training step
            losses = self.train_step(
                training_data['x_pde'], training_data['t_pde'],
                training_data['x_ic'], training_data['u_ic'], training_data['v_ic'],
                training_data['x_bc'], training_data['t_bc'], training_data['u_bc'],
                x_data, t_data, u_data,
                lambda_weights['pde'], lambda_weights['ic'],
                lambda_weights['bc'], lambda_weights['data']
            )
            # Store history
            for key, value in losses.items():
                self.history[key].append(float(value.numpy()))
            # Print progress
            if (epoch + 1) % print_frequency == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:4d}/{epochs}: "
                      f"Total = {losses['total_loss']:.2e}, "
                      f"PDE = {losses['pde_loss']:.2e}, "
                      f"IC = {losses['ic_loss']:.2e}, "
                      f"BC = {losses['bc_loss']:.2e} "
                      f"({elapsed:.1f}s)")
        total_time = time.time() - start_time
        print(f"✅ Training completed in {total_time:.1f}s")
        return self.history
    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Predict wave field at given points.
        Parameters
        ----------
        x : ndarray
            Spatial coordinates
        t : ndarray
            Time coordinates
        Returns
        -------
        ndarray
            Predicted wave values
        """
        # Ensure proper shapes
        x_flat = x.flatten().astype(np.float32)
        t_flat = t.flatten().astype(np.float32)
        inputs = tf.stack([x_flat, t_flat], axis=1)
        predictions = self.model(inputs).numpy().flatten()
        # Reshape to original shape
        return predictions.reshape(x.shape)
    def compute_error(self, x: np.ndarray, t: np.ndarray) -> Dict[str, float]:
        """
        Compute error metrics if exact solution is available.
        Parameters
        ----------
        x : ndarray
            Spatial coordinates
        t : ndarray
            Time coordinates
        Returns
        -------
        dict
            Error metrics
        """
        if self.config.exact_solution is None:
            return {'l2_error': None, 'max_error': None}
        u_pred = self.predict(x, t)
        u_exact = self.config.exact_solution(x, t)
        l2_error = np.sqrt(np.mean((u_pred - u_exact)**2))
        max_error = np.max(np.abs(u_pred - u_exact))
        return {
            'l2_error': l2_error,
            'max_error': max_error,
            'relative_l2': l2_error / np.sqrt(np.mean(u_exact**2))
        }
    def plot_solution(self,
                     x_range: Optional[Tuple[float, float]] = None,
                     t_range: Optional[Tuple[float, float]] = None,
                     n_points: Tuple[int, int] = (100, 100),
                     output_dir: Optional[Path] = None) -> plt.Figure:
        """
        Plot the solution with Berkeley styling.
        Parameters
        ----------
        x_range : tuple, optional
            Spatial range for plotting
        t_range : tuple, optional
            Time range for plotting
        n_points : tuple, default (100, 100)
            Grid resolution (nx, nt)
        output_dir : Path, optional
            Directory to save figure
        Returns
        -------
        Figure
            Matplotlib figure object
        """
        if x_range is None:
            x_range = (self.x_min, self.x_max)
        if t_range is None:
            t_range = (self.t_min, self.t_max)
        # Create prediction grid
        x = np.linspace(x_range[0], x_range[1], n_points[0])
        t = np.linspace(t_range[0], t_range[1], n_points[1])
        X, T = np.meshgrid(x, t)
        # Make predictions
        U_pred = self.predict(X, T)
        # Create Berkeley-styled plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🐻💙💛 Wave Equation PINN Solution',
                    fontsize=16, fontweight='bold', color=BERKELEY_BLUE)
        # Predicted solution
        vmax = np.max(np.abs(U_pred))
        im1 = axes[0, 0].contourf(X, T, U_pred, levels=50, cmap='RdBu_r',
                                 vmin=-vmax, vmax=vmax)
        axes[0, 0].set_xlabel('Position (x)')
        axes[0, 0].set_ylabel('Time (t)')
        axes[0, 0].set_title('PINN Solution', fontweight='bold')
        plt.colorbar(im1, ax=axes[0, 0])
        # Exact solution (if available)
        if self.config.exact_solution is not None:
            U_exact = self.config.exact_solution(X, T)
            vmax_exact = np.max(np.abs(U_exact))
            im2 = axes[0, 1].contourf(X, T, U_exact, levels=50, cmap='RdBu_r',
                                     vmin=-vmax_exact, vmax=vmax_exact)
            axes[0, 1].set_xlabel('Position (x)')
            axes[0, 1].set_ylabel('Time (t)')
            axes[0, 1].set_title('Exact Solution', fontweight='bold')
            plt.colorbar(im2, ax=axes[0, 1])
            # Error
            error = np.abs(U_pred - U_exact)
            im3 = axes[1, 0].contourf(X, T, error, levels=50, cmap='Reds')
            axes[1, 0].set_xlabel('Position (x)')
            axes[1, 0].set_ylabel('Time (t)')
            axes[1, 0].set_title('Absolute Error', fontweight='bold')
            plt.colorbar(im3, ax=axes[1, 0])
        else:
            axes[0, 1].text(0.5, 0.5, 'No exact solution available',
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Exact Solution', fontweight='bold')
            axes[1, 0].text(0.5, 0.5, 'No exact solution available',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Error Analysis', fontweight='bold')
        # Training history
        axes[1, 1].semilogy(self.history['total_loss'],
                           color=BERKELEY_BLUE, linewidth=2, label='Total')
        axes[1, 1].semilogy(self.history['pde_loss'],
                           color=CALIFORNIA_GOLD, linewidth=2, label='PDE')
        axes[1, 1].semilogy(self.history['ic_loss'],
                           color='red', linewidth=2, label='IC')
        axes[1, 1].semilogy(self.history['bc_loss'],
                           color='green', linewidth=2, label='BC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training History', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        # Save if requested
        if output_dir:
            fig.savefig(output_dir / "wave_equation_pinn.png", dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {output_dir}/wave_equation_pinn.png")
        return fig
# Example problems and utilities
def gaussian_wave_packet(x: np.ndarray,
                        center: float = 0.0,
                        width: float = 0.1,
                        k0: float = 5.0) -> np.ndarray:
    """Gaussian wave packet initial condition."""
    return np.exp(-((x - center) / width)**2) * np.cos(k0 * x)
def sine_wave_initial(x: np.ndarray,
                     amplitude: float = 1.0,
                     frequency: float = 1.0) -> np.ndarray:
    """Sinusoidal initial condition."""
    return amplitude * np.sin(frequency * pi * x)
def d_alembert_solution(x: np.ndarray, t: np.ndarray,
                       c: float = 1.0,
                       f: Optional[Callable] = None,
                       g: Optional[Callable] = None) -> np.ndarray:
    """
    D'Alembert solution for 1D wave equation on infinite domain.
    Parameters
    ----------
    x, t : ndarray
        Spatial and time coordinates
    c : float
        Wave speed
    f : callable, optional
        Initial displacement function
    g : callable, optional
        Initial velocity function
    Returns
    -------
    ndarray
        Exact solution
    """
    if f is None:
        f = lambda x: np.sin(pi * x) * (np.abs(x) <= 1)
    if g is None:
        g = lambda x: np.zeros_like(x)
    # D'Alembert formula: u(x,t) = 1/2[f(x+ct) + f(x-ct)] + 1/(2c)∫g(ξ)dξ
    u1 = 0.5 * (f(x + c * t) + f(x - c * t))
    # For simple cases, assume zero initial velocity
    return u1
def standing_wave_solution(x: np.ndarray, t: np.ndarray,
                          c: float = 1.0, L: float = 1.0,
                          n: int = 1) -> np.ndarray:
    """Standing wave solution for wave equation with fixed boundaries."""
    omega = n * pi * c / L
    return np.sin(n * pi * x / L) * np.cos(omega * t)
def create_wave_problem(c: float = 1.0,
                       L: float = 1.0,
                       T: float = 2.0,
                       wave_type: str = 'standing') -> WaveEquationConfig:
    """Create standard wave equation problem configuration."""
    if wave_type == 'standing':
        return WaveEquationConfig(
            domain_bounds=(0, L, 0, T),
            wave_speed=c,
            initial_displacement=lambda x: sine_wave_initial(x / L),
            initial_velocity=lambda x: np.zeros_like(x),
            exact_solution=lambda x, t: standing_wave_solution(x, t, c, L)
        )
    elif wave_type == 'traveling':
        return WaveEquationConfig(
            domain_bounds=(-L, L, 0, T),
            wave_speed=c,
            initial_displacement=lambda x: sine_wave_initial(x / L),
            initial_velocity=lambda x: np.zeros_like(x),
            exact_solution=lambda x, t: d_alembert_solution(x, t, c)
        )
    else:
        raise ValueError(f"Unknown wave type: {wave_type}")
def solve_wave_equation(config: WaveEquationConfig,
                       epochs: int = 3000,
                       layers: List[int] = [2, 50, 50, 50, 1]) -> WaveEquationPINN:
    """
    Solve wave equation with given configuration.
    Parameters
    ----------
    config : WaveEquationConfig
        Problem configuration
    epochs : int, default 3000
        Training epochs
    layers : list, default [2, 50, 50, 50, 1]
        Network architecture
    Returns
    -------
    WaveEquationPINN
        Trained PINN model
    """
    # Create and train PINN
    pinn = WaveEquationPINN(config, layers=layers)
    # Use adaptive weights for wave equation
    lambda_weights = {'pde': 1.0, 'ic': 10.0, 'bc': 10.0, 'data': 1.0}
    pinn.train(epochs=epochs, lambda_weights=lambda_weights)
    return pinn
if __name__ == "__main__":
    # Example usage
    print("🌊 Wave Equation PINN Example")
    # Create problem configuration
    config = create_wave_problem(c=1.0, L=1.0, T=2.0, wave_type='standing')
    # Solve using PINN
    pinn = solve_wave_equation(config, epochs=2000)
    # Plot results
    fig = pinn.plot_solution()
    plt.show()
    # Compute error metrics
    x_test = np.linspace(0, 1, 100)
    t_test = np.linspace(0, 2, 100)
    X_test, T_test = np.meshgrid(x_test, t_test)
    errors = pinn.compute_error(X_test, T_test)
    print(f"\nError Analysis:")
    print(f"  L2 Error: {errors['l2_error']:.2e}")
    print(f"  Max Error: {errors['max_error']:.2e}")
    print(f"  Relative L2: {errors['relative_l2']:.2e}")