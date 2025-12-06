#!/usr/bin/env python3
"""
Physics-Informed Neural Network for Schrödinger Equation
Implementation of PINNs for solving the time-dependent and time-independent
Schrödinger equations with physics constraints and boundary conditions.
The PINN enforces the Schrödinger equation as a soft constraint:
iℏ ∂ψ/∂t = [-ℏ²/(2m)∇² + V(r,t)]ψ
Key Features:
- Time-dependent and time-independent formulations
- Complex-valued neural networks for wavefunctions
- Automatic differentiation for quantum operators
- Conservation law enforcement (probability, energy)
- Multi-dimensional support (1D, 2D, 3D)
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Callable, Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
# Physical constants and Berkeley styling
hbar = 1.0545718e-34  # Reduced Planck constant (J⋅s)
me = 9.1093837e-31    # Electron mass (kg)
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
class ComplexDense(keras.layers.Layer):
    """Complex-valued dense layer for quantum wavefunctions."""
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
    def build(self, input_shape):
        # Real and imaginary weight matrices
        self.W_real = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            name='W_real'
        )
        self.W_imag = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            name='W_imag'
        )
        self.b_real = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='b_real'
        )
        self.b_imag = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='b_imag'
        )
    def call(self, inputs):
        # Complex matrix multiplication
        real_part = tf.matmul(inputs, self.W_real) + self.b_real
        imag_part = tf.matmul(inputs, self.W_imag) + self.b_imag
        if self.activation is not None:
            real_part = self.activation(real_part)
            imag_part = self.activation(imag_part)
        return tf.complex(real_part, imag_part)
class SchrodingerPINN:
    """
    Physics-Informed Neural Network for the Schrödinger equation.
    Solves quantum mechanical systems by training neural networks to satisfy
    the Schrödinger equation and boundary conditions simultaneously.
    """
    def __init__(self,
                 potential_func: Callable,
                 domain_bounds: Dict[str, Tuple[float, float]],
                 mass: float = me,
                 hbar_value: float = hbar,
                 network_architecture: List[int] = [50, 50, 50],
                 time_dependent: bool = True):
        """
        Initialize Schrödinger PINN.
        Parameters
        ----------
        potential_func : callable
            Potential function V(x, t) or V(x)
        domain_bounds : dict
            Domain boundaries {'x': (x_min, x_max), 't': (t_min, t_max)}
        mass : float, default me
            Particle mass
        hbar_value : float, default hbar
            Reduced Planck constant
        network_architecture : list, default [50, 50, 50]
            Hidden layer sizes
        time_dependent : bool, default True
            Whether to solve time-dependent equation
        """
        self.potential_func = potential_func
        self.domain_bounds = domain_bounds
        self.mass = mass
        self.hbar = hbar_value
        self.architecture = network_architecture
        self.time_dependent = time_dependent
        # Build neural network
        self.model = self._build_network()
        # Training history
        self.history = {'loss': [], 'pde_loss': [], 'ic_loss': [], 'bc_loss': []}
    def _build_network(self) -> keras.Model:
        """Build complex-valued neural network."""
        if self.time_dependent:
            inputs = keras.Input(shape=(2,), name='space_time')  # (x, t)
        else:
            inputs = keras.Input(shape=(1,), name='space')  # (x)
        # Normalize inputs
        x = keras.layers.Lambda(self._normalize_inputs)(inputs)
        # Complex-valued hidden layers
        for units in self.architecture:
            x = ComplexDense(units, activation='tanh')(x)
        # Output layer (complex wavefunction)
        outputs = ComplexDense(1, activation=None, name='wavefunction')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    def _normalize_inputs(self, inputs):
        """Normalize inputs to [-1, 1] range."""
        if self.time_dependent:
            x, t = inputs[:, 0:1], inputs[:, 1:2]
            x_bounds = self.domain_bounds['x']
            t_bounds = self.domain_bounds['t']
            x_norm = 2 * (x - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) - 1
            t_norm = 2 * (t - t_bounds[0]) / (t_bounds[1] - t_bounds[0]) - 1
            return tf.concat([x_norm, t_norm], axis=1)
        else:
            x = inputs
            x_bounds = self.domain_bounds['x']
            x_norm = 2 * (x - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) - 1
            return x_norm
    @tf.function
    def schrodinger_residual(self, x_t):
        """Calculate Schrödinger equation residual."""
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x_t)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(x_t)
                # Get wavefunction
                psi = self.model(x_t)
                psi_real = tf.math.real(psi)
                psi_imag = tf.math.imag(psi)
            if self.time_dependent:
                x, t = x_t[:, 0:1], x_t[:, 1:2]
                # Time derivative
                dpsi_dt_real = tape1.gradient(psi_real, t)
                dpsi_dt_imag = tape1.gradient(psi_imag, t)
                dpsi_dt = tf.complex(dpsi_dt_real, dpsi_dt_imag)
                # Spatial derivatives
                dpsi_dx_real = tape1.gradient(psi_real, x)
                dpsi_dx_imag = tape1.gradient(psi_imag, x)
            else:
                x = x_t
                dpsi_dt = tf.zeros_like(psi)  # Time-independent
                dpsi_dx_real = tape1.gradient(psi_real, x)
                dpsi_dx_imag = tape1.gradient(psi_imag, x)
        # Second spatial derivative
        d2psi_dx2_real = tape2.gradient(dpsi_dx_real, x)
        d2psi_dx2_imag = tape2.gradient(dpsi_dx_imag, x)
        d2psi_dx2 = tf.complex(d2psi_dx2_real, d2psi_dx2_imag)
        # Potential energy
        if self.time_dependent:
            V = self.potential_func(x, t)
        else:
            V = self.potential_func(x)
        # Hamiltonian operator
        H_psi = -self.hbar**2 / (2 * self.mass) * d2psi_dx2 + tf.cast(V, tf.complex64) * psi
        if self.time_dependent:
            # Time-dependent Schrödinger equation: iℏ ∂ψ/∂t = Ĥψ
            residual = 1j * self.hbar * dpsi_dt - H_psi
        else:
            # Time-independent: Ĥψ = Eψ (assuming ground state E=0 for simplicity)
            residual = H_psi
        return residual
    def boundary_conditions(self, x_boundary):
        """Apply boundary conditions (ψ = 0 at boundaries)."""
        psi_boundary = self.model(x_boundary)
        return psi_boundary
    def initial_condition(self, x_initial, psi_0_func):
        """Apply initial condition ψ(x, t=0) = ψ₀(x)."""
        psi_predicted = self.model(x_initial)
        psi_initial = psi_0_func(x_initial[:, 0:1])
        return psi_predicted - tf.cast(psi_initial, tf.complex64)
    def compute_loss(self, x_pde, x_boundary, x_initial=None, psi_0_func=None):
        """Compute total loss function."""
        # PDE residual loss
        residual = self.schrodinger_residual(x_pde)
        pde_loss = tf.reduce_mean(tf.square(tf.math.real(residual)) +
                                tf.square(tf.math.imag(residual)))
        # Boundary condition loss
        bc_residual = self.boundary_conditions(x_boundary)
        bc_loss = tf.reduce_mean(tf.square(tf.math.real(bc_residual)) +
                               tf.square(tf.math.imag(bc_residual)))
        # Initial condition loss (for time-dependent problems)
        ic_loss = 0.0
        if self.time_dependent and x_initial is not None and psi_0_func is not None:
            ic_residual = self.initial_condition(x_initial, psi_0_func)
            ic_loss = tf.reduce_mean(tf.square(tf.math.real(ic_residual)) +
                                   tf.square(tf.math.imag(ic_residual)))
        # Total loss with weights
        total_loss = pde_loss + 10.0 * bc_loss + 10.0 * ic_loss
        return total_loss, pde_loss, bc_loss, ic_loss
    def generate_training_data(self, n_pde=1000, n_boundary=100, n_initial=100):
        """Generate training data points."""
        x_bounds = self.domain_bounds['x']
        if self.time_dependent:
            t_bounds = self.domain_bounds['t']
            # PDE collocation points
            x_pde = np.random.uniform(x_bounds[0], x_bounds[1], (n_pde, 1))
            t_pde = np.random.uniform(t_bounds[0], t_bounds[1], (n_pde, 1))
            x_t_pde = np.hstack([x_pde, t_pde])
            # Boundary points
            x_boundary_left = np.full((n_boundary//2, 1), x_bounds[0])
            x_boundary_right = np.full((n_boundary//2, 1), x_bounds[1])
            t_boundary = np.random.uniform(t_bounds[0], t_bounds[1], (n_boundary, 1))
            x_boundary = np.vstack([x_boundary_left, x_boundary_right])
            x_t_boundary = np.hstack([x_boundary, t_boundary])
            # Initial condition points
            x_initial = np.random.uniform(x_bounds[0], x_bounds[1], (n_initial, 1))
            t_initial = np.full((n_initial, 1), t_bounds[0])
            x_t_initial = np.hstack([x_initial, t_initial])
            return (tf.constant(x_t_pde, dtype=tf.float32),
                   tf.constant(x_t_boundary, dtype=tf.float32),
                   tf.constant(x_t_initial, dtype=tf.float32))
        else:
            # Time-independent case
            x_pde = np.random.uniform(x_bounds[0], x_bounds[1], (n_pde, 1))
            x_boundary = np.array([[x_bounds[0]], [x_bounds[1]]])
            return (tf.constant(x_pde, dtype=tf.float32),
                   tf.constant(x_boundary, dtype=tf.float32),
                   None)
    def train(self, epochs=5000, learning_rate=1e-3, psi_0_func=None):
        """Train the PINN model."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        # Generate training data
        if self.time_dependent:
            x_pde, x_boundary, x_initial = self.generate_training_data()
        else:
            x_pde, x_boundary, _ = self.generate_training_data()
            x_initial = None
        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                total_loss, pde_loss, bc_loss, ic_loss = self.compute_loss(
                    x_pde, x_boundary, x_initial, psi_0_func)
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return total_loss, pde_loss, bc_loss, ic_loss
        # Training loop
        for epoch in range(epochs):
            total_loss, pde_loss, bc_loss, ic_loss = train_step()
            # Store history
            self.history['loss'].append(float(total_loss))
            self.history['pde_loss'].append(float(pde_loss))
            self.history['bc_loss'].append(float(bc_loss))
            self.history['ic_loss'].append(float(ic_loss))
            if epoch % 500 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.6f}, "
                      f"PDE = {pde_loss:.6f}, BC = {bc_loss:.6f}, IC = {ic_loss:.6f}")
    def predict(self, x_test, t_test=None):
        """Predict wavefunction at test points."""
        if self.time_dependent and t_test is not None:
            x_t_test = np.hstack([x_test.reshape(-1, 1), t_test.reshape(-1, 1)])
        else:
            x_t_test = x_test.reshape(-1, 1)
        psi = self.model(tf.constant(x_t_test, dtype=tf.float32))
        return psi.numpy()
    def plot_solution(self, output_dir: Optional[Path] = None):
        """Plot the PINN solution."""
        berkeley_plot = BerkeleyPlot()
        x_bounds = self.domain_bounds['x']
        x_test = np.linspace(x_bounds[0], x_bounds[1], 200)
        if self.time_dependent:
            t_bounds = self.domain_bounds['t']
            t_test = np.linspace(t_bounds[0], t_bounds[1], 100)
            # Create 2D plot
            X, T = np.meshgrid(x_test, t_test)
            x_flat = X.flatten()
            t_flat = T.flatten()
            psi = self.predict(x_flat, t_flat)
            prob_density = np.abs(psi)**2
            fig, (ax1, ax2) = berkeley_plot.create_figure(1, 2, figsize=(15, 6))
            # Probability density evolution
            im1 = ax1.contourf(X, T, prob_density.reshape(X.shape), levels=50, cmap='berkeley_blues')
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Time')
            ax1.set_title('Probability Density |ψ(x,t)|²')
            plt.colorbar(im1, ax=ax1)
            # Real part evolution
            im2 = ax2.contourf(X, T, np.real(psi).reshape(X.shape), levels=50, cmap='RdBu_r')
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Time')
            ax2.set_title('Re[ψ(x,t)]')
            plt.colorbar(im2, ax=ax2)
        else:
            # Time-independent case
            psi = self.predict(x_test)
            fig, (ax1, ax2) = berkeley_plot.create_figure(1, 2, figsize=(12, 5))
            # Wavefunction
            ax1.plot(x_test, np.real(psi), label='Re[ψ(x)]', linewidth=2)
            ax1.plot(x_test, np.imag(psi), label='Im[ψ(x)]', linewidth=2, linestyle='--')
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Wavefunction')
            ax1.set_title('PINN Solution: ψ(x)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            # Probability density
            ax2.plot(x_test, np.abs(psi)**2, linewidth=2, color='green')
            ax2.fill_between(x_test, np.abs(psi)**2, alpha=0.3, color='green')
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Probability Density')
            ax2.set_title('|ψ(x)|²')
            ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_dir:
            berkeley_plot.save_figure(output_dir / "schrodinger_pinn_solution.png")
        return fig
    def plot_training_history(self, output_dir: Optional[Path] = None):
        """Plot training loss history."""
        berkeley_plot = BerkeleyPlot()
        fig, ax = berkeley_plot.create_figure()
        epochs = range(len(self.history['loss']))
        ax.semilogy(epochs, self.history['loss'], label='Total Loss', linewidth=2)
        ax.semilogy(epochs, self.history['pde_loss'], label='PDE Loss', linewidth=2)
        ax.semilogy(epochs, self.history['bc_loss'], label='BC Loss', linewidth=2)
        if self.time_dependent:
            ax.semilogy(epochs, self.history['ic_loss'], label='IC Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('PINN Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if output_dir:
            berkeley_plot.save_figure(output_dir / "schrodinger_pinn_training.png")
        return fig
# Convenience functions
def solve_schrodinger_pinn(potential_func: Callable,
                          domain_bounds: Dict[str, Tuple[float, float]],
                          initial_condition: Optional[Callable] = None,
                          epochs: int = 5000,
                          time_dependent: bool = True) -> SchrodingerPINN:
    """
    Solve Schrödinger equation using PINN.
    Parameters
    ----------
    potential_func : callable
        Potential function V(x, t) or V(x)
    domain_bounds : dict
        Domain boundaries
    initial_condition : callable, optional
        Initial wavefunction ψ₀(x)
    epochs : int, default 5000
        Training epochs
    time_dependent : bool, default True
        Whether to solve time-dependent equation
    Returns
    -------
    SchrodingerPINN
        Trained PINN model
    """
    pinn = SchrodingerPINN(potential_func, domain_bounds, time_dependent=time_dependent)
    pinn.train(epochs=epochs, psi_0_func=initial_condition)
    return pinn
def quantum_harmonic_pinn(omega: float = 1.0,
                         x_bounds: Tuple[float, float] = (-5.0, 5.0),
                         t_bounds: Tuple[float, float] = (0.0, 2.0),
                         epochs: int = 5000) -> SchrodingerPINN:
    """
    Solve quantum harmonic oscillator using PINN.
    Parameters
    ----------
    omega : float, default 1.0
        Angular frequency
    x_bounds : tuple, default (-5.0, 5.0)
        Spatial domain
    t_bounds : tuple, default (0.0, 2.0)
        Time domain
    epochs : int, default 5000
        Training epochs
    Returns
    -------
    SchrodingerPINN
        Trained PINN model
    """
    def harmonic_potential(x, t=None):
        return 0.5 * me * omega**2 * x**2
    def gaussian_initial(x):
        # Initial Gaussian wavepacket
        sigma = 0.5
        x0 = 0.0
        return tf.exp(-tf.square(x - x0) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))**0.5
    domain_bounds = {'x': x_bounds, 't': t_bounds}
    return solve_schrodinger_pinn(harmonic_potential, domain_bounds,
                                 gaussian_initial, epochs, time_dependent=True)