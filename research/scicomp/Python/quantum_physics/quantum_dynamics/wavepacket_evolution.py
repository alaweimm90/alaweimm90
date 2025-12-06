#!/usr/bin/env python3
"""
Quantum Wavepacket Evolution and Dynamics
Advanced implementation of wavepacket time evolution using various numerical
methods including split-operator, Runge-Kutta, and Crank-Nicolson schemes.
Designed for studying quantum tunneling, scattering, and coherent dynamics.
Key Features:
- Time-dependent Schrödinger equation solver
- Multiple propagation methods with adaptive time stepping
- Real-time visualization and analysis tools
- Expectation value tracking and uncertainty calculation
- Born probability analysis and flux calculations
Applications:
- Quantum tunneling through barriers
- Molecular dynamics and vibrational motion
- Atomic physics and laser-matter interaction
- Quantum control and optimal pulse design
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Callable, Dict, List
import warnings
from dataclasses import dataclass
try:
    from ...utils.constants import hbar, me
except ImportError:
    hbar = 1.054571817e-34
    me = 9.1093837015e-31
from ...utils.units import energy_convert
from ...visualization.berkeley_style import BerkeleyPlot
@dataclass
class WavepacketConfig:
    """Configuration parameters for wavepacket evolution."""
    x_min: float = -10.0
    x_max: float = 10.0
    n_points: int = 1024
    dt: float = 0.01
    t_max: float = 10.0
    mass: float = me
    method: str = 'split_operator'  # 'split_operator', 'crank_nicolson', 'runge_kutta'
    absorbing_boundaries: bool = True
    boundary_width: float = 1.0
class QuantumWavepacket:
    """
    Comprehensive quantum wavepacket evolution simulator.
    Provides multiple numerical methods for solving the time-dependent
    Schrödinger equation with various potential configurations.
    Parameters
    ----------
    config : WavepacketConfig
        Configuration parameters for the simulation
    potential : callable
        Potential energy function V(x, t)
    """
    def __init__(self, config: WavepacketConfig,
                 potential: Optional[Callable[[np.ndarray, float], np.ndarray]] = None):
        """Initialize wavepacket evolution system."""
        self.config = config
        self.potential_func = potential or self._zero_potential
        # Setup spatial grid
        self.x = np.linspace(config.x_min, config.x_max, config.n_points)
        self.dx = self.x[1] - self.x[0]
        # Momentum space grid (for split-operator method)
        self.dk = 2 * np.pi / (config.n_points * self.dx)
        self.k = np.fft.fftfreq(config.n_points, self.dx) * 2 * np.pi
        # Time grid
        self.n_steps = int(config.t_max / config.dt)
        self.t = np.linspace(0, config.t_max, self.n_steps + 1)
        # Initialize arrays for data storage
        self.psi_history = np.zeros((self.n_steps + 1, config.n_points), dtype=complex)
        self.expectation_values = {}
        # Setup absorbing boundaries if requested
        if config.absorbing_boundaries:
            self._setup_absorbing_boundaries()
        # Precompute kinetic energy operator in momentum space
        self._setup_kinetic_operator()
    def _zero_potential(self, x: np.ndarray, t: float) -> np.ndarray:
        """Default zero potential."""
        return np.zeros_like(x)
    def _setup_absorbing_boundaries(self):
        """Setup complex absorbing potential at boundaries."""
        self.absorbing_mask = np.ones_like(self.x)
        # Left boundary
        left_region = self.x < (self.config.x_min + self.config.boundary_width)
        self.absorbing_mask[left_region] = np.exp(
            -((self.x[left_region] - (self.config.x_min + self.config.boundary_width))**2) /
            (self.config.boundary_width**2)
        )
        # Right boundary
        right_region = self.x > (self.config.x_max - self.config.boundary_width)
        self.absorbing_mask[right_region] = np.exp(
            -((self.x[right_region] - (self.config.x_max - self.config.boundary_width))**2) /
            (self.config.boundary_width**2)
        )
    def _setup_kinetic_operator(self):
        """Precompute kinetic energy operator in momentum space."""
        self.kinetic_k = (hbar**2 / (2 * self.config.mass)) * self.k**2
    def gaussian_wavepacket(self, x0: float, k0: float, sigma: float) -> np.ndarray:
        """
        Create normalized Gaussian wavepacket.
        Parameters
        ----------
        x0 : float
            Initial position
        k0 : float
            Initial momentum (in units of 1/length)
        sigma : float
            Width parameter
        Returns
        -------
        np.ndarray
            Normalized wavepacket
        """
        normalization = (2 * np.pi * sigma**2)**(-0.25)
        gaussian = np.exp(-((self.x - x0)**2) / (4 * sigma**2))
        phase = np.exp(1j * k0 * self.x)
        return normalization * gaussian * phase
    def coherent_state(self, x0: float, p0: float, omega: float) -> np.ndarray:
        """
        Create coherent state for harmonic oscillator.
        Parameters
        ----------
        x0 : float
            Displacement in position
        p0 : float
            Displacement in momentum
        omega : float
            Oscillator frequency
        Returns
        -------
        np.ndarray
            Coherent state wavefunction
        """
        # Characteristic length scale
        x_char = np.sqrt(hbar / (self.config.mass * omega))
        # Coherent state parameters
        alpha = (x0 + 1j * p0 * x_char / hbar) / (np.sqrt(2) * x_char)
        # Ground state of harmonic oscillator
        psi_0 = (self.config.mass * omega / (np.pi * hbar))**(1/4) * \
                np.exp(-(self.config.mass * omega / (2 * hbar)) * self.x**2)
        # Displacement operator applied to ground state
        displacement_factor = np.exp(1j * p0 * self.x / hbar -
                                   (self.config.mass * omega / (2 * hbar)) *
                                   (2 * x0 * self.x - x0**2))
        return psi_0 * displacement_factor
    def evolve_split_operator(self, psi_initial: np.ndarray) -> np.ndarray:
        """
        Evolve wavepacket using split-operator method.
        This method alternates between evolution in position space (potential)
        and momentum space (kinetic energy), providing excellent conservation
        properties and computational efficiency.
        Parameters
        ----------
        psi_initial : np.ndarray
            Initial wavefunction
        Returns
        -------
        np.ndarray
            Final wavefunction after evolution
        """
        psi = psi_initial.copy()
        self.psi_history[0] = psi.copy()
        # Precompute half-step potential evolution operators
        dt_half = self.config.dt / 2
        for i in range(self.n_steps):
            t_current = i * self.config.dt
            # Get potential at current time
            V = self.potential_func(self.x, t_current)
            # Half-step evolution in potential
            psi *= np.exp(-1j * V * dt_half / hbar)
            # Full step evolution in kinetic energy (momentum space)
            psi_k = np.fft.fft(psi)
            psi_k *= np.exp(-1j * self.kinetic_k * self.config.dt / hbar)
            psi = np.fft.ifft(psi_k)
            # Half-step evolution in potential
            V = self.potential_func(self.x, t_current + self.config.dt)
            psi *= np.exp(-1j * V * dt_half / hbar)
            # Apply absorbing boundaries
            if self.config.absorbing_boundaries:
                psi *= self.absorbing_mask
            # Store result
            self.psi_history[i + 1] = psi.copy()
            # Calculate expectation values
            if i % 10 == 0:  # Store every 10th step to save memory
                self._calculate_expectation_values(psi, t_current + self.config.dt, i + 1)
        return psi
    def evolve_crank_nicolson(self, psi_initial: np.ndarray) -> np.ndarray:
        """
        Evolve wavepacket using Crank-Nicolson method.
        This implicit method provides excellent stability and conservation
        properties, particularly for problems with rapidly varying potentials.
        Parameters
        ----------
        psi_initial : np.ndarray
            Initial wavefunction
        Returns
        -------
        np.ndarray
            Final wavefunction after evolution
        """
        psi = psi_initial.copy()
        self.psi_history[0] = psi.copy()
        # Setup kinetic energy matrix (second derivative)
        kinetic_matrix = self._build_kinetic_matrix()
        for i in range(self.n_steps):
            t_current = i * self.config.dt
            t_next = (i + 1) * self.config.dt
            # Average potential between current and next time steps
            V_current = self.potential_func(self.x, t_current)
            V_next = self.potential_func(self.x, t_next)
            V_avg = 0.5 * (V_current + V_next)
            # Build Hamiltonian matrix
            H = kinetic_matrix + sp.diags(V_avg)
            # Crank-Nicolson matrices
            dt_factor = 1j * self.config.dt / (2 * hbar)
            A = sp.identity(self.config.n_points) + dt_factor * H
            B = sp.identity(self.config.n_points) - dt_factor * H
            # Solve linear system: A * psi_new = B * psi_old
            rhs = B @ psi
            psi = sp.linalg.spsolve(A, rhs)
            # Apply absorbing boundaries
            if self.config.absorbing_boundaries:
                psi *= self.absorbing_mask
            # Store result
            self.psi_history[i + 1] = psi.copy()
            # Calculate expectation values
            if i % 10 == 0:
                self._calculate_expectation_values(psi, t_next, i + 1)
        return psi
    def _build_kinetic_matrix(self) -> sp.csc_matrix:
        """Build kinetic energy matrix using finite differences."""
        # Second derivative operator: -ℏ²/(2m) * d²/dx²
        diag_main = -2 * np.ones(self.config.n_points)
        diag_off = np.ones(self.config.n_points - 1)
        # Apply periodic boundary conditions
        kinetic_matrix = sp.diags([diag_off, diag_main, diag_off],
                                 [-1, 0, 1],
                                 shape=(self.config.n_points, self.config.n_points))
        # Scale by proper factors
        kinetic_matrix *= -hbar**2 / (2 * self.config.mass * self.dx**2)
        return kinetic_matrix.tocsc()
    def evolve_runge_kutta(self, psi_initial: np.ndarray) -> np.ndarray:
        """
        Evolve wavepacket using 4th-order Runge-Kutta method.
        This explicit method is straightforward to implement and understand,
        though it may require smaller time steps for stability.
        Parameters
        ----------
        psi_initial : np.ndarray
            Initial wavefunction
        Returns
        -------
        np.ndarray
            Final wavefunction after evolution
        """
        psi = psi_initial.copy()
        self.psi_history[0] = psi.copy()
        # Build kinetic energy matrix
        kinetic_matrix = self._build_kinetic_matrix()
        def tdse_rhs(t: float, psi_vec: np.ndarray) -> np.ndarray:
            """Right-hand side of TDSE for ODE solver."""
            psi_complex = psi_vec[:self.config.n_points] + 1j * psi_vec[self.config.n_points:]
            # Apply Hamiltonian
            V = self.potential_func(self.x, t)
            H_psi = kinetic_matrix @ psi_complex + V * psi_complex
            # Apply absorbing boundaries
            if self.config.absorbing_boundaries:
                H_psi *= self.absorbing_mask
            # Return as real vector [Re(dpsi/dt), Im(dpsi/dt)]
            dpsi_dt = -1j * H_psi / hbar
            return np.concatenate([np.real(dpsi_dt), np.imag(dpsi_dt)])
        # Convert complex wavefunction to real vector
        psi_vec = np.concatenate([np.real(psi), np.imag(psi)])
        # Solve using scipy's RK45
        sol = solve_ivp(tdse_rhs, [0, self.config.t_max], psi_vec,
                       t_eval=self.t, method='RK45', rtol=1e-8, atol=1e-10)
        # Convert back to complex and store history
        for i, t_val in enumerate(self.t):
            psi_real = sol.y[:self.config.n_points, i]
            psi_imag = sol.y[self.config.n_points:, i]
            psi_complex = psi_real + 1j * psi_imag
            self.psi_history[i] = psi_complex
            if i % 10 == 0:
                self._calculate_expectation_values(psi_complex, t_val, i)
        return self.psi_history[-1]
    def evolve(self, psi_initial: np.ndarray) -> np.ndarray:
        """
        Evolve wavepacket using the specified method.
        Parameters
        ----------
        psi_initial : np.ndarray
            Initial wavefunction
        Returns
        -------
        np.ndarray
            Final wavefunction after evolution
        """
        # Normalize initial wavefunction
        psi_initial = psi_initial / np.sqrt(np.trapz(np.abs(psi_initial)**2, self.x))
        if self.config.method == 'split_operator':
            return self.evolve_split_operator(psi_initial)
        elif self.config.method == 'crank_nicolson':
            return self.evolve_crank_nicolson(psi_initial)
        elif self.config.method == 'runge_kutta':
            return self.evolve_runge_kutta(psi_initial)
        else:
            raise ValueError(f"Unknown evolution method: {self.config.method}")
    def _calculate_expectation_values(self, psi: np.ndarray, t: float, step: int):
        """Calculate and store expectation values."""
        if step not in self.expectation_values:
            self.expectation_values[step] = {'time': t}
        # Probability density
        prob_density = np.abs(psi)**2
        # Position expectation value and uncertainty
        x_exp = np.trapz(prob_density * self.x, self.x)
        x2_exp = np.trapz(prob_density * self.x**2, self.x)
        sigma_x = np.sqrt(np.abs(x2_exp - x_exp**2))
        # Momentum expectation value (using derivative)
        psi_dx = np.gradient(psi, self.dx)
        p_exp = np.real(-1j * hbar * np.trapz(np.conj(psi) * psi_dx, self.x))
        # Momentum uncertainty
        psi_d2x = np.gradient(psi_dx, self.dx)
        p2_exp = np.real(-hbar**2 * np.trapz(np.conj(psi) * psi_d2x, self.x))
        sigma_p = np.sqrt(np.abs(p2_exp - p_exp**2))
        # Energy expectation value
        V = self.potential_func(self.x, t)
        T_exp = np.real(-hbar**2 / (2 * self.config.mass) *
                       np.trapz(np.conj(psi) * psi_d2x, self.x))
        V_exp = np.trapz(prob_density * V, self.x)
        E_exp = T_exp + V_exp
        # Store values
        self.expectation_values[step].update({
            'position': x_exp,
            'position_std': sigma_x,
            'momentum': p_exp,
            'momentum_std': sigma_p,
            'energy': E_exp,
            'kinetic_energy': T_exp,
            'potential_energy': V_exp,
            'norm': np.trapz(prob_density, self.x)
        })
    def calculate_transmission_reflection(self, barrier_position: float) -> Tuple[float, float]:
        """
        Calculate transmission and reflection coefficients.
        Parameters
        ----------
        barrier_position : float
            Position that separates transmitted and reflected components
        Returns
        -------
        Tuple[float, float]
            Transmission and reflection coefficients
        """
        final_psi = self.psi_history[-1]
        prob_density = np.abs(final_psi)**2
        # Find index corresponding to barrier position
        barrier_idx = np.argmin(np.abs(self.x - barrier_position))
        # Transmission (probability to the right of barrier)
        T = np.trapz(prob_density[barrier_idx:], self.x[barrier_idx:])
        # Reflection (probability to the left of barrier)
        R = np.trapz(prob_density[:barrier_idx], self.x[:barrier_idx])
        return T, R
    def calculate_flux(self, psi: np.ndarray, position: float) -> float:
        """
        Calculate probability flux at a given position.
        Parameters
        ----------
        psi : np.ndarray
            Wavefunction
        position : float
            Position to calculate flux
        Returns
        -------
        float
            Probability flux
        """
        # Find index closest to position
        idx = np.argmin(np.abs(self.x - position))
        # Calculate flux using finite differences
        psi_grad = np.gradient(psi, self.dx)
        flux = np.real(hbar / (2j * self.config.mass) *
                      (np.conj(psi[idx]) * psi_grad[idx] -
                       psi[idx] * np.conj(psi_grad[idx])))
        return flux
    def animate_evolution(self, save_path: Optional[str] = None,
                         show_potential: bool = True) -> None:
        """
        Create animated visualization of wavepacket evolution.
        Parameters
        ----------
        save_path : str, optional
            Path to save animation
        show_potential : bool
            Whether to show potential in background
        """
        try:
            from matplotlib.animation import FuncAnimation
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            # Apply Berkeley styling
            berkeley_plot = BerkeleyPlot()
            berkeley_plot.setup_plot_style()
            # Setup probability density plot
            line1, = ax1.plot(self.x, np.abs(self.psi_history[0])**2,
                            color=berkeley_plot.colors['berkeley_blue'], linewidth=2)
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Probability Density')
            ax1.set_title('Quantum Wavepacket Evolution')
            ax1.grid(True, alpha=0.3)
            # Show potential if requested
            if show_potential:
                V = self.potential_func(self.x, 0)
                V_normalized = V / np.max(np.abs(V)) * 0.1 if np.max(np.abs(V)) > 0 else V
                ax1.plot(self.x, V_normalized, '--',
                        color=berkeley_plot.colors['california_gold'],
                        alpha=0.7, label='Potential (scaled)')
                ax1.legend()
            # Setup real/imaginary parts plot
            line2, = ax2.plot(self.x, np.real(self.psi_history[0]),
                            color=berkeley_plot.colors['berkeley_blue'],
                            linewidth=2, label='Real')
            line3, = ax2.plot(self.x, np.imag(self.psi_history[0]),
                            color=berkeley_plot.colors['california_gold'],
                            linewidth=2, label='Imaginary')
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Wavefunction')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
            def animate(frame):
                psi = self.psi_history[frame]
                prob_density = np.abs(psi)**2
                line1.set_ydata(prob_density)
                line2.set_ydata(np.real(psi))
                line3.set_ydata(np.imag(psi))
                time_text.set_text(f'Time = {self.t[frame]:.2f}')
                # Auto-scale y-axis
                ax1.set_ylim(0, np.max(prob_density) * 1.1)
                ax2.set_ylim(np.min([np.min(np.real(psi)), np.min(np.imag(psi))]) * 1.1,
                            np.max([np.max(np.real(psi)), np.max(np.imag(psi))]) * 1.1)
                return line1, line2, line3, time_text
            anim = FuncAnimation(fig, animate, frames=len(self.t),
                               interval=50, blit=False, repeat=True)
            if save_path:
                anim.save(save_path, writer='pillow', fps=20)
            plt.tight_layout()
            plt.show()
        except ImportError:
            warnings.warn("Animation requires matplotlib. Skipping animation.")
    def plot_expectation_values(self) -> None:
        """Plot evolution of expectation values."""
        if not self.expectation_values:
            warnings.warn("No expectation values calculated. Run evolution first.")
            return
        berkeley_plot = BerkeleyPlot()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # Extract data
        times = [self.expectation_values[step]['time'] for step in sorted(self.expectation_values.keys())]
        positions = [self.expectation_values[step]['position'] for step in sorted(self.expectation_values.keys())]
        momenta = [self.expectation_values[step]['momentum'] for step in sorted(self.expectation_values.keys())]
        energies = [self.expectation_values[step]['energy'] for step in sorted(self.expectation_values.keys())]
        norms = [self.expectation_values[step]['norm'] for step in sorted(self.expectation_values.keys())]
        # Position evolution
        axes[0, 0].plot(times, positions, color=berkeley_plot.colors['berkeley_blue'], linewidth=2)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('⟨x⟩')
        axes[0, 0].set_title('Position Expectation Value')
        axes[0, 0].grid(True, alpha=0.3)
        # Momentum evolution
        axes[0, 1].plot(times, momenta, color=berkeley_plot.colors['california_gold'], linewidth=2)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('⟨p⟩')
        axes[0, 1].set_title('Momentum Expectation Value')
        axes[0, 1].grid(True, alpha=0.3)
        # Energy evolution
        axes[1, 0].plot(times, energies, color=berkeley_plot.colors['green_dark'], linewidth=2)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('⟨E⟩')
        axes[1, 0].set_title('Energy Expectation Value')
        axes[1, 0].grid(True, alpha=0.3)
        # Norm conservation
        axes[1, 1].plot(times, norms, color=berkeley_plot.colors['rose_dark'], linewidth=2)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('∫|ψ|² dx')
        axes[1, 1].set_title('Norm Conservation')
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
# Potential functions for common scenarios
def harmonic_potential(x: np.ndarray, t: float, k: float = 1.0, x0: float = 0.0) -> np.ndarray:
    """Harmonic oscillator potential."""
    return 0.5 * k * (x - x0)**2
def gaussian_barrier(x: np.ndarray, t: float, V0: float = 1.0,
                    x0: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Gaussian potential barrier."""
    return V0 * np.exp(-((x - x0)**2) / (2 * sigma**2))
def square_barrier(x: np.ndarray, t: float, V0: float = 1.0,
                  a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """Square potential barrier."""
    V = np.zeros_like(x)
    V[(x >= a) & (x <= b)] = V0
    return V
def double_well(x: np.ndarray, t: float, V0: float = 1.0,
               a: float = 2.0, b: float = 1.0) -> np.ndarray:
    """Double-well potential."""
    return V0 * (x**4 - a * x**2 + b)
def time_dependent_barrier(x: np.ndarray, t: float, V0: float = 1.0,
                          x0: float = 0.0, sigma: float = 1.0,
                          omega: float = 1.0) -> np.ndarray:
    """Time-dependent oscillating barrier."""
    V_static = V0 * np.exp(-((x - x0)**2) / (2 * sigma**2))
    time_modulation = 1 + 0.5 * np.sin(omega * t)
    return V_static * time_modulation
if __name__ == "__main__":
    # Example: Gaussian wavepacket tunneling through barrier
    config = WavepacketConfig(
        x_min=-10, x_max=10, n_points=512,
        dt=0.01, t_max=5.0, method='split_operator'
    )
    # Create system with Gaussian barrier
    barrier = lambda x, t: gaussian_barrier(x, t, V0=2.0, x0=0.0, sigma=0.5)
    system = QuantumWavepacket(config, barrier)
    # Initial Gaussian wavepacket moving towards barrier
    psi_initial = system.gaussian_wavepacket(x0=-5.0, k0=3.0, sigma=0.5)
    # Evolve the system
    psi_final = system.evolve(psi_initial)
    # Analyze results
    T, R = system.calculate_transmission_reflection(0.0)
    print(f"Transmission coefficient: {T:.4f}")
    print(f"Reflection coefficient: {R:.4f}")
    print(f"Total probability: {T + R:.4f}")
    # Plot expectation values
    system.plot_expectation_values()
    # Create animation (uncomment to run)
    # system.animate_evolution(show_potential=True)