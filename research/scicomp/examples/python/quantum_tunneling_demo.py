#!/usr/bin/env python3
"""
Quantum Tunneling Demonstration
Comprehensive example showcasing quantum tunneling analysis using the SciComp
framework. Demonstrates wavepacket scattering through potential barriers with
various barrier heights and widths, transmission coefficient calculations,
and Berkeley-styled visualizations.
Key Demonstrations:
- Wavepacket construction and time evolution
- Transmission and reflection coefficient calculation
- Resonant tunneling through double barriers
- WKB approximation comparison
- Interactive parameter exploration
Educational Objectives:
- Understand quantum mechanical tunneling
- Visualize wave-particle duality
- Explore parameter dependencies
- Compare classical vs quantum behavior
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
import warnings
warnings.filterwarnings('ignore')
# Import SciComp modules
from Python.quantum_physics.quantum_dynamics.quantum_tunneling import QuantumTunneling, TunnelingConfig
from Python.quantum_physics.quantum_dynamics.wavepacket_evolution import WavepacketEvolution, WavepacketConfig
from Python.visualization.berkeley_style import BerkeleyPlot
from Python.utils.constants import hbar, me, e
def main():
    """Main demonstration function."""
    print("=" * 70)
    print("SciComp")
    print("Quantum Tunneling Demonstration")
    print("=" * 70)
    print()
    # Run different tunneling scenarios
    demo_single_barrier_tunneling()
    demo_double_barrier_resonance()
    demo_energy_dependence()
    demo_barrier_width_dependence()
    demo_wkb_approximation()
    print("\nQuantum Tunneling demonstration completed!")
    print("All visualizations use Berkeley color scheme and styling.")
def demo_single_barrier_tunneling():
    """Demonstrate tunneling through a single rectangular barrier."""
    print("1. Single Barrier Tunneling Analysis")
    print("-" * 40)
    # Configuration
    config = TunnelingConfig(
        x_domain=(-20.0, 20.0),
        n_points=1000,
        barrier_type='rectangular',
        barrier_height=2.0,  # eV
        barrier_width=4.0,   # nm
        barrier_center=0.0,
        particle_energy=1.5,  # eV - below barrier
        mass=me,
        use_atomic_units=False
    )
    # Create tunneling system
    tunneling = QuantumTunneling(config)
    # Calculate transmission properties
    results = tunneling.calculate_transmission_coefficient()
    print(f"Barrier height: {config.barrier_height:.2f} eV")
    print(f"Barrier width: {config.barrier_width:.2f} nm")
    print(f"Particle energy: {config.particle_energy:.2f} eV")
    print(f"Transmission coefficient: {results['transmission']:.4f}")
    print(f"Reflection coefficient: {results['reflection']:.4f}")
    print(f"Conservation check: T + R = {results['transmission'] + results['reflection']:.6f}")
    # Visualize barrier and wavefunction
    tunneling.plot_barrier_and_wavefunction()
    # Animate wavepacket tunneling
    animate_wavepacket_tunneling(config)
    print()
def demo_double_barrier_resonance():
    """Demonstrate resonant tunneling through double barriers."""
    print("2. Double Barrier Resonant Tunneling")
    print("-" * 40)
    # Configuration for double barrier
    config = TunnelingConfig(
        x_domain=(-25.0, 25.0),
        n_points=1200,
        barrier_type='double_rectangular',
        barrier_height=3.0,    # eV
        barrier_width=2.0,     # nm
        barrier_separation=8.0, # nm
        particle_energy=2.5,   # eV
        mass=me
    )
    tunneling = QuantumTunneling(config)
    # Calculate transmission vs energy
    energies = np.linspace(0.5, 4.0, 100)
    transmissions = []
    for E in energies:
        config.particle_energy = E
        tunneling = QuantumTunneling(config)
        result = tunneling.calculate_transmission_coefficient()
        transmissions.append(result['transmission'])
    # Find resonance peaks
    transmissions = np.array(transmissions)
    peaks = find_resonance_peaks(energies, transmissions)
    print(f"Resonance energies found at: {peaks} eV")
    # Plot transmission vs energy
    plot_transmission_vs_energy(energies, transmissions, peaks)
    print()
def demo_energy_dependence():
    """Demonstrate how transmission depends on particle energy."""
    print("3. Energy Dependence of Tunneling")
    print("-" * 40)
    # Fixed barrier parameters
    barrier_height = 2.0  # eV
    barrier_width = 3.0   # nm
    # Energy range
    energies = np.linspace(0.1, 3.0, 200)
    transmissions_classical = []
    transmissions_quantum = []
    for E in energies:
        config = TunnelingConfig(
            barrier_height=barrier_height,
            barrier_width=barrier_width,
            particle_energy=E,
            mass=me
        )
        tunneling = QuantumTunneling(config)
        result = tunneling.calculate_transmission_coefficient()
        # Quantum result
        transmissions_quantum.append(result['transmission'])
        # Classical result (step function)
        if E > barrier_height:
            T_classical = 1.0
        else:
            T_classical = 0.0
        transmissions_classical.append(T_classical)
    # Plot comparison
    plot_classical_vs_quantum_transmission(
        energies, transmissions_classical, transmissions_quantum, barrier_height
    )
    print(f"Below barrier (E < {barrier_height} eV): Quantum allows tunneling, Classical forbids")
    print(f"Above barrier (E > {barrier_height} eV): Both allow transmission")
    print()
def demo_barrier_width_dependence():
    """Demonstrate exponential dependence on barrier width."""
    print("4. Barrier Width Dependence")
    print("-" * 40)
    # Fixed parameters
    barrier_height = 2.0  # eV
    particle_energy = 1.0  # eV (below barrier)
    # Barrier width range
    widths = np.linspace(1.0, 10.0, 50)
    transmissions = []
    for width in widths:
        config = TunnelingConfig(
            barrier_height=barrier_height,
            barrier_width=width,
            particle_energy=particle_energy,
            mass=me
        )
        tunneling = QuantumTunneling(config)
        result = tunneling.calculate_transmission_coefficient()
        transmissions.append(result['transmission'])
    transmissions = np.array(transmissions)
    # Fit exponential decay
    from scipy.optimize import curve_fit
    def exponential_decay(x, A, gamma):
        return A * np.exp(-gamma * x)
    # Fit to data
    popt, _ = curve_fit(exponential_decay, widths, transmissions)
    A_fit, gamma_fit = popt
    print(f"Transmission ∝ exp(-γd) with γ = {gamma_fit:.3f} nm⁻¹")
    # Plot width dependence
    plot_width_dependence(widths, transmissions, A_fit, gamma_fit)
    print()
def demo_wkb_approximation():
    """Compare exact quantum results with WKB approximation."""
    print("5. WKB Approximation Comparison")
    print("-" * 40)
    # Parameters
    barrier_height = 3.0  # eV
    barrier_width = 5.0   # nm
    energies = np.linspace(0.5, 2.8, 100)
    transmissions_exact = []
    transmissions_wkb = []
    for E in energies:
        config = TunnelingConfig(
            barrier_height=barrier_height,
            barrier_width=barrier_width,
            particle_energy=E,
            mass=me
        )
        tunneling = QuantumTunneling(config)
        # Exact quantum result
        result_exact = tunneling.calculate_transmission_coefficient()
        transmissions_exact.append(result_exact['transmission'])
        # WKB approximation
        result_wkb = tunneling.calculate_wkb_transmission()
        transmissions_wkb.append(result_wkb['transmission'])
    # Plot comparison
    plot_wkb_comparison(energies, transmissions_exact, transmissions_wkb)
    # Calculate agreement
    mse = np.mean((np.array(transmissions_exact) - np.array(transmissions_wkb))**2)
    print(f"Mean squared error between exact and WKB: {mse:.2e}")
    print("WKB approximation works best for thick barriers and low energies")
    print()
def animate_wavepacket_tunneling(tunneling_config):
    """Create animation of wavepacket tunneling through barrier."""
    print("Creating wavepacket tunneling animation...")
    # Wavepacket configuration
    wp_config = WavepacketConfig(
        x_domain=tunneling_config.x_domain,
        nx=tunneling_config.n_points,
        dt=0.01,
        total_time=30.0,
        initial_position=-10.0,
        initial_momentum=1.0,
        sigma=2.0,
        mass=tunneling_config.mass
    )
    # Create potential barrier
    x = np.linspace(wp_config.x_domain[0], wp_config.x_domain[1], wp_config.nx)
    V = create_barrier_potential(x, tunneling_config)
    # Set up wavepacket evolution
    wp_evolution = WavepacketEvolution(wp_config)
    wp_evolution.set_potential(V)
    # Run evolution
    results = wp_evolution.evolve()
    # Create animation
    create_tunneling_animation(results, V, tunneling_config)
def create_barrier_potential(x, config):
    """Create potential barrier based on configuration."""
    V = np.zeros_like(x)
    if config.barrier_type == 'rectangular':
        mask = np.abs(x - config.barrier_center) < config.barrier_width / 2
        V[mask] = config.barrier_height
    elif config.barrier_type == 'double_rectangular':
        # First barrier
        mask1 = np.abs(x - (-config.barrier_separation/2)) < config.barrier_width / 2
        # Second barrier
        mask2 = np.abs(x - (config.barrier_separation/2)) < config.barrier_width / 2
        V[mask1 | mask2] = config.barrier_height
    return V
def find_resonance_peaks(energies, transmissions, threshold=0.8):
    """Find resonance peaks in transmission spectrum."""
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(transmissions, height=threshold, distance=5)
    return energies[peaks]
def plot_transmission_vs_energy(energies, transmissions, peaks):
    """Plot transmission coefficient vs energy."""
    berkeley_plot = BerkeleyPlot()
    fig, ax = plt.subplots(figsize=(10, 6))
    # Main transmission curve
    ax.plot(energies, transmissions, linewidth=2,
           color=berkeley_plot.colors['berkeley_blue'],
           label='Transmission Coefficient')
    # Mark resonance peaks
    for peak in peaks:
        ax.axvline(peak, color=berkeley_plot.colors['california_gold'],
                  linestyle='--', alpha=0.7, label='Resonance' if peak == peaks[0] else "")
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Transmission Coefficient')
    ax.set_title('Double Barrier Resonant Tunneling')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()
def plot_classical_vs_quantum_transmission(energies, T_classical, T_quantum, barrier_height):
    """Plot classical vs quantum transmission."""
    berkeley_plot = BerkeleyPlot()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(energies, T_classical, '--', linewidth=3,
           color=berkeley_plot.colors['founders_rock'],
           label='Classical (Step Function)')
    ax.plot(energies, T_quantum, '-', linewidth=2,
           color=berkeley_plot.colors['berkeley_blue'],
           label='Quantum Mechanical')
    ax.axvline(barrier_height, color=berkeley_plot.colors['california_gold'],
              linestyle=':', linewidth=2, label=f'Barrier Height ({barrier_height} eV)')
    ax.set_xlabel('Particle Energy (eV)')
    ax.set_ylabel('Transmission Coefficient')
    ax.set_title('Classical vs Quantum Transmission')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.show()
def plot_width_dependence(widths, transmissions, A_fit, gamma_fit):
    """Plot transmission vs barrier width."""
    berkeley_plot = BerkeleyPlot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Linear scale
    ax1.plot(widths, transmissions, 'o', markersize=4,
            color=berkeley_plot.colors['berkeley_blue'],
            label='Quantum Calculation')
    # Fitted exponential
    widths_fit = np.linspace(widths[0], widths[-1], 100)
    T_fit = A_fit * np.exp(-gamma_fit * widths_fit)
    ax1.plot(widths_fit, T_fit, '--', linewidth=2,
            color=berkeley_plot.colors['california_gold'],
            label=f'Fit: T ∝ e^(-{gamma_fit:.2f}d)')
    ax1.set_xlabel('Barrier Width (nm)')
    ax1.set_ylabel('Transmission Coefficient')
    ax1.set_title('Exponential Decay with Width')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # Log scale
    ax2.semilogy(widths, transmissions, 'o', markersize=4,
                color=berkeley_plot.colors['berkeley_blue'],
                label='Quantum Calculation')
    ax2.semilogy(widths_fit, T_fit, '--', linewidth=2,
                color=berkeley_plot.colors['california_gold'],
                label=f'Exponential Fit')
    ax2.set_xlabel('Barrier Width (nm)')
    ax2.set_ylabel('Transmission Coefficient (log)')
    ax2.set_title('Exponential Decay (Log Scale)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.show()
def plot_wkb_comparison(energies, T_exact, T_wkb):
    """Plot exact vs WKB approximation."""
    berkeley_plot = BerkeleyPlot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Linear scale comparison
    ax1.plot(energies, T_exact, '-', linewidth=2,
            color=berkeley_plot.colors['berkeley_blue'],
            label='Exact Quantum')
    ax1.plot(energies, T_wkb, '--', linewidth=2,
            color=berkeley_plot.colors['california_gold'],
            label='WKB Approximation')
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Transmission Coefficient')
    ax1.set_title('Exact vs WKB Approximation')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1)
    # Log scale for better comparison
    ax2.semilogy(energies, T_exact, '-', linewidth=2,
                color=berkeley_plot.colors['berkeley_blue'],
                label='Exact Quantum')
    ax2.semilogy(energies, T_wkb, '--', linewidth=2,
                color=berkeley_plot.colors['california_gold'],
                label='WKB Approximation')
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('Transmission Coefficient (log)')
    ax2.set_title('Log Scale Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.show()
def create_tunneling_animation(evolution_results, potential, config):
    """Create animation of wavepacket tunneling."""
    print("Generating tunneling animation...")
    berkeley_plot = BerkeleyPlot()
    fig, ax = plt.subplots(figsize=(12, 8))
    x = evolution_results['x']
    t = evolution_results['t']
    psi = evolution_results['psi']
    # Normalize potential for display
    V_display = potential / np.max(potential) * 0.3
    # Animation function
    def animate(frame):
        ax.clear()
        # Current wavefunction
        psi_current = psi[:, frame]
        prob_density = np.abs(psi_current)**2
        # Plot probability density
        ax.fill_between(x, 0, prob_density, alpha=0.7,
                       color=berkeley_plot.colors['berkeley_blue'],
                       label='|ψ|²')
        # Plot real and imaginary parts
        ax.plot(x, np.real(psi_current), '--', linewidth=1.5,
               color=berkeley_plot.colors['california_gold'],
               label='Re[ψ]')
        ax.plot(x, np.imag(psi_current), ':', linewidth=1.5,
               color=berkeley_plot.colors['founders_rock'],
               label='Im[ψ]')
        # Plot potential barrier
        ax.fill_between(x, 0, V_display, alpha=0.5,
                       color='gray', label='Potential Barrier')
        ax.set_xlabel('Position (nm)')
        ax.set_ylabel('Wavefunction Amplitude')
        ax.set_title(f'Quantum Tunneling Animation (t = {t[frame]:.2f})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 0.5)
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(t),
                        interval=50, blit=False, repeat=True)
    plt.tight_layout()
    plt.show()
    print("Animation completed!")
if __name__ == "__main__":
    main()