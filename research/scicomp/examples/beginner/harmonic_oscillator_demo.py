#!/usr/bin/env python3
"""
Quantum Harmonic Oscillator Demo
A beginner-friendly demonstration of quantum harmonic oscillator physics
using the SciComp package. Shows eigenstate calculations, time evolution,
and coherent state dynamics with Berkeley-styled visualizations.
Key Concepts Demonstrated:
- Energy quantization: E_n = ℏω(n + 1/2)
- Wavefunction shapes and probability densities
- Coherent states and classical-like motion
- Uncertainty relations and quantum fluctuations
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# Add the Python package to path (for development)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Python"))
from quantum_physics.quantum_dynamics.harmonic_oscillator import QuantumHarmonic
from visualization.berkeley_style import BerkeleyPlot, BERKELEY_BLUE, CALIFORNIA_GOLD
def main():
    """Run harmonic oscillator demonstration."""
    print("🔬 Quantum Harmonic Oscillator Demo")
    print("=" * 50)
    # Create output directory
    output_dir = Path("output/harmonic_oscillator")
    output_dir.mkdir(parents=True, exist_ok=True)
    # System parameters
    omega = 1.0  # Angular frequency (natural units)
    mass = 1.0   # Mass (natural units)
    # Initialize quantum harmonic oscillator
    print(f"📊 Initializing quantum harmonic oscillator...")
    print(f"   Angular frequency: ω = {omega}")
    print(f"   Mass: m = {mass}")
    qho = QuantumHarmonic(omega=omega, mass=mass, x_max=6.0, n_points=1000)
    # Characteristic scales
    print(f"   Characteristic length: x₀ = {qho.x0:.3e} m")
    print(f"   Zero-point energy: E₀ = {qho.E0:.3e} J")
    # Part 1: Energy eigenvalues and eigenstates
    print("\n🎯 Part 1: Energy Levels and Wavefunctions")
    print("-" * 40)
    demo_eigenstates(qho, output_dir)
    # Part 2: Coherent states
    print("\n🌟 Part 2: Coherent States")
    print("-" * 40)
    demo_coherent_states(qho, output_dir)
    # Part 3: Time evolution
    print("\n⏰ Part 3: Time Evolution")
    print("-" * 40)
    demo_time_evolution(qho, output_dir)
    # Part 4: Wigner functions (phase space)
    print("\n🎨 Part 4: Phase Space Representations")
    print("-" * 40)
    demo_wigner_functions(qho, output_dir)
    print(f"\n✅ Demo completed! Results saved to: {output_dir}")
    print("🐻💙💛 Crafted with precision at UC Berkeley")
def demo_eigenstates(qho: QuantumHarmonic, output_dir: Path):
    """Demonstrate energy eigenstates."""
    # Calculate first few eigenstates
    n_states = 5
    energies = []
    wavefunctions = []
    print(f"   Computing first {n_states} eigenstates...")
    for n in range(n_states):
        energy = qho.energy(n)
        psi_n = qho.eigenstate(n)
        energies.append(energy)
        wavefunctions.append(psi_n)
        print(f"   n={n}: E_{n} = {energy/qho.E0:.1f} E₀")
    # Create Berkeley-styled plot
    berkeley_plot = BerkeleyPlot(figsize=(12, 8))
    fig, (ax1, ax2) = berkeley_plot.create_figure(1, 2)
    # Plot wavefunctions
    colors = [BERKELEY_BLUE, CALIFORNIA_GOLD, '#00553A', '#770747', '#431170']
    for n, (psi_n, color) in enumerate(zip(wavefunctions, colors)):
        ax1.plot(qho.x / qho.x0, np.real(psi_n),
                color=color, linewidth=2, label=f'ψ_{n}(x)')
    ax1.set_xlabel('Position (x/x₀)')
    ax1.set_ylabel('Wavefunction')
    ax1.set_title('Harmonic Oscillator Eigenstates', fontweight='bold', color=BERKELEY_BLUE)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Plot probability densities
    for n, (psi_n, color) in enumerate(zip(wavefunctions, colors)):
        prob_density = np.abs(psi_n)**2
        ax2.plot(qho.x / qho.x0, prob_density,
                color=color, linewidth=2, label=f'|ψ_{n}(x)|²')
        ax2.fill_between(qho.x / qho.x0, prob_density, alpha=0.2, color=color)
    # Add classical turning points
    for n in range(n_states):
        # Classical turning points: E = (1/2)mω²x²
        x_classical = np.sqrt(2 * energies[n] / (mass * omega**2)) / qho.x0
        ax2.axvline(x_classical, color=colors[n], linestyle='--', alpha=0.5)
        ax2.axvline(-x_classical, color=colors[n], linestyle='--', alpha=0.5)
    ax2.set_xlabel('Position (x/x₀)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Probability Densities', fontweight='bold', color=BERKELEY_BLUE)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    berkeley_plot.save_figure(output_dir / "eigenstates.png")
    plt.close()
def demo_coherent_states(qho: QuantumHarmonic, output_dir: Path):
    """Demonstrate coherent states."""
    # Generate coherent states with different amplitudes
    alphas = [0.5, 1.0, 2.0]
    print(f"   Computing coherent states for α = {alphas}")
    berkeley_plot = BerkeleyPlot(figsize=(14, 5))
    fig, axes = berkeley_plot.create_figure(1, 3)
    for i, alpha in enumerate(alphas):
        # Generate coherent state
        psi_coherent = qho.coherent_state(alpha)
        # Plot wavefunction
        axes[i].plot(qho.x / qho.x0, np.real(psi_coherent),
                    color=BERKELEY_BLUE, linewidth=2, label='Re[ψ(x)]')
        axes[i].plot(qho.x / qho.x0, np.imag(psi_coherent),
                    color=CALIFORNIA_GOLD, linewidth=2, linestyle='--', label='Im[ψ(x)]')
        axes[i].fill_between(qho.x / qho.x0, np.abs(psi_coherent)**2,
                           alpha=0.3, color='green', label='|ψ(x)|²')
        # Calculate expectation values
        x_exp = qho.expectation_value(psi_coherent, 'x')
        p_exp = qho.expectation_value(psi_coherent, 'p')
        axes[i].set_title(f'Coherent State α={alpha}\n⟨x⟩={np.real(x_exp)/qho.x0:.2f}x₀',
                         fontweight='bold', color=BERKELEY_BLUE)
        axes[i].set_xlabel('Position (x/x₀)')
        axes[i].set_ylabel('Wavefunction')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        print(f"   α={alpha}: ⟨x⟩ = {np.real(x_exp)/qho.x0:.2f} x₀, ⟨p⟩ = {np.real(p_exp):.2e}")
    plt.tight_layout()
    berkeley_plot.save_figure(output_dir / "coherent_states.png")
    plt.close()
def demo_time_evolution(qho: QuantumHarmonic, output_dir: Path):
    """Demonstrate time evolution."""
    print("   Computing time evolution of Gaussian wavepacket...")
    # Create initial Gaussian wavepacket (coherent state)
    alpha = 1.5
    psi_initial = qho.coherent_state(alpha)
    # Time evolution parameters
    t_max = 2 * np.pi / omega  # One classical period
    n_times = 20
    times = np.linspace(0, t_max, n_times)
    # Evolve wavefunction
    psi_trajectory = qho.time_evolution(psi_initial, times)
    # Calculate expectation values over time
    x_trajectory = []
    p_trajectory = []
    for psi_t in psi_trajectory:
        x_exp = np.real(qho.expectation_value(psi_t, 'x'))
        p_exp = np.real(qho.expectation_value(psi_t, 'p'))
        x_trajectory.append(x_exp)
        p_trajectory.append(p_exp)
    x_trajectory = np.array(x_trajectory)
    p_trajectory = np.array(p_trajectory)
    # Create phase space plot
    berkeley_plot = BerkeleyPlot(figsize=(12, 5))
    fig, (ax1, ax2) = berkeley_plot.create_figure(1, 2)
    # Plot expectation values vs time
    ax1.plot(times * omega / (2*np.pi), x_trajectory / qho.x0,
            color=BERKELEY_BLUE, linewidth=2, marker='o', label='⟨x(t)⟩')
    ax1.set_xlabel('Time (periods)')
    ax1.set_ylabel('Position (x₀)')
    ax1.set_title('Position vs Time', fontweight='bold', color=BERKELEY_BLUE)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # Phase space trajectory
    ax2.plot(x_trajectory / qho.x0, p_trajectory,
            color=CALIFORNIA_GOLD, linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Position ⟨x⟩ (x₀)')
    ax2.set_ylabel('Momentum ⟨p⟩')
    ax2.set_title('Phase Space Trajectory', fontweight='bold', color=BERKELEY_BLUE)
    ax2.grid(True, alpha=0.3)
    # Add classical ellipse for comparison
    theta = np.linspace(0, 2*np.pi, 100)
    x_classical = np.abs(alpha) * qho.x0 * np.cos(theta) / qho.x0
    p_classical = np.abs(alpha) * np.sqrt(mass * omega * qho.E0) * np.sin(theta)
    ax2.plot(x_classical, p_classical, '--', color='gray', alpha=0.7, label='Classical')
    ax2.legend()
    plt.tight_layout()
    berkeley_plot.save_figure(output_dir / "time_evolution.png")
    plt.close()
    print(f"   Classical period: T = {t_max:.3f} s")
    print(f"   Quantum state oscillates with same period (coherent state property)")
def demo_wigner_functions(qho: QuantumHarmonic, output_dir: Path):
    """Demonstrate Wigner function phase space representations."""
    print("   Computing Wigner functions for different states...")
    # Calculate Wigner functions for different states
    states = [
        ("Ground State", qho.eigenstate(0)),
        ("First Excited", qho.eigenstate(1)),
        ("Coherent State", qho.coherent_state(1.5))
    ]
    berkeley_plot = BerkeleyPlot(figsize=(15, 5))
    fig, axes = berkeley_plot.create_figure(1, 3)
    for i, (name, psi) in enumerate(states):
        # Calculate Wigner function
        X, P, W = qho.wigner_function(psi, n_points=(50, 50))
        # Plot Wigner function
        im = axes[i].contourf(X / qho.x0, P, W, levels=20, cmap='RdBu_r')
        axes[i].set_xlabel('Position (x₀)')
        axes[i].set_ylabel('Momentum')
        axes[i].set_title(f'Wigner Function\n{name}',
                         fontweight='bold', color=BERKELEY_BLUE)
        # Add colorbar
        plt.colorbar(im, ax=axes[i])
    plt.tight_layout()
    berkeley_plot.save_figure(output_dir / "wigner_functions.png")
    plt.close()
    print("   Wigner functions show phase space structure:")
    print("   - Ground state: Gaussian blob (minimum uncertainty)")
    print("   - Excited states: Multiple lobes (quantum interference)")
    print("   - Coherent states: Displaced Gaussian (quasi-classical)")
if __name__ == "__main__":
    main()