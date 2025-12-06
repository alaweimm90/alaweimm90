#!/usr/bin/env python3
"""
Variational Quantum Eigensolver (VQE) Demo
Demonstrates the Variational Quantum Eigensolver algorithm for finding
ground states of quantum systems. Shows implementation with different
Hamiltonians, ansatz circuits, and optimization strategies.
Key Concepts:
- Variational principle: ⟨ψ(θ)|Ĥ|ψ(θ)⟩ ≥ E₀
- Parameterized quantum circuits (ansatz)
- Classical optimization in hybrid algorithms
- Quantum advantage for many-body problems
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import warnings
# Add the Python package to path (for development)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Python"))
from quantum_computing.algorithms.vqe import VQE, HardwareEfficientAnsatz
from visualization.berkeley_style import BerkeleyPlot, BERKELEY_BLUE, CALIFORNIA_GOLD
def main():
    """Run VQE demonstration."""
    print("⚛️  Variational Quantum Eigensolver (VQE) Demo")
    print("=" * 50)
    # Create output directory
    output_dir = Path("output/vqe_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Demo 1: Single qubit systems
    print("\n🎯 Demo 1: Single Qubit Systems")
    print("-" * 40)
    demo_single_qubit_vqe(output_dir)
    # Demo 2: Two qubit systems
    print("\n🔗 Demo 2: Two Qubit Systems")
    print("-" * 40)
    demo_two_qubit_vqe(output_dir)
    # Demo 3: Optimization comparison
    print("\n📊 Demo 3: Optimizer Comparison")
    print("-" * 40)
    demo_optimizer_comparison(output_dir)
    # Demo 4: Ansatz comparison
    print("\n🏗️  Demo 4: Ansatz Comparison")
    print("-" * 40)
    demo_ansatz_comparison(output_dir)
    print(f"\n✅ VQE demo completed! Results saved to: {output_dir}")
    print("🐻💙💛 Quantum computing at UC Berkeley")
def demo_single_qubit_vqe(output_dir: Path):
    """Demonstrate VQE on single qubit systems."""
    print("   Testing VQE on Pauli operators...")
    # Define single-qubit Hamiltonians
    hamiltonians = {
        "Pauli-Z": np.array([[1, 0], [0, -1]], dtype=complex),
        "Pauli-X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Pauli-Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Custom": np.array([[0.5, 0.2], [0.2, -0.5]], dtype=complex)
    }
    results = {}
    for name, hamiltonian in hamiltonians.items():
        print(f"   Running VQE for {name} Hamiltonian...")
        # Exact eigenvalues for comparison
        exact_energies = np.linalg.eigvals(hamiltonian)
        exact_ground_energy = np.min(np.real(exact_energies))
        # Create ansatz and VQE instance
        ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=2)
        vqe = VQE(hamiltonian, ansatz, optimizer='BFGS', seed=42)
        # Run optimization
        result = vqe.optimize()
        # Calculate error
        error = abs(result['optimal_energy'] - exact_ground_energy)
        results[name] = {
            'vqe_energy': result['optimal_energy'],
            'exact_energy': exact_ground_energy,
            'error': error,
            'success': result['success'],
            'iterations': result['num_iterations']
        }
        print(f"     Exact ground energy: {exact_ground_energy:.6f}")
        print(f"     VQE ground energy:   {result['optimal_energy']:.6f}")
        print(f"     Error:               {error:.2e}")
        print(f"     Converged:           {result['success']}")
    # Plot results
    plot_single_qubit_results(results, output_dir)
def demo_two_qubit_vqe(output_dir: Path):
    """Demonstrate VQE on two-qubit systems."""
    print("   Testing VQE on two-qubit Hamiltonians...")
    # Ising model: H = -J σᶻ₁σᶻ₂ - h(σᶻ₁ + σᶻ₂)
    def ising_hamiltonian(J: float, h: float) -> np.ndarray:
        """Create Ising model Hamiltonian."""
        # Pauli-Z matrices
        Z = np.array([[1, 0], [0, -1]])
        I = np.eye(2)
        # Two-qubit operators
        ZZ = np.kron(Z, Z)
        ZI = np.kron(Z, I)
        IZ = np.kron(I, Z)
        return -J * ZZ - h * (ZI + IZ)
    # Test different Ising parameters
    ising_params = [
        (1.0, 0.0),   # Ferromagnetic, no field
        (1.0, 0.5),   # Ferromagnetic with field
        (-1.0, 0.0),  # Antiferromagnetic, no field
        (0.5, 1.0),   # Weak coupling, strong field
    ]
    results = {}
    for J, h in ising_params:
        name = f"Ising(J={J}, h={h})"
        print(f"   Running VQE for {name}...")
        hamiltonian = ising_hamiltonian(J, h)
        exact_energies = np.linalg.eigvals(hamiltonian)
        exact_ground_energy = np.min(np.real(exact_energies))
        # Create ansatz with more layers for two qubits
        ansatz = HardwareEfficientAnsatz(num_qubits=2, num_layers=3,
                                       entanglement='linear')
        vqe = VQE(hamiltonian, ansatz, optimizer='COBYLA', seed=42)
        # Run optimization
        result = vqe.optimize()
        error = abs(result['optimal_energy'] - exact_ground_energy)
        results[name] = {
            'vqe_energy': result['optimal_energy'],
            'exact_energy': exact_ground_energy,
            'error': error,
            'success': result['success'],
            'iterations': result['num_iterations'],
            'J': J,
            'h': h
        }
        print(f"     Exact ground energy: {exact_ground_energy:.6f}")
        print(f"     VQE ground energy:   {result['optimal_energy']:.6f}")
        print(f"     Error:               {error:.2e}")
    # Plot results
    plot_two_qubit_results(results, output_dir)
def demo_optimizer_comparison(output_dir: Path):
    """Compare different classical optimizers."""
    print("   Comparing optimization algorithms...")
    # Test Hamiltonian (Ising model)
    hamiltonian = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.2, 0.0],
        [0.0, 0.2, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=complex)
    exact_ground_energy = np.min(np.real(np.linalg.eigvals(hamiltonian)))
    optimizers = ['COBYLA', 'BFGS', 'Powell', 'SLSQP']
    optimizer_results = {}
    for optimizer in optimizers:
        print(f"   Testing {optimizer} optimizer...")
        ansatz = HardwareEfficientAnsatz(num_qubits=2, num_layers=2)
        vqe = VQE(hamiltonian, ansatz, optimizer=optimizer, seed=42)
        result = vqe.optimize()
        # Analyze convergence
        convergence = vqe.analyze_convergence()
        optimizer_results[optimizer] = {
            'final_energy': result['optimal_energy'],
            'error': abs(result['optimal_energy'] - exact_ground_energy),
            'iterations': result['num_iterations'],
            'success': result['success'],
            'energy_history': convergence['energy_history'],
            'converged': convergence['converged']
        }
        print(f"     Final energy: {result['optimal_energy']:.6f}")
        print(f"     Error:        {optimizer_results[optimizer]['error']:.2e}")
        print(f"     Iterations:   {result['num_iterations']}")
    # Plot optimization comparison
    plot_optimizer_comparison(optimizer_results, exact_ground_energy, output_dir)
def demo_ansatz_comparison(output_dir: Path):
    """Compare different ansatz circuits."""
    print("   Comparing ansatz architectures...")
    # Test Hamiltonian
    hamiltonian = np.array([
        [1.0, 0.1, 0.1, 0.0],
        [0.1, -0.5, 0.2, 0.1],
        [0.1, 0.2, -0.5, 0.1],
        [0.0, 0.1, 0.1, 1.0]
    ], dtype=complex)
    exact_ground_energy = np.min(np.real(np.linalg.eigvals(hamiltonian)))
    # Different ansatz configurations
    ansatz_configs = [
        ("Linear-1", 1, 'linear'),
        ("Linear-2", 2, 'linear'),
        ("Linear-3", 3, 'linear'),
        ("Circular-2", 2, 'circular'),
    ]
    ansatz_results = {}
    for name, layers, entanglement in ansatz_configs:
        print(f"   Testing {name} ansatz...")
        ansatz = HardwareEfficientAnsatz(num_qubits=2, num_layers=layers,
                                       entanglement=entanglement)
        vqe = VQE(hamiltonian, ansatz, optimizer='BFGS', seed=42)
        result = vqe.optimize()
        ansatz_results[name] = {
            'final_energy': result['optimal_energy'],
            'error': abs(result['optimal_energy'] - exact_ground_energy),
            'iterations': result['num_iterations'],
            'num_parameters': ansatz.num_parameters,
            'success': result['success']
        }
        print(f"     Parameters: {ansatz.num_parameters}")
        print(f"     Final energy: {result['optimal_energy']:.6f}")
        print(f"     Error: {ansatz_results[name]['error']:.2e}")
    # Plot ansatz comparison
    plot_ansatz_comparison(ansatz_results, exact_ground_energy, output_dir)
def plot_single_qubit_results(results: Dict, output_dir: Path):
    """Plot single qubit VQE results."""
    berkeley_plot = BerkeleyPlot(figsize=(12, 5))
    fig, (ax1, ax2) = berkeley_plot.create_figure(1, 2)
    hamiltonians = list(results.keys())
    vqe_energies = [results[h]['vqe_energy'] for h in hamiltonians]
    exact_energies = [results[h]['exact_energy'] for h in hamiltonians]
    errors = [results[h]['error'] for h in hamiltonians]
    x = np.arange(len(hamiltonians))
    # Energy comparison
    width = 0.35
    ax1.bar(x - width/2, exact_energies, width, label='Exact',
           color=BERKELEY_BLUE, alpha=0.7)
    ax1.bar(x + width/2, vqe_energies, width, label='VQE',
           color=CALIFORNIA_GOLD, alpha=0.7)
    ax1.set_xlabel('Hamiltonian')
    ax1.set_ylabel('Ground State Energy')
    ax1.set_title('VQE vs Exact Results', fontweight='bold', color=BERKELEY_BLUE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(hamiltonians, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Error plot
    ax2.semilogy(x, errors, 'o-', color=BERKELEY_BLUE, linewidth=2, markersize=8)
    ax2.set_xlabel('Hamiltonian')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('VQE Accuracy', fontweight='bold', color=BERKELEY_BLUE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(hamiltonians, rotation=45)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    berkeley_plot.save_figure(output_dir / "single_qubit_results.png")
    plt.close()
def plot_two_qubit_results(results: Dict, output_dir: Path):
    """Plot two qubit VQE results."""
    berkeley_plot = BerkeleyPlot(figsize=(10, 6))
    fig, ax = berkeley_plot.create_figure()
    names = list(results.keys())
    errors = [results[name]['error'] for name in names]
    iterations = [results[name]['iterations'] for name in names]
    # Create scatter plot with error vs iterations
    colors = [BERKELEY_BLUE if 'J=1.0' in name else CALIFORNIA_GOLD for name in names]
    scatter = ax.scatter(iterations, errors, c=colors, s=100, alpha=0.7)
    # Add labels for each point
    for i, name in enumerate(names):
        ax.annotate(name.replace('Ising', ''),
                   (iterations[i], errors[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9)
    ax.set_xlabel('Optimization Iterations')
    ax.set_ylabel('Absolute Error')
    ax.set_yscale('log')
    ax.set_title('Two-Qubit Ising Model VQE', fontweight='bold', color=BERKELEY_BLUE)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    berkeley_plot.save_figure(output_dir / "two_qubit_results.png")
    plt.close()
def plot_optimizer_comparison(results: Dict, exact_energy: float, output_dir: Path):
    """Plot optimizer comparison."""
    berkeley_plot = BerkeleyPlot(figsize=(15, 5))
    fig, (ax1, ax2, ax3) = berkeley_plot.create_figure(1, 3)
    optimizers = list(results.keys())
    colors = [BERKELEY_BLUE, CALIFORNIA_GOLD, '#00553A', '#770747']
    # Plot convergence histories
    for i, optimizer in enumerate(optimizers):
        history = results[optimizer]['energy_history']
        ax1.plot(history, color=colors[i], linewidth=2, label=optimizer)
    ax1.axhline(y=exact_energy, color='red', linestyle='--',
               linewidth=2, label='Exact')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy')
    ax1.set_title('Convergence History', fontweight='bold', color=BERKELEY_BLUE)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Final errors
    errors = [results[opt]['error'] for opt in optimizers]
    ax2.bar(optimizers, errors, color=colors, alpha=0.7)
    ax2.set_ylabel('Final Absolute Error')
    ax2.set_title('Final Accuracy', fontweight='bold', color=BERKELEY_BLUE)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    # Iterations
    iterations = [results[opt]['iterations'] for opt in optimizers]
    ax3.bar(optimizers, iterations, color=colors, alpha=0.7)
    ax3.set_ylabel('Number of Iterations')
    ax3.set_title('Convergence Speed', fontweight='bold', color=BERKELEY_BLUE)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    berkeley_plot.save_figure(output_dir / "optimizer_comparison.png")
    plt.close()
def plot_ansatz_comparison(results: Dict, exact_energy: float, output_dir: Path):
    """Plot ansatz comparison."""
    berkeley_plot = BerkeleyPlot(figsize=(12, 5))
    fig, (ax1, ax2) = berkeley_plot.create_figure(1, 2)
    ansatzes = list(results.keys())
    errors = [results[ans]['error'] for ans in ansatzes]
    params = [results[ans]['num_parameters'] for ans in ansatzes]
    # Error vs parameters
    colors = [BERKELEY_BLUE if 'Linear' in ans else CALIFORNIA_GOLD for ans in ansatzes]
    ax1.scatter(params, errors, c=colors, s=100, alpha=0.7)
    for i, ans in enumerate(ansatzes):
        ax1.annotate(ans, (params[i], errors[i]),
                    xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel('Number of Parameters')
    ax1.set_ylabel('Absolute Error')
    ax1.set_yscale('log')
    ax1.set_title('Accuracy vs Complexity', fontweight='bold', color=BERKELEY_BLUE)
    ax1.grid(True, alpha=0.3)
    # Final energies comparison
    final_energies = [results[ans]['final_energy'] for ans in ansatzes]
    ax2.bar(ansatzes, final_energies, color=colors, alpha=0.7)
    ax2.axhline(y=exact_energy, color='red', linestyle='--',
               linewidth=2, label='Exact')
    ax2.set_ylabel('Final Energy')
    ax2.set_title('Final Energies', fontweight='bold', color=BERKELEY_BLUE)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    berkeley_plot.save_figure(output_dir / "ansatz_comparison.png")
    plt.close()
if __name__ == "__main__":
    main()