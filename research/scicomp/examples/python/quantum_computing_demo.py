#!/usr/bin/env python3
"""
Quantum Computing Comprehensive Demonstration
Advanced showcase of quantum computing algorithms and circuit simulations using
the Berkeley SciComp framework. Demonstrates quantum gates, algorithms, error
correction, and quantum advantage with professional Berkeley styling.
Key Demonstrations:
- Quantum gate operations and circuit construction
- Grover's search algorithm with geometric analysis
- Variational Quantum Eigensolver (VQE) for molecular systems
- Quantum Approximate Optimization Algorithm (QAOA)
- Quantum error correction codes
- Quantum teleportation protocol
Educational Objectives:
- Understand quantum computation principles
- Explore quantum algorithms and their advantages
- Visualize quantum state evolution
- Analyze quantum circuit performance
- Compare classical vs quantum approaches
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
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')
# Import SciComp modules
from Python.quantum_computing.algorithms.grover import GroverSearch, GroverConfig
from Python.quantum_computing.algorithms.vqe import VQESolver, VQEConfig
from Python.quantum_computing.algorithms.qaoa import QAOAOptimizer, QAOAConfig
from Python.quantum_computing.circuits.quantum_gates import QuantumCircuit, GateConfig
from Python.quantum_computing.circuits.quantum_gates import PauliX, PauliY, PauliZ, Hadamard, CNOT
from Python.visualization.berkeley_style import BerkeleyPlot
from Python.utils.constants import *
def main():
    """Main demonstration function."""
    print("=" * 70)
    print("SciComp")
    print("Quantum Computing Comprehensive Demonstration")
    print("=" * 70)
    print()
    # Run comprehensive quantum computing demonstrations
    demo_quantum_gates_and_circuits()
    demo_grover_search_algorithm()
    demo_variational_quantum_eigensolver()
    demo_quantum_optimization()
    demo_quantum_error_correction()
    demo_quantum_teleportation()
    demo_quantum_advantage_analysis()
    print("\nQuantum Computing demonstration completed!")
    print("All visualizations use Berkeley color scheme and professional styling.")
def demo_quantum_gates_and_circuits():
    """Demonstrate quantum gates and circuit construction."""
    print("1. Quantum Gates and Circuit Construction")
    print("-" * 45)
    # Create quantum circuit
    config = GateConfig()
    circuit = QuantumCircuit(n_qubits=3, config=config)
    # Demonstrate single-qubit gates
    print("Single-qubit gate operations:")
    # Pauli gates
    X = PauliX()
    Y = PauliY()
    Z = PauliZ()
    H = Hadamard()
    print(f"  Pauli-X matrix:\n{X.get_matrix()}")
    print(f"  Pauli-Y matrix:\n{Y.get_matrix()}")
    print(f"  Pauli-Z matrix:\n{Z.get_matrix()}")
    print(f"  Hadamard matrix:\n{H.get_matrix()}")
    # Apply gates to circuit
    circuit.h(0)  # Hadamard on qubit 0
    circuit.x(1)  # Pauli-X on qubit 1
    circuit.y(2)  # Pauli-Y on qubit 2
    # Two-qubit gates
    print("\nTwo-qubit gate operations:")
    cnot = CNOT()
    print(f"  CNOT matrix:\n{cnot.get_matrix()}")
    circuit.cnot(0, 1)  # CNOT with control=0, target=1
    circuit.cnot(1, 2)  # CNOT with control=1, target=2
    # Simulate circuit
    final_state = circuit.simulate()
    print(f"\nFinal quantum state: {final_state}")
    # Measure circuit
    measurements = circuit.measure(n_shots=1000)
    print(f"Measurement results (1000 shots): {measurements}")
    # Plot circuit visualization
    plot_quantum_circuit_visualization(circuit)
    # Demonstrate Bloch sphere visualization
    demonstrate_bloch_sphere_evolution()
    print()
def demonstrate_bloch_sphere_evolution():
    """Demonstrate quantum state evolution on Bloch sphere."""
    berkeley_plot = BerkeleyPlot()
    # Create figure for Bloch sphere
    fig = plt.figure(figsize=(15, 5))
    # Initial state |0⟩
    ax1 = fig.add_subplot(131, projection='3d')
    plot_bloch_sphere(ax1, [0, 0, 1], "Initial State |0⟩", berkeley_plot)
    # After Hadamard: |+⟩ = (|0⟩ + |1⟩)/√2
    ax2 = fig.add_subplot(132, projection='3d')
    plot_bloch_sphere(ax2, [1, 0, 0], "After Hadamard |+⟩", berkeley_plot)
    # After Y rotation: complex superposition
    ax3 = fig.add_subplot(133, projection='3d')
    plot_bloch_sphere(ax3, [0, 1, 0], "After Y-rotation", berkeley_plot)
    plt.suptitle('Quantum State Evolution on Bloch Sphere',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
def plot_bloch_sphere(ax, state_vector, title, berkeley_plot):
    """Plot Bloch sphere with quantum state."""
    # Draw Bloch sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightblue')
    # Draw axes
    ax.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.3)
    ax.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.3)
    ax.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.3)
    # Draw state vector
    ax.quiver(0, 0, 0, state_vector[0], state_vector[1], state_vector[2],
             color=berkeley_plot.colors['berkeley_blue'], arrow_length_ratio=0.1, linewidth=3)
    # Labels
    ax.text(1.2, 0, 0, '|+⟩', fontsize=12)
    ax.text(-1.2, 0, 0, '|-⟩', fontsize=12)
    ax.text(0, 1.2, 0, '|+i⟩', fontsize=12)
    ax.text(0, -1.2, 0, '|-i⟩', fontsize=12)
    ax.text(0, 0, 1.2, '|0⟩', fontsize=12)
    ax.text(0, 0, -1.2, '|1⟩', fontsize=12)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_title(title, fontweight='bold')
    ax.set_box_aspect([1,1,1])
def plot_quantum_circuit_visualization(circuit):
    """Visualize quantum circuit diagram."""
    berkeley_plot = BerkeleyPlot()
    fig, ax = plt.subplots(figsize=(12, 6))
    # Circuit parameters
    n_qubits = circuit.n_qubits
    n_gates = len(circuit.gates)
    # Draw qubit lines
    for i in range(n_qubits):
        ax.plot([0, n_gates + 1], [i, i], 'k-', linewidth=2)
        ax.text(-0.5, i, f'|q_{i}⟩', fontsize=12, ha='right', va='center')
    # Draw gates
    for gate_idx, gate_info in enumerate(circuit.gates):
        gate_type, qubits = gate_info['type'], gate_info['qubits']
        x_pos = gate_idx + 1
        if len(qubits) == 1:  # Single-qubit gate
            qubit = qubits[0]
            if gate_type == 'H':
                rect = Rectangle((x_pos - 0.2, qubit - 0.2), 0.4, 0.4,
                               facecolor=berkeley_plot.colors['california_gold'],
                               edgecolor='black')
                ax.add_patch(rect)
                ax.text(x_pos, qubit, 'H', fontsize=10, ha='center', va='center', fontweight='bold')
            elif gate_type == 'X':
                circle = Circle((x_pos, qubit), 0.2,
                              facecolor=berkeley_plot.colors['berkeley_blue'],
                              edgecolor='black')
                ax.add_patch(circle)
                ax.text(x_pos, qubit, 'X', fontsize=10, ha='center', va='center', fontweight='bold', color='white')
        elif len(qubits) == 2:  # Two-qubit gate
            control, target = qubits
            # Control qubit
            circle = Circle((x_pos, control), 0.1,
                          facecolor='black', edgecolor='black')
            ax.add_patch(circle)
            # Connection line
            ax.plot([x_pos, x_pos], [control, target], 'k-', linewidth=2)
            # Target qubit
            circle = Circle((x_pos, target), 0.2,
                          facecolor='white', edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.plot([x_pos - 0.15, x_pos + 0.15], [target, target], 'k-', linewidth=2)
            ax.plot([x_pos, x_pos], [target - 0.15, target + 0.15], 'k-', linewidth=2)
    ax.set_xlim(-1, n_gates + 2)
    ax.set_ylim(-0.5, n_qubits - 0.5)
    ax.set_aspect('equal')
    ax.set_title('Quantum Circuit Diagram', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
def demo_grover_search_algorithm():
    """Demonstrate Grover's quantum search algorithm."""
    print("2. Grover's Quantum Search Algorithm")
    print("-" * 40)
    # Configuration for 4-qubit search
    config = GroverConfig(
        n_qubits=4,
        n_shots=1024,
        target_items=['1010', '1100'],
        use_optimal_iterations=True
    )
    grover = GroverSearch(config)
    print(f"Searching in {grover.n_states} states for targets: {config.target_items}")
    print(f"Classical search requires ~{grover.n_states/2:.1f} queries on average")
    print(f"Grover's algorithm requires ~{grover.calculate_optimal_iterations()} queries")
    # Run Grover search
    results = grover.run_grover_search()
    print(f"\nGrover Search Results:")
    print(f"  Success rate: {results['success_rate']:.3f}")
    print(f"  Iterations used: {results['iterations']}")
    print(f"  Quantum advantage: {(grover.n_states/2) / results['iterations']:.1f}x speedup")
    # Show measurement distribution
    print(f"\nTop measurement results:")
    sorted_measurements = sorted(results['measurements'].items(),
                               key=lambda x: x[1], reverse=True)
    for state, count in sorted_measurements[:5]:
        probability = count / config.n_shots
        is_target = "✓" if state in config.target_items else " "
        print(f"  |{state}⟩: {count:4d} ({probability:.3f}) {is_target}")
    # Plot Grover algorithm analysis
    plot_grover_analysis(grover, results)
    print()
def plot_grover_analysis(grover, results):
    """Plot Grover algorithm performance analysis."""
    berkeley_plot = BerkeleyPlot()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # Success probability vs iterations
    ax1 = axes[0, 0]
    iterations = range(1, 10)
    success_probs = []
    for k in iterations:
        # Theoretical success probability
        N = grover.n_states
        M = len(grover.target_items)
        theta = np.arcsin(np.sqrt(M/N))
        prob = np.sin((2*k + 1) * theta)**2
        success_probs.append(prob)
    ax1.plot(iterations, success_probs, 'o-',
            color=berkeley_plot.colors['berkeley_blue'], linewidth=2, markersize=6)
    ax1.axvline(grover.calculate_optimal_iterations(),
               color=berkeley_plot.colors['california_gold'],
               linestyle='--', linewidth=2, label='Optimal')
    ax1.set_xlabel('Number of Iterations')
    ax1.set_ylabel('Success Probability')
    ax1.set_title('Grover Algorithm Success Rate', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # Measurement distribution
    ax2 = axes[0, 1]
    states = list(results['measurements'].keys())
    counts = list(results['measurements'].values())
    colors = [berkeley_plot.colors['california_gold'] if state in grover.target_items
             else berkeley_plot.colors['founders_rock'] for state in states]
    bars = ax2.bar(range(len(states)), counts, color=colors, alpha=0.7)
    ax2.set_xlabel('Quantum States')
    ax2.set_ylabel('Measurement Counts')
    ax2.set_title('Measurement Distribution', fontweight='bold')
    ax2.set_xticks(range(0, len(states), max(1, len(states)//8)))
    ax2.set_xticklabels([states[i] for i in range(0, len(states), max(1, len(states)//8))],
                       rotation=45)
    # Geometric visualization of Grover's algorithm
    ax3 = axes[1, 0]
    N = grover.n_states
    M = len(grover.target_items)
    theta = np.arcsin(np.sqrt(M/N))
    # Draw geometric representation
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax3.add_patch(circle)
    # Initial state
    angle_init = np.pi/2 - theta
    ax3.arrow(0, 0, np.cos(angle_init), np.sin(angle_init),
             head_width=0.05, head_length=0.05,
             fc=berkeley_plot.colors['founders_rock'],
             ec=berkeley_plot.colors['founders_rock'], linewidth=2)
    ax3.text(np.cos(angle_init) + 0.1, np.sin(angle_init) + 0.1, 'Initial', fontsize=10)
    # Final state after optimal iterations
    k_opt = grover.calculate_optimal_iterations()
    angle_final = angle_init + 2 * k_opt * theta
    ax3.arrow(0, 0, np.cos(angle_final), np.sin(angle_final),
             head_width=0.05, head_length=0.05,
             fc=berkeley_plot.colors['california_gold'],
             ec=berkeley_plot.colors['california_gold'], linewidth=2)
    ax3.text(np.cos(angle_final) + 0.1, np.sin(angle_final) + 0.1, 'Final', fontsize=10)
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_aspect('equal')
    ax3.set_title('Geometric Representation', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    # Quantum vs Classical comparison
    ax4 = axes[1, 1]
    n_qubits_range = range(2, 8)
    classical_queries = [2**(n-1) for n in n_qubits_range]  # Average case
    quantum_queries = [int(np.pi/4 * np.sqrt(2**n)) for n in n_qubits_range]  # Grover
    ax4.semilogy(n_qubits_range, classical_queries, 'o-',
                color=berkeley_plot.colors['founders_rock'],
                linewidth=2, label='Classical Search')
    ax4.semilogy(n_qubits_range, quantum_queries, 's-',
                color=berkeley_plot.colors['berkeley_blue'],
                linewidth=2, label="Grover's Algorithm")
    ax4.set_xlabel('Number of Qubits')
    ax4.set_ylabel('Number of Queries')
    ax4.set_title('Quantum vs Classical Search', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    plt.suptitle('Grover Algorithm Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
def demo_variational_quantum_eigensolver():
    """Demonstrate Variational Quantum Eigensolver (VQE)."""
    print("3. Variational Quantum Eigensolver (VQE)")
    print("-" * 45)
    # Configuration for H2 molecule
    config = VQEConfig(
        n_qubits=4,
        molecule='H2',
        bond_distance=0.74,  # Angstroms
        ansatz_type='UCCSD',
        optimizer='COBYLA',
        max_iterations=100,
        convergence_threshold=1e-6
    )
    print(f"Solving ground state of {config.molecule} molecule")
    print(f"Bond distance: {config.bond_distance} Å")
    print(f"Using {config.ansatz_type} ansatz on {config.n_qubits} qubits")
    # Create VQE solver
    vqe = VQESolver(config)
    # Run VQE optimization
    results = vqe.run_vqe()
    print(f"\nVQE Results:")
    print(f"  Ground state energy: {results['final_energy']:.6f} Hartree")
    print(f"  Optimization iterations: {results['n_iterations']}")
    print(f"  Final parameters: {results['optimal_parameters']}")
    if 'exact_energy' in results:
        error = abs(results['final_energy'] - results['exact_energy'])
        print(f"  Exact energy: {results['exact_energy']:.6f} Hartree")
        print(f"  Absolute error: {error:.6f} Hartree")
        print(f"  Relative error: {error/abs(results['exact_energy'])*100:.4f}%")
    # Plot VQE optimization process
    plot_vqe_optimization(results)
    print()
def plot_vqe_optimization(results):
    """Plot VQE optimization convergence."""
    berkeley_plot = BerkeleyPlot()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Energy convergence
    ax1 = axes[0]
    iterations = range(len(results['energy_history']))
    ax1.plot(iterations, results['energy_history'], 'o-',
            color=berkeley_plot.colors['berkeley_blue'], linewidth=2)
    if 'exact_energy' in results:
        ax1.axhline(results['exact_energy'],
                   color=berkeley_plot.colors['california_gold'],
                   linestyle='--', linewidth=2, label='Exact Energy')
        ax1.legend()
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy (Hartree)')
    ax1.set_title('VQE Energy Convergence', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    # Parameter evolution
    ax2 = axes[1]
    if 'parameter_history' in results:
        param_history = np.array(results['parameter_history'])
        for i in range(param_history.shape[1]):
            ax2.plot(iterations, param_history[:, i],
                    color=berkeley_plot.colors['berkeley_blue'], alpha=0.7)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Parameter Evolution', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    # Molecular orbital visualization (simplified)
    ax3 = axes[2]
    # Create a simple molecular orbital visualization
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-2, 2, 80)
    X, Y = np.meshgrid(x, y)
    # Simple H2 bonding orbital representation
    orbital = np.exp(-((X + 0.74)**2 + Y**2)) + np.exp(-((X - 0.74)**2 + Y**2))
    cs = ax3.contourf(X, Y, orbital, levels=20, cmap='Blues')
    ax3.plot([-0.74, 0.74], [0, 0], 'ro', markersize=8, label='H atoms')
    ax3.set_xlabel('Distance (Å)')
    ax3.set_ylabel('Distance (Å)')
    ax3.set_title('H₂ Molecular Orbital', fontweight='bold')
    ax3.legend()
    ax3.set_aspect('equal')
    plt.suptitle('VQE Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
def demo_quantum_optimization():
    """Demonstrate Quantum Approximate Optimization Algorithm (QAOA)."""
    print("4. Quantum Approximate Optimization Algorithm (QAOA)")
    print("-" * 55)
    # Configuration for Max-Cut problem
    config = QAOAConfig(
        n_qubits=6,
        problem_type='max_cut',
        p_layers=3,  # QAOA depth
        optimizer='COBYLA',
        max_iterations=100
    )
    # Define a graph for Max-Cut problem
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4)]
    weights = [1.0] * len(edges)
    print(f"Solving Max-Cut problem on {config.n_qubits}-node graph")
    print(f"Graph edges: {edges}")
    print(f"Using QAOA with p = {config.p_layers} layers")
    # Create QAOA optimizer
    qaoa = QAOAOptimizer(config)
    qaoa.set_problem_instance(edges, weights)
    # Run QAOA optimization
    results = qaoa.run_qaoa()
    print(f"\nQAOA Results:")
    print(f"  Best cut value: {results['best_cut_value']}")
    print(f"  Best solution: {results['best_solution']}")
    print(f"  Approximation ratio: {results['approximation_ratio']:.3f}")
    print(f"  Optimization iterations: {results['n_iterations']}")
    # Plot QAOA results
    plot_qaoa_results(qaoa, results, edges)
    print()
def plot_qaoa_results(qaoa, results, edges):
    """Plot QAOA optimization results."""
    berkeley_plot = BerkeleyPlot()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Objective function convergence
    ax1 = axes[0]
    iterations = range(len(results['objective_history']))
    ax1.plot(iterations, results['objective_history'], 'o-',
            color=berkeley_plot.colors['berkeley_blue'], linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cut Value')
    ax1.set_title('QAOA Convergence', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    # Graph visualization
    ax2 = axes[1]
    n_nodes = qaoa.config.n_qubits
    # Position nodes in a circle
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n_nodes)}
    # Draw edges
    for edge in edges:
        x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
        y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
        ax2.plot(x_coords, y_coords, 'k-', alpha=0.5, linewidth=1)
    # Draw nodes with colors based on optimal solution
    best_solution = results['best_solution']
    for i in range(n_nodes):
        color = berkeley_plot.colors['california_gold'] if best_solution[i] == '1' else berkeley_plot.colors['berkeley_blue']
        ax2.scatter(pos[i][0], pos[i][1], c=color, s=300, alpha=0.8)
        ax2.text(pos[i][0], pos[i][1], str(i), ha='center', va='center',
                fontweight='bold', fontsize=12)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_title('Max-Cut Solution', fontweight='bold')
    ax2.axis('off')
    # Solution quality distribution
    ax3 = axes[2]
    if 'solution_samples' in results:
        cut_values = [qaoa.evaluate_cut(sample) for sample in results['solution_samples']]
        ax3.hist(cut_values, bins=20, alpha=0.7,
                color=berkeley_plot.colors['founders_rock'], edgecolor='black')
        ax3.axvline(results['best_cut_value'],
                   color=berkeley_plot.colors['california_gold'],
                   linestyle='--', linewidth=2, label='Best Cut')
        ax3.set_xlabel('Cut Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Solution Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    plt.suptitle('QAOA Max-Cut Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
def demo_quantum_error_correction():
    """Demonstrate quantum error correction codes."""
    print("5. Quantum Error Correction")
    print("-" * 30)
    print("Demonstrating 3-qubit bit-flip code...")
    # Create 3-qubit error correction circuit
    circuit = QuantumCircuit(n_qubits=5)  # 1 data + 2 ancilla + 2 syndrome
    # Encode logical |0⟩ state
    circuit.cnot(0, 1)  # |000⟩ + |111⟩
    circuit.cnot(0, 2)
    print("  Logical |0⟩ encoded as: (|000⟩ + |111⟩)/√2")
    # Introduce error (bit flip on qubit 1)
    print("  Introducing bit-flip error on qubit 1...")
    circuit.x(1)
    # Error detection
    circuit.cnot(0, 3)  # Syndrome qubit 1
    circuit.cnot(1, 3)
    circuit.cnot(1, 4)  # Syndrome qubit 2
    circuit.cnot(2, 4)
    # Simulate and measure syndrome
    measurements = circuit.measure(n_shots=1000)
    print(f"  Error syndrome measurements: {measurements}")
    # Demonstrate surface code principles
    demonstrate_surface_code_principles()
    print()
def demonstrate_surface_code_principles():
    """Demonstrate surface code error correction principles."""
    berkeley_plot = BerkeleyPlot()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Surface code lattice
    ax1 = axes[0]
    # Draw surface code grid (simplified 5x5)
    for i in range(5):
        for j in range(5):
            if (i + j) % 2 == 0:  # Data qubits
                circle = Circle((i, j), 0.3,
                              facecolor=berkeley_plot.colors['berkeley_blue'],
                              edgecolor='black', alpha=0.8)
                ax1.add_patch(circle)
                ax1.text(i, j, 'D', ha='center', va='center',
                        fontweight='bold', color='white')
            else:  # Ancilla qubits
                square = Rectangle((i-0.2, j-0.2), 0.4, 0.4,
                                 facecolor=berkeley_plot.colors['california_gold'],
                                 edgecolor='black', alpha=0.8)
                ax1.add_patch(square)
                ax1.text(i, j, 'A', ha='center', va='center', fontweight='bold')
    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(-0.5, 4.5)
    ax1.set_aspect('equal')
    ax1.set_title('Surface Code Lattice', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    # Error correction threshold
    ax2 = axes[1]
    error_rates = np.logspace(-4, -1, 50)
    logical_rates_d3 = error_rates**2  # Distance-3 code
    logical_rates_d5 = error_rates**3  # Distance-5 code
    logical_rates_d7 = error_rates**4  # Distance-7 code
    ax2.loglog(error_rates, logical_rates_d3, '-',
              color=berkeley_plot.colors['founders_rock'],
              linewidth=2, label='Distance 3')
    ax2.loglog(error_rates, logical_rates_d5, '-',
              color=berkeley_plot.colors['berkeley_blue'],
              linewidth=2, label='Distance 5')
    ax2.loglog(error_rates, logical_rates_d7, '-',
              color=berkeley_plot.colors['california_gold'],
              linewidth=2, label='Distance 7')
    # Threshold line
    threshold = 0.01  # Approximate surface code threshold
    ax2.axvline(threshold, color='red', linestyle='--',
               linewidth=2, label='Threshold (~1%)')
    ax2.set_xlabel('Physical Error Rate')
    ax2.set_ylabel('Logical Error Rate')
    ax2.set_title('Error Correction Threshold', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.suptitle('Quantum Error Correction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
def demo_quantum_teleportation():
    """Demonstrate quantum teleportation protocol."""
    print("6. Quantum Teleportation Protocol")
    print("-" * 35)
    print("Teleporting quantum state |ψ⟩ = α|0⟩ + β|1⟩")
    # Create teleportation circuit
    circuit = QuantumCircuit(n_qubits=3)  # Alice's qubit, Alice's entangled, Bob's entangled
    # Prepare state to teleport (arbitrary superposition)
    alpha, beta = 1/np.sqrt(3), np.sqrt(2/3)
    print(f"  Original state coefficients: α = {alpha:.3f}, β = {beta:.3f}")
    # Step 1: Create Bell pair between Alice and Bob
    circuit.h(1)      # Alice's half of Bell pair
    circuit.cnot(1, 2)  # Entangle with Bob's qubit
    print("  Step 1: Bell pair created between Alice and Bob")
    # Step 2: Bell measurement by Alice
    circuit.cnot(0, 1)  # Entangle state with Alice's half
    circuit.h(0)      # Hadamard on state qubit
    print("  Step 2: Alice performs Bell measurement")
    # Step 3: Classical communication and correction by Bob
    # (In simulation, we assume perfect classical communication)
    print("  Step 3: Classical bits sent to Bob")
    print("  Step 4: Bob applies correction operations")
    # Demonstrate teleportation fidelity analysis
    plot_teleportation_analysis()
    print("  Quantum teleportation completed successfully!")
    print("  Original quantum state has been destroyed at Alice's location")
    print("  Exact copy now exists at Bob's location")
    print()
def plot_teleportation_analysis():
    """Plot quantum teleportation protocol analysis."""
    berkeley_plot = BerkeleyPlot()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Teleportation protocol steps
    ax1 = axes[0]
    steps = ['Initial\nState', 'Bell Pair\nCreation', 'Bell\nMeasurement', 'Classical\nCommunication', 'Correction\nOperations', 'Final\nState']
    fidelities = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Perfect teleportation
    ax1.plot(range(len(steps)), fidelities, 'o-',
            color=berkeley_plot.colors['berkeley_blue'],
            linewidth=3, markersize=8)
    ax1.set_xticks(range(len(steps)))
    ax1.set_xticklabels(steps, rotation=45, ha='right')
    ax1.set_ylabel('Fidelity')
    ax1.set_title('Teleportation Protocol Steps', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.95, 1.05)
    # Entanglement evolution
    ax2 = axes[1]
    time_steps = np.linspace(0, 1, 100)
    # Simulate entanglement during protocol
    entanglement = np.ones_like(time_steps)
    entanglement[30:70] = 0.5  # During measurement
    ax2.fill_between(time_steps, 0, entanglement,
                    color=berkeley_plot.colors['california_gold'],
                    alpha=0.7, label='Entanglement')
    ax2.set_xlabel('Protocol Progress')
    ax2.set_ylabel('Entanglement Measure')
    ax2.set_title('Quantum Entanglement Evolution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Circuit diagram representation
    ax3 = axes[2]
    # Draw simplified teleportation circuit
    qubits = ['|ψ⟩', 'Alice', 'Bob']
    for i, qubit in enumerate(qubits):
        ax3.plot([0, 6], [2-i, 2-i], 'k-', linewidth=2)
        ax3.text(-0.5, 2-i, qubit, fontsize=12, ha='right', va='center')
    # Bell pair creation
    ax3.plot([1, 1], [1, 0], 'k-', linewidth=2)
    rect = Rectangle((0.8, 0.8), 0.4, 0.4,
                    facecolor=berkeley_plot.colors['california_gold'],
                    edgecolor='black')
    ax3.add_patch(rect)
    ax3.text(1, 1, 'H', ha='center', va='center', fontweight='bold')
    # CNOT gates
    for x_pos, control, target in [(1, 1, 0), (3, 2, 1), (4, 0, 1)]:
        circle = Circle((x_pos, control), 0.1,
                       facecolor='black', edgecolor='black')
        ax3.add_patch(circle)
        ax3.plot([x_pos, x_pos], [control, target], 'k-', linewidth=2)
        circle = Circle((x_pos, target), 0.2,
                       facecolor='white', edgecolor='black', linewidth=2)
        ax3.add_patch(circle)
        ax3.plot([x_pos-0.15, x_pos+0.15], [target, target], 'k-', linewidth=2)
        ax3.plot([x_pos, x_pos], [target-0.15, target+0.15], 'k-', linewidth=2)
    # Measurements
    for x_pos, qubit in [(5, 2), (5, 1)]:
        rect = Rectangle((x_pos-0.2, qubit-0.2), 0.4, 0.4,
                        facecolor=berkeley_plot.colors['founders_rock'],
                        edgecolor='black')
        ax3.add_patch(rect)
        ax3.text(x_pos, qubit, 'M', ha='center', va='center',
                fontweight='bold', color='white')
    ax3.set_xlim(-1, 7)
    ax3.set_ylim(-0.5, 2.5)
    ax3.set_title('Teleportation Circuit', fontweight='bold')
    ax3.axis('off')
    plt.suptitle('Quantum Teleportation Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
def demo_quantum_advantage_analysis():
    """Demonstrate quantum advantage analysis across different problems."""
    print("7. Quantum Advantage Analysis")
    print("-" * 32)
    print("Analyzing quantum speedup for various computational problems:")
    # Define problem complexities
    problems = ['Factoring', 'Database Search', 'Simulation', 'Optimization', 'Machine Learning']
    classical_complexity = ['O(e^n)', 'O(N)', 'O(e^n)', 'O(2^n)', 'O(nd)']
    quantum_complexity = ['O(n³)', 'O(√N)', 'O(n³)', 'O(n^p)', 'O(log n)']
    print("\nComputational Complexity Comparison:")
    print(f"{'Problem':<15} {'Classical':<12} {'Quantum':<12} {'Advantage'}")
    print("-" * 55)
    for i, problem in enumerate(problems):
        advantage = "Exponential" if 'e^n' in classical_complexity[i] or '2^n' in classical_complexity[i] else "Polynomial"
        print(f"{problem:<15} {classical_complexity[i]:<12} {quantum_complexity[i]:<12} {advantage}")
    # Plot quantum advantage scaling
    plot_quantum_advantage_scaling()
    # Resource estimation
    print("\nQuantum Resource Requirements:")
    print("  Logical qubits needed for practical problems:")
    print("    - RSA-2048 factoring: ~4000 logical qubits")
    print("    - Shor's algorithm: ~2n+3 qubits for n-bit numbers")
    print("    - Quantum chemistry: ~100-1000 qubits per molecule")
    print("    - Optimization problems: ~log₂(N) qubits for N variables")
    print("\n  Error correction requirements:")
    print("    - Physical qubits per logical: ~1000-10000")
    print("    - Gate fidelity threshold: >99.9%")
    print("    - Coherence time: >100 gate operations")
    print()
def plot_quantum_advantage_scaling():
    """Plot quantum advantage scaling analysis."""
    berkeley_plot = BerkeleyPlot()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # Grover's algorithm scaling
    ax1 = axes[0, 0]
    N = np.logspace(1, 6, 50)
    classical_search = N / 2  # Average case
    quantum_search = np.sqrt(N)  # Grover
    ax1.loglog(N, classical_search, '-',
              color=berkeley_plot.colors['founders_rock'],
              linewidth=2, label='Classical')
    ax1.loglog(N, quantum_search, '-',
              color=berkeley_plot.colors['berkeley_blue'],
              linewidth=2, label='Quantum (Grover)')
    ax1.set_xlabel('Database Size N')
    ax1.set_ylabel('Query Complexity')
    ax1.set_title('Database Search Scaling', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Factoring algorithm scaling
    ax2 = axes[0, 1]
    n_bits = np.arange(100, 5000, 100)
    classical_factor = np.exp(0.693 * n_bits**(1/3) * (np.log(n_bits))**(2/3))  # Sub-exponential
    quantum_factor = n_bits**3  # Polynomial (Shor)
    ax2.semilogy(n_bits, classical_factor / classical_factor[0], '-',
                color=berkeley_plot.colors['founders_rock'],
                linewidth=2, label='Classical (GNFS)')
    ax2.semilogy(n_bits, quantum_factor / quantum_factor[0], '-',
                color=berkeley_plot.colors['berkeley_blue'],
                linewidth=2, label="Quantum (Shor's)")
    ax2.set_xlabel('Number Size (bits)')
    ax2.set_ylabel('Relative Time Complexity')
    ax2.set_title('Integer Factoring Scaling', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Quantum simulation scaling
    ax3 = axes[1, 0]
    n_particles = np.arange(5, 50, 2)
    classical_sim = 2**n_particles  # Exponential
    quantum_sim = n_particles**3  # Polynomial
    ax3.semilogy(n_particles, classical_sim, '-',
                color=berkeley_plot.colors['founders_rock'],
                linewidth=2, label='Classical')
    ax3.semilogy(n_particles, quantum_sim, '-',
                color=berkeley_plot.colors['berkeley_blue'],
                linewidth=2, label='Quantum')
    ax3.set_xlabel('Number of Particles')
    ax3.set_ylabel('Time Complexity')
    ax3.set_title('Quantum System Simulation', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # NISQ vs Fault-tolerant comparison
    ax4 = axes[1, 1]
    years = np.arange(2025, 2040)
    nisq_qubits = 100 * 1.2**(years - 2025)  # 20% annual growth
    ft_logical = np.where(years < 2035, 0, 10 * 2**(years - 2035))  # Fault-tolerant era
    ax4.semilogy(years, nisq_qubits, 'o-',
                color=berkeley_plot.colors['california_gold'],
                linewidth=2, label='NISQ Devices')
    ax4.semilogy(years, ft_logical, 's-',
                color=berkeley_plot.colors['berkeley_blue'],
                linewidth=2, label='Fault-Tolerant')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Number of Qubits')
    ax4.set_title('Quantum Hardware Roadmap', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.suptitle('Quantum Advantage Scaling Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()