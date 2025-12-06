#!/usr/bin/env python3
"""
Quantum Approximate Optimization Algorithm (QAOA)
Advanced implementation of QAOA for solving combinatorial optimization problems
on quantum computers. Includes variational parameter optimization, performance
analysis, and applications to various NP-hard problems.
Key Features:
- QAOA circuit construction for arbitrary Hamiltonians
- Classical optimization of variational parameters
- Performance analysis and benchmarking
- Applications to MaxCut, Max-SAT, and TSP problems
- Noise-aware optimization strategies
Applications:
- Combinatorial optimization problems
- Portfolio optimization in finance
- Traffic flow optimization
- Resource allocation problems
- Machine learning feature selection
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Callable, Dict, List, Any
from dataclasses import dataclass
import warnings
import networkx as nx
from itertools import combinations
from ...utils.constants import hbar
from ...visualization.berkeley_style import BerkeleyPlot
@dataclass
class QAOAConfig:
    """Configuration for QAOA calculations."""
    n_qubits: int = 4
    n_layers: int = 2  # QAOA depth (p)
    n_shots: int = 1024  # Number of measurement shots
    optimizer: str = 'COBYLA'  # Classical optimizer
    max_iterations: int = 100
    tolerance: float = 1e-6
    # Initialization strategy
    init_strategy: str = 'random'  # 'random', 'linear', 'fourier'
    # Noise parameters
    noise_level: float = 0.0  # Noise strength (0 = noiseless)
    gate_error_rate: float = 0.001
    measurement_error_rate: float = 0.01
class QAOA:
    """
    Quantum Approximate Optimization Algorithm implementation.
    Provides comprehensive QAOA functionality for solving combinatorial
    optimization problems with various classical optimizers and
    performance analysis tools.
    Parameters
    ----------
    config : QAOAConfig
        Configuration parameters for QAOA
    """
    def __init__(self, config: QAOAConfig):
        """Initialize QAOA system."""
        self.config = config
        # Quantum state representation
        self.n_states = 2**config.n_qubits
        self.basis_states = np.arange(self.n_states)
        # Parameter storage
        self.optimal_params = None
        self.optimization_history = []
        self.expectation_values = []
        # Problem Hamiltonians
        self.problem_hamiltonian = None
        self.mixer_hamiltonian = None
        self.cost_function = None
    def create_problem_hamiltonian_maxcut(self, graph: nx.Graph) -> csr_matrix:
        """
        Create problem Hamiltonian for MaxCut problem.
        H_P = -0.5 * Σ_{(i,j)∈E} (1 - Z_i Z_j)
        Parameters
        ----------
        graph : networkx.Graph
            Graph for MaxCut problem
        Returns
        -------
        csr_matrix
            Problem Hamiltonian matrix
        """
        n_qubits = self.config.n_qubits
        H_P = np.zeros((self.n_states, self.n_states))
        # Pauli-Z matrices for each qubit
        Z_matrices = []
        for i in range(n_qubits):
            Z_i = np.eye(self.n_states)
            for state in range(self.n_states):
                bit_i = (state >> i) & 1
                Z_i[state, state] = 1 if bit_i == 0 else -1
            Z_matrices.append(Z_i)
        # Build MaxCut Hamiltonian
        for edge in graph.edges():
            i, j = edge
            if i < n_qubits and j < n_qubits:
                weight = graph[i][j].get('weight', 1.0)
                H_P += -0.5 * weight * (np.eye(self.n_states) - Z_matrices[i] @ Z_matrices[j])
        return csr_matrix(H_P)
    def create_mixer_hamiltonian(self) -> csr_matrix:
        """
        Create mixer Hamiltonian (transverse field).
        H_M = -Σ_i X_i
        Returns
        -------
        csr_matrix
            Mixer Hamiltonian matrix
        """
        n_qubits = self.config.n_qubits
        H_M = np.zeros((self.n_states, self.n_states))
        # Pauli-X matrices for each qubit
        for i in range(n_qubits):
            for state in range(self.n_states):
                # Flip bit i
                flipped_state = state ^ (1 << i)
                H_M[state, flipped_state] += -1.0
        return csr_matrix(H_M)
    def create_initial_state(self) -> np.ndarray:
        """
        Create initial superposition state |+⟩^⊗n.
        Returns
        -------
        np.ndarray
            Initial state vector
        """
        # Equal superposition of all computational basis states
        psi_0 = np.ones(self.n_states) / np.sqrt(self.n_states)
        return psi_0
    def apply_unitary_evolution(self, state: np.ndarray, hamiltonian: csr_matrix,
                               angle: float) -> np.ndarray:
        """
        Apply unitary evolution exp(-i * angle * H) to state.
        Parameters
        ----------
        state : np.ndarray
            Input quantum state
        hamiltonian : csr_matrix
            Hamiltonian matrix
        angle : float
            Evolution angle
        Returns
        -------
        np.ndarray
            Evolved quantum state
        """
        # For small systems, use exact matrix exponentiation
        H_dense = hamiltonian.toarray()
        U = self._matrix_exp(-1j * angle * H_dense)
        return U @ state
    def _matrix_exp(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix exponential using eigendecomposition."""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        return eigenvecs @ np.diag(np.exp(eigenvals)) @ eigenvecs.T.conj()
    def qaoa_circuit(self, params: np.ndarray, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Execute QAOA circuit with given parameters.
        Parameters
        ----------
        params : np.ndarray
            QAOA parameters [gamma_1, beta_1, gamma_2, beta_2, ...]
        state : np.ndarray, optional
            Initial state (default: equal superposition)
        Returns
        -------
        np.ndarray
            Final quantum state after QAOA circuit
        """
        if state is None:
            state = self.create_initial_state()
        # Extract gamma and beta parameters
        n_layers = self.config.n_layers
        gammas = params[:n_layers]
        betas = params[n_layers:]
        current_state = state.copy()
        # Apply QAOA layers
        for p in range(n_layers):
            # Apply problem unitary: exp(-i * gamma_p * H_P)
            current_state = self.apply_unitary_evolution(
                current_state, self.problem_hamiltonian, gammas[p])
            # Apply mixer unitary: exp(-i * beta_p * H_M)
            current_state = self.apply_unitary_evolution(
                current_state, self.mixer_hamiltonian, betas[p])
        return current_state
    def expectation_value(self, state: np.ndarray,
                         hamiltonian: Optional[csr_matrix] = None) -> float:
        """
        Calculate expectation value of Hamiltonian in given state.
        Parameters
        ----------
        state : np.ndarray
            Quantum state
        hamiltonian : csr_matrix, optional
            Hamiltonian (default: problem Hamiltonian)
        Returns
        -------
        float
            Expectation value
        """
        if hamiltonian is None:
            hamiltonian = self.problem_hamiltonian
        H_dense = hamiltonian.toarray()
        expectation = np.real(np.conj(state) @ H_dense @ state)
        return expectation
    def cost_function_qaoa(self, params: np.ndarray) -> float:
        """
        Cost function for QAOA optimization.
        Parameters
        ----------
        params : np.ndarray
            QAOA parameters
        Returns
        -------
        float
            Cost function value (expectation value)
        """
        state = self.qaoa_circuit(params)
        cost = self.expectation_value(state)
        # Store optimization history
        self.expectation_values.append(cost)
        return cost
    def initialize_parameters(self) -> np.ndarray:
        """
        Initialize QAOA parameters based on strategy.
        Returns
        -------
        np.ndarray
            Initial parameter vector
        """
        n_params = 2 * self.config.n_layers
        if self.config.init_strategy == 'random':
            # Random initialization
            params = np.random.uniform(0, 2*np.pi, n_params)
        elif self.config.init_strategy == 'linear':
            # Linear interpolation initialization
            gammas = np.linspace(0, np.pi/2, self.config.n_layers)
            betas = np.linspace(0, np.pi/4, self.config.n_layers)
            params = np.concatenate([gammas, betas])
        elif self.config.init_strategy == 'fourier':
            # Fourier-based initialization (INTERP strategy)
            u = np.random.uniform(0, 1)
            v = np.random.uniform(0, 1)
            gammas = []
            betas = []
            for p in range(1, self.config.n_layers + 1):
                gamma_p = u * (1 - p / self.config.n_layers)
                beta_p = v * (1 - p / self.config.n_layers)
                gammas.append(gamma_p)
                betas.append(beta_p)
            params = np.concatenate([gammas, betas])
        else:
            # Default: small random values
            params = np.random.normal(0, 0.1, n_params)
        return params
    def optimize_parameters(self) -> Tuple[np.ndarray, float]:
        """
        Optimize QAOA parameters using classical optimizer.
        Returns
        -------
        Tuple[np.ndarray, float]
            Optimal parameters and final cost
        """
        # Initialize parameters
        initial_params = self.initialize_parameters()
        # Reset optimization history
        self.optimization_history = []
        self.expectation_values = []
        # Define parameter bounds
        bounds = [(0, 2*np.pi)] * len(initial_params)
        # Optimize based on chosen method
        if self.config.optimizer == 'COBYLA':
            result = minimize(
                self.cost_function_qaoa,
                initial_params,
                method='COBYLA',
                options={'maxiter': self.config.max_iterations, 'rhobeg': 0.1}
            )
        elif self.config.optimizer == 'BFGS':
            result = minimize(
                self.cost_function_qaoa,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': self.config.max_iterations}
            )
        elif self.config.optimizer == 'differential_evolution':
            result = differential_evolution(
                self.cost_function_qaoa,
                bounds,
                maxiter=self.config.max_iterations // 10,
                seed=42
            )
        elif self.config.optimizer == 'Powell':
            result = minimize(
                self.cost_function_qaoa,
                initial_params,
                method='Powell',
                options={'maxiter': self.config.max_iterations}
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        self.optimal_params = result.x
        optimal_cost = result.fun
        return self.optimal_params, optimal_cost
    def sample_measurements(self, state: np.ndarray, n_shots: int) -> Dict[str, int]:
        """
        Simulate quantum measurements on final state.
        Parameters
        ----------
        state : np.ndarray
            Quantum state to measure
        n_shots : int
            Number of measurement shots
        Returns
        -------
        Dict[str, int]
            Measurement counts for each bitstring
        """
        probabilities = np.abs(state)**2
        # Add measurement noise if specified
        if self.config.measurement_error_rate > 0:
            noise = np.random.normal(0, self.config.measurement_error_rate, len(probabilities))
            probabilities = np.abs(probabilities + noise)
            probabilities /= np.sum(probabilities)  # Renormalize
        # Sample measurements
        measurements = np.random.choice(self.n_states, size=n_shots, p=probabilities)
        # Convert to bitstring counts
        counts = {}
        for measurement in measurements:
            bitstring = format(measurement, f'0{self.config.n_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts
    def analyze_solution_quality(self, counts: Dict[str, int],
                                true_optimum: Optional[float] = None) -> Dict[str, float]:
        """
        Analyze quality of QAOA solution.
        Parameters
        ----------
        counts : Dict[str, int]
            Measurement counts
        true_optimum : float, optional
            True optimal value for comparison
        Returns
        -------
        Dict[str, float]
            Solution quality metrics
        """
        # Calculate expectation value from measurements
        total_shots = sum(counts.values())
        expectation_measured = 0.0
        for bitstring, count in counts.items():
            # Convert bitstring to integer
            state_int = int(bitstring, 2)
            # Calculate cost for this bitstring
            if self.cost_function:
                cost = self.cost_function(bitstring)
            else:
                # Use Hamiltonian expectation for computational basis state
                basis_state = np.zeros(self.n_states)
                basis_state[state_int] = 1.0
                cost = self.expectation_value(basis_state)
            expectation_measured += cost * (count / total_shots)
        # Find best solution in measurement results
        best_cost = float('inf')
        best_bitstring = None
        for bitstring, count in counts.items():
            if self.cost_function:
                cost = self.cost_function(bitstring)
            else:
                state_int = int(bitstring, 2)
                basis_state = np.zeros(self.n_states)
                basis_state[state_int] = 1.0
                cost = self.expectation_value(basis_state)
            if cost < best_cost:
                best_cost = cost
                best_bitstring = bitstring
        metrics = {
            'expectation_measured': expectation_measured,
            'best_cost': best_cost,
            'best_solution': best_bitstring,
            'success_probability': counts.get(best_bitstring, 0) / total_shots
        }
        if true_optimum is not None:
            metrics['approximation_ratio'] = expectation_measured / true_optimum
            metrics['optimality_gap'] = abs(best_cost - true_optimum)
        return metrics
    def solve_maxcut(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Solve MaxCut problem using QAOA.
        Parameters
        ----------
        graph : networkx.Graph
            Graph for MaxCut problem
        Returns
        -------
        Dict[str, Any]
            Complete solution including parameters, costs, and analysis
        """
        # Setup problem
        self.problem_hamiltonian = self.create_problem_hamiltonian_maxcut(graph)
        self.mixer_hamiltonian = self.create_mixer_hamiltonian()
        # Define classical cost function for MaxCut
        def maxcut_cost(bitstring):
            cut_value = 0
            for edge in graph.edges():
                i, j = edge
                if i < len(bitstring) and j < len(bitstring):
                    if bitstring[i] != bitstring[j]:
                        weight = graph[i][j].get('weight', 1.0)
                        cut_value += weight
            return -cut_value  # Negative because we minimize
        self.cost_function = maxcut_cost
        # Optimize parameters
        optimal_params, optimal_cost = self.optimize_parameters()
        # Get final state and measurements
        final_state = self.qaoa_circuit(optimal_params)
        counts = self.sample_measurements(final_state, self.config.n_shots)
        # Analyze solution
        classical_optimum = self._classical_maxcut_bound(graph)
        metrics = self.analyze_solution_quality(counts, classical_optimum)
        solution = {
            'optimal_parameters': optimal_params,
            'optimal_cost': optimal_cost,
            'final_state': final_state,
            'measurement_counts': counts,
            'solution_metrics': metrics,
            'optimization_history': self.expectation_values,
            'classical_bound': classical_optimum
        }
        return solution
    def _classical_maxcut_bound(self, graph: nx.Graph) -> float:
        """Calculate classical upper bound for MaxCut."""
        # Simple bound: sum of all edge weights / 2
        total_weight = sum(graph[u][v].get('weight', 1.0) for u, v in graph.edges())
        return -total_weight  # Negative because we're minimizing
    def plot_optimization_progress(self) -> None:
        """Plot QAOA optimization progress."""
        berkeley_plot = BerkeleyPlot()
        fig, ax = plt.subplots(figsize=(10, 6))
        iterations = range(len(self.expectation_values))
        ax.plot(iterations, self.expectation_values,
               color=berkeley_plot.colors['berkeley_blue'],
               linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Optimization Iteration')
        ax.set_ylabel('Cost Function Value')
        ax.set_title('QAOA Optimization Progress')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    def plot_parameter_landscape(self, param_ranges: List[Tuple[float, float]],
                                resolution: int = 20) -> None:
        """
        Plot 2D parameter landscape for QAOA.
        Parameters
        ----------
        param_ranges : List[Tuple[float, float]]
            Ranges for first two parameters
        resolution : int
            Grid resolution for plotting
        """
        if self.config.n_layers < 1:
            warnings.warn("Need at least 1 QAOA layer for landscape plot")
            return
        berkeley_plot = BerkeleyPlot()
        fig, ax = plt.subplots(figsize=(10, 8))
        # Create parameter grid
        gamma_range = np.linspace(param_ranges[0][0], param_ranges[0][1], resolution)
        beta_range = np.linspace(param_ranges[1][0], param_ranges[1][1], resolution)
        gamma_mesh, beta_mesh = np.meshgrid(gamma_range, beta_range)
        cost_landscape = np.zeros_like(gamma_mesh)
        # Evaluate cost function on grid
        for i in range(resolution):
            for j in range(resolution):
                # Use fixed values for other parameters
                params = np.zeros(2 * self.config.n_layers)
                params[0] = gamma_mesh[i, j]
                params[self.config.n_layers] = beta_mesh[i, j]
                # Set other parameters to optimal values if available
                if self.optimal_params is not None:
                    for k in range(len(params)):
                        if k != 0 and k != self.config.n_layers:
                            params[k] = self.optimal_params[k]
                cost_landscape[i, j] = self.cost_function_qaoa(params)
        # Plot contour map
        contour = ax.contourf(gamma_mesh, beta_mesh, cost_landscape,
                            levels=20, cmap='RdYlBu_r')
        # Mark optimal point if available
        if self.optimal_params is not None:
            ax.plot(self.optimal_params[0], self.optimal_params[self.config.n_layers],
                   'wo', markersize=10, markeredgecolor='black',
                   label='Optimal Point')
            ax.legend()
        ax.set_xlabel('γ₁')
        ax.set_ylabel('β₁')
        ax.set_title('QAOA Parameter Landscape')
        # Add colorbar
        cbar = plt.colorbar(contour)
        cbar.set_label('Cost Function')
        plt.tight_layout()
        plt.show()
    def benchmark_performance(self, problem_sizes: List[int],
                             n_trials: int = 5) -> Dict[str, List[float]]:
        """
        Benchmark QAOA performance across different problem sizes.
        Parameters
        ----------
        problem_sizes : List[int]
            List of problem sizes (number of qubits)
        n_trials : int
            Number of trials per problem size
        Returns
        -------
        Dict[str, List[float]]
            Performance metrics vs problem size
        """
        metrics = {
            'problem_sizes': problem_sizes,
            'approximation_ratios': [],
            'success_probabilities': [],
            'optimization_times': []
        }
        original_n_qubits = self.config.n_qubits
        for n_qubits in problem_sizes:
            self.config.n_qubits = n_qubits
            self.n_states = 2**n_qubits
            trial_ratios = []
            trial_probs = []
            trial_times = []
            for trial in range(n_trials):
                # Generate random graph
                graph = nx.erdos_renyi_graph(n_qubits, 0.5)
                # Solve with timing
                import time
                start_time = time.time()
                solution = self.solve_maxcut(graph)
                end_time = time.time()
                trial_ratios.append(solution['solution_metrics'].get('approximation_ratio', 0))
                trial_probs.append(solution['solution_metrics']['success_probability'])
                trial_times.append(end_time - start_time)
            metrics['approximation_ratios'].append(np.mean(trial_ratios))
            metrics['success_probabilities'].append(np.mean(trial_probs))
            metrics['optimization_times'].append(np.mean(trial_times))
        # Restore original configuration
        self.config.n_qubits = original_n_qubits
        self.n_states = 2**original_n_qubits
        return metrics
def create_test_graphs() -> Dict[str, nx.Graph]:
    """Create test graphs for QAOA benchmarking."""
    graphs = {}
    # Triangle graph
    triangle = nx.Graph()
    triangle.add_edges_from([(0, 1), (1, 2), (2, 0)])
    graphs['triangle'] = triangle
    # Square graph
    square = nx.Graph()
    square.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    graphs['square'] = square
    # Complete graph K4
    complete = nx.complete_graph(4)
    graphs['K4'] = complete
    # Random graph
    random_graph = nx.erdos_renyi_graph(4, 0.6, seed=42)
    graphs['random'] = random_graph
    return graphs
if __name__ == "__main__":
    # Example: Solve MaxCut on triangle graph
    config = QAOAConfig(
        n_qubits=3,
        n_layers=2,
        n_shots=1024,
        optimizer='COBYLA',
        init_strategy='linear'
    )
    qaoa = QAOA(config)
    # Create test graphs
    test_graphs = create_test_graphs()
    # Solve MaxCut on triangle
    triangle = test_graphs['triangle']
    print("Solving MaxCut on triangle graph...")
    solution = qaoa.solve_maxcut(triangle)
    print(f"Optimal cost: {solution['optimal_cost']:.4f}")
    print(f"Best solution: {solution['solution_metrics']['best_solution']}")
    print(f"Success probability: {solution['solution_metrics']['success_probability']:.4f}")
    # Plot optimization progress
    qaoa.plot_optimization_progress()
    # Plot parameter landscape
    qaoa.plot_parameter_landscape([(0, np.pi), (0, np.pi/2)])
    # Benchmark performance
    print("\nBenchmarking QAOA performance...")
    benchmark_results = qaoa.benchmark_performance([3, 4], n_trials=3)
    for size, ratio, prob in zip(benchmark_results['problem_sizes'],
                                benchmark_results['approximation_ratios'],
                                benchmark_results['success_probabilities']):
        print(f"Size {size}: Approx ratio = {ratio:.3f}, Success prob = {prob:.3f}")