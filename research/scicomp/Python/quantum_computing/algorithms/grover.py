#!/usr/bin/env python3
"""
Grover's Quantum Search Algorithm
Comprehensive implementation of Grover's algorithm for quantum database search
with various oracle constructions, amplitude amplification variants, and
performance analysis for different search problems.
Key Features:
- Classical Grover's algorithm implementation
- Amplitude amplification generalization
- Various oracle constructions for different problems
- Geometric analysis and optimal iteration calculation
- Quantum counting and search with unknown number of solutions
Applications:
- Database search and pattern matching
- Satisfiability (SAT) problem solving
- Optimization problem acceleration
- Cryptographic key search
- Machine learning feature selection
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Callable, Dict, List, Any
from dataclasses import dataclass
import warnings
from itertools import product
from ...utils.constants import hbar
from ...visualization.berkeley_style import BerkeleyPlot
@dataclass
class GroverConfig:
    """Configuration for Grover's algorithm."""
    n_qubits: int = 4
    n_shots: int = 1024  # Number of measurement shots
    # Search parameters
    target_items: Optional[List[str]] = None  # Target bitstrings
    n_solutions: Optional[int] = None  # Number of solutions (if known)
    # Algorithm variants
    use_optimal_iterations: bool = True
    max_iterations: int = 100
    # Amplitude amplification parameters
    rotation_angles: Optional[Tuple[float, float]] = None
    # Analysis parameters
    calculate_success_probability: bool = True
    geometric_analysis: bool = True
class GroverSearch:
    """
    Comprehensive Grover's algorithm implementation.
    Provides classical simulation of Grover's quantum search algorithm
    with various oracle constructions and amplitude amplification
    techniques for solving search and optimization problems.
    Parameters
    ----------
    config : GroverConfig
        Configuration parameters for Grover's algorithm
    """
    def __init__(self, config: GroverConfig):
        """Initialize Grover search system."""
        self.config = config
        # Quantum system parameters
        self.n_qubits = config.n_qubits
        self.n_states = 2**config.n_qubits
        # Target configuration
        if config.target_items is not None:
            self.target_items = set(config.target_items)
            self.n_solutions = len(self.target_items)
        else:
            self.target_items = set()
            self.n_solutions = config.n_solutions or 1
        # Results storage
        self.iteration_history = []
        self.amplitude_history = []
        self.probability_history = []
        # Create basis states
        self.basis_states = [format(i, f'0{self.n_qubits}b')
                           for i in range(self.n_states)]
        # Initialize quantum state
        self.quantum_state = None
    def create_uniform_superposition(self) -> np.ndarray:
        """
        Create uniform superposition state |s⟩ = H^⊗n|0⟩.
        Returns
        -------
        np.ndarray
            Uniform superposition state vector
        """
        # Equal amplitude for all computational basis states
        state = np.ones(self.n_states, dtype=complex) / np.sqrt(self.n_states)
        return state
    def oracle_operator(self, state: np.ndarray,
                       target_function: Optional[Callable[[str], bool]] = None) -> np.ndarray:
        """
        Apply oracle operator O_f that flips the phase of target states.
        O_f|x⟩ = (-1)^f(x)|x⟩
        Parameters
        ----------
        state : np.ndarray
            Input quantum state
        target_function : callable, optional
            Function f(x) that returns True for target states
        Returns
        -------
        np.ndarray
            State after oracle application
        """
        oracle_state = state.copy()
        for i, basis_state in enumerate(self.basis_states):
            if target_function:
                is_target = target_function(basis_state)
            else:
                is_target = basis_state in self.target_items
            if is_target:
                oracle_state[i] *= -1  # Phase flip
        return oracle_state
    def diffusion_operator(self, state: np.ndarray) -> np.ndarray:
        """
        Apply diffusion operator (inversion about average).
        D = 2|s⟩⟨s| - I
        Parameters
        ----------
        state : np.ndarray
            Input quantum state
        Returns
        -------
        np.ndarray
            State after diffusion operator
        """
        # Calculate average amplitude
        avg_amplitude = np.mean(state)
        # Inversion about average: 2*avg - amplitude
        diffused_state = 2 * avg_amplitude - state
        return diffused_state
    def grover_iteration(self, state: np.ndarray,
                        target_function: Optional[Callable[[str], bool]] = None) -> np.ndarray:
        """
        Single Grover iteration: Oracle followed by Diffusion.
        G = D * O_f
        Parameters
        ----------
        state : np.ndarray
            Input quantum state
        target_function : callable, optional
            Target function for oracle
        Returns
        -------
        np.ndarray
            State after one Grover iteration
        """
        # Apply oracle
        state_after_oracle = self.oracle_operator(state, target_function)
        # Apply diffusion operator
        state_after_diffusion = self.diffusion_operator(state_after_oracle)
        return state_after_diffusion
    def calculate_optimal_iterations(self, n_solutions: Optional[int] = None) -> int:
        """
        Calculate optimal number of Grover iterations.
        For N total states and M solutions:
        k_opt ≈ π/4 * sqrt(N/M) - 1/2
        Parameters
        ----------
        n_solutions : int, optional
            Number of solutions (uses configured value if None)
        Returns
        -------
        int
            Optimal number of iterations
        """
        if n_solutions is None:
            n_solutions = self.n_solutions
        if n_solutions <= 0 or n_solutions >= self.n_states:
            return 0
        # Grover's formula
        theta = np.arcsin(np.sqrt(n_solutions / self.n_states))
        k_opt = np.pi / (4 * theta) - 0.5
        return max(0, int(np.round(k_opt)))
    def success_probability(self, n_iterations: int,
                          n_solutions: Optional[int] = None) -> float:
        """
        Calculate theoretical success probability after k iterations.
        P_success = sin²((2k+1)θ) where sin(θ) = sqrt(M/N)
        Parameters
        ----------
        n_iterations : int
            Number of Grover iterations
        n_solutions : int, optional
            Number of solutions
        Returns
        -------
        float
            Success probability
        """
        if n_solutions is None:
            n_solutions = self.n_solutions
        if n_solutions <= 0:
            return 0.0
        theta = np.arcsin(np.sqrt(n_solutions / self.n_states))
        prob = np.sin((2 * n_iterations + 1) * theta)**2
        return prob
    def run_grover_search(self, target_function: Optional[Callable[[str], bool]] = None,
                         n_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run complete Grover search algorithm.
        Parameters
        ----------
        target_function : callable, optional
            Function defining target states
        n_iterations : int, optional
            Number of iterations (uses optimal if None)
        Returns
        -------
        Dict[str, Any]
            Complete search results
        """
        # Initialize state
        current_state = self.create_uniform_superposition()
        self.quantum_state = current_state.copy()
        # Determine number of iterations
        if n_iterations is None:
            if self.config.use_optimal_iterations:
                n_iterations = self.calculate_optimal_iterations()
            else:
                n_iterations = 1
        # Store initial state
        self.iteration_history = [current_state.copy()]
        self.amplitude_history = [current_state.copy()]
        # Calculate initial probability
        initial_prob = self._calculate_target_probability(current_state, target_function)
        self.probability_history = [initial_prob]
        print(f"Starting Grover search with {n_iterations} iterations")
        print(f"Initial success probability: {initial_prob:.6f}")
        # Run Grover iterations
        for iteration in range(n_iterations):
            current_state = self.grover_iteration(current_state, target_function)
            # Store iteration results
            self.iteration_history.append(current_state.copy())
            self.amplitude_history.append(current_state.copy())
            # Calculate success probability
            prob = self._calculate_target_probability(current_state, target_function)
            self.probability_history.append(prob)
            print(f"Iteration {iteration + 1}: Success probability = {prob:.6f}")
        # Final measurements
        final_measurements = self.measure_state(current_state, self.config.n_shots)
        # Analyze results
        success_rate = self._analyze_measurements(final_measurements, target_function)
        results = {
            'final_state': current_state,
            'n_iterations': n_iterations,
            'measurements': final_measurements,
            'success_rate': success_rate,
            'probability_history': self.probability_history,
            'optimal_iterations': self.calculate_optimal_iterations(),
            'theoretical_success_prob': self.success_probability(n_iterations)
        }
        return results
    def _calculate_target_probability(self, state: np.ndarray,
                                    target_function: Optional[Callable[[str], bool]] = None) -> float:
        """Calculate probability of measuring target states."""
        total_prob = 0.0
        for i, basis_state in enumerate(self.basis_states):
            if target_function:
                is_target = target_function(basis_state)
            else:
                is_target = basis_state in self.target_items
            if is_target:
                total_prob += abs(state[i])**2
        return total_prob
    def measure_state(self, state: np.ndarray, n_shots: int) -> Dict[str, int]:
        """
        Simulate quantum measurements.
        Parameters
        ----------
        state : np.ndarray
            Quantum state to measure
        n_shots : int
            Number of measurement shots
        Returns
        -------
        Dict[str, int]
            Measurement counts for each basis state
        """
        probabilities = np.abs(state)**2
        # Sample measurements
        measurements = np.random.choice(
            range(self.n_states),
            size=n_shots,
            p=probabilities
        )
        # Convert to counts
        counts = {}
        for measurement in measurements:
            bitstring = self.basis_states[measurement]
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts
    def _analyze_measurements(self, measurements: Dict[str, int],
                            target_function: Optional[Callable[[str], bool]] = None) -> float:
        """Analyze measurement results to calculate success rate."""
        total_shots = sum(measurements.values())
        successful_shots = 0
        for bitstring, count in measurements.items():
            if target_function:
                is_target = target_function(bitstring)
            else:
                is_target = bitstring in self.target_items
            if is_target:
                successful_shots += count
        return successful_shots / total_shots if total_shots > 0 else 0.0
    def amplitude_amplification(self, initial_state: np.ndarray,
                              oracle_function: Callable[[np.ndarray], np.ndarray],
                              amplification_function: Callable[[np.ndarray], np.ndarray],
                              n_iterations: int) -> np.ndarray:
        """
        Generalized amplitude amplification algorithm.
        Parameters
        ----------
        initial_state : np.ndarray
            Initial quantum state
        oracle_function : callable
            Oracle operator
        amplification_function : callable
            Amplification operator
        n_iterations : int
            Number of amplification iterations
        Returns
        -------
        np.ndarray
            Final amplified state
        """
        current_state = initial_state.copy()
        for _ in range(n_iterations):
            # Apply oracle
            current_state = oracle_function(current_state)
            # Apply amplification operator
            current_state = amplification_function(current_state)
        return current_state
    def quantum_counting(self, target_function: Callable[[str], bool],
                        precision: int = 4) -> Tuple[int, float]:
        """
        Estimate number of solutions using quantum counting.
        Parameters
        ----------
        target_function : callable
            Function defining target states
        precision : int
            Number of precision qubits
        Returns
        -------
        Tuple[int, float]
            Estimated number of solutions and confidence
        """
        n_trials = 2**precision
        estimates = []
        for trial in range(n_trials):
            # Run Grover with different numbers of iterations
            n_iter = trial
            # Create test state
            test_state = self.create_uniform_superposition()
            # Apply Grover iterations
            for _ in range(n_iter):
                test_state = self.grover_iteration(test_state, target_function)
            # Estimate from success probability
            prob = self._calculate_target_probability(test_state, target_function)
            if prob > 0 and prob < 1:
                # Inverse formula: M = N * sin²(θ) where P = sin²((2k+1)θ)
                if n_iter == 0:
                    estimated_theta = np.arcsin(np.sqrt(prob))
                else:
                    estimated_theta = np.arcsin(np.sqrt(prob)) / (2 * n_iter + 1)
                estimated_solutions = self.n_states * np.sin(estimated_theta)**2
                estimates.append(int(np.round(estimated_solutions)))
        if estimates:
            # Most common estimate
            from collections import Counter
            counts = Counter(estimates)
            most_common = counts.most_common(1)[0]
            estimated_count = most_common[0]
            confidence = most_common[1] / len(estimates)
        else:
            estimated_count = 0
            confidence = 0.0
        return estimated_count, confidence
    def geometric_visualization(self) -> None:
        """Visualize Grover's algorithm in the geometric picture."""
        if not self.probability_history:
            print("Run search first to generate data for visualization")
            return
        berkeley_plot = BerkeleyPlot()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        # Plot success probability vs iterations
        iterations = range(len(self.probability_history))
        ax1.plot(iterations, self.probability_history,
                color=berkeley_plot.colors['berkeley_blue'],
                linewidth=2, marker='o', markersize=6)
        # Plot theoretical curve
        if self.n_solutions > 0:
            theory_iterations = np.linspace(0, len(iterations)-1, 100)
            theory_probs = [self.success_probability(int(k)) for k in theory_iterations]
            ax1.plot(theory_iterations, theory_probs, '--',
                    color=berkeley_plot.colors['california_gold'],
                    linewidth=2, label='Theoretical')
        # Mark optimal point
        optimal_iter = self.calculate_optimal_iterations()
        if optimal_iter < len(self.probability_history):
            ax1.axvline(optimal_iter, color='red', linestyle=':',
                       label=f'Optimal ({optimal_iter})')
        ax1.set_xlabel('Grover Iterations')
        ax1.set_ylabel('Success Probability')
        ax1.set_title('Success Probability vs Iterations')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        # Geometric representation (2D projection)
        if self.n_solutions > 0:
            theta = np.arcsin(np.sqrt(self.n_solutions / self.n_states))
            # Create unit circle
            angles = np.linspace(0, 2*np.pi, 100)
            circle_x = np.cos(angles)
            circle_y = np.sin(angles)
            ax2.plot(circle_x, circle_y, 'k--', alpha=0.3)
            # Plot state evolution
            for i, prob in enumerate(self.probability_history):
                angle = np.arcsin(np.sqrt(prob))
                x = np.sqrt(1 - prob)  # Amplitude of non-target states
                y = np.sqrt(prob)      # Amplitude of target states
                color = berkeley_plot.colors['berkeley_blue'] if i == 0 else berkeley_plot.colors['california_gold']
                marker = 'o' if i == 0 or i == len(self.probability_history)-1 else '.'
                size = 8 if marker == 'o' else 4
                ax2.plot(x, y, marker=marker, color=color, markersize=size)
                if i == 0:
                    ax2.annotate('Start', (x, y), xytext=(5, 5),
                               textcoords='offset points')
                elif i == len(self.probability_history) - 1:
                    ax2.annotate('Final', (x, y), xytext=(5, 5),
                               textcoords='offset points')
            # Mark target axis
            ax2.axhline(0, color='k', linewidth=0.5)
            ax2.axvline(0, color='k', linewidth=0.5)
            ax2.set_xlabel('Amplitude of Non-target States')
            ax2.set_ylabel('Amplitude of Target States')
            ax2.set_title('Geometric Representation')
            ax2.set_xlim(-0.1, 1.1)
            ax2.set_ylim(-0.1, 1.1)
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
        plt.tight_layout()
        plt.show()
    def benchmark_performance(self, problem_sizes: List[int],
                            solution_ratios: List[float]) -> Dict[str, List]:
        """
        Benchmark Grover's algorithm performance.
        Parameters
        ----------
        problem_sizes : List[int]
            List of problem sizes (n_qubits)
        solution_ratios : List[float]
            List of solution density ratios
        Returns
        -------
        Dict[str, List]
            Performance metrics
        """
        results = {
            'problem_sizes': [],
            'solution_ratios': [],
            'optimal_iterations': [],
            'speedup_factors': [],
            'success_probabilities': []
        }
        original_n_qubits = self.config.n_qubits
        for n_qubits in problem_sizes:
            for ratio in solution_ratios:
                # Update configuration
                self.config.n_qubits = n_qubits
                self.n_qubits = n_qubits
                self.n_states = 2**n_qubits
                self.basis_states = [format(i, f'0{n_qubits}b')
                                   for i in range(self.n_states)]
                # Calculate number of solutions
                n_solutions = max(1, int(ratio * self.n_states))
                self.n_solutions = n_solutions
                # Generate random target items
                import random
                all_states = list(range(self.n_states))
                target_indices = random.sample(all_states, n_solutions)
                self.target_items = {self.basis_states[i] for i in target_indices}
                # Calculate metrics
                optimal_k = self.calculate_optimal_iterations()
                classical_search = self.n_states // 2  # Expected classical steps
                speedup = classical_search / max(1, optimal_k)
                success_prob = self.success_probability(optimal_k)
                # Store results
                results['problem_sizes'].append(n_qubits)
                results['solution_ratios'].append(ratio)
                results['optimal_iterations'].append(optimal_k)
                results['speedup_factors'].append(speedup)
                results['success_probabilities'].append(success_prob)
                print(f"n={n_qubits}, ratio={ratio:.3f}: k_opt={optimal_k}, "
                      f"speedup={speedup:.1f}x, P_success={success_prob:.3f}")
        # Restore original configuration
        self.config.n_qubits = original_n_qubits
        self.n_qubits = original_n_qubits
        self.n_states = 2**original_n_qubits
        return results
# Example oracle functions
def satisfiability_oracle(bitstring: str, clauses: List[List[Tuple[int, bool]]]) -> bool:
    """
    Oracle for Boolean satisfiability problem.
    Parameters
    ----------
    bitstring : str
        Assignment of variables
    clauses : List[List[Tuple[int, bool]]]
        SAT clauses, each clause is list of (variable_index, negated) pairs
    Returns
    -------
    bool
        True if assignment satisfies all clauses
    """
    assignment = [bit == '1' for bit in bitstring]
    for clause in clauses:
        clause_satisfied = False
        for var_idx, negated in clause:
            if var_idx < len(assignment):
                var_value = not assignment[var_idx] if negated else assignment[var_idx]
                if var_value:
                    clause_satisfied = True
                    break
        if not clause_satisfied:
            return False
    return True
def pattern_matching_oracle(bitstring: str, pattern: str,
                           database: List[str]) -> bool:
    """
    Oracle for pattern matching in database.
    Parameters
    ----------
    bitstring : str
        Index into database (binary representation)
    pattern : str
        Pattern to search for
    database : List[str]
        Database of strings
    Returns
    -------
    bool
        True if database entry matches pattern
    """
    index = int(bitstring, 2)
    if index < len(database):
        return pattern in database[index]
    return False
if __name__ == "__main__":
    # Example: Search for specific bitstrings
    config = GroverConfig(
        n_qubits=4,
        target_items=['1010', '1100', '0011'],  # 3 target states
        n_shots=1024,
        use_optimal_iterations=True
    )
    grover = GroverSearch(config)
    print("Grover's Algorithm Demo")
    print(f"Search space: {grover.n_states} states ({grover.n_qubits} qubits)")
    print(f"Target states: {config.target_items}")
    print(f"Number of solutions: {grover.n_solutions}")
    # Run search
    results = grover.run_grover_search()
    print(f"\nResults:")
    print(f"Iterations used: {results['n_iterations']}")
    print(f"Optimal iterations: {results['optimal_iterations']}")
    print(f"Success rate: {results['success_rate']:.4f}")
    print(f"Theoretical success probability: {results['theoretical_success_prob']:.4f}")
    # Show most frequent measurements
    sorted_measurements = sorted(results['measurements'].items(),
                               key=lambda x: x[1], reverse=True)
    print(f"\nTop measurement results:")
    for bitstring, count in sorted_measurements[:5]:
        is_target = bitstring in config.target_items
        print(f"  {bitstring}: {count} counts {'✓' if is_target else ''}")
    # Visualize results
    grover.geometric_visualization()
    # Example: SAT problem
    print("\n" + "="*50)
    print("SAT Problem Example")
    # Simple 3-SAT instance: (x₀ ∨ ¬x₁ ∨ x₂) ∧ (¬x₀ ∨ x₁ ∨ ¬x₂) ∧ (x₀ ∨ x₁ ∨ x₂)
    clauses = [
        [(0, False), (1, True), (2, False)],   # x₀ ∨ ¬x₁ ∨ x₂
        [(0, True), (1, False), (2, True)],   # ¬x₀ ∨ x₁ ∨ ¬x₂
        [(0, False), (1, False), (2, False)]  # x₀ ∨ x₁ ∨ x₂
    ]
    def sat_oracle(bitstring):
        return satisfiability_oracle(bitstring, clauses)
    # Find satisfying assignments
    sat_config = GroverConfig(n_qubits=3, n_shots=1024)
    sat_grover = GroverSearch(sat_config)
    # First, find how many solutions exist by testing all
    satisfying_assignments = []
    for i in range(2**3):
        bitstring = format(i, '03b')
        if sat_oracle(bitstring):
            satisfying_assignments.append(bitstring)
    print(f"Satisfying assignments: {satisfying_assignments}")
    # Run Grover search for SAT
    sat_grover.n_solutions = len(satisfying_assignments)
    sat_results = sat_grover.run_grover_search(target_function=sat_oracle)
    print(f"SAT search success rate: {sat_results['success_rate']:.4f}")
    # Benchmark performance
    print("\n" + "="*50)
    print("Performance Benchmark")
    benchmark_results = grover.benchmark_performance(
        problem_sizes=[3, 4, 5],
        solution_ratios=[0.1, 0.25, 0.5]
    )
    print("Benchmark completed. See results above.")