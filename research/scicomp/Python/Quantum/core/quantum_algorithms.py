"""
Quantum algorithms implementation.
This module provides implementations of fundamental quantum algorithms:
- Quantum Fourier Transform
- Phase estimation
- Amplitude amplification
- Quantum walks
"""
import numpy as np
from typing import Union, List, Tuple, Optional, Callable
from scipy.linalg import expm
import scipy.sparse as sp
class QuantumFourierTransform:
    """Quantum Fourier Transform and related algorithms."""
    @staticmethod
    def qft_matrix(n_qubits: int) -> np.ndarray:
        """
        Generate QFT matrix for n qubits.
        Args:
            n_qubits: Number of qubits
        Returns:
            QFT matrix
        """
        N = 2**n_qubits
        omega = np.exp(2j * np.pi / N)
        # Create QFT matrix
        qft = np.zeros((N, N), dtype=complex)
        for j in range(N):
            for k in range(N):
                qft[j, k] = omega**(j * k) / np.sqrt(N)
        return qft
    @staticmethod
    def inverse_qft_matrix(n_qubits: int) -> np.ndarray:
        """
        Generate inverse QFT matrix.
        Args:
            n_qubits: Number of qubits
        Returns:
            Inverse QFT matrix
        """
        return QuantumFourierTransform.qft_matrix(n_qubits).conj().T
    @staticmethod
    def qft_circuit(state: np.ndarray, n_qubits: int) -> np.ndarray:
        """
        Apply QFT using circuit decomposition.
        Args:
            state: Input state vector
            n_qubits: Number of qubits
        Returns:
            Transformed state
        """
        N = 2**n_qubits
        state = state.copy()
        # Apply Hadamard and controlled rotation gates
        for j in range(n_qubits):
            # Apply Hadamard to qubit j
            state = QuantumFourierTransform._apply_hadamard(state, j, n_qubits)
            # Apply controlled rotations
            for k in range(j + 1, n_qubits):
                angle = 2 * np.pi / (2**(k - j + 1))
                state = QuantumFourierTransform._apply_controlled_phase(
                    state, k, j, angle, n_qubits
                )
        # Swap qubits
        for i in range(n_qubits // 2):
            state = QuantumFourierTransform._swap_qubits(
                state, i, n_qubits - i - 1, n_qubits
            )
        return state
    @staticmethod
    def _apply_hadamard(state: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
        """Apply Hadamard gate to specific qubit."""
        N = 2**n_qubits
        new_state = np.zeros_like(state)
        for i in range(N):
            bit = (i >> (n_qubits - qubit - 1)) & 1
            if bit == 0:
                j = i | (1 << (n_qubits - qubit - 1))
                new_state[i] += (state[i] + state[j]) / np.sqrt(2)
                new_state[j] += (state[i] - state[j]) / np.sqrt(2)
        return new_state
    @staticmethod
    def _apply_controlled_phase(state: np.ndarray, control: int, target: int,
                               angle: float, n_qubits: int) -> np.ndarray:
        """Apply controlled phase rotation."""
        N = 2**n_qubits
        phase = np.exp(1j * angle)
        for i in range(N):
            control_bit = (i >> (n_qubits - control - 1)) & 1
            target_bit = (i >> (n_qubits - target - 1)) & 1
            if control_bit == 1 and target_bit == 1:
                state[i] *= phase
        return state
    @staticmethod
    def _swap_qubits(state: np.ndarray, q1: int, q2: int, n_qubits: int) -> np.ndarray:
        """Swap two qubits in state vector."""
        N = 2**n_qubits
        new_state = state.copy()
        for i in range(N):
            bit1 = (i >> (n_qubits - q1 - 1)) & 1
            bit2 = (i >> (n_qubits - q2 - 1)) & 1
            if bit1 != bit2:
                # Swap bits
                j = i ^ (1 << (n_qubits - q1 - 1)) ^ (1 << (n_qubits - q2 - 1))
                new_state[j] = state[i]
        return new_state
class PhaseEstimation:
    """Quantum phase estimation algorithm."""
    @staticmethod
    def estimate_phase(unitary: np.ndarray, eigenstate: np.ndarray,
                      n_precision: int) -> float:
        """
        Estimate phase of eigenvalue using QPE.
        Args:
            unitary: Unitary operator
            eigenstate: Eigenstate of unitary
            n_precision: Number of precision qubits
        Returns:
            Estimated phase in [0, 1)
        """
        n_work = int(np.log2(len(eigenstate)))
        n_total = n_precision + n_work
        # Initialize state |0⟩^⊗n_precision ⊗ |eigenstate⟩
        state = np.zeros(2**n_total, dtype=complex)
        for i in range(len(eigenstate)):
            state[i] = eigenstate[i]
        # Apply Hadamard to precision qubits
        for j in range(n_precision):
            state = PhaseEstimation._apply_hadamard_to_qubit(state, j, n_total)
        # Apply controlled-U operations
        for j in range(n_precision):
            power = 2**(n_precision - j - 1)
            U_power = np.linalg.matrix_power(unitary, power)
            state = PhaseEstimation._apply_controlled_unitary(
                state, j, U_power, n_precision, n_work
            )
        # Apply inverse QFT to precision qubits
        state = PhaseEstimation._apply_inverse_qft_partial(state, n_precision, n_total)
        # Measure precision qubits
        probabilities = np.abs(state)**2
        probabilities = probabilities.reshape(2**n_precision, 2**n_work)
        measurement_probs = np.sum(probabilities, axis=1)
        # Find most likely phase
        measured_value = np.argmax(measurement_probs)
        phase = measured_value / (2**n_precision)
        return phase
    @staticmethod
    def _apply_hadamard_to_qubit(state: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
        """Apply Hadamard to specific qubit."""
        N = 2**n_qubits
        new_state = state.copy()
        for i in range(N):
            if (i >> (n_qubits - qubit - 1)) & 1 == 0:
                j = i | (1 << (n_qubits - qubit - 1))
                temp = new_state[i]
                new_state[i] = (new_state[i] + new_state[j]) / np.sqrt(2)
                new_state[j] = (temp - new_state[j]) / np.sqrt(2)
        return new_state
    @staticmethod
    def _apply_controlled_unitary(state: np.ndarray, control: int, U: np.ndarray,
                                 n_precision: int, n_work: int) -> np.ndarray:
        """Apply controlled unitary operation."""
        n_total = n_precision + n_work
        N = 2**n_total
        new_state = state.copy()
        for i in range(N):
            if (i >> (n_total - control - 1)) & 1 == 1:
                # Control qubit is |1⟩, apply U to work qubits
                work_indices = []
                for j in range(2**n_work):
                    idx = (i & ~((2**n_work - 1))) | j
                    work_indices.append(idx)
                work_state = state[work_indices]
                work_state = U @ work_state
                for j, idx in enumerate(work_indices):
                    new_state[idx] = work_state[j]
        return new_state
    @staticmethod
    def _apply_inverse_qft_partial(state: np.ndarray, n_precision: int,
                                  n_total: int) -> np.ndarray:
        """Apply inverse QFT to precision qubits only."""
        # Create inverse QFT matrix for precision qubits
        iqft = QuantumFourierTransform.inverse_qft_matrix(n_precision)
        # Reshape state to separate precision and work qubits
        state_reshaped = state.reshape(2**n_precision, 2**(n_total - n_precision))
        # Apply inverse QFT to precision qubits
        state_reshaped = iqft @ state_reshaped
        return state_reshaped.flatten()
class AmplitudeAmplification:
    """Grover's algorithm and amplitude amplification."""
    @staticmethod
    def grover_operator(oracle: np.ndarray, n_qubits: int) -> np.ndarray:
        """
        Construct Grover operator G = -AS₀A†O.
        Args:
            oracle: Oracle matrix marking target states
            n_qubits: Number of qubits
        Returns:
            Grover operator
        """
        N = 2**n_qubits
        # Create uniform superposition state
        s = np.ones(N) / np.sqrt(N)
        # Inversion about average operator
        A = 2 * np.outer(s, s) - np.eye(N)
        # Grover operator
        G = A @ oracle
        return G
    @staticmethod
    def grover_search(oracle: Callable, n_qubits: int,
                     n_iterations: Optional[int] = None) -> int:
        """
        Perform Grover's search algorithm.
        Args:
            oracle: Oracle function marking target items
            n_qubits: Number of qubits
            n_iterations: Number of Grover iterations (auto if None)
        Returns:
            Index of found item
        """
        N = 2**n_qubits
        # Initialize uniform superposition
        state = np.ones(N) / np.sqrt(N)
        # Estimate number of marked items if not provided
        if n_iterations is None:
            # Estimate using random sampling
            n_samples = min(100, N // 4)
            marked_count = sum(oracle(i) for i in np.random.randint(0, N, n_samples))
            M = max(1, int(marked_count * N / n_samples))
            n_iterations = int(np.pi / 4 * np.sqrt(N / M))
        # Apply Grover iterations
        for _ in range(n_iterations):
            # Oracle
            for i in range(N):
                if oracle(i):
                    state[i] *= -1
            # Inversion about average
            avg = np.mean(state)
            state = 2 * avg - state
        # Measure
        probabilities = np.abs(state)**2
        measured = np.random.choice(N, p=probabilities)
        return measured
    @staticmethod
    def amplitude_estimation(oracle: np.ndarray, n_qubits: int,
                           n_precision: int) -> float:
        """
        Estimate amplitude of marked states.
        Args:
            oracle: Oracle matrix
            n_qubits: Number of work qubits
            n_precision: Number of precision qubits
        Returns:
            Estimated amplitude
        """
        N = 2**n_qubits
        # Create Grover operator
        G = AmplitudeAmplification.grover_operator(oracle, n_qubits)
        # Initial state (uniform superposition)
        initial = np.ones(N) / np.sqrt(N)
        # Use phase estimation to find eigenvalues
        phase = PhaseEstimation.estimate_phase(G, initial, n_precision)
        # Convert phase to amplitude
        theta = 2 * np.arcsin(np.sqrt(phase))
        amplitude = np.sin(theta / 2)
        return amplitude
class QuantumWalk:
    """Quantum walk algorithms."""
    @staticmethod
    def discrete_walk_line(n_steps: int, n_positions: int,
                          coin_operator: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Discrete quantum walk on a line.
        Args:
            n_steps: Number of walk steps
            n_positions: Number of positions
            coin_operator: 2x2 coin operator (Hadamard if None)
        Returns:
            Final probability distribution
        """
        if coin_operator is None:
            # Hadamard coin
            coin_operator = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        # Initialize state at center position
        center = n_positions // 2
        state = np.zeros((n_positions, 2), dtype=complex)
        state[center, 0] = 1 / np.sqrt(2)
        state[center, 1] = 1 / np.sqrt(2)
        for _ in range(n_steps):
            # Apply coin operator
            for pos in range(n_positions):
                state[pos] = coin_operator @ state[pos]
            # Shift operator
            new_state = np.zeros_like(state)
            for pos in range(n_positions):
                # Move left for |0⟩ coin state
                if pos > 0:
                    new_state[pos - 1, 0] += state[pos, 0]
                # Move right for |1⟩ coin state
                if pos < n_positions - 1:
                    new_state[pos + 1, 1] += state[pos, 1]
            state = new_state
        # Calculate probability distribution
        probability = np.sum(np.abs(state)**2, axis=1)
        return probability
    @staticmethod
    def continuous_walk_graph(adjacency: np.ndarray, time: float,
                            initial_node: int) -> np.ndarray:
        """
        Continuous-time quantum walk on graph.
        Args:
            adjacency: Adjacency matrix of graph
            time: Evolution time
            initial_node: Starting node
        Returns:
            Probability distribution over nodes
        """
        n_nodes = adjacency.shape[0]
        # Graph Laplacian (Hamiltonian)
        degree = np.sum(adjacency, axis=1)
        laplacian = np.diag(degree) - adjacency
        # Time evolution
        U = expm(-1j * laplacian * time)
        # Initial state
        state = np.zeros(n_nodes, dtype=complex)
        state[initial_node] = 1
        # Evolve state
        final_state = U @ state
        # Probability distribution
        probability = np.abs(final_state)**2
        return probability
    @staticmethod
    def szegedy_walk(transition_matrix: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Szegedy quantum walk.
        Args:
            transition_matrix: Classical random walk transition matrix
            n_steps: Number of walk steps
        Returns:
            Quantum walk operator
        """
        n = transition_matrix.shape[0]
        # Create bipartite graph representation
        # Quantum state space is n² dimensional
        walk_operator = np.zeros((n*n, n*n), dtype=complex)
        # Build reflection operators
        for i in range(n):
            for j in range(n):
                idx1 = i * n + j
                # First reflection
                for k in range(n):
                    idx2 = i * n + k
                    walk_operator[idx1, idx2] += (
                        2 * np.sqrt(transition_matrix[i, j] * transition_matrix[i, k])
                    )
                if i == j:
                    walk_operator[idx1, idx1] -= 1
        # Apply second reflection
        for i in range(n):
            for j in range(n):
                idx1 = i * n + j
                for k in range(n):
                    idx2 = k * n + j
                    walk_operator[idx1, idx2] += (
                        2 * np.sqrt(transition_matrix[i, j] * transition_matrix[k, j])
                    )
                if i == j:
                    walk_operator[idx1, idx1] -= 1
        # Apply walk operator n_steps times
        result = np.linalg.matrix_power(walk_operator, n_steps)
        return result