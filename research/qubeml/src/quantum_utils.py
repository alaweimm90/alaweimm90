"""
Quantum state operations and measurements.

Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT License © 2025
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import warnings


def create_bell_state(state_type: str = "phi_plus") -> np.ndarray:
    """Bell states for entanglement tests.
    
    phi_plus = (|00> + |11>)/√2, phi_minus = (|00> - |11>)/√2
    psi_plus = (|01> + |10>)/√2, psi_minus = (|01> - |10>)/√2
    """
    bell_states = {
        "phi_plus": np.array([1, 0, 0, 1]) / np.sqrt(2),
        "phi_minus": np.array([1, 0, 0, -1]) / np.sqrt(2),
        "psi_plus": np.array([0, 1, 1, 0]) / np.sqrt(2),
        "psi_minus": np.array([0, 1, -1, 0]) / np.sqrt(2),
    }
    
    if state_type not in bell_states:
        raise ValueError(f"Unknown Bell state: {state_type}")
    
    return bell_states[state_type]


def pauli_matrices() -> dict:
    """
    Return the Pauli matrices.
    
    Returns:
        Dictionary containing Pauli matrices I, X, Y, Z
    """
    return {
        "I": np.array([[1, 0], [0, 1]], dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }


def state_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Calculate the fidelity between two quantum states.
    
    Args:
        state1: First quantum state vector
        state2: Second quantum state vector
    
    Returns:
        Fidelity between the states (0 to 1)
    """
    state1 = state1.flatten()
    state2 = state2.flatten()
    
    # Normalize states
    state1 = state1 / np.linalg.norm(state1)
    state2 = state2 / np.linalg.norm(state2)
    
    # Calculate fidelity
    fidelity = np.abs(np.vdot(state1, state2)) ** 2
    
    return float(fidelity)


def create_ghz_state(n_qubits: int) -> np.ndarray:
    """
    Create a GHZ (Greenberger–Horne–Zeilinger) state.
    
    Args:
        n_qubits: Number of qubits
    
    Returns:
        GHZ state as a numpy array
    """
    if n_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits")
    
    dim = 2**n_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0 / np.sqrt(2)  # |000...0>
    state[-1] = 1.0 / np.sqrt(2)  # |111...1>
    
    return state


def measure_state(state: np.ndarray, n_shots: int = 1000) -> dict:
    """
    Simulate measurement of a quantum state.
    
    Args:
        state: Quantum state vector
        n_shots: Number of measurement shots
    
    Returns:
        Dictionary with measurement counts
    """
    state = state.flatten()
    n_qubits = int(np.log2(len(state)))
    
    # Calculate probabilities
    probs = np.abs(state) ** 2
    probs = probs / np.sum(probs)  # Normalize
    
    # Sample measurements
    outcomes = np.random.choice(len(state), size=n_shots, p=probs)
    
    # Convert to binary strings and count
    counts = {}
    for outcome in outcomes:
        binary = format(outcome, f"0{n_qubits}b")
        counts[binary] = counts.get(binary, 0) + 1
    
    return counts


def calculate_entanglement_entropy(state: np.ndarray, partition: List[int]) -> float:
    """
    Calculate the entanglement entropy of a bipartite system.
    
    Args:
        state: Quantum state vector
        partition: List of qubit indices for the first partition
    
    Returns:
        Von Neumann entropy of the reduced density matrix
    """
    state = state.flatten()
    n_qubits = int(np.log2(len(state)))
    
    # Reshape state to matrix form
    state_matrix = state.reshape([2] * n_qubits)
    
    # Create axes for partial trace
    axes_to_trace = [i for i in range(n_qubits) if i not in partition]
    
    if not axes_to_trace:
        warnings.warn("No qubits to trace out, returning 0")
        return 0.0
    
    # Calculate reduced density matrix
    rho = np.tensordot(state_matrix, state_matrix.conj(), axes=(axes_to_trace, axes_to_trace))
    
    # Reshape to square matrix
    dim = 2 ** len(partition)
    rho = rho.reshape(dim, dim)
    
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros
    
    # Calculate von Neumann entropy
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    
    return float(entropy)


def apply_noise(state: np.ndarray, noise_prob: float = 0.01, noise_type: str = "depolarizing") -> np.ndarray:
    """
    Apply noise to a quantum state.
    
    Args:
        state: Quantum state vector
        noise_prob: Probability of noise
        noise_type: Type of noise ('depolarizing', 'phase_flip', 'bit_flip')
    
    Returns:
        Noisy state vector
    """
    state = state.copy()
    n_qubits = int(np.log2(len(state)))
    
    if noise_type == "depolarizing":
        # Mix with maximally mixed state
        mixed_state = np.ones_like(state) / len(state)
        state = (1 - noise_prob) * state + noise_prob * mixed_state
        
    elif noise_type == "phase_flip":
        # Random phase flips
        for i in range(len(state)):
            if np.random.random() < noise_prob:
                state[i] *= -1
                
    elif noise_type == "bit_flip":
        # Random bit flips in computational basis
        for i in range(len(state)):
            if np.random.random() < noise_prob:
                # Flip one random bit in the binary representation
                bit_to_flip = np.random.randint(n_qubits)
                flipped_index = i ^ (1 << bit_to_flip)
                state[i], state[flipped_index] = state[flipped_index], state[i]
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Renormalize
    state = state / np.linalg.norm(state)
    
    return state


def quantum_fourier_transform(n_qubits: int) -> np.ndarray:
    """
    Create quantum Fourier transform circuit.
    
    TODO: Implement QFT for educational purposes
    This will be useful for quantum algorithms like Shor's algorithm
    """
    # TODO: Implement QFT matrix construction
    # Reference: Nielsen & Chuang Chapter 5
    # Need to check phase factors: omega = exp(2πi/2^k)
    pass


# NOTE: Look up VQE implementation for molecular systems
# Useful papers:
# - Peruzzo et al. (2014) - Original VQE paper
# - McClean et al. (2016) - Theory of variational quantum simulation
# BOOKMARK: https://arxiv.org/abs/1304.3061


def simulate_quantum_walk(steps: int) -> np.ndarray:
    """
    Simulate quantum walk on a line.
    
    TODO: Add quantum walk simulation for educational examples
    """
    # Placeholder implementation
    positions = np.arange(-steps, steps+1)
    # TODO: Calculate probability distribution after quantum walk
    return positions