"""
Hamiltonian Builder

Constructs quantum Hamiltonians for optimization problems.

Author: Meshal Alawein
Date: 2025-11-18
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class HamiltonianBuilder:
    """
    Builds quantum Hamiltonians for various optimization problems.

    Supports:
    - QUBO to Hamiltonian conversion
    - Ising model Hamiltonians
    - Pauli operator representations
    - Time-dependent Hamiltonians for annealing
    """

    def __init__(self):
        """Initialize Hamiltonian builder."""
        self.pauli_i = np.array([[1, 0], [0, 1]], dtype=complex)
        self.pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

        logger.info("Initialized HamiltonianBuilder")

    def build_ising_hamiltonian(
        self,
        h: np.ndarray,
        J: np.ndarray
    ) -> np.ndarray:
        """
        Build Ising Hamiltonian matrix.

        H = sum_i h_i σ_z^i + sum_{i<j} J_{ij} σ_z^i σ_z^j

        Args:
            h: Local field coefficients
            J: Coupling coefficients

        Returns:
            Hamiltonian matrix (2^n × 2^n)
        """
        n = len(h)
        dim = 2 ** n
        H = np.zeros((dim, dim), dtype=complex)

        # Add local field terms
        for i in range(n):
            sigma_z_i = self._pauli_z_on_qubit(i, n)
            H += h[i] * sigma_z_i

        # Add coupling terms
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0:
                    sigma_z_i = self._pauli_z_on_qubit(i, n)
                    sigma_z_j = self._pauli_z_on_qubit(j, n)
                    H += J[i, j] * (sigma_z_i @ sigma_z_j)

        logger.info(f"Built Ising Hamiltonian for {n} qubits")

        return H

    def _pauli_z_on_qubit(self, qubit: int, n_qubits: int) -> np.ndarray:
        """
        Create Pauli-Z operator on specific qubit.

        Args:
            qubit: Target qubit index
            n_qubits: Total number of qubits

        Returns:
            Full Pauli-Z operator matrix
        """
        ops = [self.pauli_i if i != qubit else self.pauli_z
               for i in range(n_qubits)]
        return self._tensor_product(ops)

    def _tensor_product(self, matrices: List[np.ndarray]) -> np.ndarray:
        """Compute tensor product of matrices."""
        result = matrices[0]
        for mat in matrices[1:]:
            result = np.kron(result, mat)
        return result

    def build_qaoa_hamiltonian(
        self,
        problem_hamiltonian: np.ndarray,
        mixing_hamiltonian: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build QAOA Hamiltonians (problem and mixing).

        Args:
            problem_hamiltonian: Problem Hamiltonian H_C
            mixing_hamiltonian: Mixing Hamiltonian H_B (default: sum of Pauli-X)

        Returns:
            Tuple of (problem_hamiltonian, mixing_hamiltonian)
        """
        n_qubits = int(np.log2(problem_hamiltonian.shape[0]))

        if mixing_hamiltonian is None:
            # Default mixing Hamiltonian: sum of Pauli-X
            mixing_hamiltonian = np.zeros_like(problem_hamiltonian)
            for i in range(n_qubits):
                ops = [self.pauli_i if j != i else self.pauli_x
                       for j in range(n_qubits)]
                mixing_hamiltonian += self._tensor_product(ops)

        logger.info(f"Built QAOA Hamiltonians for {n_qubits} qubits")

        return problem_hamiltonian, mixing_hamiltonian

    def build_transverse_field_hamiltonian(
        self,
        ising_h: np.ndarray,
        ising_J: np.ndarray,
        transverse_field: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build transverse field Ising model for quantum annealing.

        H(s) = A(s) H_transverse + B(s) H_problem

        Args:
            ising_h: Ising local fields
            ising_J: Ising couplings
            transverse_field: Strength of transverse field

        Returns:
            Tuple of (problem_hamiltonian, transverse_hamiltonian)
        """
        n = len(ising_h)

        # Build problem Hamiltonian
        H_problem = self.build_ising_hamiltonian(ising_h, ising_J)

        # Build transverse field Hamiltonian
        H_transverse = np.zeros((2**n, 2**n), dtype=complex)
        for i in range(n):
            ops = [self.pauli_i if j != i else self.pauli_x
                   for j in range(n)]
            H_transverse += transverse_field * self._tensor_product(ops)

        logger.info(f"Built transverse field Hamiltonian with field strength {transverse_field}")

        return H_problem, H_transverse

    def build_qubo_hamiltonian(
        self,
        Q: np.ndarray
    ) -> np.ndarray:
        """
        Build Hamiltonian from QUBO matrix.

        Uses transformation: x_i = (1 - σ_z^i) / 2

        Args:
            Q: QUBO matrix

        Returns:
            Hamiltonian matrix
        """
        n = Q.shape[0]

        # Convert QUBO to Ising
        h = np.zeros(n)
        J = np.zeros((n, n))

        # Diagonal terms
        for i in range(n):
            h[i] = -0.5 * Q[i, i]
            for j in range(n):
                if i != j:
                    h[i] -= 0.25 * Q[i, j]

        # Off-diagonal terms
        for i in range(n):
            for j in range(i + 1, n):
                J[i, j] = -0.25 * (Q[i, j] + Q[j, i])
                J[j, i] = J[i, j]

        # Build Hamiltonian
        H = self.build_ising_hamiltonian(h, J)

        # Add constant offset
        offset = 0.25 * np.sum(Q) + 0.5 * np.sum(np.diag(Q))
        H += offset * np.eye(2**n)

        logger.info(f"Built QUBO Hamiltonian for {n} variables")

        return H

    def get_pauli_decomposition(
        self,
        hamiltonian: np.ndarray,
        threshold: float = 1e-10
    ) -> List[Tuple[complex, str]]:
        """
        Decompose Hamiltonian into Pauli operators.

        Args:
            hamiltonian: Hamiltonian matrix
            threshold: Threshold for considering terms

        Returns:
            List of (coefficient, pauli_string) tuples

        TODO: Implement efficient Pauli decomposition algorithm
        """
        n_qubits = int(np.log2(hamiltonian.shape[0]))
        pauli_terms = []

        # Generate all Pauli strings
        pauli_ops = ['I', 'X', 'Y', 'Z']

        # This is a placeholder for full implementation
        # For now, return simplified decomposition
        logger.warning("Using simplified Pauli decomposition (full implementation TODO)")

        # Extract diagonal elements (Z terms)
        for i in range(2**n_qubits):
            coeff = hamiltonian[i, i]
            if np.abs(coeff) > threshold:
                # Convert index to Z-basis measurement
                bitstring = format(i, f'0{n_qubits}b')
                pauli_string = ''.join('I' if b == '0' else 'Z' for b in bitstring)
                pauli_terms.append((complex(coeff), pauli_string[::-1]))

        return pauli_terms

    def build_vqe_ansatz_hamiltonian(
        self,
        n_qubits: int,
        layers: int = 1,
        entanglement: str = 'linear'
    ) -> List[Dict[str, Any]]:
        """
        Build VQE ansatz circuit description.

        Args:
            n_qubits: Number of qubits
            layers: Number of ansatz layers
            entanglement: Entanglement pattern ('linear', 'circular', 'full')

        Returns:
            List of gate descriptions for ansatz

        TODO: Implement full VQE ansatz builder
        """
        ansatz = []

        for layer in range(layers):
            # Single-qubit rotation layer
            for qubit in range(n_qubits):
                ansatz.append({
                    'gate': 'RY',
                    'qubit': qubit,
                    'parameter': f'theta_{layer}_{qubit}_y',
                })
                ansatz.append({
                    'gate': 'RZ',
                    'qubit': qubit,
                    'parameter': f'theta_{layer}_{qubit}_z',
                })

            # Entanglement layer
            if entanglement == 'linear':
                for qubit in range(n_qubits - 1):
                    ansatz.append({
                        'gate': 'CNOT',
                        'control': qubit,
                        'target': qubit + 1,
                    })
            elif entanglement == 'circular':
                for qubit in range(n_qubits):
                    ansatz.append({
                        'gate': 'CNOT',
                        'control': qubit,
                        'target': (qubit + 1) % n_qubits,
                    })
            elif entanglement == 'full':
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        ansatz.append({
                            'gate': 'CNOT',
                            'control': i,
                            'target': j,
                        })

        num_parameters = 2 * n_qubits * layers
        logger.info(f"Built VQE ansatz with {layers} layers, "
                   f"{num_parameters} parameters, {entanglement} entanglement")

        return ansatz

    def annealing_schedule(
        self,
        t: float,
        total_time: float,
        schedule_type: str = 'linear'
    ) -> Tuple[float, float]:
        """
        Calculate annealing schedule coefficients.

        H(t) = A(t) * H_initial + B(t) * H_final

        Args:
            t: Current time
            total_time: Total annealing time
            schedule_type: Type of schedule ('linear', 'exponential', 'sigmoid')

        Returns:
            Tuple of (A(t), B(t)) coefficients
        """
        s = t / total_time  # Normalized time [0, 1]

        if schedule_type == 'linear':
            A = 1 - s
            B = s
        elif schedule_type == 'exponential':
            # Faster initial decrease
            A = np.exp(-5 * s)
            B = 1 - A
        elif schedule_type == 'sigmoid':
            # S-shaped curve
            k = 10  # Steepness
            A = 1 / (1 + np.exp(k * (s - 0.5)))
            B = 1 - A
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        return float(A), float(B)

    def estimate_ground_state_energy(
        self,
        hamiltonian: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Estimate ground state energy and state.

        Args:
            hamiltonian: Hamiltonian matrix

        Returns:
            Tuple of (ground_energy, ground_state)
        """
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        ground_energy = eigenvalues[0]
        ground_state = eigenvectors[:, 0]

        logger.info(f"Ground state energy: {ground_energy:.6f}")

        return float(np.real(ground_energy)), ground_state


__all__ = ['HamiltonianBuilder']