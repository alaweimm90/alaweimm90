"""
Ising Model Encoder

Converts optimization problems to Ising Hamiltonian formulation suitable for
quantum annealing and variational quantum algorithms.

Mathematical Background:
The Ising model has the form:
    H = sum_i h_i σ_i + sum_{i<j} J_{ij} σ_i σ_j
where σ_i ∈ {-1, +1} are spin variables.

Relationship to QUBO:
    x_i = (σ_i + 1) / 2  (maps spin to binary)
    σ_i = 2x_i - 1       (maps binary to spin)

Author: Meshal Alawein
Date: 2025-11-18
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from Librex.quantum.adapters.qubo_converter import QUBOProblem

logger = logging.getLogger(__name__)


@dataclass
class IsingProblem:
    """
    Ising model problem representation.

    Attributes:
        h: Linear coefficients (local fields)
        J: Quadratic coefficients (couplings)
        offset: Constant energy offset
        num_spins: Number of spin variables
        metadata: Additional problem metadata
    """
    h: np.ndarray  # Linear terms (local fields)
    J: np.ndarray  # Quadratic terms (couplings)
    offset: float = 0.0
    num_spins: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.num_spins is None:
            self.num_spins = len(self.h)
        # Ensure J has zero diagonal (no self-interaction)
        np.fill_diagonal(self.J, 0)
        # Make J symmetric
        self.J = (self.J + self.J.T) / 2

    def energy(self, spins: np.ndarray) -> float:
        """
        Calculate Ising energy for a spin configuration.

        Args:
            spins: Array of spin values in {-1, +1}

        Returns:
            Energy value
        """
        linear_energy = np.dot(self.h, spins)
        quadratic_energy = np.dot(spins, np.dot(self.J, spins))
        return float(linear_energy + quadratic_energy + self.offset)


class IsingEncoder:
    """
    Encodes optimization problems in Ising model formulation.

    The Ising model is particularly useful for:
    - Quantum annealing (D-Wave)
    - QAOA implementations
    - VQE for optimization
    - Physics-inspired optimization algorithms
    """

    def __init__(self):
        """Initialize Ising encoder."""
        logger.info("Initialized Ising encoder")

    def qubo_to_ising(self, qubo_problem: QUBOProblem) -> IsingProblem:
        """
        Convert QUBO problem to Ising model.

        The conversion uses the transformation:
            x_i = (σ_i + 1) / 2

        This gives:
            x^T Q x = (1/4) σ^T J σ + (1/2) h^T σ + offset

        Args:
            qubo_problem: QUBO problem to convert

        Returns:
            IsingProblem: Equivalent Ising model
        """
        Q = qubo_problem.Q
        n = Q.shape[0]

        # Initialize Ising parameters
        h = np.zeros(n)
        J = np.zeros((n, n))

        # Convert QUBO to Ising
        # Q[i,i] terms contribute to h[i]
        # Q[i,j] terms (i != j) contribute to J[i,j]

        # Diagonal terms contribute to linear coefficients
        for i in range(n):
            h[i] = 0.5 * Q[i, i]
            for j in range(n):
                if i != j:
                    h[i] += 0.25 * Q[i, j]

        # Off-diagonal terms contribute to couplings
        for i in range(n):
            for j in range(i + 1, n):
                J[i, j] = 0.25 * (Q[i, j] + Q[j, i])
                J[j, i] = J[i, j]

        # Calculate offset
        offset = qubo_problem.offset
        offset += 0.25 * np.sum(Q)  # From the transformation

        metadata = qubo_problem.metadata.copy() if qubo_problem.metadata else {}
        metadata['converted_from'] = 'QUBO'

        logger.info(f"Converted QUBO (n={n}) to Ising model")

        return IsingProblem(h=h, J=J, offset=offset, metadata=metadata)

    def ising_to_qubo(self, ising_problem: IsingProblem) -> QUBOProblem:
        """
        Convert Ising model to QUBO problem.

        The conversion uses the transformation:
            σ_i = 2x_i - 1

        Args:
            ising_problem: Ising problem to convert

        Returns:
            QUBOProblem: Equivalent QUBO problem
        """
        h = ising_problem.h
        J = ising_problem.J
        n = len(h)

        # Initialize QUBO matrix
        Q = np.zeros((n, n))

        # Convert linear terms
        for i in range(n):
            Q[i, i] = -2 * h[i]
            for j in range(n):
                if i != j:
                    Q[i, i] -= J[i, j]

        # Convert quadratic terms
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] = 2 * J[i, j]
                Q[j, i] = Q[i, j]

        # Calculate offset
        offset = ising_problem.offset
        offset += np.sum(h) + np.sum(J)

        metadata = ising_problem.metadata.copy() if ising_problem.metadata else {}
        metadata['converted_from'] = 'Ising'

        logger.info(f"Converted Ising model (n={n}) to QUBO")

        return QUBOProblem(Q=Q, offset=offset, metadata=metadata)

    def encode_qap_ising(
        self,
        flow_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        penalty_weight: float = 1000.0
    ) -> IsingProblem:
        """
        Direct encoding of QAP as Ising model.

        This provides a more natural encoding for quantum systems than going
        through QUBO intermediate representation.

        Args:
            flow_matrix: Flow matrix between facilities
            distance_matrix: Distance matrix between locations
            penalty_weight: Weight for constraint penalties

        Returns:
            IsingProblem: Ising model for QAP

        TODO: Implement direct Ising encoding without QUBO intermediate
        """
        # For now, use QUBO as intermediate
        from Librex.quantum.adapters.qubo_converter import QUBOConverter

        qubo_converter = QUBOConverter(penalty_weight=penalty_weight)
        qubo_problem = qubo_converter.convert_qap_to_qubo(flow_matrix, distance_matrix)
        ising_problem = self.qubo_to_ising(qubo_problem)

        # Update metadata
        ising_problem.metadata['problem_type'] = 'QAP'
        ising_problem.metadata['encoding'] = 'direct_ising'

        return ising_problem

    def encode_tsp_ising(
        self,
        distance_matrix: np.ndarray,
        penalty_weight: float = 1000.0
    ) -> IsingProblem:
        """
        Direct encoding of TSP as Ising model.

        Args:
            distance_matrix: Distance matrix between cities
            penalty_weight: Weight for constraint penalties

        Returns:
            IsingProblem: Ising model for TSP

        TODO: Implement direct Ising encoding with improved constraint handling
        """
        from Librex.quantum.adapters.qubo_converter import QUBOConverter

        qubo_converter = QUBOConverter(penalty_weight=penalty_weight)
        qubo_problem = qubo_converter.convert_tsp_to_qubo(distance_matrix)
        ising_problem = self.qubo_to_ising(qubo_problem)

        ising_problem.metadata['problem_type'] = 'TSP'
        ising_problem.metadata['encoding'] = 'direct_ising'

        return ising_problem

    def encode_maxcut_ising(
        self,
        adjacency_matrix: np.ndarray
    ) -> IsingProblem:
        """
        Direct encoding of Max-Cut as Ising model.

        Max-Cut maps naturally to Ising: maximize sum of w[i,j] * (1 - σ_i * σ_j) / 2

        Args:
            adjacency_matrix: Graph adjacency matrix

        Returns:
            IsingProblem: Ising model for Max-Cut
        """
        n = adjacency_matrix.shape[0]
        h = np.zeros(n)
        J = np.zeros((n, n))

        # Max-Cut Ising: H = -sum_{(i,j) in E} w[i,j] * σ_i * σ_j
        # We minimize energy, so negative sign for maximization
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency_matrix[i, j] != 0:
                    J[i, j] = -adjacency_matrix[i, j] / 2
                    J[j, i] = J[i, j]

        metadata = {
            'problem_type': 'Max-Cut',
            'n_vertices': n,
            'n_edges': np.count_nonzero(adjacency_matrix) // 2,
            'encoding': 'direct_ising',
        }

        logger.info(f"Encoded Max-Cut (n={n}) as Ising model")

        return IsingProblem(h=h, J=J, offset=0.0, metadata=metadata)

    def spin_to_binary(self, spins: np.ndarray) -> np.ndarray:
        """
        Convert spin configuration {-1, +1} to binary {0, 1}.

        Args:
            spins: Spin configuration

        Returns:
            Binary configuration
        """
        return ((spins + 1) / 2).astype(int)

    def binary_to_spin(self, binary: np.ndarray) -> np.ndarray:
        """
        Convert binary configuration {0, 1} to spin {-1, +1}.

        Args:
            binary: Binary configuration

        Returns:
            Spin configuration
        """
        return 2 * binary - 1

    def decode_ising_solution(
        self,
        spins: np.ndarray,
        ising_problem: IsingProblem
    ) -> Dict[str, Any]:
        """
        Decode Ising solution to problem-specific format.

        Args:
            spins: Spin configuration {-1, +1}
            ising_problem: Original Ising problem

        Returns:
            Decoded solution dictionary
        """
        metadata = ising_problem.metadata or {}
        problem_type = metadata.get('problem_type', 'unknown')

        # Convert to binary for decoding
        binary = self.spin_to_binary(spins)

        if problem_type == 'QAP':
            n = int(np.sqrt(len(spins)))
            assignment = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(n):
                    if binary[i * n + j] > 0.5:
                        assignment[i] = j
            return {
                'assignment': assignment,
                'energy': ising_problem.energy(spins),
                'is_valid': len(np.unique(assignment)) == n,
            }
        elif problem_type == 'TSP':
            n = int(np.sqrt(len(spins)))
            tour = np.zeros(n, dtype=int)
            for t in range(n):
                for i in range(n):
                    if binary[i * n + t] > 0.5:
                        tour[t] = i
            return {
                'tour': tour,
                'energy': ising_problem.energy(spins),
                'is_valid': len(np.unique(tour)) == n,
            }
        elif problem_type == 'Max-Cut':
            partition = (spins > 0).astype(int)
            return {
                'partition': partition,
                'cut_value': -ising_problem.energy(spins),  # Convert to maximization
                'is_valid': True,
            }
        else:
            return {
                'spins': spins,
                'binary': binary,
                'energy': ising_problem.energy(spins),
                'is_valid': True,
            }

    def create_transverse_field_hamiltonian(
        self,
        ising_problem: IsingProblem,
        transverse_field_strength: float = 1.0
    ) -> Tuple[IsingProblem, np.ndarray]:
        """
        Create transverse field Ising Hamiltonian for quantum annealing.

        H_total = A(t) * H_transverse + B(t) * H_problem

        Args:
            ising_problem: Problem Hamiltonian
            transverse_field_strength: Strength of transverse field

        Returns:
            Tuple of (problem_hamiltonian, transverse_field_terms)

        TODO: Implement time-dependent annealing schedule
        """
        n = ising_problem.num_spins
        transverse_field = np.ones(n) * transverse_field_strength

        logger.info(f"Created transverse field Hamiltonian with strength {transverse_field_strength}")

        return ising_problem, transverse_field


__all__ = ['IsingEncoder', 'IsingProblem']