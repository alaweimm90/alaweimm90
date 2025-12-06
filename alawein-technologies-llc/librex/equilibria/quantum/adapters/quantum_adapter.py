"""
Quantum Problem Adapter

High-level adapter for converting classical optimization problems to quantum formulations
and managing the conversion pipeline.

Author: Meshal Alawein
Date: 2025-11-18
"""

import logging
from typing import Dict, Any, Optional, Union
import numpy as np

from Librex.core.interfaces import StandardizedProblem, StandardizedSolution
from Librex.quantum.adapters.qubo_converter import QUBOConverter, QUBOProblem
from Librex.quantum.adapters.ising_encoder import IsingEncoder, IsingProblem

logger = logging.getLogger(__name__)


class QuantumProblemAdapter:
    """
    Unified adapter for quantum problem conversion.

    This adapter provides a high-level interface for converting classical
    optimization problems to quantum formulations (QUBO/Ising) and back.
    """

    def __init__(
        self,
        penalty_weight: float = 1000.0,
        encoding: str = 'auto'
    ):
        """
        Initialize quantum problem adapter.

        Args:
            penalty_weight: Weight for constraint penalty terms
            encoding: Preferred encoding ('qubo', 'ising', or 'auto')
        """
        self.penalty_weight = penalty_weight
        self.encoding = encoding
        self.qubo_converter = QUBOConverter(penalty_weight=penalty_weight)
        self.ising_encoder = IsingEncoder()

        logger.info(f"Initialized QuantumProblemAdapter with encoding='{encoding}'")

    def convert_to_quantum(
        self,
        problem: Union[StandardizedProblem, Dict[str, Any]],
        target_format: Optional[str] = None
    ) -> Union[QUBOProblem, IsingProblem]:
        """
        Convert classical problem to quantum formulation.

        Args:
            problem: Classical optimization problem (StandardizedProblem or dict)
            target_format: Target format ('qubo', 'ising', or None for auto)

        Returns:
            Quantum problem in requested format

        Raises:
            ValueError: If problem type is not supported
        """
        # Determine target format
        if target_format is None:
            target_format = self.encoding if self.encoding != 'auto' else 'qubo'

        # Convert dict to StandardizedProblem if needed
        if isinstance(problem, dict):
            problem = self._dict_to_standardized(problem)

        # Get problem type
        problem_type = self._detect_problem_type(problem)
        logger.info(f"Converting {problem_type} to {target_format} format")

        # Convert to QUBO first
        if problem_type == 'QAP':
            qubo_problem = self._convert_qap(problem)
        elif problem_type == 'TSP':
            qubo_problem = self._convert_tsp(problem)
        elif problem_type == 'Max-Cut':
            qubo_problem = self._convert_maxcut(problem)
        else:
            qubo_problem = self.qubo_converter.convert_from_standardized(problem)

        # Convert to Ising if requested
        if target_format == 'ising':
            return self.ising_encoder.qubo_to_ising(qubo_problem)
        else:
            return qubo_problem

    def _detect_problem_type(self, problem: StandardizedProblem) -> str:
        """
        Detect problem type from StandardizedProblem.

        Args:
            problem: Standardized problem

        Returns:
            Problem type string
        """
        metadata = problem.problem_metadata or {}

        # Check explicit problem type
        if 'problem_type' in metadata:
            return metadata['problem_type']

        # Try to infer from metadata
        if 'flow_matrix' in metadata and 'distance_matrix' in metadata:
            return 'QAP'
        elif 'distance_matrix' in metadata or 'cities' in metadata:
            return 'TSP'
        elif 'adjacency_matrix' in metadata or 'graph' in metadata:
            return 'Max-Cut'
        else:
            return 'unknown'

    def _convert_qap(self, problem: StandardizedProblem) -> QUBOProblem:
        """Convert QAP to QUBO."""
        metadata = problem.problem_metadata or {}
        flow = metadata.get('flow_matrix')
        distance = metadata.get('distance_matrix')

        if flow is None or distance is None:
            # Try to extract from objective_matrix if structured appropriately
            if problem.objective_matrix is not None:
                n = int(np.sqrt(problem.dimension))
                # Assume objective_matrix encodes QAP directly
                logger.warning("Extracting QAP matrices from objective_matrix")
                flow = np.random.randn(n, n)  # Placeholder
                distance = problem.objective_matrix[:n, :n]

        return self.qubo_converter.convert_qap_to_qubo(flow, distance)

    def _convert_tsp(self, problem: StandardizedProblem) -> QUBOProblem:
        """Convert TSP to QUBO."""
        metadata = problem.problem_metadata or {}
        distance = metadata.get('distance_matrix', problem.objective_matrix)

        if distance is None:
            raise ValueError("TSP requires distance_matrix")

        return self.qubo_converter.convert_tsp_to_qubo(distance)

    def _convert_maxcut(self, problem: StandardizedProblem) -> QUBOProblem:
        """Convert Max-Cut to QUBO."""
        metadata = problem.problem_metadata or {}
        adjacency = metadata.get('adjacency_matrix', problem.objective_matrix)

        if adjacency is None:
            raise ValueError("Max-Cut requires adjacency_matrix")

        return self.qubo_converter.convert_maxcut_to_qubo(adjacency)

    def _dict_to_standardized(self, problem_dict: Dict[str, Any]) -> StandardizedProblem:
        """
        Convert dictionary problem representation to StandardizedProblem.

        Args:
            problem_dict: Problem as dictionary

        Returns:
            StandardizedProblem
        """
        return StandardizedProblem(
            dimension=problem_dict.get('dimension', 0),
            objective_matrix=problem_dict.get('objective_matrix'),
            objective_function=problem_dict.get('objective_function'),
            constraint_matrix=problem_dict.get('constraint_matrix'),
            problem_metadata=problem_dict.get('metadata', problem_dict)
        )

    def convert_from_quantum(
        self,
        quantum_solution: np.ndarray,
        quantum_problem: Union[QUBOProblem, IsingProblem],
        original_problem: Optional[StandardizedProblem] = None
    ) -> StandardizedSolution:
        """
        Convert quantum solution back to classical format.

        Args:
            quantum_solution: Solution from quantum solver (binary or spin)
            quantum_problem: Quantum problem that was solved
            original_problem: Original classical problem (optional)

        Returns:
            StandardizedSolution
        """
        # Determine solution type
        if isinstance(quantum_problem, IsingProblem):
            # Convert spin to binary
            binary_solution = self.ising_encoder.spin_to_binary(quantum_solution)
            decoded = self.ising_encoder.decode_ising_solution(
                quantum_solution, quantum_problem
            )
            objective_value = decoded.get('energy', 0.0)
        else:
            # QUBO solution is already binary
            binary_solution = quantum_solution
            decoded = self.qubo_converter.decode_qubo_solution(
                quantum_solution, quantum_problem
            )
            objective_value = decoded.get('objective', 0.0)

        # Extract problem-specific solution
        if 'assignment' in decoded:
            # QAP solution
            solution_vector = decoded['assignment']
        elif 'tour' in decoded:
            # TSP solution
            solution_vector = decoded['tour']
        elif 'partition' in decoded:
            # Max-Cut solution
            solution_vector = decoded['partition']
        else:
            solution_vector = binary_solution

        return StandardizedSolution(
            vector=solution_vector,
            objective_value=objective_value,
            is_valid=decoded.get('is_valid', True),
            metadata=decoded
        )

    def estimate_quantum_resources(
        self,
        problem: Union[StandardizedProblem, QUBOProblem, IsingProblem]
    ) -> Dict[str, Any]:
        """
        Estimate quantum resources required for solving the problem.

        Args:
            problem: Problem to analyze

        Returns:
            Dictionary with resource estimates
        """
        # Determine problem size
        if isinstance(problem, StandardizedProblem):
            if problem.problem_metadata and 'problem_type' in problem.problem_metadata:
                problem_type = problem.problem_metadata['problem_type']
                if problem_type in ['QAP', 'TSP']:
                    # n^2 binary variables for n-sized QAP/TSP
                    n = int(np.sqrt(problem.dimension))
                    num_qubits = n * n
                else:
                    num_qubits = problem.dimension
            else:
                num_qubits = problem.dimension
        elif isinstance(problem, QUBOProblem):
            num_qubits = problem.num_variables
        elif isinstance(problem, IsingProblem):
            num_qubits = problem.num_spins
        else:
            num_qubits = 0

        # Estimate circuit depth for QAOA
        qaoa_layers = max(3, min(10, num_qubits // 10))
        qaoa_depth = qaoa_layers * (2 * num_qubits + num_qubits * (num_qubits - 1) // 2)

        # Estimate for VQE
        vqe_parameters = 2 * qaoa_layers  # Typical for hardware-efficient ansatz

        # Check NISQ feasibility
        nisq_feasible = num_qubits <= 50  # Typical NISQ limit
        ideal_for_nisq = num_qubits <= 20  # Good performance expected

        return {
            'num_qubits': num_qubits,
            'nisq_feasible': nisq_feasible,
            'ideal_for_nisq': ideal_for_nisq,
            'qaoa_circuit_depth': qaoa_depth,
            'qaoa_layers_recommended': qaoa_layers,
            'vqe_parameters': vqe_parameters,
            'annealing_feasible': num_qubits <= 5000,  # D-Wave limit
            'gate_based_feasible': num_qubits <= 100,  # Near-term gate-based limit
            'estimated_runtime_factor': 2 ** (num_qubits / 10),  # Exponential scaling
            'classical_simulation_limit': num_qubits <= 30,
        }

    def validate_quantum_problem(
        self,
        quantum_problem: Union[QUBOProblem, IsingProblem]
    ) -> Dict[str, Any]:
        """
        Validate quantum problem formulation.

        Args:
            quantum_problem: Quantum problem to validate

        Returns:
            Validation results
        """
        issues = []
        warnings = []

        if isinstance(quantum_problem, QUBOProblem):
            Q = quantum_problem.Q
            # Check symmetry
            if not np.allclose(Q, Q.T):
                issues.append("QUBO matrix is not symmetric")
            # Check for very large coefficients
            max_coeff = np.max(np.abs(Q))
            if max_coeff > 1e6:
                warnings.append(f"Very large coefficients detected (max={max_coeff:.2e})")
            # Check sparsity
            sparsity = 1 - np.count_nonzero(Q) / Q.size
            if sparsity > 0.9:
                warnings.append(f"QUBO matrix is very sparse ({sparsity:.1%})")

        elif isinstance(quantum_problem, IsingProblem):
            h = quantum_problem.h
            J = quantum_problem.J
            # Check J symmetry
            if not np.allclose(J, J.T):
                issues.append("Ising coupling matrix is not symmetric")
            # Check diagonal
            if np.any(np.diag(J) != 0):
                issues.append("Ising coupling matrix has non-zero diagonal")
            # Check coefficient ranges
            max_h = np.max(np.abs(h))
            max_J = np.max(np.abs(J))
            if max_h > 1e3 or max_J > 1e3:
                warnings.append(f"Large Ising coefficients (max_h={max_h:.2e}, max_J={max_J:.2e})")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'problem_type': type(quantum_problem).__name__,
        }


__all__ = ['QuantumProblemAdapter']