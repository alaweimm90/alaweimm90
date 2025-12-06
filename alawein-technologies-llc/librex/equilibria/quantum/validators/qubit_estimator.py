"""
Qubit Requirements Estimator

Estimates qubit requirements and quantum resources for optimization problems.

Author: Meshal Alawein
Date: 2025-11-18
"""

import logging
import math
from typing import Dict, Any, Optional
import numpy as np

from Librex.core.interfaces import StandardizedProblem

logger = logging.getLogger(__name__)


class QubitEstimator:
    """
    Estimates quantum resource requirements for optimization problems.

    Provides detailed analysis of:
    - Qubit count requirements
    - Circuit depth estimates
    - Gate count estimates
    - Memory requirements
    - Runtime estimates
    """

    def __init__(self):
        """Initialize qubit estimator."""
        logger.info("Initialized QubitEstimator")

    def estimate_qubits(
        self,
        problem: StandardizedProblem,
        encoding: str = 'binary'
    ) -> Dict[str, Any]:
        """
        Estimate qubit requirements for a problem.

        Args:
            problem: Optimization problem
            encoding: Variable encoding ('binary', 'one-hot', 'gray', 'unary')

        Returns:
            Qubit requirement analysis
        """
        metadata = problem.problem_metadata or {}
        problem_type = metadata.get('problem_type', 'unknown')

        # Base qubit calculation
        if problem_type == 'QAP':
            n = metadata.get('n_facilities', int(np.sqrt(problem.dimension)))
            base_qubits = self._qap_qubits(n, encoding)
        elif problem_type == 'TSP':
            n = metadata.get('n_cities', int(np.sqrt(problem.dimension)))
            base_qubits = self._tsp_qubits(n, encoding)
        elif problem_type == 'Max-Cut':
            n = metadata.get('n_vertices', problem.dimension)
            base_qubits = self._maxcut_qubits(n, encoding)
        else:
            base_qubits = self._generic_qubits(problem.dimension, encoding)

        # Calculate ancilla requirements
        ancilla_qubits = self._estimate_ancilla_qubits(base_qubits, problem)

        # Total requirements
        total_qubits = base_qubits + ancilla_qubits

        # Create detailed report
        report = {
            'problem_type': problem_type,
            'encoding': encoding,
            'logical_qubits': base_qubits,
            'ancilla_qubits': ancilla_qubits,
            'total_qubits': total_qubits,
            'feasibility': self._assess_feasibility(total_qubits),
            'encoding_efficiency': self._calculate_encoding_efficiency(
                problem, base_qubits, encoding
            ),
        }

        # Add hardware-specific estimates
        report['hardware_requirements'] = self._hardware_requirements(total_qubits)

        logger.info(f"Estimated {total_qubits} total qubits for {problem_type} "
                   f"({base_qubits} logical + {ancilla_qubits} ancilla)")

        return report

    def _qap_qubits(self, n: int, encoding: str) -> int:
        """Calculate qubits for QAP."""
        if encoding == 'binary':
            # One qubit per (facility, location) pair
            return n * n
        elif encoding == 'one-hot':
            # Same as binary for permutation problems
            return n * n
        elif encoding == 'gray':
            # Log encoding for positions
            return n * math.ceil(math.log2(n))
        elif encoding == 'unary':
            # Unary encoding (less efficient)
            return n * (n - 1) // 2
        else:
            return n * n

    def _tsp_qubits(self, n: int, encoding: str) -> int:
        """Calculate qubits for TSP."""
        if encoding == 'binary':
            # One qubit per (city, position) pair
            return n * n
        elif encoding == 'one-hot':
            return n * n
        elif encoding == 'gray':
            # Log encoding for tour positions
            return n * math.ceil(math.log2(n))
        elif encoding == 'unary':
            # Edge-based encoding
            return n * (n - 1) // 2
        else:
            return n * n

    def _maxcut_qubits(self, n: int, encoding: str) -> int:
        """Calculate qubits for Max-Cut."""
        # Max-Cut only needs one qubit per vertex
        return n

    def _generic_qubits(self, dimension: int, encoding: str) -> int:
        """Calculate qubits for generic problem."""
        if encoding == 'binary':
            return dimension
        elif encoding == 'gray':
            # Assume we can compress by log factor
            return math.ceil(math.log2(dimension)) * (dimension // 4)
        else:
            return dimension

    def _estimate_ancilla_qubits(
        self,
        logical_qubits: int,
        problem: StandardizedProblem
    ) -> int:
        """
        Estimate ancillary qubit requirements.

        Ancilla qubits are needed for:
        - Error correction
        - Constraint enforcement
        - Intermediate computations
        """
        ancilla = 0

        # Constraint enforcement
        if problem.constraint_matrix is not None:
            num_constraints = problem.constraint_matrix.shape[0]
            ancilla += num_constraints  # One ancilla per constraint

        # Intermediate computations (typically 10-20% of logical qubits)
        ancilla += max(1, logical_qubits // 10)

        # Error correction overhead (for fault-tolerant computation)
        # Not applicable for NISQ, but included for completeness
        # ancilla += logical_qubits * 10  # Conservative estimate

        return ancilla

    def _assess_feasibility(self, total_qubits: int) -> Dict[str, bool]:
        """Assess feasibility on different quantum platforms."""
        return {
            'nisq_feasible': total_qubits <= 50,
            'nisq_ideal': total_qubits <= 20,
            'annealing_feasible': total_qubits <= 5000,
            'simulation_feasible': total_qubits <= 30,
            'future_feasible': total_qubits <= 1000,  # ~5-10 years
        }

    def _calculate_encoding_efficiency(
        self,
        problem: StandardizedProblem,
        qubits: int,
        encoding: str
    ) -> float:
        """
        Calculate encoding efficiency (information per qubit).

        Returns value between 0 and 1, where 1 is optimal.
        """
        # Calculate information content
        metadata = problem.problem_metadata or {}
        problem_type = metadata.get('problem_type', 'unknown')

        if problem_type in ['QAP', 'TSP']:
            n = int(np.sqrt(problem.dimension))
            # Information content: log2(n!)
            info_bits = sum(math.log2(i) for i in range(1, n + 1))
        else:
            # Generic: assume full binary space
            info_bits = problem.dimension

        # Efficiency = information / qubits used
        efficiency = min(1.0, info_bits / max(qubits, 1))

        return efficiency

    def _hardware_requirements(self, total_qubits: int) -> Dict[str, Any]:
        """Estimate hardware requirements based on qubit count."""
        requirements = {}

        # Gate-based quantum computers
        requirements['gate_based'] = {
            'minimum_connectivity': 'linear' if total_qubits <= 10 else 'all-to-all',
            'coherence_time_needed': f"{total_qubits * 10} μs",  # Rough estimate
            'gate_fidelity_needed': 0.99 ** (total_qubits * 10),  # Compound errors
            'recommended_platform': self._recommend_platform(total_qubits),
        }

        # Quantum annealers
        requirements['annealing'] = {
            'topology': 'chimera' if total_qubits <= 200 else 'pegasus',
            'chain_length_estimate': max(1, total_qubits // 50),
            'annealing_time': f"{20 + total_qubits // 100} μs",
        }

        return requirements

    def _recommend_platform(self, qubits: int) -> str:
        """Recommend quantum platform based on qubit requirements."""
        if qubits <= 5:
            return "Any NISQ device (IBM, Rigetti, IonQ)"
        elif qubits <= 20:
            return "High-quality NISQ (IonQ, Quantinuum)"
        elif qubits <= 50:
            return "Large NISQ (IBM Eagle, Google Sycamore)"
        elif qubits <= 400:
            return "Next-gen NISQ (IBM Condor)"
        elif qubits <= 5000:
            return "Quantum annealer (D-Wave Advantage)"
        else:
            return "Future fault-tolerant quantum computer"

    def estimate_circuit_depth(
        self,
        problem: StandardizedProblem,
        algorithm: str = 'qaoa'
    ) -> Dict[str, Any]:
        """
        Estimate quantum circuit depth for solving the problem.

        Args:
            problem: Optimization problem
            algorithm: Quantum algorithm to use

        Returns:
            Circuit depth analysis
        """
        # Get base qubit count
        qubit_estimate = self.estimate_qubits(problem)
        n_qubits = qubit_estimate['logical_qubits']

        depth_info = {
            'algorithm': algorithm,
            'n_qubits': n_qubits,
        }

        if algorithm == 'qaoa':
            # QAOA depth: p * (mixing + problem) layers
            p_layers = min(10, max(1, n_qubits // 5))
            mixing_depth = n_qubits  # Single-qubit rotations
            problem_depth = n_qubits * (n_qubits - 1) // 2  # Two-qubit gates
            total_depth = p_layers * (mixing_depth + problem_depth)

            depth_info.update({
                'p_layers': p_layers,
                'mixing_depth_per_layer': mixing_depth,
                'problem_depth_per_layer': problem_depth,
                'total_depth': total_depth,
                'two_qubit_gates': p_layers * n_qubits * (n_qubits - 1) // 2,
                'single_qubit_gates': p_layers * n_qubits * 2,
            })

        elif algorithm == 'vqe':
            # VQE with hardware-efficient ansatz
            layers = min(8, max(1, n_qubits // 3))
            depth_per_layer = 3 * n_qubits  # Ry, Rz, CNOT ladder
            total_depth = layers * depth_per_layer

            depth_info.update({
                'ansatz_layers': layers,
                'depth_per_layer': depth_per_layer,
                'total_depth': total_depth,
                'parameters': layers * n_qubits * 2,
            })

        elif algorithm == 'quantum_annealing':
            # Annealing doesn't have circuit depth, but has annealing time
            annealing_time = 20 * (1 + n_qubits // 100)  # microseconds

            depth_info.update({
                'annealing_time_us': annealing_time,
                'total_depth': None,  # Not applicable
            })

        else:
            # Generic estimate
            depth_info.update({
                'total_depth': n_qubits * n_qubits,
                'estimated': True,
            })

        # Estimate runtime (very rough)
        if depth_info.get('total_depth'):
            # Assume 1 μs per gate, 1000 shots
            runtime_ms = depth_info['total_depth'] * 1000 / 1000
            depth_info['estimated_runtime_ms'] = runtime_ms

        logger.info(f"Estimated circuit depth for {algorithm}: {depth_info.get('total_depth', 'N/A')}")

        return depth_info

    def estimate_memory_requirements(
        self,
        problem: StandardizedProblem,
        simulation: bool = True
    ) -> Dict[str, Any]:
        """
        Estimate memory requirements for quantum computation.

        Args:
            problem: Optimization problem
            simulation: Whether estimating for classical simulation

        Returns:
            Memory requirement analysis
        """
        qubit_estimate = self.estimate_qubits(problem)
        n_qubits = qubit_estimate['total_qubits']

        memory_info = {
            'n_qubits': n_qubits,
        }

        if simulation:
            # Classical simulation memory: 2^n complex numbers
            state_vector_size = 2 ** n_qubits
            bytes_per_amplitude = 16  # Complex double
            memory_bytes = state_vector_size * bytes_per_amplitude

            memory_info.update({
                'simulation_type': 'state_vector',
                'state_vector_size': state_vector_size,
                'memory_bytes': memory_bytes,
                'memory_gb': memory_bytes / (1024 ** 3),
                'feasible': memory_bytes < 16 * (1024 ** 3),  # 16 GB limit
            })

            # Suggest alternatives for large problems
            if not memory_info['feasible']:
                memory_info['alternatives'] = [
                    "Use tensor network simulation",
                    "Use sampling-based methods",
                    "Run on actual quantum hardware",
                ]

        else:
            # Quantum hardware memory (classical control)
            memory_info.update({
                'classical_memory_mb': n_qubits * 0.1,  # Control parameters
                'quantum_memory': 'Coherent quantum state',
                'feasible': True,
            })

        logger.info(f"Memory requirements: {memory_info.get('memory_gb', 0):.2f} GB "
                   f"(simulation={simulation})")

        return memory_info

    def compare_encodings(
        self,
        problem: StandardizedProblem
    ) -> Dict[str, Any]:
        """
        Compare different qubit encodings for the problem.

        Args:
            problem: Optimization problem

        Returns:
            Comparison of encoding strategies
        """
        encodings = ['binary', 'one-hot', 'gray', 'unary']
        comparison = {}

        for encoding in encodings:
            try:
                estimate = self.estimate_qubits(problem, encoding)
                comparison[encoding] = {
                    'qubits': estimate['total_qubits'],
                    'efficiency': estimate['encoding_efficiency'],
                    'feasible': estimate['feasibility']['nisq_feasible'],
                }
            except Exception as e:
                comparison[encoding] = {
                    'error': str(e),
                    'supported': False,
                }

        # Find best encoding
        valid_encodings = {k: v for k, v in comparison.items()
                          if 'qubits' in v and v.get('feasible', False)}

        if valid_encodings:
            best_encoding = min(valid_encodings.keys(),
                              key=lambda k: valid_encodings[k]['qubits'])
            comparison['recommended'] = best_encoding
            comparison['recommendation_reason'] = "Minimum qubit count while feasible"
        else:
            comparison['recommended'] = 'binary'
            comparison['recommendation_reason'] = "Default encoding"

        return comparison


__all__ = ['QubitEstimator']