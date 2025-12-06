"""
Quantum Problem Validator

Validates whether optimization problems are suitable for quantum computing
and checks hardware constraints.

Author: Meshal Alawein
Date: 2025-11-18
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

from Librex.core.interfaces import StandardizedProblem

logger = logging.getLogger(__name__)


class QuantumProblemValidator:
    """
    Validates optimization problems for quantum computing suitability.

    Checks for:
    - Problem size constraints (NISQ limitations)
    - Problem structure suitability
    - Hardware compatibility
    - Numerical stability
    """

    # Hardware limitations (current state-of-the-art)
    NISQ_QUBIT_LIMIT = 50  # Typical NISQ device limit
    IDEAL_NISQ_QUBITS = 20  # Sweet spot for current devices
    ANNEALING_QUBIT_LIMIT = 5000  # D-Wave Advantage limit
    SIMULATION_LIMIT = 30  # Classical simulation feasibility

    def __init__(self, hardware_type: str = 'nisq'):
        """
        Initialize validator.

        Args:
            hardware_type: Target hardware ('nisq', 'annealing', 'simulator', 'ideal')
        """
        self.hardware_type = hardware_type
        self.qubit_limit = self._get_qubit_limit(hardware_type)

        logger.info(f"Initialized validator for {hardware_type} hardware "
                   f"(qubit limit: {self.qubit_limit})")

    def _get_qubit_limit(self, hardware_type: str) -> int:
        """Get qubit limit for hardware type."""
        limits = {
            'nisq': self.NISQ_QUBIT_LIMIT,
            'annealing': self.ANNEALING_QUBIT_LIMIT,
            'simulator': self.SIMULATION_LIMIT,
            'ideal': float('inf'),
        }
        return limits.get(hardware_type, self.NISQ_QUBIT_LIMIT)

    def validate_problem(
        self,
        problem: StandardizedProblem
    ) -> Dict[str, Any]:
        """
        Validate if problem is suitable for quantum optimization.

        Args:
            problem: Standardized optimization problem

        Returns:
            Validation results dictionary
        """
        results = {
            'is_suitable': True,
            'issues': [],
            'warnings': [],
            'recommendations': [],
        }

        # Check problem size
        size_check = self._check_problem_size(problem)
        results.update(size_check)

        # Check problem structure
        structure_check = self._check_problem_structure(problem)
        results['structure'] = structure_check
        if structure_check.get('issues'):
            results['issues'].extend(structure_check['issues'])

        # Check numerical stability
        stability_check = self._check_numerical_stability(problem)
        results['numerical_stability'] = stability_check
        if stability_check.get('warnings'):
            results['warnings'].extend(stability_check['warnings'])

        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(
            problem, results
        )

        # Overall suitability
        results['is_suitable'] = (
            len(results['issues']) == 0 and
            results.get('size_feasible', False)
        )

        return results

    def _check_problem_size(
        self,
        problem: StandardizedProblem
    ) -> Dict[str, Any]:
        """Check if problem size is feasible for quantum hardware."""
        metadata = problem.problem_metadata or {}
        problem_type = metadata.get('problem_type', 'unknown')

        # Calculate required qubits
        if problem_type in ['QAP', 'TSP']:
            # These require n^2 qubits for n-sized problem
            n = int(np.sqrt(problem.dimension)) if problem.dimension > 0 else 0
            required_qubits = n * n
            problem_size = n
        else:
            required_qubits = problem.dimension
            problem_size = problem.dimension

        # Check against limits
        size_feasible = required_qubits <= self.qubit_limit
        ideal_size = required_qubits <= self.IDEAL_NISQ_QUBITS
        simulatable = required_qubits <= self.SIMULATION_LIMIT

        # Generate size analysis
        analysis = {
            'required_qubits': required_qubits,
            'problem_size': problem_size,
            'size_feasible': size_feasible,
            'ideal_size': ideal_size,
            'classically_simulatable': simulatable,
            'hardware_type': self.hardware_type,
            'qubit_limit': self.qubit_limit,
        }

        # Add warnings/issues
        if not size_feasible:
            analysis['issues'] = [
                f"Problem requires {required_qubits} qubits, "
                f"exceeds {self.hardware_type} limit of {self.qubit_limit}"
            ]
        elif not ideal_size and self.hardware_type == 'nisq':
            analysis['warnings'] = [
                f"Problem size ({required_qubits} qubits) is large for NISQ. "
                f"Performance may be limited by noise."
            ]

        return analysis

    def _check_problem_structure(
        self,
        problem: StandardizedProblem
    ) -> Dict[str, Any]:
        """Check if problem structure is suitable for quantum optimization."""
        issues = []
        info = {}

        if problem.objective_matrix is not None:
            matrix = problem.objective_matrix
            n = matrix.shape[0]

            # Check matrix properties
            is_symmetric = np.allclose(matrix, matrix.T)
            sparsity = 1 - np.count_nonzero(matrix) / matrix.size
            condition_number = np.linalg.cond(matrix) if n > 0 else 1

            info['is_symmetric'] = is_symmetric
            info['sparsity'] = sparsity
            info['condition_number'] = condition_number

            # Check for suitability
            if not is_symmetric and problem.problem_metadata.get('problem_type') in ['QAP', 'TSP']:
                issues.append("Objective matrix is not symmetric")

            if condition_number > 1e10:
                issues.append(f"Matrix is ill-conditioned (condition number: {condition_number:.2e})")

            if sparsity < 0.5:
                info['structure_type'] = 'dense'
            else:
                info['structure_type'] = 'sparse'

        # Check constraints
        if problem.constraint_matrix is not None:
            info['has_constraints'] = True
            info['num_constraints'] = problem.constraint_matrix.shape[0]
            if self.hardware_type == 'annealing':
                issues.append("Quantum annealing has limited constraint handling")
        else:
            info['has_constraints'] = False

        return {'info': info, 'issues': issues}

    def _check_numerical_stability(
        self,
        problem: StandardizedProblem
    ) -> Dict[str, Any]:
        """Check numerical stability for quantum computation."""
        warnings = []
        info = {}

        if problem.objective_matrix is not None:
            matrix = problem.objective_matrix

            # Check coefficient magnitudes
            max_coeff = np.max(np.abs(matrix))
            min_nonzero = np.min(np.abs(matrix[matrix != 0])) if np.any(matrix != 0) else 0
            dynamic_range = max_coeff / min_nonzero if min_nonzero > 0 else float('inf')

            info['max_coefficient'] = max_coeff
            info['min_nonzero_coefficient'] = min_nonzero
            info['dynamic_range'] = dynamic_range

            # Generate warnings
            if max_coeff > 1e6:
                warnings.append(f"Very large coefficients detected (max: {max_coeff:.2e})")

            if dynamic_range > 1e9:
                warnings.append(f"Large dynamic range ({dynamic_range:.2e}) may cause precision issues")

            # Check for special values
            if np.any(np.isnan(matrix)):
                warnings.append("Matrix contains NaN values")
            if np.any(np.isinf(matrix)):
                warnings.append("Matrix contains infinite values")

        return {'info': info, 'warnings': warnings}

    def _generate_recommendations(
        self,
        problem: StandardizedProblem,
        validation_results: Dict[str, Any]
    ) -> list:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Size-based recommendations
        required_qubits = validation_results.get('required_qubits', 0)
        if required_qubits > self.NISQ_QUBIT_LIMIT:
            if self.hardware_type == 'nisq':
                recommendations.append(
                    f"Consider quantum-inspired classical algorithms or "
                    f"hybrid quantum-classical approaches"
                )
                recommendations.append(
                    f"Problem decomposition may help reduce qubit requirements"
                )
        elif required_qubits > self.IDEAL_NISQ_QUBITS:
            recommendations.append(
                f"Consider using error mitigation techniques for better results"
            )

        # Structure-based recommendations
        structure = validation_results.get('structure', {})
        if structure.get('info', {}).get('sparsity', 0) > 0.8:
            recommendations.append(
                "Sparse problem structure is well-suited for quantum optimization"
            )

        # Numerical stability recommendations
        stability = validation_results.get('numerical_stability', {})
        if stability.get('info', {}).get('dynamic_range', 0) > 1e6:
            recommendations.append(
                "Consider rescaling coefficients to improve numerical stability"
            )

        # Hardware recommendations
        if self.hardware_type == 'nisq' and required_qubits <= self.ANNEALING_QUBIT_LIMIT:
            if not structure.get('info', {}).get('has_constraints', False):
                recommendations.append(
                    "This problem may be suitable for quantum annealing"
                )

        return recommendations

    def estimate_success_probability(
        self,
        problem: StandardizedProblem,
        method: str = 'qaoa'
    ) -> float:
        """
        Estimate success probability for solving the problem on quantum hardware.

        Args:
            problem: Optimization problem
            method: Quantum algorithm ('qaoa', 'vqe', 'annealing')

        Returns:
            Estimated success probability (0-1)

        Note: This is a rough heuristic based on problem characteristics.
        """
        base_probability = 0.5  # Base success rate

        # Adjust for problem size
        metadata = problem.problem_metadata or {}
        problem_type = metadata.get('problem_type', 'unknown')

        if problem_type in ['QAP', 'TSP']:
            n = int(np.sqrt(problem.dimension))
            size_factor = max(0.1, 1.0 - (n - 5) * 0.05)  # Decrease with size
        else:
            size_factor = max(0.1, 1.0 - (problem.dimension - 10) * 0.02)

        # Adjust for hardware type
        hardware_factors = {
            'nisq': 0.7,
            'annealing': 0.8,
            'simulator': 1.0,
            'ideal': 1.0,
        }
        hardware_factor = hardware_factors.get(self.hardware_type, 0.7)

        # Adjust for method
        method_factors = {
            'qaoa': 0.8,
            'vqe': 0.75,
            'annealing': 0.85,
        }
        method_factor = method_factors.get(method, 0.7)

        # Calculate final probability
        success_probability = base_probability * size_factor * hardware_factor * method_factor

        logger.info(f"Estimated success probability: {success_probability:.2%} "
                   f"(size_factor={size_factor:.2f}, hardware={hardware_factor:.2f}, "
                   f"method={method_factor:.2f})")

        return success_probability

    def recommend_quantum_algorithm(
        self,
        problem: StandardizedProblem
    ) -> Dict[str, Any]:
        """
        Recommend the best quantum algorithm for the problem.

        Args:
            problem: Optimization problem

        Returns:
            Algorithm recommendation with reasoning
        """
        metadata = problem.problem_metadata or {}
        problem_type = metadata.get('problem_type', 'unknown')

        # Default recommendation
        recommendation = {
            'algorithm': 'qaoa',
            'reasoning': [],
            'alternatives': [],
            'parameters': {},
        }

        # Problem-specific recommendations
        if problem_type in ['QAP', 'TSP']:
            n = int(np.sqrt(problem.dimension))
            if n <= 5:
                recommendation['algorithm'] = 'vqe'
                recommendation['reasoning'].append("Small problem size suitable for VQE")
                recommendation['parameters']['ansatz'] = 'hardware_efficient'
            else:
                recommendation['algorithm'] = 'qaoa'
                recommendation['reasoning'].append("QAOA is effective for combinatorial problems")
                recommendation['parameters']['p_layers'] = min(5, max(2, n // 2))

            if not problem.constraint_matrix:
                recommendation['alternatives'].append('quantum_annealing')

        elif problem_type == 'Max-Cut':
            recommendation['algorithm'] = 'qaoa'
            recommendation['reasoning'].append("QAOA was originally designed for Max-Cut")
            recommendation['parameters']['p_layers'] = 3

        # Hardware-based adjustments
        if self.hardware_type == 'annealing':
            if not problem.constraint_matrix:
                recommendation['algorithm'] = 'quantum_annealing'
                recommendation['reasoning'].append("Hardware is quantum annealer")
                recommendation['parameters']['annealing_time'] = 20  # microseconds

        return recommendation


__all__ = ['QuantumProblemValidator']