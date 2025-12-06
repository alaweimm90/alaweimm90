"""
Quantum to Classical Result Converter

Converts quantum computation results to classical optimization solutions.

Author: Meshal Alawein
Date: 2025-11-18
"""

import logging
from typing import Dict, Any, Union, Optional
import numpy as np

from Librex.core.interfaces import StandardizedSolution

logger = logging.getLogger(__name__)


def quantum_to_classical_result(
    quantum_result: Dict[str, Any],
    problem_metadata: Optional[Dict[str, Any]] = None
) -> StandardizedSolution:
    """
    Convert quantum optimization result to classical format.

    Args:
        quantum_result: Result from quantum optimizer containing:
            - 'solution': Binary/spin array or bitstring
            - 'energy' or 'objective': Objective value
            - 'measurements': Optional measurement counts
            - 'metadata': Optional additional data
        problem_metadata: Original problem metadata

    Returns:
        StandardizedSolution compatible with classical optimizers
    """
    # Extract solution
    if 'solution' in quantum_result:
        solution = quantum_result['solution']
        if isinstance(solution, str):
            # Convert bitstring to array
            solution = np.array([int(b) for b in solution])
        elif not isinstance(solution, np.ndarray):
            solution = np.array(solution)
    elif 'bitstring' in quantum_result:
        # Convert bitstring
        solution = np.array([int(b) for b in quantum_result['bitstring']])
    elif 'spins' in quantum_result:
        # Convert spins to binary
        spins = np.array(quantum_result['spins'])
        solution = ((spins + 1) / 2).astype(int)
    else:
        raise ValueError("No solution found in quantum result")

    # Extract objective value
    objective_value = (
        quantum_result.get('objective') or
        quantum_result.get('energy') or
        quantum_result.get('objective_value', 0.0)
    )

    # Determine validity
    is_valid = quantum_result.get('is_valid', True)
    if 'feasible' in quantum_result:
        is_valid = quantum_result['feasible']

    # Convert solution based on problem type
    if problem_metadata:
        problem_type = problem_metadata.get('problem_type', 'generic')
        solution = _convert_solution_format(solution, problem_type)

    # Build metadata
    metadata = {
        'quantum_solver': quantum_result.get('solver', 'unknown'),
        'quantum_backend': quantum_result.get('backend', 'unknown'),
    }

    # Add measurement statistics if available
    if 'measurements' in quantum_result:
        measurements = quantum_result['measurements']
        total_shots = sum(measurements.values())
        best_count = max(measurements.values())
        metadata['measurement_stats'] = {
            'total_shots': total_shots,
            'unique_solutions': len(measurements),
            'best_probability': best_count / total_shots,
        }

    # Add convergence info if available
    if 'convergence' in quantum_result:
        metadata['convergence'] = quantum_result['convergence']

    # Add any additional quantum-specific metadata
    for key in ['circuit_depth', 'n_qubits', 'optimization_level', 'iterations']:
        if key in quantum_result:
            metadata[key] = quantum_result[key]

    # Merge with existing metadata
    if quantum_result.get('metadata'):
        metadata.update(quantum_result['metadata'])

    logger.info(f"Converted quantum result: objective={objective_value:.6f}, "
               f"valid={is_valid}")

    return StandardizedSolution(
        vector=solution,
        objective_value=float(objective_value),
        is_valid=is_valid,
        metadata=metadata
    )


def _convert_solution_format(
    solution: np.ndarray,
    problem_type: str
) -> np.ndarray:
    """
    Convert solution to problem-specific format.

    Args:
        solution: Raw solution array
        problem_type: Type of optimization problem

    Returns:
        Formatted solution
    """
    if problem_type == 'QAP':
        # Convert binary matrix to assignment
        n = int(np.sqrt(len(solution)))
        assignment = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if solution[i * n + j] > 0.5:
                    assignment[i] = j
        return assignment

    elif problem_type == 'TSP':
        # Convert binary matrix to tour
        n = int(np.sqrt(len(solution)))
        tour = np.zeros(n, dtype=int)
        for t in range(n):
            for i in range(n):
                if solution[i * n + t] > 0.5:
                    tour[t] = i
        return tour

    elif problem_type == 'Max-Cut':
        # Binary partition
        return (solution > 0.5).astype(int)

    else:
        # Generic: return as-is
        return solution


__all__ = ['quantum_to_classical_result']