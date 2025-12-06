"""
Quantum State Decoder

Decodes quantum measurement results into classical optimization solutions.

Author: Meshal Alawein
Date: 2025-11-18
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class QuantumStateDecoder:
    """
    Decodes quantum states and measurement results into classical solutions.

    Handles:
    - Bitstring to solution mapping
    - Probability distribution analysis
    - Solution extraction from measurements
    - Error correction and validation
    """

    def __init__(self):
        """Initialize quantum state decoder."""
        logger.info("Initialized QuantumStateDecoder")

    def decode_bitstring(
        self,
        bitstring: str,
        problem_type: str = 'generic'
    ) -> np.ndarray:
        """
        Decode a bitstring to a solution vector.

        Args:
            bitstring: Binary string (e.g., '0110')
            problem_type: Type of optimization problem

        Returns:
            Solution vector
        """
        # Convert bitstring to numpy array
        binary_array = np.array([int(b) for b in bitstring])

        if problem_type == 'QAP':
            return self._decode_qap_bitstring(binary_array)
        elif problem_type == 'TSP':
            return self._decode_tsp_bitstring(binary_array)
        elif problem_type == 'Max-Cut':
            return binary_array  # Direct mapping
        else:
            return binary_array

    def _decode_qap_bitstring(self, binary: np.ndarray) -> np.ndarray:
        """
        Decode QAP bitstring to assignment.

        Binary variables represent x[i,j] = 1 if facility i -> location j.
        """
        n = int(np.sqrt(len(binary)))
        assignment = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(n):
                if binary[i * n + j] > 0.5:
                    assignment[i] = j

        return assignment

    def _decode_tsp_bitstring(self, binary: np.ndarray) -> np.ndarray:
        """
        Decode TSP bitstring to tour.

        Binary variables represent x[i,t] = 1 if city i visited at time t.
        """
        n = int(np.sqrt(len(binary)))
        tour = np.zeros(n, dtype=int)

        for t in range(n):
            for i in range(n):
                if binary[i * n + t] > 0.5:
                    tour[t] = i

        return tour

    def decode_measurement_results(
        self,
        measurements: Dict[str, int],
        problem_type: str = 'generic',
        top_k: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Decode quantum measurement results.

        Args:
            measurements: Dictionary mapping bitstrings to counts
            problem_type: Type of optimization problem
            top_k: Number of top solutions to return

        Returns:
            List of decoded solutions with metadata
        """
        if not measurements:
            logger.warning("Empty measurement results")
            return []

        # Calculate probabilities
        total_counts = sum(measurements.values())
        probabilities = {bs: count / total_counts
                        for bs, count in measurements.items()}

        # Sort by probability
        sorted_results = sorted(probabilities.items(),
                              key=lambda x: x[1], reverse=True)

        # Decode top-k solutions
        solutions = []
        for i, (bitstring, prob) in enumerate(sorted_results[:top_k]):
            solution = self.decode_bitstring(bitstring, problem_type)
            solutions.append({
                'rank': i + 1,
                'bitstring': bitstring,
                'solution': solution,
                'probability': prob,
                'counts': measurements[bitstring],
                'is_valid': self._validate_solution(solution, problem_type),
            })

        logger.info(f"Decoded {len(solutions)} solutions from {len(measurements)} measurements")

        return solutions

    def _validate_solution(
        self,
        solution: np.ndarray,
        problem_type: str
    ) -> bool:
        """
        Validate if decoded solution is feasible.

        Args:
            solution: Decoded solution
            problem_type: Problem type

        Returns:
            True if solution is valid
        """
        if problem_type in ['QAP', 'TSP']:
            # Check if it's a valid permutation
            n = len(solution)
            return len(np.unique(solution)) == n and np.all(solution < n)
        elif problem_type == 'Max-Cut':
            # Binary partition is always valid
            return True
        else:
            # Generic validation
            return not np.any(np.isnan(solution))

    def extract_best_feasible_solution(
        self,
        measurements: Dict[str, int],
        problem_type: str = 'generic',
        objective_function: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Extract the best feasible solution from measurements.

        Args:
            measurements: Measurement results
            problem_type: Problem type
            objective_function: Optional function to evaluate solutions

        Returns:
            Best feasible solution with metadata
        """
        # Decode all solutions
        all_solutions = self.decode_measurement_results(
            measurements, problem_type, top_k=len(measurements)
        )

        # Filter feasible solutions
        feasible = [sol for sol in all_solutions if sol['is_valid']]

        if not feasible:
            logger.warning("No feasible solutions found")
            # Return best infeasible solution
            return all_solutions[0] if all_solutions else None

        # If objective function provided, evaluate solutions
        if objective_function:
            for sol in feasible:
                sol['objective_value'] = objective_function(sol['solution'])

            # Sort by objective value
            feasible.sort(key=lambda x: x.get('objective_value', float('inf')))

        # Return best feasible solution
        best = feasible[0]
        logger.info(f"Found best feasible solution with probability {best['probability']:.3f}")

        return best

    def analyze_solution_distribution(
        self,
        measurements: Dict[str, int],
        problem_type: str = 'generic'
    ) -> Dict[str, Any]:
        """
        Analyze the distribution of solutions from measurements.

        Args:
            measurements: Measurement results
            problem_type: Problem type

        Returns:
            Statistical analysis of solution distribution
        """
        if not measurements:
            return {'error': 'No measurements provided'}

        # Basic statistics
        total_counts = sum(measurements.values())
        num_unique = len(measurements)

        # Calculate probabilities
        probabilities = [count / total_counts for count in measurements.values()]

        # Entropy calculation
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
        max_entropy = np.log2(num_unique)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Top solution analysis
        top_bitstring = max(measurements.keys(), key=lambda k: measurements[k])
        top_probability = measurements[top_bitstring] / total_counts

        # Decode solutions for validity analysis
        solutions = self.decode_measurement_results(
            measurements, problem_type, top_k=min(100, len(measurements))
        )
        num_valid = sum(1 for sol in solutions if sol['is_valid'])
        validity_rate = num_valid / len(solutions) if solutions else 0

        analysis = {
            'total_measurements': total_counts,
            'unique_solutions': num_unique,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'top_solution': {
                'bitstring': top_bitstring,
                'probability': top_probability,
                'counts': measurements[top_bitstring],
            },
            'validity_rate': validity_rate,
            'concentration': top_probability,  # How concentrated is the distribution
            'diversity': normalized_entropy,   # How diverse are the solutions
        }

        # Quality assessment
        if top_probability > 0.5:
            analysis['quality'] = 'excellent'
            analysis['confidence'] = 'high'
        elif top_probability > 0.1:
            analysis['quality'] = 'good'
            analysis['confidence'] = 'medium'
        else:
            analysis['quality'] = 'poor'
            analysis['confidence'] = 'low'

        logger.info(f"Solution distribution: {num_unique} unique, "
                   f"entropy={entropy:.2f}, top_prob={top_probability:.3f}")

        return analysis

    def decode_statevector(
        self,
        statevector: np.ndarray,
        problem_type: str = 'generic',
        threshold: float = 0.001
    ) -> List[Dict[str, Any]]:
        """
        Decode solutions from a quantum statevector.

        Args:
            statevector: Quantum state amplitudes
            problem_type: Problem type
            threshold: Probability threshold for considering states

        Returns:
            List of decoded solutions above threshold
        """
        n_qubits = int(np.log2(len(statevector)))

        # Calculate probabilities
        probabilities = np.abs(statevector) ** 2

        # Find states above threshold
        solutions = []
        for i, prob in enumerate(probabilities):
            if prob > threshold:
                # Convert index to bitstring
                bitstring = format(i, f'0{n_qubits}b')
                solution = self.decode_bitstring(bitstring, problem_type)

                solutions.append({
                    'bitstring': bitstring,
                    'solution': solution,
                    'amplitude': complex(statevector[i]),
                    'probability': prob,
                    'is_valid': self._validate_solution(solution, problem_type),
                })

        # Sort by probability
        solutions.sort(key=lambda x: x['probability'], reverse=True)

        logger.info(f"Decoded {len(solutions)} solutions from statevector "
                   f"(threshold={threshold})")

        return solutions

    def correct_solution_errors(
        self,
        solution: np.ndarray,
        problem_type: str
    ) -> Tuple[np.ndarray, bool]:
        """
        Attempt to correct errors in decoded solution.

        Args:
            solution: Potentially invalid solution
            problem_type: Problem type

        Returns:
            Tuple of (corrected_solution, was_corrected)
        """
        was_corrected = False

        if problem_type in ['QAP', 'TSP']:
            # For permutation problems, ensure valid permutation
            n = len(solution)

            # Check for duplicates
            if len(np.unique(solution)) < n:
                # Find missing values
                all_values = set(range(n))
                used_values = set(solution)
                missing = list(all_values - used_values)

                # Replace duplicates with missing values
                seen = set()
                for i in range(n):
                    if solution[i] in seen and missing:
                        solution[i] = missing.pop()
                        was_corrected = True
                    seen.add(solution[i])

            # Check for out-of-range values
            if np.any(solution >= n) or np.any(solution < 0):
                solution = np.clip(solution, 0, n - 1)
                was_corrected = True

        if was_corrected:
            logger.info(f"Corrected errors in {problem_type} solution")

        return solution, was_corrected


__all__ = ['QuantumStateDecoder']