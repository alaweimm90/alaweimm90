"""
Quadratic Assignment Problem (QAP) Domain Adapter - CORRECTED VERSION
"""

from typing import Any, Dict

import numpy as np

from Librex.core.interfaces import (
    StandardizedProblem,
    StandardizedSolution,
    UniversalOptimizationInterface,
    ValidationResult,
)


class QAPAdapter(UniversalOptimizationInterface):
    """QAP to universal format adapter - MATHEMATICALALLY CORRECTED"""

    def __init__(self):
        super().__init__({
            'name': 'qap',
            'type': 'discrete',
            'constraints': ['permutation'],
            'objectives': ['minimize']
        })
        self.n = None
        self.A = None
        self.B = None

    def encode_problem(self, instance: Dict[str, np.ndarray]) -> StandardizedProblem:
        """Convert QAP instance to standardized format"""
        self.A = instance['flow_matrix']
        self.B = instance['distance_matrix']
        self.n = len(self.A)

        # FIXED: QAP should use trace formulation, not Kronecker product
        # QAP objective: min trace(A @ P @ B @ P.T) where P is permutation matrix
        # The flow × distance interactions are properly captured by the objective function

        return StandardizedProblem(
            dimension=self.n,
            objective_matrix=None,  # QAP doesn't use simple matrix formulation
            objective_function=self.compute_objective,
            constraint_matrix=self._permutation_constraints(),
            problem_metadata={
                'type': 'quadratic_assignment',
                'size': self.n,
                'flow_matrix': self.A,
                'distance_matrix': self.B,
                'mathematical_formulation': 'trace(A @ P @ B @ P.T)'
            }
        )

    def decode_solution(self, solution: StandardizedSolution) -> np.ndarray:
        """Convert solution to permutation"""
        P = solution.vector.reshape((self.n, self.n))
        return np.argmax(P, axis=1)

    def validate_solution(self, solution: np.ndarray) -> ValidationResult:
        """Check if solution is valid permutation"""
        violations = []
        magnitudes = []

        # Check if permutation
        if not self._is_permutation(solution):
            violations.append("Not a valid permutation")
            magnitudes.append(1.0)

        return ValidationResult(
            is_valid=len(violations) == 0,
            constraint_violations=violations,
            violation_magnitudes=magnitudes
        )

    def compute_objective(self, solution: np.ndarray) -> float:
        """
        Compute QAP objective value using corrected mathematical formulation.

        QAP: min Σᵢ Σⱼ aᵢⱼ × b₍π₍ᵢ₎, π₍ⱼ₎₎
        Matrix form: trace(A @ P @ B @ P.T)

        Where:
        - A is flow matrix (size n×n)
        - B is distance matrix (size n×n)
        - π is permutation vector
        - P is permutation matrix for π
        """
        P = self._permutation_to_matrix(solution)

        # CORRECTED: Use trace formulation instead of Kronecker product
        # This computes: trace(A @ P @ B @ P.T)
        # Which equals: Σᵢ Σⱼ aᵢⱼ × b₍π₍ᵢ₎, π₍ⱼ₎₎
        return float(np.trace(self.A @ P @ self.B @ P.T))

    def _permutation_constraints(self) -> np.ndarray:
        """Generate permutation matrix constraints"""
        # Doubly stochastic constraints
        constraints = np.zeros((2 * self.n, self.n * self.n))

        # Row sum = 1
        for i in range(self.n):
            constraints[i, i*self.n:(i+1)*self.n] = 1

        # Column sum = 1
        for j in range(self.n):
            constraints[self.n + j, j::self.n] = 1

        return constraints

    def _is_permutation(self, perm: np.ndarray) -> bool:
        """Check if array is a permutation"""
        return (
            len(perm) == self.n and
            np.all(np.sort(perm) == np.arange(self.n))
        )

    def _permutation_to_matrix(self, perm: np.ndarray) -> np.ndarray:
        """Convert permutation to permutation matrix"""
        P = np.zeros((self.n, self.n))
        P[np.arange(self.n), perm] = 1
        return P

    def verify_qap_properties(self) -> Dict[str, bool]:
        """
        Verify that the QAP instance has valid mathematical properties.

        Returns:
            Dictionary with validation results for various QAP properties.
        """
        if self.A is None or self.B is None:
            return {'instance_loaded': False}

        # Check matrices are square and same size
        square_check = (self.A.shape[0] == self.A.shape[1] ==
                       self.B.shape[0] == self.B.shape[1])

        # Check matrices are positive (flow and distance should be non-negative)
        positive_flow = np.all(self.A >= 0)
        positive_distance = np.all(self.B >= 0)

        # Check symmetry (common in QAP instances)
        symmetric_flow = np.allclose(self.A, self.A.T)
        symmetric_distance = np.allclose(self.B, self.B.T)

        return {
            'instance_loaded': True,
            'square_matrices': square_check,
            'positive_flows': positive_flow,
            'positive_distances': positive_distance,
            'symmetric_flows': symmetric_flow,
            'symmetric_distances': symmetric_distance,
            'mathematical_valid': square_check and positive_flow and positive_distance
        }


# Test function to verify the fix
def test_qap_formulation():
    """Test the corrected QAP formulation"""
    # Simple 3x3 test case
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    B = np.array([[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]])

    # Test identity permutation
    identity = np.array([0, 1, 2])

    adapter = QAPAdapter()
    adapter.A = A
    adapter.B = B
    adapter.n = 3

    result = adapter.compute_objective(identity)
    print(f"Identity permutation objective: {result}")

    # Expected: trace(A @ I @ B @ I.T) = trace(A @ B)
    expected = float(np.trace(A @ B))
    print(f"Expected: {expected}")
    print(f"Match: {abs(result - expected) < 1e-10}")


if __name__ == "__main__":
    test_qap_formulation()
