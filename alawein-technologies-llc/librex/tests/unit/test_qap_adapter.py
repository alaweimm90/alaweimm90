"""
Unit tests for QAP adapter mathematical correctness

Tests verify that the QAP formulation fixes have been correctly applied.
"""

import numpy as np
import pytest

from Librex.adapters.qap import QAPAdapter


def test_qap_formulation_correctness():
    """
    Verify QAP formulation matches theoretical trace formulation.

    Tests that the corrected QAP adapter uses the proper mathematical
    formulation: min trace(A @ P @ B @ P.T) instead of the incorrect
    Kronecker product formulation.
    """
    # Simple 2x2 test case with known solution
    A = np.array([[0, 1], [1, 0]])  # Flow matrix
    B = np.array([[0, 2], [2, 0]])  # Distance matrix

    adapter = QAPAdapter()
    adapter.A = A
    adapter.B = B
    adapter.n = 2

    # Test identity permutation [0, 1]
    identity = np.array([0, 1])
    result = adapter.compute_objective(identity)

    # Expected: trace(A @ I @ B @ I.T) = trace(A @ B)
    # A @ B = [[2, 0], [0, 2]]
    # trace = 4
    expected = float(np.trace(A @ B))

    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"


def test_qap_formulation_swap_permutation():
    """Test QAP objective with a swap permutation"""
    A = np.array([[0, 1], [1, 0]])
    B = np.array([[0, 2], [2, 0]])

    adapter = QAPAdapter()
    adapter.A = A
    adapter.B = B
    adapter.n = 2

    # Test swap permutation [1, 0]
    swap = np.array([1, 0])
    result = adapter.compute_objective(swap)

    # For swap: P = [[0, 1], [1, 0]]
    # A @ P @ B @ P.T should give same result due to symmetry
    P = np.array([[0, 1], [1, 0]])
    expected = float(np.trace(A @ P @ B @ P.T))

    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"


def test_qap_formulation_3x3():
    """Test QAP formulation with 3x3 instance"""
    A = np.array([[0, 1, 2],
                  [1, 0, 3],
                  [2, 3, 0]])

    B = np.array([[0, 5, 10],
                  [5, 0, 15],
                  [10, 15, 0]])

    adapter = QAPAdapter()
    adapter.A = A
    adapter.B = B
    adapter.n = 3

    # Test identity permutation
    identity = np.array([0, 1, 2])
    result = adapter.compute_objective(identity)

    # Expected: trace(A @ B)
    expected = float(np.trace(A @ B))

    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"


def test_qap_is_permutation():
    """Test permutation validation"""
    adapter = QAPAdapter()
    adapter.n = 3

    # Valid permutation
    assert adapter._is_permutation(np.array([0, 1, 2]))
    assert adapter._is_permutation(np.array([2, 0, 1]))
    assert adapter._is_permutation(np.array([1, 2, 0]))

    # Invalid permutations
    assert not adapter._is_permutation(np.array([0, 0, 1]))  # Duplicate
    assert not adapter._is_permutation(np.array([0, 1, 3]))  # Out of range
    assert not adapter._is_permutation(np.array([0, 1]))     # Wrong length


def test_qap_verify_properties():
    """Test QAP instance property verification"""
    # Valid symmetric QAP instance
    A = np.array([[0, 1], [1, 0]])
    B = np.array([[0, 2], [2, 0]])

    adapter = QAPAdapter()
    adapter.A = A
    adapter.B = B
    adapter.n = 2

    properties = adapter.verify_qap_properties()

    assert properties['instance_loaded']
    assert properties['square_matrices']
    assert properties['positive_flows']
    assert properties['positive_distances']
    assert properties['symmetric_flows']
    assert properties['symmetric_distances']
    assert properties['mathematical_valid']


def test_qap_negative_flows_detected():
    """Test that negative flows are detected in validation"""
    # Invalid QAP with negative flow
    A = np.array([[0, -1], [1, 0]])
    B = np.array([[0, 2], [2, 0]])

    adapter = QAPAdapter()
    adapter.A = A
    adapter.B = B
    adapter.n = 2

    properties = adapter.verify_qap_properties()

    assert not properties['positive_flows']
    assert not properties['mathematical_valid']


def test_qap_encode_problem():
    """Test encoding QAP instance to standardized format"""
    flow = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    distance = np.array([[0, 5, 10], [5, 0, 15], [10, 15, 0]])

    adapter = QAPAdapter()
    problem = adapter.encode_problem({
        'flow_matrix': flow,
        'distance_matrix': distance
    })

    assert problem.dimension == 3
    assert problem.objective_function is not None
    assert problem.objective_function == adapter.compute_objective
    assert problem.problem_metadata['type'] == 'quadratic_assignment'
    assert problem.problem_metadata['size'] == 3
    assert np.array_equal(problem.problem_metadata['flow_matrix'], flow)
    assert np.array_equal(problem.problem_metadata['distance_matrix'], distance)


def test_qap_decode_solution():
    """Test decoding StandardizedSolution to permutation"""
    from Librex.core.interfaces import StandardizedSolution

    adapter = QAPAdapter()
    adapter.n = 3

    # Create solution with permutation matrix representation
    # Permutation [1, 2, 0] as matrix
    solution_vector = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ]).flatten()

    solution = StandardizedSolution(
        vector=solution_vector,
        objective_value=100.0,
        is_valid=True
    )

    decoded = adapter.decode_solution(solution)
    expected = np.array([1, 2, 0])

    assert np.array_equal(decoded, expected)


def test_qap_validate_solution_valid():
    """Test validation of valid permutation"""
    adapter = QAPAdapter()
    adapter.n = 3

    valid_perm = np.array([2, 0, 1])
    result = adapter.validate_solution(valid_perm)

    assert result.is_valid
    assert len(result.constraint_violations) == 0
    assert len(result.violation_magnitudes) == 0


def test_qap_validate_solution_invalid():
    """Test validation of invalid permutation"""
    adapter = QAPAdapter()
    adapter.n = 3

    # Invalid - has duplicate
    invalid_perm = np.array([0, 0, 1])
    result = adapter.validate_solution(invalid_perm)

    assert not result.is_valid
    assert len(result.constraint_violations) > 0
    assert "permutation" in result.constraint_violations[0].lower()


def test_qap_permutation_to_matrix():
    """Test conversion of permutation to permutation matrix"""
    adapter = QAPAdapter()
    adapter.n = 3

    perm = np.array([1, 2, 0])
    P = adapter._permutation_to_matrix(perm)

    expected = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])

    assert np.array_equal(P, expected)
    # Verify it's a valid permutation matrix (doubly stochastic)
    assert np.allclose(P.sum(axis=0), 1)
    assert np.allclose(P.sum(axis=1), 1)


def test_qap_permutation_constraints():
    """Test generation of permutation matrix constraints"""
    adapter = QAPAdapter()
    adapter.n = 3

    constraints = adapter._permutation_constraints()

    # Should have 2n constraints (n row sums + n column sums)
    assert constraints.shape[0] == 6  # 2 * 3
    assert constraints.shape[1] == 9  # 3 * 3

    # Verify row sum constraints
    for i in range(3):
        row_constraint = constraints[i]
        # Should have 1s in positions corresponding to row i of matrix
        assert np.sum(row_constraint) == 3

    # Verify column sum constraints
    for j in range(3):
        col_constraint = constraints[3 + j]
        # Should have 1s in positions corresponding to column j of matrix
        assert np.sum(col_constraint) == 3


def test_qap_asymmetric_instance():
    """Test QAP with asymmetric matrices"""
    # Asymmetric flow matrix
    flow = np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0]])
    distance = np.array([[0, 8, 15], [8, 0, 13], [15, 13, 0]])

    adapter = QAPAdapter()
    problem = adapter.encode_problem({
        'flow_matrix': flow,
        'distance_matrix': distance
    })

    properties = adapter.verify_qap_properties()
    assert not properties['symmetric_flows']
    assert properties['symmetric_distances']
    assert properties['mathematical_valid']  # Still valid, just asymmetric


def test_qap_large_instance():
    """Test QAP with larger instance size"""
    n = 10
    np.random.seed(42)
    flow = np.random.randint(0, 20, (n, n))
    distance = np.random.randint(1, 50, (n, n))

    adapter = QAPAdapter()
    problem = adapter.encode_problem({
        'flow_matrix': flow,
        'distance_matrix': distance
    })

    assert problem.dimension == n
    assert adapter.n == n

    # Test with identity permutation
    identity = np.arange(n)
    obj = adapter.compute_objective(identity)
    assert obj >= 0  # Objective should be non-negative


def test_qap_zero_matrices():
    """Test QAP with zero flow/distance matrices"""
    flow = np.zeros((3, 3))
    distance = np.zeros((3, 3))

    adapter = QAPAdapter()
    problem = adapter.encode_problem({
        'flow_matrix': flow,
        'distance_matrix': distance
    })

    perm = np.array([0, 1, 2])
    obj = adapter.compute_objective(perm)

    assert obj == 0.0  # Zero matrices should give zero objective


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
