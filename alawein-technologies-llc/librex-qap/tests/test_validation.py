"""
Unit tests for Librex.QAP/validation.py module.

This test suite addresses the 13% coverage gap in validation.py.
Target: Increase validation.py coverage from 13% to 90%+

Tests cover:
- QAP problem validation
- Permutation validation
- Doubly-stochastic matrix validation
- Error message generation
"""

import numpy as np
import pytest
from Librex.QAP.validation import (
    validate_qap_problem,
    validate_permutation,
    validate_permutation_vector,
    validate_doubly_stochastic,
)


# ============================================================================
# TEST CLASS: QAP Problem Validation
# ============================================================================


class TestQAPProblemValidation:
    """Test validate_qap_problem function."""

    def test_validate_qap_problem_valid(self):
        """Test validation passes for valid QAP problem."""
        A = np.random.rand(5, 5)
        B = np.random.rand(5, 5)
        # Should not raise
        validate_qap_problem(A, B)

    def test_validate_qap_problem_non_square_A(self):
        """Test validation fails for non-square A."""
        A = np.random.rand(5, 6)
        B = np.random.rand(5, 5)
        with pytest.raises(ValueError, match="must be square"):
            validate_qap_problem(A, B)

    def test_validate_qap_problem_non_square_B(self):
        """Test validation fails for non-square B."""
        A = np.random.rand(5, 5)
        B = np.random.rand(5, 6)
        with pytest.raises(ValueError, match="must be square"):
            validate_qap_problem(A, B)

    def test_validate_qap_problem_size_mismatch(self):
        """Test validation fails when A and B have different sizes."""
        A = np.random.rand(5, 5)
        B = np.random.rand(6, 6)
        with pytest.raises(ValueError, match="same shape"):
            validate_qap_problem(A, B)

    def test_validate_qap_problem_wrong_dimensions(self):
        """Test validation fails for wrong number of dimensions."""
        A = np.random.rand(5, 5, 2)  # 3D array
        B = np.random.rand(5, 5)
        with pytest.raises(ValueError):
            validate_qap_problem(A, B)

    def test_validate_qap_problem_n1(self):
        """Test validation passes for n=1 problem."""
        A = np.array([[5.0]])
        B = np.array([[3.0]])
        validate_qap_problem(A, B)  # Should not raise


# ============================================================================
# TEST CLASS: Permutation Validation
# ============================================================================


class TestPermutationValidation:
    """Test validate_permutation function."""

    def test_validate_permutation_valid_vector(self):
        """Test validation passes for valid permutation vector."""
        perm = np.array([2, 0, 3, 1])
        validate_permutation_vector(perm)  # Should not raise

    def test_validate_permutation_valid_matrix(self):
        """Test validation passes for valid permutation matrix."""
        P = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])
        validate_permutation(P)  # Should not raise

    def test_validate_permutation_duplicate_in_vector(self):
        """Test validation fails for duplicate values in vector."""
        perm = np.array([0, 1, 1, 2])  # Duplicate 1
        with pytest.raises(ValueError, match="duplicate"):
            validate_permutation_vector(perm)

    def test_validate_permutation_out_of_range(self):
        """Test validation fails for out-of-range indices."""
        perm = np.array([0, 1, 5, 3])  # 5 is out of range for n=4
        with pytest.raises(ValueError):
            validate_permutation_vector(perm)

    def test_validate_permutation_negative(self):
        """Test validation fails for negative indices."""
        perm = np.array([0, -1, 2, 3])
        with pytest.raises(ValueError):
            validate_permutation_vector(perm)

    def test_validate_permutation_matrix_non_binary(self):
        """Test validation fails for non-binary permutation matrix."""
        P = np.array([
            [0.5, 0.5, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])
        with pytest.raises(ValueError, match="binary|0 or 1"):
            validate_permutation(P)

    def test_validate_permutation_matrix_multiple_ones_in_row(self):
        """Test validation fails for multiple 1s in a row."""
        P = np.array([
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ])
        with pytest.raises(ValueError, match="exactly one 1|row"):
            validate_permutation(P)

    def test_validate_permutation_matrix_multiple_ones_in_col(self):
        """Test validation fails for multiple 1s in a column."""
        P = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ])
        with pytest.raises(ValueError, match="exactly one 1|column"):
            validate_permutation(P)

    def test_validate_permutation_empty_row(self):
        """Test validation fails for row with no 1s."""
        P = np.array([
            [1, 0, 0],
            [0, 0, 0],  # Empty row
            [0, 1, 0],
        ])
        with pytest.raises(ValueError):
            validate_permutation(P)


# ============================================================================
# TEST CLASS: Doubly-Stochastic Validation
# ============================================================================


class TestDoublyStochasticValidation:
    """Test validate_doubly_stochastic function."""

    def test_validate_doubly_stochastic_valid(self):
        """Test validation passes for valid doubly-stochastic matrix."""
        X = np.array([
            [0.3, 0.4, 0.3],
            [0.3, 0.2, 0.5],
            [0.4, 0.4, 0.2],
        ])
        validate_doubly_stochastic(X, tol=1e-9)  # Should not raise

    def test_validate_doubly_stochastic_identity(self):
        """Test validation passes for identity permutation."""
        X = np.eye(5)
        validate_doubly_stochastic(X)  # Should not raise

    def test_validate_doubly_stochastic_uniform(self):
        """Test validation passes for uniform matrix."""
        n = 10
        X = np.ones((n, n)) / n
        validate_doubly_stochastic(X)  # Should not raise

    def test_validate_doubly_stochastic_negative_values(self):
        """Test validation fails for negative values."""
        X = np.array([
            [0.5, 0.6, -0.1],
            [0.3, 0.2, 0.5],
            [0.2, 0.2, 0.6],
        ])
        with pytest.raises(ValueError, match="non-negative"):
            validate_doubly_stochastic(X)

    def test_validate_doubly_stochastic_row_sum_violation(self):
        """Test validation fails when row sums != 1."""
        X = np.array([
            [0.4, 0.4, 0.4],  # Row sum = 1.2
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.2],
        ])
        with pytest.raises(ValueError, match="row"):
            validate_doubly_stochastic(X, tol=1e-6)

    def test_validate_doubly_stochastic_col_sum_violation(self):
        """Test validation fails when column sums != 1."""
        # Create matrix with exact row sums but wrong column sums
        X = np.array([
            [0.5, 0.3, 0.2],  # Row sum = 1.0
            [0.3, 0.3, 0.4],  # Row sum = 1.0
            [0.1, 0.2, 0.7],  # Row sum = 1.0
        ])
        # Column sums: [0.9, 0.8, 1.3] - all wrong
        with pytest.raises(ValueError, match="column|row"):
            validate_doubly_stochastic(X, tol=1e-6)

    def test_validate_doubly_stochastic_non_square(self):
        """Test validation fails for non-square matrix."""
        X = np.random.rand(3, 4)
        with pytest.raises(ValueError, match="square"):
            validate_doubly_stochastic(X)

    def test_validate_doubly_stochastic_negative_check(self):
        """Test validation detects negative values correctly."""
        X = np.ones((3, 3)) / 3
        X[1, 1] = -0.1
        with pytest.raises(ValueError, match="non-negative"):
            validate_doubly_stochastic(X)

    def test_validate_doubly_stochastic_tolerance(self):
        """Test validation with different tolerance levels."""
        X = np.array([
            [0.35, 0.33, 0.32],
            [0.33, 0.35, 0.32],
            [0.32, 0.32, 0.36],
        ])
        # Should pass with loose tolerance (rows sum to 1.0, cols sum to 1.0)
        validate_doubly_stochastic(X, tol=0.01)


# ============================================================================
# TEST CLASS: Edge Cases
# ============================================================================


class TestValidationEdgeCases:
    """Test edge cases in validation."""

    def test_validate_qap_problem_n1(self):
        """Test QAP validation for n=1."""
        A = np.array([[1.0]])
        B = np.array([[2.0]])
        validate_qap_problem(A, B)

    def test_validate_permutation_n1(self):
        """Test permutation validation for n=1."""
        perm = np.array([0])
        validate_permutation_vector(perm)

        P = np.array([[1.0]])
        validate_permutation(P)

    def test_validate_doubly_stochastic_n1(self):
        """Test doubly-stochastic validation for n=1."""
        X = np.array([[1.0]])
        validate_doubly_stochastic(X)

    def test_validate_qap_problem_large(self):
        """Test QAP validation for large matrix."""
        n = 200
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        validate_qap_problem(A, B)

    def test_validate_permutation_large(self):
        """Test permutation validation for large permutation."""
        n = 200
        perm = np.random.permutation(n)
        validate_permutation_vector(perm)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
