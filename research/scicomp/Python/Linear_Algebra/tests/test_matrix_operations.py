"""
Tests for Matrix Operations
Comprehensive test suite for matrix operations including basic arithmetic,
decompositions, properties, and special matrices.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import warnings
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.matrix_operations import MatrixOperations, MatrixDecompositions, SpecialMatrices
class TestMatrixOperations:
    """Test basic matrix operations."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.A = np.array([[1, 2], [3, 4]])
        self.B = np.array([[5, 6], [7, 8]])
        self.C = np.array([[1, 2, 3], [4, 5, 6]])
        self.v = np.array([1, 2])
        self.w = np.array([2, 3, 1])
    def test_validate_matrix(self):
        """Test matrix validation."""
        # Valid matrix
        A_valid = MatrixOperations.validate_matrix(self.A)
        assert A_valid.shape == (2, 2)
        # List to array conversion
        A_list = [[1, 2], [3, 4]]
        A_converted = MatrixOperations.validate_matrix(A_list)
        assert_array_almost_equal(A_converted, self.A)
        # Invalid dimensions
        with pytest.raises(ValueError):
            MatrixOperations.validate_matrix(np.array([1, 2, 3]))  # 1D
        with pytest.raises(ValueError):
            MatrixOperations.validate_matrix(np.array([]))  # Empty
    def test_matrix_multiply(self):
        """Test matrix multiplication."""
        # Matrix-matrix multiplication
        result = MatrixOperations.matrix_multiply(self.A, self.B)
        expected = np.array([[19, 22], [43, 50]])
        assert_array_almost_equal(result, expected)
        # Matrix-vector multiplication
        result = MatrixOperations.matrix_multiply(self.A, self.v)
        expected = np.array([5, 11])
        assert_array_almost_equal(result, expected)
        # Incompatible dimensions
        with pytest.raises(ValueError):
            MatrixOperations.matrix_multiply(self.A, self.C)
    def test_matrix_add_subtract(self):
        """Test matrix addition and subtraction."""
        # Addition
        result = MatrixOperations.matrix_add(self.A, self.B)
        expected = np.array([[6, 8], [10, 12]])
        assert_array_almost_equal(result, expected)
        # Subtraction
        result = MatrixOperations.matrix_subtract(self.A, self.B)
        expected = np.array([[-4, -4], [-4, -4]])
        assert_array_almost_equal(result, expected)
        # Incompatible shapes
        with pytest.raises(ValueError):
            MatrixOperations.matrix_add(self.A, self.C)
    def test_matrix_power(self):
        """Test matrix power."""
        # A^2
        result = MatrixOperations.matrix_power(self.A, 2)
        expected = self.A @ self.A
        assert_array_almost_equal(result, expected)
        # A^0 = I
        result = MatrixOperations.matrix_power(self.A, 0)
        expected = np.eye(2)
        assert_array_almost_equal(result, expected)
        # Non-square matrix
        with pytest.raises(ValueError):
            MatrixOperations.matrix_power(self.C, 2)
        # Negative power
        with pytest.raises(ValueError):
            MatrixOperations.matrix_power(self.A, -1)
    def test_transpose(self):
        """Test matrix transpose."""
        result = MatrixOperations.transpose(self.A)
        expected = np.array([[1, 3], [2, 4]])
        assert_array_almost_equal(result, expected)
        result = MatrixOperations.transpose(self.C)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        assert_array_almost_equal(result, expected)
    def test_conjugate_transpose(self):
        """Test conjugate transpose."""
        # Real matrix
        result = MatrixOperations.conjugate_transpose(self.A)
        expected = self.A.T
        assert_array_almost_equal(result, expected)
        # Complex matrix
        A_complex = np.array([[1+2j, 3-1j], [2+1j, 4]])
        result = MatrixOperations.conjugate_transpose(A_complex)
        expected = np.array([[1-2j, 2-1j], [3+1j, 4]])
        assert_array_almost_equal(result, expected)
    def test_trace(self):
        """Test matrix trace."""
        result = MatrixOperations.trace(self.A)
        expected = 5  # 1 + 4
        assert result == expected
        # Non-square matrix
        with pytest.raises(ValueError):
            MatrixOperations.trace(self.C)
    def test_determinant(self):
        """Test matrix determinant."""
        result = MatrixOperations.determinant(self.A)
        expected = -2  # 1*4 - 2*3
        assert_allclose(result, expected)
        # 3x3 matrix
        A_3x3 = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        result = MatrixOperations.determinant(A_3x3)
        expected = 1  # Calculated by hand
        assert_allclose(result, expected)
        # Non-square matrix
        with pytest.raises(ValueError):
            MatrixOperations.determinant(self.C)
    def test_rank(self):
        """Test matrix rank."""
        # Full rank matrix
        result = MatrixOperations.rank(self.A)
        assert result == 2
        # Rank deficient matrix
        singular_matrix = np.array([[1, 2], [2, 4]])  # Rank 1
        result = MatrixOperations.rank(singular_matrix)
        assert result == 1
        # Zero matrix
        zero_matrix = np.zeros((3, 3))
        result = MatrixOperations.rank(zero_matrix)
        assert result == 0
    def test_condition_number(self):
        """Test condition number."""
        # Well-conditioned matrix
        result = MatrixOperations.condition_number(self.A)
        assert result > 1  # Should be finite and > 1
        # Ill-conditioned matrix (Hilbert matrix)
        H = np.array([[1, 1/2, 1/3], [1/2, 1/3, 1/4], [1/3, 1/4, 1/5]])
        result = MatrixOperations.condition_number(H)
        assert result > 100  # Hilbert matrices are ill-conditioned
    def test_norms(self):
        """Test matrix norms."""
        # Frobenius norm
        result = MatrixOperations.frobenius_norm(self.A)
        expected = np.sqrt(1**2 + 2**2 + 3**2 + 4**2)
        assert_allclose(result, expected)
        # Spectral norm
        result = MatrixOperations.spectral_norm(self.A)
        U, s, Vt = np.linalg.svd(self.A)
        expected = s[0]  # Largest singular value
        assert_allclose(result, expected)
        # Nuclear norm
        result = MatrixOperations.nuclear_norm(self.A)
        expected = np.sum(s)  # Sum of singular values
        assert_allclose(result, expected)
class TestMatrixDecompositions:
    """Test matrix decomposition algorithms."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.A = np.random.randn(4, 4)
        self.A_spd = self.A.T @ self.A + 0.1 * np.eye(4)  # Symmetric positive definite
        self.A_rect = np.random.randn(5, 3)
    def test_lu_decomposition(self):
        """Test LU decomposition."""
        P, L, U = MatrixDecompositions.lu_decomposition(self.A)
        # Check dimensions
        assert P.shape == (4, 4)
        assert L.shape == (4, 4)
        assert U.shape == (4, 4)
        # Check decomposition: PA = LU
        assert_allclose(P @ self.A, L @ U, rtol=1e-10)
        # Check L is lower triangular with unit diagonal
        assert_allclose(np.triu(L, 1), 0, atol=1e-15)
        assert_allclose(np.diag(L), 1, rtol=1e-15)
        # Check U is upper triangular
        assert_allclose(np.tril(U, -1), 0, atol=1e-15)
    def test_qr_decomposition(self):
        """Test QR decomposition."""
        # Full QR
        Q, R = MatrixDecompositions.qr_decomposition(self.A_rect, mode='full')
        # Check decomposition: A = QR
        assert_allclose(self.A_rect, Q @ R, rtol=1e-10)
        # Check Q is orthogonal
        assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), rtol=1e-10)
        # Check R is upper triangular
        assert_allclose(np.tril(R, -1), 0, atol=1e-15)
        # Economic QR
        Q_econ, R_econ = MatrixDecompositions.qr_decomposition(self.A_rect, mode='economic')
        assert Q_econ.shape == (5, 3)
        assert R_econ.shape == (3, 3)
        assert_allclose(self.A_rect, Q_econ @ R_econ, rtol=1e-10)
    def test_cholesky_decomposition(self):
        """Test Cholesky decomposition."""
        # Lower triangular factor
        L = MatrixDecompositions.cholesky_decomposition(self.A_spd, lower=True)
        # Check decomposition: A = LL^T
        assert_allclose(self.A_spd, L @ L.T, rtol=1e-10)
        # Check L is lower triangular
        assert_allclose(np.triu(L, 1), 0, atol=1e-15)
        # Upper triangular factor
        U = MatrixDecompositions.cholesky_decomposition(self.A_spd, lower=False)
        assert_allclose(self.A_spd, U.T @ U, rtol=1e-10)
        # Non-positive definite matrix should fail
        A_not_spd = np.array([[1, 2], [2, 1]])
        with pytest.raises(ValueError):
            MatrixDecompositions.cholesky_decomposition(A_not_spd)
    def test_svd(self):
        """Test Singular Value Decomposition."""
        # Full SVD
        U, s, Vt = MatrixDecompositions.svd(self.A_rect, full_matrices=True)
        # Check decomposition: A = U @ diag(s) @ Vt
        assert_allclose(self.A_rect, U @ np.diag(s) @ Vt, rtol=1e-10)
        # Check orthogonality
        assert_allclose(U.T @ U, np.eye(U.shape[1]), rtol=1e-10)
        assert_allclose(Vt @ Vt.T, np.eye(Vt.shape[0]), rtol=1e-10)
        # Check singular values are non-negative and sorted
        assert np.all(s >= 0)
        assert np.all(s[:-1] >= s[1:])
        # Singular values only
        s_only = MatrixDecompositions.svd(self.A_rect, compute_uv=False)
        assert_allclose(s, s_only)
    def test_eigendecomposition(self):
        """Test eigenvalue decomposition."""
        # General matrix
        eigenvals, eigenvecs = MatrixDecompositions.eigendecomposition(self.A)
        # Check decomposition: A @ v = λ @ v for each eigenpair
        for i in range(len(eigenvals)):
            lhs = self.A @ eigenvecs[:, i]
            rhs = eigenvals[i] * eigenvecs[:, i]
            assert_allclose(lhs, rhs, rtol=1e-10)
        # Check that eigenvalues are sorted by magnitude
        mags = np.abs(eigenvals)
        assert np.all(mags[:-1] >= mags[1:])
    def test_symmetric_eigendecomposition(self):
        """Test symmetric eigenvalue decomposition."""
        eigenvals, eigenvecs = MatrixDecompositions.symmetric_eigendecomposition(self.A_spd)
        # Check decomposition
        for i in range(len(eigenvals)):
            lhs = self.A_spd @ eigenvecs[:, i]
            rhs = eigenvals[i] * eigenvecs[:, i]
            assert_allclose(lhs, rhs, rtol=1e-10)
        # Check that eigenvalues are real and sorted
        assert np.all(np.isreal(eigenvals))
        assert np.all(eigenvals[:-1] <= eigenvals[1:])  # Ascending order
        # Check that eigenvectors are orthonormal
        assert_allclose(eigenvecs.T @ eigenvecs, np.eye(4), rtol=1e-10)
        # For SPD matrix, all eigenvalues should be positive
        assert np.all(eigenvals > 0)
    def test_schur_decomposition(self):
        """Test Schur decomposition."""
        T, Z = MatrixDecompositions.schur_decomposition(self.A)
        # Check decomposition: A = Z @ T @ Z^H
        assert_allclose(self.A, Z @ T @ Z.T, rtol=1e-10)
        # Check Z is orthogonal
        assert_allclose(Z.T @ Z, np.eye(4), rtol=1e-10)
        # Check T is upper triangular (or quasi-upper for real Schur)
        if np.allclose(T, np.triu(T)):
            # Upper triangular
            assert_allclose(np.tril(T, -1), 0, atol=1e-12)
class TestSpecialMatrices:
    """Test special matrix properties and operations."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Symmetric matrix
        A = np.random.randn(3, 3)
        self.symmetric = A + A.T
        # Hermitian matrix
        A_complex = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        self.hermitian = A_complex + A_complex.conj().T
        # Orthogonal matrix
        Q, _ = np.linalg.qr(np.random.randn(3, 3))
        self.orthogonal = Q
        # Positive definite matrix
        A = np.random.randn(3, 3)
        self.positive_definite = A.T @ A + 0.1 * np.eye(3)
        # Positive semidefinite matrix (rank deficient)
        A = np.random.randn(3, 2)
        self.positive_semidefinite = A @ A.T
    def test_is_symmetric(self):
        """Test symmetry detection."""
        assert SpecialMatrices.is_symmetric(self.symmetric)
        assert not SpecialMatrices.is_symmetric(np.random.randn(3, 3))
        # Non-square matrix
        assert not SpecialMatrices.is_symmetric(np.random.randn(3, 2))
    def test_is_hermitian(self):
        """Test Hermitian property detection."""
        assert SpecialMatrices.is_hermitian(self.hermitian)
        assert SpecialMatrices.is_hermitian(self.symmetric)  # Real symmetric is Hermitian
        assert not SpecialMatrices.is_hermitian(np.random.randn(3, 3) + 1j * np.random.randn(3, 3))
    def test_is_orthogonal(self):
        """Test orthogonality detection."""
        assert SpecialMatrices.is_orthogonal(self.orthogonal)
        assert SpecialMatrices.is_orthogonal(np.eye(3))  # Identity is orthogonal
        assert not SpecialMatrices.is_orthogonal(np.random.randn(3, 3))
    def test_is_unitary(self):
        """Test unitary property detection."""
        assert SpecialMatrices.is_unitary(self.orthogonal)  # Real orthogonal is unitary
        # Complex unitary matrix
        A = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        Q, _ = np.linalg.qr(A)
        assert SpecialMatrices.is_unitary(Q)
    def test_is_positive_definite(self):
        """Test positive definiteness detection."""
        assert SpecialMatrices.is_positive_definite(self.positive_definite)
        assert not SpecialMatrices.is_positive_definite(self.positive_semidefinite)
        assert not SpecialMatrices.is_positive_definite(np.random.randn(3, 3))
    def test_is_positive_semidefinite(self):
        """Test positive semidefiniteness detection."""
        assert SpecialMatrices.is_positive_semidefinite(self.positive_definite)
        assert SpecialMatrices.is_positive_semidefinite(self.positive_semidefinite)
        # Negative definite matrix
        negative_definite = -self.positive_definite
        assert not SpecialMatrices.is_positive_semidefinite(negative_definite)
    def test_make_symmetric(self):
        """Test symmetrization."""
        A = np.random.randn(3, 3)
        A_sym = SpecialMatrices.make_symmetric(A)
        assert SpecialMatrices.is_symmetric(A_sym)
        assert_allclose(A_sym, (A + A.T) / 2)
    def test_make_hermitian(self):
        """Test Hermitian-ization."""
        A = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        A_herm = SpecialMatrices.make_hermitian(A)
        assert SpecialMatrices.is_hermitian(A_herm)
        assert_allclose(A_herm, (A + A.conj().T) / 2)
    def test_gram_schmidt(self):
        """Test Gram-Schmidt orthogonalization."""
        A = np.random.randn(4, 3)
        Q = SpecialMatrices.gram_schmidt(A, normalize=True)
        # Check orthonormality
        assert_allclose(Q.T @ Q, np.eye(3), rtol=1e-10)
        # Check that column space is preserved
        # (Q should span the same space as A)
        for i in range(3):
            # Each column of A should be expressible as linear combination of Q columns
            coeffs = np.linalg.lstsq(Q, A[:, i], rcond=None)[0]
            reconstructed = Q @ coeffs
            assert_allclose(reconstructed, A[:, i], rtol=1e-10)
    def test_householder_reflector(self):
        """Test Householder reflector construction."""
        x = np.array([3, 4, 0])
        H = SpecialMatrices.householder_reflector(x)
        # Check that H is orthogonal
        assert SpecialMatrices.is_orthogonal(H)
        # Check that Hx = ||x|| e_1
        Hx = H @ x
        expected = np.array([np.linalg.norm(x), 0, 0])
        assert_allclose(np.abs(Hx), np.abs(expected), rtol=1e-10)
    def test_givens_rotation(self):
        """Test Givens rotation construction."""
        a, b = 3.0, 4.0
        c, s, G = SpecialMatrices.givens_rotation(a, b)
        # Check rotation properties
        assert_allclose(c**2 + s**2, 1, rtol=1e-15)
        # Check that rotation zeros out b
        result = G @ np.array([a, b])
        assert_allclose(result[1], 0, atol=1e-15)
        # Check that G is orthogonal
        assert SpecialMatrices.is_orthogonal(G)
def test_create_test_matrices():
    """Test test matrix creation."""
    matrices = MatrixOperations.create_test_matrices()
    assert 'random_3x3' in matrices
    assert 'symmetric_4x4' in matrices
    assert 'positive_definite_3x3' in matrices
    assert 'orthogonal_4x4' in matrices
    assert 'singular_3x3' in matrices
    assert 'hilbert_5x5' in matrices
    # Check properties
    assert SpecialMatrices.is_symmetric(matrices['symmetric_4x4'])
    assert SpecialMatrices.is_positive_definite(matrices['positive_definite_3x3'])
    assert SpecialMatrices.is_orthogonal(matrices['orthogonal_4x4'])
    # Singular matrix should have zero determinant
    assert abs(np.linalg.det(matrices['singular_3x3'])) < 1e-10
    # Hilbert matrix should be ill-conditioned
    assert np.linalg.cond(matrices['hilbert_5x5']) > 1e5
if __name__ == "__main__":
    pytest.main([__file__])