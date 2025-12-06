"""Matrix operations for scientific computing."""
import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix, issparse
from typing import Union, Tuple, Optional, List
import warnings
class MatrixOperations:
    """Matrix operations including decompositions, eigenvalues, and norms."""
    
    def multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiplication (instance method for compatibility)."""
        return self.matrix_multiply(A, B)
    
    def eigenvalues(self, matrix: np.ndarray) -> np.ndarray:
        """Compute eigenvalues of a matrix."""
        eigenvals, _ = self.eigendecomposition(matrix)
        return eigenvals
    
    def inverse(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix inverse."""
        matrix = self.validate_matrix(matrix, "matrix")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square for inversion")
        return np.linalg.inv(matrix)
    @staticmethod
    def validate_matrix(matrix: np.ndarray, name: str = "matrix") -> np.ndarray:
        """Validate matrix input and convert to numpy array."""
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be 2-dimensional")
        if matrix.size == 0:
            raise ValueError(f"{name} cannot be empty")
        return matrix
    @staticmethod
    def validate_vector(vector: np.ndarray, name: str = "vector") -> np.ndarray:
        """Validate vector input and convert to numpy array."""
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        if vector.ndim == 2 and min(vector.shape) == 1:
            vector = vector.flatten()
        elif vector.ndim != 1:
            raise ValueError(f"{name} must be 1-dimensional")
        return vector
    @staticmethod
    def matrix_multiply(A: np.ndarray, B: np.ndarray,
                       check_compatibility: bool = True) -> np.ndarray:
        """
        Matrix multiplication with dimension checking.
        Parameters:
            A: First matrix
            B: Second matrix or vector
            check_compatibility: Whether to check dimension compatibility
        Returns:
            Matrix product A @ B
        """
        A = MatrixOperations.validate_matrix(A, "A")
        if B.ndim == 1:
            B = MatrixOperations.validate_vector(B, "B")
        else:
            B = MatrixOperations.validate_matrix(B, "B")
        if check_compatibility:
            if A.shape[1] != B.shape[0]:
                raise ValueError(f"Incompatible dimensions: A{A.shape} @ B{B.shape}")
        # TODO: Consider CUDA acceleration for large matrices (>1000x1000)
        return A @ B
    @staticmethod
    def matrix_add(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix addition with dimension checking."""
        A = MatrixOperations.validate_matrix(A, "A")
        B = MatrixOperations.validate_matrix(B, "B")
        if A.shape != B.shape:
            raise ValueError(f"Incompatible shapes: A{A.shape} + B{B.shape}")
        return A + B
    @staticmethod
    def matrix_subtract(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix subtraction with dimension checking."""
        A = MatrixOperations.validate_matrix(A, "A")
        B = MatrixOperations.validate_matrix(B, "B")
        if A.shape != B.shape:
            raise ValueError(f"Incompatible shapes: A{A.shape} - B{B.shape}")
        return A - B
    @staticmethod
    def matrix_power(A: np.ndarray, n: int) -> np.ndarray:
        """
        Matrix power A^n using repeated squaring.
        Parameters:
            A: Square matrix
            n: Non-negative integer power
        Returns:
            A^n
        """
        A = MatrixOperations.validate_matrix(A, "A")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square for matrix power")
        if n < 0:
            raise ValueError("Power must be non-negative")
        if n == 0:
            return np.eye(A.shape[0])
        return np.linalg.matrix_power(A, n)
    @staticmethod
    def transpose(A: np.ndarray) -> np.ndarray:
        """Matrix transpose."""
        A = MatrixOperations.validate_matrix(A, "A")
        return A.T
    @staticmethod
    def conjugate_transpose(A: np.ndarray) -> np.ndarray:
        """Conjugate transpose (Hermitian transpose)."""
        A = MatrixOperations.validate_matrix(A, "A")
        return A.conj().T
    @staticmethod
    def trace(A: np.ndarray) -> complex:
        """Matrix trace (sum of diagonal elements)."""
        A = MatrixOperations.validate_matrix(A, "A")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square to compute trace")
        return np.trace(A)
    @staticmethod
    def determinant(A: np.ndarray) -> complex:
        """Matrix determinant."""
        A = MatrixOperations.validate_matrix(A, "A")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square to compute determinant")
        return np.linalg.det(A)
    @staticmethod
    def rank(A: np.ndarray, tolerance: Optional[float] = None) -> int:
        """
        Matrix rank using SVD.
        Parameters:
            A: Input matrix
            tolerance: Tolerance for rank determination
        Returns:
            Matrix rank
        """
        A = MatrixOperations.validate_matrix(A, "A")
        if tolerance is None:
            # Convert to float to handle integer matrices
            A_float = A.astype(float)
            # Handle edge case of zero matrix
            if np.all(A_float == 0):
                return 0
            tolerance = np.finfo(A_float.dtype).eps * max(A.shape) * np.max(np.abs(A_float))
        return np.linalg.matrix_rank(A, tol=tolerance)
    @staticmethod
    def condition_number(A: np.ndarray, p: Union[None, int, str] = None) -> float:
        """
        Matrix condition number.
        Parameters:
            A: Input matrix
            p: Order of the norm (None, 1, -1, 2, -2, inf, -inf, 'fro')
        Returns:
            Condition number
        """
        A = MatrixOperations.validate_matrix(A, "A")
        return np.linalg.cond(A, p)
    @staticmethod
    def frobenius_norm(A: np.ndarray) -> float:
        """Frobenius norm of matrix."""
        A = MatrixOperations.validate_matrix(A, "A")
        return np.linalg.norm(A, 'fro')
    @staticmethod
    def spectral_norm(A: np.ndarray) -> float:
        """Spectral norm (largest singular value)."""
        A = MatrixOperations.validate_matrix(A, "A")
        return np.linalg.norm(A, 2)
    @staticmethod
    def nuclear_norm(A: np.ndarray) -> float:
        """Nuclear norm (sum of singular values)."""
        A = MatrixOperations.validate_matrix(A, "A")
        return np.linalg.norm(A, 'nuc')
class MatrixDecompositions:
    """
    Matrix decomposition algorithms.
    Features:
    - LU decomposition with partial pivoting
    - QR decomposition (Householder and Givens)
    - Cholesky decomposition
    - Singular Value Decomposition (SVD)
    - Eigenvalue decomposition
    - Schur decomposition
    """
    @staticmethod
    def lu_decomposition(A: np.ndarray, permute_l: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        LU decomposition with partial pivoting.
        Parameters:
            A: Input matrix
            permute_l: Whether to apply permutations to L
        Returns:
            Tuple of (P, L, U) where PA = LU
        """
        A = MatrixOperations.validate_matrix(A, "A")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square for LU decomposition")
        P, L, U = la.lu(A, permute_l=permute_l)
        return P, L, U
    @staticmethod
    def qr_decomposition(A: np.ndarray, mode: str = 'full') -> Tuple[np.ndarray, np.ndarray]:
        """
        QR decomposition using Householder reflections.
        Parameters:
            A: Input matrix
            mode: 'full' or 'economic'
        Returns:
            Tuple of (Q, R) where A = QR
        """
        A = MatrixOperations.validate_matrix(A, "A")
        if mode not in ['full', 'economic']:
            raise ValueError("Mode must be 'full' or 'economic'")
        if mode == 'economic':
            Q, R = la.qr(A, mode='economic')
        else:
            Q, R = la.qr(A)
        return Q, R
    @staticmethod
    def cholesky_decomposition(A: np.ndarray, lower: bool = True) -> np.ndarray:
        """
        Cholesky decomposition for positive definite matrices.
        Parameters:
            A: Positive definite matrix
            lower: Whether to return lower triangular factor
        Returns:
            Cholesky factor L (if lower=True) or U (if lower=False)
        """
        A = MatrixOperations.validate_matrix(A, "A")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square for Cholesky decomposition")
        # Check if matrix is symmetric (within tolerance)
        if not np.allclose(A, A.T):
            warnings.warn("Matrix is not symmetric, results may be unreliable")
        try:
            L = la.cholesky(A, lower=lower)
            return L
        except la.LinAlgError as e:
            raise ValueError("Matrix is not positive definite") from e
    @staticmethod
    def svd(A: np.ndarray, full_matrices: bool = True,
            compute_uv: bool = True) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        Singular Value Decomposition.
        Parameters:
            A: Input matrix
            full_matrices: Whether to compute full U and Vt matrices
            compute_uv: Whether to compute U and Vt
        Returns:
            If compute_uv=True: (U, s, Vt) where A = U @ diag(s) @ Vt
            If compute_uv=False: s (singular values only)
        """
        A = MatrixOperations.validate_matrix(A, "A")
        return la.svd(A, full_matrices=full_matrices, compute_uv=compute_uv)
    @staticmethod
    def eigendecomposition(A: np.ndarray, right: bool = True,
                          left: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigenvalue decomposition for general matrices.
        Parameters:
            A: Square matrix
            right: Whether to compute right eigenvectors
            left: Whether to compute left eigenvectors
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        A = MatrixOperations.validate_matrix(A, "A")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square for eigendecomposition")
        eigenvals, eigenvecs = la.eig(A, right=right, left=left)
        # Sort by eigenvalue magnitude (descending)
        idx = np.argsort(np.abs(eigenvals))[::-1]
        eigenvals = eigenvals[idx]
        if eigenvecs is not None:
            if isinstance(eigenvecs, tuple):
                eigenvecs = (eigenvecs[0][:, idx], eigenvecs[1][:, idx])
            else:
                eigenvecs = eigenvecs[:, idx]
        return eigenvals, eigenvecs
    @staticmethod
    def symmetric_eigendecomposition(A: np.ndarray, subset_by_index: Optional[Tuple[int, int]] = None,
                                   subset_by_value: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigenvalue decomposition for symmetric/Hermitian matrices.
        Parameters:
            A: Symmetric/Hermitian matrix
            subset_by_index: Tuple (il, iu) to compute eigenvalues il through iu
            subset_by_value: Tuple (vl, vu) to compute eigenvalues in range [vl, vu]
        Returns:
            Tuple of (eigenvalues, eigenvectors) sorted in ascending order
        """
        A = MatrixOperations.validate_matrix(A, "A")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square")
        # Check if matrix is symmetric/Hermitian
        if np.iscomplexobj(A):
            if not np.allclose(A, A.conj().T):
                warnings.warn("Matrix is not Hermitian, results may be unreliable")
        else:
            if not np.allclose(A, A.T):
                warnings.warn("Matrix is not symmetric, results may be unreliable")
        eigenvals, eigenvecs = la.eigh(A, subset_by_index=subset_by_index,
                                      subset_by_value=subset_by_value)
        return eigenvals, eigenvecs
    @staticmethod
    def schur_decomposition(A: np.ndarray, output: str = 'real') -> Tuple[np.ndarray, np.ndarray]:
        """
        Schur decomposition.
        Parameters:
            A: Square matrix
            output: 'real' or 'complex'
        Returns:
            Tuple of (T, Z) where A = Z @ T @ Z.H
        """
        A = MatrixOperations.validate_matrix(A, "A")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square for Schur decomposition")
        T, Z = la.schur(A, output=output)
        return T, Z
class SpecialMatrices:
    """
    Special matrix types and properties.
    Features:
    - Matrix property checking (symmetric, orthogonal, etc.)
    - Special matrix generation
    - Matrix transformations
    """
    @staticmethod
    def is_symmetric(A: np.ndarray, tolerance: float = 1e-12) -> bool:
        """Check if matrix is symmetric."""
        A = MatrixOperations.validate_matrix(A, "A")
        if A.shape[0] != A.shape[1]:
            return False
        return np.allclose(A, A.T, atol=tolerance)
    @staticmethod
    def is_hermitian(A: np.ndarray, tolerance: float = 1e-12) -> bool:
        """Check if matrix is Hermitian."""
        A = MatrixOperations.validate_matrix(A, "A")
        if A.shape[0] != A.shape[1]:
            return False
        return np.allclose(A, A.conj().T, atol=tolerance)
    @staticmethod
    def is_orthogonal(A: np.ndarray, tolerance: float = 1e-12) -> bool:
        """Check if matrix is orthogonal (A.T @ A = I)."""
        A = MatrixOperations.validate_matrix(A, "A")
        if A.shape[0] != A.shape[1]:
            return False
        ATA = A.T @ A
        I = np.eye(A.shape[0])
        return np.allclose(ATA, I, atol=tolerance)
    @staticmethod
    def is_unitary(A: np.ndarray, tolerance: float = 1e-12) -> bool:
        """Check if matrix is unitary (A.H @ A = I)."""
        A = MatrixOperations.validate_matrix(A, "A")
        if A.shape[0] != A.shape[1]:
            return False
        AHA = A.conj().T @ A
        I = np.eye(A.shape[0])
        return np.allclose(AHA, I, atol=tolerance)
    @staticmethod
    def is_positive_definite(A: np.ndarray) -> bool:
        """Check if matrix is positive definite."""
        A = MatrixOperations.validate_matrix(A, "A")
        if A.shape[0] != A.shape[1]:
            return False
        if not SpecialMatrices.is_symmetric(A):
            return False
        try:
            la.cholesky(A)
            return True
        except la.LinAlgError:
            return False
    @staticmethod
    def is_positive_semidefinite(A: np.ndarray) -> bool:
        """Check if matrix is positive semidefinite."""
        A = MatrixOperations.validate_matrix(A, "A")
        if A.shape[0] != A.shape[1]:
            return False
        if not SpecialMatrices.is_symmetric(A):
            return False
        eigenvals = la.eigvals(A)
        return np.all(eigenvals >= -1e-12)  # Allow small numerical errors
    @staticmethod
    def make_symmetric(A: np.ndarray) -> np.ndarray:
        """Make matrix symmetric by (A + A.T) / 2."""
        A = MatrixOperations.validate_matrix(A, "A")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square")
        return (A + A.T) / 2
    @staticmethod
    def make_hermitian(A: np.ndarray) -> np.ndarray:
        """Make matrix Hermitian by (A + A.H) / 2."""
        A = MatrixOperations.validate_matrix(A, "A")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square")
        return (A + A.conj().T) / 2
    @staticmethod
    def gram_schmidt(A: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Gram-Schmidt orthogonalization of matrix columns.
        Parameters:
            A: Input matrix
            normalize: Whether to normalize the orthogonal vectors
        Returns:
            Matrix with orthogonal (or orthonormal) columns
        """
        A = MatrixOperations.validate_matrix(A, "A")
        m, n = A.shape
        Q = np.zeros_like(A)
        for j in range(n):
            v = A[:, j].copy()
            # Orthogonalize against previous vectors
            for i in range(j):
                proj = np.dot(Q[:, i], v) * Q[:, i]
                v = v - proj
            # Normalize if requested
            if normalize:
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-12:  # Avoid division by zero
                    v = v / norm_v
                else:
                    warnings.warn(f"Column {j} is linearly dependent")
            Q[:, j] = v
        return Q
    @staticmethod
    def householder_reflector(x: np.ndarray) -> np.ndarray:
        """
        Compute Householder reflector matrix.
        Parameters:
            x: Input vector
        Returns:
            Householder matrix H such that Hx = ||x||e_1
        """
        x = MatrixOperations.validate_vector(x, "x")
        n = len(x)
        e1 = np.zeros(n)
        e1[0] = 1
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-12:
            return np.eye(n)
        v = x + np.sign(x[0]) * norm_x * e1
        v = v / np.linalg.norm(v)
        H = np.eye(n) - 2 * np.outer(v, v)
        return H
    @staticmethod
    def givens_rotation(a: float, b: float) -> Tuple[float, float, np.ndarray]:
        """
        Compute Givens rotation matrix.
        Parameters:
            a, b: Values to rotate
        Returns:
            Tuple of (c, s, G) where G is 2x2 Givens matrix
        """
        if abs(b) < 1e-12:
            c, s = 1.0, 0.0
        elif abs(a) < 1e-12:
            c, s = 0.0, 1.0
        elif abs(b) > abs(a):
            t = a / b
            s = 1.0 / np.sqrt(1 + t**2)
            c = s * t
        else:
            t = b / a
            c = 1.0 / np.sqrt(1 + t**2)
            s = c * t
        G = np.array([[c, s], [-s, c]])
        return c, s, G
def create_test_matrices() -> dict:
    """Create a set of test matrices for validation."""
    matrices = {}
    # Random matrices
    np.random.seed(42)
    matrices['random_3x3'] = np.random.randn(3, 3)
    matrices['random_5x5'] = np.random.randn(5, 5)
    # Symmetric matrix
    A = np.random.randn(4, 4)
    matrices['symmetric_4x4'] = A + A.T
    # Positive definite matrix
    A = np.random.randn(3, 3)
    matrices['positive_definite_3x3'] = A.T @ A + 0.1 * np.eye(3)
    # Orthogonal matrix (from QR decomposition)
    A = np.random.randn(4, 4)
    Q, _ = la.qr(A)
    matrices['orthogonal_4x4'] = Q
    # Singular matrix
    matrices['singular_3x3'] = np.array([[1, 2, 3], [2, 4, 6], [1, 2, 3]])
    # Hilbert matrix (ill-conditioned)
    n = 5
    matrices['hilbert_5x5'] = np.array([[1/(i+j+1) for j in range(n)] for i in range(n)])
    return matrices