"""
Linear Systems Solvers for Scientific Computing
Comprehensive algorithms for solving linear systems Ax = b including
direct methods, iterative methods, and specialized solvers.
"""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Union, Tuple, Optional, Callable, Dict, Any
import warnings
from dataclasses import dataclass
from .matrix_operations import MatrixOperations, MatrixDecompositions
@dataclass
class SolverResult:
    """Result container for linear system solvers."""
    solution: np.ndarray
    success: bool
    iterations: int
    residual_norm: float
    info: Dict[str, Any]
class DirectSolvers:
    """
    Direct methods for solving linear systems.
    Features:
    - LU decomposition with partial pivoting
    - Cholesky decomposition for symmetric positive definite systems
    - QR decomposition for overdetermined systems
    - SVD for rank-deficient systems
    - Specialized algorithms for structured matrices
    """
    @staticmethod
    def lu_solve(A: np.ndarray, b: np.ndarray, check_finite: bool = True) -> SolverResult:
        """
        Solve Ax = b using LU decomposition with partial pivoting.
        Parameters:
            A: Coefficient matrix
            b: Right-hand side vector
            check_finite: Whether to check for finite values
        Returns:
            SolverResult with solution and metadata
        """
        A = MatrixOperations.validate_matrix(A, "A")
        b = MatrixOperations.validate_vector(b, "b")
        if A.shape[0] != len(b):
            raise ValueError(f"Matrix and vector dimensions incompatible: {A.shape[0]} != {len(b)}")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square for LU solve")
        try:
            x = la.solve(A, b, check_finite=check_finite)
            # Compute residual
            residual = A @ x - b
            residual_norm = np.linalg.norm(residual)
            return SolverResult(
                solution=x,
                success=True,
                iterations=1,
                residual_norm=residual_norm,
                info={'method': 'LU', 'condition_number': np.linalg.cond(A)}
            )
        except la.LinAlgError as e:
            warnings.warn(f"LU solve failed: {e}")
            return SolverResult(
                solution=np.full(A.shape[1], np.nan),
                success=False,
                iterations=0,
                residual_norm=np.inf,
                info={'method': 'LU', 'error': str(e)}
            )
    @staticmethod
    def cholesky_solve(A: np.ndarray, b: np.ndarray, lower: bool = True) -> SolverResult:
        """
        Solve Ax = b using Cholesky decomposition for symmetric positive definite A.
        Parameters:
            A: Symmetric positive definite matrix
            b: Right-hand side vector
            lower: Whether to use lower triangular factor
        Returns:
            SolverResult with solution and metadata
        """
        A = MatrixOperations.validate_matrix(A, "A")
        b = MatrixOperations.validate_vector(b, "b")
        if A.shape[0] != len(b):
            raise ValueError(f"Matrix and vector dimensions incompatible: {A.shape[0]} != {len(b)}")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square for Cholesky solve")
        try:
            # Check if matrix is symmetric
            if not np.allclose(A, A.T):
                warnings.warn("Matrix is not symmetric, Cholesky may fail")
            x = la.solve(A, b, assume_a='pos')
            # Compute residual
            residual = A @ x - b
            residual_norm = np.linalg.norm(residual)
            return SolverResult(
                solution=x,
                success=True,
                iterations=1,
                residual_norm=residual_norm,
                info={'method': 'Cholesky', 'condition_number': np.linalg.cond(A)}
            )
        except la.LinAlgError as e:
            warnings.warn(f"Cholesky solve failed: {e}")
            return SolverResult(
                solution=np.full(A.shape[1], np.nan),
                success=False,
                iterations=0,
                residual_norm=np.inf,
                info={'method': 'Cholesky', 'error': str(e)}
            )
    @staticmethod
    def qr_solve(A: np.ndarray, b: np.ndarray, mode: str = 'full') -> SolverResult:
        """
        Solve Ax = b using QR decomposition (supports overdetermined systems).
        Parameters:
            A: Coefficient matrix (m x n, m >= n)
            b: Right-hand side vector
            mode: QR decomposition mode ('full' or 'economic')
        Returns:
            SolverResult with solution and metadata
        """
        A = MatrixOperations.validate_matrix(A, "A")
        b = MatrixOperations.validate_vector(b, "b")
        if A.shape[0] != len(b):
            raise ValueError(f"Matrix and vector dimensions incompatible: {A.shape[0]} != {len(b)}")
        try:
            Q, R = MatrixDecompositions.qr_decomposition(A, mode=mode)
            # Solve Rx = Q^T b
            Qt_b = Q.T @ b
            if mode == 'economic':
                x = la.solve_triangular(R, Qt_b)
            else:
                x = la.solve_triangular(R[:A.shape[1], :], Qt_b[:A.shape[1]])
            # Compute residual
            residual = A @ x - b
            residual_norm = np.linalg.norm(residual)
            return SolverResult(
                solution=x,
                success=True,
                iterations=1,
                residual_norm=residual_norm,
                info={
                    'method': 'QR',
                    'condition_number': np.linalg.cond(A),
                    'overdetermined': A.shape[0] > A.shape[1]
                }
            )
        except la.LinAlgError as e:
            warnings.warn(f"QR solve failed: {e}")
            return SolverResult(
                solution=np.full(A.shape[1], np.nan),
                success=False,
                iterations=0,
                residual_norm=np.inf,
                info={'method': 'QR', 'error': str(e)}
            )
    @staticmethod
    def svd_solve(A: np.ndarray, b: np.ndarray, rcond: Optional[float] = None) -> SolverResult:
        """
        Solve Ax = b using SVD (handles rank-deficient systems).
        Parameters:
            A: Coefficient matrix
            b: Right-hand side vector
            rcond: Relative condition number for rank determination
        Returns:
            SolverResult with solution and metadata
        """
        A = MatrixOperations.validate_matrix(A, "A")
        b = MatrixOperations.validate_vector(b, "b")
        if A.shape[0] != len(b):
            raise ValueError(f"Matrix and vector dimensions incompatible: {A.shape[0]} != {len(b)}")
        try:
            x, residuals, rank, s = la.lstsq(A, b, rcond=rcond)
            # Compute residual norm
            if residuals.size > 0:
                residual_norm = np.sqrt(residuals[0])
            else:
                residual = A @ x - b
                residual_norm = np.linalg.norm(residual)
            return SolverResult(
                solution=x,
                success=True,
                iterations=1,
                residual_norm=residual_norm,
                info={
                    'method': 'SVD',
                    'rank': rank,
                    'singular_values': s,
                    'condition_number': s[0] / s[-1] if s[-1] > 0 else np.inf
                }
            )
        except la.LinAlgError as e:
            warnings.warn(f"SVD solve failed: {e}")
            return SolverResult(
                solution=np.full(A.shape[1], np.nan),
                success=False,
                iterations=0,
                residual_norm=np.inf,
                info={'method': 'SVD', 'error': str(e)}
            )
    @staticmethod
    def tridiagonal_solve(diag: np.ndarray, upper: np.ndarray, lower: np.ndarray,
                         b: np.ndarray) -> SolverResult:
        """
        Solve tridiagonal system using Thomas algorithm.
        Parameters:
            diag: Main diagonal
            upper: Upper diagonal
            lower: Lower diagonal
            b: Right-hand side vector
        Returns:
            SolverResult with solution and metadata
        """
        diag = MatrixOperations.validate_vector(diag, "diag")
        upper = MatrixOperations.validate_vector(upper, "upper")
        lower = MatrixOperations.validate_vector(lower, "lower")
        b = MatrixOperations.validate_vector(b, "b")
        n = len(diag)
        if len(upper) != n - 1:
            raise ValueError(f"Upper diagonal must have length {n-1}, got {len(upper)}")
        if len(lower) != n - 1:
            raise ValueError(f"Lower diagonal must have length {n-1}, got {len(lower)}")
        if len(b) != n:
            raise ValueError(f"RHS vector must have length {n}, got {len(b)}")
        try:
            # Create tridiagonal matrix
            A = np.diag(diag) + np.diag(upper, 1) + np.diag(lower, -1)
            # Thomas algorithm (more efficient than general solver)
            x = np.zeros(n)
            # Forward elimination
            c_prime = np.zeros(n - 1)
            d_prime = np.zeros(n)
            c_prime[0] = upper[0] / diag[0]
            d_prime[0] = b[0] / diag[0]
            for i in range(1, n - 1):
                denom = diag[i] - lower[i-1] * c_prime[i-1]
                c_prime[i] = upper[i] / denom
                d_prime[i] = (b[i] - lower[i-1] * d_prime[i-1]) / denom
            d_prime[n-1] = (b[n-1] - lower[n-2] * d_prime[n-2]) / (diag[n-1] - lower[n-2] * c_prime[n-2])
            # Back substitution
            x[n-1] = d_prime[n-1]
            for i in range(n-2, -1, -1):
                x[i] = d_prime[i] - c_prime[i] * x[i+1]
            # Compute residual
            residual = A @ x - b
            residual_norm = np.linalg.norm(residual)
            return SolverResult(
                solution=x,
                success=True,
                iterations=1,
                residual_norm=residual_norm,
                info={'method': 'Thomas (Tridiagonal)'}
            )
        except Exception as e:
            warnings.warn(f"Tridiagonal solve failed: {e}")
            return SolverResult(
                solution=np.full(n, np.nan),
                success=False,
                iterations=0,
                residual_norm=np.inf,
                info={'method': 'Thomas (Tridiagonal)', 'error': str(e)}
            )
class IterativeSolvers:
    """
    Iterative methods for solving linear systems.
    Features:
    - Jacobi iteration
    - Gauss-Seidel iteration
    - Successive Over-Relaxation (SOR)
    - Conjugate Gradient method
    - GMRES method
    - BiCGSTAB method
    """
    @staticmethod
    def jacobi(A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None,
               max_iterations: int = 1000, tolerance: float = 1e-6) -> SolverResult:
        """
        Solve Ax = b using Jacobi iteration.
        Parameters:
            A: Coefficient matrix
            b: Right-hand side vector
            x0: Initial guess
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        Returns:
            SolverResult with solution and metadata
        """
        A = MatrixOperations.validate_matrix(A, "A")
        b = MatrixOperations.validate_vector(b, "b")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square for Jacobi iteration")
        if A.shape[0] != len(b):
            raise ValueError(f"Matrix and vector dimensions incompatible: {A.shape[0]} != {len(b)}")
        n = A.shape[0]
        if x0 is None:
            x = np.zeros(n)
        else:
            x = MatrixOperations.validate_vector(x0, "x0").copy()
        # Check diagonal dominance
        diag = np.diag(A)
        if np.any(np.abs(diag) < 1e-12):
            warnings.warn("Matrix has near-zero diagonal elements, Jacobi may not converge")
        # Extract diagonal and off-diagonal parts
        D = np.diag(diag)
        R = A - D
        residuals = []
        for iteration in range(max_iterations):
            # Jacobi update: x^(k+1) = D^(-1) * (b - R * x^(k))
            x_new = (b - R @ x) / diag
            # Check convergence
            residual = A @ x_new - b
            residual_norm = np.linalg.norm(residual)
            residuals.append(residual_norm)
            if residual_norm < tolerance:
                return SolverResult(
                    solution=x_new,
                    success=True,
                    iterations=iteration + 1,
                    residual_norm=residual_norm,
                    info={'method': 'Jacobi', 'residuals': residuals}
                )
            x = x_new
        return SolverResult(
            solution=x,
            success=False,
            iterations=max_iterations,
            residual_norm=residuals[-1] if residuals else np.inf,
            info={'method': 'Jacobi', 'residuals': residuals, 'converged': False}
        )
    @staticmethod
    def gauss_seidel(A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None,
                     max_iterations: int = 1000, tolerance: float = 1e-6) -> SolverResult:
        """
        Solve Ax = b using Gauss-Seidel iteration.
        Parameters:
            A: Coefficient matrix
            b: Right-hand side vector
            x0: Initial guess
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        Returns:
            SolverResult with solution and metadata
        """
        A = MatrixOperations.validate_matrix(A, "A")
        b = MatrixOperations.validate_vector(b, "b")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square for Gauss-Seidel iteration")
        if A.shape[0] != len(b):
            raise ValueError(f"Matrix and vector dimensions incompatible: {A.shape[0]} != {len(b)}")
        n = A.shape[0]
        if x0 is None:
            x = np.zeros(n)
        else:
            x = MatrixOperations.validate_vector(x0, "x0").copy()
        residuals = []
        for iteration in range(max_iterations):
            x_old = x.copy()
            # Gauss-Seidel update
            for i in range(n):
                if abs(A[i, i]) < 1e-12:
                    warnings.warn(f"Near-zero diagonal element at position {i}")
                    continue
                sum_ax = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
                x[i] = (b[i] - sum_ax) / A[i, i]
            # Check convergence
            residual = A @ x - b
            residual_norm = np.linalg.norm(residual)
            residuals.append(residual_norm)
            if residual_norm < tolerance:
                return SolverResult(
                    solution=x,
                    success=True,
                    iterations=iteration + 1,
                    residual_norm=residual_norm,
                    info={'method': 'Gauss-Seidel', 'residuals': residuals}
                )
        return SolverResult(
            solution=x,
            success=False,
            iterations=max_iterations,
            residual_norm=residuals[-1] if residuals else np.inf,
            info={'method': 'Gauss-Seidel', 'residuals': residuals, 'converged': False}
        )
    @staticmethod
    def sor(A: np.ndarray, b: np.ndarray, omega: float = 1.0, x0: Optional[np.ndarray] = None,
            max_iterations: int = 1000, tolerance: float = 1e-6) -> SolverResult:
        """
        Solve Ax = b using Successive Over-Relaxation (SOR).
        Parameters:
            A: Coefficient matrix
            b: Right-hand side vector
            omega: Relaxation parameter (0 < omega < 2)
            x0: Initial guess
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        Returns:
            SolverResult with solution and metadata
        """
        if not (0 < omega < 2):
            warnings.warn("SOR parameter omega should be in range (0, 2) for convergence")
        A = MatrixOperations.validate_matrix(A, "A")
        b = MatrixOperations.validate_vector(b, "b")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square for SOR iteration")
        if A.shape[0] != len(b):
            raise ValueError(f"Matrix and vector dimensions incompatible: {A.shape[0]} != {len(b)}")
        n = A.shape[0]
        if x0 is None:
            x = np.zeros(n)
        else:
            x = MatrixOperations.validate_vector(x0, "x0").copy()
        residuals = []
        for iteration in range(max_iterations):
            x_old = x.copy()
            # SOR update
            for i in range(n):
                if abs(A[i, i]) < 1e-12:
                    warnings.warn(f"Near-zero diagonal element at position {i}")
                    continue
                sum_ax = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
                x_gs = (b[i] - sum_ax) / A[i, i]  # Gauss-Seidel update
                x[i] = (1 - omega) * x_old[i] + omega * x_gs  # SOR update
            # Check convergence
            residual = A @ x - b
            residual_norm = np.linalg.norm(residual)
            residuals.append(residual_norm)
            if residual_norm < tolerance:
                return SolverResult(
                    solution=x,
                    success=True,
                    iterations=iteration + 1,
                    residual_norm=residual_norm,
                    info={'method': 'SOR', 'omega': omega, 'residuals': residuals}
                )
        return SolverResult(
            solution=x,
            success=False,
            iterations=max_iterations,
            residual_norm=residuals[-1] if residuals else np.inf,
            info={'method': 'SOR', 'omega': omega, 'residuals': residuals, 'converged': False}
        )
    @staticmethod
    def conjugate_gradient(A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None,
                          max_iterations: Optional[int] = None, tolerance: float = 1e-6,
                          preconditioner: Optional[np.ndarray] = None) -> SolverResult:
        """
        Solve Ax = b using Conjugate Gradient method (for symmetric positive definite A).
        Parameters:
            A: Symmetric positive definite matrix
            b: Right-hand side vector
            x0: Initial guess
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            preconditioner: Preconditioner matrix
        Returns:
            SolverResult with solution and metadata
        """
        A = MatrixOperations.validate_matrix(A, "A")
        b = MatrixOperations.validate_vector(b, "b")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square for CG")
        if A.shape[0] != len(b):
            raise ValueError(f"Matrix and vector dimensions incompatible: {A.shape[0]} != {len(b)}")
        # Check if matrix is symmetric
        if not np.allclose(A, A.T):
            warnings.warn("Matrix is not symmetric, CG may not converge")
        n = A.shape[0]
        if x0 is None:
            x = np.zeros(n)
        else:
            x = MatrixOperations.validate_vector(x0, "x0").copy()
        if max_iterations is None:
            max_iterations = n
        # Use scipy's CG implementation for robustness
        try:
            if preconditioner is not None:
                M = spla.LinearOperator((n, n), matvec=lambda v: la.solve(preconditioner, v))
            else:
                M = None
            x, info = spla.cg(A, b, x0=x, maxiter=max_iterations, rtol=tolerance, M=M)
            residual = A @ x - b
            residual_norm = np.linalg.norm(residual)
            return SolverResult(
                solution=x,
                success=(info == 0),
                iterations=max_iterations if info != 0 else -1,  # scipy doesn't return iteration count
                residual_norm=residual_norm,
                info={'method': 'CG', 'scipy_info': info}
            )
        except Exception as e:
            warnings.warn(f"CG solve failed: {e}")
            return SolverResult(
                solution=np.full(n, np.nan),
                success=False,
                iterations=0,
                residual_norm=np.inf,
                info={'method': 'CG', 'error': str(e)}
            )
    @staticmethod
    def gmres(A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None,
              max_iterations: Optional[int] = None, tolerance: float = 1e-6,
              restart: Optional[int] = None) -> SolverResult:
        """
        Solve Ax = b using GMRES method.
        Parameters:
            A: Coefficient matrix
            b: Right-hand side vector
            x0: Initial guess
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            restart: GMRES restart parameter
        Returns:
            SolverResult with solution and metadata
        """
        A = MatrixOperations.validate_matrix(A, "A")
        b = MatrixOperations.validate_vector(b, "b")
        if A.shape[0] != len(b):
            raise ValueError(f"Matrix and vector dimensions incompatible: {A.shape[0]} != {len(b)}")
        n = A.shape[0]
        if x0 is None:
            x0 = np.zeros(n)
        else:
            x0 = MatrixOperations.validate_vector(x0, "x0")
        if max_iterations is None:
            max_iterations = min(n, 1000)
        try:
            x, info = spla.gmres(A, b, x0=x0, maxiter=max_iterations, rtol=tolerance, restart=restart)
            residual = A @ x - b
            residual_norm = np.linalg.norm(residual)
            return SolverResult(
                solution=x,
                success=(info == 0),
                iterations=max_iterations if info != 0 else -1,
                residual_norm=residual_norm,
                info={'method': 'GMRES', 'scipy_info': info, 'restart': restart}
            )
        except Exception as e:
            warnings.warn(f"GMRES solve failed: {e}")
            return SolverResult(
                solution=np.full(A.shape[1], np.nan),
                success=False,
                iterations=0,
                residual_norm=np.inf,
                info={'method': 'GMRES', 'error': str(e)}
            )
class LinearSystemUtils:
    """
    Utility functions for linear systems.
    Features:
    - System conditioning analysis
    - Residual analysis
    - Solver selection heuristics
    """
    @staticmethod
    def analyze_system(A: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        """
        Analyze linear system properties to guide solver selection.
        Parameters:
            A: Coefficient matrix
            b: Right-hand side vector
        Returns:
            Dictionary with system properties
        """
        A = MatrixOperations.validate_matrix(A, "A")
        b = MatrixOperations.validate_vector(b, "b")
        analysis = {
            'matrix_shape': A.shape,
            'vector_length': len(b),
            'is_square': A.shape[0] == A.shape[1],
            'is_overdetermined': A.shape[0] > A.shape[1],
            'is_underdetermined': A.shape[0] < A.shape[1]
        }
        if A.shape[0] == A.shape[1]:
            # Square matrix analysis
            analysis['determinant'] = np.linalg.det(A)
            analysis['is_singular'] = abs(analysis['determinant']) < 1e-12
            analysis['condition_number'] = np.linalg.cond(A)
            analysis['is_well_conditioned'] = analysis['condition_number'] < 1e12
            # Symmetry check
            analysis['is_symmetric'] = np.allclose(A, A.T)
            if analysis['is_symmetric']:
                eigenvals = np.linalg.eigvals(A)
                analysis['is_positive_definite'] = np.all(eigenvals > 1e-12)
                analysis['is_positive_semidefinite'] = np.all(eigenvals >= -1e-12)
            # Diagonal dominance
            diag = np.abs(np.diag(A))
            off_diag_sum = np.sum(np.abs(A), axis=1) - diag
            analysis['is_diagonally_dominant'] = np.all(diag >= off_diag_sum)
        return analysis
    @staticmethod
    def recommend_solver(A: np.ndarray, b: np.ndarray) -> str:
        """
        Recommend appropriate solver based on system properties.
        Parameters:
            A: Coefficient matrix
            b: Right-hand side vector
        Returns:
            Recommended solver name
        """
        analysis = LinearSystemUtils.analyze_system(A, b)
        if not analysis['is_square']:
            return 'qr_solve'  # Overdetermined/underdetermined
        if analysis['is_singular']:
            return 'svd_solve'  # Singular system
        if analysis.get('is_positive_definite', False):
            return 'cholesky_solve'  # Symmetric positive definite
        if not analysis['is_well_conditioned']:
            return 'svd_solve'  # Ill-conditioned
        if A.shape[0] > 1000 and analysis.get('is_diagonally_dominant', False):
            return 'gauss_seidel'  # Large, well-conditioned iterative
        return 'lu_solve'  # Default for general case
    @staticmethod
    def solve_auto(A: np.ndarray, b: np.ndarray, **kwargs) -> SolverResult:
        """
        Automatically select and apply appropriate solver.
        Parameters:
            A: Coefficient matrix
            b: Right-hand side vector
            **kwargs: Additional parameters passed to solver
        Returns:
            SolverResult from chosen solver
        """
        solver_name = LinearSystemUtils.recommend_solver(A, b)
        if solver_name == 'lu_solve':
            return DirectSolvers.lu_solve(A, b, **kwargs)
        elif solver_name == 'cholesky_solve':
            return DirectSolvers.cholesky_solve(A, b, **kwargs)
        elif solver_name == 'qr_solve':
            return DirectSolvers.qr_solve(A, b, **kwargs)
        elif solver_name == 'svd_solve':
            return DirectSolvers.svd_solve(A, b, **kwargs)
        elif solver_name == 'gauss_seidel':
            return IterativeSolvers.gauss_seidel(A, b, **kwargs)
        else:
            return DirectSolvers.lu_solve(A, b, **kwargs)  # Fallback
def create_test_systems() -> dict:
    """Create test linear systems for validation."""
    systems = {}
    np.random.seed(42)
    # Well-conditioned system
    A = np.random.randn(5, 5)
    A = A + 0.1 * np.eye(5)  # Make well-conditioned
    b = np.random.randn(5)
    systems['well_conditioned'] = (A, b)
    # Symmetric positive definite
    A = np.random.randn(4, 4)
    A = A.T @ A + 0.1 * np.eye(4)
    b = np.random.randn(4)
    systems['symmetric_pd'] = (A, b)
    # Tridiagonal system
    n = 6
    diag = np.full(n, 2.0)
    upper = np.full(n-1, -1.0)
    lower = np.full(n-1, -1.0)
    b = np.ones(n)
    systems['tridiagonal'] = (diag, upper, lower, b)
    # Overdetermined system
    A = np.random.randn(8, 5)
    b = np.random.randn(8)
    systems['overdetermined'] = (A, b)
    # Ill-conditioned (Hilbert matrix)
    n = 5
    A = np.array([[1/(i+j+1) for j in range(n)] for i in range(n)])
    b = np.ones(n)
    systems['ill_conditioned'] = (A, b)
    return systems