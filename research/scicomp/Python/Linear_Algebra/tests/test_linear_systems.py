"""
Tests for Linear Systems Solvers
Comprehensive test suite for linear system solvers including direct methods,
iterative methods, and system analysis utilities.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import scipy.linalg as la
import warnings
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.linear_systems import (DirectSolvers, IterativeSolvers, LinearSystemUtils,
                                SolverResult, create_test_systems)
from core.matrix_operations import MatrixOperations, SpecialMatrices
class TestSolverResult:
    """Test SolverResult dataclass."""
    def test_solver_result_creation(self):
        """Test SolverResult creation and attributes."""
        solution = np.array([1, 2, 3])
        result = SolverResult(
            solution=solution,
            success=True,
            iterations=10,
            residual_norm=1e-8,
            info={'method': 'test'}
        )
        assert_array_almost_equal(result.solution, solution)
        assert result.success is True
        assert result.iterations == 10
        assert result.residual_norm == 1e-8
        assert result.info['method'] == 'test'
class TestDirectSolvers:
    """Test direct solution methods."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Well-conditioned square system
        self.A = np.array([[2, 1], [1, 3]])
        self.x_true = np.array([1, 2])
        self.b = self.A @ self.x_true
        # Symmetric positive definite
        A_base = np.random.randn(4, 4)
        self.A_spd = A_base.T @ A_base + 0.1 * np.eye(4)
        self.x_true_spd = np.random.randn(4)
        self.b_spd = self.A_spd @ self.x_true_spd
        # Overdetermined system
        self.A_over = np.random.randn(6, 4)
        self.x_true_over = np.random.randn(4)
        self.b_over = self.A_over @ self.x_true_over
        # Tridiagonal system
        n = 5
        self.diag = 2 * np.ones(n)
        self.upper = -np.ones(n-1)
        self.lower = -np.ones(n-1)
        self.x_true_tri = np.random.randn(n)
        A_tri = np.diag(self.diag) + np.diag(self.upper, 1) + np.diag(self.lower, -1)
        self.b_tri = A_tri @ self.x_true_tri
    def test_lu_solve(self):
        """Test LU decomposition solver."""
        result = DirectSolvers.lu_solve(self.A, self.b)
        assert result.success
        assert result.iterations == 1
        assert_array_almost_equal(result.solution, self.x_true, decimal=10)
        assert result.residual_norm < 1e-12
        assert result.info['method'] == 'LU'
        assert 'condition_number' in result.info
        # Test with finite checking
        result_finite = DirectSolvers.lu_solve(self.A, self.b, check_finite=False)
        assert result_finite.success
        assert_array_almost_equal(result_finite.solution, self.x_true, decimal=10)
        # Incompatible dimensions
        with pytest.raises(ValueError):
            DirectSolvers.lu_solve(self.A, np.array([1, 2, 3]))
        # Non-square matrix
        with pytest.raises(ValueError):
            DirectSolvers.lu_solve(self.A_over, self.b_over)
        # Singular matrix
        A_singular = np.array([[1, 2], [2, 4]])
        b_singular = np.array([1, 2])
        result_singular = DirectSolvers.lu_solve(A_singular, b_singular)
        assert not result_singular.success
        assert np.all(np.isnan(result_singular.solution))
    def test_cholesky_solve(self):
        """Test Cholesky decomposition solver."""
        result = DirectSolvers.cholesky_solve(self.A_spd, self.b_spd)
        assert result.success
        assert result.iterations == 1
        assert_array_almost_equal(result.solution, self.x_true_spd, decimal=8)
        assert result.residual_norm < 1e-10
        assert result.info['method'] == 'Cholesky'
        # Test lower triangular option
        result_lower = DirectSolvers.cholesky_solve(self.A_spd, self.b_spd, lower=True)
        assert result_lower.success
        assert_array_almost_equal(result_lower.solution, self.x_true_spd, decimal=8)
        # Non-symmetric matrix (should warn)
        A_nonsym = np.array([[2, 1], [0, 3]])
        b_nonsym = np.array([1, 2])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_nonsym = DirectSolvers.cholesky_solve(A_nonsym, b_nonsym)
            assert len(w) == 1
            assert "not symmetric" in str(w[0].message)
        # Non-positive definite matrix
        A_not_pd = np.array([[1, 2], [2, 1]])
        b_not_pd = np.array([1, 2])
        result_not_pd = DirectSolvers.cholesky_solve(A_not_pd, b_not_pd)
        assert not result_not_pd.success
    def test_qr_solve(self):
        """Test QR decomposition solver."""
        # Square system
        result = DirectSolvers.qr_solve(self.A, self.b)
        assert result.success
        assert_array_almost_equal(result.solution, self.x_true, decimal=10)
        # Overdetermined system
        result_over = DirectSolvers.qr_solve(self.A_over, self.b_over)
        assert result_over.success
        assert_array_almost_equal(result_over.solution, self.x_true_over, decimal=8)
        assert result_over.info['overdetermined'] is True
        # Economic mode
        result_econ = DirectSolvers.qr_solve(self.A_over, self.b_over, mode='economic')
        assert result_econ.success
        assert_array_almost_equal(result_econ.solution, self.x_true_over, decimal=8)
        # Rank deficient system
        A_rank_def = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1]])
        b_rank_def = np.array([1, 2, 1])
        result_rank_def = DirectSolvers.qr_solve(A_rank_def, b_rank_def)
        # Should still work but may have large residual for inconsistent system
        assert result_rank_def.success
    def test_svd_solve(self):
        """Test SVD solver."""
        # Well-conditioned system
        result = DirectSolvers.svd_solve(self.A, self.b)
        assert result.success
        assert_array_almost_equal(result.solution, self.x_true, decimal=8)
        assert result.info['method'] == 'SVD'
        assert 'rank' in result.info
        assert 'singular_values' in result.info
        # Overdetermined system
        result_over = DirectSolvers.svd_solve(self.A_over, self.b_over)
        assert result_over.success
        assert_array_almost_equal(result_over.solution, self.x_true_over, decimal=6)
        # Rank deficient system
        A_rank_def = np.array([[1, 2], [2, 4]])
        x_consistent = np.array([1, 0.5])  # Choose solution in range
        b_consistent = A_rank_def @ x_consistent
        result_rank_def = DirectSolvers.svd_solve(A_rank_def, b_consistent)
        assert result_rank_def.success
        assert result_rank_def.info['rank'] == 1
        # Verify solution is in range (Ax = b should hold)
        residual = A_rank_def @ result_rank_def.solution - b_consistent
        assert np.linalg.norm(residual) < 1e-10
        # Custom tolerance
        result_custom = DirectSolvers.svd_solve(self.A, self.b, rcond=1e-12)
        assert result_custom.success
    def test_tridiagonal_solve(self):
        """Test tridiagonal system solver."""
        result = DirectSolvers.tridiagonal_solve(self.diag, self.upper, self.lower, self.b_tri)
        assert result.success
        assert result.iterations == 1
        assert_array_almost_equal(result.solution, self.x_true_tri, decimal=10)
        assert result.residual_norm < 1e-12
        assert result.info['method'] == 'Thomas (Tridiagonal)'
        # Dimension mismatches
        with pytest.raises(ValueError):
            DirectSolvers.tridiagonal_solve(self.diag, self.upper[:-1], self.lower, self.b_tri)
        with pytest.raises(ValueError):
            DirectSolvers.tridiagonal_solve(self.diag, self.upper, self.lower[:-1], self.b_tri)
        with pytest.raises(ValueError):
            DirectSolvers.tridiagonal_solve(self.diag, self.upper, self.lower, self.b_tri[:-1])
    def test_direct_solver_edge_cases(self):
        """Test edge cases for direct solvers."""
        # Very small system
        A_small = np.array([[2]])
        b_small = np.array([4])
        result_small = DirectSolvers.lu_solve(A_small, b_small)
        assert result_small.success
        assert_allclose(result_small.solution, [2])
        # Identity system
        A_identity = np.eye(3)
        b_identity = np.array([1, 2, 3])
        result_identity = DirectSolvers.lu_solve(A_identity, b_identity)
        assert result_identity.success
        assert_array_almost_equal(result_identity.solution, b_identity)
        # Complex system
        A_complex = np.array([[1+1j, 2], [0, 1-1j]])
        x_complex = np.array([1, 1+1j])
        b_complex = A_complex @ x_complex
        result_complex = DirectSolvers.lu_solve(A_complex, b_complex)
        assert result_complex.success
        assert_array_almost_equal(result_complex.solution, x_complex)
class TestIterativeSolvers:
    """Test iterative solution methods."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Diagonally dominant system (ensures convergence)
        n = 5
        A = np.random.randn(n, n)
        A = A + A.T  # Make symmetric
        A = A + (np.abs(np.sum(A, axis=1)) + 1) * np.eye(n)  # Make diagonally dominant
        self.A_dd = A
        self.x_true_dd = np.random.randn(n)
        self.b_dd = self.A_dd @ self.x_true_dd
        # Symmetric positive definite (good for CG)
        A_base = np.random.randn(n, n)
        self.A_spd = A_base.T @ A_base + 0.1 * np.eye(n)
        self.x_true_spd = np.random.randn(n)
        self.b_spd = self.A_spd @ self.x_true_spd
        # Ill-conditioned system
        self.A_ill = self.create_hilbert_matrix(4)
        self.x_true_ill = np.ones(4)
        self.b_ill = self.A_ill @ self.x_true_ill
    def create_hilbert_matrix(self, n):
        """Create Hilbert matrix for testing."""
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H[i, j] = 1.0 / (i + j + 1)
        return H
    def test_jacobi(self):
        """Test Jacobi iteration."""
        result = IterativeSolvers.jacobi(self.A_dd, self.b_dd, tolerance=1e-8, max_iterations=1000)
        assert result.success
        assert result.iterations < 100  # Should converge reasonably fast
        assert_array_almost_equal(result.solution, self.x_true_dd, decimal=6)
        assert result.residual_norm < 1e-8
        assert result.info['method'] == 'Jacobi'
        assert 'residuals' in result.info
        # Custom initial guess
        x0 = np.ones(len(self.x_true_dd))
        result_x0 = IterativeSolvers.jacobi(self.A_dd, self.b_dd, x0=x0, tolerance=1e-6)
        assert result_x0.success
        # Non-convergent case (non-diagonally dominant)
        A_bad = np.array([[1, 2], [3, 1]])  # Not diagonally dominant
        b_bad = np.array([1, 1])
        result_bad = IterativeSolvers.jacobi(A_bad, b_bad, max_iterations=10, tolerance=1e-6)
        # May or may not converge, but should handle gracefully
        # Near-zero diagonal warning
        A_zero_diag = np.array([[1e-15, 1], [1, 2]])
        b_zero_diag = np.array([1, 1])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            IterativeSolvers.jacobi(A_zero_diag, b_zero_diag, max_iterations=5)
            assert len(w) >= 1
            assert any("near-zero diagonal" in str(warning.message) for warning in w)
        # Non-square matrix
        with pytest.raises(ValueError):
            A_rect = np.random.randn(3, 4)
            b_rect = np.random.randn(3)
            IterativeSolvers.jacobi(A_rect, b_rect)
        # Dimension mismatch
        with pytest.raises(ValueError):
            IterativeSolvers.jacobi(self.A_dd, self.b_dd[:-1])
    def test_gauss_seidel(self):
        """Test Gauss-Seidel iteration."""
        result = IterativeSolvers.gauss_seidel(self.A_dd, self.b_dd, tolerance=1e-8, max_iterations=1000)
        assert result.success
        assert result.iterations < 100  # Should converge faster than Jacobi
        assert_array_almost_equal(result.solution, self.x_true_dd, decimal=6)
        assert result.residual_norm < 1e-8
        assert result.info['method'] == 'Gauss-Seidel'
        # Compare with Jacobi (GS should be faster for diagonally dominant matrices)
        result_jacobi = IterativeSolvers.jacobi(self.A_dd, self.b_dd, tolerance=1e-6)
        if result_jacobi.success:
            assert result.iterations <= result_jacobi.iterations
        # Near-zero diagonal handling
        A_zero_diag = self.A_dd.copy()
        A_zero_diag[0, 0] = 1e-15
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_zero = IterativeSolvers.gauss_seidel(A_zero_diag, self.b_dd, max_iterations=5)
            assert len(w) >= 1
    def test_sor(self):
        """Test Successive Over-Relaxation."""
        # Optimal omega for SPD matrix
        eigenvals = np.linalg.eigvals(self.A_spd)
        rho_jacobi = (np.max(eigenvals) - np.min(eigenvals)) / (np.max(eigenvals) + np.min(eigenvals))
        omega_opt = 2 / (1 + np.sqrt(1 - rho_jacobi**2))
        result = IterativeSolvers.sor(self.A_spd, self.b_spd, omega=omega_opt, tolerance=1e-8)
        assert result.success
        assert_array_almost_equal(result.solution, self.x_true_spd, decimal=6)
        assert result.info['method'] == 'SOR'
        assert result.info['omega'] == omega_opt
        # Test different omega values
        result_under = IterativeSolvers.sor(self.A_dd, self.b_dd, omega=0.5, tolerance=1e-6, max_iterations=200)
        result_over = IterativeSolvers.sor(self.A_dd, self.b_dd, omega=1.5, tolerance=1e-6, max_iterations=200)
        # Both should work, but may have different convergence rates
        if result_under.success and result_over.success:
            # Under-relaxation should be more stable but slower
            pass
        # Warning for bad omega values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            IterativeSolvers.sor(self.A_dd, self.b_dd, omega=2.5, max_iterations=5)
            assert len(w) >= 1
            assert "omega should be in range" in str(w[0].message)
    def test_conjugate_gradient(self):
        """Test Conjugate Gradient method."""
        result = IterativeSolvers.conjugate_gradient(self.A_spd, self.b_spd, tolerance=1e-10)
        assert result.success
        assert_array_almost_equal(result.solution, self.x_true_spd, decimal=8)
        assert result.info['method'] == 'CG'
        # Should converge in at most n iterations for SPD matrix
        n = len(self.x_true_spd)
        assert result.iterations <= n
        # Custom initial guess
        x0 = np.ones(len(self.x_true_spd))
        result_x0 = IterativeSolvers.conjugate_gradient(self.A_spd, self.b_spd, x0=x0, tolerance=1e-8)
        assert result_x0.success
        # Custom max iterations
        result_limited = IterativeSolvers.conjugate_gradient(self.A_spd, self.b_spd, max_iterations=5, tolerance=1e-12)
        # May not converge with limited iterations
        # Non-symmetric matrix warning
        A_nonsym = np.random.randn(4, 4) + 2 * np.eye(4)
        b_nonsym = np.random.randn(4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            IterativeSolvers.conjugate_gradient(A_nonsym, b_nonsym, max_iterations=5)
            assert len(w) >= 1
            assert "not symmetric" in str(w[0].message)
        # Preconditioner test
        M = np.diag(np.diag(self.A_spd))  # Jacobi preconditioner
        result_prec = IterativeSolvers.conjugate_gradient(self.A_spd, self.b_spd,
                                                         preconditioner=M, tolerance=1e-8)
        assert result_prec.success
    def test_gmres(self):
        """Test GMRES method."""
        # Works on general matrices
        A_general = np.random.randn(5, 5) + 2 * np.eye(5)
        x_true_general = np.random.randn(5)
        b_general = A_general @ x_true_general
        result = IterativeSolvers.gmres(A_general, b_general, tolerance=1e-8)
        assert result.success
        assert_array_almost_equal(result.solution, x_true_general, decimal=6)
        assert result.info['method'] == 'GMRES'
        # Custom parameters
        result_restart = IterativeSolvers.gmres(A_general, b_general, restart=10, tolerance=1e-6)
        assert result_restart.success
        assert result_restart.info['restart'] == 10
        # Initial guess
        x0 = np.ones(len(x_true_general))
        result_x0 = IterativeSolvers.gmres(A_general, b_general, x0=x0, tolerance=1e-6)
        assert result_x0.success
        # Limited iterations
        result_limited = IterativeSolvers.gmres(A_general, b_general, max_iterations=5, tolerance=1e-12)
        # May not converge with limited iterations
    def test_iterative_solver_convergence(self):
        """Test convergence properties of iterative solvers."""
        # Well-conditioned system
        methods = [
            ('jacobi', IterativeSolvers.jacobi),
            ('gauss_seidel', IterativeSolvers.gauss_seidel),
            ('conjugate_gradient', IterativeSolvers.conjugate_gradient)
        ]
        tolerance = 1e-6
        results = {}
        for method_name, method in methods:
            if method_name == 'conjugate_gradient':
                result = method(self.A_spd, self.b_spd, tolerance=tolerance)
            else:
                result = method(self.A_dd, self.b_dd, tolerance=tolerance)
            if result.success:
                results[method_name] = result.iterations
        # CG should be most efficient for SPD systems
        if 'conjugate_gradient' in results:
            cg_iters = results['conjugate_gradient']
            assert cg_iters <= len(self.x_true_spd)  # Theoretical bound
        # Gauss-Seidel should converge faster than Jacobi for diagonally dominant matrices
        if 'jacobi' in results and 'gauss_seidel' in results:
            assert results['gauss_seidel'] <= results['jacobi']
class TestLinearSystemUtils:
    """Test linear system utility functions."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Square well-conditioned
        self.A_square = np.random.randn(4, 4) + 2 * np.eye(4)
        self.b_square = np.random.randn(4)
        # Overdetermined
        self.A_over = np.random.randn(6, 4)
        self.b_over = np.random.randn(6)
        # Underdetermined
        self.A_under = np.random.randn(3, 5)
        self.b_under = np.random.randn(3)
        # SPD matrix
        A_base = np.random.randn(4, 4)
        self.A_spd = A_base.T @ A_base + 0.1 * np.eye(4)
        self.b_spd = np.random.randn(4)
        # Singular matrix
        self.A_singular = np.array([[1, 2], [2, 4]])
        self.b_singular = np.array([1, 2])
        # Ill-conditioned matrix
        self.A_ill = self.create_hilbert_matrix(4)
        self.b_ill = np.random.randn(4)
    def create_hilbert_matrix(self, n):
        """Create Hilbert matrix."""
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H[i, j] = 1.0 / (i + j + 1)
        return H
    def test_analyze_system(self):
        """Test system analysis."""
        # Square system
        analysis = LinearSystemUtils.analyze_system(self.A_square, self.b_square)
        assert analysis['matrix_shape'] == self.A_square.shape
        assert analysis['vector_length'] == len(self.b_square)
        assert analysis['is_square'] is True
        assert analysis['is_overdetermined'] is False
        assert analysis['is_underdetermined'] is False
        assert 'determinant' in analysis
        assert 'condition_number' in analysis
        assert 'is_symmetric' in analysis
        assert 'is_diagonally_dominant' in analysis
        # Overdetermined system
        analysis_over = LinearSystemUtils.analyze_system(self.A_over, self.b_over)
        assert analysis_over['is_overdetermined'] is True
        assert analysis_over['is_square'] is False
        # Underdetermined system
        analysis_under = LinearSystemUtils.analyze_system(self.A_under, self.b_under)
        assert analysis_under['is_underdetermined'] is True
        assert analysis_under['is_square'] is False
        # SPD system
        analysis_spd = LinearSystemUtils.analyze_system(self.A_spd, self.b_spd)
        assert analysis_spd['is_symmetric'] is True
        assert analysis_spd['is_positive_definite'] is True
        # Singular system
        analysis_singular = LinearSystemUtils.analyze_system(self.A_singular, self.b_singular)
        assert analysis_singular['is_singular'] is True
        assert abs(analysis_singular['determinant']) < 1e-10
        # Ill-conditioned system
        analysis_ill = LinearSystemUtils.analyze_system(self.A_ill, self.b_ill)
        assert analysis_ill['is_well_conditioned'] is False
        assert analysis_ill['condition_number'] > 1e12
    def test_recommend_solver(self):
        """Test solver recommendation."""
        # SPD system should recommend Cholesky
        recommendation_spd = LinearSystemUtils.recommend_solver(self.A_spd, self.b_spd)
        assert recommendation_spd == 'cholesky_solve'
        # Overdetermined should recommend QR
        recommendation_over = LinearSystemUtils.recommend_solver(self.A_over, self.b_over)
        assert recommendation_over == 'qr_solve'
        # Singular should recommend SVD
        recommendation_singular = LinearSystemUtils.recommend_solver(self.A_singular, self.b_singular)
        assert recommendation_singular == 'svd_solve'
        # Ill-conditioned should recommend SVD
        recommendation_ill = LinearSystemUtils.recommend_solver(self.A_ill, self.b_ill)
        assert recommendation_ill == 'svd_solve'
        # Well-conditioned square should recommend LU
        recommendation_square = LinearSystemUtils.recommend_solver(self.A_square, self.b_square)
        assert recommendation_square == 'lu_solve'
        # Large diagonally dominant might recommend iterative
        n_large = 1001  # Larger than threshold
        A_large_dd = np.eye(n_large) * 10 + np.random.randn(n_large, n_large) * 0.1
        b_large = np.random.randn(n_large)
        recommendation_large = LinearSystemUtils.recommend_solver(A_large_dd, b_large)
        # Should recommend iterative method for large diagonally dominant
        assert recommendation_large in ['gauss_seidel', 'lu_solve']
    def test_solve_auto(self):
        """Test automatic solver selection and application."""
        # SPD system
        result_spd = LinearSystemUtils.solve_auto(self.A_spd, self.b_spd)
        assert result_spd.success
        assert result_spd.info['method'] == 'Cholesky'
        # Overdetermined system
        result_over = LinearSystemUtils.solve_auto(self.A_over, self.b_over)
        assert result_over.success
        assert result_over.info['method'] == 'QR'
        # Square system
        result_square = LinearSystemUtils.solve_auto(self.A_square, self.b_square)
        assert result_square.success
        assert result_square.info['method'] == 'LU'
        # Pass additional parameters
        result_with_params = LinearSystemUtils.solve_auto(self.A_square, self.b_square, check_finite=False)
        assert result_with_params.success
        # Verify solution quality
        residual = self.A_square @ result_square.solution - self.b_square
        assert np.linalg.norm(residual) < 1e-10
class TestCreateTestSystems:
    """Test test system creation function."""
    def test_create_test_systems(self):
        """Test creation of test linear systems."""
        systems = create_test_systems()
        expected_systems = [
            'well_conditioned',
            'symmetric_pd',
            'tridiagonal',
            'overdetermined',
            'ill_conditioned'
        ]
        for system_name in expected_systems:
            assert system_name in systems
        # Check well-conditioned system
        A, b = systems['well_conditioned']
        assert A.shape == (5, 5)
        assert len(b) == 5
        assert np.linalg.cond(A) < 1e6  # Should be well-conditioned
        # Check symmetric positive definite
        A_spd, b_spd = systems['symmetric_pd']
        assert A_spd.shape == (4, 4)
        assert np.allclose(A_spd, A_spd.T)  # Symmetric
        assert SpecialMatrices.is_positive_definite(A_spd)
        # Check tridiagonal (special format)
        diag, upper, lower, b_tri = systems['tridiagonal']
        assert len(diag) == 6
        assert len(upper) == 5
        assert len(lower) == 5
        assert len(b_tri) == 6
        # Check overdetermined
        A_over, b_over = systems['overdetermined']
        assert A_over.shape[0] > A_over.shape[1]  # More rows than columns
        # Check ill-conditioned
        A_ill, b_ill = systems['ill_conditioned']
        assert np.linalg.cond(A_ill) > 1e6  # Should be ill-conditioned
        # Test that systems can be solved
        for system_name, system_data in systems.items():
            if system_name == 'tridiagonal':
                diag, upper, lower, b = system_data
                result = DirectSolvers.tridiagonal_solve(diag, upper, lower, b)
            else:
                A, b = system_data
                result = LinearSystemUtils.solve_auto(A, b)
            # Most systems should be solvable (ill-conditioned might have issues)
            if system_name != 'ill_conditioned':
                assert result.success, f"Failed to solve {system_name} system"
class TestIntegrationTests:
    """Integration tests combining multiple components."""
    def test_solver_comparison(self):
        """Compare different solvers on the same problem."""
        np.random.seed(42)
        # Create SPD system
        A_base = np.random.randn(5, 5)
        A = A_base.T @ A_base + 0.1 * np.eye(5)
        x_true = np.random.randn(5)
        b = A @ x_true
        # Test multiple solvers
        solvers = [
            ('LU', lambda: DirectSolvers.lu_solve(A, b)),
            ('Cholesky', lambda: DirectSolvers.cholesky_solve(A, b)),
            ('QR', lambda: DirectSolvers.qr_solve(A, b)),
            ('SVD', lambda: DirectSolvers.svd_solve(A, b)),
            ('CG', lambda: IterativeSolvers.conjugate_gradient(A, b, tolerance=1e-10)),
            ('Auto', lambda: LinearSystemUtils.solve_auto(A, b))
        ]
        solutions = {}
        for solver_name, solver_func in solvers:
            result = solver_func()
            if result.success:
                solutions[solver_name] = result.solution
                # Check solution quality
                residual = A @ result.solution - b
                assert np.linalg.norm(residual) < 1e-8, f"{solver_name} residual too large"
        # All solutions should be similar
        reference_solution = solutions['LU']
        for solver_name, solution in solutions.items():
            error = np.linalg.norm(solution - reference_solution)
            assert error < 1e-6, f"{solver_name} solution differs from LU solution"
    def test_condition_number_effect(self):
        """Test how condition number affects solver performance."""
        np.random.seed(42)
        n = 10
        condition_numbers = [1e2, 1e6, 1e10, 1e14]
        for cond_target in condition_numbers:
            # Create matrix with specific condition number
            eigenvals = np.linspace(1, cond_target, n)
            Q, _ = np.linalg.qr(np.random.randn(n, n))
            A = Q @ np.diag(eigenvals) @ Q.T
            x_true = np.random.randn(n)
            b = A @ x_true
            # Test different solvers
            result_lu = DirectSolvers.lu_solve(A, b)
            result_svd = DirectSolvers.svd_solve(A, b)
            result_cg = IterativeSolvers.conjugate_gradient(A, b, tolerance=1e-12, max_iterations=n)
            if cond_target < 1e12:
                # Well-conditioned cases
                assert result_lu.success
                assert result_svd.success
                error_lu = np.linalg.norm(result_lu.solution - x_true)
                error_svd = np.linalg.norm(result_svd.solution - x_true)
                # Errors should be reasonable
                assert error_lu < 1e-6
                assert error_svd < 1e-6
                if result_cg.success:
                    error_cg = np.linalg.norm(result_cg.solution - x_true)
                    assert error_cg < 1e-6
            else:
                # Ill-conditioned cases - SVD should be more robust
                assert result_svd.success
                error_svd = np.linalg.norm(result_svd.solution - x_true)
                if result_lu.success:
                    error_lu = np.linalg.norm(result_lu.solution - x_true)
                    # SVD should be more accurate for ill-conditioned systems
                    assert error_svd <= error_lu * 10  # Allow some tolerance
    def test_large_sparse_system(self):
        """Test handling of larger sparse-like systems."""
        # Create tridiagonal system (sparse structure)
        n = 100
        diag = 2 * np.ones(n)
        upper = -np.ones(n-1)
        lower = -np.ones(n-1)
        # Create full matrix for comparison
        A_full = np.diag(diag) + np.diag(upper, 1) + np.diag(lower, -1)
        x_true = np.random.randn(n)
        b = A_full @ x_true
        # Compare solvers
        result_tridiag = DirectSolvers.tridiagonal_solve(diag, upper, lower, b)
        result_lu = DirectSolvers.lu_solve(A_full, b)
        result_cg = IterativeSolvers.conjugate_gradient(A_full, b, tolerance=1e-10)
        assert result_tridiag.success
        assert result_lu.success
        assert result_cg.success
        # Check solution accuracy
        error_tridiag = np.linalg.norm(result_tridiag.solution - x_true)
        error_lu = np.linalg.norm(result_lu.solution - x_true)
        error_cg = np.linalg.norm(result_cg.solution - x_true)
        assert error_tridiag < 1e-10
        assert error_lu < 1e-10
        assert error_cg < 1e-8
        # Specialized solver should be fastest (in practice)
        # This is hard to test reliably, but check that it works
        assert result_tridiag.info['method'] == 'Thomas (Tridiagonal)'
if __name__ == "__main__":
    pytest.main([__file__])