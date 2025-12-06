"""
Comprehensive unit tests for Librex.QAP/utils.py core functions.

This test suite addresses critical coverage gaps identified by Agent 3.
Target: Increase utils.py coverage from 39% to 80%+

Priority tests:
- QAPProblem class methods
- QAPPipeline.solve() - CRITICAL
- Mathematical gradients
- Saddle point detection/escape
- Local search correctness
"""

import numpy as np
import pytest
from pathlib import Path
from Librex.QAP.utils import (
    QAPProblem,
    QAPPipeline,
    gradient_qap,
    entropy_gradient,
    constraint_forces,
    sinkhorn_projection,
    spectral_init,
    is_saddle,
    reverse_time_escape,
    hungarian_round,
    local_search_2opt,
    qap_cost,
    swap_delta,
    load_qaplib_instance,
    permutation_matrix,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def small_qap_problem():
    """Small symmetric QAP problem for testing."""
    n = 5
    np.random.seed(42)
    A = np.random.randint(1, 10, (n, n)).astype(float)
    B = np.random.randint(1, 10, (n, n)).astype(float)
    A = (A + A.T) / 2
    B = (B + B.T) / 2
    return QAPProblem(A, B, best_known=100.0, instance_name="test_small")


@pytest.fixture
def medium_qap_problem():
    """Medium QAP problem."""
    n = 12
    np.random.seed(123)
    A = np.random.randint(1, 20, (n, n)).astype(float)
    B = np.random.randint(1, 20, (n, n)).astype(float)
    A = (A + A.T) / 2
    B = (B + B.T) / 2
    return QAPProblem(A, B, best_known=500.0, instance_name="test_medium")


@pytest.fixture
def identity_permutation():
    """Identity permutation matrix."""
    n = 5
    return np.eye(n)


@pytest.fixture
def doubly_stochastic_matrix():
    """Valid doubly-stochastic matrix."""
    n = 4
    X = np.array([
        [0.3, 0.2, 0.25, 0.25],
        [0.25, 0.3, 0.2, 0.25],
        [0.2, 0.25, 0.3, 0.25],
        [0.25, 0.25, 0.25, 0.25],
    ])
    return X


# ============================================================================
# TEST CLASS: QAPProblem
# ============================================================================


class TestQAPProblem:
    """Test QAPProblem class methods."""

    def test_construction_basic(self, small_qap_problem):
        """Test basic construction."""
        assert small_qap_problem.n == 5
        assert small_qap_problem.instance_name == "test_small"
        assert small_qap_problem.best_known == 100.0

    def test_construction_validation_shape_mismatch(self):
        """Test construction fails on shape mismatch."""
        A = np.random.rand(5, 5)
        B = np.random.rand(6, 6)
        with pytest.raises(AssertionError):
            QAPProblem(A, B)

    def test_construction_validation_non_square(self):
        """Test construction fails on non-square matrices."""
        A = np.random.rand(5, 6)
        B = np.random.rand(5, 6)
        with pytest.raises(AssertionError):
            QAPProblem(A, B)

    def test_objective_identity(self, small_qap_problem, identity_permutation):
        """Test objective on identity permutation."""
        obj = small_qap_problem.objective(identity_permutation)
        # Should equal trace(A @ B)
        expected = np.trace(small_qap_problem.A @ small_qap_problem.B)
        assert np.isclose(obj, expected)

    def test_objective_random_permutation(self, small_qap_problem):
        """Test objective on random permutation."""
        perm = np.array([1, 2, 3, 4, 0])
        P = np.zeros((5, 5))
        P[np.arange(5), perm] = 1
        obj = small_qap_problem.objective(P)
        assert obj > 0
        assert np.isfinite(obj)

    def test_gap_with_best_known(self, small_qap_problem, identity_permutation):
        """Test gap calculation when best_known is set."""
        small_qap_problem.best_known = 100.0
        gap = small_qap_problem.gap(identity_permutation)
        assert np.isfinite(gap)

    def test_gap_without_best_known(self):
        """Test gap returns inf when best_known is None."""
        A = np.eye(3)
        B = np.eye(3)
        problem = QAPProblem(A, B, best_known=None)
        gap = problem.gap(np.eye(3))
        assert gap == float('inf')

    def test_ds_violation_perfect_matrix(self, small_qap_problem, doubly_stochastic_matrix):
        """Test ds_violation on perfect doubly-stochastic matrix."""
        X = doubly_stochastic_matrix
        # Pad to size 5
        n = small_qap_problem.n
        X_padded = np.eye(n) / n
        X_padded[:4, :4] = X
        violation = small_qap_problem.ds_violation(X_padded)
        # Will have violations due to padding, but original should be small
        problem_4 = QAPProblem(np.eye(4), np.eye(4))
        violation = problem_4.ds_violation(X)
        assert violation < 1e-10

    def test_ds_violation_violated_matrix(self, small_qap_problem):
        """Test ds_violation on matrix with violations."""
        X = np.random.rand(5, 5)
        violation = small_qap_problem.ds_violation(X)
        assert violation > 0.1  # Should have significant violations

    def test_is_permutation_valid(self, small_qap_problem, identity_permutation):
        """Test is_permutation on valid permutation."""
        assert small_qap_problem.is_permutation(identity_permutation)

    def test_is_permutation_invalid_non_binary(self, small_qap_problem):
        """Test is_permutation rejects non-binary matrix."""
        X = np.ones((5, 5)) / 5  # Doubly-stochastic but not permutation
        assert not small_qap_problem.is_permutation(X)

    def test_is_permutation_invalid_multiple_ones(self, small_qap_problem):
        """Test is_permutation rejects matrix with multiple 1s per row."""
        P = np.array([
            [1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ])
        assert not small_qap_problem.is_permutation(P)


# ============================================================================
# TEST CLASS: Mathematical Gradients
# ============================================================================


class TestMathematicalGradients:
    """Test gradient computations for correctness."""

    def test_gradient_qap_shape(self, small_qap_problem):
        """Test gradient_qap returns correct shape."""
        X = np.ones((5, 5)) / 5
        grad = gradient_qap(small_qap_problem.A, small_qap_problem.B, X)
        assert grad.shape == (5, 5)

    def test_gradient_qap_finite_difference(self):
        """Verify gradient_qap using finite differences."""
        n = 4
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        X = spectral_init(A, B)

        # Analytical gradient
        grad = gradient_qap(A, B, X)

        # Numerical gradient (sample a few points for speed)
        eps = 1e-7
        for i in [0, 1, 2]:
            for j in [0, 1, 2]:
                X_plus = X.copy()
                X_plus[i, j] += eps

                X_minus = X.copy()
                X_minus[i, j] -= eps

                f_plus = np.trace(A @ X_plus @ B @ X_plus.T)
                f_minus = np.trace(A @ X_minus @ B @ X_minus.T)

                grad_numerical = (f_plus - f_minus) / (2 * eps)

                assert np.isclose(grad[i, j], grad_numerical, rtol=1e-4, atol=1e-6), \
                    f"Gradient mismatch at ({i},{j}): {grad[i,j]} vs {grad_numerical}"

    def test_entropy_gradient_shape(self):
        """Test entropy_gradient returns correct shape."""
        X = np.random.rand(5, 5) + 0.1
        grad = entropy_gradient(X)
        assert grad.shape == (5, 5)

    def test_entropy_gradient_numerical_stability(self):
        """Test entropy_gradient handles near-zero values."""
        X = np.ones((5, 5)) * 1e-12
        grad = entropy_gradient(X, epsilon=1e-10)
        assert np.all(np.isfinite(grad))
        assert not np.any(np.isnan(grad))

    def test_constraint_forces_row_violation(self):
        """Test constraint_forces on row violation."""
        X = np.ones((3, 3)) / 3
        X[0, :] *= 1.5  # Violate row 0
        forces = constraint_forces(X, lambda_r=1.0, lambda_c=1.0)
        assert forces.shape == (3, 3)
        # Forces should be nonzero
        assert np.linalg.norm(forces) > 0

    def test_constraint_forces_both_violations(self):
        """Test constraint_forces with both row and column violations."""
        X = np.random.rand(4, 4) + 0.5
        forces = constraint_forces(X, lambda_r=2.0, lambda_c=2.0)
        assert forces.shape == (4, 4)
        assert np.all(np.isfinite(forces))


# ============================================================================
# TEST CLASS: Saddle Point Handling
# ============================================================================


class TestSaddlePointHandling:
    """Test saddle point detection and escape."""

    def test_is_saddle_small_gradient(self):
        """Test is_saddle detects small gradient."""
        grad = np.ones((5, 5)) * 1e-8
        assert is_saddle(grad, threshold=1e-5)

    def test_is_saddle_large_gradient(self):
        """Test is_saddle rejects large gradient."""
        grad = np.ones((5, 5))
        assert not is_saddle(grad, threshold=1e-5)

    def test_is_saddle_threshold_edge(self):
        """Test is_saddle at threshold boundary."""
        threshold = 1e-4
        grad = np.ones((3, 3)) * (threshold * 0.9 / np.sqrt(9))
        assert is_saddle(grad, threshold=threshold)

    def test_reverse_time_escape_basic(self, small_qap_problem):
        """Test reverse_time_escape basic operation."""
        X = spectral_init(small_qap_problem.A, small_qap_problem.B)
        grad = gradient_qap(small_qap_problem.A, small_qap_problem.B, X)

        X_escaped = reverse_time_escape(X, grad, dt=0.05, steps=5)

        assert X_escaped.shape == X.shape
        assert np.all(X_escaped >= 0)
        # Should be different from input
        assert not np.allclose(X_escaped, X)

    def test_reverse_time_escape_preserves_feasibility(self, small_qap_problem):
        """Test reverse_time_escape maintains doubly-stochastic property."""
        X = spectral_init(small_qap_problem.A, small_qap_problem.B)
        grad = gradient_qap(small_qap_problem.A, small_qap_problem.B, X)

        X_escaped = reverse_time_escape(X, grad, dt=0.1, steps=10)

        # Should be doubly-stochastic
        row_sums = X_escaped.sum(axis=1)
        col_sums = X_escaped.sum(axis=0)
        assert np.allclose(row_sums, 1.0, atol=1e-5)
        assert np.allclose(col_sums, 1.0, atol=1e-5)


# ============================================================================
# TEST CLASS: Spectral Initialization
# ============================================================================


class TestSpectralInitialization:
    """Test spectral initialization strategies."""

    def test_spectral_init_svd_strategy(self, small_qap_problem):
        """Test spectral_init with SVD strategy."""
        X = spectral_init(small_qap_problem.A, small_qap_problem.B, strategy='svd')
        assert X.shape == (5, 5)
        # Should be doubly-stochastic
        assert np.allclose(X.sum(axis=0), 1.0, atol=1e-5)
        assert np.allclose(X.sum(axis=1), 1.0, atol=1e-5)

    def test_spectral_init_eig_strategy(self, small_qap_problem):
        """Test spectral_init with eigenvalue strategy (PREVIOUSLY UNTESTED)."""
        X = spectral_init(small_qap_problem.A, small_qap_problem.B, strategy='eig')
        assert X.shape == (5, 5)
        # Should be doubly-stochastic
        assert np.allclose(X.sum(axis=0), 1.0, atol=1e-5)
        assert np.allclose(X.sum(axis=1), 1.0, atol=1e-5)

    def test_spectral_init_different_strategies_differ(self, small_qap_problem):
        """Test that different strategies produce different results."""
        X_svd = spectral_init(small_qap_problem.A, small_qap_problem.B, strategy='svd')
        X_eig = spectral_init(small_qap_problem.A, small_qap_problem.B, strategy='eig')
        # Should not be identical
        assert not np.allclose(X_svd, X_eig, atol=1e-3)


# ============================================================================
# TEST CLASS: Local Search
# ============================================================================


class TestLocalSearch:
    """Test 2-opt local search."""

    def test_local_search_2opt_basic(self, small_qap_problem, identity_permutation):
        """Test local_search_2opt basic operation."""
        P_refined = local_search_2opt(
            identity_permutation,
            small_qap_problem.A,
            small_qap_problem.B,
            max_iter=10
        )
        assert P_refined.shape == (5, 5)
        assert small_qap_problem.is_permutation(P_refined)

    def test_local_search_2opt_improves_or_maintains(self, small_qap_problem):
        """Test local_search_2opt improves or maintains objective."""
        # Start with random permutation
        perm = np.array([2, 1, 4, 3, 0])
        P = np.zeros((5, 5))
        P[np.arange(5), perm] = 1

        obj_before = small_qap_problem.objective(P)

        P_refined = local_search_2opt(P, small_qap_problem.A, small_qap_problem.B, max_iter=50)

        obj_after = small_qap_problem.objective(P_refined)

        # Should not increase objective
        assert obj_after <= obj_before + 1e-10

    def test_local_search_2opt_timeout(self, medium_qap_problem):
        """Test local_search_2opt respects timeout."""
        P = np.eye(12)
        import time
        start = time.time()
        P_refined = local_search_2opt(P, medium_qap_problem.A, medium_qap_problem.B,
                                      max_iter=10000, max_time=0.5)
        elapsed = time.time() - start
        # Should terminate around timeout (with some tolerance)
        assert elapsed < 1.5


# ============================================================================
# TEST CLASS: QAPPipeline (CRITICAL)
# ============================================================================


class TestQAPPipeline:
    """Test full QAPPipeline.solve() - PREVIOUSLY COMPLETELY UNTESTED."""

    def test_pipeline_solve_basic(self, small_qap_problem):
        """Test QAPPipeline.solve() basic execution."""
        pipeline = QAPPipeline(
            use_fft=False,  # Disable for small problem
            use_reverse_time=True,
            use_momentum=True,
            use_2opt=True
        )

        P, history = pipeline.solve(small_qap_problem, max_time=2.0)

        # Check output structure
        assert P.shape == (5, 5)
        assert small_qap_problem.is_permutation(P)

        # Check history
        assert 'times' in history
        assert 'objectives' in history
        assert 'gaps' in history
        assert 'ds_violations' in history
        assert 'grad_norms' in history
        assert 'saddle_escapes' in history

        # Check history has data
        assert len(history['times']) > 0
        assert len(history['objectives']) > 0

    def test_pipeline_solve_convergence(self, small_qap_problem):
        """Test QAPPipeline.solve() produces decreasing objective."""
        pipeline = QAPPipeline(use_fft=False, use_reverse_time=False, use_momentum=False)

        P, history = pipeline.solve(small_qap_problem, max_time=3.0)

        objectives = history['objectives']
        # Objective should decrease or stay same
        for i in range(1, len(objectives)):
            assert objectives[i] <= objectives[i-1] + 1e-6  # Allow tiny numerical error

    def test_pipeline_solve_timeout(self, medium_qap_problem):
        """Test QAPPipeline.solve() respects time limit."""
        pipeline = QAPPipeline()
        import time

        start = time.time()
        P, history = pipeline.solve(medium_qap_problem, max_time=1.0)
        elapsed = time.time() - start

        # Should terminate around timeout
        assert elapsed < 2.0
        assert history['times'][-1] <= 2.0


# ============================================================================
# TEST CLASS: Utility Functions
# ============================================================================


class TestSwapDelta:
    """Test swap_delta calculation."""

    def test_swap_delta_same_positions(self):
        """Test swap_delta returns 0 for i=j."""
        A = np.random.rand(4, 4)
        B = np.random.rand(4, 4)
        perm = np.array([0, 1, 2, 3])
        delta = swap_delta(A, B, perm, 1, 1)
        assert delta == 0.0

    def test_swap_delta_calculation(self):
        """Test swap_delta calculation is correct."""
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5, 6], [7, 8]], dtype=float)
        perm = np.array([0, 1])

        delta = swap_delta(A, B, perm, 0, 1)

        # Verify by computing full costs
        cost_before = qap_cost(A, B, perm)
        perm_swap = np.array([1, 0])
        cost_after = qap_cost(A, B, perm_swap)

        assert np.isclose(delta, cost_after - cost_before, atol=1e-10)


class TestPermutationMatrix:
    """Test permutation_matrix helper."""

    def test_permutation_matrix_structure(self):
        """Permutation matrices must be binary with one 1 per row/column."""
        perm = np.array([2, 0, 1])
        P = permutation_matrix(perm)
        assert P.shape == (3, 3)
        assert np.all((P == 0.0) | (P == 1.0))
        assert np.allclose(P.sum(axis=0), 1.0)
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_permutation_matrix_trace_equivalence(self):
        """qap_cost must match trace(A @ P @ B @ P.T) for symmetric inputs."""
        A = np.array([[0, 2, 1], [2, 0, 3], [1, 3, 0]], dtype=float)
        B = np.array([[0, 5, 4], [5, 0, 6], [4, 6, 0]], dtype=float)
        perm = np.array([1, 2, 0])
        P = permutation_matrix(perm)
        trace_cost = np.trace(A @ P @ B @ P.T)
        direct_cost = qap_cost(A, B, perm)
        assert np.isclose(trace_cost, direct_cost, atol=1e-10)


class TestLoadQAPLib:
    """Test load_qaplib_instance helper."""

    def test_load_qaplib_instance_parses_file(self, tmp_path):
        """Parser should read flattened QAPLIB format into matrices."""
        n = 2
        flow = np.array([[0, 1], [2, 0]], dtype=float)
        distance = np.array([[0, 3], [4, 0]], dtype=float)
        tokens = [str(n)] + [str(int(x)) for x in flow.flatten()] + [
            str(int(x)) for x in distance.flatten()
        ]
        data_dir = tmp_path
        (data_dir / "tiny.dat").write_text("\n".join([" ".join(tokens[:1]), " ".join(tokens[1:])]))
        loaded_flow, loaded_distance = load_qaplib_instance("tiny", data_dir)
        assert np.array_equal(loaded_flow, flow)
        assert np.array_equal(loaded_distance, distance)

    def test_load_qaplib_instance_invalid_count(self, tmp_path):
        """Parser should raise when entry count does not match 2 * n^2."""
        data_dir = tmp_path
        (data_dir / "bad.dat").write_text("2\n1 2 3")  # insufficient data
        with pytest.raises(ValueError, match="expected 8 entries"):
            load_qaplib_instance("bad", data_dir)


# ============================================================================
# TEST CLASS: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases for robustness."""

    def test_qap_cost_n1(self):
        """Test QAP on trivial n=1 problem."""
        A = np.array([[5.0]])
        B = np.array([[3.0]])
        perm = np.array([0])
        cost = qap_cost(A, B, perm)
        assert cost == 15.0

    def test_qap_cost_n2(self):
        """Test QAP on n=2 problem."""
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5, 6], [7, 8]], dtype=float)

        perm_id = np.array([0, 1])
        cost_id = qap_cost(A, B, perm_id)

        perm_swap = np.array([1, 0])
        cost_swap = qap_cost(A, B, perm_swap)

        # Should produce different costs
        assert cost_id != cost_swap

    def test_sinkhorn_already_feasible(self):
        """Test Sinkhorn on already doubly-stochastic matrix."""
        X = np.eye(5) / 5
        for i in range(5):
            X[i, :] = 1.0 / 5

        X_proj = sinkhorn_projection(X, max_iter=20, tol=1e-9)

        # Should converge quickly
        assert np.allclose(X_proj, X, atol=1e-6)

    def test_sinkhorn_projection_zero_row(self):
        """Sinkhorn projection should handle zero rows without NaNs."""
        X = np.array([
            [0.0, 0.0, 0.0],
            [0.2, 0.3, 0.5],
            [0.4, 0.4, 0.2],
        ])
        X_proj = sinkhorn_projection(X, max_iter=10, tol=1e-6)
        assert np.all(np.isfinite(X_proj))
        # Result should still approximate a DS matrix
        assert np.allclose(X_proj.sum(axis=1), 1.0, atol=1e-6)
        assert np.allclose(X_proj.sum(axis=0), 1.0, atol=1e-6)

    def test_hungarian_identity_input(self):
        """Test Hungarian on identity-like input."""
        X = np.eye(5) + np.random.rand(5, 5) * 0.01
        X = X / X.sum(axis=1, keepdims=True)

        P = hungarian_round(X)

        # Should return valid permutation close to identity
        assert np.allclose(P.sum(axis=0), 1.0)
        assert np.allclose(P.sum(axis=1), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
