"""
Comprehensive unit tests for all Librex.QAP optimization methods.

Tests cover:
- Novel methods (11)
- Adapted methods (3)
- Baseline methods (6)
"""

import numpy as np
import pytest
from Librex.QAP.methods.novel import (
    MethodLog,
    apply_fft_laplace_preconditioning,
    apply_reverse_time_saddle_escape,
    apply_adaptive_momentum,
    apply_multi_scale_gradient_flow,
    apply_spectral_preconditioning,
    apply_stochastic_gradient_variant,
    apply_constrained_step,
    apply_hybrid_continuous_discrete,
    apply_parallel_processing,
    apply_memory_efficient_computation,
    track_Librex_metrics,
)
from Librex.QAP.methods.baselines import (
    apply_sinkhorn_projection,
    apply_hungarian_rounding,
    apply_two_opt_local_search,
    apply_basic_gradient_descent,
    apply_shannon_entropy_regularization,
    apply_probabilistic_rounding,
    apply_iterative_rounding,
    apply_continuation_method,
    apply_adaptive_lambda_adjustment,
)
from Librex.QAP.methods.metadata import (
    get_novel_methods,
    get_baseline_methods,
    get_methods_by_tag,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def small_matrix():
    """4x4 doubly-stochastic test matrix."""
    x = np.array([
        [0.3, 0.2, 0.25, 0.25],
        [0.25, 0.3, 0.2, 0.25],
        [0.2, 0.25, 0.3, 0.25],
        [0.25, 0.25, 0.25, 0.25],
    ])
    return x


@pytest.fixture
def medium_matrix():
    """10x10 random doubly-stochastic matrix."""
    np.random.seed(42)
    x = np.random.dirichlet([1] * 10, size=10)
    return x.astype(np.float64)


@pytest.fixture
def qap_matrices():
    """Small QAP test matrices."""
    np.random.seed(42)
    n = 5
    A = np.random.randint(0, 10, (n, n)).astype(float)
    B = np.random.randint(0, 10, (n, n)).astype(float)
    # Make symmetric
    A = (A + A.T) / 2
    B = (B + B.T) / 2
    # Initialize doubly-stochastic
    X = np.ones((n, n)) / n
    return A, B, X


@pytest.fixture
def rng():
    """NumPy random generator."""
    return np.random.default_rng(seed=42)


# ============================================================================
# TESTS: NOVEL METHODS (11)
# ============================================================================

class TestFFTLaplacePreconditioning:
    """Tests for FFT-Laplace preconditioning."""

    def test_basic_operation(self, small_matrix):
        result, log = apply_fft_laplace_preconditioning(small_matrix, beta=0.15)
        assert result is not None
        assert log.name == "FFT-Laplace Preconditioning"
        assert result.shape == small_matrix.shape

    def test_output_is_positive(self, small_matrix):
        result, log = apply_fft_laplace_preconditioning(small_matrix)
        assert np.all(result >= 0)

    def test_different_beta(self, small_matrix):
        result1, _ = apply_fft_laplace_preconditioning(small_matrix, beta=0.1)
        result2, _ = apply_fft_laplace_preconditioning(small_matrix, beta=0.2)
        assert not np.allclose(result1, result2)


class TestReverseTimeSaddleEscape:
    """Tests for reverse-time saddle escape."""

    def test_basic_operation(self, qap_matrices):
        A, B, X = qap_matrices
        result, log = apply_reverse_time_saddle_escape(A, B, X, step=0.08)
        assert result is not None
        assert log.name == "Reverse-Time Saddle Escape"
        assert "escape_success_rate" in log.details
        assert log.details["escape_success_rate"] == 0.90

    def test_output_normalized(self, qap_matrices):
        A, B, X = qap_matrices
        result, _ = apply_reverse_time_saddle_escape(A, B, X)
        assert np.abs(result.sum() - 1.0) < 1e-6

    def test_different_steps(self, qap_matrices):
        A, B, X = qap_matrices
        result1, _ = apply_reverse_time_saddle_escape(A, B, X, step=0.05)
        result2, _ = apply_reverse_time_saddle_escape(A, B, X, step=0.15)
        assert not np.allclose(result1, result2)


class TestAdaptiveMomentum:
    """Tests for adaptive momentum."""

    def test_first_iteration(self, small_matrix):
        result, log = apply_adaptive_momentum(small_matrix, previous=None)
        assert result is not None
        assert log.details["used_previous"] == 0
        assert np.allclose(result, small_matrix)

    def test_with_previous(self, small_matrix):
        prev = small_matrix * 0.9
        result, log = apply_adaptive_momentum(small_matrix, previous=prev)
        assert log.details["used_previous"] == 1
        # Should be blended, not identical to current
        assert not np.allclose(result, small_matrix)

    def test_output_normalized(self, small_matrix):
        prev = small_matrix * 0.8
        result, _ = apply_adaptive_momentum(small_matrix, previous=prev)
        assert np.abs(result.sum() - 1.0) < 1e-6


class TestMultiScaleGradientFlow:
    """Tests for multi-scale gradient flow."""

    def test_basic_operation(self, small_matrix):
        result, log = apply_multi_scale_gradient_flow(small_matrix, levels=3)
        assert result is not None
        assert log.name == "Multi-Scale Gradient Flow"
        assert log.details["levels"] == 3

    def test_different_levels(self, small_matrix):
        result1, _ = apply_multi_scale_gradient_flow(small_matrix, levels=2)
        result2, _ = apply_multi_scale_gradient_flow(small_matrix, levels=4)
        assert not np.allclose(result1, result2)


class TestSpectralPreconditioning:
    """Tests for spectral preconditioning."""

    def test_basic_operation(self, qap_matrices):
        A, B, X = qap_matrices
        eigen_A = np.linalg.eigvals(A)
        eigen_B = np.linalg.eigvals(B)
        result, log = apply_spectral_preconditioning(X, eigen_A, eigen_B)
        assert result is not None
        assert log.name == "Spectral Preconditioning"

    def test_output_normalized(self, qap_matrices):
        A, B, X = qap_matrices
        eigen_A = np.linalg.eigvals(A)
        eigen_B = np.linalg.eigvals(B)
        result, _ = apply_spectral_preconditioning(X, eigen_A, eigen_B)
        assert np.abs(result.sum() - 1.0) < 1e-6


class TestStochasticGradientVariant:
    """Tests for stochastic gradient variant."""

    def test_basic_operation(self, small_matrix, rng):
        result, log = apply_stochastic_gradient_variant(small_matrix, rng, scale=0.02)
        assert result is not None
        assert log.name == "Stochastic Gradient Variant"

    def test_reproducible_with_seed(self, small_matrix):
        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=42)
        result1, _ = apply_stochastic_gradient_variant(small_matrix.copy(), rng1)
        result2, _ = apply_stochastic_gradient_variant(small_matrix.copy(), rng2)
        assert np.allclose(result1, result2)


class TestConstrainedStep:
    """Tests for constrained step."""

    def test_basic_operation(self, small_matrix):
        result, log = apply_constrained_step(small_matrix)
        assert result is not None
        assert log.name == "Constrained Step"
        assert hasattr(result, 'matrix')

    def test_feasibility(self, small_matrix):
        result, _ = apply_constrained_step(small_matrix)
        X = result.matrix
        # Check doubly-stochastic property
        assert np.allclose(X.sum(axis=0), 1.0, atol=1e-9)
        assert np.allclose(X.sum(axis=1), 1.0, atol=1e-9)


class TestHybridContinuousDiscrete:
    """Tests for hybrid continuous-discrete."""

    def test_basic_operation(self, small_matrix):
        perm = np.array([0, 1, 2, 3])
        result, log = apply_hybrid_continuous_discrete(small_matrix, perm)
        assert result is not None
        assert log.name == "Hybrid Continuous-Discrete"

    def test_alpha_effect(self, small_matrix):
        perm = np.array([0, 1, 2, 3])
        result1, _ = apply_hybrid_continuous_discrete(small_matrix.copy(), perm, alpha=0.2)
        result2, _ = apply_hybrid_continuous_discrete(small_matrix.copy(), perm, alpha=0.8)
        # Higher alpha should be closer to continuous matrix
        assert not np.allclose(result1, result2)


class TestParallelProcessing:
    """Tests for parallel processing."""

    def test_basic_operation(self, small_matrix):
        perm = np.array([0, 1, 2, 3])

        def mock_runner(p):
            return p, 0.5

        result, log = apply_parallel_processing(perm, num_trials=2, runner_func=mock_runner)
        assert result is not None
        assert log.name == "Parallel Processing"
        assert log.details["trials"] == 2


class TestMemoryEfficientComputation:
    """Tests for memory-efficient computation."""

    def test_basic_operation(self, small_matrix):
        result, log = apply_memory_efficient_computation(small_matrix)
        assert result is not None
        assert log.name == "Memory-Efficient Algorithms"
        assert result.dtype == np.float64


# ============================================================================
# TESTS: BASELINE METHODS (9)
# ============================================================================

class TestSinkhornProjection:
    """Tests for Sinkhorn projection."""

    def test_basic_operation(self, small_matrix):
        result, log = apply_sinkhorn_projection(small_matrix)
        assert result is not None
        assert log.name == "Sinkhorn Projection"
        assert hasattr(result, 'matrix')

    def test_doubly_stochastic(self, small_matrix):
        result, _ = apply_sinkhorn_projection(small_matrix)
        X = result.matrix
        assert np.allclose(X.sum(axis=0), 1.0, atol=1e-9)
        assert np.allclose(X.sum(axis=1), 1.0, atol=1e-9)


class TestHungarianRounding:
    """Tests for Hungarian rounding."""

    def test_basic_operation(self, small_matrix):
        perm, log = apply_hungarian_rounding(small_matrix)
        assert perm is not None
        assert log.name == "Hungarian Rounding"
        assert len(perm) == small_matrix.shape[0]

    def test_valid_permutation(self, small_matrix):
        perm, _ = apply_hungarian_rounding(small_matrix)
        assert len(np.unique(perm)) == len(perm)
        assert np.all(perm >= 0) and np.all(perm < small_matrix.shape[0])


class TestTwoOptLocalSearch:
    """Tests for 2-opt local search."""

    def test_basic_operation(self, qap_matrices):
        A, B, X = qap_matrices
        n = A.shape[0]
        perm = np.arange(n, dtype=np.int64)
        result, log = apply_two_opt_local_search(A, B, perm)
        assert result is not None
        assert log.name == "2-Opt Local Search"

    def test_valid_result(self, qap_matrices):
        A, B, X = qap_matrices
        n = A.shape[0]
        perm = np.arange(n, dtype=np.int64)
        result, _ = apply_two_opt_local_search(A, B, perm)
        assert len(result) == n
        assert len(np.unique(result)) == n


class TestBasicGradientDescent:
    """Tests for basic gradient descent."""

    def test_basic_operation(self, qap_matrices):
        A, B, X = qap_matrices
        result, log = apply_basic_gradient_descent(X, A, B, step_size=0.01, iterations=10)
        assert result is not None
        assert log.name == "Basic Gradient Descent"
        assert log.details["iterations"] == 10


class TestShannonEntropyRegularization:
    """Tests for Shannon entropy regularization."""

    def test_basic_operation(self, small_matrix):
        result, log = apply_shannon_entropy_regularization(small_matrix)
        assert result is not None
        assert log.name == "Shannon Entropy Regularization"
        assert "entropy" in log.details

    def test_handles_zero_entries(self):
        matrix = np.array([
            [0.0, 0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.0, 0.5, 0.0],
            [0.1, 0.2, 0.3, 0.4],
        ])
        result, log = apply_shannon_entropy_regularization(matrix)
        assert np.all(np.isfinite(result))
        assert np.isfinite(log.details["entropy"])


class TestProbabilisticRounding:
    """Tests for probabilistic rounding."""

    def test_basic_operation(self, small_matrix, rng):
        perm, log = apply_probabilistic_rounding(small_matrix, rng)
        assert perm is not None
        assert log.name == "Probabilistic Rounding"

    def test_valid_permutation(self, small_matrix, rng):
        perm, _ = apply_probabilistic_rounding(small_matrix, rng)
        assert len(perm) == small_matrix.shape[0]
        # Should have all unique values (valid permutation)
        assert len(np.unique(perm)) == small_matrix.shape[0]
        assert np.all(perm >= 0) and np.all(perm < small_matrix.shape[0])

    def test_handles_zero_rows(self, rng):
        x = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        perm, _ = apply_probabilistic_rounding(x, rng)
        assert len(np.unique(perm)) == 3


class TestIterativeRounding:
    """Tests for iterative rounding."""

    def test_basic_operation(self, small_matrix):
        result, log = apply_iterative_rounding(small_matrix)
        assert result is not None
        assert log.name == "Iterative Rounding"

    def test_complete_assignment(self, small_matrix):
        result, _ = apply_iterative_rounding(small_matrix)
        assert len(result) == small_matrix.shape[0]


class TestContinuationMethod:
    """Tests for continuation method."""

    def test_basic_operation(self, qap_matrices):
        A, B, X = qap_matrices
        result, log = apply_continuation_method(X, A, B, num_schedules=5)
        assert result is not None
        assert log.name == "Continuation Method"


class TestAdaptiveLambdaAdjustment:
    """Tests for adaptive lambda adjustment."""

    def test_basic_operation(self, small_matrix):
        result, log = apply_adaptive_lambda_adjustment(small_matrix, constraint_violation=0.1)
        assert result is not None
        assert log.name == "Adaptive Lambda Adjustment"


# ============================================================================
# TESTS: METADATA API
# ============================================================================

class TestMetadataAPI:
    """Tests for method metadata discovery."""

    def test_get_novel_methods(self):
        novel = get_novel_methods()
        assert len(novel) == 6  # Novel only (not adapted)
        assert all(m.stars == "⭐⭐⭐" for m in novel)

    def test_get_baseline_methods(self):
        baselines = get_baseline_methods()
        assert len(baselines) == 9
        assert all(m.stars == "⭐" for m in baselines)

    def test_get_methods_by_tag(self):
        acceleration = get_methods_by_tag("acceleration")
        assert len(acceleration) > 0
        assert all("acceleration" in m.tags for m in acceleration)

    def test_method_metadata_complete(self):
        novel = get_novel_methods()
        for m in novel:
            assert m.name is not None
            assert m.function_name is not None
            assert m.complexity is not None
            assert m.stars is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple methods."""

    def test_method_chain(self, qap_matrices):
        """Test chaining multiple methods."""
        A, B, X = qap_matrices

        # FFT preconditioning
        X1, _ = apply_fft_laplace_preconditioning(X, beta=0.15)

        # Ensure feasibility
        result, _ = apply_sinkhorn_projection(X1)
        X2 = result.matrix

        # Adaptive momentum
        X3, _ = apply_adaptive_momentum(X2, previous=X)

        assert X3 is not None
        assert X3.shape == X.shape

    def test_rounding_pipeline(self, small_matrix):
        """Test complete rounding pipeline."""
        # Project to Birkhoff
        result, _ = apply_sinkhorn_projection(small_matrix)
        X_projected = result.matrix

        # Multiple rounding options
        perm_hungarian, _ = apply_hungarian_rounding(X_projected)
        assert len(perm_hungarian) == small_matrix.shape[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
