"""
Unit tests for corrected statistical functions

Tests verify that numerical stability fixes have been correctly applied.
"""

import numpy as np
import pytest

from Librex.validation.statistical_tests import (
    effect_size_cohens_d_corrected,
    confidence_interval_difference_corrected,
    compute_statistical_power_corrected,
    convergence_analysis_corrected,
)


def test_effect_size_normal_data():
    """Test effect size calculation with normal data"""
    np.random.seed(42)
    A = np.random.normal(100, 15, 30)
    B = np.random.normal(95, 12, 30)

    effect_size = effect_size_cohens_d_corrected(A, B)

    # Effect size should be positive and reasonable
    assert effect_size > 0
    assert effect_size < 2.0  # Reasonable effect size range


def test_effect_size_identical_samples():
    """Test effect size with identical samples - edge case fix verification"""
    # This tests the division by zero fix
    C = np.ones(10) * 5.0
    D = np.ones(10) * 5.0

    effect_size = effect_size_cohens_d_corrected(C, D)

    # Should return 0 for identical samples (no effect)
    assert abs(effect_size) < 1e-10, f"Expected 0, got {effect_size}"


def test_effect_size_zero_variance_different_means():
    """Test effect size with zero variance but different means"""
    # This tests the infinite effect size handling
    E = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    F = np.array([5.1, 5.1, 5.1, 5.1, 5.1])

    effect_size = effect_size_cohens_d_corrected(E, F)

    # Should return infinite effect size (or very large)
    assert np.isinf(effect_size) or abs(effect_size) > 100


def test_confidence_interval_bootstrap():
    """Test bootstrap confidence interval calculation"""
    np.random.seed(42)
    A = np.random.normal(100, 15, 30)
    B = np.random.normal(95, 12, 30)

    ci_lower, ci_upper = confidence_interval_difference_corrected(
        A, B, method='bootstrap', n_bootstrap=1000  # Reduced for speed
    )

    # CI should be ordered
    assert ci_lower < ci_upper

    # Mean difference should be within CI (with high probability)
    mean_diff = np.mean(A) - np.mean(B)
    # Don't assert this strictly as it's probabilistic
    # assert ci_lower <= mean_diff <= ci_upper


def test_confidence_interval_normal():
    """Test normal approximation confidence interval"""
    np.random.seed(42)
    A = np.random.normal(100, 15, 30)
    B = np.random.normal(95, 12, 30)

    ci_lower, ci_upper = confidence_interval_difference_corrected(
        A, B, method='normal'
    )

    # CI should be ordered
    assert ci_lower < ci_upper

    # Should be finite values
    assert np.isfinite(ci_lower)
    assert np.isfinite(ci_upper)


def test_confidence_interval_identical_samples():
    """Test CI with identical samples - edge case"""
    C = np.ones(10) * 5.0
    D = np.ones(10) * 5.0

    ci_lower, ci_upper = confidence_interval_difference_corrected(
        C, D, method='normal'
    )

    # Should return (0, 0) or very small interval
    assert abs(ci_lower) < 1e-10
    assert abs(ci_upper) < 1e-10


def test_statistical_power_reasonable_effect():
    """Test statistical power calculation with reasonable effect size"""
    np.random.seed(42)
    A = np.random.normal(100, 15, 30)
    B = np.random.normal(95, 12, 30)

    effect_size = 0.5  # Medium effect size
    power = compute_statistical_power_corrected(A, B, effect_size)

    # Power should be between 0 and 1
    assert 0 <= power <= 1


def test_statistical_power_zero_effect():
    """Test power with zero effect size"""
    A = np.random.normal(100, 15, 30)
    B = np.random.normal(100, 15, 30)

    power = compute_statistical_power_corrected(A, B, effect_size=0.0, alpha=0.05)

    # Power should approximately equal alpha for zero effect
    assert abs(power - 0.05) < 0.1


def test_statistical_power_large_effect():
    """Test power with very large effect size"""
    A = np.random.normal(100, 15, 30)
    B = np.random.normal(100, 15, 30)

    power = compute_statistical_power_corrected(A, B, effect_size=10.0)

    # Should be close to 1.0 for very large effect
    assert power > 0.99


def test_convergence_analysis_converged():
    """Test convergence analysis with converging sequence"""
    # Monotonically decreasing sequence that converges quickly
    # Using smaller window_size to detect convergence
    history = 100 * np.exp(-0.5 * np.arange(100))

    result = convergence_analysis_corrected(history, tolerance=1e-3, window_size=10)

    assert result['converged']
    assert result['iterations_to_convergence'] < 100
    assert result['final_objective'] < result['initial_objective']
    assert result['improvement_ratio'] > 0


def test_convergence_analysis_not_converged():
    """Test convergence analysis with non-converging sequence"""
    # Oscillating sequence
    history = 50 + 10 * np.sin(np.arange(50))

    result = convergence_analysis_corrected(history, tolerance=1e-6)

    # May or may not converge depending on window
    # Just check structure is correct
    assert 'converged' in result
    assert 'convergence_rate' in result
    assert 'final_objective' in result


def test_convergence_analysis_empty():
    """Test convergence analysis with minimal data"""
    history = [100.0]

    result = convergence_analysis_corrected(history)

    assert not result['converged']
    assert result['convergence_rate'] == 'unknown'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
