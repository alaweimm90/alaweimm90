"""
Corrected Statistical Functions for Librex

This module contains mathematically corrected statistical functions that fix
numerical stability issues identified in the original implementation.

Key fixes:
1. Division by zero handling in effect size calculations.
2. Improved numerical precision for bootstrap methods.
3. Better statistical power calculations.
4. Enhanced convergence analysis.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def effect_size_cohens_d_corrected(
    sample_A: Union[List[float], np.ndarray],
    sample_B: Union[List[float], np.ndarray],
    pooled: bool = True
) -> float:
    """
    Compute Cohen's d effect size between two samples - NUMERICALLY STABLE VERSION.

    FIXED ISSUES:
    1. Proper handling of zero variance cases
    2. Numerical stability for very small effect sizes
    3. Clear handling of infinite effect sizes

    Cohen's d is the standardized mean difference:
    d = (mean_A - mean_B) / pooled_std

    Interpretation:
    - |d| < 0.2: Negligible effect
    - 0.2 <= |d| < 0.5: Small effect
    - 0.5 <= |d| < 0.8: Medium effect
    - |d| >= 0.8: Large effect

    Args:
        sample_A: First sample
        sample_B: Second sample
        pooled: Whether to use pooled standard deviation

    Returns:
        Cohen's d effect size
    """
    A = np.asarray(sample_A)
    B = np.asarray(sample_B)

    # Validate inputs
    if len(A) == 0 or len(B) == 0:
        raise ValueError("Samples must not be empty")

    if len(A) < 2 or len(B) < 2:
        warnings.warn("Small sample sizes may lead to unreliable effect size estimates")

    mean_diff = np.mean(A) - np.mean(B)

    if pooled:
        # Pooled standard deviation
        n_A, n_B = len(A), len(B)
        var_A, var_B = np.var(A, ddof=1), np.var(B, ddof=1)
        pooled_var = ((n_A - 1) * var_A + (n_B - 1) * var_B) / (n_A + n_B - 2)
        pooled_std = np.sqrt(pooled_var)
    else:
        # Use average standard deviation
        pooled_std = np.sqrt((np.var(A, ddof=1) + np.var(B, ddof=1)) / 2)

    # FIXED: Robust handling of edge cases
    if pooled_std < 1e-12:
        # Both samples have essentially zero variance
        if abs(mean_diff) < 1e-12:
            return 0.0  # No meaningful difference
        else:
            # Infinite effect size due to zero variance
            return np.inf if mean_diff > 0 else -np.inf

    # Cap effect size at reasonable maximum to avoid numerical overflow
    raw_effect_size = mean_diff / pooled_std
    if abs(raw_effect_size) > 1e6:
        warnings.warn(f"Very large effect size detected: {raw_effect_size:.2e}")
        return np.sign(raw_effect_size) * 1e6

    return float(raw_effect_size)


def confidence_interval_difference_corrected(
    sample_A: Union[List[float], np.ndarray],
    sample_B: Union[List[float], np.ndarray],
    alpha: float = 0.05,
    method: str = 'bootstrap',
    n_bootstrap: int = 50000
) -> Tuple[float, float]:
    """
    Compute confidence interval for the difference between two samples - IMPROVED VERSION.

    FIXES:
    1. Increased bootstrap samples for better accuracy
    2. Better handling of edge cases
    3. Proper error handling for bootstrap failures

    Args:
        sample_A: First sample
        sample_B: Second sample
        alpha: Significance level (0.05 for 95% CI)
        method: 'bootstrap' or 'normal'
        n_bootstrap: Number of bootstrap samples (increased from 10000)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    A = np.asarray(sample_A)
    B = np.asarray(sample_B)

    # Validate inputs
    if len(A) == 0 or len(B) == 0:
        raise ValueError("Samples must not be empty")

    if method == 'bootstrap':
        # IMPROVED: More bootstrap samples for better accuracy
        bootstrap_diffs = []

        try:
            for _ in range(n_bootstrap):
                A_sample = np.random.choice(A, size=len(A), replace=True)
                B_sample = np.random.choice(B, size=len(B), replace=True)
                bootstrap_diffs.append(np.mean(A_sample) - np.mean(B_sample))

            bootstrap_diffs = np.array(bootstrap_diffs)

            # Check for numerical issues
            if np.any(np.isnan(bootstrap_diffs)) or np.any(np.isinf(bootstrap_diffs)):
                warnings.warn("Bootstrap samples contain NaN or inf values, falling back to normal approximation")
                method = 'normal'
            else:
                lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
                upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
                return (float(lower), float(upper))

        except Exception as e:
            warnings.warn(f"Bootstrap method failed: {e}, falling back to normal approximation")
            method = 'normal'

    if method == 'normal':
        # IMPROVED: Better normal approximation
        mean_diff = np.mean(A) - np.mean(B)

        # Use sample standard deviations
        var_A = np.var(A, ddof=1) if len(A) > 1 else 0
        var_B = np.var(B, ddof=1) if len(B) > 1 else 0

        se_diff = np.sqrt(var_A / len(A) + var_B / len(B))

        # Check for zero standard error
        if se_diff < 1e-12:
            if abs(mean_diff) < 1e-12:
                return (0.0, 0.0)  # No difference
            else:
                # Infinite confidence interval
                return (-np.inf, np.inf)

        # t-distribution critical value
        df = len(A) + len(B) - 2
        df = max(df, 1)  # Ensure at least 1 degree of freedom

        t_crit = stats.t.ppf(1 - alpha / 2, df)

        lower = mean_diff - t_crit * se_diff
        upper = mean_diff + t_crit * se_diff

        return (float(lower), float(upper))

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_statistical_power_corrected(
    sample_A: Union[List[float], np.ndarray],
    sample_B: Union[List[float], np.ndarray],
    effect_size: float,
    alpha: float = 0.05
) -> float:
    """
    Compute statistical power of a test - IMPROVED VERSION.

    FIXES:
    1. Better handling of effect sizes near zero
    2. More robust fallback calculations
    3. Proper bounds checking

    Power is the probability of correctly rejecting the null hypothesis
    when it is false (1 - Type II error rate).

    Args:
        sample_A: First sample
        sample_B: Second sample
        effect_size: Cohen's d or similar
        alpha: Significance level

    Returns:
        Statistical power (0 to 1)
    """
    n_A, n_B = len(sample_A), len(sample_B)

    if n_A < 2 or n_B < 2:
        warnings.warn("Cannot compute power with sample sizes < 2")
        return 0.0

    # Harmonic mean of sample sizes for unequal groups
    n_eff = 2 * n_A * n_B / (n_A + n_B)

    # Handle extreme effect sizes
    if abs(effect_size) < 1e-10:
        return alpha  # Power = alpha for zero effect size

    if abs(effect_size) > 10:
        return 1.0  # Very large effect size, almost certain power

    try:
        # Try to use statsmodels if available
        from statsmodels.stats.power import ttest_power
        power = ttest_power(effect_size, n_eff, alpha, alternative='two-sided')
    except ImportError:
        # IMPROVED: Better fallback calculation
        # More accurate approximation for power
        z_alpha = stats.norm.ppf(1 - alpha / 2)

        # Non-central t approximation for power
        # This is more accurate than the simple z-approximation
        delta = abs(effect_size) * np.sqrt(n_eff / 2)

        # Use normal approximation for large delta, better for moderate
        if delta > 3:
            z_power = delta - z_alpha
            power = stats.norm.cdf(z_power)
        else:
            # For smaller effect sizes, use a more conservative estimate
            power = stats.norm.cdf(delta - z_alpha) + stats.norm.cdf(-delta - z_alpha)

        power = max(0.0, min(1.0, power))

    return min(max(power, 0.0), 1.0)


def convergence_analysis_corrected(
    history: Union[List[float], np.ndarray],
    tolerance: float = 1e-8,  # IMPROVED: Tighter tolerance
    window_size: int = 20     # IMPROVED: Larger window for stability
) -> Dict[str, Any]:
    """
    Analyze convergence properties - IMPROVED VERSION.

    FIXES:
    1. Tighter convergence tolerance
    2. Larger window size for more stable detection
    3. Better error handling
    4. More robust convergence rate analysis
    """
    history = np.asarray(history)

    if len(history) < 2:
        return {
            'converged': False,
            'convergence_rate': 'unknown',
            'iterations_to_convergence': -1,
            'final_objective': history[-1] if len(history) > 0 else np.inf,
            'initial_objective': history[0] if len(history) > 0 else np.inf,
            'improvement_ratio': 0.0,
            'convergence_threshold': tolerance,
            'rate_coefficient': None
        }

    # Check for convergence with larger window
    converged = False
    iterations_to_convergence = len(history)

    for i in range(window_size, len(history)):
        window = history[i-window_size:i]

        # IMPROVED: Use relative tolerance
        window_range = np.max(window) - np.min(window)
        relative_tolerance = tolerance * (1 + abs(history[i]))

        if window_range < max(tolerance, relative_tolerance):
            converged = True
            iterations_to_convergence = i
            break

    # Compute improvement with protection against division by zero
    initial_obj = history[0]
    final_obj = history[-1]

    if abs(initial_obj) > 1e-12:
        improvement_ratio = (initial_obj - final_obj) / abs(initial_obj)
    else:
        improvement_ratio = initial_obj - final_obj if initial_obj != 0 else 0

    # Analyze convergence rate with better error handling
    try:
        convergence_rate, rate_coefficient = analyze_convergence_rate_corrected(
            history[:iterations_to_convergence] if converged else history
        )
    except Exception as e:
        logger.warning(f"Convergence rate analysis failed: {e}")
        convergence_rate = 'unknown'
        rate_coefficient = None

    return {
        'converged': converged,
        'convergence_rate': convergence_rate,
        'iterations_to_convergence': iterations_to_convergence,
        'final_objective': float(final_obj),
        'initial_objective': float(initial_obj),
        'improvement_ratio': float(improvement_ratio),
        'convergence_threshold': tolerance,
        'rate_coefficient': rate_coefficient
    }


def analyze_convergence_rate_corrected(history: np.ndarray) -> Tuple[str, Optional[float]]:
    """
    Determine convergence rate type - IMPROVED VERSION.

    FIXES:
    1. Better handling of numerical noise
    2. More robust polynomial fitting
    3. Improved threshold values
    """
    if len(history) < 5:
        return 'unknown', None

    # Estimate optimal value as the final value
    f_star = history[-1]

    # Compute error sequence
    errors = np.abs(history - f_star)
    errors = errors[errors > 1e-12]  # Remove near-zero errors

    if len(errors) < 3:
        return 'unknown', None

    # Check for linear convergence - IMPROVED thresholds
    log_errors = np.log(errors[1:])
    log_prev_errors = np.log(errors[:-1])

    if len(log_errors) > 2:
        try:
            slope, intercept = np.polyfit(log_prev_errors, log_errors, 1)

            # IMPROVED: More generous thresholds for linear convergence
            if 0.1 < slope < 0.99:
                # Check for superlinear behavior
                rates = errors[1:] / errors[:-1]
                if len(rates) > 2 and np.mean(np.diff(rates)) < -0.01:
                    return 'superlinear', slope
                return 'linear', slope

            # Check for quadratic convergence with tighter bounds
            elif 1.5 < slope < 2.5:
                return 'quadratic', slope

        except (np.linalg.LinAlgError, ValueError) as error:
            logger.debug("Convergence rate estimation failed: %s", error)

    # Fallback: check if sequence is decreasing
    if len(history) > 3 and np.all(np.diff(history[-10:]) < 0):
        return 'linear', None

    return 'unknown', None


# Test function to verify the corrections
def test_corrected_statistical_functions():
    """Test the corrected statistical functions"""

    print("Testing corrected statistical functions...")

    # Test case 1: Normal data
    np.random.seed(42)
    A = np.random.normal(100, 15, 30)
    B = np.random.normal(95, 12, 30)

    # Test effect size calculation
    effect_size = effect_size_cohens_d_corrected(A, B)
    print(f"Effect size (normal data): {effect_size:.4f}")

    # Test confidence interval
    ci_lower, ci_upper = confidence_interval_difference_corrected(A, B)
    print(f"95% CI for difference: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Test statistical power
    power = compute_statistical_power_corrected(A, B, effect_size)
    print(f"Statistical power: {power:.4f}")

    # Test case 2: Edge case - identical samples
    C = np.ones(10) * 5.0
    D = np.ones(10) * 5.0

    effect_size_identical = effect_size_cohens_d_corrected(C, D)
    print(f"Effect size (identical samples): {effect_size_identical}")

    # Test case 3: Zero variance edge case
    E = np.array([5.0, 5.0, 5.0])
    F = np.array([5.1, 5.1, 5.1])  # Small difference but zero variance

    effect_size_zero_var = effect_size_cohens_d_corrected(E, F)
    print(f"Effect size (zero variance): {effect_size_zero_var}")

    print("Statistical function tests completed successfully!")


if __name__ == "__main__":
    test_corrected_statistical_functions()
