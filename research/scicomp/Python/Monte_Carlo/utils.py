"""Utility Functions for Monte Carlo Methods.
This module provides utility functions for Monte Carlo simulations
including random number generation, statistical tests, and diagnostics.
Functions:
    set_seed: Set random seed for reproducibility
    compute_statistics: Compute basic statistics
    effective_sample_size: Calculate effective sample size
    autocorrelation: Calculate autocorrelation function
    gelman_rubin: Gelman-Rubin convergence diagnostic
    geweke_test: Geweke convergence test
Classes:
    RandomNumberGenerator: Advanced random number generator
    StatisticalTests: Collection of statistical tests
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple
from scipy import stats
import warnings
def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
def compute_statistics(samples: np.ndarray,
                      axis: Optional[int] = None) -> Dict[str, float]:
    """Compute basic statistics for samples.
    Args:
        samples: Sample array
        axis: Axis along which to compute statistics
    Returns:
        Dictionary of statistics
    """
    stats_dict = {
        'mean': np.mean(samples, axis=axis),
        'std': np.std(samples, axis=axis, ddof=1),
        'var': np.var(samples, axis=axis, ddof=1),
        'min': np.min(samples, axis=axis),
        'max': np.max(samples, axis=axis),
        'median': np.median(samples, axis=axis),
        'q25': np.percentile(samples, 25, axis=axis),
        'q75': np.percentile(samples, 75, axis=axis)
    }
    # Add skewness and kurtosis if scipy is available
    try:
        stats_dict['skewness'] = stats.skew(samples, axis=axis)
        stats_dict['kurtosis'] = stats.kurtosis(samples, axis=axis)
    except:
        pass
    return stats_dict
def effective_sample_size(samples: np.ndarray,
                         max_lag: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculate effective sample size using autocorrelation.
    Args:
        samples: MCMC samples (n_samples x n_dimensions)
        max_lag: Maximum lag for autocorrelation calculation
    Returns:
        Effective sample size
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    n_samples, n_dims = samples.shape
    if max_lag is None:
        max_lag = min(n_samples // 4, 200)
    ess_values = np.zeros(n_dims)
    for d in range(n_dims):
        # Calculate autocorrelation
        autocorr = autocorrelation(samples[:, d], max_lag)
        # Find integrated autocorrelation time
        tau_int = 1 + 2 * np.sum(autocorr[1:])
        # Effective sample size
        ess_values[d] = n_samples / tau_int
    return ess_values[0] if n_dims == 1 else ess_values
def autocorrelation(samples: np.ndarray,
                   max_lag: Optional[int] = None) -> np.ndarray:
    """Calculate autocorrelation function.
    Args:
        samples: Time series samples
        max_lag: Maximum lag to calculate
    Returns:
        Autocorrelation function
    """
    n_samples = len(samples)
    if max_lag is None:
        max_lag = min(n_samples // 4, 200)
    # Center the data
    centered = samples - np.mean(samples)
    # Calculate autocorrelation using FFT
    padded = np.concatenate([centered, np.zeros(n_samples)])
    fft = np.fft.fft(padded)
    autocorr_full = np.fft.ifft(fft * np.conj(fft)).real
    # Normalize and extract relevant lags
    autocorr = autocorr_full[:max_lag+1] / autocorr_full[0]
    return autocorr
def gelman_rubin(chains: List[np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate Gelman-Rubin convergence diagnostic.
    Args:
        chains: List of MCMC chains
    Returns:
        Gelman-Rubin statistic (R-hat)
    """
    if not chains:
        raise ValueError("Need at least one chain")
    # Convert to array format
    chains = [np.asarray(chain) for chain in chains]
    # Check dimensions
    n_chains = len(chains)
    n_samples = len(chains[0])
    if chains[0].ndim == 1:
        n_dims = 1
        chains = [chain.reshape(-1, 1) for chain in chains]
    else:
        n_dims = chains[0].shape[1]
    # Check all chains have same length
    for i, chain in enumerate(chains[1:], 1):
        if len(chain) != n_samples:
            raise ValueError(f"Chain {i} has different length than chain 0")
    r_hat = np.zeros(n_dims)
    for d in range(n_dims):
        # Extract dimension d from all chains
        chain_data = np.array([chain[:, d] for chain in chains])
        # Calculate between-chain and within-chain variance
        chain_means = np.mean(chain_data, axis=1)
        grand_mean = np.mean(chain_means)
        # Between-chain variance
        B = n_samples / (n_chains - 1) * np.sum((chain_means - grand_mean)**2)
        # Within-chain variance
        chain_vars = np.var(chain_data, axis=1, ddof=1)
        W = np.mean(chain_vars)
        # Marginal posterior variance
        var_plus = (n_samples - 1) / n_samples * W + B / n_samples
        # R-hat statistic
        r_hat[d] = np.sqrt(var_plus / W)
    return r_hat[0] if n_dims == 1 else r_hat
def geweke_test(samples: np.ndarray,
               first_fraction: float = 0.1,
               last_fraction: float = 0.5) -> Dict[str, float]:
    """Geweke convergence test.
    Args:
        samples: MCMC samples
        first_fraction: Fraction of samples from beginning
        last_fraction: Fraction of samples from end
    Returns:
        Dictionary with test results
    """
    n_samples = len(samples)
    # Split samples
    n_first = int(first_fraction * n_samples)
    n_last = int(last_fraction * n_samples)
    first_samples = samples[:n_first]
    last_samples = samples[-n_last:]
    # Calculate means and variances
    mean_first = np.mean(first_samples)
    mean_last = np.mean(last_samples)
    var_first = np.var(first_samples, ddof=1) / len(first_samples)
    var_last = np.var(last_samples, ddof=1) / len(last_samples)
    # Z-score
    z_score = (mean_first - mean_last) / np.sqrt(var_first + var_last)
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
    return {
        'z_score': z_score,
        'p_value': p_value,
        'mean_first': mean_first,
        'mean_last': mean_last,
        'converged': p_value > 0.05  # Common threshold
    }
class RandomNumberGenerator:
    """Advanced random number generator with multiple distributions.
    Provides access to various random number distributions commonly
    used in Monte Carlo simulations.
    """
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.seed = seed
    def uniform(self, low: float = 0.0, high: float = 1.0,
               size: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        """Generate uniform random numbers."""
        return self.rng.uniform(low, high, size)
    def normal(self, loc: float = 0.0, scale: float = 1.0,
              size: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        """Generate normal random numbers."""
        return self.rng.normal(loc, scale, size)
    def multivariate_normal(self, mean: np.ndarray, cov: np.ndarray,
                           size: Optional[int] = None) -> np.ndarray:
        """Generate multivariate normal random numbers."""
        return self.rng.multivariate_normal(mean, cov, size)
    def exponential(self, scale: float = 1.0,
                   size: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        """Generate exponential random numbers."""
        return self.rng.exponential(scale, size)
    def gamma(self, shape: float, scale: float = 1.0,
             size: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        """Generate gamma random numbers."""
        return self.rng.gamma(shape, scale, size)
    def beta(self, a: float, b: float,
            size: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        """Generate beta random numbers."""
        return self.rng.beta(a, b, size)
    def lognormal(self, mean: float = 0.0, sigma: float = 1.0,
                 size: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        """Generate lognormal random numbers."""
        return self.rng.lognormal(mean, sigma, size)
    def weibull(self, a: float,
               size: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        """Generate Weibull random numbers."""
        return self.rng.weibull(a, size)
    def choice(self, a: Union[int, np.ndarray], size: Optional[int] = None,
              replace: bool = True, p: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate random choice from array."""
        return self.rng.choice(a, size, replace, p)
    def permutation(self, x: Union[int, np.ndarray]) -> np.ndarray:
        """Generate random permutation."""
        return self.rng.permutation(x)
    def shuffle(self, x: np.ndarray) -> None:
        """Shuffle array in-place."""
        self.rng.shuffle(x)
class StatisticalTests:
    """Collection of statistical tests for Monte Carlo diagnostics.
    Provides various statistical tests commonly used to assess
    the quality and convergence of Monte Carlo simulations.
    """
    @staticmethod
    def kolmogorov_smirnov_test(samples1: np.ndarray,
                               samples2: np.ndarray) -> Dict[str, float]:
        """Kolmogorov-Smirnov two-sample test.
        Args:
            samples1: First sample
            samples2: Second sample
        Returns:
            Dictionary with test results
        """
        statistic, p_value = stats.ks_2samp(samples1, samples2)
        return {
            'statistic': statistic,
            'p_value': p_value,
            'samples_different': p_value < 0.05
        }
    @staticmethod
    def anderson_darling_test(samples: np.ndarray,
                             distribution: str = 'norm') -> Dict[str, Any]:
        """Anderson-Darling test for distribution fitting.
        Args:
            samples: Sample data
            distribution: Distribution to test against
        Returns:
            Dictionary with test results
        """
        try:
            result = stats.anderson(samples, dist=distribution)
            return {
                'statistic': result.statistic,
                'critical_values': result.critical_values,
                'significance_levels': result.significance_level,
                'distribution': distribution
            }
        except Exception as e:
            warnings.warn(f"Anderson-Darling test failed: {e}")
            return {'error': str(e)}
    @staticmethod
    def shapiro_wilk_test(samples: np.ndarray) -> Dict[str, float]:
        """Shapiro-Wilk test for normality.
        Args:
            samples: Sample data
        Returns:
            Dictionary with test results
        """
        if len(samples) > 5000:
            # Subsample for efficiency
            indices = np.random.choice(len(samples), 5000, replace=False)
            samples = samples[indices]
        statistic, p_value = stats.shapiro(samples)
        return {
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
    @staticmethod
    def jarque_bera_test(samples: np.ndarray) -> Dict[str, float]:
        """Jarque-Bera test for normality.
        Args:
            samples: Sample data
        Returns:
            Dictionary with test results
        """
        statistic, p_value = stats.jarque_bera(samples)
        return {
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
    @staticmethod
    def ljung_box_test(samples: np.ndarray, lags: int = 10) -> Dict[str, float]:
        """Ljung-Box test for autocorrelation.
        Args:
            samples: Sample data
            lags: Number of lags to test
        Returns:
            Dictionary with test results
        """
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            result = acorr_ljungbox(samples, lags=lags, return_df=False)
            return {
                'statistic': result[0][-1],  # Last lag statistic
                'p_value': result[1][-1],    # Last lag p-value
                'no_autocorrelation': result[1][-1] > 0.05
            }
        except ImportError:
            warnings.warn("statsmodels not available for Ljung-Box test")
            return {'error': 'statsmodels not available'}
        except Exception as e:
            warnings.warn(f"Ljung-Box test failed: {e}")
            return {'error': str(e)}
    @staticmethod
    def runs_test(samples: np.ndarray) -> Dict[str, float]:
        """Runs test for randomness.
        Args:
            samples: Sample data
        Returns:
            Dictionary with test results
        """
        # Convert to binary sequence (above/below median)
        median = np.median(samples)
        binary = (samples > median).astype(int)
        # Count runs
        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                runs += 1
        # Expected runs and variance under null hypothesis
        n1 = np.sum(binary)
        n2 = len(binary) - n1
        if n1 == 0 or n2 == 0:
            return {'error': 'All values above or below median'}
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / \
                       ((n1 + n2)**2 * (n1 + n2 - 1))
        # Z-score
        z_score = (runs - expected_runs) / np.sqrt(variance_runs)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
        return {
            'runs': runs,
            'expected_runs': expected_runs,
            'z_score': z_score,
            'p_value': p_value,
            'is_random': p_value > 0.05
        }
def heidelberg_welch_test(samples: np.ndarray,
                         alpha: float = 0.05,
                         eps: float = 0.1) -> Dict[str, Any]:
    """Heidelberg-Welch convergence test.
    Args:
        samples: MCMC samples
        alpha: Significance level
        eps: Target accuracy
    Returns:
        Dictionary with test results
    """
    n = len(samples)
    # Halfwidth test
    mean_estimate = np.mean(samples)
    std_estimate = np.std(samples, ddof=1)
    # Critical value
    z_alpha = stats.norm.ppf(1 - alpha/2)
    # Halfwidth
    halfwidth = z_alpha * std_estimate / np.sqrt(n)
    # Test if halfwidth is small enough
    halfwidth_test = halfwidth < eps * np.abs(mean_estimate)
    # Stationarity test (simplified version)
    # Split chain into segments and test for equality of means
    n_segments = 4
    segment_size = n // n_segments
    segment_means = []
    for i in range(n_segments):
        start = i * segment_size
        end = start + segment_size
        segment_means.append(np.mean(samples[start:end]))
    # F-test for equality of means
    try:
        f_stat, p_value = stats.f_oneway(*[samples[i*segment_size:(i+1)*segment_size]
                                         for i in range(n_segments)])
        stationarity_test = p_value > alpha
    except:
        stationarity_test = False
        p_value = 0.0
    return {
        'halfwidth_test': halfwidth_test,
        'stationarity_test': stationarity_test,
        'halfwidth': halfwidth,
        'mean_estimate': mean_estimate,
        'stationarity_p_value': p_value,
        'converged': halfwidth_test and stationarity_test
    }