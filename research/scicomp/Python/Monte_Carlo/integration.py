"""Monte Carlo Integration Methods.
This module implements various Monte Carlo integration techniques
for scientific computing applications.
Classes:
    MonteCarloIntegrator: Standard Monte Carlo integration
    QuasiMonteCarloIntegrator: Low-discrepancy sequences
    AdaptiveMonteCarloIntegrator: Adaptive variance reduction
Functions:
    monte_carlo_integrate: Standard MC integration
    quasi_monte_carlo_integrate: QMC integration
    importance_sampling_integrate: Importance sampling
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
from typing import Callable, Union, Tuple, List, Optional, Dict, Any
from scipy import stats
from dataclasses import dataclass
import warnings
from .utils import set_seed, compute_statistics
from .constants import MATHEMATICAL_CONSTANTS
@dataclass
class IntegrationResult:
    """Result of Monte Carlo integration.
    Attributes:
        value: Estimated integral value
        error: Estimated error
        variance: Sample variance
        n_samples: Number of samples used
        convergence_history: History of estimates
        efficiency: Computational efficiency metric
    """
    value: float
    error: float
    variance: float
    n_samples: int
    convergence_history: List[float]
    efficiency: float
    metadata: Dict[str, Any]
class MonteCarloIntegrator:
    """Standard Monte Carlo integrator.
    Implements basic Monte Carlo integration with variance estimation
    and convergence monitoring.
    Parameters:
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information
    """
    def __init__(self, random_state: Optional[int] = None, verbose: bool = False):
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.RandomState(random_state)
    def integrate(self,
                 func: Callable,
                 bounds: Union[Tuple[float, float], List[Tuple[float, float]]],
                 n_samples: int = 10000,
                 **kwargs) -> IntegrationResult:
        """Integrate function using Monte Carlo method.
        Args:
            func: Function to integrate
            bounds: Integration bounds (1D or multi-dimensional)
            n_samples: Number of random samples
            **kwargs: Additional arguments
        Returns:
            IntegrationResult: Integration results
        """
        # Handle 1D vs multi-dimensional integration
        if isinstance(bounds[0], (int, float)):
            return self._integrate_1d(func, bounds, n_samples, **kwargs)
        else:
            return self._integrate_nd(func, bounds, n_samples, **kwargs)
    def _integrate_1d(self, func: Callable, bounds: Tuple[float, float],
                     n_samples: int, **kwargs) -> IntegrationResult:
        """1D Monte Carlo integration."""
        a, b = bounds
        volume = b - a
        # Generate random samples
        samples = self.rng.uniform(a, b, n_samples)
        # Evaluate function
        try:
            function_values = np.array([func(x) for x in samples])
        except:
            # Vectorized evaluation if possible
            function_values = func(samples)
        # Compute integral estimate
        integral_estimate = volume * np.mean(function_values)
        # Compute variance and error
        variance = np.var(function_values, ddof=1)
        error = volume * np.sqrt(variance / n_samples)
        # Convergence history
        convergence_history = self._compute_convergence_history(function_values, volume)
        return IntegrationResult(
            value=integral_estimate,
            error=error,
            variance=variance,
            n_samples=n_samples,
            convergence_history=convergence_history,
            efficiency=1.0 / variance if variance > 0 else np.inf,
            metadata={'method': 'monte_carlo', 'dimension': 1}
        )
    def _integrate_nd(self, func: Callable, bounds: List[Tuple[float, float]],
                     n_samples: int, **kwargs) -> IntegrationResult:
        """Multi-dimensional Monte Carlo integration."""
        dimension = len(bounds)
        volume = np.prod([b - a for a, b in bounds])
        # Generate random samples
        samples = np.zeros((n_samples, dimension))
        for i, (a, b) in enumerate(bounds):
            samples[:, i] = self.rng.uniform(a, b, n_samples)
        # Evaluate function
        function_values = np.array([func(sample) for sample in samples])
        # Compute integral estimate
        integral_estimate = volume * np.mean(function_values)
        # Compute variance and error
        variance = np.var(function_values, ddof=1)
        error = volume * np.sqrt(variance / n_samples)
        # Convergence history
        convergence_history = self._compute_convergence_history(function_values, volume)
        return IntegrationResult(
            value=integral_estimate,
            error=error,
            variance=variance,
            n_samples=n_samples,
            convergence_history=convergence_history,
            efficiency=1.0 / variance if variance > 0 else np.inf,
            metadata={'method': 'monte_carlo', 'dimension': dimension}
        )
    def _compute_convergence_history(self, function_values: np.ndarray,
                                   volume: float) -> List[float]:
        """Compute convergence history."""
        n_points = 20
        sample_sizes = np.logspace(2, np.log10(len(function_values)), n_points, dtype=int)
        sample_sizes = np.unique(sample_sizes)
        history = []
        for n in sample_sizes:
            if n <= len(function_values):
                partial_estimate = volume * np.mean(function_values[:n])
                history.append(partial_estimate)
        return history
class QuasiMonteCarloIntegrator:
    """Quasi-Monte Carlo integrator using low-discrepancy sequences.
    Implements QMC integration using Sobol sequences for better
    convergence than standard Monte Carlo.
    """
    def __init__(self, random_state: Optional[int] = None, verbose: bool = False):
        self.random_state = random_state
        self.verbose = verbose
    def integrate(self,
                 func: Callable,
                 bounds: Union[Tuple[float, float], List[Tuple[float, float]]],
                 n_samples: int = 10000,
                 sequence_type: str = 'sobol',
                 **kwargs) -> IntegrationResult:
        """Integrate using quasi-Monte Carlo method.
        Args:
            func: Function to integrate
            bounds: Integration bounds
            n_samples: Number of samples
            sequence_type: Type of low-discrepancy sequence
            **kwargs: Additional arguments
        Returns:
            IntegrationResult: Integration results
        """
        try:
            from scipy.stats import qmc
        except ImportError:
            warnings.warn("scipy.stats.qmc not available, falling back to standard MC")
            integrator = MonteCarloIntegrator(self.random_state, self.verbose)
            return integrator.integrate(func, bounds, n_samples, **kwargs)
        # Handle bounds format
        if isinstance(bounds[0], (int, float)):
            bounds = [bounds]
        dimension = len(bounds)
        # Generate low-discrepancy sequence
        if sequence_type == 'sobol':
            sampler = qmc.Sobol(d=dimension, seed=self.random_state)
        elif sequence_type == 'halton':
            sampler = qmc.Halton(d=dimension, seed=self.random_state)
        else:
            raise ValueError(f"Unknown sequence type: {sequence_type}")
        # Generate samples
        unit_samples = sampler.random(n_samples)
        # Transform to integration domain
        samples = np.zeros_like(unit_samples)
        volume = 1.0
        for i, (a, b) in enumerate(bounds):
            samples[:, i] = a + (b - a) * unit_samples[:, i]
            volume *= (b - a)
        # Evaluate function
        if dimension == 1:
            function_values = np.array([func(sample[0]) for sample in samples])
        else:
            function_values = np.array([func(sample) for sample in samples])
        # Compute integral estimate
        integral_estimate = volume * np.mean(function_values)
        # Estimate error (QMC error is O(n^{-1}) vs O(n^{-1/2}) for MC)
        variance = np.var(function_values, ddof=1)
        # QMC error estimation is more complex, use conservative estimate
        error = volume * np.sqrt(variance / n_samples) / np.sqrt(np.log(n_samples))
        # Convergence history
        convergence_history = self._compute_convergence_history(function_values, volume)
        return IntegrationResult(
            value=integral_estimate,
            error=error,
            variance=variance,
            n_samples=n_samples,
            convergence_history=convergence_history,
            efficiency=1.0 / variance if variance > 0 else np.inf,
            metadata={'method': 'quasi_monte_carlo', 'sequence': sequence_type, 'dimension': dimension}
        )
    def _compute_convergence_history(self, function_values: np.ndarray,
                                   volume: float) -> List[float]:
        """Compute convergence history."""
        n_points = 20
        sample_sizes = np.logspace(2, np.log10(len(function_values)), n_points, dtype=int)
        sample_sizes = np.unique(sample_sizes)
        history = []
        for n in sample_sizes:
            if n <= len(function_values):
                partial_estimate = volume * np.mean(function_values[:n])
                history.append(partial_estimate)
        return history
class AdaptiveMonteCarloIntegrator:
    """Adaptive Monte Carlo integrator with variance reduction.
    Implements adaptive importance sampling and stratified sampling
    for improved efficiency.
    """
    def __init__(self, random_state: Optional[int] = None, verbose: bool = False):
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.RandomState(random_state)
    def integrate(self,
                 func: Callable,
                 bounds: Union[Tuple[float, float], List[Tuple[float, float]]],
                 n_samples: int = 10000,
                 adaptation_strategy: str = 'importance',
                 **kwargs) -> IntegrationResult:
        """Adaptive Monte Carlo integration.
        Args:
            func: Function to integrate
            bounds: Integration bounds
            n_samples: Number of samples
            adaptation_strategy: Adaptation strategy ('importance', 'stratified')
            **kwargs: Additional arguments
        Returns:
            IntegrationResult: Integration results
        """
        if adaptation_strategy == 'importance':
            return self._importance_sampling_integration(func, bounds, n_samples, **kwargs)
        elif adaptation_strategy == 'stratified':
            return self._stratified_sampling_integration(func, bounds, n_samples, **kwargs)
        else:
            raise ValueError(f"Unknown adaptation strategy: {adaptation_strategy}")
    def _importance_sampling_integration(self, func: Callable, bounds, n_samples: int, **kwargs):
        """Integration using importance sampling."""
        # Handle bounds format
        if isinstance(bounds[0], (int, float)):
            bounds = [bounds]
        dimension = len(bounds)
        volume = np.prod([b - a for a, b in bounds])
        # Initial exploration phase
        n_explore = min(1000, n_samples // 4)
        explore_samples = np.zeros((n_explore, dimension))
        for i, (a, b) in enumerate(bounds):
            explore_samples[:, i] = self.rng.uniform(a, b, n_explore)
        # Evaluate function for exploration
        if dimension == 1:
            explore_values = np.array([func(sample[0]) for sample in explore_samples])
        else:
            explore_values = np.array([func(sample) for sample in explore_samples])
        # Adaptive importance distribution (simplified)
        # Use samples with higher function values more frequently
        abs_values = np.abs(explore_values)
        if np.sum(abs_values) > 0:
            importance_weights = abs_values / np.sum(abs_values)
        else:
            importance_weights = np.ones(n_explore) / n_explore
        # Generate importance samples
        n_importance = n_samples - n_explore
        importance_indices = self.rng.choice(n_explore, size=n_importance, p=importance_weights)
        importance_samples = explore_samples[importance_indices]
        # Add some noise to importance samples
        noise_scale = 0.1 * np.array([b - a for a, b in bounds])
        noise = self.rng.normal(0, noise_scale, (n_importance, dimension))
        importance_samples += noise
        # Clip to bounds
        for i, (a, b) in enumerate(bounds):
            importance_samples[:, i] = np.clip(importance_samples[:, i], a, b)
        # Evaluate function at importance samples
        if dimension == 1:
            importance_values = np.array([func(sample[0]) for sample in importance_samples])
        else:
            importance_values = np.array([func(sample) for sample in importance_samples])
        # Combine all samples
        all_values = np.concatenate([explore_values, importance_values])
        # Compute integral estimate
        integral_estimate = volume * np.mean(all_values)
        # Compute variance and error
        variance = np.var(all_values, ddof=1)
        error = volume * np.sqrt(variance / n_samples)
        # Convergence history
        convergence_history = self._compute_convergence_history(all_values, volume)
        return IntegrationResult(
            value=integral_estimate,
            error=error,
            variance=variance,
            n_samples=n_samples,
            convergence_history=convergence_history,
            efficiency=1.0 / variance if variance > 0 else np.inf,
            metadata={'method': 'adaptive_importance', 'dimension': dimension}
        )
    def _stratified_sampling_integration(self, func: Callable, bounds, n_samples: int, **kwargs):
        """Integration using stratified sampling."""
        # Handle bounds format
        if isinstance(bounds[0], (int, float)):
            bounds = [bounds]
        dimension = len(bounds)
        # Determine number of strata per dimension
        strata_per_dim = max(2, int(np.power(n_samples, 1.0/dimension) / 4))
        total_strata = strata_per_dim ** dimension
        samples_per_stratum = n_samples // total_strata
        all_values = []
        total_volume = 0
        # Generate stratified samples
        for stratum_idx in range(total_strata):
            # Determine stratum bounds
            stratum_bounds = []
            idx = stratum_idx
            for i, (a, b) in enumerate(bounds):
                stratum_i = idx % strata_per_dim
                idx //= strata_per_dim
                stratum_width = (b - a) / strata_per_dim
                stratum_a = a + stratum_i * stratum_width
                stratum_b = a + (stratum_i + 1) * stratum_width
                stratum_bounds.append((stratum_a, stratum_b))
            # Generate samples within stratum
            stratum_samples = np.zeros((samples_per_stratum, dimension))
            stratum_volume = 1.0
            for i, (sa, sb) in enumerate(stratum_bounds):
                stratum_samples[:, i] = self.rng.uniform(sa, sb, samples_per_stratum)
                stratum_volume *= (sb - sa)
            # Evaluate function in stratum
            if dimension == 1:
                stratum_values = np.array([func(sample[0]) for sample in stratum_samples])
            else:
                stratum_values = np.array([func(sample) for sample in stratum_samples])
            # Weight by stratum volume
            weighted_values = stratum_values * stratum_volume
            all_values.extend(weighted_values)
            total_volume += stratum_volume
        all_values = np.array(all_values)
        # Compute integral estimate
        integral_estimate = np.sum(all_values) / len(all_values)
        # Compute variance and error
        variance = np.var(all_values, ddof=1)
        error = np.sqrt(variance / len(all_values))
        # Convergence history
        convergence_history = self._compute_convergence_history(all_values, 1.0)
        return IntegrationResult(
            value=integral_estimate,
            error=error,
            variance=variance,
            n_samples=len(all_values),
            convergence_history=convergence_history,
            efficiency=1.0 / variance if variance > 0 else np.inf,
            metadata={'method': 'stratified', 'strata_per_dim': strata_per_dim, 'dimension': dimension}
        )
    def _compute_convergence_history(self, function_values: np.ndarray,
                                   volume: float) -> List[float]:
        """Compute convergence history."""
        n_points = 20
        sample_sizes = np.logspace(2, np.log10(len(function_values)), n_points, dtype=int)
        sample_sizes = np.unique(sample_sizes)
        history = []
        for n in sample_sizes:
            if n <= len(function_values):
                partial_estimate = volume * np.mean(function_values[:n])
                history.append(partial_estimate)
        return history
# Convenience functions
def monte_carlo_integrate(func: Callable,
                         bounds: Union[Tuple[float, float], List[Tuple[float, float]]],
                         n_samples: int = 10000,
                         random_state: Optional[int] = None,
                         **kwargs) -> IntegrationResult:
    """Convenience function for Monte Carlo integration.
    Args:
        func: Function to integrate
        bounds: Integration bounds
        n_samples: Number of samples
        random_state: Random seed
        **kwargs: Additional arguments
    Returns:
        IntegrationResult: Integration results
    """
    integrator = MonteCarloIntegrator(random_state=random_state)
    return integrator.integrate(func, bounds, n_samples, **kwargs)
def quasi_monte_carlo_integrate(func: Callable,
                               bounds: Union[Tuple[float, float], List[Tuple[float, float]]],
                               n_samples: int = 10000,
                               sequence_type: str = 'sobol',
                               random_state: Optional[int] = None,
                               **kwargs) -> IntegrationResult:
    """Convenience function for quasi-Monte Carlo integration.
    Args:
        func: Function to integrate
        bounds: Integration bounds
        n_samples: Number of samples
        sequence_type: Low-discrepancy sequence type
        random_state: Random seed
        **kwargs: Additional arguments
    Returns:
        IntegrationResult: Integration results
    """
    integrator = QuasiMonteCarloIntegrator(random_state=random_state)
    return integrator.integrate(func, bounds, n_samples, sequence_type=sequence_type, **kwargs)
def importance_sampling_integrate(func: Callable,
                                 bounds: Union[Tuple[float, float], List[Tuple[float, float]]],
                                 n_samples: int = 10000,
                                 random_state: Optional[int] = None,
                                 **kwargs) -> IntegrationResult:
    """Convenience function for importance sampling integration.
    Args:
        func: Function to integrate
        bounds: Integration bounds
        n_samples: Number of samples
        random_state: Random seed
        **kwargs: Additional arguments
    Returns:
        IntegrationResult: Integration results
    """
    integrator = AdaptiveMonteCarloIntegrator(random_state=random_state)
    return integrator.integrate(func, bounds, n_samples, adaptation_strategy='importance', **kwargs)