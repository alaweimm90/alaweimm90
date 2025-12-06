"""Physical Constants and Standard Distributions for Monte Carlo Methods.
This module provides physical constants, mathematical constants,
and standard probability distributions commonly used in Monte Carlo simulations.
Constants:
    PHYSICAL_CONSTANTS: Physical constants with 2018 CODATA values
    MATHEMATICAL_CONSTANTS: Mathematical constants (π, e, etc.)
Functions:
    get_distribution: Get standard probability distribution
    standard_distributions: Dictionary of standard distributions
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
from typing import Dict, Any, Callable, Optional, Union
from scipy import stats
import warnings
# Physical Constants (2018 CODATA values)
PHYSICAL_CONSTANTS = {
    # Fundamental constants
    'c': 299792458.0,  # Speed of light in vacuum (m/s)
    'h': 6.62607015e-34,  # Planck constant (J⋅s)
    'hbar': 1.054571817e-34,  # Reduced Planck constant (J⋅s)
    'e': 1.602176634e-19,  # Elementary charge (C)
    'k_B': 1.380649e-23,  # Boltzmann constant (J/K)
    'N_A': 6.02214076e23,  # Avogadro constant (mol⁻¹)
    'R': 8.314462618,  # Gas constant (J/(mol⋅K))
    'sigma_SB': 5.670374419e-8,  # Stefan-Boltzmann constant (W/(m²⋅K⁴))
    # Electromagnetic constants
    'mu_0': 4e-7 * np.pi,  # Magnetic permeability of vacuum (H/m)
    'epsilon_0': 8.8541878128e-12,  # Electric permittivity of vacuum (F/m)
    'Z_0': 376.730313668,  # Impedance of vacuum (Ω)
    'alpha': 7.2973525693e-3,  # Fine structure constant
    # Particle masses (kg)
    'm_e': 9.1093837015e-31,  # Electron mass
    'm_p': 1.67262192369e-27,  # Proton mass
    'm_n': 1.67492749804e-27,  # Neutron mass
    'm_u': 1.66053906660e-27,  # Atomic mass unit
    # Other useful constants
    'g': 9.80665,  # Standard gravity (m/s²)
    'atm': 101325.0,  # Standard atmosphere (Pa)
    'cal': 4.184,  # Thermochemical calorie (J)
    'eV': 1.602176634e-19,  # Electron volt (J)
}
# Mathematical Constants
MATHEMATICAL_CONSTANTS = {
    'pi': np.pi,
    'e': np.e,
    'golden_ratio': (1 + np.sqrt(5)) / 2,
    'euler_gamma': 0.5772156649015329,
    'sqrt_2': np.sqrt(2),
    'sqrt_pi': np.sqrt(np.pi),
    'ln_2': np.log(2),
    'ln_10': np.log(10),
    'log10_e': np.log10(np.e),
}
# Standard Probability Distributions
def get_distribution(name: str, **params) -> stats.rv_continuous:
    """Get a standard probability distribution.
    Args:
        name: Distribution name
        **params: Distribution parameters
    Returns:
        Scipy distribution object
    Examples:
        >>> normal = get_distribution('normal', loc=0, scale=1)
        >>> uniform = get_distribution('uniform', loc=-1, scale=2)
    """
    distributions = {
        'normal': stats.norm,
        'uniform': stats.uniform,
        'exponential': stats.expon,
        'gamma': stats.gamma,
        'beta': stats.beta,
        'lognormal': stats.lognorm,
        'weibull': stats.weibull_min,
        'pareto': stats.pareto,
        'chi2': stats.chi2,
        't': stats.t,
        'f': stats.f,
        'cauchy': stats.cauchy,
        'laplace': stats.laplace,
        'logistic': stats.logistic,
        'rayleigh': stats.rayleigh,
        'maxwell': stats.maxwell,
        'rice': stats.rice,
        'poisson': stats.poisson,
        'binomial': stats.binom,
        'geometric': stats.geom,
        'negative_binomial': stats.nbinom,
        'hypergeometric': stats.hypergeom,
        'multinomial': stats.multinomial,
        'multivariate_normal': stats.multivariate_normal,
        'dirichlet': stats.dirichlet,
        'wishart': stats.wishart,
        'inverse_wishart': stats.invwishart
    }
    if name not in distributions:
        available = list(distributions.keys())
        raise ValueError(f"Unknown distribution '{name}'. Available: {available}")
    try:
        return distributions[name](**params)
    except Exception as e:
        raise ValueError(f"Failed to create {name} distribution with params {params}: {e}")
# Predefined standard distributions
standard_distributions = {
    'standard_normal': lambda: get_distribution('normal', loc=0, scale=1),
    'unit_uniform': lambda: get_distribution('uniform', loc=0, scale=1),
    'standard_exponential': lambda: get_distribution('exponential', scale=1),
    'standard_gamma': lambda: get_distribution('gamma', a=1, scale=1),
    'standard_cauchy': lambda: get_distribution('cauchy', loc=0, scale=1),
    'standard_laplace': lambda: get_distribution('laplace', loc=0, scale=1),
    'unit_circle_uniform': lambda: get_distribution('uniform', loc=0, scale=2*np.pi),
    'bernoulli_half': lambda: get_distribution('binomial', n=1, p=0.5),
    'standard_chi2': lambda: get_distribution('chi2', df=1),
    'student_t': lambda df=1: get_distribution('t', df=df),
}
class DistributionSampler:
    """Advanced distribution sampler with caching and validation.
    Provides efficient sampling from various distributions with
    automatic parameter validation and result caching.
    """
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self._distribution_cache = {}
    def sample(self,
               distribution: Union[str, stats.rv_continuous, Callable],
               size: Optional[Union[int, tuple]] = None,
               **params) -> np.ndarray:
        """Sample from a distribution.
        Args:
            distribution: Distribution name, scipy distribution, or callable
            size: Sample size
            **params: Distribution parameters
        Returns:
            Generated samples
        """
        if isinstance(distribution, str):
            dist = self._get_cached_distribution(distribution, **params)
            return dist.rvs(size=size, random_state=self.rng)
        elif hasattr(distribution, 'rvs'):
            return distribution.rvs(size=size, random_state=self.rng)
        elif callable(distribution):
            return distribution(size)
        else:
            raise ValueError("Invalid distribution type")
    def _get_cached_distribution(self, name: str, **params) -> stats.rv_continuous:
        """Get distribution with caching."""
        cache_key = (name, tuple(sorted(params.items())))
        if cache_key not in self._distribution_cache:
            self._distribution_cache[cache_key] = get_distribution(name, **params)
        return self._distribution_cache[cache_key]
    def pdf(self,
            distribution: Union[str, stats.rv_continuous],
            x: np.ndarray,
            **params) -> np.ndarray:
        """Evaluate probability density function."""
        if isinstance(distribution, str):
            dist = self._get_cached_distribution(distribution, **params)
            return dist.pdf(x)
        elif hasattr(distribution, 'pdf'):
            return distribution.pdf(x)
        else:
            raise ValueError("Invalid distribution type")
    def logpdf(self,
               distribution: Union[str, stats.rv_continuous],
               x: np.ndarray,
               **params) -> np.ndarray:
        """Evaluate log probability density function."""
        if isinstance(distribution, str):
            dist = self._get_cached_distribution(distribution, **params)
            return dist.logpdf(x)
        elif hasattr(distribution, 'logpdf'):
            return distribution.logpdf(x)
        else:
            raise ValueError("Invalid distribution type")
    def cdf(self,
            distribution: Union[str, stats.rv_continuous],
            x: np.ndarray,
            **params) -> np.ndarray:
        """Evaluate cumulative distribution function."""
        if isinstance(distribution, str):
            dist = self._get_cached_distribution(distribution, **params)
            return dist.cdf(x)
        elif hasattr(distribution, 'cdf'):
            return distribution.cdf(x)
        else:
            raise ValueError("Invalid distribution type")
# Physics-specific distributions
def maxwell_boltzmann_3d(temperature: float, mass: float) -> Callable:
    """Maxwell-Boltzmann distribution for 3D velocities.
    Args:
        temperature: Temperature (K)
        mass: Particle mass (kg)
    Returns:
        Sampling function
    """
    k_B = PHYSICAL_CONSTANTS['k_B']
    sigma = np.sqrt(k_B * temperature / mass)
    def sample(size=None):
        if size is None:
            return np.random.normal(0, sigma, 3)
        elif isinstance(size, int):
            return np.random.normal(0, sigma, (size, 3))
        else:
            return np.random.normal(0, sigma, size + (3,))
    return sample
def planck_distribution(temperature: float) -> stats.rv_continuous:
    """Planck distribution for blackbody radiation.
    Args:
        temperature: Temperature (K)
    Returns:
        Planck distribution object
    """
    h = PHYSICAL_CONSTANTS['h']
    c = PHYSICAL_CONSTANTS['c']
    k_B = PHYSICAL_CONSTANTS['k_B']
    # Custom Planck distribution (simplified implementation)
    class PlanckDistribution(stats.rv_continuous):
        def _pdf(self, f):
            # Frequency-based Planck distribution
            hf_kT = h * f / (k_B * temperature)
            return (8 * np.pi * h * f**3 / c**3) / (np.exp(hf_kT) - 1)
    return PlanckDistribution(a=0, name='planck')
def fermi_dirac_distribution(temperature: float, chemical_potential: float) -> Callable:
    """Fermi-Dirac distribution.
    Args:
        temperature: Temperature (K)
        chemical_potential: Chemical potential (J)
    Returns:
        Fermi-Dirac probability function
    """
    k_B = PHYSICAL_CONSTANTS['k_B']
    def probability(energy):
        return 1 / (1 + np.exp((energy - chemical_potential) / (k_B * temperature)))
    return probability
def bose_einstein_distribution(temperature: float, chemical_potential: float) -> Callable:
    """Bose-Einstein distribution.
    Args:
        temperature: Temperature (K)
        chemical_potential: Chemical potential (J)
    Returns:
        Bose-Einstein occupation function
    """
    k_B = PHYSICAL_CONSTANTS['k_B']
    def occupation(energy):
        return 1 / (np.exp((energy - chemical_potential) / (k_B * temperature)) - 1)
    return occupation
# Convenience functions for common use cases
def create_multivariate_normal(mean: np.ndarray,
                               cov: np.ndarray,
                               random_state: Optional[int] = None) -> Callable:
    """Create multivariate normal sampler.
    Args:
        mean: Mean vector
        cov: Covariance matrix
        random_state: Random seed
    Returns:
        Sampling function
    """
    rng = np.random.RandomState(random_state)
    def sample(size=None):
        return rng.multivariate_normal(mean, cov, size)
    return sample
def create_mixture_distribution(components: list,
                                weights: Optional[np.ndarray] = None,
                                random_state: Optional[int] = None) -> Callable:
    """Create mixture distribution sampler.
    Args:
        components: List of (distribution, params) tuples
        weights: Component weights
        random_state: Random seed
    Returns:
        Sampling function
    """
    if weights is None:
        weights = np.ones(len(components)) / len(components)
    else:
        weights = np.asarray(weights)
        weights = weights / np.sum(weights)
    rng = np.random.RandomState(random_state)
    sampler = DistributionSampler(random_state)
    def sample(size=None):
        if size is None:
            size = 1
            single_sample = True
        else:
            single_sample = False
        # Choose components
        component_indices = rng.choice(len(components), size=size, p=weights)
        # Sample from chosen components
        samples = []
        for i in range(size):
            comp_idx = component_indices[i]
            dist_name, params = components[comp_idx]
            sample_val = sampler.sample(dist_name, size=None, **params)
            samples.append(sample_val)
        result = np.array(samples)
        return result[0] if single_sample else result
    return sample
# Unit conversion utilities
UNIT_CONVERSIONS = {
    'energy': {
        'J_to_eV': 1 / PHYSICAL_CONSTANTS['eV'],
        'eV_to_J': PHYSICAL_CONSTANTS['eV'],
        'J_to_cal': 1 / PHYSICAL_CONSTANTS['cal'],
        'cal_to_J': PHYSICAL_CONSTANTS['cal'],
        'J_to_erg': 1e7,
        'erg_to_J': 1e-7
    },
    'temperature': {
        'K_to_C': lambda T: T - 273.15,
        'C_to_K': lambda T: T + 273.15,
        'K_to_F': lambda T: (T - 273.15) * 9/5 + 32,
        'F_to_K': lambda T: (T - 32) * 5/9 + 273.15
    },
    'pressure': {
        'Pa_to_atm': 1 / PHYSICAL_CONSTANTS['atm'],
        'atm_to_Pa': PHYSICAL_CONSTANTS['atm'],
        'Pa_to_bar': 1e-5,
        'bar_to_Pa': 1e5,
        'Pa_to_mmHg': 760 / PHYSICAL_CONSTANTS['atm'],
        'mmHg_to_Pa': PHYSICAL_CONSTANTS['atm'] / 760
    }
}
def convert_units(value: float, from_unit: str, to_unit: str,
                  unit_type: str) -> float:
    """Convert between units.
    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit
        unit_type: Type of unit (energy, temperature, pressure)
    Returns:
        Converted value
    """
    if unit_type not in UNIT_CONVERSIONS:
        raise ValueError(f"Unknown unit type: {unit_type}")
    conversions = UNIT_CONVERSIONS[unit_type]
    conversion_key = f"{from_unit}_to_{to_unit}"
    if conversion_key in conversions:
        factor_or_func = conversions[conversion_key]
        if callable(factor_or_func):
            return factor_or_func(value)
        else:
            return value * factor_or_func
    else:
        available = list(conversions.keys())
        raise ValueError(f"Unknown conversion {conversion_key}. Available: {available}")