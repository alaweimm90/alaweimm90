"""Monte Carlo Methods for Scientific Computing.
This package implements comprehensive Monte Carlo simulation methods
for scientific computing applications including integration, sampling,
optimization, and uncertainty quantification.
Modules:
    integration: Monte Carlo integration methods
    sampling: Advanced sampling techniques (MCMC, importance sampling)
    optimization: Monte Carlo optimization algorithms
    uncertainty: Uncertainty quantification and propagation
    visualization: Berkeley-themed plotting utilities
    utils: Utility functions and random number generators
    constants: Physical constants and distributions
Examples:
    >>> from Monte_Carlo import integration, sampling
    >>> result = integration.monte_carlo_integrate(lambda x: x**2, 0, 1, n_samples=10000)
    >>> samples = sampling.metropolis_hastings(target_pdf, initial_state, n_samples=5000)
Author: Berkeley SciComp Team
Date: 2024
"""
from .integration import (
    MonteCarloIntegrator,
    QuasiMonteCarloIntegrator,
    AdaptiveMonteCarloIntegrator,
    monte_carlo_integrate,
    quasi_monte_carlo_integrate,
    importance_sampling_integrate
)
from .sampling import (
    MetropolisHastings,
    HamiltonianMonteCarlo,
    ImportanceSampler,
    RejectionSampler,
    GibbsSampler,
    metropolis_hastings,
    hamiltonian_monte_carlo,
    importance_sampling,
    rejection_sampling
)
from .optimization import (
    SimulatedAnnealing,
    GeneticAlgorithm,
    ParticleSwarmOptimization,
    CrossEntropyMethod,
    simulated_annealing,
    genetic_algorithm,
    particle_swarm,
    cross_entropy_method
)
from .uncertainty import (
    UncertaintyQuantifier,
    SensitivityAnalyzer,
    PolynomialChaos,
    monte_carlo_uncertainty,
    sensitivity_analysis,
    polynomial_chaos_expansion
)
from .visualization import (
    MonteCarloVisualizer,
    plot_convergence,
    plot_samples,
    plot_distributions,
    plot_sensitivity
)
from .utils import (
    RandomNumberGenerator,
    StatisticalTests,
    set_seed,
    compute_statistics,
    effective_sample_size,
    autocorrelation
)
from .constants import (
    PHYSICAL_CONSTANTS,
    MATHEMATICAL_CONSTANTS,
    get_distribution,
    standard_distributions
)
__all__ = [
    # Integration
    'MonteCarloIntegrator',
    'QuasiMonteCarloIntegrator',
    'AdaptiveMonteCarloIntegrator',
    'monte_carlo_integrate',
    'quasi_monte_carlo_integrate',
    'importance_sampling_integrate',
    # Sampling
    'MetropolisHastings',
    'HamiltonianMonteCarlo',
    'ImportanceSampler',
    'RejectionSampler',
    'GibbsSampler',
    'metropolis_hastings',
    'hamiltonian_monte_carlo',
    'importance_sampling',
    'rejection_sampling',
    # Optimization
    'SimulatedAnnealing',
    'GeneticAlgorithm',
    'ParticleSwarmOptimization',
    'CrossEntropyMethod',
    'simulated_annealing',
    'genetic_algorithm',
    'particle_swarm',
    'cross_entropy_method',
    # Uncertainty
    'UncertaintyQuantifier',
    'SensitivityAnalyzer',
    'PolynomialChaos',
    'monte_carlo_uncertainty',
    'sensitivity_analysis',
    'polynomial_chaos_expansion',
    # Visualization
    'MonteCarloVisualizer',
    'plot_convergence',
    'plot_samples',
    'plot_distributions',
    'plot_sensitivity',
    # Utils
    'RandomNumberGenerator',
    'StatisticalTests',
    'set_seed',
    'compute_statistics',
    'effective_sample_size',
    'autocorrelation',
    # Constants
    'PHYSICAL_CONSTANTS',
    'MATHEMATICAL_CONSTANTS',
    'get_distribution',
    'standard_distributions'
]
__version__ = '1.0.0'
__author__ = 'Berkeley SciComp Team'
__email__ = 'scicomp@berkeley.edu'