# integration
**Module:** `Python/Monte_Carlo/integration.py`
## Overview
Monte Carlo Integration Methods.
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
## Functions
### `monte_carlo_integrate(func, bounds, n_samples, random_state)`
Convenience function for Monte Carlo integration.
Args:
func: Function to integrate
bounds: Integration bounds
n_samples: Number of samples
random_state: Random seed
**kwargs: Additional arguments
Returns:
IntegrationResult: Integration results
**Source:** [Line 476](Python/Monte_Carlo/integration.py#L476)
### `quasi_monte_carlo_integrate(func, bounds, n_samples, sequence_type, random_state)`
Convenience function for quasi-Monte Carlo integration.
Args:
func: Function to integrate
bounds: Integration bounds
n_samples: Number of samples
sequence_type: Low-discrepancy sequence type
random_state: Random seed
**kwargs: Additional arguments
Returns:
IntegrationResult: Integration results
**Source:** [Line 497](Python/Monte_Carlo/integration.py#L497)
### `importance_sampling_integrate(func, bounds, n_samples, random_state)`
Convenience function for importance sampling integration.
Args:
func: Function to integrate
bounds: Integration bounds
n_samples: Number of samples
random_state: Random seed
**kwargs: Additional arguments
Returns:
IntegrationResult: Integration results
**Source:** [Line 520](Python/Monte_Carlo/integration.py#L520)
## Classes
### `IntegrationResult`
Result of Monte Carlo integration.
Attributes:
value: Estimated integral value
error: Estimated error
variance: Sample variance
n_samples: Number of samples used
convergence_history: History of estimates
efficiency: Computational efficiency metric
**Class Source:** [Line 31](Python/Monte_Carlo/integration.py#L31)
### `MonteCarloIntegrator`
Standard Monte Carlo integrator.
Implements basic Monte Carlo integration with variance estimation
and convergence monitoring.
Parameters:
random_state: Random seed for reproducibility
verbose: Whether to print progress information
#### Methods
##### `__init__(self, random_state, verbose)`
*No documentation available.*
**Source:** [Line 62](Python/Monte_Carlo/integration.py#L62)
##### `integrate(self, func, bounds, n_samples)`
Integrate function using Monte Carlo method.
Args:
func: Function to integrate
bounds: Integration bounds (1D or multi-dimensional)
n_samples: Number of random samples
**kwargs: Additional arguments
Returns:
IntegrationResult: Integration results
**Source:** [Line 67](Python/Monte_Carlo/integration.py#L67)
##### `_integrate_1d(self, func, bounds, n_samples)`
1D Monte Carlo integration.
**Source:** [Line 89](Python/Monte_Carlo/integration.py#L89)
##### `_integrate_nd(self, func, bounds, n_samples)`
Multi-dimensional Monte Carlo integration.
**Source:** [Line 125](Python/Monte_Carlo/integration.py#L125)
##### `_compute_convergence_history(self, function_values, volume)`
Compute convergence history.
**Source:** [Line 159](Python/Monte_Carlo/integration.py#L159)
**Class Source:** [Line 51](Python/Monte_Carlo/integration.py#L51)
### `QuasiMonteCarloIntegrator`
Quasi-Monte Carlo integrator using low-discrepancy sequences.
Implements QMC integration using Sobol sequences for better
convergence than standard Monte Carlo.
#### Methods
##### `__init__(self, random_state, verbose)`
*No documentation available.*
**Source:** [Line 182](Python/Monte_Carlo/integration.py#L182)
##### `integrate(self, func, bounds, n_samples, sequence_type)`
Integrate using quasi-Monte Carlo method.
Args:
func: Function to integrate
bounds: Integration bounds
n_samples: Number of samples
sequence_type: Type of low-discrepancy sequence
**kwargs: Additional arguments
Returns:
IntegrationResult: Integration results
**Source:** [Line 186](Python/Monte_Carlo/integration.py#L186)
##### `_compute_convergence_history(self, function_values, volume)`
Compute convergence history.
**Source:** [Line 262](Python/Monte_Carlo/integration.py#L262)
**Class Source:** [Line 175](Python/Monte_Carlo/integration.py#L175)
### `AdaptiveMonteCarloIntegrator`
Adaptive Monte Carlo integrator with variance reduction.
Implements adaptive importance sampling and stratified sampling
for improved efficiency.
#### Methods
##### `__init__(self, random_state, verbose)`
*No documentation available.*
**Source:** [Line 285](Python/Monte_Carlo/integration.py#L285)
##### `integrate(self, func, bounds, n_samples, adaptation_strategy)`
Adaptive Monte Carlo integration.
Args:
func: Function to integrate
bounds: Integration bounds
n_samples: Number of samples
adaptation_strategy: Adaptation strategy ('importance', 'stratified')
**kwargs: Additional arguments
Returns:
IntegrationResult: Integration results
**Source:** [Line 290](Python/Monte_Carlo/integration.py#L290)
##### `_importance_sampling_integration(self, func, bounds, n_samples)`
Integration using importance sampling.
**Source:** [Line 315](Python/Monte_Carlo/integration.py#L315)
##### `_stratified_sampling_integration(self, func, bounds, n_samples)`
Integration using stratified sampling.
**Source:** [Line 387](Python/Monte_Carlo/integration.py#L387)
##### `_compute_convergence_history(self, function_values, volume)`
Compute convergence history.
**Source:** [Line 459](Python/Monte_Carlo/integration.py#L459)
**Class Source:** [Line 278](Python/Monte_Carlo/integration.py#L278)