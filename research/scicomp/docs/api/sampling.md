# sampling
**Module:** `Python/Monte_Carlo/sampling.py`
## Overview
Monte Carlo Sampling Methods.
This module implements various Monte Carlo sampling techniques
including MCMC methods, importance sampling, and rejection sampling.
Classes:
MetropolisHastings: Metropolis-Hastings MCMC sampler
HamiltonianMonteCarlo: Hamiltonian Monte Carlo sampler
ImportanceSampler: Importance sampling
RejectionSampler: Rejection sampling
GibbsSampler: Gibbs sampling for multivariate distributions
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `metropolis_hastings(log_prob_func, initial_state, n_samples, burn_in, proposal_cov, random_state)`
Convenience function for Metropolis-Hastings sampling.
**Source:** [Line 578](Python/Monte_Carlo/sampling.py#L578)
### `hamiltonian_monte_carlo(log_prob_func, grad_log_prob_func, initial_state, n_samples, burn_in, step_size, n_leapfrog, random_state)`
Convenience function for Hamiltonian Monte Carlo sampling.
**Source:** [Line 590](Python/Monte_Carlo/sampling.py#L590)
### `importance_sampling(target_log_prob, proposal_sampler, proposal_log_prob, n_samples, random_state)`
Convenience function for importance sampling.
**Source:** [Line 605](Python/Monte_Carlo/sampling.py#L605)
### `rejection_sampling(target_func, proposal_sampler, envelope_constant, n_samples, random_state)`
Convenience function for rejection sampling.
**Source:** [Line 616](Python/Monte_Carlo/sampling.py#L616)
## Classes
### `SamplingResult`
Result of Monte Carlo sampling.
Attributes:
samples: Generated samples
log_probabilities: Log probabilities of samples
acceptance_rate: Acceptance rate for MCMC methods
effective_sample_size: Effective sample size
autocorrelation_time: Autocorrelation time
diagnostics: Additional diagnostic information
**Class Source:** [Line 27](Python/Monte_Carlo/sampling.py#L27)
### `MetropolisHastings`
Metropolis-Hastings MCMC sampler.
Implements the Metropolis-Hastings algorithm for sampling from
arbitrary probability distributions.
Parameters:
log_prob_func: Log probability function
proposal_cov: Proposal covariance matrix
random_state: Random seed for reproducibility
verbose: Whether to print progress information
#### Methods
##### `__init__(self, log_prob_func, proposal_cov, random_state, verbose)`
*No documentation available.*
**Source:** [Line 59](Python/Monte_Carlo/sampling.py#L59)
##### `sample(self, initial_state, n_samples, burn_in, thin, adapt_proposal)`
Generate samples using Metropolis-Hastings.
Args:
initial_state: Initial state of the chain
n_samples: Number of samples to generate
burn_in: Number of burn-in samples
thin: Thinning factor
adapt_proposal: Whether to adapt proposal distribution
Returns:
SamplingResult: Sampling results
**Source:** [Line 75](Python/Monte_Carlo/sampling.py#L75)
##### `_adapt_proposal(self, samples, accepted)`
Adapt proposal covariance matrix.
**Source:** [Line 168](Python/Monte_Carlo/sampling.py#L168)
**Class Source:** [Line 46](Python/Monte_Carlo/sampling.py#L46)
### `HamiltonianMonteCarlo`
Hamiltonian Monte Carlo sampler.
Implements HMC using Hamiltonian dynamics for efficient sampling
from high-dimensional distributions.
#### Methods
##### `__init__(self, log_prob_func, grad_log_prob_func, step_size, n_leapfrog, random_state, verbose)`
*No documentation available.*
**Source:** [Line 195](Python/Monte_Carlo/sampling.py#L195)
##### `sample(self, initial_state, n_samples, burn_in, thin)`
Generate samples using Hamiltonian Monte Carlo.
Args:
initial_state: Initial state of the chain
n_samples: Number of samples to generate
burn_in: Number of burn-in samples
thin: Thinning factor
Returns:
SamplingResult: Sampling results
**Source:** [Line 210](Python/Monte_Carlo/sampling.py#L210)
##### `_leapfrog(self, q, p)`
Leapfrog integration step.
**Source:** [Line 305](Python/Monte_Carlo/sampling.py#L305)
**Class Source:** [Line 188](Python/Monte_Carlo/sampling.py#L188)
### `ImportanceSampler`
Importance sampling for estimating expectations.
Uses importance sampling to estimate expectations of functions
under target distributions.
#### Methods
##### `__init__(self, target_log_prob, proposal_sampler, proposal_log_prob, random_state)`
*No documentation available.*
**Source:** [Line 337](Python/Monte_Carlo/sampling.py#L337)
##### `sample(self, n_samples)`
Generate importance samples.
Args:
n_samples: Number of samples to generate
Returns:
SamplingResult: Sampling results with importance weights
**Source:** [Line 348](Python/Monte_Carlo/sampling.py#L348)
##### `estimate_expectation(self, func, n_samples)`
Estimate expectation of function under target distribution.
Args:
func: Function to compute expectation of
n_samples: Number of samples to use
Returns:
Tuple of (estimate, standard_error)
**Source:** [Line 392](Python/Monte_Carlo/sampling.py#L392)
**Class Source:** [Line 330](Python/Monte_Carlo/sampling.py#L330)
### `RejectionSampler`
Rejection sampling for generating samples from arbitrary distributions.
Uses rejection sampling to generate samples from distributions
that can be evaluated up to a normalization constant.
#### Methods
##### `__init__(self, target_func, proposal_sampler, envelope_constant, random_state)`
*No documentation available.*
**Source:** [Line 425](Python/Monte_Carlo/sampling.py#L425)
##### `sample(self, n_samples, max_iterations)`
Generate samples using rejection sampling.
Args:
n_samples: Number of samples to generate
max_iterations: Maximum number of proposals to try
Returns:
SamplingResult: Sampling results
**Source:** [Line 436](Python/Monte_Carlo/sampling.py#L436)
**Class Source:** [Line 418](Python/Monte_Carlo/sampling.py#L418)
### `GibbsSampler`
Gibbs sampler for multivariate distributions.
Implements Gibbs sampling for multivariate distributions where
conditional distributions are known.
#### Methods
##### `__init__(self, conditional_samplers, random_state, verbose)`
*No documentation available.*
**Source:** [Line 501](Python/Monte_Carlo/sampling.py#L501)
##### `sample(self, initial_state, n_samples, burn_in, thin)`
Generate samples using Gibbs sampling.
Args:
initial_state: Initial state of the chain
n_samples: Number of samples to generate
burn_in: Number of burn-in samples
thin: Thinning factor
Returns:
SamplingResult: Sampling results
**Source:** [Line 510](Python/Monte_Carlo/sampling.py#L510)
**Class Source:** [Line 494](Python/Monte_Carlo/sampling.py#L494)