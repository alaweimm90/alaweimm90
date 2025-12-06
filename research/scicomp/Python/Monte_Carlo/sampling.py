"""Monte Carlo Sampling Methods.
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
"""
import numpy as np
from typing import Callable, Union, Tuple, List, Optional, Dict, Any
from scipy import stats, optimize
from dataclasses import dataclass
import warnings
from .utils import set_seed, compute_statistics, effective_sample_size, autocorrelation
@dataclass
class SamplingResult:
    """Result of Monte Carlo sampling.
    Attributes:
        samples: Generated samples
        log_probabilities: Log probabilities of samples
        acceptance_rate: Acceptance rate for MCMC methods
        effective_sample_size: Effective sample size
        autocorrelation_time: Autocorrelation time
        diagnostics: Additional diagnostic information
    """
    samples: np.ndarray
    log_probabilities: np.ndarray
    acceptance_rate: float
    effective_sample_size: float
    autocorrelation_time: float
    diagnostics: Dict[str, Any]
class MetropolisHastings:
    """Metropolis-Hastings MCMC sampler.
    Implements the Metropolis-Hastings algorithm for sampling from
    arbitrary probability distributions.
    Parameters:
        log_prob_func: Log probability function
        proposal_cov: Proposal covariance matrix
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information
    """
    def __init__(self,
                 log_prob_func: Callable,
                 proposal_cov: Optional[np.ndarray] = None,
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        self.log_prob_func = log_prob_func
        self.proposal_cov = proposal_cov
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.RandomState(random_state)
        # Adaptive parameters
        self.adaptive = True
        self.target_acceptance = 0.44  # Optimal for high dimensions
        self.adaptation_window = 50
    def sample(self,
               initial_state: np.ndarray,
               n_samples: int,
               burn_in: int = 1000,
               thin: int = 1,
               adapt_proposal: bool = True) -> SamplingResult:
        """Generate samples using Metropolis-Hastings.
        Args:
            initial_state: Initial state of the chain
            n_samples: Number of samples to generate
            burn_in: Number of burn-in samples
            thin: Thinning factor
            adapt_proposal: Whether to adapt proposal distribution
        Returns:
            SamplingResult: Sampling results
        """
        initial_state = np.asarray(initial_state)
        dimension = len(initial_state)
        # Initialize proposal covariance
        if self.proposal_cov is None:
            self.proposal_cov = 0.1 * np.eye(dimension)
        # Total number of iterations
        total_iterations = burn_in + n_samples * thin
        # Storage
        all_samples = np.zeros((total_iterations, dimension))
        all_log_probs = np.zeros(total_iterations)
        accepted = np.zeros(total_iterations, dtype=bool)
        # Initialize chain
        current_state = initial_state.copy()
        current_log_prob = self.log_prob_func(current_state)
        # Sampling loop
        for i in range(total_iterations):
            # Propose new state
            proposal = self.rng.multivariate_normal(current_state, self.proposal_cov)
            proposal_log_prob = self.log_prob_func(proposal)
            # Metropolis-Hastings acceptance probability
            log_alpha = min(0, proposal_log_prob - current_log_prob)
            # Accept or reject
            if np.log(self.rng.random()) < log_alpha:
                current_state = proposal
                current_log_prob = proposal_log_prob
                accepted[i] = True
            # Store sample
            all_samples[i] = current_state
            all_log_probs[i] = current_log_prob
            # Adaptive proposal (during burn-in)
            if adapt_proposal and i < burn_in and i > 0 and i % self.adaptation_window == 0:
                self._adapt_proposal(all_samples[:i+1], accepted[:i+1])
            # Progress reporting
            if self.verbose and (i + 1) % 1000 == 0:
                current_acceptance = np.mean(accepted[max(0, i-999):i+1])
                print(f"Iteration {i+1}/{total_iterations}, "
                      f"Acceptance rate: {current_acceptance:.3f}")
        # Extract post-burn-in samples
        post_burnin_samples = all_samples[burn_in::thin]
        post_burnin_log_probs = all_log_probs[burn_in::thin]
        post_burnin_accepted = accepted[burn_in::thin]
        # Compute diagnostics
        acceptance_rate = np.mean(post_burnin_accepted)
        ess = effective_sample_size(post_burnin_samples)
        autocorr_time = autocorrelation(post_burnin_samples)
        diagnostics = {
            'total_iterations': total_iterations,
            'burn_in': burn_in,
            'thin': thin,
            'proposal_cov': self.proposal_cov.copy(),
            'gelman_rubin': None  # Would need multiple chains
        }
        return SamplingResult(
            samples=post_burnin_samples,
            log_probabilities=post_burnin_log_probs,
            acceptance_rate=acceptance_rate,
            effective_sample_size=ess,
            autocorrelation_time=autocorr_time,
            diagnostics=diagnostics
        )
    def _adapt_proposal(self, samples: np.ndarray, accepted: np.ndarray):
        """Adapt proposal covariance matrix."""
        recent_samples = samples[-self.adaptation_window:]
        recent_acceptance = np.mean(accepted[-self.adaptation_window:])
        # Update covariance based on sample covariance
        if len(recent_samples) > 1:
            sample_cov = np.cov(recent_samples.T)
            # Regularize to avoid singular matrices
            sample_cov += 1e-6 * np.eye(sample_cov.shape[0])
            # Scale based on acceptance rate
            if recent_acceptance > self.target_acceptance:
                scale_factor = 1.1
            else:
                scale_factor = 0.9
            self.proposal_cov = scale_factor * sample_cov
class HamiltonianMonteCarlo:
    """Hamiltonian Monte Carlo sampler.
    Implements HMC using Hamiltonian dynamics for efficient sampling
    from high-dimensional distributions.
    """
    def __init__(self,
                 log_prob_func: Callable,
                 grad_log_prob_func: Callable,
                 step_size: float = 0.1,
                 n_leapfrog: int = 10,
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        self.log_prob_func = log_prob_func
        self.grad_log_prob_func = grad_log_prob_func
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.RandomState(random_state)
    def sample(self,
               initial_state: np.ndarray,
               n_samples: int,
               burn_in: int = 1000,
               thin: int = 1) -> SamplingResult:
        """Generate samples using Hamiltonian Monte Carlo.
        Args:
            initial_state: Initial state of the chain
            n_samples: Number of samples to generate
            burn_in: Number of burn-in samples
            thin: Thinning factor
        Returns:
            SamplingResult: Sampling results
        """
        initial_state = np.asarray(initial_state)
        dimension = len(initial_state)
        # Total number of iterations
        total_iterations = burn_in + n_samples * thin
        # Storage
        all_samples = np.zeros((total_iterations, dimension))
        all_log_probs = np.zeros(total_iterations)
        accepted = np.zeros(total_iterations, dtype=bool)
        # Initialize chain
        current_q = initial_state.copy()
        current_log_prob = self.log_prob_func(current_q)
        # Sampling loop
        for i in range(total_iterations):
            # Sample momentum
            current_p = self.rng.normal(0, 1, dimension)
            # Store initial state
            q_initial = current_q.copy()
            p_initial = current_p.copy()
            # Leapfrog integration
            q, p = self._leapfrog(current_q, current_p)
            # Compute acceptance probability
            proposed_log_prob = self.log_prob_func(q)
            # Hamiltonian at initial and proposed states
            H_initial = -current_log_prob + 0.5 * np.sum(p_initial**2)
            H_proposed = -proposed_log_prob + 0.5 * np.sum(p**2)
            # Accept or reject
            log_alpha = min(0, H_initial - H_proposed)
            if np.log(self.rng.random()) < log_alpha:
                current_q = q
                current_log_prob = proposed_log_prob
                accepted[i] = True
            # Store sample
            all_samples[i] = current_q
            all_log_probs[i] = current_log_prob
            # Progress reporting
            if self.verbose and (i + 1) % 1000 == 0:
                current_acceptance = np.mean(accepted[max(0, i-999):i+1])
                print(f"Iteration {i+1}/{total_iterations}, "
                      f"Acceptance rate: {current_acceptance:.3f}")
        # Extract post-burn-in samples
        post_burnin_samples = all_samples[burn_in::thin]
        post_burnin_log_probs = all_log_probs[burn_in::thin]
        post_burnin_accepted = accepted[burn_in::thin]
        # Compute diagnostics
        acceptance_rate = np.mean(post_burnin_accepted)
        ess = effective_sample_size(post_burnin_samples)
        autocorr_time = autocorrelation(post_burnin_samples)
        diagnostics = {
            'total_iterations': total_iterations,
            'burn_in': burn_in,
            'thin': thin,
            'step_size': self.step_size,
            'n_leapfrog': self.n_leapfrog
        }
        return SamplingResult(
            samples=post_burnin_samples,
            log_probabilities=post_burnin_log_probs,
            acceptance_rate=acceptance_rate,
            effective_sample_size=ess,
            autocorrelation_time=autocorr_time,
            diagnostics=diagnostics
        )
    def _leapfrog(self, q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Leapfrog integration step."""
        q = q.copy()
        p = p.copy()
        # Half step for momentum
        p += 0.5 * self.step_size * self.grad_log_prob_func(q)
        # Full steps
        for _ in range(self.n_leapfrog):
            # Full step for position
            q += self.step_size * p
            # Full step for momentum (except last iteration)
            p += self.step_size * self.grad_log_prob_func(q)
        # Half step for momentum
        p -= 0.5 * self.step_size * self.grad_log_prob_func(q)
        # Negate momentum for detailed balance
        p = -p
        return q, p
class ImportanceSampler:
    """Importance sampling for estimating expectations.
    Uses importance sampling to estimate expectations of functions
    under target distributions.
    """
    def __init__(self,
                 target_log_prob: Callable,
                 proposal_sampler: Callable,
                 proposal_log_prob: Callable,
                 random_state: Optional[int] = None):
        self.target_log_prob = target_log_prob
        self.proposal_sampler = proposal_sampler
        self.proposal_log_prob = proposal_log_prob
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    def sample(self, n_samples: int) -> SamplingResult:
        """Generate importance samples.
        Args:
            n_samples: Number of samples to generate
        Returns:
            SamplingResult: Sampling results with importance weights
        """
        # Generate samples from proposal distribution
        samples = self.proposal_sampler(n_samples)
        # Compute log probabilities
        target_log_probs = np.array([self.target_log_prob(x) for x in samples])
        proposal_log_probs = np.array([self.proposal_log_prob(x) for x in samples])
        # Compute importance weights
        log_weights = target_log_probs - proposal_log_probs
        # Normalize weights
        max_log_weight = np.max(log_weights)
        weights = np.exp(log_weights - max_log_weight)
        normalized_weights = weights / np.sum(weights)
        # Effective sample size
        ess = 1.0 / np.sum(normalized_weights**2)
        diagnostics = {
            'weights': weights,
            'normalized_weights': normalized_weights,
            'max_weight': np.max(weights),
            'min_weight': np.min(weights),
            'weight_variance': np.var(weights)
        }
        return SamplingResult(
            samples=samples,
            log_probabilities=target_log_probs,
            acceptance_rate=1.0,  # All samples are accepted
            effective_sample_size=ess,
            autocorrelation_time=0.0,  # Independent samples
            diagnostics=diagnostics
        )
    def estimate_expectation(self, func: Callable, n_samples: int) -> Tuple[float, float]:
        """Estimate expectation of function under target distribution.
        Args:
            func: Function to compute expectation of
            n_samples: Number of samples to use
        Returns:
            Tuple of (estimate, standard_error)
        """
        result = self.sample(n_samples)
        # Evaluate function
        func_values = np.array([func(x) for x in result.samples])
        # Importance sampling estimate
        weights = result.diagnostics['normalized_weights']
        estimate = np.sum(weights * func_values)
        # Estimate standard error
        variance_estimate = np.sum(weights * (func_values - estimate)**2)
        standard_error = np.sqrt(variance_estimate / result.effective_sample_size)
        return estimate, standard_error
class RejectionSampler:
    """Rejection sampling for generating samples from arbitrary distributions.
    Uses rejection sampling to generate samples from distributions
    that can be evaluated up to a normalization constant.
    """
    def __init__(self,
                 target_func: Callable,
                 proposal_sampler: Callable,
                 envelope_constant: float,
                 random_state: Optional[int] = None):
        self.target_func = target_func
        self.proposal_sampler = proposal_sampler
        self.envelope_constant = envelope_constant
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    def sample(self, n_samples: int, max_iterations: int = None) -> SamplingResult:
        """Generate samples using rejection sampling.
        Args:
            n_samples: Number of samples to generate
            max_iterations: Maximum number of proposals to try
        Returns:
            SamplingResult: Sampling results
        """
        if max_iterations is None:
            max_iterations = n_samples * 100  # Conservative upper bound
        samples = []
        target_values = []
        n_accepted = 0
        n_proposed = 0
        while n_accepted < n_samples and n_proposed < max_iterations:
            # Generate proposal
            proposal = self.proposal_sampler(1)[0]
            n_proposed += 1
            # Evaluate target function
            target_value = self.target_func(proposal)
            # Accept or reject
            u = self.rng.random()
            if u * self.envelope_constant <= target_value:
                samples.append(proposal)
                target_values.append(target_value)
                n_accepted += 1
        if n_accepted < n_samples:
            warnings.warn(f"Only generated {n_accepted}/{n_samples} samples "
                         f"after {n_proposed} proposals")
        samples = np.array(samples)
        target_values = np.array(target_values)
        acceptance_rate = n_accepted / n_proposed if n_proposed > 0 else 0.0
        diagnostics = {
            'n_proposed': n_proposed,
            'envelope_constant': self.envelope_constant,
            'efficiency': acceptance_rate
        }
        return SamplingResult(
            samples=samples,
            log_probabilities=np.log(target_values),
            acceptance_rate=acceptance_rate,
            effective_sample_size=float(len(samples)),
            autocorrelation_time=0.0,  # Independent samples
            diagnostics=diagnostics
        )
class GibbsSampler:
    """Gibbs sampler for multivariate distributions.
    Implements Gibbs sampling for multivariate distributions where
    conditional distributions are known.
    """
    def __init__(self,
                 conditional_samplers: List[Callable],
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        self.conditional_samplers = conditional_samplers
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.RandomState(random_state)
    def sample(self,
               initial_state: np.ndarray,
               n_samples: int,
               burn_in: int = 1000,
               thin: int = 1) -> SamplingResult:
        """Generate samples using Gibbs sampling.
        Args:
            initial_state: Initial state of the chain
            n_samples: Number of samples to generate
            burn_in: Number of burn-in samples
            thin: Thinning factor
        Returns:
            SamplingResult: Sampling results
        """
        initial_state = np.asarray(initial_state)
        dimension = len(initial_state)
        if len(self.conditional_samplers) != dimension:
            raise ValueError("Number of conditional samplers must match dimension")
        # Total number of iterations
        total_iterations = burn_in + n_samples * thin
        # Storage
        all_samples = np.zeros((total_iterations, dimension))
        # Initialize chain
        current_state = initial_state.copy()
        # Sampling loop
        for i in range(total_iterations):
            # Update each component
            for j in range(dimension):
                current_state[j] = self.conditional_samplers[j](current_state)
            # Store sample
            all_samples[i] = current_state.copy()
            # Progress reporting
            if self.verbose and (i + 1) % 1000 == 0:
                print(f"Iteration {i+1}/{total_iterations}")
        # Extract post-burn-in samples
        post_burnin_samples = all_samples[burn_in::thin]
        # Compute diagnostics
        ess = effective_sample_size(post_burnin_samples)
        autocorr_time = autocorrelation(post_burnin_samples)
        diagnostics = {
            'total_iterations': total_iterations,
            'burn_in': burn_in,
            'thin': thin
        }
        return SamplingResult(
            samples=post_burnin_samples,
            log_probabilities=np.zeros(len(post_burnin_samples)),  # Not computed
            acceptance_rate=1.0,  # Always accept in Gibbs
            effective_sample_size=ess,
            autocorrelation_time=autocorr_time,
            diagnostics=diagnostics
        )
# Convenience functions
def metropolis_hastings(log_prob_func: Callable,
                       initial_state: np.ndarray,
                       n_samples: int,
                       burn_in: int = 1000,
                       proposal_cov: Optional[np.ndarray] = None,
                       random_state: Optional[int] = None,
                       **kwargs) -> SamplingResult:
    """Convenience function for Metropolis-Hastings sampling."""
    sampler = MetropolisHastings(log_prob_func, proposal_cov, random_state)
    return sampler.sample(initial_state, n_samples, burn_in, **kwargs)
def hamiltonian_monte_carlo(log_prob_func: Callable,
                           grad_log_prob_func: Callable,
                           initial_state: np.ndarray,
                           n_samples: int,
                           burn_in: int = 1000,
                           step_size: float = 0.1,
                           n_leapfrog: int = 10,
                           random_state: Optional[int] = None,
                           **kwargs) -> SamplingResult:
    """Convenience function for Hamiltonian Monte Carlo sampling."""
    sampler = HamiltonianMonteCarlo(log_prob_func, grad_log_prob_func,
                                   step_size, n_leapfrog, random_state)
    return sampler.sample(initial_state, n_samples, burn_in, **kwargs)
def importance_sampling(target_log_prob: Callable,
                       proposal_sampler: Callable,
                       proposal_log_prob: Callable,
                       n_samples: int,
                       random_state: Optional[int] = None) -> SamplingResult:
    """Convenience function for importance sampling."""
    sampler = ImportanceSampler(target_log_prob, proposal_sampler,
                               proposal_log_prob, random_state)
    return sampler.sample(n_samples)
def rejection_sampling(target_func: Callable,
                      proposal_sampler: Callable,
                      envelope_constant: float,
                      n_samples: int,
                      random_state: Optional[int] = None,
                      **kwargs) -> SamplingResult:
    """Convenience function for rejection sampling."""
    sampler = RejectionSampler(target_func, proposal_sampler,
                              envelope_constant, random_state)
    return sampler.sample(n_samples, **kwargs)