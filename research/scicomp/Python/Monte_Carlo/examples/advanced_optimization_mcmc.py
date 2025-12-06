#!/usr/bin/env python3
"""
Advanced Monte Carlo: Optimization and MCMC Example
This example demonstrates advanced Monte Carlo techniques including
sophisticated optimization algorithms, Hamiltonian Monte Carlo,
and applications to complex scientific problems.
Topics covered:
- Global optimization with Monte Carlo methods
- Hamiltonian Monte Carlo for high-dimensional sampling
- Bayesian parameter estimation
- Advanced MCMC diagnostics
- Physics-informed applications
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Monte_Carlo import (
    SimulatedAnnealing,
    GeneticAlgorithm,
    ParticleSwarmOptimization,
    HamiltonianMonteCarlo,
    MetropolisHastings,
    plot_convergence,
    plot_samples,
    effective_sample_size,
    gelman_rubin,
    MonteCarloVisualizer,
    PHYSICAL_CONSTANTS
)
def main():
    """Main function demonstrating advanced Monte Carlo methods."""
    print("🚀 Berkeley SciComp: Advanced Monte Carlo Methods")
    print("=" * 55)
    # Example 1: Global Optimization Comparison
    print("\n1. Global Optimization Algorithm Comparison")
    print("-" * 45)
    def rastrigin_function(x):
        """
        Rastrigin function - a challenging multimodal optimization problem.
        Global minimum at x = [0, 0, ...] with f(x) = 0.
        """
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    def ackley_function(x):
        """
        Ackley function - another challenging test function.
        Global minimum at x = [0, 0, ...] with f(x) = 0.
        """
        a, b, c = 20, 0.2, 2*np.pi
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        return (-a * np.exp(-b * np.sqrt(sum1/n)) -
                np.exp(sum2/n) + a + np.exp(1))
    # Test on 5D Rastrigin function
    bounds = [(-5.12, 5.12)] * 5  # 5-dimensional problem
    print("Optimizing 5D Rastrigin function...")
    print("Global minimum: f(0,0,0,0,0) = 0")
    # Simulated Annealing
    print("\nSimulated Annealing:")
    sa_optimizer = SimulatedAnnealing(
        initial_temperature=100.0,
        max_iterations=10000,
        random_state=42,
        verbose=False
    )
    sa_result = sa_optimizer.optimize(rastrigin_function, bounds)
    print(f"Best solution: {sa_result.x_optimal}")
    print(f"Best value: {sa_result.f_optimal:.6f}")
    print(f"Function evaluations: {sa_result.n_evaluations}")
    print(f"Final temperature: {sa_result.metadata['final_temperature']:.6f}")
    # Genetic Algorithm
    print("\nGenetic Algorithm:")
    ga_optimizer = GeneticAlgorithm(
        population_size=50,
        n_generations=100,
        random_state=42,
        verbose=False
    )
    ga_result = ga_optimizer.optimize(rastrigin_function, bounds)
    print(f"Best solution: {ga_result.x_optimal}")
    print(f"Best value: {ga_result.f_optimal:.6f}")
    print(f"Function evaluations: {ga_result.n_evaluations}")
    # Particle Swarm Optimization
    print("\nParticle Swarm Optimization:")
    pso_optimizer = ParticleSwarmOptimization(
        n_particles=30,
        max_iterations=200,
        random_state=42,
        verbose=False
    )
    pso_result = pso_optimizer.optimize(rastrigin_function, bounds)
    print(f"Best solution: {pso_result.x_optimal}")
    print(f"Best value: {pso_result.f_optimal:.6f}")
    print(f"Function evaluations: {pso_result.n_evaluations}")
    # Compare convergence
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot convergence histories
    ax.semilogy(sa_result.convergence_history, label='Simulated Annealing',
               color='#003262', linewidth=2)
    ax.semilogy(ga_result.convergence_history, label='Genetic Algorithm',
               color='#FDB515', linewidth=2)
    ax.semilogy(pso_result.convergence_history, label='Particle Swarm',
               color='#3B7EA1', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Function Value')
    ax.set_title('Global Optimization Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
    # Example 2: Hamiltonian Monte Carlo for Bayesian Inference
    print("\n2. Hamiltonian Monte Carlo: Bayesian Parameter Estimation")
    print("-" * 55)
    # Generate synthetic experimental data
    np.random.seed(42)
    # True parameters for data generation
    true_params = {'a': 2.5, 'b': -1.2, 'sigma': 0.3}
    # Generate noisy observations of y = a*x + b + noise
    n_obs = 50
    x_obs = np.linspace(0, 5, n_obs)
    y_true = true_params['a'] * x_obs + true_params['b']
    y_obs = y_true + np.random.normal(0, true_params['sigma'], n_obs)
    print(f"Generated {n_obs} observations from linear model: y = ax + b + ε")
    print(f"True parameters: a={true_params['a']}, b={true_params['b']}, σ={true_params['sigma']}")
    def log_posterior(params):
        """
        Log posterior for Bayesian linear regression.
        Prior: a ~ N(0, 10), b ~ N(0, 10), log(σ) ~ N(0, 1)
        Likelihood: y ~ N(ax + b, σ²)
        """
        a, b, log_sigma = params
        sigma = np.exp(log_sigma)
        # Prior log probabilities
        log_prior_a = stats.norm.logpdf(a, 0, 10)
        log_prior_b = stats.norm.logpdf(b, 0, 10)
        log_prior_sigma = stats.norm.logpdf(log_sigma, 0, 1)
        # Likelihood
        y_pred = a * x_obs + b
        log_likelihood = np.sum(stats.norm.logpdf(y_obs, y_pred, sigma))
        return log_prior_a + log_prior_b + log_prior_sigma + log_likelihood
    def grad_log_posterior(params):
        """Gradient of log posterior (required for HMC)."""
        a, b, log_sigma = params
        sigma = np.exp(log_sigma)
        # Predicted values
        y_pred = a * x_obs + b
        residuals = y_obs - y_pred
        # Gradients
        grad_a = -a/100 + np.sum(residuals * x_obs) / (sigma**2)  # Prior N(0,10) -> var=100
        grad_b = -b/100 + np.sum(residuals) / (sigma**2)
        # For log_sigma: d/d(log_sigma) = sigma * d/d_sigma
        grad_log_sigma = (-log_sigma +
                         n_obs +
                         np.sum(residuals**2) / (sigma**2))
        return np.array([grad_a, grad_b, grad_log_sigma])
    # Run Hamiltonian Monte Carlo
    print("\nRunning Hamiltonian Monte Carlo...")
    hmc_sampler = HamiltonianMonteCarlo(
        log_prob_func=log_posterior,
        grad_log_prob_func=grad_log_posterior,
        step_size=0.01,
        n_leapfrog=10,
        random_state=42,
        verbose=True
    )
    # Initial state (near MAP estimate)
    initial_state = np.array([2.0, -1.0, np.log(0.5)])
    hmc_result = hmc_sampler.sample(
        initial_state=initial_state,
        n_samples=5000,
        burn_in=1000
    )
    print(f"HMC Acceptance rate: {hmc_result.acceptance_rate:.3f}")
    print(f"Effective sample size: {hmc_result.effective_sample_size:.1f}")
    # Extract parameter samples
    param_samples = hmc_result.samples
    a_samples = param_samples[:, 0]
    b_samples = param_samples[:, 1]
    sigma_samples = np.exp(param_samples[:, 2])  # Transform back from log
    # Posterior statistics
    print("\nPosterior Statistics:")
    print("Parameter | True Value | Posterior Mean | 95% Credible Interval")
    print("-" * 65)
    params_true = [true_params['a'], true_params['b'], true_params['sigma']]
    params_names = ['a', 'b', 'σ']
    params_samples = [a_samples, b_samples, sigma_samples]
    for name, true_val, samples in zip(params_names, params_true, params_samples):
        posterior_mean = np.mean(samples)
        ci_lower, ci_upper = np.percentile(samples, [2.5, 97.5])
        print(f"{name:9} | {true_val:10.3f} | {posterior_mean:13.3f} | [{ci_lower:.3f}, {ci_upper:.3f}]")
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # Parameter traces
    for i, (name, samples) in enumerate(zip(params_names, params_samples)):
        axes[0, i].plot(samples[:1000], color='#003262', alpha=0.7)
        axes[0, i].set_title(f'{name} Trace')
        axes[0, i].set_ylabel(name)
        axes[0, i].grid(True, alpha=0.3)
        # Parameter histograms
        axes[1, i].hist(samples, bins=50, density=True, alpha=0.7, color='#003262')
        axes[1, i].axvline(params_true[i], color='#FDB515', linewidth=2,
                          linestyle='--', label='True Value')
        axes[1, i].set_title(f'{name} Posterior')
        axes[1, i].set_xlabel(name)
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # Example 3: MCMC Diagnostics and Convergence
    print("\n3. MCMC Diagnostics and Convergence Assessment")
    print("-" * 45)
    # Run multiple chains for convergence diagnostics
    print("Running multiple MCMC chains for convergence assessment...")
    n_chains = 4
    chain_results = []
    for chain_id in range(n_chains):
        # Different initial states for each chain
        initial_state = initial_state + np.random.normal(0, 0.5, 3)
        chain_result = hmc_sampler.sample(
            initial_state=initial_state,
            n_samples=2000,
            burn_in=500
        )
        chain_results.append(chain_result.samples)
    # Gelman-Rubin diagnostic
    print("\nGelman-Rubin Diagnostic (R-hat values):")
    print("Parameter | R-hat | Interpretation")
    print("-" * 35)
    for i, name in enumerate(params_names):
        chains_param = [chain[:, i] for chain in chain_results]
        r_hat = gelman_rubin(chains_param)
        if r_hat < 1.01:
            interpretation = "Excellent"
        elif r_hat < 1.05:
            interpretation = "Good"
        elif r_hat < 1.1:
            interpretation = "Acceptable"
        else:
            interpretation = "Poor"
        print(f"{name:9} | {r_hat:.4f} | {interpretation}")
    print("\nR-hat < 1.01: Excellent convergence")
    print("R-hat < 1.05: Good convergence")
    print("R-hat > 1.1:  Poor convergence")
    # Effective sample size
    print("\nEffective Sample Size:")
    print("Parameter | ESS   | ESS/N")
    print("-" * 25)
    for i, name in enumerate(params_names):
        all_samples = np.concatenate([chain[:, i] for chain in chain_results])
        ess = effective_sample_size(all_samples)
        ess_ratio = ess / len(all_samples)
        print(f"{name:9} | {ess:5.0f} | {ess_ratio:.3f}")
    # Example 4: Advanced MCMC: Adaptive Proposals
    print("\n4. Adaptive MCMC with Custom Proposals")
    print("-" * 35)
    class AdaptiveMetropolisHastings:
        """Adaptive Metropolis-Hastings with proposal adaptation."""
        def __init__(self, log_prob_func, initial_cov=None, adaptation_interval=50):
            self.log_prob_func = log_prob_func
            self.adaptation_interval = adaptation_interval
            self.initial_cov = initial_cov
        def sample(self, initial_state, n_samples, burn_in=1000):
            """Sample with adaptive proposal covariance."""
            dimension = len(initial_state)
            if self.initial_cov is None:
                proposal_cov = 0.1 * np.eye(dimension)
            else:
                proposal_cov = self.initial_cov.copy()
            # Storage
            samples = np.zeros((n_samples + burn_in, dimension))
            log_probs = np.zeros(n_samples + burn_in)
            accepted = np.zeros(n_samples + burn_in, dtype=bool)
            # Initialize
            current_state = initial_state.copy()
            current_log_prob = self.log_prob_func(current_state)
            for i in range(n_samples + burn_in):
                # Propose new state
                proposal = np.random.multivariate_normal(current_state, proposal_cov)
                proposal_log_prob = self.log_prob_func(proposal)
                # Accept/reject
                log_alpha = min(0, proposal_log_prob - current_log_prob)
                if np.log(np.random.random()) < log_alpha:
                    current_state = proposal
                    current_log_prob = proposal_log_prob
                    accepted[i] = True
                samples[i] = current_state
                log_probs[i] = current_log_prob
                # Adapt proposal covariance
                if i > 100 and i % self.adaptation_interval == 0:
                    # Use recent samples to update covariance
                    recent_samples = samples[max(0, i-500):i+1]
                    if len(recent_samples) > dimension:
                        sample_cov = np.cov(recent_samples.T)
                        # Adaptive scaling
                        acceptance_rate = np.mean(accepted[max(0, i-500):i+1])
                        if acceptance_rate > 0.5:
                            scale_factor = 1.1
                        elif acceptance_rate < 0.2:
                            scale_factor = 0.9
                        else:
                            scale_factor = 1.0
                        proposal_cov = scale_factor * (sample_cov + 1e-6 * np.eye(dimension))
            # Return post-burn-in samples
            post_burnin_samples = samples[burn_in:]
            acceptance_rate = np.mean(accepted[burn_in:])
            return {
                'samples': post_burnin_samples,
                'acceptance_rate': acceptance_rate,
                'final_cov': proposal_cov
            }
    print("Running adaptive MCMC...")
    adaptive_mcmc = AdaptiveMetropolisHastings(log_posterior)
    adaptive_result = adaptive_mcmc.sample(initial_state, n_samples=3000, burn_in=1000)
    print(f"Adaptive MCMC acceptance rate: {adaptive_result['acceptance_rate']:.3f}")
    # Compare with standard MCMC
    standard_mcmc = MetropolisHastings(log_posterior, random_state=42)
    standard_result = standard_mcmc.sample(initial_state, n_samples=3000, burn_in=1000)
    print(f"Standard MCMC acceptance rate: {standard_result.acceptance_rate:.3f}")
    # Compare effective sample sizes
    adaptive_ess = np.mean([effective_sample_size(adaptive_result['samples'][:, i])
                           for i in range(3)])
    standard_ess = np.mean([effective_sample_size(standard_result.samples[:, i])
                           for i in range(3)])
    print(f"Adaptive MCMC ESS: {adaptive_ess:.0f}")
    print(f"Standard MCMC ESS: {standard_ess:.0f}")
    print(f"Efficiency improvement: {adaptive_ess/standard_ess:.2f}x")
    # Example 5: Physics Application - Quantum Harmonic Oscillator
    print("\n5. Physics Application: Bayesian Analysis of Quantum Data")
    print("-" * 55)
    def quantum_harmonic_oscillator_energy(n, omega, hbar=PHYSICAL_CONSTANTS['hbar']):
        """Energy levels of quantum harmonic oscillator."""
        return hbar * omega * (n + 0.5)
    # Simulate experimental measurement of energy levels
    true_omega = 2e14  # rad/s (typical molecular vibration)
    n_levels = [0, 1, 2, 3, 4]  # Measured energy levels
    # "Experimental" energies with measurement uncertainty
    true_energies = [quantum_harmonic_oscillator_energy(n, true_omega) for n in n_levels]
    measurement_error = 0.02  # 2% relative uncertainty
    np.random.seed(123)
    measured_energies = [E * (1 + np.random.normal(0, measurement_error))
                        for E in true_energies]
    print("Quantum harmonic oscillator energy measurement:")
    print("Level | True Energy (J) | Measured Energy (J)")
    print("-" * 45)
    for n, E_true, E_meas in zip(n_levels, true_energies, measured_energies):
        print(f"{n:5d} | {E_true:.6e} | {E_meas:.6e}")
    def log_posterior_quantum(params):
        """Log posterior for quantum oscillator frequency estimation."""
        log_omega, log_rel_error = params
        omega = np.exp(log_omega)
        rel_error = np.exp(log_rel_error)
        # Priors
        log_prior_omega = stats.norm.logpdf(log_omega, np.log(1e14), 1)  # Broad prior
        log_prior_error = stats.norm.logpdf(log_rel_error, np.log(0.01), 0.5)  # ~1% error
        # Likelihood
        log_likelihood = 0
        for n, E_meas in zip(n_levels, measured_energies):
            E_pred = quantum_harmonic_oscillator_energy(n, omega)
            sigma = rel_error * E_pred  # Relative error model
            log_likelihood += stats.norm.logpdf(E_meas, E_pred, sigma)
        return log_prior_omega + log_prior_error + log_likelihood
    # MCMC for quantum parameter estimation
    print("\nBayesian estimation of oscillator frequency...")
    quantum_mcmc = MetropolisHastings(log_posterior_quantum, random_state=42)
    initial_quantum = np.array([np.log(1.5e14), np.log(0.015)])
    quantum_result = quantum_mcmc.sample(
        initial_state=initial_quantum,
        n_samples=5000,
        burn_in=1000
    )
    # Transform samples back
    omega_samples = np.exp(quantum_result.samples[:, 0])
    error_samples = np.exp(quantum_result.samples[:, 1])
    # Results
    omega_mean = np.mean(omega_samples)
    omega_ci = np.percentile(omega_samples, [2.5, 97.5])
    print(f"True frequency: {true_omega:.2e} rad/s")
    print(f"Estimated frequency: {omega_mean:.2e} rad/s")
    print(f"95% credible interval: [{omega_ci[0]:.2e}, {omega_ci[1]:.2e}] rad/s")
    relative_error = abs(omega_mean - true_omega) / true_omega * 100
    print(f"Relative error: {relative_error:.2f}%")
    # Final visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Frequency posterior
    ax1.hist(omega_samples/1e14, bins=50, density=True, alpha=0.7, color='#003262')
    ax1.axvline(true_omega/1e14, color='#FDB515', linewidth=2,
               linestyle='--', label='True Value')
    ax1.set_xlabel('Frequency (×10¹⁴ rad/s)')
    ax1.set_ylabel('Posterior Density')
    ax1.set_title('Quantum Oscillator Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Measurement error posterior
    ax2.hist(error_samples*100, bins=50, density=True, alpha=0.7, color='#003262')
    ax2.axvline(measurement_error*100, color='#FDB515', linewidth=2,
               linestyle='--', label='True Value')
    ax2.set_xlabel('Relative Error (%)')
    ax2.set_ylabel('Posterior Density')
    ax2.set_title('Measurement Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("\n" + "=" * 55)
    print("✅ Advanced Monte Carlo examples completed!")
    print("\nKey achievements:")
    print("• Solved challenging optimization problems with multiple algorithms")
    print("• Performed sophisticated Bayesian inference with HMC")
    print("• Implemented comprehensive MCMC diagnostics")
    print("• Developed adaptive sampling strategies")
    print("• Applied methods to realistic physics problems")
    print("• Achieved convergence assessment and uncertainty quantification")
if __name__ == "__main__":
    main()