#!/usr/bin/env python3
"""
Beginner Monte Carlo Methods Example
This example demonstrates basic Monte Carlo integration and sampling
for scientific computing applications using the Berkeley SciComp framework.
Topics covered:
- Basic Monte Carlo integration
- Simple random sampling
- Convergence analysis
- Basic visualization
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Monte_Carlo import (
    monte_carlo_integrate,
    MetropolisHastings,
    plot_convergence,
    plot_samples,
    set_seed
)
def main():
    """Main function demonstrating basic Monte Carlo methods."""
    print("🎲 Berkeley SciComp: Beginner Monte Carlo Methods")
    print("=" * 50)
    # Set random seed for reproducibility
    set_seed(42)
    # Example 1: Monte Carlo Integration
    print("\n1. Monte Carlo Integration")
    print("-" * 30)
    # Define a simple function to integrate: f(x) = x^2
    def f(x):
        return x**2
    # Integrate from 0 to 1 (analytical answer = 1/3)
    bounds = (0, 1)
    n_samples = 10000
    print(f"Integrating f(x) = x² from {bounds[0]} to {bounds[1]}")
    print(f"Analytical answer: 1/3 = {1/3:.6f}")
    # Perform Monte Carlo integration
    result = monte_carlo_integrate(f, bounds, n_samples)
    print(f"Monte Carlo estimate: {result.value:.6f}")
    print(f"Error estimate: ±{result.error:.6f}")
    print(f"Relative error: {abs(result.value - 1/3) / (1/3) * 100:.2f}%")
    # Plot convergence
    fig = plot_convergence(
        result.convergence_history,
        title="Monte Carlo Integration Convergence",
        ylabel="Integral Estimate",
        true_value=1/3
    )
    plt.show()
    # Example 2: Estimating π using Monte Carlo
    print("\n2. Estimating π using Monte Carlo")
    print("-" * 30)
    def estimate_pi_monte_carlo(n_samples):
        """Estimate π by sampling points in a unit square."""
        # Generate random points in [0,1] x [0,1]
        x = np.random.random(n_samples)
        y = np.random.random(n_samples)
        # Count points inside unit circle
        inside_circle = (x**2 + y**2) <= 1
        # π/4 = (area of quarter circle) / (area of unit square)
        pi_estimate = 4 * np.sum(inside_circle) / n_samples
        return pi_estimate
    # Estimate π with different sample sizes
    sample_sizes = [100, 1000, 10000, 100000]
    pi_estimates = []
    print(f"True value of π: {np.pi:.6f}")
    print("\nSample Size | π Estimate | Error")
    print("-" * 35)
    for n in sample_sizes:
        pi_est = estimate_pi_monte_carlo(n)
        pi_estimates.append(pi_est)
        error = abs(pi_est - np.pi)
        print(f"{n:10d} | {pi_est:9.6f} | {error:.6f}")
    # Example 3: Basic Sampling from Normal Distribution
    print("\n3. Sampling from Normal Distribution")
    print("-" * 35)
    # Define a simple log probability function (standard normal)
    def log_normal_pdf(x):
        return -0.5 * np.sum(x**2)
    # Create Metropolis-Hastings sampler
    sampler = MetropolisHastings(log_normal_pdf, random_state=42)
    # Generate samples
    initial_state = np.array([0.0])  # Start from origin
    n_samples = 5000
    print(f"Generating {n_samples} samples from standard normal distribution")
    sampling_result = sampler.sample(
        initial_state=initial_state,
        n_samples=n_samples,
        burn_in=1000
    )
    print(f"Acceptance rate: {sampling_result.acceptance_rate:.3f}")
    print(f"Effective sample size: {sampling_result.effective_sample_size:.1f}")
    # Analyze samples
    samples = sampling_result.samples.flatten()
    sample_mean = np.mean(samples)
    sample_std = np.std(samples, ddof=1)
    print(f"Sample mean: {sample_mean:.3f} (expected: 0.000)")
    print(f"Sample std: {sample_std:.3f} (expected: 1.000)")
    # Plot sample distribution
    fig = plot_samples(
        samples,
        title="Samples from Standard Normal Distribution",
        plot_type='histogram'
    )
    plt.show()
    # Example 4: Central Limit Theorem Demonstration
    print("\n4. Central Limit Theorem Demonstration")
    print("-" * 40)
    def demonstrate_clt(n_samples=1000, sample_sizes=[1, 5, 10, 30]):
        """Demonstrate CLT with uniform random variables."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, n in enumerate(sample_sizes):
            # Generate sample means
            sample_means = []
            for _ in range(n_samples):
                # Sample n uniform random variables
                uniform_samples = np.random.uniform(-1, 1, n)
                sample_means.append(np.mean(uniform_samples))
            # Plot histogram
            axes[i].hist(sample_means, bins=30, density=True, alpha=0.7,
                        color='#003262', edgecolor='#003262')
            # Theoretical normal curve
            theoretical_mean = 0  # E[uniform(-1,1)] = 0
            theoretical_std = np.sqrt(1/3) / np.sqrt(n)  # Var[uniform(-1,1)]/n = 1/3/n
            x = np.linspace(-3*theoretical_std, 3*theoretical_std, 100)
            theoretical_pdf = (1 / (theoretical_std * np.sqrt(2*np.pi))) * \
                             np.exp(-0.5 * (x / theoretical_std)**2)
            axes[i].plot(x, theoretical_pdf, 'r-', linewidth=2,
                        label='Theoretical Normal')
            axes[i].set_title(f'Sample Size n={n}')
            axes[i].set_xlabel('Sample Mean')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        fig.suptitle('Central Limit Theorem: Sample Means of Uniform Variables',
                    fontsize=14)
        plt.tight_layout()
        plt.show()
    demonstrate_clt()
    # Example 5: Monte Carlo Error Analysis
    print("\n5. Monte Carlo Error Analysis")
    print("-" * 30)
    def analyze_mc_error():
        """Analyze how Monte Carlo error decreases with sample size."""
        # Function to integrate: sin(x) from 0 to π
        def sin_func(x):
            return np.sin(x)
        bounds = (0, np.pi)
        true_value = 2.0  # ∫₀^π sin(x) dx = 2
        sample_sizes = np.logspace(2, 5, 20).astype(int)  # 100 to 100,000
        errors = []
        theoretical_errors = []
        print("Sample Size | MC Error | Theoretical Error (1/√n)")
        print("-" * 50)
        for n in sample_sizes:
            # Perform integration
            result = monte_carlo_integrate(sin_func, bounds, n)
            # Actual error
            actual_error = abs(result.value - true_value)
            errors.append(actual_error)
            # Theoretical error (proportional to 1/√n)
            # Using estimated variance from a small sample
            small_result = monte_carlo_integrate(sin_func, bounds, 1000)
            theoretical_error = small_result.error * np.sqrt(1000) / np.sqrt(n)
            theoretical_errors.append(theoretical_error)
            if n <= 10000:  # Only print for smaller samples
                print(f"{n:10d} | {actual_error:8.6f} | {theoretical_error:8.6f}")
        # Plot error scaling
        plt.figure(figsize=(10, 6))
        plt.loglog(sample_sizes, errors, 'o-', label='Actual Error',
                  color='#003262', markersize=6)
        plt.loglog(sample_sizes, theoretical_errors, '--',
                  label='1/√n scaling', color='#FDB515', linewidth=2)
        plt.xlabel('Number of Samples')
        plt.ylabel('Absolute Error')
        plt.title('Monte Carlo Error Scaling')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    analyze_mc_error()
    print("\n" + "=" * 50)
    print("✅ Beginner Monte Carlo examples completed!")
    print("Key takeaways:")
    print("• Monte Carlo methods provide statistical estimates")
    print("• Error decreases as 1/√n with sample size")
    print("• Acceptance rates indicate sampler efficiency")
    print("• CLT explains why MC methods work")
if __name__ == "__main__":
    main()