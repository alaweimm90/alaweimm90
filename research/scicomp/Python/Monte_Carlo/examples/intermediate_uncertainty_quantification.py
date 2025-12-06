#!/usr/bin/env python3
"""
Intermediate Monte Carlo: Uncertainty Quantification Example
This example demonstrates advanced Monte Carlo techniques for uncertainty
quantification and sensitivity analysis in scientific computing.
Topics covered:
- Uncertainty propagation through complex models
- Global sensitivity analysis (Sobol indices)
- Advanced sampling techniques (LHS, importance sampling)
- Variance reduction methods
- Physics-based examples
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
    UncertaintyQuantifier,
    SensitivityAnalyzer,
    monte_carlo_uncertainty,
    sensitivity_analysis,
    plot_sensitivity,
    MonteCarloVisualizer,
    PHYSICAL_CONSTANTS
)
def main():
    """Main function demonstrating uncertainty quantification."""
    print("🔬 Berkeley SciComp: Uncertainty Quantification")
    print("=" * 50)
    # Example 1: Pendulum Period Uncertainty Analysis
    print("\n1. Simple Pendulum Period Analysis")
    print("-" * 35)
    def pendulum_period(inputs):
        """
        Calculate period of simple pendulum with uncertainty.
        Parameters:
        - L: length (m)
        - g: gravitational acceleration (m/s²)
        - theta0: initial angle (rad)
        Returns period accounting for large angle corrections.
        """
        L, g, theta0 = inputs
        # Small angle approximation: T = 2π√(L/g)
        T_small = 2 * np.pi * np.sqrt(L / g)
        # Large angle correction (first-order)
        # T ≈ T_small * (1 + θ₀²/16)
        correction = 1 + (theta0**2) / 16
        return T_small * correction
    # Define input uncertainty distributions
    input_distributions = [
        {'distribution': 'normal', 'loc': 1.0, 'scale': 0.01},      # L: 1.0 ± 0.01 m
        {'distribution': 'normal', 'loc': 9.81, 'scale': 0.05},    # g: 9.81 ± 0.05 m/s²
        {'distribution': 'uniform', 'loc': 0.1, 'scale': 0.4}      # θ₀: 0.1 to 0.5 rad
    ]
    variable_names = ['Length (m)', 'Gravity (m/s²)', 'Initial Angle (rad)']
    print("Analyzing pendulum period uncertainty...")
    print("Input uncertainties:")
    print("- Length: 1.00 ± 0.01 m (normal)")
    print("- Gravity: 9.81 ± 0.05 m/s² (normal)")
    print("- Initial angle: 0.1 to 0.5 rad (uniform)")
    # Perform uncertainty quantification
    uq = UncertaintyQuantifier(random_state=42, verbose=True)
    # Standard Monte Carlo
    result_mc = uq.propagate_uncertainty(
        model=pendulum_period,
        input_distributions=input_distributions,
        n_samples=10000,
        method='monte_carlo'
    )
    print(f"\nPendulum Period Results (Monte Carlo):")
    print(f"Mean: {result_mc.mean:.4f} s")
    print(f"Std: {result_mc.std:.4f} s")
    print(f"95% CI: [{result_mc.confidence_intervals['95%'][0]:.4f}, "
          f"{result_mc.confidence_intervals['95%'][1]:.4f}] s")
    # Latin Hypercube Sampling
    result_lhs = uq.propagate_uncertainty(
        model=pendulum_period,
        input_distributions=input_distributions,
        n_samples=1000,  # Fewer samples needed with LHS
        method='latin_hypercube'
    )
    print(f"\nPendulum Period Results (Latin Hypercube):")
    print(f"Mean: {result_lhs.mean:.4f} s")
    print(f"Std: {result_lhs.std:.4f} s")
    # Compare sampling efficiency
    efficiency_ratio = result_mc.std / result_lhs.std
    print(f"\nSampling efficiency: LHS is {efficiency_ratio:.2f}x more efficient")
    # Visualize results
    visualizer = MonteCarloVisualizer()
    # Plot output distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(result_mc.samples, bins=50, alpha=0.7, density=True,
            color='#003262', label='Monte Carlo')
    ax1.hist(result_lhs.samples, bins=30, alpha=0.7, density=True,
            color='#FDB515', label='Latin Hypercube')
    ax1.set_xlabel('Period (s)')
    ax1.set_ylabel('Density')
    ax1.set_title('Output Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Plot input samples (2D projection)
    ax2.scatter(result_mc.input_samples[:500, 0], result_mc.input_samples[:500, 1],
               alpha=0.6, s=20, color='#003262', label='Monte Carlo')
    ax2.scatter(result_lhs.input_samples[:500, 0], result_lhs.input_samples[:500, 1],
               alpha=0.6, s=20, color='#FDB515', label='Latin Hypercube')
    ax2.set_xlabel('Length (m)')
    ax2.set_ylabel('Gravity (m/s²)')
    ax2.set_title('Input Sample Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # Example 2: Global Sensitivity Analysis
    print("\n2. Global Sensitivity Analysis")
    print("-" * 30)
    print("Performing Sobol sensitivity analysis...")
    # Perform sensitivity analysis
    sa_result = sensitivity_analysis(
        model=pendulum_period,
        input_distributions=input_distributions,
        n_samples=5000,
        variable_names=variable_names
    )
    print("Sensitivity Analysis Results:")
    print("Variable        | First-order | Total-order")
    print("-" * 45)
    for i, name in enumerate(variable_names):
        print(f"{name:15} | {sa_result.first_order[i]:10.4f} | {sa_result.total_order[i]:10.4f}")
    # Plot sensitivity indices
    plot_sensitivity(sa_result, plot_type='bar',
                    title='Pendulum Period Sensitivity Analysis')
    plt.show()
    # Example 3: Heat Transfer Problem
    print("\n3. Heat Transfer Uncertainty Analysis")
    print("-" * 35)
    def heat_transfer_model(inputs):
        """
        1D steady-state heat conduction with convection boundary.
        Parameters:
        - k: thermal conductivity (W/m·K)
        - h: convection coefficient (W/m²·K)
        - T_inf: ambient temperature (K)
        - q_gen: heat generation rate (W/m³)
        - L: length (m)
        Returns: maximum temperature in the rod
        """
        k, h, T_inf, q_gen, L = inputs
        # Analytical solution for 1D heat conduction with generation
        # and convection boundary condition
        # Biot number
        Bi = h * L / k
        # Dimensionless heat generation
        m_squared = h / k
        if m_squared > 0:
            m = np.sqrt(m_squared)
            # Temperature rise due to generation and convection
            # Simplified formula for demonstration
            delta_T = (q_gen * L) / (h + k/L) + (q_gen * L**2) / (2 * k)
        else:
            delta_T = (q_gen * L**2) / (2 * k)
        # Maximum temperature
        T_max = T_inf + delta_T
        return T_max
    # Define input uncertainties for heat transfer problem
    heat_inputs = [
        {'distribution': 'normal', 'loc': 50.0, 'scale': 5.0},     # k: 50 ± 5 W/m·K
        {'distribution': 'lognormal', 's': 0.3, 'scale': 25.0},   # h: ~25 W/m²·K (log-normal)
        {'distribution': 'normal', 'loc': 300.0, 'scale': 2.0},   # T_inf: 300 ± 2 K
        {'distribution': 'uniform', 'loc': 1000, 'scale': 2000},  # q_gen: 1000-3000 W/m³
        {'distribution': 'normal', 'loc': 0.1, 'scale': 0.005}    # L: 0.1 ± 0.005 m
    ]
    heat_variables = ['k (W/m·K)', 'h (W/m²·K)', 'T_∞ (K)', 'q_gen (W/m³)', 'L (m)']
    print("Heat transfer rod maximum temperature analysis")
    print("Input uncertainties:")
    print("- Thermal conductivity: 50 ± 5 W/m·K")
    print("- Convection coefficient: ~25 W/m²·K (log-normal)")
    print("- Ambient temperature: 300 ± 2 K")
    print("- Heat generation: 1000-3000 W/m³ (uniform)")
    print("- Length: 0.1 ± 0.005 m")
    # Uncertainty propagation
    heat_result = monte_carlo_uncertainty(
        model=heat_transfer_model,
        input_distributions=heat_inputs,
        n_samples=10000
    )
    print(f"\nMaximum Temperature Results:")
    print(f"Mean: {heat_result.mean:.2f} K")
    print(f"Std: {heat_result.std:.2f} K")
    print(f"95% CI: [{heat_result.confidence_intervals['95%'][0]:.2f}, "
          f"{heat_result.confidence_intervals['95%'][1]:.2f}] K")
    # Sensitivity analysis for heat transfer
    heat_sa = sensitivity_analysis(
        model=heat_transfer_model,
        input_distributions=heat_inputs,
        n_samples=3000,
        variable_names=heat_variables
    )
    print("\nHeat Transfer Sensitivity Analysis:")
    print("Variable           | First-order | Total-order")
    print("-" * 50)
    for i, name in enumerate(heat_variables):
        print(f"{name:18} | {heat_sa.first_order[i]:10.4f} | {heat_sa.total_order[i]:10.4f}")
    # Tornado plot for heat transfer
    plot_sensitivity(heat_sa, plot_type='tornado',
                    title='Heat Transfer Temperature Sensitivity')
    plt.show()
    # Example 4: Model Validation and Uncertainty
    print("\n4. Model Validation Under Uncertainty")
    print("-" * 35)
    def validate_with_uncertainty():
        """Demonstrate model validation accounting for input uncertainty."""
        # "Experimental" data (synthetic with noise)
        np.random.seed(123)
        experimental_conditions = np.array([
            [1.0, 9.81, 0.2],   # Nominal conditions
            [0.95, 9.81, 0.3],  # Different length
            [1.05, 9.81, 0.15], # Different length and angle
        ])
        # Synthetic experimental measurements with noise
        true_periods = [pendulum_period(cond) for cond in experimental_conditions]
        measurement_noise = 0.02  # 2% measurement uncertainty
        measured_periods = [t + np.random.normal(0, measurement_noise * t)
                           for t in true_periods]
        print("Model validation data:")
        print("Condition | Measured | Model Prediction | Uncertainty")
        print("-" * 55)
        # Validate each condition
        for i, (condition, measured) in enumerate(zip(experimental_conditions, measured_periods)):
            # Create input distributions around experimental conditions
            L, g, theta = condition
            validation_inputs = [
                {'distribution': 'normal', 'loc': L, 'scale': 0.005},      # ±0.5% length
                {'distribution': 'normal', 'loc': g, 'scale': 0.02},       # ±0.2% gravity
                {'distribution': 'normal', 'loc': theta, 'scale': 0.01}    # ±0.01 rad angle
            ]
            # Propagate uncertainty
            val_result = monte_carlo_uncertainty(
                model=pendulum_period,
                input_distributions=validation_inputs,
                n_samples=5000
            )
            # Check if measurement falls within prediction uncertainty
            prediction_mean = val_result.mean
            prediction_std = val_result.std
            # Z-score for validation
            z_score = abs(measured - prediction_mean) / prediction_std
            is_valid = z_score < 2.0  # 95% confidence level
            status = "✓ PASS" if is_valid else "✗ FAIL"
            print(f"{i+1:9d} | {measured:8.4f} | {prediction_mean:8.4f} ± {prediction_std:.4f} | {status}")
        print("\nValidation complete. All predictions should fall within ±2σ.")
    validate_with_uncertainty()
    # Example 5: Advanced Variance Reduction
    print("\n5. Variance Reduction Techniques")
    print("-" * 30)
    def demonstrate_variance_reduction():
        """Compare standard MC with variance reduction techniques."""
        # Function with high variance: oscillatory integrand
        def oscillatory_function(x):
            return np.sin(20 * x) * np.exp(-x)
        bounds = (0, 5)
        n_samples = 10000
        # Standard Monte Carlo
        from Monte_Carlo import monte_carlo_integrate
        std_result = monte_carlo_integrate(oscillatory_function, bounds, n_samples)
        # Stratified sampling (approximate)
        def stratified_integration(func, bounds, n_samples, n_strata=10):
            """Simple stratified sampling implementation."""
            a, b = bounds
            stratum_width = (b - a) / n_strata
            samples_per_stratum = n_samples // n_strata
            total_estimate = 0
            total_variance = 0
            for i in range(n_strata):
                stratum_a = a + i * stratum_width
                stratum_b = stratum_a + stratum_width
                # Sample within stratum
                stratum_samples = np.random.uniform(stratum_a, stratum_b, samples_per_stratum)
                stratum_values = [func(x) for x in stratum_samples]
                stratum_estimate = stratum_width * np.mean(stratum_values)
                stratum_variance = stratum_width**2 * np.var(stratum_values, ddof=1) / samples_per_stratum
                total_estimate += stratum_estimate
                total_variance += stratum_variance
            return total_estimate, np.sqrt(total_variance)
        strat_estimate, strat_error = stratified_integration(
            oscillatory_function, bounds, n_samples
        )
        print("Variance Reduction Comparison:")
        print(f"Standard MC:     {std_result.value:.6f} ± {std_result.error:.6f}")
        print(f"Stratified MC:   {strat_estimate:.6f} ± {strat_error:.6f}")
        variance_reduction = (std_result.error / strat_error)**2
        print(f"Variance reduction factor: {variance_reduction:.2f}")
    demonstrate_variance_reduction()
    print("\n" + "=" * 50)
    print("✅ Uncertainty quantification examples completed!")
    print("\nKey insights:")
    print("• Input uncertainties propagate nonlinearly through models")
    print("• Sensitivity analysis identifies most important parameters")
    print("• Advanced sampling can improve efficiency significantly")
    print("• Model validation must account for uncertainty")
    print("• Variance reduction techniques can reduce computational cost")
if __name__ == "__main__":
    main()