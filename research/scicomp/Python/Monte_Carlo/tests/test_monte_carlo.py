#!/usr/bin/env python3
"""
Comprehensive Test Suite for Monte Carlo Methods Package
This test suite provides thorough testing of all Monte Carlo components
in the Berkeley SciComp framework.
Test Coverage:
- Integration methods (MC, QMC, adaptive)
- Sampling methods (MCMC, importance, rejection)
- Optimization algorithms (SA, GA, PSO, CEM)
- Uncertainty quantification and sensitivity analysis
- Utility functions and diagnostics
- Visualization components
Author: Berkeley SciComp Team
Date: 2024
"""
import unittest
import numpy as np
import warnings
from typing import List, Dict, Any
import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import Monte Carlo modules
from Monte_Carlo import (
    # Integration
    MonteCarloIntegrator,
    QuasiMonteCarloIntegrator,
    AdaptiveMonteCarloIntegrator,
    monte_carlo_integrate,
    # Sampling
    MetropolisHastings,
    HamiltonianMonteCarlo,
    ImportanceSampler,
    RejectionSampler,
    GibbsSampler,
    # Optimization
    SimulatedAnnealing,
    GeneticAlgorithm,
    ParticleSwarmOptimization,
    CrossEntropyMethod,
    # Uncertainty quantification
    UncertaintyQuantifier,
    SensitivityAnalyzer,
    PolynomialChaos,
    # Utilities
    RandomNumberGenerator,
    StatisticalTests,
    set_seed,
    compute_statistics,
    effective_sample_size,
    autocorrelation,
    gelman_rubin,
    # Constants
    PHYSICAL_CONSTANTS,
    MATHEMATICAL_CONSTANTS,
    get_distribution,
    # Visualization
    MonteCarloVisualizer
)
class TestMonteCarloBerkeley(unittest.TestCase):
    """Comprehensive test suite for Monte Carlo methods."""
    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        set_seed(self.seed)
        self.tolerance = 0.1  # 10% tolerance for stochastic methods
        # Test functions
        self.quadratic = lambda x: x**2
        self.linear_2d = lambda x: x[0] + 2*x[1]
        self.rosenbrock = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
        # Test data
        self.n_samples = 1000  # Small for fast testing
        self.bounds_1d = (0, 1)
        self.bounds_2d = [(-2, 2), (-2, 2)]
class TestIntegration(TestMonteCarloBerkeley):
    """Test Monte Carlo integration methods."""
    def test_monte_carlo_integration_1d(self):
        """Test 1D Monte Carlo integration."""
        # Integrate x^2 from 0 to 1 (analytical: 1/3)
        integrator = MonteCarloIntegrator(random_state=self.seed)
        result = integrator.integrate(self.quadratic, self.bounds_1d, self.n_samples)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.value, 1/3, delta=self.tolerance)
        self.assertGreater(result.error, 0)
        self.assertEqual(result.n_samples, self.n_samples)
    def test_monte_carlo_integration_2d(self):
        """Test 2D Monte Carlo integration."""
        # Integrate x + 2y over [-2,2] x [-2,2] (analytical: 0)
        integrator = MonteCarloIntegrator(random_state=self.seed)
        result = integrator.integrate(self.linear_2d, self.bounds_2d, self.n_samples)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.value, 0, delta=self.tolerance)
        self.assertEqual(result.metadata['dimension'], 2)
    def test_quasi_monte_carlo_integration(self):
        """Test quasi-Monte Carlo integration."""
        integrator = QuasiMonteCarloIntegrator(random_state=self.seed)
        result = integrator.integrate(self.quadratic, self.bounds_1d, self.n_samples)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.value, 1/3, delta=self.tolerance)
        self.assertEqual(result.metadata['method'], 'quasi_monte_carlo')
    def test_adaptive_monte_carlo_integration(self):
        """Test adaptive Monte Carlo integration."""
        integrator = AdaptiveMonteCarloIntegrator(random_state=self.seed)
        result = integrator.integrate(self.quadratic, self.bounds_1d, self.n_samples,
                                    adaptation_strategy='importance')
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.value, 1/3, delta=self.tolerance)
        self.assertEqual(result.metadata['method'], 'adaptive_importance')
    def test_convenience_function(self):
        """Test convenience integration function."""
        result = monte_carlo_integrate(self.quadratic, self.bounds_1d,
                                     self.n_samples, random_state=self.seed)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.value, 1/3, delta=self.tolerance)
class TestSampling(TestMonteCarloBerkeley):
    """Test Monte Carlo sampling methods."""
    def test_metropolis_hastings_normal(self):
        """Test Metropolis-Hastings sampling from normal distribution."""
        # Standard normal distribution
        log_prob = lambda x: -0.5 * np.sum(x**2)
        sampler = MetropolisHastings(log_prob, random_state=self.seed)
        result = sampler.sample(
            initial_state=np.array([0.0]),
            n_samples=self.n_samples,
            burn_in=100
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result.samples), self.n_samples)
        self.assertGreater(result.acceptance_rate, 0.1)
        self.assertLess(result.acceptance_rate, 0.9)
        # Check sample statistics
        samples = result.samples.flatten()
        sample_mean = np.mean(samples)
        sample_std = np.std(samples, ddof=1)
        self.assertAlmostEqual(sample_mean, 0, delta=0.2)
        self.assertAlmostEqual(sample_std, 1, delta=0.2)
    def test_hamiltonian_monte_carlo(self):
        """Test Hamiltonian Monte Carlo sampling."""
        # 2D Gaussian
        log_prob = lambda x: -0.5 * np.sum(x**2)
        grad_log_prob = lambda x: -x
        sampler = HamiltonianMonteCarlo(
            log_prob, grad_log_prob,
            step_size=0.1, n_leapfrog=5,
            random_state=self.seed
        )
        result = sampler.sample(
            initial_state=np.array([0.0, 0.0]),
            n_samples=500,  # Smaller for HMC
            burn_in=100
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.samples.shape, (500, 2))
        self.assertGreater(result.acceptance_rate, 0.5)  # HMC should have high acceptance
    def test_importance_sampling(self):
        """Test importance sampling."""
        # Target: standard normal, Proposal: wider normal
        target_log_prob = lambda x: -0.5 * x**2 - 0.5 * np.log(2*np.pi)
        proposal_sampler = lambda n: np.random.normal(0, 2, n)
        proposal_log_prob = lambda x: -0.5 * (x/2)**2 - 0.5 * np.log(2*np.pi*4)
        sampler = ImportanceSampler(
            target_log_prob, proposal_sampler, proposal_log_prob,
            random_state=self.seed
        )
        result = sampler.sample(self.n_samples)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.samples), self.n_samples)
        self.assertEqual(result.acceptance_rate, 1.0)  # All samples accepted
        self.assertIn('weights', result.diagnostics)
    def test_rejection_sampling(self):
        """Test rejection sampling."""
        # Sample from triangular distribution using uniform envelope
        target_func = lambda x: 2 * x if 0 <= x <= 1 else 0
        proposal_sampler = lambda n: np.random.uniform(0, 1, n)
        envelope_constant = 2.0
        sampler = RejectionSampler(
            target_func, proposal_sampler, envelope_constant,
            random_state=self.seed
        )
        result = sampler.sample(self.n_samples // 2)  # Expecting ~50% acceptance
        self.assertIsNotNone(result)
        self.assertGreater(len(result.samples), 0)
        self.assertGreater(result.acceptance_rate, 0.3)
        self.assertLess(result.acceptance_rate, 0.7)
    def test_gibbs_sampling(self):
        """Test Gibbs sampling."""
        # Simple 2D case where conditionals are normal
        def conditional_x1(state):
            # x1 | x2 ~ N(0.5*x2, 1)
            return np.random.normal(0.5 * state[1], 1)
        def conditional_x2(state):
            # x2 | x1 ~ N(0.5*x1, 1)
            return np.random.normal(0.5 * state[0], 1)
        sampler = GibbsSampler(
            [conditional_x1, conditional_x2],
            random_state=self.seed
        )
        result = sampler.sample(
            initial_state=np.array([0.0, 0.0]),
            n_samples=self.n_samples,
            burn_in=100
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.samples.shape, (self.n_samples, 2))
        self.assertEqual(result.acceptance_rate, 1.0)  # Gibbs always accepts
class TestOptimization(TestMonteCarloBerkeley):
    """Test Monte Carlo optimization algorithms."""
    def test_simulated_annealing(self):
        """Test simulated annealing optimization."""
        # Minimize Rosenbrock function (minimum at [1, 1])
        optimizer = SimulatedAnnealing(
            initial_temperature=10.0,
            max_iterations=1000,
            random_state=self.seed,
            verbose=False
        )
        result = optimizer.optimize(self.rosenbrock, self.bounds_2d)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertLess(result.f_optimal, 10)  # Should find reasonable minimum
        self.assertEqual(len(result.x_optimal), 2)
    def test_genetic_algorithm(self):
        """Test genetic algorithm optimization."""
        optimizer = GeneticAlgorithm(
            population_size=20,
            n_generations=50,
            random_state=self.seed,
            verbose=False
        )
        result = optimizer.optimize(self.rosenbrock, self.bounds_2d)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertLess(result.f_optimal, 50)  # GA may not converge as well
        self.assertGreater(result.n_evaluations, 0)
    def test_particle_swarm_optimization(self):
        """Test particle swarm optimization."""
        optimizer = ParticleSwarmOptimization(
            n_particles=15,
            max_iterations=100,
            random_state=self.seed,
            verbose=False
        )
        result = optimizer.optimize(self.rosenbrock, self.bounds_2d)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertLess(result.f_optimal, 20)
        self.assertIn('n_particles', result.metadata)
    def test_cross_entropy_method(self):
        """Test cross-entropy method optimization."""
        optimizer = CrossEntropyMethod(
            population_size=50,
            max_iterations=30,
            random_state=self.seed,
            verbose=False
        )
        result = optimizer.optimize(self.rosenbrock, self.bounds_2d)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertLess(result.f_optimal, 30)
        self.assertIn('elite_fraction', result.metadata)
class TestUncertaintyQuantification(TestMonteCarloBerkeley):
    """Test uncertainty quantification methods."""
    def test_uncertainty_propagation(self):
        """Test uncertainty propagation through model."""
        # Simple linear model: y = 2*x1 + x2
        model = lambda x: 2*x[0] + x[1]
        input_distributions = [
            {'distribution': 'normal', 'loc': 0, 'scale': 1},
            {'distribution': 'normal', 'loc': 1, 'scale': 0.5}
        ]
        uq = UncertaintyQuantifier(random_state=self.seed)
        result = uq.propagate_uncertainty(
            model, input_distributions, n_samples=self.n_samples
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.mean, 1, delta=0.2)  # E[2*0 + 1] = 1
        self.assertGreater(result.std, 0)
        self.assertIn('95%', result.confidence_intervals)
    def test_sensitivity_analysis(self):
        """Test Sobol sensitivity analysis."""
        # Additive model: y = x1 + 2*x2 + 3*x3
        model = lambda x: x[0] + 2*x[1] + 3*x[2]
        input_distributions = [
            {'distribution': 'uniform', 'loc': 0, 'scale': 1},
            {'distribution': 'uniform', 'loc': 0, 'scale': 1},
            {'distribution': 'uniform', 'loc': 0, 'scale': 1}
        ]
        sa = SensitivityAnalyzer(random_state=self.seed)
        result = sa.analyze(
            model, input_distributions, n_samples=500,  # Small for speed
            variable_names=['X1', 'X2', 'X3']
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result.first_order), 3)
        self.assertEqual(len(result.total_order), 3)
        self.assertEqual(len(result.variable_names), 3)
        # For additive model, first-order = total-order
        np.testing.assert_allclose(
            result.first_order, result.total_order, atol=0.1
        )
    def test_polynomial_chaos_expansion(self):
        """Test polynomial chaos expansion."""
        # Simple quadratic model
        model = lambda x: x[0]**2 + x[1]
        input_distributions = [
            {'distribution': 'normal', 'loc': 0, 'scale': 1},
            {'distribution': 'normal', 'loc': 0, 'scale': 1}
        ]
        pce = PolynomialChaos(random_state=self.seed)
        # This will issue a warning since PCE is not fully implemented
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_info = pce.fit(model, input_distributions, polynomial_order=2, n_samples=100)
        self.assertIsNotNone(fit_info)
        self.assertIn('n_training_samples', fit_info)
class TestUtilities(TestMonteCarloBerkeley):
    """Test utility functions."""
    def test_random_number_generator(self):
        """Test random number generator class."""
        rng = RandomNumberGenerator(seed=self.seed)
        # Test various distributions
        normal_samples = rng.normal(0, 1, 100)
        uniform_samples = rng.uniform(0, 1, 100)
        self.assertEqual(len(normal_samples), 100)
        self.assertEqual(len(uniform_samples), 100)
        self.assertTrue(np.all(uniform_samples >= 0))
        self.assertTrue(np.all(uniform_samples <= 1))
    def test_statistical_tests(self):
        """Test statistical test functions."""
        # Generate test data
        normal_data = np.random.normal(0, 1, 200)
        uniform_data = np.random.uniform(0, 1, 200)
        # Kolmogorov-Smirnov test
        ks_result = StatisticalTests.kolmogorov_smirnov_test(normal_data, uniform_data)
        self.assertIn('statistic', ks_result)
        self.assertIn('p_value', ks_result)
        self.assertTrue(ks_result['samples_different'])  # Should be different
        # Shapiro-Wilk test
        sw_result = StatisticalTests.shapiro_wilk_test(normal_data)
        self.assertIn('statistic', sw_result)
        self.assertIn('p_value', sw_result)
    def test_compute_statistics(self):
        """Test statistics computation."""
        data = np.random.normal(5, 2, 1000)
        stats = compute_statistics(data)
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('median', stats)
        self.assertAlmostEqual(stats['mean'], 5, delta=0.2)
        self.assertAlmostEqual(stats['std'], 2, delta=0.2)
    def test_effective_sample_size(self):
        """Test effective sample size calculation."""
        # Uncorrelated samples should have ESS ≈ N
        uncorrelated = np.random.normal(0, 1, 1000)
        ess = effective_sample_size(uncorrelated)
        self.assertIsInstance(ess, float)
        self.assertGreater(ess, 500)  # Should be reasonably high
    def test_autocorrelation(self):
        """Test autocorrelation function."""
        # White noise should have low autocorrelation
        white_noise = np.random.normal(0, 1, 1000)
        autocorr = autocorrelation(white_noise, max_lag=50)
        self.assertEqual(len(autocorr), 51)  # 0 to max_lag
        self.assertAlmostEqual(autocorr[0], 1.0, delta=1e-10)  # Lag 0 = 1
    def test_gelman_rubin_diagnostic(self):
        """Test Gelman-Rubin convergence diagnostic."""
        # Create multiple chains from same distribution
        chains = [np.random.normal(0, 1, 500) for _ in range(4)]
        r_hat = gelman_rubin(chains)
        self.assertIsInstance(r_hat, float)
        self.assertLess(r_hat, 1.2)  # Should indicate convergence
class TestConstants(TestMonteCarloBerkeley):
    """Test physical constants and distributions."""
    def test_physical_constants(self):
        """Test physical constants."""
        self.assertIn('c', PHYSICAL_CONSTANTS)
        self.assertIn('h', PHYSICAL_CONSTANTS)
        self.assertIn('k_B', PHYSICAL_CONSTANTS)
        # Check some known values
        self.assertEqual(PHYSICAL_CONSTANTS['c'], 299792458.0)
        self.assertAlmostEqual(PHYSICAL_CONSTANTS['h'], 6.62607015e-34)
    def test_mathematical_constants(self):
        """Test mathematical constants."""
        self.assertIn('pi', MATHEMATICAL_CONSTANTS)
        self.assertIn('e', MATHEMATICAL_CONSTANTS)
        self.assertAlmostEqual(MATHEMATICAL_CONSTANTS['pi'], np.pi)
        self.assertAlmostEqual(MATHEMATICAL_CONSTANTS['e'], np.e)
    def test_get_distribution(self):
        """Test distribution creation."""
        # Normal distribution
        normal_dist = get_distribution('normal', loc=0, scale=1)
        self.assertIsNotNone(normal_dist)
        # Sample from distribution
        samples = normal_dist.rvs(100)
        self.assertEqual(len(samples), 100)
        # PDF evaluation
        pdf_vals = normal_dist.pdf(np.array([0, 1, -1]))
        self.assertEqual(len(pdf_vals), 3)
        self.assertGreater(pdf_vals[0], pdf_vals[1])  # PDF higher at mean
class TestVisualization(TestMonteCarloBerkeley):
    """Test visualization components."""
    def test_monte_carlo_visualizer(self):
        """Test Monte Carlo visualizer class."""
        visualizer = MonteCarloVisualizer()
        self.assertIsNotNone(visualizer)
        self.assertEqual(visualizer.style, 'berkeley')
    def test_plot_convergence(self):
        """Test convergence plotting (without display)."""
        # Import plotting functions
        from Monte_Carlo.visualization import plot_convergence
        # Generate fake convergence data
        convergence_data = [1.0 / (i + 1) for i in range(100)]
        # This should not raise an error
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            fig = plot_convergence(convergence_data, title="Test Convergence")
            self.assertIsNotNone(fig)
        except ImportError:
            self.skipTest("Matplotlib not available")
class TestIntegration_E2E(TestMonteCarloBerkeley):
    """End-to-end integration tests."""
    def test_complete_workflow(self):
        """Test complete Monte Carlo workflow."""
        # Define a physics problem: damped harmonic oscillator
        def oscillator_amplitude(params):
            """Final amplitude of damped harmonic oscillator."""
            omega0, gamma, t = params
            if gamma >= omega0:  # Overdamped
                return np.exp(-gamma * t)
            else:  # Underdamped
                omega_d = np.sqrt(omega0**2 - gamma**2)
                return np.exp(-gamma * t) * np.cos(omega_d * t)
        # Define parameter uncertainties
        input_distributions = [
            {'distribution': 'normal', 'loc': 10.0, 'scale': 0.5},  # omega0
            {'distribution': 'uniform', 'loc': 0.5, 'scale': 1.0},  # gamma
            {'distribution': 'normal', 'loc': 1.0, 'scale': 0.1}    # t
        ]
        # 1. Uncertainty quantification
        uq_result = monte_carlo_uncertainty(
            oscillator_amplitude, input_distributions,
            n_samples=500, random_state=self.seed
        )
        self.assertIsNotNone(uq_result)
        self.assertGreater(uq_result.std, 0)
        # 2. Sensitivity analysis
        sa_result = sensitivity_analysis(
            oscillator_amplitude, input_distributions,
            n_samples=300, random_state=self.seed
        )
        self.assertIsNotNone(sa_result)
        self.assertEqual(len(sa_result.first_order), 3)
        # 3. Optimization (find parameters that maximize amplitude)
        def negative_amplitude(params):
            return -abs(oscillator_amplitude(params))
        bounds = [(8, 12), (0.1, 2), (0.5, 1.5)]
        optimizer = SimulatedAnnealing(
            max_iterations=500, random_state=self.seed, verbose=False
        )
        opt_result = optimizer.optimize(negative_amplitude, bounds)
        self.assertIsNotNone(opt_result)
        self.assertTrue(opt_result.success)
        print(f"✅ Complete workflow test passed!")
        print(f"   UQ mean: {uq_result.mean:.4f} ± {uq_result.std:.4f}")
        print(f"   SA first-order indices: {sa_result.first_order}")
        print(f"   Optimal parameters: {opt_result.x_optimal}")
def run_monte_carlo_tests():
    """Run all Monte Carlo tests with detailed reporting."""
    print("🧪 Berkeley SciComp: Monte Carlo Test Suite")
    print("=" * 50)
    # Create test suite
    test_classes = [
        TestIntegration,
        TestSampling,
        TestOptimization,
        TestUncertaintyQuantification,
        TestUtilities,
        TestConstants,
        TestVisualization,
        TestIntegration_E2E
    ]
    # Run tests
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    for test_class in test_classes:
        print(f"\n🔬 Testing {test_class.__name__}")
        print("-" * 30)
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
        result = runner.run(suite)
        total_tests += result.testsRun
        passed_tests += result.testsRun - len(result.failures) - len(result.errors)
        if result.failures:
            failed_tests.extend([f"{test_class.__name__}: {f[0]}" for f in result.failures])
        if result.errors:
            failed_tests.extend([f"{test_class.__name__}: {e[0]}" for e in result.errors])
    # Summary
    print("\n" + "=" * 50)
    print("🎯 TEST SUMMARY")
    print("=" * 50)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    if failed_tests:
        print("\n❌ Failed tests:")
        for test in failed_tests:
            print(f"  - {test}")
    else:
        print("\n✅ All tests passed! Monte Carlo package is ready for use.")
    print("\n🎲 Test coverage areas:")
    print("  ✓ Monte Carlo integration (standard, quasi, adaptive)")
    print("  ✓ MCMC sampling (Metropolis-Hastings, HMC, Gibbs)")
    print("  ✓ Advanced sampling (importance, rejection)")
    print("  ✓ Global optimization (SA, GA, PSO, CEM)")
    print("  ✓ Uncertainty quantification and sensitivity analysis")
    print("  ✓ Statistical utilities and diagnostics")
    print("  ✓ Physical constants and distributions")
    print("  ✓ Berkeley-themed visualization")
    print("  ✓ End-to-end workflows")
    return passed_tests == total_tests
if __name__ == "__main__":
    success = run_monte_carlo_tests()
    sys.exit(0 if success else 1)