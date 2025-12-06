"""
Test Suite for Stochastic Processes Module
==========================================
Comprehensive tests for stochastic processes functionality.
Author: Berkeley SciComp Team
Date: 2024
"""
import pytest
import numpy as np
import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Python.Stochastic.stochastic_processes import (
    BrownianMotion, RandomWalk, StochasticDifferentialEquation,
    OrnsteinUhlenbeck, JumpDiffusion
)
class TestBrownianMotion:
    """Test Brownian Motion processes."""
    @pytest.fixture
    def bm(self):
        """Create BrownianMotion instance with fixed seed."""
        return BrownianMotion(drift=0.1, volatility=0.3, seed=42)
    def test_standard_brownian_motion(self, bm):
        """Test standard Brownian motion generation."""
        T, n_steps = 10.0, 1000
        t, W = bm.generate_path(T, n_steps, x0=0.0)
        assert len(t) == n_steps + 1
        assert len(W) == n_steps + 1
        assert W[0] == 0.0
        # Check statistical properties
        increments = np.diff(W)
        dt = T / n_steps
        # Mean should be close to drift * dt
        expected_mean = bm.drift * dt
        assert np.mean(increments) == pytest.approx(expected_mean, abs=0.05)
        # Variance should be close to volatility^2 * dt
        expected_var = bm.volatility**2 * dt
        assert np.var(increments) == pytest.approx(expected_var, rel=0.2)
    def test_geometric_brownian_motion(self, bm):
        """Test geometric Brownian motion."""
        T, n_steps = 5.0, 500
        S0 = 100.0
        t, S = bm.generate_geometric_path(T, n_steps, S0)
        assert len(S) == n_steps + 1
        assert S[0] == S0
        assert np.all(S > 0)  # GBM should always be positive
        # Log returns should be normally distributed
        log_returns = np.diff(np.log(S))
        # Perform Jarque-Bera test for normality (relaxed)
        from scipy import stats
        _, p_value = stats.jarque_bera(log_returns)
        assert p_value > 0.01  # Relaxed significance level
    def test_brownian_bridge(self, bm):
        """Test Brownian bridge generation."""
        T, n_steps = 1.0, 100
        x0, xT = 0.0, 1.0
        t, bridge = bm.generate_bridge(T, n_steps, x0, xT)
        assert len(bridge) == n_steps + 1
        assert bridge[0] == pytest.approx(x0)
        assert bridge[-1] == pytest.approx(xT)
        # Bridge should have maximum variance in the middle
        variances = []
        n_samples = 100
        for _ in range(n_samples):
            _, b = bm.generate_bridge(T, n_steps, x0, xT)
            variances.append(b)
        var_profile = np.var(variances, axis=0)
        max_var_idx = np.argmax(var_profile[1:-1]) + 1
        assert n_steps // 3 < max_var_idx < 2 * n_steps // 3
    def test_fractional_brownian_motion(self, bm):
        """Test fractional Brownian motion."""
        T, n_steps = 1.0, 100
        # Test different Hurst parameters
        for H in [0.3, 0.5, 0.7]:
            t, fBm = bm.fractional_brownian_motion(T, n_steps, H)
            assert len(fBm) == n_steps + 1
            assert fBm[0] == pytest.approx(0.0, abs=1e-10)
            # Check self-similarity property (approximately)
            # Var(X(t)) ~ t^(2H)
            mid_point = n_steps // 2
            var_mid = np.var([bm.fractional_brownian_motion(T, n_steps, H)[1][mid_point]
                             for _ in range(50)])
            expected_var = (T/2)**(2*H)
            assert var_mid == pytest.approx(expected_var, rel=0.5)
class TestRandomWalk:
    """Test Random Walk processes."""
    @pytest.fixture
    def rw(self):
        """Create RandomWalk instance with fixed seed."""
        return RandomWalk(seed=42)
    def test_simple_walk_1d(self, rw):
        """Test 1D random walk."""
        n_steps = 1000
        # Unbiased walk
        position = rw.simple_walk_1d(n_steps, p=0.5)
        assert len(position) == n_steps + 1
        assert position[0] == 0
        # Check that steps are ±1
        steps = np.diff(position)
        assert np.all(np.abs(steps) == 1)
        # Biased walk
        position_biased = rw.simple_walk_1d(n_steps, p=0.7)
        # Should tend to move right
        assert np.mean(np.diff(position_biased)) > 0
    def test_simple_walk_2d(self, rw):
        """Test 2D random walk."""
        n_steps = 1000
        x, y = rw.simple_walk_2d(n_steps)
        assert len(x) == n_steps + 1
        assert len(y) == n_steps + 1
        assert x[0] == 0 and y[0] == 0
        # Check step sizes (should be unit steps)
        step_sizes = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        assert np.allclose(step_sizes, 1.0, rtol=1e-10)
        # Distance from origin should grow as sqrt(n)
        final_distance = np.sqrt(x[-1]**2 + y[-1]**2)
        expected_distance = np.sqrt(n_steps)
        assert final_distance < 3 * expected_distance  # Within 3 std devs
    def test_levy_walk(self, rw):
        """Test Lévy walk."""
        n_steps = 500
        # Test Gaussian case (alpha=2)
        position_gauss = rw.levy_walk(n_steps, alpha=2.0)
        assert len(position_gauss) == n_steps + 1
        assert position_gauss[0] == 0
        # Test heavy-tailed case
        position_levy = rw.levy_walk(n_steps, alpha=1.5)
        assert len(position_levy) == n_steps + 1
        # Lévy walks should have larger jumps
        steps_gauss = np.diff(position_gauss)
        steps_levy = np.diff(position_levy)
        # Check for heavy tails (kurtosis should be higher for Lévy)
        from scipy import stats
        kurt_gauss = stats.kurtosis(steps_gauss)
        kurt_levy = stats.kurtosis(steps_levy)
        assert kurt_levy > kurt_gauss or np.max(np.abs(steps_levy)) > np.max(np.abs(steps_gauss))
    def test_self_avoiding_walk(self, rw):
        """Test self-avoiding walk."""
        n_steps = 50  # Smaller for self-avoiding
        x, y = rw.self_avoiding_walk_2d(n_steps)
        assert len(x) <= n_steps + 1  # May be shorter if trapped
        assert x[0] == 0 and y[0] == 0
        # Check self-avoiding property
        positions = list(zip(x, y))
        assert len(positions) == len(set(positions))  # No repeated positions
class TestStochasticDifferentialEquation:
    """Test SDE solvers."""
    @pytest.fixture
    def sde(self):
        """Create SDE solver with fixed seed."""
        return StochasticDifferentialEquation(seed=42)
    def test_euler_maruyama(self, sde):
        """Test Euler-Maruyama method."""
        # Define simple SDE: dX = -X dt + 0.1 dW (OU process)
        drift = lambda x, t: -x
        diffusion = lambda x, t: 0.1
        x0, T, n_steps = 1.0, 5.0, 1000
        t, X = sde.euler_maruyama(drift, diffusion, x0, T, n_steps)
        assert len(X) == n_steps + 1
        assert X[0] == x0
        # Solution should decay toward 0 (mean-reverting)
        assert abs(X[-1]) < abs(x0)
        # Check stability
        assert np.all(np.isfinite(X))
        assert np.max(np.abs(X)) < 10  # Should be bounded
    def test_milstein(self, sde):
        """Test Milstein method."""
        # Linear SDE with state-dependent diffusion
        drift = lambda x, t: -0.5 * x
        diffusion = lambda x, t: 0.3 * np.sqrt(abs(x) + 1)
        diffusion_prime = lambda x, t: 0.3 * 0.5 / np.sqrt(abs(x) + 1)
        x0, T, n_steps = 1.0, 2.0, 500
        t, X = sde.milstein(drift, diffusion, diffusion_prime, x0, T, n_steps)
        assert len(X) == n_steps + 1
        assert X[0] == x0
        assert np.all(np.isfinite(X))
        # Compare with Euler-Maruyama (Milstein should be more accurate)
        t_em, X_em = sde.euler_maruyama(drift, diffusion, x0, T, n_steps)
        # Both should be stable
        assert np.all(np.isfinite(X_em))
    def test_strong_convergence(self, sde):
        """Test strong convergence of SDE methods."""
        # Use simple SDE with known solution properties
        drift = lambda x, t: 0.0
        diffusion = lambda x, t: 1.0
        x0, T = 0.0, 1.0
        # Compare different step sizes
        errors = []
        step_sizes = [100, 200, 400, 800]
        for n_steps in step_sizes:
            # Run multiple simulations
            n_sims = 20
            final_values = []
            for _ in range(n_sims):
                _, X = sde.euler_maruyama(drift, diffusion, x0, T, n_steps)
                final_values.append(X[-1])
            # Theoretical: X(T) ~ N(0, T)
            error = np.abs(np.mean(final_values) - 0.0)
            errors.append(error)
        # Errors should decrease with smaller step size
        assert errors[-1] < errors[0]
class TestOrnsteinUhlenbeck:
    """Test Ornstein-Uhlenbeck process."""
    def test_mean_reversion(self):
        """Test mean-reverting property."""
        ou = OrnsteinUhlenbeck(theta=2.0, mu=1.0, sigma=0.3, seed=42)
        # Start far from mean
        x0 = 5.0
        T, n_steps = 10.0, 1000
        t, X = ou.generate_path(T, n_steps, x0)
        assert len(X) == n_steps + 1
        assert X[0] == x0
        # Should revert toward mu
        final_mean = np.mean(X[n_steps//2:])
        assert abs(final_mean - ou.mu) < abs(x0 - ou.mu)
        assert abs(final_mean - ou.mu) < 0.5
    def test_stationary_distribution(self):
        """Test stationary distribution properties."""
        ou = OrnsteinUhlenbeck(theta=1.0, mu=0.0, sigma=1.0, seed=42)
        # Generate long path
        T, n_steps = 100.0, 10000
        t, X = ou.generate_path(T, n_steps, x0=0.0)
        # After transient, should match stationary distribution
        X_stationary = X[n_steps//2:]
        mean_theory, var_theory = ou.stationary_distribution()
        assert np.mean(X_stationary) == pytest.approx(mean_theory, abs=0.1)
        assert np.var(X_stationary) == pytest.approx(var_theory, rel=0.2)
class TestJumpDiffusion:
    """Test Jump-Diffusion process."""
    def test_jump_diffusion(self):
        """Test jump-diffusion process generation."""
        jd = JumpDiffusion(drift=0.05, volatility=0.2,
                          jump_rate=0.5, jump_mean=0.0, jump_std=0.1,
                          seed=42)
        S0 = 100.0
        T, n_steps = 5.0, 1000
        t, S = jd.generate_path(T, n_steps, S0)
        assert len(S) == n_steps + 1
        assert S[0] == S0
        assert np.all(S > 0)  # Should remain positive
        # Check for jumps (large changes)
        returns = np.diff(np.log(S))
        large_moves = np.abs(returns) > 3 * np.std(returns)
        # Should have some jumps
        n_jumps = np.sum(large_moves)
        expected_jumps = jd.jump_rate * T
        assert n_jumps > 0
        # Rough check (Poisson variability)
        assert n_jumps < 5 * expected_jumps
    def test_no_jumps_case(self):
        """Test jump-diffusion with zero jump rate."""
        jd = JumpDiffusion(drift=0.05, volatility=0.2,
                          jump_rate=0.0, jump_mean=0.0, jump_std=0.1,
                          seed=42)
        S0 = 100.0
        T, n_steps = 2.0, 500
        t, S = jd.generate_path(T, n_steps, S0)
        # Should behave like geometric Brownian motion
        log_returns = np.diff(np.log(S))
        # No extreme outliers expected
        assert np.max(np.abs(log_returns)) < 5 * np.std(log_returns)
@pytest.mark.integration
class TestIntegration:
    """Integration tests for stochastic processes."""
    def test_monte_carlo_option_pricing(self):
        """Test option pricing using geometric Brownian motion."""
        # European call option parameters
        S0 = 100.0  # Initial stock price
        K = 105.0   # Strike price
        T = 1.0     # Time to maturity
        r = 0.05    # Risk-free rate
        sigma = 0.2 # Volatility
        # Monte Carlo simulation
        n_paths = 1000
        n_steps = 252  # Daily steps
        payoffs = []
        for i in range(n_paths):
            bm = BrownianMotion(drift=r, volatility=sigma, seed=i)
            _, S = bm.generate_geometric_path(T, n_steps, S0)
            payoff = max(S[-1] - K, 0)
            payoffs.append(payoff)
        # Discounted expected payoff
        option_price = np.exp(-r * T) * np.mean(payoffs)
        # Black-Scholes formula for comparison
        from scipy.stats import norm
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        bs_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        # Monte Carlo should be close to Black-Scholes
        assert option_price == pytest.approx(bs_price, rel=0.1)
    def test_ensemble_statistics(self):
        """Test ensemble statistics of stochastic processes."""
        # Generate ensemble of Brownian paths
        n_paths = 100
        T, n_steps = 1.0, 100
        paths = []
        for i in range(n_paths):
            bm = BrownianMotion(drift=0.0, volatility=1.0, seed=i)
            t, W = bm.generate_path(T, n_steps)
            paths.append(W)
        paths = np.array(paths)
        # Check ensemble statistics at each time
        mean_path = np.mean(paths, axis=0)
        var_path = np.var(paths, axis=0)
        # Mean should be close to 0
        assert np.all(np.abs(mean_path) < 0.2)
        # Variance should grow linearly with time
        expected_var = t
        assert np.allclose(var_path, expected_var, rtol=0.3)
if __name__ == '__main__':
    pytest.main([__file__, '-v'])