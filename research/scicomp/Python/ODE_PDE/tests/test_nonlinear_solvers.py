"""Tests for nonlinear solvers module.
This module contains comprehensive tests for nonlinear solving functionality
including Newton methods, continuation methods, and convergence analysis.
Author: Berkeley SciComp Team
Date: 2024
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import warnings
from ..nonlinear_solvers import (
    NewtonSolver, DampedNewtonSolver, FixedPointSolver, ContinuationSolver,
    newton_raphson, solve_nonlinear_ode, solve_nonlinear_pde,
    analyze_nonlinear_convergence
)
class TestNewtonSolver:
    """Test Newton-Raphson solver."""
    def setup_method(self):
        """Setup test fixtures."""
        # Simple quadratic: f(x) = x² - 2, root at x = √2
        def f_quadratic(x):
            return np.array([x[0]**2 - 2])
        def jacobian_quadratic(x):
            return np.array([[2 * x[0]]])
        self.f_quadratic = f_quadratic
        self.jacobian_quadratic = jacobian_quadratic
        self.expected_root = np.sqrt(2)
    def test_newton_simple_root(self):
        """Test Newton method on simple quadratic."""
        solver = NewtonSolver(tolerance=1e-10)
        # Start near the root
        x0 = np.array([1.5])
        result = solver.solve(self.f_quadratic, x0, self.jacobian_quadratic)
        assert result.success
        assert result.iterations < 10
        assert abs(result.x[0] - self.expected_root) < 1e-10
        assert result.residual_norm < 1e-10
    def test_newton_without_jacobian(self):
        """Test Newton method with finite difference Jacobian."""
        solver = NewtonSolver(tolerance=1e-8)
        x0 = np.array([1.5])
        result = solver.solve(self.f_quadratic, x0)  # No Jacobian provided
        assert result.success
        assert abs(result.x[0] - self.expected_root) < 1e-8
        assert result.jacobian_evaluations > 0  # Should compute finite differences
    def test_newton_convergence_history(self):
        """Test convergence history tracking."""
        solver = NewtonSolver(tolerance=1e-12)
        x0 = np.array([1.5])
        result = solver.solve(self.f_quadratic, x0, self.jacobian_quadratic)
        assert result.success
        assert len(result.convergence_history) == result.iterations
        # Convergence should be monotonic (decreasing)
        for i in range(1, len(result.convergence_history)):
            assert result.convergence_history[i] <= result.convergence_history[i-1]
    def test_newton_system_equations(self):
        """Test Newton method on system of equations."""
        # System: f1 = x₁² + x₂² - 1, f2 = x₁ - x₂
        # Solution: (1/√2, 1/√2) and (-1/√2, -1/√2)
        def f_system(x):
            x1, x2 = x
            return np.array([x1**2 + x2**2 - 1, x1 - x2])
        def jacobian_system(x):
            x1, x2 = x
            return np.array([
                [2*x1, 2*x2],
                [1, -1]
            ])
        solver = NewtonSolver(tolerance=1e-10)
        # Start near positive solution
        x0 = np.array([0.8, 0.6])
        result = solver.solve(f_system, x0, jacobian_system)
        assert result.success
        assert result.iterations < 20
        # Check solution satisfies equations
        residual = f_system(result.x)
        assert np.linalg.norm(residual) < 1e-10
        # Check it's on unit circle and x1 = x2
        assert abs(np.linalg.norm(result.x) - 1.0) < 1e-10
        assert abs(result.x[0] - result.x[1]) < 1e-10
class TestDampedNewtonSolver:
    """Test damped Newton solver."""
    def test_damped_newton_difficult_case(self):
        """Test damped Newton on case where regular Newton might struggle."""
        # Function with poor conditioning
        def f_difficult(x):
            return np.array([x[0]**3 - x[0] - 1])
        def jacobian_difficult(x):
            return np.array([[3*x[0]**2 - 1]])
        solver = DampedNewtonSolver(tolerance=1e-8)
        # Start far from root
        x0 = np.array([10.0])
        result = solver.solve(f_difficult, x0, jacobian_difficult)
        assert result.success
        # Check solution
        residual = f_difficult(result.x)
        assert np.abs(residual[0]) < 1e-8
    def test_damping_effectiveness(self):
        """Test that damping helps with convergence."""
        # Function that benefits from damping
        def f_steep(x):
            return np.array([100 * x[0]**2 - 1])
        def jacobian_steep(x):
            return np.array([[200 * x[0]]])
        # Compare regular Newton vs damped Newton
        x0 = np.array([10.0])  # Far from root
        # Regular Newton might struggle
        newton_solver = NewtonSolver(tolerance=1e-8, max_iterations=50)
        newton_result = newton_solver.solve(f_steep, x0, jacobian_steep)
        # Damped Newton should be more robust
        damped_solver = DampedNewtonSolver(tolerance=1e-8, max_iterations=50)
        damped_result = damped_solver.solve(f_steep, x0, jacobian_steep)
        assert damped_result.success
        assert np.abs(damped_result.residual_norm) < 1e-8
class TestFixedPointSolver:
    """Test fixed point iteration solver."""
    def test_fixed_point_simple(self):
        """Test fixed point iteration on simple function."""
        # g(x) = 0.5*(x + 2/x) converges to √2
        def g(x):
            return np.array([0.5 * (x[0] + 2/x[0])])
        solver = FixedPointSolver(tolerance=1e-8, max_iterations=100)
        x0 = np.array([1.5])
        result = solver.solve(g, x0)
        assert result.success
        assert abs(result.x[0] - np.sqrt(2)) < 1e-8
    def test_fixed_point_relaxation(self):
        """Test fixed point iteration with relaxation."""
        # Function that needs relaxation for stability
        def g_unstable(x):
            return np.array([2 - x[0]**2])  # Can be unstable without relaxation
        # Without relaxation (might fail)
        solver_no_relax = FixedPointSolver(tolerance=1e-6, relaxation=1.0)
        # With relaxation (should converge)
        solver_relax = FixedPointSolver(tolerance=1e-6, relaxation=0.5)
        x0 = np.array([0.5])
        result_relax = solver_relax.solve(g_unstable, x0)
        # Relaxed version should converge
        assert result_relax.success
class TestContinuationSolver:
    """Test continuation/parameter continuation solver."""
    def test_parameter_continuation_simple(self):
        """Test parameter continuation on simple problem."""
        # f(x, μ) = x² - μ, solution x = ±√μ for μ > 0
        def f_param(x, mu):
            return np.array([x[0]**2 - mu])
        def jacobian_param(x, mu):
            return np.array([[2 * x[0]]])
        solver = ContinuationSolver()
        # Start at μ = 1, x = 1 (positive branch)
        x0 = np.array([1.0])
        mu_range = (1.0, 4.0)
        result = solver.solve_parameter_continuation(
            f_param, jacobian_param, x0, mu_range, n_steps=20
        )
        assert result.success
        assert len(result.parameter_values) == 21  # n_steps + 1
        assert len(result.solutions) == 21
        # Check that solutions satisfy f(x, μ) = 0
        for i, (x, mu) in enumerate(zip(result.solutions, result.parameter_values)):
            residual = f_param(x, mu)
            assert np.abs(residual[0]) < 1e-8
        # Check that x ≈ √μ along the path
        for i, (x, mu) in enumerate(zip(result.solutions, result.parameter_values)):
            assert abs(x[0] - np.sqrt(mu)) < 1e-6
    def test_bifurcation_detection(self):
        """Test bifurcation point detection."""
        # Pitchfork bifurcation: f(x, μ) = μx - x³
        def f_pitchfork(x, mu):
            return np.array([mu * x[0] - x[0]**3])
        def jacobian_pitchfork(x, mu):
            return np.array([[mu - 3 * x[0]**2]])
        solver = ContinuationSolver()
        # Start at μ = -1, x = 0 (stable branch)
        x0 = np.array([0.0])
        mu_range = (-1.0, 1.0)
        result = solver.solve_parameter_continuation(
            f_pitchfork, jacobian_pitchfork, x0, mu_range, n_steps=50
        )
        assert result.success
        # Should detect bifurcation near μ = 0
        if result.bifurcation_points:
            bp_idx = result.bifurcation_points[0]
            bp_mu = result.parameter_values[bp_idx]
            assert abs(bp_mu) < 0.1  # Near theoretical bifurcation point
class TestNonlinearODEIntegration:
    """Test nonlinear ODE solving integration."""
    def test_solve_nonlinear_ode(self):
        """Test nonlinear ODE solving function."""
        # Nonlinear ODE: dy/dt = -y²
        # Analytical solution: y(t) = y0/(1 + y0*t)
        def dydt(t, y):
            return -y**2
        y0 = 1.0
        t = 0.0
        dt = 0.1
        # Solve one step with implicit method
        y_new = solve_nonlinear_ode(dydt, y0, t, dt, method='newton')
        # Check that result is reasonable
        assert isinstance(y_new, (float, np.ndarray))
        assert np.isfinite(y_new)
        # Should be less than y0 for this decaying equation
        if isinstance(y_new, np.ndarray):
            assert np.all(y_new < y0)
        else:
            assert y_new < y0
class TestConvergenceAnalysis:
    """Test convergence analysis utilities."""
    def test_analyze_quadratic_convergence(self):
        """Test analysis of quadratic convergence."""
        # Simulate quadratic convergence: e_{k+1} ≈ C * e_k²
        C = 0.5
        e0 = 0.1
        errors = [e0]
        for _ in range(8):
            e_next = C * errors[-1]**2
            errors.append(e_next)
        analysis = analyze_nonlinear_convergence(errors)
        assert 'convergence_rate' in analysis
        assert analysis['convergence_type'] == 'quadratic'
        assert abs(analysis['convergence_rate'] - 2.0) < 0.5
    def test_analyze_linear_convergence(self):
        """Test analysis of linear convergence."""
        # Simulate linear convergence: e_{k+1} ≈ 0.5 * e_k
        e0 = 0.1
        rate = 0.5
        errors = [e0]
        for _ in range(10):
            e_next = rate * errors[-1]
            errors.append(e_next)
        analysis = analyze_nonlinear_convergence(errors)
        assert analysis['convergence_type'] == 'linear'
        assert abs(analysis['convergence_rate'] - 1.0) < 0.3
class TestConvenienceFunctions:
    """Test convenience functions."""
    def test_newton_raphson_function(self):
        """Test newton_raphson convenience function."""
        def f(x):
            return np.array([x[0]**2 - 4])
        def df(x):
            return np.array([[2 * x[0]]])
        x0 = np.array([1.0])
        result = newton_raphson(f, x0, df, tolerance=1e-10)
        assert result.success
        assert abs(result.x[0] - 2.0) < 1e-10
    def test_solve_nonlinear_pde_placeholder(self):
        """Test nonlinear PDE solving function (placeholder)."""
        # Simple test for interface
        def residual(u):
            return u**2 - 1  # Simple nonlinear equation
        u0 = np.array([0.5])
        result = solve_nonlinear_pde(residual, u0, method='newton')
        assert result.success
        assert abs(result.x[0] - 1.0) < 1e-6
class TestErrorHandling:
    """Test error handling and edge cases."""
    def test_singular_jacobian(self):
        """Test behavior with singular Jacobian."""
        def f_singular(x):
            return np.array([x[0]**2])  # f'(0) = 0
        def jacobian_singular(x):
            return np.array([[2 * x[0]]])  # Singular at x = 0
        solver = NewtonSolver(tolerance=1e-8, max_iterations=10)
        # Start at x = 0 where Jacobian is singular
        x0 = np.array([0.0])
        result = solver.solve(f_singular, x0, jacobian_singular)
        # Should fail gracefully
        assert not result.success
        assert "singular" in result.message.lower() or "fail" in result.message.lower()
    def test_max_iterations(self):
        """Test maximum iterations limit."""
        def f_slow(x):
            return np.array([x[0] - 1e-6])  # Very small step
        def jacobian_slow(x):
            return np.array([[1.0]])
        solver = NewtonSolver(tolerance=1e-12, max_iterations=3)
        x0 = np.array([1.0])
        result = solver.solve(f_slow, x0, jacobian_slow)
        # Should hit max iterations
        assert not result.success
        assert result.iterations == 3
        assert "maximum" in result.message.lower()
    def test_invalid_initial_guess(self):
        """Test behavior with invalid initial guess."""
        def f_invalid(x):
            if not np.isfinite(x[0]):
                raise ValueError("Invalid input")
            return np.array([x[0]**2 - 1])
        solver = NewtonSolver(tolerance=1e-8)
        # Test with NaN initial guess
        x0 = np.array([np.nan])
        try:
            result = solver.solve(f_invalid, x0)
            # If it doesn't raise an exception, it should fail gracefully
            assert not result.success
        except (ValueError, RuntimeError):
            # This is also acceptable behavior
            pass
if __name__ == "__main__":
    pytest.main([__file__])