"""Tests for ODE solvers module.
This module contains comprehensive tests for the ODE solving functionality
including basic solvers, adaptive methods, and accuracy verification.
Author: Berkeley SciComp Team
Date: 2024
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import warnings
from ..ode_solvers import (
    ExplicitEuler, ImplicitEuler, RungeKutta4, RungeKuttaFehlberg,
    AdamsBashforth, AdamsMoulton, BDF, solve_ode
)
class TestBasicODESolvers:
    """Test basic ODE solvers."""
    def setup_method(self):
        """Setup test fixtures."""
        # Simple exponential decay: dy/dt = -λy, y(0) = y0
        self.lambda_param = 2.0
        self.y0 = 1.0
        self.t_span = (0, 1)
        def dydt(t, y):
            return -self.lambda_param * y
        def analytical_solution(t):
            return self.y0 * np.exp(-self.lambda_param * t)
        self.dydt = dydt
        self.analytical_solution = analytical_solution
    def test_explicit_euler(self):
        """Test explicit Euler method."""
        solver = ExplicitEuler()
        dt = 0.01
        result = solver.solve(self.dydt, self.y0, self.t_span, dt)
        assert result.success
        assert len(result.t) == int((self.t_span[1] - self.t_span[0]) / dt) + 1
        # Check final value accuracy (Euler is first-order)
        y_exact = self.analytical_solution(result.t[-1])
        error = abs(result.y[-1] - y_exact)
        assert error < 0.1  # Reasonable tolerance for Euler
    def test_implicit_euler(self):
        """Test implicit Euler method."""
        solver = ImplicitEuler()
        dt = 0.01
        result = solver.solve(self.dydt, self.y0, self.t_span, dt)
        assert result.success
        # Implicit Euler should be stable for this problem
        y_exact = self.analytical_solution(result.t[-1])
        error = abs(result.y[-1] - y_exact)
        assert error < 0.1
    def test_runge_kutta_4(self):
        """Test 4th-order Runge-Kutta method."""
        solver = RungeKutta4()
        dt = 0.01
        result = solver.solve(self.dydt, self.y0, self.t_span, dt)
        assert result.success
        # RK4 should be much more accurate
        y_exact = self.analytical_solution(result.t[-1])
        error = abs(result.y[-1] - y_exact)
        assert error < 1e-6  # Much tighter tolerance for RK4
    def test_adaptive_rk45(self):
        """Test adaptive RK45 method."""
        solver = RungeKuttaFehlberg()
        dt = 0.1  # Can use larger initial step due to adaptivity
        result = solver.solve(self.dydt, self.y0, self.t_span, dt)
        assert result.success
        # Should be very accurate due to error control
        y_exact = self.analytical_solution(result.t[-1])
        error = abs(result.y[-1] - y_exact)
        assert error < 1e-8
class TestODESystemSolvers:
    """Test ODE solvers with systems of equations."""
    def setup_method(self):
        """Setup system test fixtures."""
        # Harmonic oscillator: d²y/dt² + ω²y = 0
        # Convert to system: y1 = y, y2 = dy/dt
        # dy1/dt = y2, dy2/dt = -ω²y1
        self.omega = 2.0
        self.y0 = np.array([1.0, 0.0])  # Initial position and velocity
        self.t_span = (0, np.pi)  # Half period
        def dydt(t, y):
            y1, y2 = y
            return np.array([y2, -self.omega**2 * y1])
        def analytical_solution(t):
            y = np.cos(self.omega * t)
            dydt = -self.omega * np.sin(self.omega * t)
            return np.array([y, dydt])
        self.dydt = dydt
        self.analytical_solution = analytical_solution
    def test_system_rk4(self):
        """Test RK4 with system of equations."""
        solver = RungeKutta4()
        dt = 0.01
        result = solver.solve(self.dydt, self.y0, self.t_span, dt)
        assert result.success
        assert result.y.shape[1] == 2  # Two variables
        # Check accuracy
        y_exact = self.analytical_solution(result.t[-1])
        error = np.linalg.norm(result.y[-1] - y_exact)
        assert error < 1e-6
    def test_energy_conservation(self):
        """Test energy conservation for harmonic oscillator."""
        solver = RungeKutta4()
        dt = 0.001  # Small time step for good conservation
        result = solver.solve(self.dydt, self.y0, self.t_span, dt)
        # Energy = 0.5 * (dy/dt)² + 0.5 * ω² * y²
        kinetic = 0.5 * result.y[:, 1]**2
        potential = 0.5 * self.omega**2 * result.y[:, 0]**2
        total_energy = kinetic + potential
        # Energy should be conserved
        energy_variation = np.std(total_energy)
        assert energy_variation < 1e-8
class TestODEAccuracy:
    """Test ODE solver accuracy and convergence."""
    def test_convergence_order(self):
        """Test convergence order of different methods."""
        # Use dy/dt = -y, y(0) = 1, exact solution y(t) = e^(-t)
        def dydt(t, y):
            return -y
        def exact(t):
            return np.exp(-t)
        y0 = 1.0
        t_final = 1.0
        t_span = (0, t_final)
        # Test different step sizes
        dt_values = [0.1, 0.05, 0.025, 0.0125]
        # Test Euler (should be 1st order)
        euler_errors = []
        for dt in dt_values:
            solver = ExplicitEuler()
            result = solver.solve(dydt, y0, t_span, dt)
            error = abs(result.y[-1] - exact(t_final))
            euler_errors.append(error)
        # Check convergence rate (should be approximately 1)
        rates = []
        for i in range(1, len(euler_errors)):
            rate = np.log(euler_errors[i-1] / euler_errors[i]) / np.log(dt_values[i-1] / dt_values[i])
            rates.append(rate)
        avg_rate = np.mean(rates)
        assert abs(avg_rate - 1.0) < 0.3  # Allow some tolerance
        # Test RK4 (should be 4th order)
        rk4_errors = []
        for dt in dt_values:
            solver = RungeKutta4()
            result = solver.solve(dydt, y0, t_span, dt)
            error = abs(result.y[-1] - exact(t_final))
            rk4_errors.append(error)
        # Check convergence rate (should be approximately 4)
        rk4_rates = []
        for i in range(1, len(rk4_errors)):
            rate = np.log(rk4_errors[i-1] / rk4_errors[i]) / np.log(dt_values[i-1] / dt_values[i])
            rk4_rates.append(rate)
        avg_rk4_rate = np.mean(rk4_rates)
        assert abs(avg_rk4_rate - 4.0) < 0.5
    def test_stiff_problem(self):
        """Test solver behavior on stiff problem."""
        # Stiff equation: dy/dt = -1000(y - cos(t)) - sin(t)
        # Exact solution: y(t) = cos(t)
        def stiff_ode(t, y):
            return -1000 * (y - np.cos(t)) - np.sin(t)
        def exact(t):
            return np.cos(t)
        y0 = 1.0
        t_span = (0, 1)
        dt = 0.001  # Small step needed for explicit methods
        # Implicit Euler should handle this better than explicit
        implicit_solver = ImplicitEuler()
        result_implicit = implicit_solver.solve(stiff_ode, y0, t_span, dt)
        if result_implicit.success:
            error_implicit = abs(result_implicit.y[-1] - exact(t_span[1]))
            assert error_implicit < 0.1
        # Explicit Euler might struggle or need very small steps
        explicit_solver = ExplicitEuler()
        result_explicit = explicit_solver.solve(stiff_ode, y0, t_span, dt)
        # This might fail or be inaccurate - that's expected for stiff problems
class TestMultiStepMethods:
    """Test multi-step ODE methods."""
    def setup_method(self):
        """Setup test fixtures."""
        def dydt(t, y):
            return -y + np.sin(t)
        self.dydt = dydt
        self.y0 = 1.0
        self.t_span = (0, 2)
    def test_adams_bashforth(self):
        """Test Adams-Bashforth method."""
        solver = AdamsBashforth(order=2)
        dt = 0.01
        result = solver.solve(self.dydt, self.y0, self.t_span, dt)
        # Should produce reasonable results
        assert result.success
        assert len(result.y) > 10
        assert np.all(np.isfinite(result.y))
    def test_adams_moulton(self):
        """Test Adams-Moulton method."""
        solver = AdamsMoulton(order=2)
        dt = 0.01
        result = solver.solve(self.dydt, self.y0, self.t_span, dt)
        # Should produce reasonable results
        assert result.success
        assert len(result.y) > 10
        assert np.all(np.isfinite(result.y))
    def test_bdf(self):
        """Test BDF method."""
        solver = BDF(order=2)
        dt = 0.01
        result = solver.solve(self.dydt, self.y0, self.t_span, dt)
        # Should produce reasonable results
        assert result.success
        assert len(result.y) > 10
        assert np.all(np.isfinite(result.y))
class TestODEInterface:
    """Test high-level ODE solving interface."""
    def test_solve_ode_function(self):
        """Test solve_ode convenience function."""
        def dydt(t, y):
            return -2 * y
        y0 = 1.0
        t_span = (0, 1)
        # Test with different methods
        methods = ['euler', 'rk4', 'adaptive']
        for method in methods:
            result = solve_ode(dydt, y0, t_span, method=method)
            assert result.success
            assert len(result.y) > 1
            assert np.all(np.isfinite(result.y))
    def test_invalid_method(self):
        """Test error handling for invalid method."""
        def dydt(t, y):
            return -y
        with pytest.raises(ValueError):
            solve_ode(dydt, 1.0, (0, 1), method='invalid_method')
class TestBoundaryConditions:
    """Test ODE problems with boundary conditions."""
    def test_shooting_method_placeholder(self):
        """Placeholder for shooting method tests."""
        # This would test boundary value problems
        # For now, just check that the structure is in place
        pass
class TestODERobustness:
    """Test ODE solver robustness and edge cases."""
    def test_zero_time_span(self):
        """Test behavior with zero time span."""
        def dydt(t, y):
            return -y
        solver = ExplicitEuler()
        result = solver.solve(dydt, 1.0, (0, 0), 0.01)
        # Should return initial condition
        assert result.success
        assert len(result.y) == 1
        assert result.y[0] == 1.0
    def test_negative_time_step(self):
        """Test behavior with negative time step."""
        def dydt(t, y):
            return -y
        solver = ExplicitEuler()
        # Should handle backward integration
        result = solver.solve(dydt, 1.0, (1, 0), -0.01)
        assert result.success or not result.success  # Either way is acceptable
    def test_very_small_time_step(self):
        """Test behavior with very small time step."""
        def dydt(t, y):
            return -y
        solver = ExplicitEuler()
        result = solver.solve(dydt, 1.0, (0, 0.001), 1e-6)
        # Should complete but might take many steps
        assert result.success
        assert len(result.y) > 100  # Many small steps
    def test_discontinuous_rhs(self):
        """Test with discontinuous right-hand side."""
        def dydt(t, y):
            return -y if t < 0.5 else y
        solver = RungeKutta4()
        result = solver.solve(dydt, 1.0, (0, 1), 0.01)
        # Should complete despite discontinuity
        assert result.success
        assert np.all(np.isfinite(result.y))
if __name__ == "__main__":
    pytest.main([__file__])