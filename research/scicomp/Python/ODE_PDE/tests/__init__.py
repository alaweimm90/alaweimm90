"""Test suite for ODE_PDE package.
This package contains comprehensive tests for the ODE_PDE module including:
- ODE solvers (explicit, implicit, adaptive)
- PDE solvers (finite difference, spectral, FEM)
- Nonlinear solvers (Newton, continuation)
- Adaptive methods and error control
- Stability analysis
- Utility functions
To run all tests:
    pytest tests/
To run specific test modules:
    pytest tests/test_ode_solvers.py
    pytest tests/test_pde_solvers.py
    pytest tests/test_nonlinear_solvers.py
Author: Berkeley SciComp Team
Date: 2024
"""
__version__ = "1.0.0"
__author__ = "Berkeley SciComp Team"
# Test configuration
pytest_plugins = []
# Test utilities that can be imported by test modules
import numpy as np
def assert_solution_accuracy(numerical, analytical, tolerance=1e-6):
    """Assert that numerical solution matches analytical within tolerance."""
    error = np.max(np.abs(numerical - analytical))
    assert error < tolerance, f"Solution error {error:.2e} exceeds tolerance {tolerance:.2e}"
def assert_convergence_order(errors, step_sizes, expected_order, tolerance=0.5):
    """Assert that convergence order matches expected value."""
    if len(errors) < 2:
        raise ValueError("Need at least 2 error values to compute convergence order")
    # Compute convergence rates
    rates = []
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i-1] > 0:
            rate = np.log(errors[i-1] / errors[i]) / np.log(step_sizes[i-1] / step_sizes[i])
            rates.append(rate)
    if not rates:
        raise ValueError("Could not compute any convergence rates")
    avg_rate = np.mean(rates)
    assert abs(avg_rate - expected_order) < tolerance, \
        f"Convergence order {avg_rate:.2f} differs from expected {expected_order} by more than {tolerance}"
def create_test_mesh_1d(domain, n_points):
    """Create a simple 1D test mesh."""
    return np.linspace(domain[0], domain[1], n_points)
def create_test_function_1d():
    """Create a test function and its derivatives for 1D problems."""
    def f(x):
        return np.sin(np.pi * x)
    def df_dx(x):
        return np.pi * np.cos(np.pi * x)
    def d2f_dx2(x):
        return -np.pi**2 * np.sin(np.pi * x)
    return f, df_dx, d2f_dx2
def create_test_ode_system():
    """Create a test ODE system (harmonic oscillator)."""
    omega = 2.0
    def dydt(t, y):
        return np.array([y[1], -omega**2 * y[0]])
    def jacobian(t, y):
        return np.array([[0, 1], [-omega**2, 0]])
    def analytical_solution(t, y0, v0):
        c1 = y0
        c2 = v0 / omega
        return np.array([
            c1 * np.cos(omega * t) + c2 * np.sin(omega * t),
            -c1 * omega * np.sin(omega * t) + c2 * omega * np.cos(omega * t)
        ])
    return dydt, jacobian, analytical_solution, omega
def create_exponential_decay_ode():
    """Create exponential decay ODE for testing."""
    lambda_param = 2.0
    def dydt(t, y):
        return -lambda_param * y
    def analytical_solution(t, y0):
        return y0 * np.exp(-lambda_param * t)
    return dydt, analytical_solution, lambda_param
# Common test parameters
DEFAULT_TOLERANCE = 1e-6
DEFAULT_TIME_SPAN = (0, 1)
DEFAULT_DOMAIN_1D = (0, 1)
DEFAULT_GRID_SIZE = 51
# Berkeley colors for test plotting (if needed)
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'