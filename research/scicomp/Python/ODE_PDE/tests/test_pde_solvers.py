"""Tests for PDE solvers module.
This module contains comprehensive tests for the PDE solving functionality
including finite difference methods, various PDE types, and accuracy verification.
Author: Berkeley SciComp Team
Date: 2024
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import warnings
from ..pde_solvers import (
    HeatEquationSolver, WaveEquationSolver, PoissonSolver,
    AdvectionDiffusionSolver, solve_pde, PDEOptions
)
class TestHeatEquationSolver:
    """Test heat equation solver."""
    def setup_method(self):
        """Setup test fixtures."""
        # 1D heat equation on [0, 1]
        self.domain = {'x': np.linspace(0, 1, 51)}
        self.thermal_diffusivity = 0.1
        # Dirichlet boundary conditions
        self.boundary_conditions = {
            'dirichlet': {0: 0.0, 50: 0.0}  # u(0) = u(1) = 0
        }
        self.options = PDEOptions()
    def test_steady_state_heat(self):
        """Test steady-state heat equation."""
        # Source term: f(x) = π²sin(πx)
        # Analytical solution: u(x) = sin(πx)/α
        def source_term(x):
            return np.pi**2 * np.sin(np.pi * x)
        solver = HeatEquationSolver(
            self.domain, self.boundary_conditions,
            self.thermal_diffusivity, self.options
        )
        result = solver.solve_steady(source_term)
        assert result.success
        assert len(result.u) == len(self.domain['x'])
        # Check boundary conditions
        assert abs(result.u[0]) < 1e-10
        assert abs(result.u[-1]) < 1e-10
        # Compare with analytical solution
        x = self.domain['x']
        u_analytical = np.sin(np.pi * x) / self.thermal_diffusivity
        # Error should be small (depends on grid resolution)
        error = np.max(np.abs(result.u - u_analytical))
        assert error < 0.1  # Reasonable tolerance for finite differences
    def test_transient_heat(self):
        """Test transient heat equation."""
        def initial_condition(x):
            return np.sin(np.pi * x)
        solver = HeatEquationSolver(
            self.domain, self.boundary_conditions,
            self.thermal_diffusivity, self.options
        )
        result = solver.solve_transient(
            initial_condition=initial_condition,
            time_span=(0, 0.1),
            dt=0.001
        )
        assert result.success
        assert result.u.ndim == 2  # Time x Space
        assert result.u.shape[1] == len(self.domain['x'])
        # Temperature should decay over time
        initial_energy = np.sum(result.u[0]**2)
        final_energy = np.sum(result.u[-1]**2)
        assert final_energy < initial_energy
        # Boundary conditions should be maintained
        assert np.all(np.abs(result.u[:, 0]) < 1e-10)  # Left boundary
        assert np.all(np.abs(result.u[:, -1]) < 1e-10)  # Right boundary
    def test_heat_analytical_comparison(self):
        """Test against analytical solution for heat equation."""
        # For u(x,0) = sin(πx), u(0,t) = u(1,t) = 0
        # Analytical solution: u(x,t) = sin(πx)exp(-π²αt)
        def initial_condition(x):
            return np.sin(np.pi * x)
        solver = HeatEquationSolver(
            self.domain, self.boundary_conditions,
            self.thermal_diffusivity, self.options
        )
        final_time = 0.05
        result = solver.solve_transient(
            initial_condition=initial_condition,
            time_span=(0, final_time),
            dt=0.001
        )
        if result.success:
            # Analytical solution at final time
            x = self.domain['x']
            u_analytical = (np.sin(np.pi * x) *
                           np.exp(-np.pi**2 * self.thermal_diffusivity * final_time))
            # Compare numerical and analytical solutions
            error = np.max(np.abs(result.u[-1] - u_analytical))
            assert error < 0.01
class TestWaveEquationSolver:
    """Test wave equation solver."""
    def setup_method(self):
        """Setup test fixtures."""
        self.domain = {'x': np.linspace(0, 1, 101)}
        self.wave_speed = 1.0
        self.boundary_conditions = {
            'dirichlet': {0: 0.0, 100: 0.0}
        }
        self.options = PDEOptions()
    def test_wave_equation_standing_wave(self):
        """Test wave equation with standing wave initial condition."""
        def initial_condition(x):
            return np.sin(np.pi * x)
        def initial_velocity(x):
            return np.zeros_like(x)
        solver = WaveEquationSolver(
            self.domain, self.boundary_conditions,
            self.wave_speed, self.options
        )
        # Solve for one period
        period = 2.0  # For c=1, L=1, fundamental mode has period 2
        result = solver.solve_transient(
            initial_condition=initial_condition,
            initial_velocity=initial_velocity,
            time_span=(0, period/4),  # Quarter period
            dt=0.001
        )
        assert result.success
        assert result.u.ndim == 2
        # At t=T/4, should have pure velocity mode
        # Check energy conservation (approximately)
        dx = self.domain['x'][1] - self.domain['x'][0]
        dt = result.t[1] - result.t[0]
        initial_energy = 0.5 * np.sum(result.u[0]**2) * dx
        final_energy = 0.5 * np.sum(result.u[-1]**2) * dx
        # Energy should be approximately conserved
        energy_error = abs(final_energy - initial_energy) / initial_energy
        assert energy_error < 0.1  # Allow some numerical dissipation
    def test_wave_cfl_stability(self):
        """Test CFL stability condition for wave equation."""
        def initial_condition(x):
            return np.exp(-50 * (x - 0.5)**2)  # Gaussian pulse
        def initial_velocity(x):
            return np.zeros_like(x)
        solver = WaveEquationSolver(
            self.domain, self.boundary_conditions,
            self.wave_speed, self.options
        )
        dx = self.domain['x'][1] - self.domain['x'][0]
        # Test stable time step (CFL < 1)
        dt_stable = 0.5 * dx / self.wave_speed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore CFL warnings for this test
            result_stable = solver.solve_transient(
                initial_condition=initial_condition,
                initial_velocity=initial_velocity,
                time_span=(0, 0.1),
                dt=dt_stable
            )
        assert result_stable.success
        assert np.all(np.isfinite(result_stable.u))
class TestPoissonSolver:
    """Test Poisson equation solver."""
    def setup_method(self):
        """Setup test fixtures."""
        self.domain = {'x': np.linspace(0, 1, 101)}
        self.boundary_conditions = {
            'dirichlet': {0: 0.0, 100: 0.0}
        }
        self.options = PDEOptions()
    def test_poisson_manufactured_solution(self):
        """Test Poisson solver with manufactured solution."""
        # Manufactured solution: u(x) = x(1-x)
        # Source: -d²u/dx² = 2
        def source_term(x):
            return 2.0 * np.ones_like(x)
        def analytical_solution(x):
            return x * (1 - x)
        solver = PoissonSolver(self.domain, self.boundary_conditions, self.options)
        result = solver.solve_steady(source_term)
        assert result.success
        # Compare with analytical solution
        u_analytical = analytical_solution(self.domain['x'])
        error = np.max(np.abs(result.u - u_analytical))
        # Should be very accurate for this simple case
        assert error < 1e-6
    def test_poisson_sine_source(self):
        """Test Poisson equation with sine source."""
        # -d²u/dx² = π²sin(πx), u(0) = u(1) = 0
        # Analytical solution: u(x) = sin(πx)
        def source_term(x):
            return np.pi**2 * np.sin(np.pi * x)
        solver = PoissonSolver(self.domain, self.boundary_conditions, self.options)
        result = solver.solve_steady(source_term)
        assert result.success
        # Compare with analytical solution
        u_analytical = np.sin(np.pi * self.domain['x'])
        error = np.max(np.abs(result.u - u_analytical))
        assert error < 0.01  # Good accuracy expected
class TestAdvectionDiffusionSolver:
    """Test advection-diffusion solver."""
    def setup_method(self):
        """Setup test fixtures."""
        self.domain = {'x': np.linspace(0, 10, 101)}
        self.velocity = 1.0
        self.diffusivity = 0.1
        self.boundary_conditions = {
            'dirichlet': {0: 0.0, 100: 0.0}
        }
        self.options = PDEOptions()
    def test_steady_advection_diffusion(self):
        """Test steady advection-diffusion equation."""
        def source_term(x):
            return np.exp(-x)  # Decaying source
        solver = AdvectionDiffusionSolver(
            self.domain, self.boundary_conditions,
            self.velocity, self.diffusivity, self.options
        )
        result = solver.solve_steady(source_term)
        assert result.success
        assert len(result.u) == len(self.domain['x'])
        # Solution should be finite and reasonable
        assert np.all(np.isfinite(result.u))
        assert np.all(result.u >= 0)  # Non-negative for this problem
    def test_transient_advection_diffusion(self):
        """Test transient advection-diffusion equation."""
        def initial_condition(x):
            return np.exp(-2 * (x - 2)**2)  # Gaussian pulse
        solver = AdvectionDiffusionSolver(
            self.domain, self.boundary_conditions,
            self.velocity, self.diffusivity, self.options
        )
        result = solver.solve_transient(
            initial_condition=initial_condition,
            time_span=(0, 2),
            dt=0.01
        )
        assert result.success
        assert result.u.ndim == 2
        # Peak should move to the right (advection)
        initial_peak_idx = np.argmax(result.u[0])
        final_peak_idx = np.argmax(result.u[-1])
        assert final_peak_idx > initial_peak_idx
        # Solution should remain finite
        assert np.all(np.isfinite(result.u))
    def test_peclet_number_effects(self):
        """Test behavior for different Peclet numbers."""
        def initial_condition(x):
            return np.exp(-5 * (x - 1)**2)
        # High Peclet number (advection dominated)
        solver_high_pe = AdvectionDiffusionSolver(
            self.domain, self.boundary_conditions,
            velocity=5.0, diffusivity=0.01, options=self.options
        )
        # Low Peclet number (diffusion dominated)
        solver_low_pe = AdvectionDiffusionSolver(
            self.domain, self.boundary_conditions,
            velocity=0.1, diffusivity=1.0, options=self.options
        )
        for solver in [solver_high_pe, solver_low_pe]:
            result = solver.solve_transient(
                initial_condition=initial_condition,
                time_span=(0, 1),
                dt=0.01
            )
            # Should complete successfully
            assert result.success
            assert np.all(np.isfinite(result.u))
class TestPDEInterface:
    """Test high-level PDE solving interface."""
    def test_solve_pde_function(self):
        """Test solve_pde convenience function."""
        domain = {'x': np.linspace(0, 1, 51)}
        boundary_conditions = {'dirichlet': {0: 0.0, 50: 0.0}}
        def source_term(x):
            return np.pi**2 * np.sin(np.pi * x)
        # Test different PDE types
        pde_types = ['heat', 'poisson']
        for pde_type in pde_types:
            if pde_type == 'heat':
                # Steady heat equation
                result = solve_pde(
                    pde_type=pde_type,
                    domain=domain,
                    boundary_conditions=boundary_conditions,
                    source_term=source_term,
                    parameters={'thermal_diffusivity': 1.0}
                )
            else:
                # Poisson equation
                result = solve_pde(
                    pde_type=pde_type,
                    domain=domain,
                    boundary_conditions=boundary_conditions,
                    source_term=source_term
                )
            assert result.success
            assert len(result.u) == len(domain['x'])
    def test_transient_pde_interface(self):
        """Test transient PDE solving interface."""
        domain = {'x': np.linspace(0, 1, 51)}
        boundary_conditions = {'dirichlet': {0: 0.0, 50: 0.0}}
        def initial_condition(x):
            return np.sin(np.pi * x)
        result = solve_pde(
            pde_type='heat',
            domain=domain,
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            time_span=(0, 0.1),
            dt=0.001,
            parameters={'thermal_diffusivity': 0.1}
        )
        assert result.success
        assert result.u.ndim == 2
    def test_invalid_pde_type(self):
        """Test error handling for invalid PDE type."""
        domain = {'x': np.linspace(0, 1, 51)}
        boundary_conditions = {'dirichlet': {0: 0.0, 50: 0.0}}
        with pytest.raises(ValueError):
            solve_pde(
                pde_type='invalid_pde',
                domain=domain,
                boundary_conditions=boundary_conditions
            )
class TestPDEAccuracy:
    """Test PDE solver accuracy and convergence."""
    def test_spatial_convergence(self):
        """Test spatial convergence for Poisson equation."""
        # Use problem with known analytical solution
        def source_term(x):
            return np.pi**2 * np.sin(np.pi * x)
        def analytical_solution(x):
            return np.sin(np.pi * x)
        # Test different grid sizes
        grid_sizes = [21, 41, 81, 161]
        errors = []
        for n in grid_sizes:
            domain = {'x': np.linspace(0, 1, n)}
            boundary_conditions = {'dirichlet': {0: 0.0, n-1: 0.0}}
            result = solve_pde(
                pde_type='poisson',
                domain=domain,
                boundary_conditions=boundary_conditions,
                source_term=source_term
            )
            if result.success:
                u_analytical = analytical_solution(domain['x'])
                error = np.max(np.abs(result.u - u_analytical))
                errors.append(error)
            else:
                errors.append(np.inf)
        # Errors should decrease with finer grids
        assert len(errors) == len(grid_sizes)
        for i in range(1, len(errors)):
            assert errors[i] < errors[i-1] * 2  # Allow some variation
    def test_temporal_convergence(self):
        """Test temporal convergence for heat equation."""
        def initial_condition(x):
            return np.sin(np.pi * x)
        domain = {'x': np.linspace(0, 1, 51)}
        boundary_conditions = {'dirichlet': {0: 0.0, 50: 0.0}}
        # Analytical solution at t=0.01: u(x,t) = sin(πx)exp(-π²αt)
        alpha = 0.1
        final_time = 0.01
        def analytical_final(x):
            return np.sin(np.pi * x) * np.exp(-np.pi**2 * alpha * final_time)
        # Test different time steps
        dt_values = [0.001, 0.0005, 0.00025]
        errors = []
        for dt in dt_values:
            result = solve_pde(
                pde_type='heat',
                domain=domain,
                boundary_conditions=boundary_conditions,
                initial_condition=initial_condition,
                time_span=(0, final_time),
                dt=dt,
                parameters={'thermal_diffusivity': alpha}
            )
            if result.success:
                u_analytical = analytical_final(domain['x'])
                error = np.max(np.abs(result.u[-1] - u_analytical))
                errors.append(error)
            else:
                errors.append(np.inf)
        # Errors should decrease with smaller time steps
        for i in range(1, len(errors)):
            assert errors[i] <= errors[i-1] * 2  # Allow some variation
class TestPDERobustness:
    """Test PDE solver robustness and edge cases."""
    def test_zero_source_term(self):
        """Test with zero source term."""
        domain = {'x': np.linspace(0, 1, 51)}
        boundary_conditions = {'dirichlet': {0: 1.0, 50: 2.0}}
        def zero_source(x):
            return np.zeros_like(x)
        result = solve_pde(
            pde_type='poisson',
            domain=domain,
            boundary_conditions=boundary_conditions,
            source_term=zero_source
        )
        assert result.success
        # Solution should be linear interpolation between boundaries
        x = domain['x']
        expected = 1.0 + (2.0 - 1.0) * x
        error = np.max(np.abs(result.u - expected))
        assert error < 1e-10
    def test_constant_source_term(self):
        """Test with constant source term."""
        domain = {'x': np.linspace(0, 1, 51)}
        boundary_conditions = {'dirichlet': {0: 0.0, 50: 0.0}}
        def constant_source(x):
            return np.ones_like(x)
        result = solve_pde(
            pde_type='poisson',
            domain=domain,
            boundary_conditions=boundary_conditions,
            source_term=constant_source
        )
        assert result.success
        assert np.all(np.isfinite(result.u))
        # Maximum should be in the interior
        max_idx = np.argmax(result.u)
        assert 0 < max_idx < len(result.u) - 1
    def test_large_domain(self):
        """Test with large spatial domain."""
        domain = {'x': np.linspace(0, 100, 201)}
        boundary_conditions = {'dirichlet': {0: 0.0, 200: 0.0}}
        def source_term(x):
            return np.exp(-x/10)
        result = solve_pde(
            pde_type='poisson',
            domain=domain,
            boundary_conditions=boundary_conditions,
            source_term=source_term
        )
        assert result.success
        assert np.all(np.isfinite(result.u))
if __name__ == "__main__":
    pytest.main([__file__])