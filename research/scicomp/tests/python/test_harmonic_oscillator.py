#!/usr/bin/env python3
"""
Test Suite for Quantum Harmonic Oscillator
Comprehensive tests for the quantum harmonic oscillator implementation
including eigenstate accuracy, time evolution unitarity, and coherent states.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import sys
from pathlib import Path
# Add the Python package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Python"))
from quantum_physics.quantum_dynamics.harmonic_oscillator import (
    QuantumHarmonic, harmonic_eigenstate, coherent_state
)
from utils.constants import hbar, me
class TestQuantumHarmonic:
    """Test suite for QuantumHarmonic class."""
    @pytest.fixture
    def qho(self):
        """Create a QuantumHarmonic instance for testing."""
        return QuantumHarmonic(omega=1.0, mass=me, n_max=20, x_max=8.0, n_points=500)
    def test_initialization(self, qho):
        """Test proper initialization of QuantumHarmonic."""
        assert qho.omega == 1.0
        assert qho.mass == me
        assert qho.n_max == 20
        assert len(qho.x) == 500
        assert qho.x0 > 0  # Characteristic length should be positive
        assert qho.E0 > 0  # Zero-point energy should be positive
    def test_energy_levels(self, qho):
        """Test energy level formula: E_n = ℏω(n + 1/2)."""
        for n in range(10):
            expected_energy = hbar * qho.omega * (n + 0.5)
            calculated_energy = qho.energy(n)
            assert_allclose(calculated_energy, expected_energy, rtol=1e-12)
    def test_energy_ordering(self, qho):
        """Test that energy levels are properly ordered."""
        energies = [qho.energy(n) for n in range(10)]
        # Check that energies increase monotonically
        for i in range(1, len(energies)):
            assert energies[i] > energies[i-1]
    def test_eigenstate_normalization(self, qho):
        """Test that eigenstates are properly normalized."""
        for n in range(5):
            psi_n = qho.eigenstate(n)
            norm_squared = np.trapz(np.abs(psi_n)**2, qho.x)
            assert_allclose(norm_squared, 1.0, rtol=1e-8)
    def test_eigenstate_orthogonality(self, qho):
        """Test orthogonality of different eigenstates."""
        # Test a few pairs of states
        test_pairs = [(0, 1), (0, 2), (1, 2), (1, 3)]
        for n, m in test_pairs:
            psi_n = qho.eigenstate(n)
            psi_m = qho.eigenstate(m)
            overlap = np.trapz(np.conj(psi_n) * psi_m, qho.x)
            assert_allclose(np.abs(overlap), 0.0, atol=1e-10)
    def test_ground_state_properties(self, qho):
        """Test specific properties of the ground state."""
        psi_0 = qho.eigenstate(0)
        # Ground state should be real and even
        assert_allclose(np.imag(psi_0), 0.0, atol=1e-12)
        # Check even parity: ψ(-x) = ψ(x)
        n_points = len(qho.x)
        mid = n_points // 2
        left_half = psi_0[:mid]
        right_half = psi_0[mid+1:][::-1]  # Reverse right half
        if len(left_half) == len(right_half):
            assert_allclose(left_half, right_half, rtol=1e-6)
    def test_coherent_state_normalization(self, qho):
        """Test that coherent states are normalized."""
        alphas = [0.5, 1.0, 1.5, 2.0]
        for alpha in alphas:
            psi_coh = qho.coherent_state(alpha)
            norm_squared = np.trapz(np.abs(psi_coh)**2, qho.x)
            assert_allclose(norm_squared, 1.0, rtol=1e-6)
    def test_coherent_state_displacement(self, qho):
        """Test that coherent states are displaced in position."""
        alpha = 1.5
        psi_coh = qho.coherent_state(alpha)
        # Calculate position expectation value
        x_exp = qho.expectation_value(psi_coh, 'x')
        # For real α, expectation value should be approximately √2 α x₀
        expected_x = np.sqrt(2) * alpha * qho.x0
        assert_allclose(np.real(x_exp), expected_x, rtol=0.1)
    def test_time_evolution_unitarity(self, qho):
        """Test that time evolution preserves norm."""
        # Initial state (first excited state)
        psi_initial = qho.eigenstate(1)
        # Evolve for different times
        times = np.linspace(0, 2*np.pi/qho.omega, 10)
        for t in times:
            psi_t = qho.time_evolution(psi_initial, t)
            norm = qho.calculate_norm(psi_t)
            assert_allclose(norm, 1.0, rtol=1e-10)
    def test_energy_eigenstate_time_evolution(self, qho):
        """Test that energy eigenstates acquire only phase under time evolution."""
        n = 2
        psi_initial = qho.eigenstate(n)
        # Evolve for a quarter period
        t = np.pi / (2 * qho.omega)
        psi_t = qho.time_evolution(psi_initial, t)
        # Calculate expected phase
        expected_phase = np.exp(-1j * qho.energy(n) * t / hbar)
        expected_psi = expected_phase * psi_initial
        # Check that the evolved state matches expected
        assert_allclose(psi_t, expected_psi, rtol=1e-8)
    def test_energy_expectation_values(self, qho):
        """Test energy expectation values for eigenstates."""
        for n in range(5):
            psi_n = qho.eigenstate(n)
            energy_exp = qho.expectation_value(psi_n, 'H')
            energy_exact = qho.energy(n)
            assert_allclose(np.real(energy_exp), energy_exact, rtol=1e-6)
    def test_position_momentum_uncertainty(self, qho):
        """Test position-momentum uncertainty relation."""
        # Test for ground state
        psi_0 = qho.eigenstate(0)
        x_exp = qho.expectation_value(psi_0, 'x')
        x2_exp = qho.expectation_value(psi_0, 'x2')
        p_exp = qho.expectation_value(psi_0, 'p')
        p2_exp = qho.expectation_value(psi_0, 'p2')
        # Calculate uncertainties
        sigma_x_squared = np.real(x2_exp - x_exp**2)
        sigma_p_squared = np.real(p2_exp - p_exp**2)
        # Uncertainty product
        uncertainty_product = sigma_x_squared * sigma_p_squared
        # For ground state, should equal minimum (ℏ/2)²
        expected_minimum = (hbar / 2)**2
        assert uncertainty_product >= expected_minimum * 0.99  # Allow small numerical error
    def test_classical_turning_points(self, qho):
        """Test that probability density decreases at classical turning points."""
        for n in [0, 1, 2]:
            psi_n = qho.eigenstate(n)
            prob_density = np.abs(psi_n)**2
            # Classical turning point: E_n = (1/2)mω²x²
            energy = qho.energy(n)
            x_turning = np.sqrt(2 * energy / (qho.mass * qho.omega**2))
            # Find indices near turning points
            idx_turning = np.argmin(np.abs(qho.x - x_turning))
            idx_beyond = min(idx_turning + 50, len(qho.x) - 1)
            # Probability should be much smaller beyond classical turning point
            # (this tests the exponential decay in the classically forbidden region)
            if idx_beyond < len(prob_density):
                assert prob_density[idx_beyond] < prob_density[idx_turning] * 0.5
    def test_hermite_polynomial_recursion(self, qho):
        """Test that eigenstates satisfy Hermite polynomial properties."""
        # Test that higher n states have more oscillations
        psi_0 = qho.eigenstate(0)
        psi_2 = qho.eigenstate(2)
        # Count sign changes (crude measure of oscillations)
        def count_sign_changes(arr):
            return np.sum(np.diff(np.sign(arr)) != 0)
        sign_changes_0 = count_sign_changes(np.real(psi_0))
        sign_changes_2 = count_sign_changes(np.real(psi_2))
        # Higher quantum number should have more oscillations
        assert sign_changes_2 > sign_changes_0
    def test_invalid_quantum_number(self, qho):
        """Test error handling for invalid quantum numbers."""
        with pytest.raises(ValueError):
            qho.eigenstate(-1)
        with pytest.raises(ValueError):
            qho.energy(-1)
    def test_wigner_function_properties(self, qho):
        """Test basic properties of Wigner function."""
        # Test for ground state
        psi_0 = qho.eigenstate(0)
        X, P, W = qho.wigner_function(psi_0, n_points=(30, 30))
        # Wigner function should integrate to 1/(2πℏ) in phase space
        dx = X[1, 0] - X[0, 0]
        dp = P[0, 1] - P[0, 0]
        integral = np.sum(W) * dx * dp
        expected_integral = 1.0 / (2 * np.pi * hbar)
        # Note: This is approximate due to finite grid
        assert_allclose(integral, expected_integral, rtol=0.5)
        # For ground state, Wigner function should be positive everywhere
        # (This is a special property of Gaussian states)
        assert np.all(W >= -1e-10)  # Allow small numerical errors
class TestConvenienceFunctions:
    """Test convenience functions."""
    def test_harmonic_eigenstate_function(self):
        """Test standalone harmonic_eigenstate function."""
        omega = 2.0
        x = np.linspace(-5, 5, 200)
        # Compare with QuantumHarmonic class
        qho = QuantumHarmonic(omega, x_max=5.0, n_points=200)
        for n in range(3):
            psi_function = harmonic_eigenstate(n, omega, x)
            psi_class = qho.eigenstate(n)
            # Should be very similar (up to numerical differences)
            correlation = np.abs(np.trapz(np.conj(psi_function) * psi_class, x))
            assert correlation > 0.99
    def test_coherent_state_function(self):
        """Test standalone coherent_state function."""
        alpha = 1.0
        omega = 1.5
        x = np.linspace(-6, 6, 300)
        # Compare with QuantumHarmonic class
        qho = QuantumHarmonic(omega, x_max=6.0, n_points=300)
        psi_function = coherent_state(alpha, omega, x)
        psi_class = qho.coherent_state(alpha)
        # Check normalization
        norm_function = np.sqrt(np.trapz(np.abs(psi_function)**2, x))
        assert_allclose(norm_function, 1.0, rtol=1e-6)
        # Check similarity (allowing for phase differences)
        overlap = np.abs(np.trapz(np.conj(psi_function) * psi_class, x))
        assert overlap > 0.99
@pytest.mark.parametrize("omega", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("n_max", [10, 20])
def test_parameter_variations(omega, n_max):
    """Test QuantumHarmonic with different parameters."""
    qho = QuantumHarmonic(omega=omega, n_max=n_max)
    # Basic functionality should work for all parameter combinations
    psi_0 = qho.eigenstate(0)
    norm = qho.calculate_norm(psi_0)
    assert_allclose(norm, 1.0, rtol=1e-8)
    energy_0 = qho.energy(0)
    expected_energy_0 = hbar * omega * 0.5
    assert_allclose(energy_0, expected_energy_0, rtol=1e-12)
class TestPerformance:
    """Performance and scaling tests."""
    def test_large_quantum_numbers(self):
        """Test behavior with large quantum numbers."""
        qho = QuantumHarmonic(omega=1.0, n_max=50)
        # Should handle reasonably large n
        n_large = 20
        psi_n = qho.eigenstate(n_large)
        norm = qho.calculate_norm(psi_n)
        assert_allclose(norm, 1.0, rtol=1e-6)
    @pytest.mark.slow
    def test_high_resolution_grid(self):
        """Test with high-resolution spatial grid."""
        qho = QuantumHarmonic(omega=1.0, n_points=2000)
        # High resolution should give more accurate results
        psi_0 = qho.eigenstate(0)
        energy_0 = qho.expectation_value(psi_0, 'H')
        expected_energy_0 = qho.energy(0)
        # Should be very accurate with high resolution
        assert_allclose(np.real(energy_0), expected_energy_0, rtol=1e-10)
if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])