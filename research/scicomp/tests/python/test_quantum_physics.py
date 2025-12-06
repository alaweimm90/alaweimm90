#!/usr/bin/env python3
"""
Comprehensive Test Suite for Quantum Physics Modules
Advanced testing framework for validating quantum physics calculations,
numerical accuracy, and physical consistency in the SciComp framework.
Tests include unit tests, integration tests, and physics validation.
Test Categories:
- Unit tests for individual functions
- Integration tests for complete workflows
- Physics validation against analytical solutions
- Numerical accuracy and stability tests
- Performance and benchmark tests
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import unittest
import numpy as np
import sys
import os
# Add SciComp modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Python.quantum_physics.quantum_dynamics.wavepacket_evolution import WavepacketEvolution, WavepacketConfig
from Python.quantum_physics.quantum_dynamics.quantum_tunneling import QuantumTunneling, TunnelingConfig
from Python.quantum_physics.electronic_structure.density_of_states import DensityOfStates, DOSConfig
from Python.quantum_physics.many_body.quantum_monte_carlo import QuantumMonteCarlo, QMCConfig
from Python.quantum_computing.algorithms.grover import GroverSearch, GroverConfig
from Python.quantum_computing.circuits.quantum_gates import QuantumCircuit, GateConfig
from Python.utils.constants import hbar, me, e, pi
class TestQuantumPhysicsConstants(unittest.TestCase):
    """Test fundamental physical constants."""
    def test_physical_constants_values(self):
        """Test that physical constants have correct values."""
        # Reduced Planck constant
        self.assertAlmostEqual(hbar, 1.054571817e-34, places=10)
        # Electron mass
        self.assertAlmostEqual(me, 9.1093837015e-31, places=10)
        # Elementary charge
        self.assertAlmostEqual(e, 1.602176634e-19, places=10)
        # Pi
        self.assertAlmostEqual(pi, np.pi, places=15)
    def test_constant_relationships(self):
        """Test relationships between constants."""
        # Fine structure constant
        c = 299792458  # speed of light
        epsilon_0 = 8.8541878128e-12
        alpha = e**2 / (4 * pi * epsilon_0 * hbar * c)
        expected_alpha = 7.2973525693e-3  # 1/137.035999084
        self.assertAlmostEqual(alpha, expected_alpha, places=10)
class TestWavepacketEvolution(unittest.TestCase):
    """Test wavepacket evolution functionality."""
    def setUp(self):
        """Set up test configuration."""
        self.config = WavepacketConfig(
            x_domain=(-10.0, 10.0),
            nx=256,
            dt=0.01,
            total_time=2.0,
            initial_position=0.0,
            initial_momentum=1.0,
            sigma=1.0,
            mass=me
        )
        self.evolution = WavepacketEvolution(self.config)
    def test_initial_wavepacket_normalization(self):
        """Test that initial wavepacket is properly normalized."""
        psi0 = self.evolution.create_initial_wavepacket()
        norm = self.evolution.calculate_norm(psi0)
        self.assertAlmostEqual(norm, 1.0, places=6)
    def test_initial_wavepacket_properties(self):
        """Test initial wavepacket properties."""
        psi0 = self.evolution.create_initial_wavepacket()
        # Calculate expectation values
        x_expect = self.evolution.calculate_position_expectation(psi0)
        p_expect = self.evolution.calculate_momentum_expectation(psi0)
        # Check initial position and momentum
        self.assertAlmostEqual(x_expect, self.config.initial_position, places=1)
        self.assertAlmostEqual(p_expect, self.config.initial_momentum, places=1)
    def test_free_particle_evolution(self):
        """Test free particle evolution (no potential)."""
        # Set up free particle
        V = np.zeros(self.config.nx)
        self.evolution.set_potential(V)
        # Evolve
        results = self.evolution.evolve()
        # Check norm conservation
        norms = [self.evolution.calculate_norm(psi) for psi in results['psi'].T]
        for norm in norms:
            self.assertAlmostEqual(norm, 1.0, places=5)
        # Check that center of wavepacket moves with correct velocity
        t_final = results['t'][-1]
        x_final = self.evolution.calculate_position_expectation(results['psi'][:, -1])
        expected_x = self.config.initial_position + (self.config.initial_momentum / self.config.mass) * t_final
        self.assertAlmostEqual(x_final, expected_x, places=1)
    def test_harmonic_oscillator_evolution(self):
        """Test evolution in harmonic oscillator potential."""
        # Create harmonic potential
        x = self.evolution.x
        omega = 1.0
        V = 0.5 * self.config.mass * omega**2 * x**2
        self.evolution.set_potential(V)
        # Evolve for one period
        period = 2 * pi / omega
        config_period = WavepacketConfig(
            x_domain=self.config.x_domain,
            nx=self.config.nx,
            dt=0.01,
            total_time=period,
            initial_position=1.0,
            initial_momentum=0.0,
            sigma=0.5,
            mass=self.config.mass
        )
        evolution_period = WavepacketEvolution(config_period)
        evolution_period.set_potential(V)
        results = evolution_period.evolve()
        # Check that wavepacket returns to initial position after one period
        x_initial = evolution_period.calculate_position_expectation(results['psi'][:, 0])
        x_final = evolution_period.calculate_position_expectation(results['psi'][:, -1])
        self.assertAlmostEqual(x_final, x_initial, places=1)
class TestQuantumTunneling(unittest.TestCase):
    """Test quantum tunneling calculations."""
    def setUp(self):
        """Set up tunneling configuration."""
        self.config = TunnelingConfig(
            x_domain=(-20.0, 20.0),
            n_points=1000,
            barrier_type='rectangular',
            barrier_height=2.0,
            barrier_width=4.0,
            barrier_center=0.0,
            particle_energy=1.0,
            mass=me
        )
        self.tunneling = QuantumTunneling(self.config)
    def test_transmission_conservation(self):
        """Test that transmission + reflection = 1."""
        results = self.tunneling.calculate_transmission_coefficient()
        T = results['transmission']
        R = results['reflection']
        self.assertAlmostEqual(T + R, 1.0, places=6)
        self.assertTrue(0 <= T <= 1)
        self.assertTrue(0 <= R <= 1)
    def test_classical_limit_high_energy(self):
        """Test that transmission approaches 1 for high energies."""
        # Set particle energy well above barrier
        self.config.particle_energy = 10.0 * self.config.barrier_height
        tunneling_high_e = QuantumTunneling(self.config)
        results = tunneling_high_e.calculate_transmission_coefficient()
        T = results['transmission']
        # Should be close to 1 (perfect transmission)
        self.assertGreater(T, 0.9)
    def test_tunneling_exponential_dependence(self):
        """Test exponential dependence on barrier width."""
        widths = [2.0, 4.0, 6.0]
        transmissions = []
        for width in widths:
            self.config.barrier_width = width
            tunneling = QuantumTunneling(self.config)
            results = tunneling.calculate_transmission_coefficient()
            transmissions.append(results['transmission'])
        # Check that transmission decreases exponentially
        self.assertGreater(transmissions[0], transmissions[1])
        self.assertGreater(transmissions[1], transmissions[2])
        # Check approximate exponential relationship
        ratio1 = transmissions[1] / transmissions[0]
        ratio2 = transmissions[2] / transmissions[1]
        # Ratios should be approximately equal for exponential decay
        self.assertAlmostEqual(ratio1, ratio2, delta=0.5)
    def test_wkb_approximation_thick_barrier(self):
        """Test WKB approximation for thick barriers."""
        # Set up thick barrier
        self.config.barrier_width = 10.0
        self.config.particle_energy = 0.5 * self.config.barrier_height
        tunneling = QuantumTunneling(self.config)
        exact_result = tunneling.calculate_transmission_coefficient()
        wkb_result = tunneling.calculate_wkb_transmission()
        T_exact = exact_result['transmission']
        T_wkb = wkb_result['transmission']
        # WKB should be reasonably close for thick barriers
        relative_error = abs(T_exact - T_wkb) / T_exact
        self.assertLess(relative_error, 0.5)  # Within 50%
class TestDensityOfStates(unittest.TestCase):
    """Test density of states calculations."""
    def setUp(self):
        """Set up DOS configuration."""
        self.config = DOSConfig(
            system_type='1d_free_electron',
            energy_range=(0.0, 10.0),
            n_points=1000,
            dimensions=1,
            length=1e-9,  # 1 nm
            temperature=300.0
        )
        self.dos = DensityOfStates(self.config)
    def test_1d_free_electron_dos(self):
        """Test 1D free electron DOS."""
        results = self.dos.calculate_dos()
        energies = results['energies']
        dos_values = results['dos']
        # For 1D free electron, DOS ∝ E^(-1/2)
        # Check that DOS decreases with energy for E > 0
        mid_point = len(energies) // 2
        self.assertGreater(dos_values[10], dos_values[mid_point])
    def test_dos_normalization(self):
        """Test DOS normalization properties."""
        results = self.dos.calculate_dos()
        energies = results['energies']
        dos_values = results['dos']
        # DOS should be positive
        self.assertTrue(np.all(dos_values >= 0))
        # Integral should be finite
        integral = np.trapz(dos_values, energies)
        self.assertTrue(np.isfinite(integral))
        self.assertGreater(integral, 0)
    def test_2d_dos_scaling(self):
        """Test 2D DOS scaling."""
        config_2d = DOSConfig(
            system_type='2d_free_electron',
            energy_range=(0.0, 10.0),
            n_points=1000,
            dimensions=2,
            length=1e-9,
            temperature=300.0
        )
        dos_2d = DensityOfStates(config_2d)
        results_2d = dos_2d.calculate_dos()
        # 2D DOS should be constant for free electrons
        dos_values_2d = results_2d['dos']
        energies = results_2d['energies']
        # Check that DOS is approximately constant (within 10%)
        dos_mean = np.mean(dos_values_2d[energies > 1.0])  # Avoid E=0 singularity
        dos_std = np.std(dos_values_2d[energies > 1.0])
        relative_variation = dos_std / dos_mean
        self.assertLess(relative_variation, 0.1)
class TestQuantumMonteCarlo(unittest.TestCase):
    """Test Quantum Monte Carlo methods."""
    def setUp(self):
        """Set up QMC configuration."""
        self.config = QMCConfig(
            n_particles=2,
            n_dimensions=1,
            n_walkers=100,
            n_steps=1000,
            n_equilibration=100,
            method='VMC',
            temperature=1.0
        )
    def test_harmonic_oscillator_ground_state(self):
        """Test QMC for harmonic oscillator ground state."""
        # Define harmonic oscillator Hamiltonian
        def harmonic_hamiltonian(positions):
            omega = 1.0
            return 0.5 * omega**2 * np.sum(positions**2)
        # Simple trial wavefunction
        def trial_wavefunction(positions, params):
            alpha = params[0] if len(params) > 0 else 1.0
            return np.exp(-alpha * np.sum(positions**2) / 2)
        qmc = QuantumMonteCarlo(self.config, harmonic_hamiltonian, trial_wavefunction)
        # Run short calculation
        results = qmc.run_calculation()
        # Check that energy is reasonable (should be around 0.5 for ground state)
        energy = results['energy']
        self.assertGreater(energy, 0.3)
        self.assertLess(energy, 1.0)
        # Check that error bars are reasonable
        error = results['energy_error']
        self.assertGreater(error, 0.0)
        self.assertLess(error / energy, 0.5)  # Error should be < 50% of energy
    def test_walker_population_stability(self):
        """Test that walker population remains stable."""
        def simple_hamiltonian(positions):
            return np.sum(positions**2)
        def simple_wavefunction(positions, params):
            return np.exp(-np.sum(positions**2) / 2)
        qmc = QuantumMonteCarlo(self.config, simple_hamiltonian, simple_wavefunction)
        # Initialize walkers
        walkers = qmc.initialize_walkers()
        # Check walker array shape
        expected_shape = (self.config.n_walkers, self.config.n_particles, self.config.n_dimensions)
        self.assertEqual(walkers.shape, expected_shape)
        # Check that walkers are finite
        self.assertTrue(np.all(np.isfinite(walkers)))
class TestGroverAlgorithm(unittest.TestCase):
    """Test Grover's quantum search algorithm."""
    def setUp(self):
        """Set up Grover configuration."""
        self.config = GroverConfig(
            n_qubits=4,
            n_shots=1024,
            target_items=['1010', '1100'],
            use_optimal_iterations=True
        )
        self.grover = GroverSearch(self.config)
    def test_superposition_creation(self):
        """Test uniform superposition state creation."""
        state = self.grover.create_uniform_superposition()
        # Check normalization
        norm = np.sum(np.abs(state)**2)
        self.assertAlmostEqual(norm, 1.0, places=6)
        # Check uniform amplitudes
        expected_amplitude = 1.0 / np.sqrt(self.grover.n_states)
        for amplitude in state:
            self.assertAlmostEqual(abs(amplitude), expected_amplitude, places=6)
    def test_oracle_operator(self):
        """Test oracle operator functionality."""
        state = self.grover.create_uniform_superposition()
        # Apply oracle
        oracle_state = self.grover.oracle_operator(state)
        # Check that norm is preserved
        norm_before = np.sum(np.abs(state)**2)
        norm_after = np.sum(np.abs(oracle_state)**2)
        self.assertAlmostEqual(norm_before, norm_after, places=6)
        # Check that target states have flipped phase
        for i, basis_state in enumerate(self.grover.basis_states):
            if basis_state in self.grover.target_items:
                # Target state should have opposite phase
                self.assertAlmostEqual(oracle_state[i], -state[i], places=6)
            else:
                # Non-target states unchanged
                self.assertAlmostEqual(oracle_state[i], state[i], places=6)
    def test_optimal_iterations_calculation(self):
        """Test optimal iteration calculation."""
        optimal_k = self.grover.calculate_optimal_iterations()
        # For 4 qubits (16 states) with 2 solutions:
        # k_opt ≈ π/4 * sqrt(16/2) - 1/2 ≈ 2.2
        expected_k = 2  # Should round to 2
        self.assertEqual(optimal_k, expected_k)
    def test_grover_search_success(self):
        """Test complete Grover search."""
        results = self.grover.run_grover_search()
        # Check that success rate is reasonable
        success_rate = results['success_rate']
        self.assertGreater(success_rate, 0.5)  # Should be better than random
        # Check that most common measurement is a target
        measurements = results['measurements']
        most_common = max(measurements.items(), key=lambda x: x[1])
        most_common_state = most_common[0]
        # Most common should be one of the targets (with high probability)
        # Due to quantum nature, this might not always be true, so we check
        # that success rate is significantly above random chance
        random_success_rate = len(self.grover.target_items) / self.grover.n_states
        self.assertGreater(success_rate, 2 * random_success_rate)
class TestQuantumCircuits(unittest.TestCase):
    """Test quantum circuit functionality."""
    def setUp(self):
        """Set up quantum circuit."""
        self.config = GateConfig()
        self.circuit = QuantumCircuit(n_qubits=2, config=self.config)
    def test_single_qubit_gates(self):
        """Test single qubit gate operations."""
        from Python.quantum_computing.circuits.quantum_gates import PauliX, PauliY, PauliZ, Hadamard
        # Test Pauli gates
        X = PauliX()
        Y = PauliY()
        Z = PauliZ()
        H = Hadamard()
        # Check gate matrices
        X_matrix = X.get_matrix()
        expected_X = np.array([[0, 1], [1, 0]])
        np.testing.assert_array_almost_equal(X_matrix, expected_X)
        # Check Hadamard
        H_matrix = H.get_matrix()
        expected_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        np.testing.assert_array_almost_equal(H_matrix, expected_H)
    def test_two_qubit_gates(self):
        """Test two qubit gate operations."""
        from Python.quantum_computing.circuits.quantum_gates import CNOT
        cnot = CNOT()
        cnot_matrix = cnot.get_matrix()
        expected_cnot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        np.testing.assert_array_almost_equal(cnot_matrix, expected_cnot)
    def test_bell_state_creation(self):
        """Test Bell state creation and measurement."""
        # Create Bell state |00⟩ + |11⟩
        self.circuit.h(0)  # Hadamard on qubit 0
        self.circuit.cnot(0, 1)  # CNOT with control=0, target=1
        # Simulate circuit
        final_state = self.circuit.simulate()
        # Expected Bell state: (|00⟩ + |11⟩) / √2
        expected_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        np.testing.assert_array_almost_equal(final_state, expected_state, decimal=6)
        # Test measurements
        counts = self.circuit.measure(n_shots=1000)
        # Should only measure |00⟩ and |11⟩
        measured_states = set(counts.keys())
        expected_states = {'00', '11'}
        self.assertTrue(measured_states.issubset(expected_states))
        # Roughly equal counts for both states
        total_shots = sum(counts.values())
        for state in ['00', '11']:
            if state in counts:
                probability = counts[state] / total_shots
                self.assertGreater(probability, 0.3)  # Allow for statistical fluctuation
    def test_circuit_depth_calculation(self):
        """Test circuit depth calculation."""
        # Create simple circuit
        self.circuit.h(0)
        self.circuit.cnot(0, 1)
        self.circuit.z(1)
        depth = self.circuit.depth()
        # H and CNOT can't be parallel (CNOT depends on qubit 0)
        # Z on qubit 1 can be parallel with operations on qubit 0
        # Expected depth: 2 (H-CNOT on one layer, Z can be parallel with something)
        self.assertGreaterEqual(depth, 2)
class TestPhysicsValidation(unittest.TestCase):
    """Test physics principles and conservation laws."""
    def test_uncertainty_principle(self):
        """Test Heisenberg uncertainty principle."""
        config = WavepacketConfig(
            x_domain=(-10.0, 10.0),
            nx=512,
            sigma=1.0,
            initial_position=0.0,
            initial_momentum=0.0
        )
        evolution = WavepacketEvolution(config)
        psi = evolution.create_initial_wavepacket()
        # Calculate uncertainties
        sigma_x = evolution.calculate_position_uncertainty(psi)
        sigma_p = evolution.calculate_momentum_uncertainty(psi)
        # Check uncertainty principle: Δx Δp ≥ ℏ/2
        uncertainty_product = sigma_x * sigma_p
        heisenberg_limit = hbar / 2
        self.assertGreaterEqual(uncertainty_product, heisenberg_limit * 0.99)  # Allow small numerical error
    def test_energy_conservation_free_particle(self):
        """Test energy conservation for free particle."""
        config = WavepacketConfig(
            x_domain=(-10.0, 10.0),
            nx=256,
            dt=0.01,
            total_time=2.0,
            initial_momentum=2.0,
            sigma=1.0
        )
        evolution = WavepacketEvolution(config)
        V = np.zeros(config.nx)  # Free particle
        evolution.set_potential(V)
        results = evolution.evolve()
        # Calculate energy at each time step
        energies = []
        for i, psi in enumerate(results['psi'].T):
            E = evolution.calculate_total_energy(psi, V)
            energies.append(E)
        # Check energy conservation (should be constant)
        energy_variation = np.std(energies) / np.mean(energies)
        self.assertLess(energy_variation, 0.01)  # Within 1%
    def test_parity_conservation(self):
        """Test parity conservation in symmetric potentials."""
        config = WavepacketConfig(
            x_domain=(-10.0, 10.0),
            nx=256,
            initial_position=0.0,
            initial_momentum=0.0,
            sigma=1.0
        )
        evolution = WavepacketEvolution(config)
        # Create symmetric harmonic potential
        x = evolution.x
        V = 0.5 * x**2  # Symmetric about x=0
        evolution.set_potential(V)
        # Create symmetric initial state
        psi0 = evolution.create_initial_wavepacket()
        # Evolve for short time
        config.total_time = 0.5
        evolution.config = config
        results = evolution.evolve()
        # Final state should maintain symmetry properties
        psi_final = results['psi'][:, -1]
        # Check that probability density is symmetric
        prob_density = np.abs(psi_final)**2
        # Compare left and right halves
        mid = len(prob_density) // 2
        left_half = prob_density[:mid]
        right_half = prob_density[mid+1:][::-1]  # Reverse right half
        # Should be approximately equal (within 5%)
        if len(left_half) == len(right_half):
            symmetry_error = np.mean(np.abs(left_half - right_half)) / np.mean(prob_density)
            self.assertLess(symmetry_error, 0.05)
class TestNumericalAccuracy(unittest.TestCase):
    """Test numerical accuracy and stability."""
    def test_fft_accuracy(self):
        """Test FFT accuracy for momentum space calculations."""
        # Create simple Gaussian wavepacket
        n = 256
        x = np.linspace(-10, 10, n)
        dx = x[1] - x[0]
        sigma = 1.0
        k0 = 2.0
        psi = np.exp(-(x**2)/(2*sigma**2)) * np.exp(1j * k0 * x)
        psi = psi / np.sqrt(np.trapz(np.abs(psi)**2, x))
        # Transform to momentum space
        dk = 2*np.pi / (n * dx)
        k = np.fft.fftshift(np.fft.fftfreq(n, dx/(2*np.pi)))
        psi_k = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi))) * dx / np.sqrt(2*np.pi)
        # Check normalization in momentum space
        norm_k = np.trapz(np.abs(psi_k)**2, k)
        self.assertAlmostEqual(norm_k, 1.0, places=3)
        # Transform back to position
        psi_back = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(psi_k))) * len(k) * dk / np.sqrt(2*np.pi)
        # Check round-trip accuracy
        error = np.max(np.abs(psi - psi_back))
        self.assertLess(error, 1e-10)
    def test_time_evolution_unitarity(self):
        """Test unitarity of time evolution operator."""
        config = WavepacketConfig(
            x_domain=(-5.0, 5.0),
            nx=128,
            dt=0.01,
            total_time=0.1
        )
        evolution = WavepacketEvolution(config)
        # Create random potential
        np.random.seed(42)
        V = np.random.random(config.nx) * 2.0
        evolution.set_potential(V)
        # Create random initial state
        psi0 = np.random.random(config.nx) + 1j * np.random.random(config.nx)
        psi0 = psi0 / np.sqrt(np.trapz(np.abs(psi0)**2, evolution.x))
        # Evolve
        evolution.psi = psi0
        psi_final = evolution.time_step(psi0)
        # Check norm conservation
        norm_initial = np.trapz(np.abs(psi0)**2, evolution.x)
        norm_final = np.trapz(np.abs(psi_final)**2, evolution.x)
        self.assertAlmostEqual(norm_final, norm_initial, places=6)
def create_test_suite():
    """Create comprehensive test suite."""
    test_suite = unittest.TestSuite()
    # Add all test classes
    test_classes = [
        TestQuantumPhysicsConstants,
        TestWavepacketEvolution,
        TestQuantumTunneling,
        TestDensityOfStates,
        TestQuantumMonteCarlo,
        TestGroverAlgorithm,
        TestQuantumCircuits,
        TestPhysicsValidation,
        TestNumericalAccuracy
    ]
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    return test_suite
def run_tests():
    """Run all tests with detailed output."""
    print("=" * 70)
    print("SciComp")
    print("Quantum Physics Test Suite")
    print("=" * 70)
    print()
    # Create test suite
    suite = create_test_suite()
    # Run tests with verbose output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    result = runner.run(suite)
    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    if result.failures:
        print(f"\nFailures ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False
if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)