#!/usr/bin/env python3
"""
Test suite for Berkeley SciComp Quantum modules.
Comprehensive tests for quantum states, operators, and algorithms.
"""
import unittest
import numpy as np
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from Quantum.core.quantum_states import QuantumState, BellStates, EntanglementMeasures
from Quantum.core.quantum_operators import PauliOperators, QuantumGates, HamiltonianOperators
from Quantum.core.quantum_algorithms import QuantumFourierTransform, AmplitudeAmplification
class TestQuantumStates(unittest.TestCase):
    """Test quantum state operations."""
    def setUp(self):
        """Set up test fixtures."""
        self.tolerance = 1e-10
    def test_quantum_state_creation(self):
        """Test quantum state creation and normalization."""
        # Pure state
        psi = np.array([1, 0, 0, 0])
        state = QuantumState(psi)
        self.assertEqual(len(state.state_vector), 4)
        self.assertAlmostEqual(np.linalg.norm(state.state_vector), 1.0, places=10)
    def test_bell_states(self):
        """Test Bell state creation."""
        phi_plus = BellStates.phi_plus()
        # Check normalization
        self.assertAlmostEqual(np.linalg.norm(phi_plus.state_vector), 1.0, places=10)
        # Check correct form |Φ+⟩ = (|00⟩ + |11⟩)/√2
        expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        np.testing.assert_array_almost_equal(phi_plus.state_vector, expected, decimal=10)
    def test_entanglement_measures(self):
        """Test entanglement quantification."""
        # Bell state should have maximum entanglement
        phi_plus = BellStates.phi_plus()
        concurrence = EntanglementMeasures.concurrence(phi_plus)
        # Bell states have concurrence = 1
        self.assertAlmostEqual(concurrence, 1.0, places=8)
        # Separable state should have zero entanglement
        separable = QuantumState(np.kron([1, 0], [1, 0]))  # |00⟩
        conc_sep = EntanglementMeasures.concurrence(separable)
        self.assertAlmostEqual(conc_sep, 0.0, places=8)
class TestQuantumOperators(unittest.TestCase):
    """Test quantum operators."""
    def test_pauli_matrices(self):
        """Test Pauli matrix properties."""
        # Test anticommutation relations
        sigma_x = PauliOperators.X
        sigma_y = PauliOperators.Y
        sigma_z = PauliOperators.Z
        # {σ_x, σ_y} = 0
        anticomm = PauliOperators.anticommutator(sigma_x, sigma_y)
        np.testing.assert_array_almost_equal(anticomm, np.zeros((2, 2)), decimal=10)
        # σ_x² = I
        np.testing.assert_array_almost_equal(sigma_x @ sigma_x, np.eye(2), decimal=10)
    def test_quantum_gates(self):
        """Test quantum gate properties."""
        # Hadamard gate is unitary
        H = QuantumGates.H
        np.testing.assert_array_almost_equal(H @ H.conj().T, np.eye(2), decimal=10)
        # CNOT gate is unitary
        cnot = QuantumGates.CNOT()
        np.testing.assert_array_almost_equal(cnot @ cnot.conj().T, np.eye(4), decimal=10)
    def test_hamiltonian_operators(self):
        """Test Hamiltonian construction."""
        # Ising model
        H_ising = HamiltonianOperators.ising_1d(2, J=1.0, h=0.5)
        # Should be Hermitian
        np.testing.assert_array_almost_equal(H_ising, H_ising.conj().T, decimal=10)
        # Check dimensions
        self.assertEqual(H_ising.shape, (4, 4))
class TestQuantumAlgorithms(unittest.TestCase):
    """Test quantum algorithms."""
    def test_qft_matrix(self):
        """Test QFT matrix properties."""
        qft_2 = QuantumFourierTransform.qft_matrix(2)
        # QFT should be unitary
        np.testing.assert_array_almost_equal(qft_2 @ qft_2.conj().T, np.eye(4), decimal=10)
        # Check dimensions
        self.assertEqual(qft_2.shape, (4, 4))
    def test_amplitude_amplification(self):
        """Test amplitude amplification setup."""
        # Create simple oracle (marks |11⟩ state)
        oracle = np.eye(4)
        oracle[3, 3] = -1  # Flip sign of |11⟩
        G = AmplitudeAmplification.grover_operator(oracle, 2)
        # Grover operator should be unitary
        np.testing.assert_array_almost_equal(G @ G.conj().T, np.eye(4), decimal=8)
def run_quantum_tests():
    """Run all quantum module tests."""
    print("\n" + "="*60)
    print("Running Berkeley SciComp Quantum Module Tests")
    print("="*60)
    # Create test suite
    test_classes = [TestQuantumStates, TestQuantumOperators, TestQuantumAlgorithms]
    total_tests = 0
    total_failures = 0
    total_errors = 0
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 40)
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
    # Summary
    print("\n" + "="*60)
    print("QUANTUM MODULE TEST SUMMARY")
    print("="*60)
    print(f"Total tests run: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    success_rate = (total_tests - total_failures - total_errors) / total_tests * 100
    print(f"Success rate: {success_rate:.1f}%")
    if total_failures == 0 and total_errors == 0:
        print("✓ All Quantum module tests PASSED!")
        return True
    else:
        print("✗ Some Quantum module tests FAILED!")
        return False
if __name__ == '__main__':
    success = run_quantum_tests()
    sys.exit(0 if success else 1)