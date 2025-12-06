"""Quantum utilities test suite."""

import numpy as np
import pytest
from src.quantum_utils import (
    create_bell_state,
    pauli_matrices,
    state_fidelity,
    create_ghz_state,
    measure_state,
    calculate_entanglement_entropy,
    apply_noise
)


class TestQuantumUtils:
    
    def test_bell_states(self):
        """Bell states normalized and entangled."""
        # Test all four Bell states
        bell_types = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]
        
        for bell_type in bell_types:
            state = create_bell_state(bell_type)
            
            # Check normalization
            assert np.allclose(np.linalg.norm(state), 1.0)
            
            # Check dimension
            assert len(state) == 4
            
            # Check entanglement (all amplitudes should have magnitude 1/sqrt(2))
            assert np.allclose(np.abs(state[state != 0]), 1/np.sqrt(2))
    
    def test_pauli_matrices(self):
        """Test Pauli matrices properties."""
        paulis = pauli_matrices()
        
        # Check keys
        assert set(paulis.keys()) == {"I", "X", "Y", "Z"}
        
        # Check dimensions
        for matrix in paulis.values():
            assert matrix.shape == (2, 2)
        
        # Check Pauli algebra: X^2 = Y^2 = Z^2 = I
        assert np.allclose(paulis["X"] @ paulis["X"], paulis["I"])
        assert np.allclose(paulis["Y"] @ paulis["Y"], paulis["I"])
        assert np.allclose(paulis["Z"] @ paulis["Z"], paulis["I"])
        
        # Check anticommutation: XY = -YX = iZ
        assert np.allclose(paulis["X"] @ paulis["Y"], 1j * paulis["Z"])
        assert np.allclose(paulis["Y"] @ paulis["X"], -1j * paulis["Z"])
    
    def test_state_fidelity(self):
        """Test state fidelity calculation."""
        # Identical states should have fidelity 1
        state1 = np.array([1, 0, 0, 0])
        assert np.allclose(state_fidelity(state1, state1), 1.0)
        
        # Orthogonal states should have fidelity 0
        state2 = np.array([0, 1, 0, 0])
        assert np.allclose(state_fidelity(state1, state2), 0.0)
        
        # Test with Bell states
        phi_plus = create_bell_state("phi_plus")
        phi_minus = create_bell_state("phi_minus")
        
        # Same state
        assert np.allclose(state_fidelity(phi_plus, phi_plus), 1.0)
        
        # Different Bell states are orthogonal
        assert np.allclose(state_fidelity(phi_plus, phi_minus), 0.0)
    
    def test_ghz_state(self):
        """Test GHZ state creation."""
        # Test different qubit numbers
        for n_qubits in [2, 3, 4]:
            state = create_ghz_state(n_qubits)
            
            # Check dimension
            assert len(state) == 2**n_qubits
            
            # Check normalization
            assert np.allclose(np.linalg.norm(state), 1.0)
            
            # Check that only first and last basis states have non-zero amplitude
            assert np.allclose(np.abs(state[0]), 1/np.sqrt(2))
            assert np.allclose(np.abs(state[-1]), 1/np.sqrt(2))
            assert np.allclose(state[1:-1], 0)
    
    def test_measure_state(self):
        """Test quantum state measurement."""
        # Test computational basis state
        state = np.array([1, 0, 0, 0])
        counts = measure_state(state, n_shots=1000)
        
        # Should only measure "00"
        assert "00" in counts
        assert counts["00"] == 1000
        
        # Test superposition state
        state = np.array([1, 1, 0, 0]) / np.sqrt(2)
        counts = measure_state(state, n_shots=10000)
        
        # Should measure "00" and "01" with roughly equal probability
        assert "00" in counts
        assert "01" in counts
        assert abs(counts["00"] - 5000) < 500  # Within statistical fluctuation
        assert abs(counts["01"] - 5000) < 500
    
    def test_entanglement_entropy(self):
        """Test entanglement entropy calculation."""
        # Product state should have zero entropy
        state = np.array([1, 0, 0, 0])
        entropy = calculate_entanglement_entropy(state, [0])
        assert np.allclose(entropy, 0.0)
        
        # Maximally entangled state should have entropy of 1
        bell_state = create_bell_state("phi_plus")
        entropy = calculate_entanglement_entropy(bell_state, [0])
        assert np.allclose(entropy, 1.0)  # 1 bit of entanglement
    
    def test_apply_noise(self):
        """Test noise application."""
        state = np.array([1, 0, 0, 0])
        
        # No noise should preserve the state
        noisy_state = apply_noise(state, noise_prob=0.0)
        assert np.allclose(state, noisy_state)
        
        # Depolarizing noise
        noisy_state = apply_noise(state, noise_prob=0.1, noise_type="depolarizing")
        assert np.allclose(np.linalg.norm(noisy_state), 1.0)  # Still normalized
        
        # Phase flip noise
        noisy_state = apply_noise(state, noise_prob=0.1, noise_type="phase_flip")
        assert np.allclose(np.linalg.norm(noisy_state), 1.0)
        
        # Bit flip noise
        noisy_state = apply_noise(state, noise_prob=0.1, noise_type="bit_flip")
        assert np.allclose(np.linalg.norm(noisy_state), 1.0)


if __name__ == "__main__":
    pytest.main([__file__])