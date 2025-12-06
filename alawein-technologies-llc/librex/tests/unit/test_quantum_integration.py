"""
Unit tests for quantum computing integration.

Tests the quantum module functionality without requiring quantum libraries.
"""

import pytest
import numpy as np
from unittest.mock import patch

from Librex.core.interfaces import StandardizedProblem
from Librex.quantum import (
    check_quantum_availability,
    QISKIT_AVAILABLE,
    PENNYLANE_AVAILABLE,
)


class TestQuantumAvailability:
    """Test quantum library availability checking."""

    def test_availability_flags(self):
        """Test that availability flags are boolean."""
        assert isinstance(QISKIT_AVAILABLE, bool)
        assert isinstance(PENNYLANE_AVAILABLE, bool)

    def test_check_quantum_availability(self):
        """Test quantum availability checking function."""
        # Should return bool
        assert isinstance(check_quantum_availability(), bool)
        assert isinstance(check_quantum_availability('qiskit'), bool)
        assert isinstance(check_quantum_availability('pennylane'), bool)

    def test_invalid_library_check(self):
        """Test checking for invalid library raises error."""
        with pytest.raises(ValueError):
            check_quantum_availability('invalid_library')


class TestQUBOConversion:
    """Test QUBO conversion functionality."""

    def test_qubo_converter_import(self):
        """Test that QUBO converter can be imported."""
        from Librex.quantum.adapters import QUBOConverter
        assert QUBOConverter is not None

    def test_qap_to_qubo_conversion(self):
        """Test QAP to QUBO conversion."""
        from Librex.quantum.adapters import QUBOConverter

        n = 3
        flow = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        distance = np.array([[0, 2, 3], [2, 0, 1], [3, 1, 0]])

        converter = QUBOConverter(penalty_weight=100.0)
        qubo = converter.convert_qap_to_qubo(flow, distance)

        assert qubo.num_variables == n * n
        assert qubo.Q.shape == (n * n, n * n)
        assert np.allclose(qubo.Q, qubo.Q.T)  # Symmetric

    def test_tsp_to_qubo_conversion(self):
        """Test TSP to QUBO conversion."""
        from Librex.quantum.adapters import QUBOConverter

        distance = np.array([
            [0, 10, 15],
            [10, 0, 20],
            [15, 20, 0]
        ])

        converter = QUBOConverter()
        qubo = converter.convert_tsp_to_qubo(distance)

        assert qubo.num_variables == 9
        assert qubo.Q.shape == (9, 9)

    def test_maxcut_to_qubo_conversion(self):
        """Test Max-Cut to QUBO conversion."""
        from Librex.quantum.adapters import QUBOConverter

        adjacency = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])

        converter = QUBOConverter()
        qubo = converter.convert_maxcut_to_qubo(adjacency)

        assert qubo.num_variables == 3
        assert qubo.Q.shape == (3, 3)


class TestIsingEncoding:
    """Test Ising model encoding."""

    def test_ising_encoder_import(self):
        """Test that Ising encoder can be imported."""
        from Librex.quantum.adapters import IsingEncoder
        assert IsingEncoder is not None

    def test_qubo_to_ising(self):
        """Test QUBO to Ising conversion."""
        from Librex.quantum.adapters import QUBOConverter, IsingEncoder

        # Create simple QUBO
        Q = np.array([[1, 2], [2, 3]])
        from Librex.quantum.adapters.qubo_converter import QUBOProblem
        qubo = QUBOProblem(Q=Q)

        encoder = IsingEncoder()
        ising = encoder.qubo_to_ising(qubo)

        assert len(ising.h) == 2
        assert ising.J.shape == (2, 2)
        assert np.allclose(ising.J, ising.J.T)
        assert np.all(np.diag(ising.J) == 0)

    def test_spin_binary_conversion(self):
        """Test spin to binary conversion."""
        from Librex.quantum.adapters import IsingEncoder

        encoder = IsingEncoder()

        # Test spin to binary
        spins = np.array([1, -1, 1, -1])
        binary = encoder.spin_to_binary(spins)
        assert np.array_equal(binary, [1, 0, 1, 0])

        # Test binary to spin
        binary = np.array([0, 1, 0, 1])
        spins = encoder.binary_to_spin(binary)
        assert np.array_equal(spins, [-1, 1, -1, 1])


class TestQuantumValidation:
    """Test quantum problem validation."""

    def test_problem_validator_import(self):
        """Test that problem validator can be imported."""
        from Librex.quantum.validators import QuantumProblemValidator
        assert QuantumProblemValidator is not None

    def test_problem_validation(self):
        """Test problem validation for quantum suitability."""
        from Librex.quantum.validators import QuantumProblemValidator

        # Small problem (should be suitable)
        small_problem = StandardizedProblem(
            dimension=16,
            objective_matrix=np.random.randn(16, 16),
            problem_metadata={'problem_type': 'QAP', 'n_facilities': 4}
        )

        validator = QuantumProblemValidator('nisq')
        validation = validator.validate_problem(small_problem)

        assert 'is_suitable' in validation
        assert 'required_qubits' in validation
        assert validation['required_qubits'] == 16

        # Large problem (should not be suitable for NISQ)
        large_problem = StandardizedProblem(
            dimension=100,
            objective_matrix=np.random.randn(100, 100),
            problem_metadata={'problem_type': 'QAP', 'n_facilities': 10}
        )

        validation = validator.validate_problem(large_problem)
        assert validation['required_qubits'] == 100
        assert not validation['size_feasible']  # Too large for NISQ

    def test_algorithm_recommendation(self):
        """Test quantum algorithm recommendation."""
        from Librex.quantum.validators import QuantumProblemValidator

        problem = StandardizedProblem(
            dimension=16,
            objective_matrix=np.random.randn(16, 16),
            problem_metadata={'problem_type': 'QAP', 'n_facilities': 4}
        )

        validator = QuantumProblemValidator()
        recommendation = validator.recommend_quantum_algorithm(problem)

        assert 'algorithm' in recommendation
        assert recommendation['algorithm'] in ['qaoa', 'vqe', 'quantum_annealing']
        assert 'reasoning' in recommendation


class TestQubitEstimation:
    """Test qubit requirement estimation."""

    def test_qubit_estimator_import(self):
        """Test that qubit estimator can be imported."""
        from Librex.quantum.validators import QubitEstimator
        assert QubitEstimator is not None

    def test_qubit_estimation(self):
        """Test qubit requirement estimation."""
        from Librex.quantum.validators import QubitEstimator

        problem = StandardizedProblem(
            dimension=16,
            problem_metadata={'problem_type': 'QAP', 'n_facilities': 4}
        )

        estimator = QubitEstimator()
        estimate = estimator.estimate_qubits(problem)

        assert 'logical_qubits' in estimate
        assert 'ancilla_qubits' in estimate
        assert 'total_qubits' in estimate
        assert estimate['logical_qubits'] == 16

    def test_circuit_depth_estimation(self):
        """Test circuit depth estimation."""
        from Librex.quantum.validators import QubitEstimator

        problem = StandardizedProblem(
            dimension=9,
            problem_metadata={'problem_type': 'TSP', 'n_cities': 3}
        )

        estimator = QubitEstimator()
        depth = estimator.estimate_circuit_depth(problem, 'qaoa')

        assert 'total_depth' in depth
        assert 'p_layers' in depth
        assert depth['n_qubits'] == 9

    def test_encoding_comparison(self):
        """Test comparing different encodings."""
        from Librex.quantum.validators import QubitEstimator

        problem = StandardizedProblem(
            dimension=16,
            problem_metadata={'problem_type': 'QAP', 'n_facilities': 4}
        )

        estimator = QubitEstimator()
        comparison = estimator.compare_encodings(problem)

        assert 'binary' in comparison
        assert 'recommended' in comparison
        # Total qubits includes ancilla
        assert comparison['binary']['qubits'] >= 16


class TestQuantumUtilities:
    """Test quantum utility functions."""

    def test_state_decoder_import(self):
        """Test that state decoder can be imported."""
        from Librex.quantum.utils import QuantumStateDecoder
        assert QuantumStateDecoder is not None

    def test_bitstring_decoding(self):
        """Test decoding bitstrings to solutions."""
        from Librex.quantum.utils import QuantumStateDecoder

        decoder = QuantumStateDecoder()

        # Test QAP decoding
        bitstring = '100010001'  # Identity permutation for 3x3
        solution = decoder.decode_bitstring(bitstring, 'QAP')
        assert np.array_equal(solution, [0, 1, 2])

        # Test generic decoding
        bitstring = '1010'
        solution = decoder.decode_bitstring(bitstring, 'generic')
        assert np.array_equal(solution, [1, 0, 1, 0])

    def test_hamiltonian_builder_import(self):
        """Test that Hamiltonian builder can be imported."""
        from Librex.quantum.utils import HamiltonianBuilder
        assert HamiltonianBuilder is not None

    def test_ising_hamiltonian_building(self):
        """Test building Ising Hamiltonian."""
        from Librex.quantum.utils import HamiltonianBuilder

        builder = HamiltonianBuilder()

        h = np.array([1.0, -1.0])
        J = np.array([[0, 0.5], [0.5, 0]])

        H = builder.build_ising_hamiltonian(h, J)

        assert H.shape == (4, 4)
        assert np.allclose(H, H.conj().T)  # Hermitian

    def test_result_converter_import(self):
        """Test that result converter can be imported."""
        from Librex.quantum.utils import quantum_to_classical_result
        assert quantum_to_classical_result is not None


class TestQuantumMethods:
    """Test quantum optimization methods (without quantum libraries)."""

    def test_method_imports(self):
        """Test that quantum methods can be imported."""
        # These should import even without quantum libraries
        # but will raise errors when called without libraries
        from Librex.quantum.methods import (
            quantum_annealing_optimize,
            qaoa_optimize,
            vqe_optimize
        )
        assert quantum_annealing_optimize is not None
        assert qaoa_optimize is not None
        assert vqe_optimize is not None

    @pytest.mark.skipif(
        not check_quantum_availability(),
        reason="Quantum libraries not installed"
    )
    def test_quantum_annealing_simulation(self):
        """Test quantum annealing with simulation."""
        from Librex.quantum.methods import quantum_annealing_optimize

        problem = StandardizedProblem(
            dimension=4,
            objective_matrix=np.random.randn(4, 4),
            problem_metadata={'problem_type': 'generic'}
        )

        config = {'num_reads': 10, 'seed': 42}
        result = quantum_annealing_optimize(problem, config)

        assert 'solution' in result
        assert 'objective' in result
        assert 'is_valid' in result


class TestQuantumAdapter:
    """Test high-level quantum problem adapter."""

    def test_quantum_adapter_import(self):
        """Test that quantum adapter can be imported."""
        from Librex.quantum.adapters import QuantumProblemAdapter
        assert QuantumProblemAdapter is not None

    def test_problem_conversion(self):
        """Test converting problems to quantum format."""
        from Librex.quantum.adapters import QuantumProblemAdapter

        adapter = QuantumProblemAdapter()

        problem = StandardizedProblem(
            dimension=9,
            objective_matrix=np.random.randn(9, 9),
            problem_metadata={
                'problem_type': 'QAP',
                'flow_matrix': np.random.randn(3, 3),
                'distance_matrix': np.random.randn(3, 3),
                'n_facilities': 3
            }
        )

        # Convert to QUBO
        qubo = adapter.convert_to_quantum(problem, target_format='qubo')
        assert qubo.num_variables == 9

        # Convert to Ising
        ising = adapter.convert_to_quantum(problem, target_format='ising')
        assert ising.num_spins == 9

    def test_resource_estimation(self):
        """Test quantum resource estimation."""
        from Librex.quantum.adapters import QuantumProblemAdapter

        adapter = QuantumProblemAdapter()

        problem = StandardizedProblem(
            dimension=16,
            problem_metadata={'problem_type': 'TSP', 'n_cities': 4}
        )

        resources = adapter.estimate_quantum_resources(problem)

        assert 'num_qubits' in resources
        assert 'nisq_feasible' in resources
        assert resources['num_qubits'] == 16


if __name__ == '__main__':
    pytest.main([__file__, '-v'])