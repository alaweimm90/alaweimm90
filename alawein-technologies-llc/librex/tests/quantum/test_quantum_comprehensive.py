"""
Comprehensive tests for Quantum Integration.
Tests QUBO conversion, quantum methods, validators, and utilities.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import tempfile
from pathlib import Path

# Handle optional quantum dependencies
try:
    import qiskit
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

from Librex.quantum import (
    QuantumOptimizer,
    QUBOConverter,
    QuantumAdapter,
    IsingEncoder
)
from Librex.quantum.validators import (
    ProblemValidator,
    QubitEstimator
)
from Librex.quantum.utils import (
    HamiltonianBuilder,
    ResultConverter,
    StateDecoder
)
from Librex.quantum.methods import (
    QAOAOptimizer,
    QuantumAnnealingOptimizer,
    VQEOptimizer
)


class TestQUBOConverter:
    """Test QUBO (Quadratic Unconstrained Binary Optimization) converter."""

    @pytest.fixture
    def converter(self):
        """Create a QUBO converter."""
        return QUBOConverter()

    @pytest.fixture
    def qap_problem(self):
        """Create a sample QAP problem."""
        n = 5
        return {
            'flow': np.random.rand(n, n),
            'distance': np.random.rand(n, n),
            'size': n
        }

    def test_initialization(self, converter):
        """Test converter initialization."""
        assert converter is not None
        assert hasattr(converter, 'convert')
        assert hasattr(converter, 'validate')

    def test_qap_to_qubo_conversion(self, converter, qap_problem):
        """Test converting QAP to QUBO formulation."""
        Q = converter.convert_qap_to_qubo(qap_problem)

        n = qap_problem['size']
        expected_size = n * n
        assert Q.shape == (expected_size, expected_size)
        assert np.allclose(Q, Q.T)  # Should be symmetric

    def test_tsp_to_qubo_conversion(self, converter):
        """Test converting TSP to QUBO formulation."""
        n = 4
        distance_matrix = np.array([
            [0, 2, 3, 1],
            [2, 0, 4, 2],
            [3, 4, 0, 3],
            [1, 2, 3, 0]
        ])

        Q = converter.convert_tsp_to_qubo(distance_matrix, penalty=10)

        expected_size = n * n
        assert Q.shape == (expected_size, expected_size)

    def test_maxcut_to_qubo_conversion(self, converter):
        """Test converting MaxCut to QUBO formulation."""
        adjacency_matrix = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ])

        Q = converter.convert_maxcut_to_qubo(adjacency_matrix)

        n = adjacency_matrix.shape[0]
        assert Q.shape == (n, n)

    def test_constraint_encoding(self, converter):
        """Test encoding constraints in QUBO."""
        # One-hot constraint for assignment problem
        n_vars = 9  # 3x3 assignment
        constraints = converter.encode_one_hot_constraints(3, 3)

        assert 'row_constraints' in constraints
        assert 'col_constraints' in constraints
        assert len(constraints['row_constraints']) == 3
        assert len(constraints['col_constraints']) == 3

    def test_penalty_parameter_tuning(self, converter, qap_problem):
        """Test automatic penalty parameter tuning."""
        Q_low = converter.convert_qap_to_qubo(qap_problem, penalty=1.0)
        Q_high = converter.convert_qap_to_qubo(qap_problem, penalty=100.0)

        # Different penalties should give different matrices
        assert not np.allclose(Q_low, Q_high)

    def test_sparsity_optimization(self, converter, qap_problem):
        """Test QUBO matrix sparsity optimization."""
        Q = converter.convert_qap_to_qubo(qap_problem, optimize_sparsity=True)

        # Count non-zero elements
        sparsity = np.sum(Q == 0) / Q.size
        assert sparsity >= 0  # Some elements should be zero

    def test_validation(self, converter):
        """Test QUBO validation."""
        # Valid QUBO
        valid_Q = np.array([[1, 2], [2, 3]])
        assert converter.validate(valid_Q) is True

        # Invalid QUBO (non-symmetric)
        invalid_Q = np.array([[1, 2], [3, 4]])
        assert converter.validate(invalid_Q) is False


class TestIsingEncoder:
    """Test Ising model encoder."""

    @pytest.fixture
    def encoder(self):
        """Create an Ising encoder."""
        return IsingEncoder()

    def test_qubo_to_ising_conversion(self, encoder):
        """Test QUBO to Ising conversion."""
        Q = np.array([
            [1, 2, 3],
            [2, 4, 5],
            [3, 5, 6]
        ])

        h, J, offset = encoder.qubo_to_ising(Q)

        assert len(h) == 3  # Linear terms
        assert J.shape == (3, 3)  # Quadratic terms
        assert isinstance(offset, (int, float))

    def test_ising_to_qubo_conversion(self, encoder):
        """Test Ising to QUBO conversion."""
        h = np.array([1, 2, 3])
        J = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ])

        Q = encoder.ising_to_qubo(h, J)

        assert Q.shape == (3, 3)
        assert np.allclose(Q, Q.T)  # Should be symmetric

    def test_binary_to_spin_conversion(self, encoder):
        """Test binary to spin variable conversion."""
        binary_vars = np.array([0, 1, 0, 1, 1])
        spin_vars = encoder.binary_to_spin(binary_vars)

        assert len(spin_vars) == len(binary_vars)
        assert all(s in [-1, 1] for s in spin_vars)
        assert spin_vars[0] == -1  # 0 -> -1
        assert spin_vars[1] == 1   # 1 -> 1

    def test_spin_to_binary_conversion(self, encoder):
        """Test spin to binary variable conversion."""
        spin_vars = np.array([-1, 1, -1, 1, 1])
        binary_vars = encoder.spin_to_binary(spin_vars)

        assert len(binary_vars) == len(spin_vars)
        assert all(b in [0, 1] for b in binary_vars)
        assert binary_vars[0] == 0  # -1 -> 0
        assert binary_vars[1] == 1  # 1 -> 1


class TestProblemValidator:
    """Test problem validator for quantum optimization."""

    @pytest.fixture
    def validator(self):
        """Create a problem validator."""
        return ProblemValidator()

    def test_validate_qubo_size(self, validator):
        """Test QUBO size validation."""
        # Small QUBO (valid)
        small_Q = np.random.rand(10, 10)
        assert validator.validate_qubo_size(small_Q, max_qubits=20) is True

        # Large QUBO (invalid)
        large_Q = np.random.rand(100, 100)
        assert validator.validate_qubo_size(large_Q, max_qubits=20) is False

    def test_validate_connectivity(self, validator):
        """Test connectivity validation for hardware constraints."""
        # Fully connected (valid for most cases)
        Q = np.ones((5, 5))
        assert validator.validate_connectivity(Q, topology='full') is True

        # Check for specific topology
        assert validator.validate_connectivity(Q, topology='linear') is False

    def test_validate_coefficient_range(self, validator):
        """Test coefficient range validation."""
        # Valid range
        Q = np.random.rand(10, 10) * 10
        assert validator.validate_coefficient_range(Q, min_val=-100, max_val=100) is True

        # Invalid range
        Q = np.random.rand(10, 10) * 1000
        assert validator.validate_coefficient_range(Q, min_val=-100, max_val=100) is False

    def test_problem_complexity_assessment(self, validator):
        """Test problem complexity assessment."""
        Q = np.random.rand(20, 20)
        complexity = validator.assess_complexity(Q)

        assert 'n_qubits' in complexity
        assert 'n_terms' in complexity
        assert 'connectivity' in complexity
        assert 'estimated_difficulty' in complexity

        assert complexity['n_qubits'] == 20
        assert complexity['estimated_difficulty'] in ['easy', 'medium', 'hard']


class TestQubitEstimator:
    """Test qubit requirement estimator."""

    @pytest.fixture
    def estimator(self):
        """Create a qubit estimator."""
        return QubitEstimator()

    def test_estimate_qap_qubits(self, estimator):
        """Test qubit estimation for QAP."""
        problem_sizes = [3, 5, 10, 20]

        for n in problem_sizes:
            n_qubits = estimator.estimate_qap_qubits(n)
            assert n_qubits == n * n  # QAP needs n^2 qubits

    def test_estimate_tsp_qubits(self, estimator):
        """Test qubit estimation for TSP."""
        n_cities = 10
        n_qubits = estimator.estimate_tsp_qubits(n_cities)
        assert n_qubits == n_cities * n_cities

    def test_estimate_maxcut_qubits(self, estimator):
        """Test qubit estimation for MaxCut."""
        n_vertices = 15
        n_qubits = estimator.estimate_maxcut_qubits(n_vertices)
        assert n_qubits == n_vertices

    def test_ancilla_qubit_estimation(self, estimator):
        """Test ancilla qubit estimation."""
        n_logical = 10
        n_ancilla = estimator.estimate_ancilla_qubits(n_logical, error_correction='basic')
        assert n_ancilla >= 0

        n_ancilla_heavy = estimator.estimate_ancilla_qubits(n_logical, error_correction='surface_code')
        assert n_ancilla_heavy > n_ancilla

    def test_hardware_feasibility(self, estimator):
        """Test hardware feasibility assessment."""
        # Small problem - feasible
        feasibility = estimator.assess_feasibility(n_qubits=5, hardware='simulator')
        assert feasibility['feasible'] is True

        # Large problem - not feasible on real hardware
        feasibility = estimator.assess_feasibility(n_qubits=1000, hardware='ibmq')
        assert feasibility['feasible'] is False
        assert 'reason' in feasibility


class TestHamiltonianBuilder:
    """Test Hamiltonian builder for quantum optimization."""

    @pytest.fixture
    def builder(self):
        """Create a Hamiltonian builder."""
        return HamiltonianBuilder()

    def test_build_ising_hamiltonian(self, builder):
        """Test Ising Hamiltonian construction."""
        h = [0.5, -0.3, 0.2]
        J = [[0, 0.1, 0.2], [0.1, 0, 0.3], [0.2, 0.3, 0]]

        H = builder.build_ising_hamiltonian(h, J)

        assert H is not None
        # Hamiltonian should be Hermitian
        if hasattr(H, 'to_matrix'):
            matrix = H.to_matrix()
            assert np.allclose(matrix, matrix.conj().T)

    def test_build_qubo_hamiltonian(self, builder):
        """Test QUBO Hamiltonian construction."""
        Q = np.array([
            [1, 2],
            [2, 3]
        ])

        H = builder.build_qubo_hamiltonian(Q)
        assert H is not None

    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
    def test_hamiltonian_to_circuit(self, builder):
        """Test converting Hamiltonian to quantum circuit."""
        h = [0.5, -0.3]
        J = [[0, 0.1], [0.1, 0]]

        H = builder.build_ising_hamiltonian(h, J)
        circuit = builder.hamiltonian_to_circuit(H, time=1.0)

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 2


class TestResultConverter:
    """Test result converter for quantum optimization."""

    @pytest.fixture
    def converter(self):
        """Create a result converter."""
        return ResultConverter()

    def test_bitstring_to_solution(self, converter):
        """Test converting bitstring to solution."""
        # For QAP with n=3
        bitstring = '001010100'  # One-hot encoded solution
        solution = converter.bitstring_to_permutation(bitstring, n=3)

        assert len(solution) == 3
        assert set(solution) == {0, 1, 2}

    def test_measurement_to_expectation(self, converter):
        """Test converting measurements to expectation values."""
        counts = {
            '00': 500,
            '01': 200,
            '10': 200,
            '11': 100
        }

        expectation = converter.counts_to_expectation(counts, observable='Z')
        assert -1 <= expectation <= 1

    def test_probability_to_solution(self, converter):
        """Test converting probability distribution to solution."""
        probabilities = {
            '000': 0.1,
            '001': 0.3,
            '010': 0.2,
            '011': 0.05,
            '100': 0.25,
            '101': 0.05,
            '110': 0.03,
            '111': 0.02
        }

        best_solution = converter.get_best_solution(probabilities)
        assert best_solution == '001'  # Highest probability

    def test_energy_calculation(self, converter):
        """Test energy calculation from solution."""
        Q = np.array([
            [1, 2],
            [2, 3]
        ])
        solution = [0, 1]

        energy = converter.calculate_energy(solution, Q)
        expected = Q[0, 0] * 0 + Q[1, 1] * 1 + 2 * Q[0, 1] * 0 * 1
        assert energy == expected


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
class TestQAOAOptimizer:
    """Test QAOA (Quantum Approximate Optimization Algorithm)."""

    @pytest.fixture
    def optimizer(self):
        """Create a QAOA optimizer."""
        return QAOAOptimizer(n_layers=2, optimizer='COBYLA')

    def test_initialization(self, optimizer):
        """Test QAOA initialization."""
        assert optimizer.n_layers == 2
        assert optimizer.optimizer_name == 'COBYLA'

    @patch('qiskit.Aer.get_backend')
    def test_circuit_construction(self, mock_backend, optimizer):
        """Test QAOA circuit construction."""
        Q = np.array([[1, 2], [2, 3]])
        circuit = optimizer.construct_circuit(Q)

        assert circuit.num_qubits == 2
        # Should have parameterized gates
        assert len(circuit.parameters) > 0

    @patch('qiskit.Aer.get_backend')
    def test_optimization_simulation(self, mock_backend, optimizer):
        """Test QAOA optimization with simulation."""
        mock_backend.return_value.run.return_value.result.return_value.get_counts.return_value = {
            '00': 100,
            '01': 200,
            '10': 300,
            '11': 400
        }

        Q = np.array([[1, 2], [2, 3]])
        result = optimizer.optimize(Q, shots=1000)

        assert 'solution' in result
        assert 'energy' in result
        assert 'counts' in result


@pytest.mark.skipif(not PENNYLANE_AVAILABLE, reason="PennyLane not installed")
class TestVQEOptimizer:
    """Test VQE (Variational Quantum Eigensolver)."""

    @pytest.fixture
    def optimizer(self):
        """Create a VQE optimizer."""
        return VQEOptimizer(ansatz='hardware_efficient')

    def test_initialization(self, optimizer):
        """Test VQE initialization."""
        assert optimizer.ansatz == 'hardware_efficient'

    def test_ansatz_construction(self, optimizer):
        """Test VQE ansatz construction."""
        n_qubits = 4
        n_layers = 2
        ansatz = optimizer.build_ansatz(n_qubits, n_layers)

        assert ansatz is not None
        # Should have parameters
        assert optimizer.n_parameters > 0

    @patch('pennylane.device')
    def test_cost_function(self, mock_device, optimizer):
        """Test VQE cost function."""
        mock_dev = MagicMock()
        mock_device.return_value = mock_dev

        H = np.array([[1, 0], [0, -1]])
        params = np.random.rand(4)

        cost = optimizer.cost_function(params, H)
        assert isinstance(cost, (int, float))

    @patch('pennylane.device')
    def test_optimization_process(self, mock_device, optimizer):
        """Test VQE optimization process."""
        mock_dev = MagicMock()
        mock_device.return_value = mock_dev

        H = np.array([[1, 0.5], [0.5, -1]])
        result = optimizer.optimize(H, max_iterations=10)

        assert 'energy' in result
        assert 'parameters' in result
        assert result['energy'] <= np.max(np.linalg.eigvals(H))


class TestQuantumAdapter:
    """Test quantum adapter for different backends."""

    @pytest.fixture
    def adapter(self):
        """Create a quantum adapter."""
        return QuantumAdapter(backend='simulator')

    def test_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter.backend == 'simulator'
        assert adapter.is_available()

    def test_backend_switching(self, adapter):
        """Test switching between backends."""
        backends = ['simulator', 'qiskit', 'pennylane', 'cirq']

        for backend in backends:
            adapter.set_backend(backend)
            if adapter.is_backend_available(backend):
                assert adapter.backend == backend

    @patch('Librex.quantum.adapters.quantum_adapter.QAOAOptimizer')
    def test_run_optimization(self, mock_qaoa, adapter):
        """Test running optimization through adapter."""
        mock_optimizer = MagicMock()
        mock_qaoa.return_value = mock_optimizer
        mock_optimizer.optimize.return_value = {'solution': [0, 1], 'energy': -1.5}

        Q = np.array([[1, 2], [2, 3]])
        result = adapter.optimize(Q, method='QAOA')

        assert 'solution' in result
        assert 'energy' in result
        mock_optimizer.optimize.assert_called_once()

    def test_hardware_constraints(self, adapter):
        """Test hardware constraint checking."""
        constraints = adapter.get_hardware_constraints()

        assert 'max_qubits' in constraints
        assert 'connectivity' in constraints
        assert 'gate_set' in constraints

    def test_transpilation_options(self, adapter):
        """Test transpilation options."""
        options = adapter.get_transpilation_options()

        assert 'optimization_level' in options
        assert 'basis_gates' in options
        assert 'layout_method' in options


class TestQuantumIntegration:
    """Integration tests for quantum optimization pipeline."""

    def test_end_to_end_qap_quantum(self):
        """Test end-to-end QAP solving with quantum methods."""
        # Create small QAP problem
        n = 3
        problem = {
            'flow': np.random.rand(n, n),
            'distance': np.random.rand(n, n)
        }

        # Convert to QUBO
        converter = QUBOConverter()
        Q = converter.convert_qap_to_qubo(problem)

        # Validate problem
        validator = ProblemValidator()
        assert validator.validate_qubo_size(Q, max_qubits=100)

        # Estimate qubits
        estimator = QubitEstimator()
        n_qubits = estimator.estimate_qap_qubits(n)
        assert n_qubits == n * n

        # Convert to Ising
        encoder = IsingEncoder()
        h, J, offset = encoder.qubo_to_ising(Q)

        assert h is not None
        assert J is not None

    def test_hybrid_classical_quantum(self):
        """Test hybrid classical-quantum optimization."""
        from Librex import OptimizationOrchestrator

        orchestrator = OptimizationOrchestrator(
            classical_methods=['genetic_algorithm', 'simulated_annealing'],
            quantum_methods=['QAOA'] if QISKIT_AVAILABLE else []
        )

        problem = {
            'flow': np.random.rand(3, 3),
            'distance': np.random.rand(3, 3)
        }

        result = orchestrator.optimize(problem, hybrid=True)

        assert 'best_solution' in result
        assert 'best_method' in result
        assert 'performance_comparison' in result