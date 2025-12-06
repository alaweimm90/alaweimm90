"""
Quantum operators and gates.
This module provides quantum operators including:
- Pauli matrices
- Common quantum gates
- Operator utilities
- Time evolution operators
"""
import numpy as np
from typing import Union, List, Tuple, Optional
from scipy.linalg import expm
import scipy.sparse as sp
class PauliOperators:
    """Pauli matrices and operations."""
    # Single-qubit Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    @classmethod
    def pauli_string(cls, operators: str) -> np.ndarray:
        """
        Create multi-qubit Pauli operator from string.
        Args:
            operators: String of Pauli operators (e.g., 'XYZ')
        Returns:
            Multi-qubit Pauli operator
        """
        pauli_map = {'I': cls.I, 'X': cls.X, 'Y': cls.Y, 'Z': cls.Z}
        if not operators:
            return np.array([[1]], dtype=complex)
        result = pauli_map[operators[0].upper()]
        for op in operators[1:]:
            result = np.kron(result, pauli_map[op.upper()])
        return result
    @classmethod
    def commutator(cls, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Calculate commutator [A, B] = AB - BA."""
        return A @ B - B @ A
    @classmethod
    def anticommutator(cls, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Calculate anticommutator {A, B} = AB + BA."""
        return A @ B + B @ A
class QuantumGates:
    """Common quantum gates."""
    # Single-qubit gates
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # Hadamard
    S = np.array([[1, 0], [0, 1j]], dtype=complex)                # Phase
    T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)  # π/8
    @staticmethod
    def Rx(theta: float) -> np.ndarray:
        """Rotation around X-axis."""
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
    @staticmethod
    def Ry(theta: float) -> np.ndarray:
        """Rotation around Y-axis."""
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    @staticmethod
    def Rz(theta: float) -> np.ndarray:
        """Rotation around Z-axis."""
        return np.array([[np.exp(-1j*theta/2), 0],
                        [0, np.exp(1j*theta/2)]], dtype=complex)
    @staticmethod
    def U3(theta: float, phi: float, lam: float) -> np.ndarray:
        """General single-qubit unitary."""
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        return np.array([
            [c, -np.exp(1j*lam)*s],
            [np.exp(1j*phi)*s, np.exp(1j*(phi+lam))*c]
        ], dtype=complex)
    @staticmethod
    def CNOT() -> np.ndarray:
        """Controlled-NOT gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
    @staticmethod
    def CZ() -> np.ndarray:
        """Controlled-Z gate."""
        return np.diag([1, 1, 1, -1]).astype(complex)
    @staticmethod
    def SWAP() -> np.ndarray:
        """SWAP gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    @staticmethod
    def Toffoli() -> np.ndarray:
        """Toffoli (CCNOT) gate."""
        gate = np.eye(8, dtype=complex)
        gate[6, 6] = 0
        gate[6, 7] = 1
        gate[7, 6] = 1
        gate[7, 7] = 0
        return gate
    @staticmethod
    def Fredkin() -> np.ndarray:
        """Fredkin (controlled-SWAP) gate."""
        gate = np.eye(8, dtype=complex)
        gate[5, 5] = 0
        gate[5, 6] = 1
        gate[6, 5] = 1
        gate[6, 6] = 0
        return gate
    @staticmethod
    def controlled_gate(gate: np.ndarray, control_qubits: int = 1) -> np.ndarray:
        """
        Create controlled version of gate.
        Args:
            gate: Unitary gate to control
            control_qubits: Number of control qubits
        Returns:
            Controlled gate
        """
        n = gate.shape[0]
        m = 2**control_qubits
        # Create controlled gate
        controlled = np.eye(m * n, dtype=complex)
        controlled[-n:, -n:] = gate
        return controlled
class HamiltonianOperators:
    """Common Hamiltonian operators."""
    @staticmethod
    def ising_1d(n_sites: int, J: float = 1.0, h: float = 0.0,
                  periodic: bool = False) -> np.ndarray:
        """
        1D Ising model Hamiltonian.
        Args:
            n_sites: Number of sites
            J: Coupling strength
            h: Transverse field strength
            periodic: Whether to use periodic boundary conditions
        Returns:
            Hamiltonian matrix
        """
        dim = 2**n_sites
        H = np.zeros((dim, dim), dtype=complex)
        # Interaction terms
        for i in range(n_sites - 1):
            Zi_Zi1 = PauliOperators.pauli_string('I' * i + 'ZZ' + 'I' * (n_sites - i - 2))
            H -= J * Zi_Zi1
        # Periodic boundary condition
        if periodic and n_sites > 2:
            Zn_Z0 = PauliOperators.pauli_string('Z' + 'I' * (n_sites - 2) + 'Z')
            H -= J * Zn_Z0
        # Transverse field
        for i in range(n_sites):
            Xi = PauliOperators.pauli_string('I' * i + 'X' + 'I' * (n_sites - i - 1))
            H -= h * Xi
        return H
    @staticmethod
    def heisenberg_1d(n_sites: int, J: Union[float, List[float]] = 1.0,
                      periodic: bool = False) -> np.ndarray:
        """
        1D Heisenberg model Hamiltonian.
        Args:
            n_sites: Number of sites
            J: Coupling strength(s) [Jx, Jy, Jz] or scalar
            periodic: Whether to use periodic boundary conditions
        Returns:
            Hamiltonian matrix
        """
        if isinstance(J, (int, float)):
            Jx = Jy = Jz = J
        else:
            Jx, Jy, Jz = J
        dim = 2**n_sites
        H = np.zeros((dim, dim), dtype=complex)
        # Nearest-neighbor interactions
        for i in range(n_sites - 1):
            # XX interaction
            XX = PauliOperators.pauli_string('I' * i + 'XX' + 'I' * (n_sites - i - 2))
            H += Jx * XX
            # YY interaction
            YY = PauliOperators.pauli_string('I' * i + 'YY' + 'I' * (n_sites - i - 2))
            H += Jy * YY
            # ZZ interaction
            ZZ = PauliOperators.pauli_string('I' * i + 'ZZ' + 'I' * (n_sites - i - 2))
            H += Jz * ZZ
        # Periodic boundary condition
        if periodic and n_sites > 2:
            # X_n X_0
            XX_n0 = PauliOperators.pauli_string('X' + 'I' * (n_sites - 2) + 'X')
            H += Jx * XX_n0
            # Y_n Y_0
            YY_n0 = PauliOperators.pauli_string('Y' + 'I' * (n_sites - 2) + 'Y')
            H += Jy * YY_n0
            # Z_n Z_0
            ZZ_n0 = PauliOperators.pauli_string('Z' + 'I' * (n_sites - 2) + 'Z')
            H += Jz * ZZ_n0
        return H
    @staticmethod
    def hubbard_1d(n_sites: int, t: float = 1.0, U: float = 1.0,
                   periodic: bool = False) -> np.ndarray:
        """
        1D Hubbard model Hamiltonian.
        Args:
            n_sites: Number of sites
            t: Hopping parameter
            U: On-site interaction
            periodic: Whether to use periodic boundary conditions
        Returns:
            Hamiltonian matrix
        """
        # Each site has 4 states: |0⟩, |↑⟩, |↓⟩, |↑↓⟩
        dim = 4**n_sites
        H = sp.lil_matrix((dim, dim), dtype=complex)
        # Create basis states
        for state in range(dim):
            config = []
            temp = state
            for _ in range(n_sites):
                config.append(temp % 4)
                temp //= 4
            # Hopping terms
            for i in range(n_sites - 1):
                # Spin up hopping
                if config[i] in [1, 3] and config[i+1] in [0, 2]:
                    new_config = config.copy()
                    new_config[i] -= 1 if config[i] == 1 else 2
                    new_config[i+1] += 1 if config[i+1] == 0 else 2
                    new_state = sum(c * 4**j for j, c in enumerate(new_config))
                    H[new_state, state] -= t
                    H[state, new_state] -= t
                # Spin down hopping
                if config[i] in [2, 3] and config[i+1] in [0, 1]:
                    new_config = config.copy()
                    new_config[i] -= 2
                    new_config[i+1] += 2
                    new_state = sum(c * 4**j for j, c in enumerate(new_config))
                    H[new_state, state] -= t
                    H[state, new_state] -= t
            # On-site interaction
            for i in range(n_sites):
                if config[i] == 3:  # Both spins on site
                    H[state, state] += U
        return H.toarray()
class TimeEvolution:
    """Time evolution operators and propagators."""
    @staticmethod
    def unitary_evolution(H: np.ndarray, t: float, hbar: float = 1.0) -> np.ndarray:
        """
        Unitary time evolution operator U(t) = exp(-iHt/ℏ).
        Args:
            H: Hamiltonian
            t: Time
            hbar: Reduced Planck constant
        Returns:
            Time evolution operator
        """
        return expm(-1j * H * t / hbar)
    @staticmethod
    def trotter_suzuki(H_list: List[np.ndarray], t: float, n_steps: int = 100,
                       order: int = 2) -> np.ndarray:
        """
        Trotter-Suzuki decomposition for time evolution.
        Args:
            H_list: List of Hamiltonian terms
            t: Total time
            n_steps: Number of Trotter steps
            order: Order of decomposition (1 or 2)
        Returns:
            Approximate time evolution operator
        """
        dt = t / n_steps
        dim = H_list[0].shape[0]
        U = np.eye(dim, dtype=complex)
        if order == 1:
            # First-order Trotter
            for _ in range(n_steps):
                for H in H_list:
                    U = expm(-1j * H * dt) @ U
        elif order == 2:
            # Second-order Trotter (Strang splitting)
            for _ in range(n_steps):
                # Forward sweep
                for H in H_list[:-1]:
                    U = expm(-1j * H * dt/2) @ U
                # Last term with full step
                U = expm(-1j * H_list[-1] * dt) @ U
                # Backward sweep
                for H in reversed(H_list[:-1]):
                    U = expm(-1j * H * dt/2) @ U
        return U
    @staticmethod
    def magnus_expansion(H: callable, t: float, order: int = 2) -> np.ndarray:
        """
        Magnus expansion for time-dependent Hamiltonians.
        Args:
            H: Time-dependent Hamiltonian H(t)
            t: Time
            order: Order of Magnus expansion
        Returns:
            Time evolution operator
        """
        # Sample Hamiltonian at different times
        n_samples = 10 * order
        dt = t / n_samples
        times = np.linspace(0, t, n_samples)
        # First-order Magnus
        Omega = np.zeros_like(H(0), dtype=complex)
        for ti in times[:-1]:
            Omega += H(ti) * dt
        Omega *= -1j
        if order >= 2:
            # Second-order correction
            Omega2 = np.zeros_like(H(0), dtype=complex)
            for i, ti in enumerate(times[:-1]):
                for j, tj in enumerate(times[:i]):
                    Omega2 += PauliOperators.commutator(H(ti), H(tj)) * dt**2
            Omega += Omega2 / 2
        return expm(Omega)
class OperatorMeasurements:
    """Measurement operations and expectation values."""
    @staticmethod
    def measure_pauli(state_vector: np.ndarray, pauli_string: str) -> float:
        """
        Measure Pauli operator expectation value.
        Args:
            state_vector: Quantum state vector
            pauli_string: Pauli operator string
        Returns:
            Expectation value
        """
        operator = PauliOperators.pauli_string(pauli_string)
        return np.real(np.vdot(state_vector, operator @ state_vector))
    @staticmethod
    def projective_measurement(state_vector: np.ndarray, basis: List[np.ndarray]) -> Tuple[int, np.ndarray]:
        """
        Perform projective measurement.
        Args:
            state_vector: Quantum state vector
            basis: Measurement basis vectors
        Returns:
            Measurement outcome and collapsed state
        """
        # Calculate probabilities
        probs = [np.abs(np.vdot(b, state_vector))**2 for b in basis]
        probs = np.array(probs)
        probs /= np.sum(probs)
        # Sample outcome
        outcome = np.random.choice(len(basis), p=probs)
        # Collapse state
        collapsed = basis[outcome] * np.vdot(basis[outcome], state_vector)
        collapsed /= np.linalg.norm(collapsed)
        return outcome, collapsed
    @staticmethod
    def povm_measurement(state_vector: np.ndarray, povm_elements: List[np.ndarray]) -> int:
        """
        Perform POVM measurement.
        Args:
            state_vector: Quantum state vector
            povm_elements: POVM elements
        Returns:
            Measurement outcome
        """
        # Calculate probabilities
        probs = [np.real(np.vdot(state_vector, E @ state_vector))
                for E in povm_elements]
        probs = np.array(probs)
        probs /= np.sum(probs)
        return np.random.choice(len(povm_elements), p=probs)