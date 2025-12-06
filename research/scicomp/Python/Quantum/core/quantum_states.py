"""Quantum state operations including pure/mixed states and entanglement."""
import numpy as np
from typing import Union, List, Tuple, Optional, Dict
from scipy.linalg import sqrtm, logm
import scipy.sparse as sp
class QuantumState:
    """Base class for quantum state representations."""
    def __init__(self, data: Union[np.ndarray, sp.spmatrix], is_density_matrix: bool = False):
        """
        Initialize quantum state.
        Args:
            data: State vector or density matrix
            is_density_matrix: Whether data is density matrix
        """
        self.is_density_matrix = is_density_matrix
        if is_density_matrix:
            self.density_matrix = np.asarray(data)
            self.dim = self.density_matrix.shape[0]
            self._validate_density_matrix()
        else:
            self.state_vector = np.asarray(data, dtype=complex).flatten()
            self.dim = len(self.state_vector)
            self._normalize_state()
            self.density_matrix = np.outer(self.state_vector, self.state_vector.conj())
    def _normalize_state(self):
        """Normalize state vector."""
        norm = np.linalg.norm(self.state_vector)
        if norm > 1e-10:
            self.state_vector /= norm
    def is_normalized(self, tolerance: float = 1e-10) -> bool:
        """Check if state is normalized."""
        if self.is_density_matrix:
            trace = np.trace(self.density_matrix)
            return abs(trace - 1.0) < tolerance
        else:
            norm = np.linalg.norm(self.state_vector)
            return abs(norm - 1.0) < tolerance
    def _validate_density_matrix(self):
        """Validate density matrix properties."""
        # Check Hermiticity
        if not np.allclose(self.density_matrix, self.density_matrix.conj().T):
            raise ValueError("Density matrix must be Hermitian")
        # Check trace
        trace = np.trace(self.density_matrix)
        if not np.isclose(trace, 1.0):
            self.density_matrix /= trace
        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(self.density_matrix)
        if np.any(eigenvalues < -1e-10):
            raise ValueError("Density matrix must be positive semi-definite")
    def expectation_value(self, operator: np.ndarray) -> complex:
        """
        Calculate expectation value of operator.
        Args:
            operator: Hermitian operator
        Returns:
            Expectation value
        """
        return np.trace(self.density_matrix @ operator)
    def purity(self) -> float:
        """Calculate state purity."""
        return np.real(np.trace(self.density_matrix @ self.density_matrix))
    def von_neumann_entropy(self) -> float:
        """Calculate von Neumann entropy."""
        eigenvalues = np.linalg.eigvalsh(self.density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return -np.sum(eigenvalues * np.log(eigenvalues))
    def fidelity(self, other: 'QuantumState') -> float:
        """
        Calculate fidelity with another state.
        Args:
            other: Another quantum state
        Returns:
            Fidelity between states
        """
        if not self.is_density_matrix and not other.is_density_matrix:
            return np.abs(np.vdot(self.state_vector, other.state_vector))**2
        sqrt_rho1 = sqrtm(self.density_matrix)
        fid_op = sqrt_rho1 @ other.density_matrix @ sqrt_rho1
        return np.real(np.trace(sqrtm(fid_op))**2)
    def partial_trace(self, subsystem_dims: List[int], keep: List[int]) -> np.ndarray:
        """
        Calculate partial trace over subsystems.
        Args:
            subsystem_dims: Dimensions of each subsystem
            keep: Indices of subsystems to keep
        Returns:
            Reduced density matrix
        """
        n_subsystems = len(subsystem_dims)
        trace_out = [i for i in range(n_subsystems) if i not in keep]
        # Reshape density matrix
        shape = subsystem_dims + subsystem_dims
        rho_reshaped = self.density_matrix.reshape(shape)
        # Trace out subsystems
        for idx in sorted(trace_out, reverse=True):
            rho_reshaped = np.trace(rho_reshaped, axis1=idx, axis2=idx+n_subsystems)
            shape = list(shape)
            del shape[idx]
            del shape[idx+n_subsystems-1]
            if len(shape) > 0:
                rho_reshaped = rho_reshaped.reshape(shape)
        # Reshape back to matrix
        kept_dim = np.prod([subsystem_dims[i] for i in keep])
        return rho_reshaped.reshape(kept_dim, kept_dim)
class EntanglementMeasures:
    """Tools for quantifying quantum entanglement."""
    @staticmethod
    def concurrence(state: QuantumState, subsystem_dims: List[int] = [2, 2]) -> float:
        """
        Calculate concurrence for two-qubit system.
        Args:
            state: Quantum state
            subsystem_dims: Dimensions of subsystems
        Returns:
            Concurrence value
        """
        if np.prod(subsystem_dims) != 4:
            raise ValueError("Concurrence only defined for two-qubit systems")
        # Pauli-Y matrix
        sigma_y = np.array([[0, -1j], [1j, 0]])
        spin_flip = np.kron(sigma_y, sigma_y)
        # Calculate concurrence
        rho = state.density_matrix
        rho_tilde = spin_flip @ rho.conj() @ spin_flip
        # Calculate sqrt(rho) carefully
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
        sqrt_rho = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.conj().T
        # Calculate R matrix
        R_matrix = sqrt_rho @ rho_tilde @ sqrt_rho
        eigenvals_R = np.linalg.eigvals(R_matrix)
        eigenvals_R = np.sqrt(np.maximum(np.real(eigenvals_R), 0))
        eigenvals_R = np.sort(eigenvals_R)[::-1]
        return max(0, eigenvals_R[0] - eigenvals_R[1] - eigenvals_R[2] - eigenvals_R[3])
    @staticmethod
    def entanglement_entropy(state: QuantumState, subsystem_dims: List[int],
                            partition: List[int]) -> float:
        """
        Calculate entanglement entropy across partition.
        Args:
            state: Quantum state
            subsystem_dims: Dimensions of subsystems
            partition: Indices of first partition
        Returns:
            Entanglement entropy
        """
        rho_reduced = state.partial_trace(subsystem_dims, partition)
        eigenvalues = np.linalg.eigvalsh(rho_reduced)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return -np.sum(eigenvalues * np.log(eigenvalues))
    @staticmethod
    def negativity(state: QuantumState, subsystem_dims: List[int],
                   partition: List[int]) -> float:
        """
        Calculate negativity across partition.
        Args:
            state: Quantum state
            subsystem_dims: Dimensions of subsystems
            partition: Indices of first partition
        Returns:
            Negativity
        """
        # Partial transpose
        n_subsystems = len(subsystem_dims)
        shape = subsystem_dims + subsystem_dims
        rho_reshaped = state.density_matrix.reshape(shape)
        # Transpose subsystems in partition
        axes = list(range(2 * n_subsystems))
        for idx in partition:
            axes[idx], axes[idx + n_subsystems] = axes[idx + n_subsystems], axes[idx]
        rho_pt = np.transpose(rho_reshaped, axes)
        total_dim = np.prod(subsystem_dims)
        rho_pt = rho_pt.reshape(total_dim, total_dim)
        # Calculate negativity
        eigenvalues = np.linalg.eigvals(rho_pt)
        return np.sum(np.abs(eigenvalues[eigenvalues < 0]))
class BellStates:
    """Common Bell states and operations."""
    @staticmethod
    def phi_plus() -> QuantumState:
        """Create |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
        state = np.zeros(4, dtype=complex)
        state[0] = 1/np.sqrt(2)  # |00⟩
        state[3] = 1/np.sqrt(2)  # |11⟩
        return QuantumState(state)
    @staticmethod
    def phi_minus() -> QuantumState:
        """Create |Φ-⟩ = (|00⟩ - |11⟩)/√2."""
        state = np.zeros(4, dtype=complex)
        state[0] = 1/np.sqrt(2)   # |00⟩
        state[3] = -1/np.sqrt(2)  # |11⟩
        return QuantumState(state)
    @staticmethod
    def psi_plus() -> QuantumState:
        """Create |Ψ+⟩ = (|01⟩ + |10⟩)/√2."""
        state = np.zeros(4, dtype=complex)
        state[1] = 1/np.sqrt(2)  # |01⟩
        state[2] = 1/np.sqrt(2)  # |10⟩
        return QuantumState(state)
    @staticmethod
    def psi_minus() -> QuantumState:
        """Create |Ψ-⟩ = (|01⟩ - |10⟩)/√2."""
        state = np.zeros(4, dtype=complex)
        state[1] = 1/np.sqrt(2)   # |01⟩
        state[2] = -1/np.sqrt(2)  # |10⟩
        return QuantumState(state)
class GHZStates:
    """Greenberger-Horne-Zeilinger states."""
    @staticmethod
    def ghz_state(n_qubits: int) -> QuantumState:
        """
        Create n-qubit GHZ state.
        Args:
            n_qubits: Number of qubits
        Returns:
            GHZ state
        """
        dim = 2**n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1/np.sqrt(2)        # |000...0⟩
        state[dim-1] = 1/np.sqrt(2)    # |111...1⟩
        return QuantumState(state)
    @staticmethod
    def w_state(n_qubits: int) -> QuantumState:
        """
        Create n-qubit W state.
        Args:
            n_qubits: Number of qubits
        Returns:
            W state
        """
        dim = 2**n_qubits
        state = np.zeros(dim, dtype=complex)
        # W state is equal superposition of states with single excitation
        for i in range(n_qubits):
            idx = 2**i
            state[idx] = 1/np.sqrt(n_qubits)
        return QuantumState(state)
class QuantumStateTomography:
    """Quantum state reconstruction from measurements."""
    @staticmethod
    def pauli_basis(n_qubits: int) -> List[np.ndarray]:
        """
        Generate Pauli basis for n qubits.
        Args:
            n_qubits: Number of qubits
        Returns:
            List of Pauli operators
        """
        pauli_1 = [
            np.eye(2),                          # I
            np.array([[0, 1], [1, 0]]),         # X
            np.array([[0, -1j], [1j, 0]]),      # Y
            np.array([[1, 0], [0, -1]])         # Z
        ]
        if n_qubits == 1:
            return pauli_1
        # Generate tensor products
        basis = []
        indices = np.ndindex(*([4] * n_qubits))
        for idx in indices:
            operator = pauli_1[idx[0]]
            for i in range(1, n_qubits):
                operator = np.kron(operator, pauli_1[idx[i]])
            basis.append(operator)
        return basis
    @staticmethod
    def linear_inversion(measurements: Dict[str, float], n_qubits: int) -> QuantumState:
        """
        Reconstruct state using linear inversion.
        Args:
            measurements: Dictionary of Pauli measurements
            n_qubits: Number of qubits
        Returns:
            Reconstructed quantum state
        """
        dim = 2**n_qubits
        basis = QuantumStateTomography.pauli_basis(n_qubits)
        # Reconstruct density matrix
        rho = np.zeros((dim, dim), dtype=complex)
        for i, pauli in enumerate(basis):
            key = f"P{i}"
            if key in measurements:
                rho += measurements[key] * pauli / dim
        return QuantumState(rho, is_density_matrix=True)
    @staticmethod
    def maximum_likelihood(measurements: np.ndarray, projectors: List[np.ndarray],
                          counts: np.ndarray, max_iter: int = 1000) -> QuantumState:
        """
        Maximum likelihood state tomography.
        Args:
            measurements: Measurement outcomes
            projectors: Measurement projectors
            counts: Number of measurements for each projector
            max_iter: Maximum iterations
        Returns:
            Reconstructed state
        """
        dim = projectors[0].shape[0]
        # Initialize with maximally mixed state
        rho = np.eye(dim) / dim
        for _ in range(max_iter):
            # E-step: calculate likelihood
            probs = np.array([np.real(np.trace(rho @ P)) for P in projectors])
            probs = np.maximum(probs, 1e-10)
            # M-step: update density matrix
            R = np.zeros((dim, dim), dtype=complex)
            for i, P in enumerate(projectors):
                R += counts[i] * measurements[i] / probs[i] * P
            # Apply transformation
            rho_sqrt = sqrtm(rho)
            G = sqrtm(rho_sqrt @ R @ rho_sqrt)
            rho_new = G @ np.linalg.inv(rho_sqrt)
            rho_new = rho_new @ rho_new.conj().T
            # Normalize
            rho_new /= np.trace(rho_new)
            # Check convergence
            if np.linalg.norm(rho_new - rho) < 1e-6:
                break
            rho = rho_new
        return QuantumState(rho, is_density_matrix=True)