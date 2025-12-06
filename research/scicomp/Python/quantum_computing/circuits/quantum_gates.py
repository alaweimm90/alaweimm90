#!/usr/bin/env python3
"""
Quantum Gates and Circuit Construction
Comprehensive implementation of quantum gates, multi-qubit operations, and
quantum circuit construction tools for quantum computing simulations and
algorithm development.
Key Features:
- Complete set of single and multi-qubit gates
- Parameterized gates and controlled operations
- Circuit composition and optimization
- Gate decomposition and synthesis
- Noise models and error simulation
Applications:
- Quantum algorithm development
- Quantum circuit optimization
- Quantum error correction
- Quantum compilation and transpilation
- Educational quantum computing tools
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Callable, Dict, List, Any
from dataclasses import dataclass
import warnings
from abc import ABC, abstractmethod
from ...utils.constants import hbar
from ...visualization.berkeley_style import BerkeleyPlot
@dataclass
class GateConfig:
    """Configuration for quantum gate operations."""
    # Precision parameters
    angle_precision: float = 1e-10
    amplitude_precision: float = 1e-12
    # Noise parameters
    gate_error_rate: float = 0.0
    decoherence_time: float = np.inf  # T1, T2 times
    # Optimization parameters
    optimize_circuits: bool = True
    max_gate_depth: int = 100
    # Simulation parameters
    use_sparse_matrices: bool = False
    parallel_execution: bool = False
class QuantumGate(ABC):
    """Abstract base class for quantum gates."""
    def __init__(self, n_qubits: int, name: str):
        """Initialize quantum gate."""
        self.n_qubits = n_qubits
        self.name = name
        self.matrix = None
        self.parameters = {}
    @abstractmethod
    def get_matrix(self) -> np.ndarray:
        """Get the matrix representation of the gate."""
        pass
    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply gate to quantum state."""
        return self.get_matrix() @ state
    def dagger(self) -> 'QuantumGate':
        """Return Hermitian conjugate of the gate."""
        conjugate_gate = type(self)(**self.parameters)
        conjugate_gate.matrix = self.get_matrix().conj().T
        conjugate_gate.name = self.name + "†"
        return conjugate_gate
    def __str__(self) -> str:
        return f"{self.name}({self.n_qubits} qubits)"
class SingleQubitGate(QuantumGate):
    """Base class for single-qubit gates."""
    def __init__(self, name: str):
        super().__init__(1, name)
class PauliX(SingleQubitGate):
    """Pauli-X (NOT) gate."""
    def __init__(self):
        super().__init__("X")
    def get_matrix(self) -> np.ndarray:
        return np.array([[0, 1], [1, 0]], dtype=complex)
class PauliY(SingleQubitGate):
    """Pauli-Y gate."""
    def __init__(self):
        super().__init__("Y")
    def get_matrix(self) -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
class PauliZ(SingleQubitGate):
    """Pauli-Z gate."""
    def __init__(self):
        super().__init__("Z")
    def get_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1]], dtype=complex)
class Hadamard(SingleQubitGate):
    """Hadamard gate."""
    def __init__(self):
        super().__init__("H")
    def get_matrix(self) -> np.ndarray:
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
class SGate(SingleQubitGate):
    """S gate (phase gate)."""
    def __init__(self):
        super().__init__("S")
    def get_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, 1j]], dtype=complex)
class TGate(SingleQubitGate):
    """T gate (π/8 gate)."""
    def __init__(self):
        super().__init__("T")
    def get_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
class RXGate(SingleQubitGate):
    """Rotation around X-axis."""
    def __init__(self, theta: float):
        super().__init__("RX")
        self.theta = theta
        self.parameters = {'theta': theta}
    def get_matrix(self) -> np.ndarray:
        cos_half = np.cos(self.theta / 2)
        sin_half = np.sin(self.theta / 2)
        return np.array([
            [cos_half, -1j * sin_half],
            [-1j * sin_half, cos_half]
        ], dtype=complex)
class RYGate(SingleQubitGate):
    """Rotation around Y-axis."""
    def __init__(self, theta: float):
        super().__init__("RY")
        self.theta = theta
        self.parameters = {'theta': theta}
    def get_matrix(self) -> np.ndarray:
        cos_half = np.cos(self.theta / 2)
        sin_half = np.sin(self.theta / 2)
        return np.array([
            [cos_half, -sin_half],
            [sin_half, cos_half]
        ], dtype=complex)
class RZGate(SingleQubitGate):
    """Rotation around Z-axis."""
    def __init__(self, phi: float):
        super().__init__("RZ")
        self.phi = phi
        self.parameters = {'phi': phi}
    def get_matrix(self) -> np.ndarray:
        return np.array([
            [np.exp(-1j * self.phi / 2), 0],
            [0, np.exp(1j * self.phi / 2)]
        ], dtype=complex)
class PhaseGate(SingleQubitGate):
    """General phase gate."""
    def __init__(self, phi: float):
        super().__init__("Phase")
        self.phi = phi
        self.parameters = {'phi': phi}
    def get_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0],
            [0, np.exp(1j * self.phi)]
        ], dtype=complex)
class U3Gate(SingleQubitGate):
    """General single-qubit unitary gate (3 parameters)."""
    def __init__(self, theta: float, phi: float, lam: float):
        super().__init__("U3")
        self.theta = theta
        self.phi = phi
        self.lam = lam
        self.parameters = {'theta': theta, 'phi': phi, 'lambda': lam}
    def get_matrix(self) -> np.ndarray:
        cos_half = np.cos(self.theta / 2)
        sin_half = np.sin(self.theta / 2)
        return np.array([
            [cos_half, -np.exp(1j * self.lam) * sin_half],
            [np.exp(1j * self.phi) * sin_half,
             np.exp(1j * (self.phi + self.lam)) * cos_half]
        ], dtype=complex)
class TwoQubitGate(QuantumGate):
    """Base class for two-qubit gates."""
    def __init__(self, name: str):
        super().__init__(2, name)
class CNOT(TwoQubitGate):
    """Controlled-NOT gate."""
    def __init__(self):
        super().__init__("CNOT")
    def get_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
class CZ(TwoQubitGate):
    """Controlled-Z gate."""
    def __init__(self):
        super().__init__("CZ")
    def get_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
class SWAP(TwoQubitGate):
    """SWAP gate."""
    def __init__(self):
        super().__init__("SWAP")
    def get_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
class iSWAP(TwoQubitGate):
    """iSWAP gate."""
    def __init__(self):
        super().__init__("iSWAP")
    def get_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
class RXXGate(TwoQubitGate):
    """RXX parametric gate."""
    def __init__(self, theta: float):
        super().__init__("RXX")
        self.theta = theta
        self.parameters = {'theta': theta}
    def get_matrix(self) -> np.ndarray:
        cos_half = np.cos(self.theta / 2)
        sin_half = np.sin(self.theta / 2)
        return np.array([
            [cos_half, 0, 0, -1j * sin_half],
            [0, cos_half, -1j * sin_half, 0],
            [0, -1j * sin_half, cos_half, 0],
            [-1j * sin_half, 0, 0, cos_half]
        ], dtype=complex)
class RYYGate(TwoQubitGate):
    """RYY parametric gate."""
    def __init__(self, theta: float):
        super().__init__("RYY")
        self.theta = theta
        self.parameters = {'theta': theta}
    def get_matrix(self) -> np.ndarray:
        cos_half = np.cos(self.theta / 2)
        sin_half = np.sin(self.theta / 2)
        return np.array([
            [cos_half, 0, 0, 1j * sin_half],
            [0, cos_half, -1j * sin_half, 0],
            [0, -1j * sin_half, cos_half, 0],
            [1j * sin_half, 0, 0, cos_half]
        ], dtype=complex)
class RZZGate(TwoQubitGate):
    """RZZ parametric gate."""
    def __init__(self, theta: float):
        super().__init__("RZZ")
        self.theta = theta
        self.parameters = {'theta': theta}
    def get_matrix(self) -> np.ndarray:
        exp_pos = np.exp(1j * self.theta / 2)
        exp_neg = np.exp(-1j * self.theta / 2)
        return np.array([
            [exp_neg, 0, 0, 0],
            [0, exp_pos, 0, 0],
            [0, 0, exp_pos, 0],
            [0, 0, 0, exp_neg]
        ], dtype=complex)
class QuantumCircuit:
    """
    Quantum circuit representation and manipulation.
    Provides tools for constructing, optimizing, and simulating
    quantum circuits with various gate sets and noise models.
    """
    def __init__(self, n_qubits: int, config: Optional[GateConfig] = None):
        """Initialize quantum circuit."""
        self.n_qubits = n_qubits
        self.config = config or GateConfig()
        # Circuit representation
        self.gates = []  # List of (gate, qubits) tuples
        self.measurements = []
        # State tracking
        self.n_states = 2**n_qubits
        self.current_state = None
        # Circuit statistics
        self.depth = 0
        self.gate_counts = {}
    def add_gate(self, gate: QuantumGate, qubits: Union[int, List[int]]) -> 'QuantumCircuit':
        """
        Add gate to circuit.
        Parameters
        ----------
        gate : QuantumGate
            Gate to add
        qubits : int or List[int]
            Target qubit(s)
        Returns
        -------
        QuantumCircuit
            Self for method chaining
        """
        if isinstance(qubits, int):
            qubits = [qubits]
        # Validate qubit indices
        for q in qubits:
            if q < 0 or q >= self.n_qubits:
                raise ValueError(f"Invalid qubit index: {q}")
        if len(qubits) != gate.n_qubits:
            raise ValueError(f"Gate requires {gate.n_qubits} qubits, got {len(qubits)}")
        # Add to circuit
        self.gates.append((gate, qubits))
        # Update statistics
        self.gate_counts[gate.name] = self.gate_counts.get(gate.name, 0) + 1
        return self
    def h(self, qubit: int) -> 'QuantumCircuit':
        """Add Hadamard gate."""
        return self.add_gate(Hadamard(), qubit)
    def x(self, qubit: int) -> 'QuantumCircuit':
        """Add Pauli-X gate."""
        return self.add_gate(PauliX(), qubit)
    def y(self, qubit: int) -> 'QuantumCircuit':
        """Add Pauli-Y gate."""
        return self.add_gate(PauliY(), qubit)
    def z(self, qubit: int) -> 'QuantumCircuit':
        """Add Pauli-Z gate."""
        return self.add_gate(PauliZ(), qubit)
    def s(self, qubit: int) -> 'QuantumCircuit':
        """Add S gate."""
        return self.add_gate(SGate(), qubit)
    def t(self, qubit: int) -> 'QuantumCircuit':
        """Add T gate."""
        return self.add_gate(TGate(), qubit)
    def rx(self, qubit: int, theta: float) -> 'QuantumCircuit':
        """Add RX rotation gate."""
        return self.add_gate(RXGate(theta), qubit)
    def ry(self, qubit: int, theta: float) -> 'QuantumCircuit':
        """Add RY rotation gate."""
        return self.add_gate(RYGate(theta), qubit)
    def rz(self, qubit: int, phi: float) -> 'QuantumCircuit':
        """Add RZ rotation gate."""
        return self.add_gate(RZGate(phi), qubit)
    def phase(self, qubit: int, phi: float) -> 'QuantumCircuit':
        """Add phase gate."""
        return self.add_gate(PhaseGate(phi), qubit)
    def u3(self, qubit: int, theta: float, phi: float, lam: float) -> 'QuantumCircuit':
        """Add general U3 gate."""
        return self.add_gate(U3Gate(theta, phi, lam), qubit)
    def cnot(self, control: int, target: int) -> 'QuantumCircuit':
        """Add CNOT gate."""
        return self.add_gate(CNOT(), [control, target])
    def cz(self, control: int, target: int) -> 'QuantumCircuit':
        """Add CZ gate."""
        return self.add_gate(CZ(), [control, target])
    def swap(self, qubit1: int, qubit2: int) -> 'QuantumCircuit':
        """Add SWAP gate."""
        return self.add_gate(SWAP(), [qubit1, qubit2])
    def controlled_gate(self, gate: QuantumGate, control: int, target: int) -> 'QuantumCircuit':
        """
        Add controlled version of single-qubit gate.
        Parameters
        ----------
        gate : QuantumGate
            Single-qubit gate to control
        control : int
            Control qubit
        target : int
            Target qubit
        Returns
        -------
        QuantumCircuit
            Self for method chaining
        """
        if gate.n_qubits != 1:
            raise ValueError("Can only create controlled versions of single-qubit gates")
        # Create controlled gate matrix
        gate_matrix = gate.get_matrix()
        controlled_matrix = np.eye(4, dtype=complex)
        controlled_matrix[2:, 2:] = gate_matrix
        # Create controlled gate object
        class ControlledGate(TwoQubitGate):
            def __init__(self, base_gate):
                super().__init__(f"C{base_gate.name}")
                self.base_gate = base_gate
                self._matrix = controlled_matrix
            def get_matrix(self):
                return self._matrix
        controlled = ControlledGate(gate)
        return self.add_gate(controlled, [control, target])
    def toffoli(self, control1: int, control2: int, target: int) -> 'QuantumCircuit':
        """Add Toffoli (CCNOT) gate."""
        # Implement using decomposition for now
        # Full Toffoli would require 3-qubit gate class
        return (self
                .h(target)
                .cnot(control2, target)
                .t(target).dagger()
                .cnot(control1, target)
                .t(target)
                .cnot(control2, target)
                .t(target).dagger()
                .cnot(control1, target)
                .t(control2)
                .t(target)
                .cnot(control1, control2)
                .h(target)
                .t(control1)
                .t(control2).dagger()
                .cnot(control1, control2))
    def get_unitary_matrix(self) -> np.ndarray:
        """
        Get the full unitary matrix for the circuit.
        Returns
        -------
        np.ndarray
            Circuit unitary matrix
        """
        circuit_matrix = np.eye(self.n_states, dtype=complex)
        for gate, qubits in self.gates:
            gate_matrix = self._expand_gate_matrix(gate.get_matrix(), qubits)
            circuit_matrix = gate_matrix @ circuit_matrix
        return circuit_matrix
    def _expand_gate_matrix(self, gate_matrix: np.ndarray,
                           qubits: List[int]) -> np.ndarray:
        """Expand gate matrix to full Hilbert space."""
        if len(qubits) == 1:
            return self._expand_single_qubit_gate(gate_matrix, qubits[0])
        elif len(qubits) == 2:
            return self._expand_two_qubit_gate(gate_matrix, qubits)
        else:
            raise NotImplementedError("Multi-qubit gates > 2 not yet implemented")
    def _expand_single_qubit_gate(self, gate_matrix: np.ndarray,
                                 qubit: int) -> np.ndarray:
        """Expand single-qubit gate to full space using tensor products."""
        expanded = np.array([[1]], dtype=complex)
        for i in range(self.n_qubits):
            if i == qubit:
                expanded = np.kron(expanded, gate_matrix)
            else:
                expanded = np.kron(expanded, np.eye(2))
        return expanded
    def _expand_two_qubit_gate(self, gate_matrix: np.ndarray,
                              qubits: List[int]) -> np.ndarray:
        """Expand two-qubit gate to full space."""
        if len(qubits) != 2:
            raise ValueError("Expected exactly 2 qubits")
        q1, q2 = sorted(qubits)
        # Create full space matrix
        full_matrix = np.zeros((self.n_states, self.n_states), dtype=complex)
        # Map gate matrix elements to full space
        for i in range(self.n_states):
            for j in range(self.n_states):
                # Extract bit values for relevant qubits
                i_bits = [(i >> k) & 1 for k in range(self.n_qubits)]
                j_bits = [(j >> k) & 1 for k in range(self.n_qubits)]
                # Check if other qubits are unchanged
                other_unchanged = all(i_bits[k] == j_bits[k]
                                    for k in range(self.n_qubits)
                                    if k not in qubits)
                if other_unchanged:
                    # Map to gate matrix indices
                    i_gate = i_bits[q2] * 2 + i_bits[q1]
                    j_gate = j_bits[q2] * 2 + j_bits[q1]
                    full_matrix[i, j] = gate_matrix[i_gate, j_gate]
        return full_matrix
    def simulate(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simulate the quantum circuit.
        Parameters
        ----------
        initial_state : np.ndarray, optional
            Initial quantum state (default: |0⟩^⊗n)
        Returns
        -------
        np.ndarray
            Final quantum state
        """
        if initial_state is None:
            # Start in |0⟩^⊗n state
            initial_state = np.zeros(self.n_states, dtype=complex)
            initial_state[0] = 1.0
        current_state = initial_state.copy()
        # Apply gates sequentially
        for gate, qubits in self.gates:
            gate_matrix = self._expand_gate_matrix(gate.get_matrix(), qubits)
            # Add noise if configured
            if self.config.gate_error_rate > 0:
                current_state = self._apply_gate_noise(current_state, gate_matrix)
            else:
                current_state = gate_matrix @ current_state
        self.current_state = current_state
        return current_state
    def _apply_gate_noise(self, state: np.ndarray,
                         gate_matrix: np.ndarray) -> np.ndarray:
        """Apply gate with noise model."""
        # Simple depolarizing noise model
        error_prob = self.config.gate_error_rate
        # Apply ideal gate
        ideal_state = gate_matrix @ state
        # Apply depolarizing noise
        if np.random.random() < error_prob:
            # Replace with maximally mixed state contribution
            mixed_contribution = np.ones(self.n_states) / np.sqrt(self.n_states)
            noisy_state = (1 - error_prob) * ideal_state + error_prob * mixed_contribution
        else:
            noisy_state = ideal_state
        return noisy_state
    def measure(self, qubits: Optional[List[int]] = None,
               n_shots: int = 1024) -> Dict[str, int]:
        """
        Measure the quantum circuit.
        Parameters
        ----------
        qubits : List[int], optional
            Qubits to measure (default: all)
        n_shots : int
            Number of measurement shots
        Returns
        -------
        Dict[str, int]
            Measurement outcomes and counts
        """
        if self.current_state is None:
            self.simulate()
        if qubits is None:
            qubits = list(range(self.n_qubits))
        # Calculate measurement probabilities
        probabilities = np.abs(self.current_state)**2
        # Sample measurements
        measurements = np.random.choice(
            range(self.n_states),
            size=n_shots,
            p=probabilities
        )
        # Convert to bitstring counts
        counts = {}
        for measurement in measurements:
            # Extract relevant qubits
            full_bitstring = format(measurement, f'0{self.n_qubits}b')
            measured_bits = ''.join(full_bitstring[-(q+1)] for q in reversed(qubits))
            counts[measured_bits] = counts.get(measured_bits, 0) + 1
        return counts
    def optimize(self) -> 'QuantumCircuit':
        """
        Optimize the quantum circuit.
        Returns
        -------
        QuantumCircuit
            Optimized circuit
        """
        if not self.config.optimize_circuits:
            return self
        optimized_gates = []
        # Simple optimization: cancel adjacent inverse gates
        i = 0
        while i < len(self.gates):
            gate, qubits = self.gates[i]
            # Look for inverse gate
            if i + 1 < len(self.gates):
                next_gate, next_qubits = self.gates[i + 1]
                # Check if gates are inverses on same qubits
                if (qubits == next_qubits and
                    self._are_inverse_gates(gate, next_gate)):
                    # Skip both gates (they cancel)
                    i += 2
                    continue
            optimized_gates.append((gate, qubits))
            i += 1
        # Create optimized circuit
        optimized_circuit = QuantumCircuit(self.n_qubits, self.config)
        optimized_circuit.gates = optimized_gates
        return optimized_circuit
    def _are_inverse_gates(self, gate1: QuantumGate, gate2: QuantumGate) -> bool:
        """Check if two gates are inverses of each other."""
        # Simple check for common cases
        inverse_pairs = {
            ('X', 'X'), ('Y', 'Y'), ('Z', 'Z'),
            ('H', 'H'), ('S', 'S†'), ('T', 'T†'),
            ('CNOT', 'CNOT'), ('SWAP', 'SWAP')
        }
        return (gate1.name, gate2.name) in inverse_pairs
    def depth(self) -> int:
        """Calculate circuit depth."""
        if not self.gates:
            return 0
        # Track when each qubit is last used
        last_used = [-1] * self.n_qubits
        depth = 0
        for layer, (gate, qubits) in enumerate(self.gates):
            # Find the latest time any of these qubits was used
            max_last_used = max(last_used[q] for q in qubits)
            current_depth = max_last_used + 1
            # Update last used times
            for q in qubits:
                last_used[q] = current_depth
            depth = max(depth, current_depth)
        return depth
    def draw(self) -> str:
        """
        Create ASCII representation of the circuit.
        Returns
        -------
        str
            ASCII circuit diagram
        """
        if not self.gates:
            return "Empty circuit"
        # Build circuit diagram
        lines = [f"q{i}: " for i in range(self.n_qubits)]
        for gate, qubits in self.gates:
            # Add gate representation
            for i in range(self.n_qubits):
                if i in qubits:
                    if len(qubits) == 1:
                        lines[i] += f"[{gate.name}]"
                    else:
                        # Multi-qubit gate
                        if i == min(qubits):
                            lines[i] += f"[{gate.name}]"
                        else:
                            lines[i] += "[•]"
                else:
                    lines[i] += "---"
                lines[i] += "---"
        return "\n".join(lines)
    def plot_circuit(self) -> None:
        """Plot the quantum circuit."""
        berkeley_plot = BerkeleyPlot()
        if not self.gates:
            print("Empty circuit - nothing to plot")
            return
        fig, ax = plt.subplots(figsize=(max(8, len(self.gates)), self.n_qubits + 1))
        # Draw qubit lines
        for i in range(self.n_qubits):
            ax.plot([0, len(self.gates) + 1], [i, i], 'k-', linewidth=1)
            ax.text(-0.5, i, f'q{i}', ha='right', va='center')
        # Draw gates
        for j, (gate, qubits) in enumerate(self.gates):
            x_pos = j + 1
            if len(qubits) == 1:
                # Single-qubit gate
                q = qubits[0]
                rect = plt.Rectangle((x_pos - 0.3, q - 0.2), 0.6, 0.4,
                                   facecolor=berkeley_plot.colors['berkeley_blue'],
                                   edgecolor='black')
                ax.add_patch(rect)
                ax.text(x_pos, q, gate.name, ha='center', va='center',
                       color='white', fontweight='bold')
            elif len(qubits) == 2:
                # Two-qubit gate
                q1, q2 = qubits
                # Draw connection line
                ax.plot([x_pos, x_pos], [min(q1, q2), max(q1, q2)],
                       'k-', linewidth=2)
                # Draw control/target symbols
                if gate.name == 'CNOT':
                    # Control qubit (filled circle)
                    ax.plot(x_pos, q1, 'ko', markersize=8)
                    # Target qubit (plus symbol)
                    circle = plt.Circle((x_pos, q2), 0.15,
                                      facecolor='white', edgecolor='black')
                    ax.add_patch(circle)
                    ax.plot([x_pos-0.1, x_pos+0.1], [q2, q2], 'k-', linewidth=2)
                    ax.plot([x_pos, x_pos], [q2-0.1, q2+0.1], 'k-', linewidth=2)
                else:
                    # General two-qubit gate
                    for q in qubits:
                        rect = plt.Rectangle((x_pos - 0.3, q - 0.2), 0.6, 0.4,
                                           facecolor=berkeley_plot.colors['california_gold'],
                                           edgecolor='black')
                        ax.add_patch(rect)
                        ax.text(x_pos, q, gate.name, ha='center', va='center',
                               color='black', fontweight='bold', fontsize=8)
        ax.set_xlim(-1, len(self.gates) + 1)
        ax.set_ylim(-0.5, self.n_qubits - 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Quantum Circuit')
        plt.tight_layout()
        plt.show()
def decompose_arbitrary_unitary(matrix: np.ndarray) -> List[QuantumGate]:
    """
    Decompose arbitrary single-qubit unitary into standard gates.
    Parameters
    ----------
    matrix : np.ndarray
        2x2 unitary matrix
    Returns
    -------
    List[QuantumGate]
        List of gates that implement the unitary
    """
    if matrix.shape != (2, 2):
        raise ValueError("Only single-qubit unitaries supported")
    # Extract parameters for U3 decomposition
    # Any single-qubit unitary can be written as U3(θ, φ, λ)
    # Calculate parameters using ZYZ Euler decomposition
    theta = 2 * np.arccos(abs(matrix[0, 0]))
    if abs(np.sin(theta/2)) < 1e-10:
        # Special case: rotation around Z-axis only
        phi = 0
        lam = np.angle(matrix[1, 1] / matrix[0, 0])
    else:
        phi = np.angle(-matrix[0, 1] / np.sin(theta/2))
        lam = np.angle(matrix[1, 0] / np.sin(theta/2))
    return [U3Gate(theta, phi, lam)]
if __name__ == "__main__":
    # Example: Create and simulate a simple quantum circuit
    print("Quantum Gates and Circuits Demo")
    # Create 3-qubit circuit
    circuit = QuantumCircuit(3)
    # Build Bell state preparation + additional gate
    circuit.h(0)           # Hadamard on qubit 0
    circuit.cnot(0, 1)     # CNOT between qubits 0 and 1
    circuit.x(2)           # X gate on qubit 2
    circuit.cz(1, 2)       # CZ between qubits 1 and 2
    print("Circuit:")
    print(circuit.draw())
    # Simulate the circuit
    final_state = circuit.simulate()
    print(f"\nFinal state amplitudes:")
    for i, amplitude in enumerate(final_state):
        if abs(amplitude) > 1e-10:
            bitstring = format(i, f'0{circuit.n_qubits}b')
            print(f"|{bitstring}⟩: {amplitude:.4f}")
    # Measure the circuit
    counts = circuit.measure(n_shots=1024)
    print(f"\nMeasurement results (1024 shots):")
    for bitstring, count in sorted(counts.items()):
        print(f"{bitstring}: {count}")
    # Calculate circuit properties
    print(f"\nCircuit properties:")
    print(f"Depth: {circuit.depth()}")
    print(f"Gate counts: {circuit.gate_counts}")
    # Plot the circuit
    circuit.plot_circuit()
    # Example: Parameterized circuit
    print("\n" + "="*50)
    print("Parameterized Circuit Example")
    param_circuit = QuantumCircuit(2)
    param_circuit.ry(0, np.pi/4)
    param_circuit.cnot(0, 1)
    param_circuit.rz(1, np.pi/3)
    print("Parameterized circuit:")
    print(param_circuit.draw())
    # Test gate decomposition
    print("\n" + "="*50)
    print("Gate Decomposition Example")
    # Create arbitrary unitary
    theta, phi, lam = np.pi/3, np.pi/4, np.pi/6
    arbitrary_unitary = U3Gate(theta, phi, lam).get_matrix()
    # Decompose it
    decomposed_gates = decompose_arbitrary_unitary(arbitrary_unitary)
    print(f"Original gate parameters: θ={theta:.3f}, φ={phi:.3f}, λ={lam:.3f}")
    print(f"Decomposed gate: {decomposed_gates[0].name} with parameters {decomposed_gates[0].parameters}")
    # Verify decomposition
    reconstructed = decomposed_gates[0].get_matrix()
    error = np.max(np.abs(arbitrary_unitary - reconstructed))
    print(f"Reconstruction error: {error:.2e}")
    print("\nDemo completed!")