# quantum_states
**Module:** `Python/Quantum/core/quantum_states.py`
## Overview
Quantum state representations and operations.
This module provides comprehensive quantum state manipulation including:
- Pure and mixed state representations
- Density matrix operations
- Entanglement measures
- State tomography
## Classes
### `QuantumState`
Base class for quantum state representations.
#### Methods
##### `__init__(self, data, is_density_matrix)`
Initialize quantum state.
Args:
data: State vector or density matrix
is_density_matrix: Whether data is density matrix
**Source:** [Line 20](Python/Quantum/core/quantum_states.py#L20)
##### `_normalize_state(self)`
Normalize state vector.
**Source:** [Line 40](Python/Quantum/core/quantum_states.py#L40)
##### `is_normalized(self, tolerance)`
Check if state is normalized.
**Source:** [Line 46](Python/Quantum/core/quantum_states.py#L46)
##### `_validate_density_matrix(self)`
Validate density matrix properties.
**Source:** [Line 55](Python/Quantum/core/quantum_states.py#L55)
##### `expectation_value(self, operator)`
Calculate expectation value of operator.
Args:
operator: Hermitian operator
Returns:
Expectation value
**Source:** [Line 71](Python/Quantum/core/quantum_states.py#L71)
##### `purity(self)`
Calculate state purity.
**Source:** [Line 83](Python/Quantum/core/quantum_states.py#L83)
##### `von_neumann_entropy(self)`
Calculate von Neumann entropy.
**Source:** [Line 87](Python/Quantum/core/quantum_states.py#L87)
##### `fidelity(self, other)`
Calculate fidelity with another state.
Args:
other: Another quantum state
Returns:
Fidelity between states
**Source:** [Line 93](Python/Quantum/core/quantum_states.py#L93)
##### `partial_trace(self, subsystem_dims, keep)`
Calculate partial trace over subsystems.
Args:
subsystem_dims: Dimensions of each subsystem
keep: Indices of subsystems to keep
Returns:
Reduced density matrix
**Source:** [Line 110](Python/Quantum/core/quantum_states.py#L110)
**Class Source:** [Line 17](Python/Quantum/core/quantum_states.py#L17)
### `EntanglementMeasures`
Tools for quantifying quantum entanglement.
#### Methods
##### `concurrence(state, subsystem_dims)`
Calculate concurrence for two-qubit system.
Args:
state: Quantum state
subsystem_dims: Dimensions of subsystems
Returns:
Concurrence value
**Source:** [Line 146](Python/Quantum/core/quantum_states.py#L146)
##### `entanglement_entropy(state, subsystem_dims, partition)`
Calculate entanglement entropy across partition.
Args:
state: Quantum state
subsystem_dims: Dimensions of subsystems
partition: Indices of first partition
Returns:
Entanglement entropy
**Source:** [Line 182](Python/Quantum/core/quantum_states.py#L182)
##### `negativity(state, subsystem_dims, partition)`
Calculate negativity across partition.
Args:
state: Quantum state
subsystem_dims: Dimensions of subsystems
partition: Indices of first partition
Returns:
Negativity
**Source:** [Line 201](Python/Quantum/core/quantum_states.py#L201)
**Class Source:** [Line 142](Python/Quantum/core/quantum_states.py#L142)
### `BellStates`
Common Bell states and operations.
#### Methods
##### `phi_plus()`
Create |Φ+⟩ = (|00⟩ + |11⟩)/√2.
**Source:** [Line 237](Python/Quantum/core/quantum_states.py#L237)
##### `phi_minus()`
Create |Φ-⟩ = (|00⟩ - |11⟩)/√2.
**Source:** [Line 245](Python/Quantum/core/quantum_states.py#L245)
##### `psi_plus()`
Create |Ψ+⟩ = (|01⟩ + |10⟩)/√2.
**Source:** [Line 253](Python/Quantum/core/quantum_states.py#L253)
##### `psi_minus()`
Create |Ψ-⟩ = (|01⟩ - |10⟩)/√2.
**Source:** [Line 261](Python/Quantum/core/quantum_states.py#L261)
**Class Source:** [Line 233](Python/Quantum/core/quantum_states.py#L233)
### `GHZStates`
Greenberger-Horne-Zeilinger states.
#### Methods
##### `ghz_state(n_qubits)`
Create n-qubit GHZ state.
Args:
n_qubits: Number of qubits
Returns:
GHZ state
**Source:** [Line 273](Python/Quantum/core/quantum_states.py#L273)
##### `w_state(n_qubits)`
Create n-qubit W state.
Args:
n_qubits: Number of qubits
Returns:
W state
**Source:** [Line 290](Python/Quantum/core/quantum_states.py#L290)
**Class Source:** [Line 269](Python/Quantum/core/quantum_states.py#L269)
### `QuantumStateTomography`
Quantum state reconstruction from measurements.
#### Methods
##### `pauli_basis(n_qubits)`
Generate Pauli basis for n qubits.
Args:
n_qubits: Number of qubits
Returns:
List of Pauli operators
**Source:** [Line 315](Python/Quantum/core/quantum_states.py#L315)
##### `linear_inversion(measurements, n_qubits)`
Reconstruct state using linear inversion.
Args:
measurements: Dictionary of Pauli measurements
n_qubits: Number of qubits
Returns:
Reconstructed quantum state
**Source:** [Line 348](Python/Quantum/core/quantum_states.py#L348)
##### `maximum_likelihood(measurements, projectors, counts, max_iter)`
Maximum likelihood state tomography.
Args:
measurements: Measurement outcomes
projectors: Measurement projectors
counts: Number of measurements for each projector
max_iter: Maximum iterations
Returns:
Reconstructed state
**Source:** [Line 373](Python/Quantum/core/quantum_states.py#L373)
**Class Source:** [Line 311](Python/Quantum/core/quantum_states.py#L311)