# quantum_operators
**Module:** `Python/Quantum/core/quantum_operators.py`
## Overview
Quantum operators and gates.
This module provides quantum operators including:
- Pauli matrices
- Common quantum gates
- Operator utilities
- Time evolution operators
## Classes
### `PauliOperators`
Pauli matrices and operations.
#### Methods
##### `pauli_string(cls, operators)`
Create multi-qubit Pauli operator from string.
Args:
operators: String of Pauli operators (e.g., 'XYZ')
Returns:
Multi-qubit Pauli operator
**Source:** [Line 27](Python/Quantum/core/quantum_operators.py#L27)
##### `commutator(cls, A, B)`
Calculate commutator [A, B] = AB - BA.
**Source:** [Line 49](Python/Quantum/core/quantum_operators.py#L49)
##### `anticommutator(cls, A, B)`
Calculate anticommutator {A, B} = AB + BA.
**Source:** [Line 54](Python/Quantum/core/quantum_operators.py#L54)
**Class Source:** [Line 17](Python/Quantum/core/quantum_operators.py#L17)
### `QuantumGates`
Common quantum gates.
#### Methods
##### `Rx(theta)`
Rotation around X-axis.
**Source:** [Line 68](Python/Quantum/core/quantum_operators.py#L68)
##### `Ry(theta)`
Rotation around Y-axis.
**Source:** [Line 75](Python/Quantum/core/quantum_operators.py#L75)
##### `Rz(theta)`
Rotation around Z-axis.
**Source:** [Line 82](Python/Quantum/core/quantum_operators.py#L82)
##### `U3(theta, phi, lam)`
General single-qubit unitary.
**Source:** [Line 88](Python/Quantum/core/quantum_operators.py#L88)
##### `CNOT()`
Controlled-NOT gate.
**Source:** [Line 98](Python/Quantum/core/quantum_operators.py#L98)
##### `CZ()`
Controlled-Z gate.
**Source:** [Line 108](Python/Quantum/core/quantum_operators.py#L108)
##### `SWAP()`
SWAP gate.
**Source:** [Line 113](Python/Quantum/core/quantum_operators.py#L113)
##### `Toffoli()`
Toffoli (CCNOT) gate.
**Source:** [Line 123](Python/Quantum/core/quantum_operators.py#L123)
##### `Fredkin()`
Fredkin (controlled-SWAP) gate.
**Source:** [Line 133](Python/Quantum/core/quantum_operators.py#L133)
##### `controlled_gate(gate, control_qubits)`
Create controlled version of gate.
Args:
gate: Unitary gate to control
control_qubits: Number of control qubits
Returns:
Controlled gate
**Source:** [Line 143](Python/Quantum/core/quantum_operators.py#L143)
**Class Source:** [Line 59](Python/Quantum/core/quantum_operators.py#L59)
### `HamiltonianOperators`
Common Hamiltonian operators.
#### Methods
##### `ising_1d(n_sites, J, h, periodic)`
1D Ising model Hamiltonian.
Args:
n_sites: Number of sites
J: Coupling strength
h: Transverse field strength
periodic: Whether to use periodic boundary conditions
Returns:
Hamiltonian matrix
**Source:** [Line 168](Python/Quantum/core/quantum_operators.py#L168)
##### `heisenberg_1d(n_sites, J, periodic)`
1D Heisenberg model Hamiltonian.
Args:
n_sites: Number of sites
J: Coupling strength(s) [Jx, Jy, Jz] or scalar
periodic: Whether to use periodic boundary conditions
Returns:
Hamiltonian matrix
**Source:** [Line 203](Python/Quantum/core/quantum_operators.py#L203)
##### `hubbard_1d(n_sites, t, U, periodic)`
1D Hubbard model Hamiltonian.
Args:
n_sites: Number of sites
t: Hopping parameter
U: On-site interaction
periodic: Whether to use periodic boundary conditions
Returns:
Hamiltonian matrix
**Source:** [Line 255](Python/Quantum/core/quantum_operators.py#L255)
**Class Source:** [Line 164](Python/Quantum/core/quantum_operators.py#L164)
### `TimeEvolution`
Time evolution operators and propagators.
#### Methods
##### `unitary_evolution(H, t, hbar)`
Unitary time evolution operator U(t) = exp(-iHt/ℏ).
Args:
H: Hamiltonian
t: Time
hbar: Reduced Planck constant
Returns:
Time evolution operator
**Source:** [Line 313](Python/Quantum/core/quantum_operators.py#L313)
##### `trotter_suzuki(H_list, t, n_steps, order)`
Trotter-Suzuki decomposition for time evolution.
Args:
H_list: List of Hamiltonian terms
t: Total time
n_steps: Number of Trotter steps
order: Order of decomposition (1 or 2)
Returns:
Approximate time evolution operator
**Source:** [Line 328](Python/Quantum/core/quantum_operators.py#L328)
##### `magnus_expansion(H, t, order)`
Magnus expansion for time-dependent Hamiltonians.
Args:
H: Time-dependent Hamiltonian H(t)
t: Time
order: Order of Magnus expansion
Returns:
Time evolution operator
**Source:** [Line 369](Python/Quantum/core/quantum_operators.py#L369)
**Class Source:** [Line 309](Python/Quantum/core/quantum_operators.py#L309)
### `OperatorMeasurements`
Measurement operations and expectation values.
#### Methods
##### `measure_pauli(state_vector, pauli_string)`
Measure Pauli operator expectation value.
Args:
state_vector: Quantum state vector
pauli_string: Pauli operator string
Returns:
Expectation value
**Source:** [Line 407](Python/Quantum/core/quantum_operators.py#L407)
##### `projective_measurement(state_vector, basis)`
Perform projective measurement.
Args:
state_vector: Quantum state vector
basis: Measurement basis vectors
Returns:
Measurement outcome and collapsed state
**Source:** [Line 422](Python/Quantum/core/quantum_operators.py#L422)
##### `povm_measurement(state_vector, povm_elements)`
Perform POVM measurement.
Args:
state_vector: Quantum state vector
povm_elements: POVM elements
Returns:
Measurement outcome
**Source:** [Line 448](Python/Quantum/core/quantum_operators.py#L448)
**Class Source:** [Line 403](Python/Quantum/core/quantum_operators.py#L403)