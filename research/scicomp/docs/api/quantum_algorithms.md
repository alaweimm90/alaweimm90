# quantum_algorithms
**Module:** `Python/Quantum/core/quantum_algorithms.py`
## Overview
Quantum algorithms implementation.
This module provides implementations of fundamental quantum algorithms:
- Quantum Fourier Transform
- Phase estimation
- Amplitude amplification
- Quantum walks
## Classes
### `QuantumFourierTransform`
Quantum Fourier Transform and related algorithms.
#### Methods
##### `qft_matrix(n_qubits)`
Generate QFT matrix for n qubits.
Args:
n_qubits: Number of qubits
Returns:
QFT matrix
**Source:** [Line 21](Python/Quantum/core/quantum_algorithms.py#L21)
##### `inverse_qft_matrix(n_qubits)`
Generate inverse QFT matrix.
Args:
n_qubits: Number of qubits
Returns:
Inverse QFT matrix
**Source:** [Line 43](Python/Quantum/core/quantum_algorithms.py#L43)
##### `qft_circuit(state, n_qubits)`
Apply QFT using circuit decomposition.
Args:
state: Input state vector
n_qubits: Number of qubits
Returns:
Transformed state
**Source:** [Line 56](Python/Quantum/core/quantum_algorithms.py#L56)
##### `_apply_hadamard(state, qubit, n_qubits)`
Apply Hadamard gate to specific qubit.
**Source:** [Line 91](Python/Quantum/core/quantum_algorithms.py#L91)
##### `_apply_controlled_phase(state, control, target, angle, n_qubits)`
Apply controlled phase rotation.
**Source:** [Line 106](Python/Quantum/core/quantum_algorithms.py#L106)
##### `_swap_qubits(state, q1, q2, n_qubits)`
Swap two qubits in state vector.
**Source:** [Line 122](Python/Quantum/core/quantum_algorithms.py#L122)
**Class Source:** [Line 17](Python/Quantum/core/quantum_algorithms.py#L17)
### `PhaseEstimation`
Quantum phase estimation algorithm.
#### Methods
##### `estimate_phase(unitary, eigenstate, n_precision)`
Estimate phase of eigenvalue using QPE.
Args:
unitary: Unitary operator
eigenstate: Eigenstate of unitary
n_precision: Number of precision qubits
Returns:
Estimated phase in [0, 1)
**Source:** [Line 143](Python/Quantum/core/quantum_algorithms.py#L143)
##### `_apply_hadamard_to_qubit(state, qubit, n_qubits)`
Apply Hadamard to specific qubit.
**Source:** [Line 191](Python/Quantum/core/quantum_algorithms.py#L191)
##### `_apply_controlled_unitary(state, control, U, n_precision, n_work)`
Apply controlled unitary operation.
**Source:** [Line 206](Python/Quantum/core/quantum_algorithms.py#L206)
##### `_apply_inverse_qft_partial(state, n_precision, n_total)`
Apply inverse QFT to precision qubits only.
**Source:** [Line 230](Python/Quantum/core/quantum_algorithms.py#L230)
**Class Source:** [Line 139](Python/Quantum/core/quantum_algorithms.py#L139)
### `AmplitudeAmplification`
Grover's algorithm and amplitude amplification.
#### Methods
##### `grover_operator(oracle, n_qubits)`
Construct Grover operator G = -AS₀A†O.
Args:
oracle: Oracle matrix marking target states
n_qubits: Number of qubits
Returns:
Grover operator
**Source:** [Line 249](Python/Quantum/core/quantum_algorithms.py#L249)
##### `grover_search(oracle, n_qubits, n_iterations)`
Perform Grover's search algorithm.
Args:
oracle: Oracle function marking target items
n_qubits: Number of qubits
n_iterations: Number of Grover iterations (auto if None)
Returns:
Index of found item
**Source:** [Line 274](Python/Quantum/core/quantum_algorithms.py#L274)
##### `amplitude_estimation(oracle, n_qubits, n_precision)`
Estimate amplitude of marked states.
Args:
oracle: Oracle matrix
n_qubits: Number of work qubits
n_precision: Number of precision qubits
Returns:
Estimated amplitude
**Source:** [Line 318](Python/Quantum/core/quantum_algorithms.py#L318)
**Class Source:** [Line 245](Python/Quantum/core/quantum_algorithms.py#L245)
### `QuantumWalk`
Quantum walk algorithms.
#### Methods
##### `discrete_walk_line(n_steps, n_positions, coin_operator)`
Discrete quantum walk on a line.
Args:
n_steps: Number of walk steps
n_positions: Number of positions
coin_operator: 2x2 coin operator (Hadamard if None)
Returns:
Final probability distribution
**Source:** [Line 353](Python/Quantum/core/quantum_algorithms.py#L353)
##### `continuous_walk_graph(adjacency, time, initial_node)`
Continuous-time quantum walk on graph.
Args:
adjacency: Adjacency matrix of graph
time: Evolution time
initial_node: Starting node
Returns:
Probability distribution over nodes
**Source:** [Line 400](Python/Quantum/core/quantum_algorithms.py#L400)
##### `szegedy_walk(transition_matrix, n_steps)`
Szegedy quantum walk.
Args:
transition_matrix: Classical random walk transition matrix
n_steps: Number of walk steps
Returns:
Quantum walk operator
**Source:** [Line 435](Python/Quantum/core/quantum_algorithms.py#L435)
**Class Source:** [Line 349](Python/Quantum/core/quantum_algorithms.py#L349)