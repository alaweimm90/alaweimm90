# state_space
**Module:** `Python/Control/core/state_space.py`
## Overview
State-Space Control Systems
Modern control theory implementations using state-space representation.
Includes system analysis, controllability/observability tests, and optimal control design.
## Classes
### `StateSpaceMatrices`
State-space system matrices container.
#### Methods
##### `__post_init__(self)`
Validate matrix dimensions.
**Source:** [Line 24](Python/Control/core/state_space.py#L24)
**Class Source:** [Line 17](Python/Control/core/state_space.py#L17)
### `StateSpaceSystem`
Linear time-invariant state-space system representation.
Represents systems of the form:
dx/dt = Ax + Bu
y = Cx + Du
Features:
- System analysis (poles, zeros, stability)
- Controllability and observability analysis
- Time and frequency response simulation
- System transformations
Examples:
>>> A = np.array([[-1, 1], [0, -2]])
>>> B = np.array([[0], [1]])
>>> C = np.array([[1, 0]])
>>> D = np.array([[0]])
>>> sys = StateSpaceSystem(A, B, C, D)
>>> print(f"System is stable: {sys.is_stable()}")
#### Methods
##### `__init__(self, A, B, C, D)`
Initialize state-space system.
Parameters:
A: System matrix (n x n)
B: Input matrix (n x m)
C: Output matrix (p x n)
D: Feedthrough matrix (p x m)
**Source:** [Line 68](Python/Control/core/state_space.py#L68)
##### `_validate_system(self)`
Validate system matrices and properties.
**Source:** [Line 81](Python/Control/core/state_space.py#L81)
##### `A(self)`
System matrix.
**Source:** [Line 92](Python/Control/core/state_space.py#L92)
##### `B(self)`
Input matrix.
**Source:** [Line 97](Python/Control/core/state_space.py#L97)
##### `C(self)`
Output matrix.
**Source:** [Line 102](Python/Control/core/state_space.py#L102)
##### `D(self)`
Feedthrough matrix.
**Source:** [Line 107](Python/Control/core/state_space.py#L107)
##### `n_states(self)`
Number of states.
**Source:** [Line 112](Python/Control/core/state_space.py#L112)
##### `n_inputs(self)`
Number of inputs.
**Source:** [Line 117](Python/Control/core/state_space.py#L117)
##### `n_outputs(self)`
Number of outputs.
**Source:** [Line 122](Python/Control/core/state_space.py#L122)
##### `poles(self)`
Compute system poles (eigenvalues of A).
**Source:** [Line 126](Python/Control/core/state_space.py#L126)
##### `is_stable(self)`
Check if system is asymptotically stable.
**Source:** [Line 130](Python/Control/core/state_space.py#L130)
##### `controllability_matrix(self)`
Compute controllability matrix.
**Source:** [Line 135](Python/Control/core/state_space.py#L135)
##### `is_controllable(self)`
Check if system is completely controllable.
**Source:** [Line 147](Python/Control/core/state_space.py#L147)
##### `observability_matrix(self)`
Compute observability matrix.
**Source:** [Line 153](Python/Control/core/state_space.py#L153)
##### `is_observable(self)`
Check if system is completely observable.
**Source:** [Line 165](Python/Control/core/state_space.py#L165)
##### `simulate(self, t, u, x0)`
Simulate system response.
Parameters:
t: Time vector
u: Input signal (array or function)
x0: Initial state (default: zeros)
Returns:
Tuple of (state_response, output_response)
**Source:** [Line 171](Python/Control/core/state_space.py#L171)
##### `step_response(self, t, input_channel)`
Compute step response.
Parameters:
t: Time vector
input_channel: Which input to apply step to
Returns:
Tuple of (state_response, output_response)
**Source:** [Line 230](Python/Control/core/state_space.py#L230)
##### `impulse_response(self, t, input_channel)`
Compute impulse response.
Parameters:
t: Time vector
input_channel: Which input to apply impulse to
Returns:
Tuple of (state_response, output_response)
**Source:** [Line 245](Python/Control/core/state_space.py#L245)
**Class Source:** [Line 45](Python/Control/core/state_space.py#L45)
### `LinearQuadraticRegulator`
Linear Quadratic Regulator (LQR) optimal controller.
Solves the optimal control problem:
min ∫(x'Qx + u'Ru + 2x'Nu) dt
Subject to: dx/dt = Ax + Bu
Examples:
>>> sys = StateSpaceSystem(A, B, C, D)
>>> Q = np.eye(sys.n_states)
>>> R = np.eye(sys.n_inputs)
>>> lqr = LinearQuadraticRegulator(sys, Q, R)
>>> K = lqr.gain_matrix()
#### Methods
##### `__init__(self, system, Q, R, N)`
Initialize LQR controller.
Parameters:
system: State-space system
Q: State weighting matrix (n x n, positive semi-definite)
R: Input weighting matrix (m x m, positive definite)
N: Cross-term matrix (n x m, optional)
**Source:** [Line 279](Python/Control/core/state_space.py#L279)
##### `_validate_weights(self)`
Validate weighting matrices.
**Source:** [Line 305](Python/Control/core/state_space.py#L305)
##### `_solve_riccati(self)`
Solve algebraic Riccati equation.
**Source:** [Line 329](Python/Control/core/state_space.py#L329)
##### `gain_matrix(self)`
Compute optimal feedback gain matrix K.
**Source:** [Line 344](Python/Control/core/state_space.py#L344)
##### `closed_loop_system(self)`
Compute closed-loop system with LQR feedback.
**Source:** [Line 350](Python/Control/core/state_space.py#L350)
##### `cost_function_value(self, x0)`
Compute optimal cost for given initial condition.
Parameters:
x0: Initial state
Returns:
Optimal cost J* = x0' P x0
**Source:** [Line 360](Python/Control/core/state_space.py#L360)
**Class Source:** [Line 262](Python/Control/core/state_space.py#L262)