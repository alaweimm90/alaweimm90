# optimal_control
**Module:** `Python/Control/core/optimal_control.py`
## Overview
Optimal Control Algorithms
Advanced optimal control methods including model predictive control,
dynamic programming, and trajectory optimization.
## Functions
### `solve_lq_tracking(A, B, Q, R, reference, horizon)`
Solve finite-horizon LQ tracking problem.
Parameters:
A: System matrix
B: Input matrix
Q: State penalty matrix
R: Input penalty matrix
reference: Reference trajectory (horizon x n_states)
horizon: Time horizon
Returns:
Tuple of (optimal_control, optimal_trajectory)
**Source:** [Line 268](Python/Control/core/optimal_control.py#L268)
## Classes
### `MPCConfig`
Model Predictive Control configuration.
**Class Source:** [Line 17](Python/Control/core/optimal_control.py#L17)
### `ModelPredictiveController`
Model Predictive Control (MPC) implementation.
Solves the finite-horizon optimal control problem at each time step:
min Σ(x'Qx + u'Ru) + xf'Qf*xf
Subject to:
x(k+1) = Ax(k) + Bu(k)
u_min ≤ u(k) ≤ u_max
du_min ≤ u(k) - u(k-1) ≤ du_max
Examples:
>>> config = MPCConfig(prediction_horizon=20, Q=Q, R=R)
>>> mpc = ModelPredictiveController(A, B, config)
>>> u_opt = mpc.solve(x_current, reference_trajectory)
#### Methods
##### `__init__(self, A, B, config)`
Initialize MPC controller.
Parameters:
A: System matrix (n x n)
B: Input matrix (n x m)
config: MPC configuration
**Source:** [Line 57](Python/Control/core/optimal_control.py#L57)
##### `_setup_weights(self)`
Setup default weighting matrices if not provided.
**Source:** [Line 79](Python/Control/core/optimal_control.py#L79)
##### `_build_prediction_matrices(self)`
Build prediction matrices for the MPC formulation.
**Source:** [Line 90](Python/Control/core/optimal_control.py#L90)
##### `solve(self, x_current, reference, u_previous)`
Solve MPC optimization problem.
Parameters:
x_current: Current state
reference: Reference trajectory (N x n_states)
u_previous: Previous control input (for rate constraints)
Returns:
Optimal control sequence (Nc x n_inputs)
**Source:** [Line 130](Python/Control/core/optimal_control.py#L130)
##### `predict_trajectory(self, x_current, u_sequence)`
Predict state trajectory given control sequence.
Parameters:
x_current: Current state
u_sequence: Control sequence (Nc x n_inputs)
Returns:
Predicted state trajectory (N x n_states)
**Source:** [Line 232](Python/Control/core/optimal_control.py#L232)
**Class Source:** [Line 39](Python/Control/core/optimal_control.py#L39)