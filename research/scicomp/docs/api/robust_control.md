# robust_control
**Module:** `Python/Control/core/robust_control.py`
## Overview
Robust Control Methods
Advanced robust control techniques including H∞ control, μ-synthesis,
and uncertainty modeling for control systems.
## Functions
### `robust_stability_margin(A, B, uncertainty, n_samples)`
Compute robust stability margin using Monte Carlo analysis.
Parameters:
A: Nominal system matrix
B: Nominal input matrix
uncertainty: Uncertainty model
n_samples: Number of Monte Carlo samples
Returns:
Dictionary with stability statistics
**Source:** [Line 267](Python/Control/core/robust_control.py#L267)
## Classes
### `UncertaintyModel`
Parametric uncertainty model for robust control.
**Class Source:** [Line 17](Python/Control/core/robust_control.py#L17)
### `HInfinityController`
H∞ controller design for robust performance.
Solves the H∞ control problem:
min ||T_zw||∞
Where T_zw is the closed-loop transfer function from disturbances w to
performance outputs z.
Examples:
>>> # Setup generalized plant
>>> P = GeneralizedPlant(A, B1, B2, C1, C2, D11, D12, D21, D22)
>>> controller = HInfinityController(P)
>>> K, gamma = controller.synthesize()
#### Methods
##### `__init__(self, A, B1, B2, C1, C2, D11, D12, D21, D22)`
Initialize H∞ controller with generalized plant.
The generalized plant has the form:
dx/dt = Ax + B1*w + B2*u
z = C1*x + D11*w + D12*u  (performance outputs)
y = C2*x + D21*w + D22*u  (measurements)
**Source:** [Line 42](Python/Control/core/robust_control.py#L42)
##### `_validate_dimensions(self)`
Validate generalized plant matrix dimensions.
**Source:** [Line 72](Python/Control/core/robust_control.py#L72)
##### `synthesize(self, gamma_max, tolerance)`
Synthesize H∞ controller using Riccati-based approach.
Parameters:
gamma_max: Maximum gamma to search
tolerance: Convergence tolerance
Returns:
Tuple of (controller_matrices, achieved_gamma)
**Source:** [Line 96](Python/Control/core/robust_control.py#L96)
##### `_test_gamma(self, gamma)`
Test if a given gamma is achievable.
**Source:** [Line 127](Python/Control/core/robust_control.py#L127)
##### `_compute_controller(self, gamma)`
Compute H∞ controller matrices for given gamma.
**Source:** [Line 170](Python/Control/core/robust_control.py#L170)
**Class Source:** [Line 25](Python/Control/core/robust_control.py#L25)
### `MuSynthesis`
μ-synthesis for robust performance with structured uncertainty.
Addresses the robust performance problem in the presence of
structured uncertainty blocks.
#### Methods
##### `__init__(self, nominal_plant, uncertainty_structure, performance_weights)`
Initialize μ-synthesis problem.
Parameters:
nominal_plant: Nominal plant matrices
uncertainty_structure: List of uncertainty block descriptions
performance_weights: Performance weighting functions
**Source:** [Line 200](Python/Control/core/robust_control.py#L200)
##### `d_k_iteration(self, max_iterations)`
Perform D-K iteration for μ-synthesis.
Parameters:
max_iterations: Maximum number of iterations
Returns:
Tuple of (controller, mu_value)
**Source:** [Line 216](Python/Control/core/robust_control.py#L216)
##### `_h_infinity_step(self, previous_controller)`
H∞ synthesis step in D-K iteration.
**Source:** [Line 250](Python/Control/core/robust_control.py#L250)
##### `_compute_d_scaling(self, controller)`
Compute optimal D-scaling matrices.
**Source:** [Line 256](Python/Control/core/robust_control.py#L256)
##### `_compute_mu_upper_bound(self, controller, d_scaling)`
Compute upper bound on structured singular value.
**Source:** [Line 261](Python/Control/core/robust_control.py#L261)
**Class Source:** [Line 192](Python/Control/core/robust_control.py#L192)