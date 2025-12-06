# quantum_light
**Module:** `Python/QuantumOptics/core/quantum_light.py`
## Overview
Quantum light states and operations.
This module provides tools for quantum optics including:
- Fock states
- Coherent states
- Squeezed states
- Photon statistics
## Classes
### `FockStates`
Fock (number) states and operations.
#### Methods
##### `fock_state(n, dim)`
Create Fock state |n⟩.
Args:
n: Photon number
dim: Hilbert space dimension
Returns:
Fock state vector
**Source:** [Line 22](Python/QuantumOptics/core/quantum_light.py#L22)
##### `creation_operator(dim, sparse)`
Creation operator a†.
Args:
dim: Hilbert space dimension
sparse: Return sparse matrix
Returns:
Creation operator
**Source:** [Line 39](Python/QuantumOptics/core/quantum_light.py#L39)
##### `annihilation_operator(dim, sparse)`
Annihilation operator a.
Args:
dim: Hilbert space dimension
sparse: Return sparse matrix
Returns:
Annihilation operator
**Source:** [Line 62](Python/QuantumOptics/core/quantum_light.py#L62)
##### `number_operator(dim, sparse)`
Number operator n = a†a.
Args:
dim: Hilbert space dimension
sparse: Return sparse matrix
Returns:
Number operator
**Source:** [Line 85](Python/QuantumOptics/core/quantum_light.py#L85)
##### `displacement_operator(alpha, dim)`
Displacement operator D(α) = exp(αa† - α*a).
Args:
alpha: Displacement parameter
dim: Hilbert space dimension
Returns:
Displacement operator
**Source:** [Line 102](Python/QuantumOptics/core/quantum_light.py#L102)
**Class Source:** [Line 18](Python/QuantumOptics/core/quantum_light.py#L18)
### `CoherentStates`
Coherent states and operations.
#### Methods
##### `coherent_state(alpha, dim, method)`
Create coherent state |α⟩.
Args:
alpha: Coherent state parameter
dim: Hilbert space dimension
method: 'displacement' or 'fock_expansion'
Returns:
Coherent state vector
**Source:** [Line 123](Python/QuantumOptics/core/quantum_light.py#L123)
##### `coherent_state_overlap(alpha, beta)`
Calculate overlap ⟨α|β⟩.
Args:
alpha: First coherent state parameter
beta: Second coherent state parameter
Returns:
Overlap
**Source:** [Line 153](Python/QuantumOptics/core/quantum_light.py#L153)
##### `husimi_q_function(state, alpha_grid)`
Calculate Husimi Q-function.
Args:
state: Quantum state vector or density matrix
alpha_grid: Grid of coherent state parameters
Returns:
Q-function values
**Source:** [Line 167](Python/QuantumOptics/core/quantum_light.py#L167)
##### `glauber_sudarshan_p_function(state, alpha, regularization)`
Calculate regularized P-function.
Args:
state: Density matrix
alpha: Coherent state parameter
regularization: Regularization parameter
Returns:
P-function value
**Source:** [Line 195](Python/QuantumOptics/core/quantum_light.py#L195)
**Class Source:** [Line 119](Python/QuantumOptics/core/quantum_light.py#L119)
### `SqueezedStates`
Squeezed states and operations.
#### Methods
##### `squeeze_operator(xi, dim)`
Squeeze operator S(ξ) = exp((ξ*a² - ξa†²)/2).
Args:
xi: Squeezing parameter (r*e^(iθ))
dim: Hilbert space dimension
Returns:
Squeeze operator
**Source:** [Line 228](Python/QuantumOptics/core/quantum_light.py#L228)
##### `squeezed_state(xi, alpha, dim)`
Create squeezed coherent state |α, ξ⟩.
Args:
xi: Squeezing parameter
alpha: Displacement parameter
dim: Hilbert space dimension
Returns:
Squeezed coherent state
**Source:** [Line 248](Python/QuantumOptics/core/quantum_light.py#L248)
##### `two_mode_squeeze_operator(xi, dim)`
Two-mode squeeze operator.
Args:
xi: Squeezing parameter
dim: Dimension per mode
Returns:
Two-mode squeeze operator
**Source:** [Line 268](Python/QuantumOptics/core/quantum_light.py#L268)
##### `quadrature_variances(state)`
Calculate quadrature variances.
Args:
state: Quantum state vector
Returns:
Variances of X and P quadratures
**Source:** [Line 287](Python/QuantumOptics/core/quantum_light.py#L287)
**Class Source:** [Line 224](Python/QuantumOptics/core/quantum_light.py#L224)
### `PhotonStatistics`
Photon counting statistics and correlations.
#### Methods
##### `photon_distribution(state)`
Calculate photon number distribution.
Args:
state: State vector or density matrix
Returns:
Probability distribution P(n)
**Source:** [Line 321](Python/QuantumOptics/core/quantum_light.py#L321)
##### `mean_photon_number(state)`
Calculate mean photon number.
Args:
state: State vector or density matrix
Returns:
Mean photon number
**Source:** [Line 339](Python/QuantumOptics/core/quantum_light.py#L339)
##### `mandel_q_parameter(state)`
Calculate Mandel Q parameter.
Args:
state: State vector or density matrix
Returns:
Mandel Q parameter
**Source:** [Line 358](Python/QuantumOptics/core/quantum_light.py#L358)
##### `g2_correlation(state, tau)`
Calculate second-order correlation g²(τ).
Args:
state: State vector or density matrix
tau: Time delay (0 for equal-time)
Returns:
g²(τ) correlation
**Source:** [Line 386](Python/QuantumOptics/core/quantum_light.py#L386)
##### `antibunching_parameter(state)`
Calculate antibunching parameter.
Args:
state: State vector or density matrix
Returns:
Antibunching parameter (negative for antibunching)
**Source:** [Line 420](Python/QuantumOptics/core/quantum_light.py#L420)
**Class Source:** [Line 317](Python/QuantumOptics/core/quantum_light.py#L317)
### `WignerFunction`
Wigner quasi-probability distribution.
#### Methods
##### `wigner_function(state, xvec, pvec)`
Calculate Wigner function.
Args:
state: State vector or density matrix
xvec: Position grid
pvec: Momentum grid
Returns:
Wigner function W(x,p)
**Source:** [Line 438](Python/QuantumOptics/core/quantum_light.py#L438)
##### `_wigner_point(rho, x, p, dim)`
Calculate Wigner function at single point.
**Source:** [Line 469](Python/QuantumOptics/core/quantum_light.py#L469)
##### `wigner_from_characteristic(chi, beta_grid, xvec, pvec)`
Calculate Wigner function from characteristic function.
Args:
chi: Characteristic function values
beta_grid: Grid for characteristic function
xvec: Position grid
pvec: Momentum grid
Returns:
Wigner function
**Source:** [Line 482](Python/QuantumOptics/core/quantum_light.py#L482)
**Class Source:** [Line 434](Python/QuantumOptics/core/quantum_light.py#L434)