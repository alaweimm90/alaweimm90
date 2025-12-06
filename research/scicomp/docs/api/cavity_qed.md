# cavity_qed
**Module:** `Python/QuantumOptics/core/cavity_qed.py`
## Overview
Cavity Quantum Electrodynamics (QED) simulations.
This module provides tools for cavity QED including:
- Jaynes-Cummings model
- Rabi oscillations
- Cavity-atom interactions
- Dissipative dynamics
## Classes
### `JaynesCummings`
Jaynes-Cummings model for cavity-atom interaction.
#### Methods
##### `__init__(self, omega_c, omega_a, g, n_max)`
Initialize Jaynes-Cummings system.
Args:
omega_c: Cavity frequency
omega_a: Atomic transition frequency
g: Coupling strength
n_max: Maximum photon number
**Source:** [Line 21](Python/QuantumOptics/core/cavity_qed.py#L21)
##### `_build_hamiltonian(self)`
Construct Jaynes-Cummings Hamiltonian.
**Source:** [Line 40](Python/QuantumOptics/core/cavity_qed.py#L40)
##### `time_evolution(self, psi0, times)`
Calculate time evolution of state.
Args:
psi0: Initial state vector
times: Time points
Returns:
States at each time point
**Source:** [Line 63](Python/QuantumOptics/core/cavity_qed.py#L63)
##### `eigenvalues(self)`
Get eigenvalues of Jaynes-Cummings Hamiltonian.
**Source:** [Line 82](Python/QuantumOptics/core/cavity_qed.py#L82)
##### `rabi_oscillations(self, n_photons, times)`
Calculate Rabi oscillations for initial Fock state.
Args:
n_photons: Initial photon number
times: Time points
Returns:
Dictionary with atomic and photon dynamics
**Source:** [Line 86](Python/QuantumOptics/core/cavity_qed.py#L86)
##### `vacuum_rabi_splitting(self)`
Calculate vacuum Rabi splitting.
Returns:
Eigenvalues and eigenvectors showing splitting
**Source:** [Line 133](Python/QuantumOptics/core/cavity_qed.py#L133)
##### `dressed_states(self, n)`
Calculate dressed states for n-photon manifold.
Args:
n: Photon number
Returns:
Dressed state energies and states
**Source:** [Line 150](Python/QuantumOptics/core/cavity_qed.py#L150)
**Class Source:** [Line 18](Python/QuantumOptics/core/cavity_qed.py#L18)
### `DissipativeJaynesCummings`
Jaynes-Cummings model with dissipation.
#### Methods
##### `__init__(self, omega_c, omega_a, g, kappa, gamma, n_max)`
Initialize dissipative system.
Args:
omega_c: Cavity frequency
omega_a: Atomic frequency
g: Coupling strength
kappa: Cavity decay rate
gamma: Atomic decay rate
n_max: Maximum photon number
**Source:** [Line 178](Python/QuantumOptics/core/cavity_qed.py#L178)
##### `_build_lindblad_operators(self)`
Construct Lindblad jump operators.
**Source:** [Line 199](Python/QuantumOptics/core/cavity_qed.py#L199)
##### `lindblad_evolution(self, rho0, times)`
Solve master equation with Lindblad terms.
Args:
rho0: Initial density matrix
times: Time points
Returns:
Density matrices at each time
**Source:** [Line 217](Python/QuantumOptics/core/cavity_qed.py#L217)
##### `steady_state(self)`
Calculate steady state using eigenvalue method.
Returns:
Steady state density matrix
**Source:** [Line 265](Python/QuantumOptics/core/cavity_qed.py#L265)
##### `_build_liouvillian(self)`
Build Liouvillian superoperator.
**Source:** [Line 289](Python/QuantumOptics/core/cavity_qed.py#L289)
**Class Source:** [Line 175](Python/QuantumOptics/core/cavity_qed.py#L175)
### `CavityModes`
Optical cavity modes and field distributions.
#### Methods
##### `hermite_gaussian(n, x, w0)`
Hermite-Gaussian mode profile.
Args:
n: Mode number
x: Position array
w0: Beam waist
Returns:
Mode profile
**Source:** [Line 321](Python/QuantumOptics/core/cavity_qed.py#L321)
##### `laguerre_gaussian(p, l, r, phi, w0)`
Laguerre-Gaussian mode profile.
Args:
p: Radial mode number
l: Azimuthal mode number
r: Radial coordinate
phi: Angular coordinate
w0: Beam waist
Returns:
Mode profile
**Source:** [Line 348](Python/QuantumOptics/core/cavity_qed.py#L348)
##### `cavity_spectrum(length, n_modes, fsr)`
Calculate cavity mode frequencies.
Args:
length: Cavity length
n_modes: Number of modes
fsr: Free spectral range (calculated if None)
Returns:
Mode frequencies
**Source:** [Line 382](Python/QuantumOptics/core/cavity_qed.py#L382)
##### `finesse(r1, r2, loss)`
Calculate cavity finesse.
Args:
r1: Mirror 1 reflectivity
r2: Mirror 2 reflectivity
loss: Round-trip loss
Returns:
Cavity finesse
**Source:** [Line 406](Python/QuantumOptics/core/cavity_qed.py#L406)
##### `mode_volume(w0, length)`
Calculate cavity mode volume.
Args:
w0: Beam waist
length: Cavity length
Returns:
Mode volume
**Source:** [Line 424](Python/QuantumOptics/core/cavity_qed.py#L424)
**Class Source:** [Line 317](Python/QuantumOptics/core/cavity_qed.py#L317)
### `PulseShaping`
Quantum pulse shaping and control.
#### Methods
##### `gaussian_pulse(t, t0, sigma, amplitude)`
Gaussian pulse shape.
Args:
t: Time array
t0: Pulse center
sigma: Pulse width
amplitude: Peak amplitude
Returns:
Pulse envelope
**Source:** [Line 442](Python/QuantumOptics/core/cavity_qed.py#L442)
##### `sech_pulse(t, t0, width, amplitude)`
Hyperbolic secant pulse.
Args:
t: Time array
t0: Pulse center
width: Pulse width
amplitude: Peak amplitude
Returns:
Pulse envelope
**Source:** [Line 459](Python/QuantumOptics/core/cavity_qed.py#L459)
##### `chirped_pulse(t, omega0, chirp, envelope)`
Chirped pulse with frequency sweep.
Args:
t: Time array
omega0: Central frequency
chirp: Chirp rate
envelope: Pulse envelope (constant if None)
Returns:
Chirped pulse
**Source:** [Line 476](Python/QuantumOptics/core/cavity_qed.py#L476)
##### `optimal_control_pulse(H0, H_control, initial, target, T, n_steps, max_iter)`
Calculate optimal control pulse using GRAPE algorithm.
Args:
H0: Drift Hamiltonian
H_control: List of control Hamiltonians
initial: Initial state
target: Target state
T: Total time
n_steps: Number of time steps
max_iter: Maximum iterations
Returns:
Optimal control pulses
**Source:** [Line 497](Python/QuantumOptics/core/cavity_qed.py#L497)
**Class Source:** [Line 438](Python/QuantumOptics/core/cavity_qed.py#L438)