# spin_transport
**Module:** `Python/Spintronics/core/spin_transport.py`
## Overview
Spin transport and spintronics device modeling.
This module provides tools for spin transport including:
- Giant magnetoresistance (GMR)
- Tunneling magnetoresistance (TMR)
- Spin diffusion equations
- Spin Hall effects
## Classes
### `SpinDiffusion`
Spin diffusion equation solver.
#### Methods
##### `__init__(self, D_up, D_down, sf_length)`
Initialize spin diffusion model.
Args:
D_up: Diffusion constant for spin-up electrons
D_down: Diffusion constant for spin-down electrons
sf_length: Spin flip length
**Source:** [Line 21](Python/Spintronics/core/spin_transport.py#L21)
##### `solve_1d(self, x, boundary_conditions, current_density)`
Solve 1D spin diffusion equation.
Args:
x: Position array
boundary_conditions: Boundary conditions
current_density: Applied current density
Returns:
Solution dictionary with charge and spin densities
**Source:** [Line 36](Python/Spintronics/core/spin_transport.py#L36)
##### `spin_injection_efficiency(self, R_contact, R_channel, polarization)`
Calculate spin injection efficiency.
Args:
R_contact: Contact resistance
R_channel: Channel resistance
polarization: Contact polarization
Returns:
Injection efficiency
**Source:** [Line 100](Python/Spintronics/core/spin_transport.py#L100)
**Class Source:** [Line 18](Python/Spintronics/core/spin_transport.py#L18)
### `MagnetoresistiveDevices`
Giant and tunneling magnetoresistance models.
#### Methods
##### `__init__(self, device_type)`
Initialize MR device.
Args:
device_type: 'gmr' or 'tmr'
**Source:** [Line 123](Python/Spintronics/core/spin_transport.py#L123)
##### `gmr_resistance(self, theta, R_parallel, R_antiparallel)`
Calculate GMR resistance vs angle.
Args:
theta: Angle between magnetizations
R_parallel: Resistance in parallel configuration
R_antiparallel: Resistance in antiparallel configuration
Returns:
Resistance
**Source:** [Line 132](Python/Spintronics/core/spin_transport.py#L132)
##### `tmr_resistance(self, theta, R_parallel, TMR_ratio)`
Calculate TMR resistance vs angle.
Args:
theta: Angle between magnetizations
R_parallel: Resistance in parallel configuration
TMR_ratio: TMR ratio (R_AP - R_P) / R_P
Returns:
Resistance
**Source:** [Line 150](Python/Spintronics/core/spin_transport.py#L150)
##### `switching_field(self, Ms, thickness, K_eff, alpha)`
Calculate switching field for free layer.
Args:
Ms: Saturation magnetization
thickness: Free layer thickness
K_eff: Effective anisotropy
alpha: Shape factor
Returns:
Switching field
**Source:** [Line 168](Python/Spintronics/core/spin_transport.py#L168)
##### `thermal_stability(self, Ms, volume, K_eff, temperature)`
Calculate thermal stability factor Δ = KV/kT.
Args:
Ms: Saturation magnetization
volume: Magnetic volume
K_eff: Effective anisotropy
temperature: Temperature
Returns:
Thermal stability factor
**Source:** [Line 187](Python/Spintronics/core/spin_transport.py#L187)
**Class Source:** [Line 120](Python/Spintronics/core/spin_transport.py#L120)
### `SpinHallEffect`
Spin Hall effect and spin-orbit coupling.
#### Methods
##### `__init__(self, spin_hall_angle, conductivity)`
Initialize spin Hall model.
Args:
spin_hall_angle: Spin Hall angle θ_SH
conductivity: Electrical conductivity
**Source:** [Line 210](Python/Spintronics/core/spin_transport.py#L210)
##### `spin_current_from_charge(self, j_charge)`
Calculate spin current from charge current via SHE.
Args:
j_charge: Charge current density vector
Returns:
Spin current density
**Source:** [Line 223](Python/Spintronics/core/spin_transport.py#L223)
##### `spin_hall_torque(self, j_charge, thickness, m, damping_like)`
Calculate spin Hall torque.
Args:
j_charge: Charge current density
thickness: Heavy metal thickness
m: Magnetization direction
damping_like: Include damping-like torque
Returns:
Torque vector
**Source:** [Line 238](Python/Spintronics/core/spin_transport.py#L238)
##### `critical_current_sot(self, Ms, thickness, K_eff, alpha)`
Calculate critical current for spin-orbit torque switching.
Args:
Ms: Saturation magnetization
thickness: Magnetic layer thickness
K_eff: Effective anisotropy
alpha: Gilbert damping
Returns:
Critical current density
**Source:** [Line 268](Python/Spintronics/core/spin_transport.py#L268)
**Class Source:** [Line 207](Python/Spintronics/core/spin_transport.py#L207)
### `RashbaEffect`
Rashba spin-orbit coupling effects.
#### Methods
##### `__init__(self, rashba_parameter)`
Initialize Rashba model.
Args:
rashba_parameter: Rashba parameter α_R (eV·m)
**Source:** [Line 294](Python/Spintronics/core/spin_transport.py#L294)
##### `rashba_field(self, k)`
Calculate effective Rashba magnetic field.
Args:
k: Wave vector
Returns:
Effective field
**Source:** [Line 305](Python/Spintronics/core/spin_transport.py#L305)
##### `spin_precession_frequency(self, k)`
Calculate spin precession frequency.
Args:
k: Wave vector magnitude
Returns:
Precession frequency
**Source:** [Line 321](Python/Spintronics/core/spin_transport.py#L321)
##### `current_induced_field(self, current_density, electric_field)`
Calculate current-induced effective field.
Args:
current_density: Current density
electric_field: Electric field
Returns:
Effective field
**Source:** [Line 336](Python/Spintronics/core/spin_transport.py#L336)
**Class Source:** [Line 291](Python/Spintronics/core/spin_transport.py#L291)
### `MagnonTransport`
Magnon-based spin transport.
#### Methods
##### `__init__(self, magnon_diffusivity, magnon_lifetime)`
Initialize magnon transport.
Args:
magnon_diffusivity: Magnon diffusion constant
magnon_lifetime: Magnon lifetime
**Source:** [Line 358](Python/Spintronics/core/spin_transport.py#L358)
##### `magnon_current(self, mu_gradient, temperature)`
Calculate magnon current from chemical potential gradient.
Args:
mu_gradient: Magnon chemical potential gradient
temperature: Temperature
Returns:
Magnon current density
**Source:** [Line 371](Python/Spintronics/core/spin_transport.py#L371)
##### `spin_seebeck_coefficient(self, temperature, magnetic_moment)`
Calculate spin Seebeck coefficient.
Args:
temperature: Temperature
magnetic_moment: Magnetic moment per unit volume
Returns:
Spin Seebeck coefficient
**Source:** [Line 391](Python/Spintronics/core/spin_transport.py#L391)
##### `magnon_drag_voltage(self, heat_current, magnon_phonon_coupling)`
Calculate voltage from magnon drag effect.
Args:
heat_current: Phonon heat current
magnon_phonon_coupling: Magnon-phonon coupling strength
Returns:
Induced voltage
**Source:** [Line 411](Python/Spintronics/core/spin_transport.py#L411)
**Class Source:** [Line 355](Python/Spintronics/core/spin_transport.py#L355)
### `SpinValve`
Complete spin valve device modeling.
#### Methods
##### `__init__(self, layers)`
Initialize spin valve structure.
Args:
layers: List of layer dictionaries with properties
**Source:** [Line 432](Python/Spintronics/core/spin_transport.py#L432)
##### `transfer_matrix_method(self, energy, k_parallel)`
Calculate transmission using transfer matrix method.
Args:
energy: Electron energy
k_parallel: Parallel momentum component
Returns:
Transmission coefficients
**Source:** [Line 442](Python/Spintronics/core/spin_transport.py#L442)
##### `iv_characteristics(self, voltages, temperature)`
Calculate I-V characteristics.
Args:
voltages: Voltage array
temperature: Temperature
Returns:
Current and conductance
**Source:** [Line 479](Python/Spintronics/core/spin_transport.py#L479)
**Class Source:** [Line 429](Python/Spintronics/core/spin_transport.py#L429)