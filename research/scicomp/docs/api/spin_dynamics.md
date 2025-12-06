# spin_dynamics
**Module:** `Python/Spintronics/core/spin_dynamics.py`
## Overview
Spin dynamics and magnetization evolution.
This module provides tools for spin dynamics including:
- Landau-Lifshitz-Gilbert equation
- Spin precession dynamics
- Magnetic anisotropy effects
- Damping mechanisms
## Classes
### `LandauLifshitzGilbert`
Landau-Lifshitz-Gilbert equation solver.
#### Methods
##### `__init__(self, gamma, alpha)`
Initialize LLG solver.
Args:
gamma: Gyromagnetic ratio (m/(A·s²))
alpha: Gilbert damping parameter
**Source:** [Line 21](Python/Spintronics/core/spin_dynamics.py#L21)
##### `effective_field(self, m)`
Calculate effective magnetic field.
Args:
m: Magnetization vector (normalized)
**params: Field parameters
Returns:
Effective field vector
**Source:** [Line 33](Python/Spintronics/core/spin_dynamics.py#L33)
##### `llg_equation(self, t, m)`
LLG equation: dm/dt = -γ/(1+α²) [m×H_eff + α m×(m×H_eff)].
Args:
t: Time
m: Magnetization vector
**params: Field parameters
Returns:
Time derivative dm/dt
**Source:** [Line 73](Python/Spintronics/core/spin_dynamics.py#L73)
##### `solve(self, m0, t_span, t_eval)`
Solve LLG equation.
Args:
m0: Initial magnetization
t_span: Time span (start, end)
t_eval: Time points for evaluation
**params: Field parameters
Returns:
Solution dictionary
**Source:** [Line 97](Python/Spintronics/core/spin_dynamics.py#L97)
##### `_calculate_energies(self, magnetization)`
Calculate magnetic energies during evolution.
**Source:** [Line 135](Python/Spintronics/core/spin_dynamics.py#L135)
**Class Source:** [Line 18](Python/Spintronics/core/spin_dynamics.py#L18)
### `SpinWaves`
Spin wave analysis and dispersion relations.
#### Methods
##### `__init__(self, Ms, A_ex, gamma)`
Initialize spin wave analysis.
Args:
Ms: Saturation magnetization
A_ex: Exchange stiffness
gamma: Gyromagnetic ratio
**Source:** [Line 170](Python/Spintronics/core/spin_dynamics.py#L170)
##### `dispersion_relation(self, k, H_ext, K_u)`
Calculate spin wave dispersion relation.
Args:
k: Wave vector array
H_ext: External field
K_u: Uniaxial anisotropy constant
Returns:
Frequency array
**Source:** [Line 184](Python/Spintronics/core/spin_dynamics.py#L184)
##### `group_velocity(self, k)`
Calculate group velocity v_g = dω/dk.
Args:
k: Wave vector array
**params: Material parameters
Returns:
Group velocity array
**Source:** [Line 211](Python/Spintronics/core/spin_dynamics.py#L211)
##### `magnon_density_of_states(self, omega, lattice_param)`
Calculate magnon density of states.
Args:
omega: Frequency array
lattice_param: Lattice parameter
Returns:
Density of states
**Source:** [Line 230](Python/Spintronics/core/spin_dynamics.py#L230)
**Class Source:** [Line 167](Python/Spintronics/core/spin_dynamics.py#L167)
### `MagneticDomains`
Magnetic domain wall dynamics.
#### Methods
##### `__init__(self, Ms, A_ex, K_u, width)`
Initialize domain wall system.
Args:
Ms: Saturation magnetization
A_ex: Exchange stiffness
K_u: Uniaxial anisotropy
width: Domain wall width
**Source:** [Line 254](Python/Spintronics/core/spin_dynamics.py#L254)
##### `bloch_wall_profile(self, x, x0)`
Bloch domain wall magnetization profile.
Args:
x: Position array
x0: Wall center position
Returns:
Magnetization components
**Source:** [Line 275](Python/Spintronics/core/spin_dynamics.py#L275)
##### `neel_wall_profile(self, x, x0)`
Néel domain wall magnetization profile.
Args:
x: Position array
x0: Wall center position
Returns:
Magnetization components
**Source:** [Line 295](Python/Spintronics/core/spin_dynamics.py#L295)
##### `walker_breakdown(self, H_field)`
Calculate Walker breakdown field and velocity.
Args:
H_field: Applied field
Returns:
Breakdown field and velocity
**Source:** [Line 315](Python/Spintronics/core/spin_dynamics.py#L315)
##### `domain_wall_energy(self, wall_type)`
Calculate domain wall energy per unit area.
Args:
wall_type: 'bloch' or 'neel'
Returns:
Wall energy density
**Source:** [Line 340](Python/Spintronics/core/spin_dynamics.py#L340)
**Class Source:** [Line 251](Python/Spintronics/core/spin_dynamics.py#L251)
### `SpinTorque`
Spin-transfer torque effects.
#### Methods
##### `__init__(self, gamma, hbar)`
Initialize spin torque calculations.
Args:
gamma: Gyromagnetic ratio
hbar: Reduced Planck constant
**Source:** [Line 365](Python/Spintronics/core/spin_dynamics.py#L365)
##### `slonczewski_torque(self, m, p, current, thickness, spin_polarization)`
Calculate Slonczewski spin-transfer torque.
Args:
m: Free layer magnetization
p: Fixed layer magnetization
current: Current density (A/m²)
thickness: Free layer thickness
spin_polarization: Spin polarization efficiency
Returns:
Torque vector
**Source:** [Line 378](Python/Spintronics/core/spin_dynamics.py#L378)
##### `field_like_torque(self, m, p, current, beta)`
Calculate field-like torque.
Args:
m: Free layer magnetization
p: Fixed layer magnetization
current: Current density
beta: Field-like torque efficiency
Returns:
Field-like torque
**Source:** [Line 407](Python/Spintronics/core/spin_dynamics.py#L407)
##### `critical_current(self, Ms, thickness, alpha, spin_polarization)`
Calculate critical current for switching.
Args:
Ms: Saturation magnetization
thickness: Layer thickness
alpha: Gilbert damping
spin_polarization: Spin polarization
Returns:
Critical current density
**Source:** [Line 429](Python/Spintronics/core/spin_dynamics.py#L429)
**Class Source:** [Line 362](Python/Spintronics/core/spin_dynamics.py#L362)
### `MagnetocrystallineAnisotropy`
Magnetocrystalline anisotropy calculations.
#### Methods
##### `__init__(self, crystal_class)`
Initialize anisotropy calculations.
Args:
crystal_class: Crystal class ('cubic', 'uniaxial', 'orthorhombic')
**Source:** [Line 452](Python/Spintronics/core/spin_dynamics.py#L452)
##### `anisotropy_energy(self, m)`
Calculate magnetocrystalline anisotropy energy.
Args:
m: Magnetization direction (normalized)
**constants: Anisotropy constants
Returns:
Anisotropy energy density
**Source:** [Line 461](Python/Spintronics/core/spin_dynamics.py#L461)
##### `anisotropy_field(self, m)`
Calculate anisotropy field H_anis = -∂E/∂m.
Args:
m: Magnetization direction
**constants: Anisotropy constants
Returns:
Anisotropy field
**Source:** [Line 507](Python/Spintronics/core/spin_dynamics.py#L507)
##### `easy_axes(self)`
Find crystallographic easy axes.
Args:
**constants: Anisotropy constants
Returns:
List of easy axis directions
**Source:** [Line 535](Python/Spintronics/core/spin_dynamics.py#L535)
**Class Source:** [Line 449](Python/Spintronics/core/spin_dynamics.py#L449)