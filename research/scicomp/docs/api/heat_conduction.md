# heat_conduction
**Module:** `Python/Thermal_Transport/core/heat_conduction.py`
## Overview
Heat conduction and thermal transport modeling.
This module provides tools for thermal transport including:
- Heat equation solvers
- Thermal conductivity models
- Phonon transport
- Thermoelectric effects
## Classes
### `HeatEquation`
Heat equation solver for various geometries.
#### Methods
##### `__init__(self, thermal_diffusivity)`
Initialize heat equation solver.
Args:
thermal_diffusivity: Thermal diffusivity α = k/(ρ·c)
**Source:** [Line 23](Python/Thermal_Transport/core/heat_conduction.py#L23)
##### `solve_1d_transient(self, x, t, initial_condition, boundary_conditions, source_term)`
Solve 1D transient heat equation.
Args:
x: Spatial grid
t: Time grid
initial_condition: Initial temperature function
boundary_conditions: Boundary condition specification
source_term: Heat source function Q(x,t)
Returns:
Temperature field T(x,t)
**Source:** [Line 32](Python/Thermal_Transport/core/heat_conduction.py#L32)
##### `solve_1d_steady_state(self, x, boundary_conditions, source_term, thermal_conductivity)`
Solve 1D steady-state heat equation.
Args:
x: Spatial grid
boundary_conditions: Boundary conditions
source_term: Heat source Q(x)
thermal_conductivity: Thermal conductivity k(x)
Returns:
Temperature distribution T(x)
**Source:** [Line 82](Python/Thermal_Transport/core/heat_conduction.py#L82)
##### `solve_2d_steady_state(self, x, y, boundary_conditions, source_term, thermal_conductivity)`
Solve 2D steady-state heat equation using finite differences.
Args:
x: X-direction grid
y: Y-direction grid
boundary_conditions: Boundary conditions
source_term: Heat source Q(x,y)
thermal_conductivity: Thermal conductivity
Returns:
Temperature field T(x,y)
**Source:** [Line 129](Python/Thermal_Transport/core/heat_conduction.py#L129)
##### `_apply_boundary_conditions_1d(self, T, n, x, t, bc)`
Apply boundary conditions for 1D transient problem.
**Source:** [Line 202](Python/Thermal_Transport/core/heat_conduction.py#L202)
##### `_apply_boundary_conditions_steady(self, A, b, x, bc)`
Apply boundary conditions for steady-state problem.
**Source:** [Line 230](Python/Thermal_Transport/core/heat_conduction.py#L230)
**Class Source:** [Line 20](Python/Thermal_Transport/core/heat_conduction.py#L20)
### `PhononTransport`
Phonon transport and Boltzmann transport equation.
#### Methods
##### `__init__(self, debye_temperature, sound_velocities)`
Initialize phonon transport model.
Args:
debye_temperature: Debye temperature
sound_velocities: Sound velocities (vL, vT1, vT2)
**Source:** [Line 253](Python/Thermal_Transport/core/heat_conduction.py#L253)
##### `debye_frequency(self, density, n_atoms)`
Calculate Debye cutoff frequency.
Args:
density: Material density
n_atoms: Number of atoms per unit volume
Returns:
Debye frequency
**Source:** [Line 267](Python/Thermal_Transport/core/heat_conduction.py#L267)
##### `phonon_specific_heat(self, temperature)`
Calculate phonon specific heat using Debye model.
Args:
temperature: Temperature
Returns:
Specific heat per unit volume
**Source:** [Line 286](Python/Thermal_Transport/core/heat_conduction.py#L286)
##### `thermal_conductivity_kinetic(self, temperature, mean_free_path, density)`
Calculate thermal conductivity using kinetic theory.
Args:
temperature: Temperature
mean_free_path: Phonon mean free path
density: Material density
Returns:
Thermal conductivity
**Source:** [Line 311](Python/Thermal_Transport/core/heat_conduction.py#L311)
##### `umklapp_scattering_rate(self, temperature, gruneisen_parameter)`
Calculate Umklapp scattering rate.
Args:
temperature: Temperature
gruneisen_parameter: Grüneisen parameter
Returns:
Umklapp scattering rate
**Source:** [Line 336](Python/Thermal_Transport/core/heat_conduction.py#L336)
##### `boundary_scattering_rate(self, characteristic_length)`
Calculate boundary scattering rate.
Args:
characteristic_length: Characteristic sample dimension
Returns:
Boundary scattering rate
**Source:** [Line 353](Python/Thermal_Transport/core/heat_conduction.py#L353)
##### `solve_bte_relaxation_time(self, temperature, scattering_rates)`
Solve BTE using relaxation time approximation.
Args:
temperature: Temperature
scattering_rates: List of scattering rates
Returns:
Thermal conductivity
**Source:** [Line 368](Python/Thermal_Transport/core/heat_conduction.py#L368)
**Class Source:** [Line 250](Python/Thermal_Transport/core/heat_conduction.py#L250)
### `ThermoelectricEffects`
Thermoelectric transport phenomena.
#### Methods
##### `__init__(self)`
Initialize thermoelectric model.
**Source:** [Line 397](Python/Thermal_Transport/core/heat_conduction.py#L397)
##### `seebeck_coefficient(self, carrier_concentration, effective_mass, temperature, scattering_parameter)`
Calculate Seebeck coefficient.
Args:
carrier_concentration: Carrier concentration
effective_mass: Effective mass
temperature: Temperature
scattering_parameter: Scattering parameter (0.5 for acoustic phonons)
Returns:
Seebeck coefficient
**Source:** [Line 402](Python/Thermal_Transport/core/heat_conduction.py#L402)
##### `electrical_conductivity(self, carrier_concentration, mobility)`
Calculate electrical conductivity.
Args:
carrier_concentration: Carrier concentration
mobility: Carrier mobility
Returns:
Electrical conductivity
**Source:** [Line 425](Python/Thermal_Transport/core/heat_conduction.py#L425)
##### `thermal_conductivity_electronic(self, electrical_conductivity, temperature, lorenz_number)`
Calculate electronic thermal conductivity.
Args:
electrical_conductivity: Electrical conductivity
temperature: Temperature
lorenz_number: Lorenz number (Wiedemann-Franz law)
Returns:
Electronic thermal conductivity
**Source:** [Line 440](Python/Thermal_Transport/core/heat_conduction.py#L440)
##### `figure_of_merit(self, seebeck, electrical_conductivity, thermal_conductivity, temperature)`
Calculate thermoelectric figure of merit ZT.
Args:
seebeck: Seebeck coefficient
electrical_conductivity: Electrical conductivity
thermal_conductivity: Total thermal conductivity
temperature: Temperature
Returns:
Figure of merit ZT
**Source:** [Line 457](Python/Thermal_Transport/core/heat_conduction.py#L457)
##### `peltier_coefficient(self, seebeck, temperature)`
Calculate Peltier coefficient using Thomson relation.
Args:
seebeck: Seebeck coefficient
temperature: Temperature
Returns:
Peltier coefficient
**Source:** [Line 476](Python/Thermal_Transport/core/heat_conduction.py#L476)
##### `thomson_coefficient(self, seebeck, temperature)`
Calculate Thomson coefficient.
Args:
seebeck: Seebeck coefficient
temperature: Temperature
Returns:
Thomson coefficient
**Source:** [Line 490](Python/Thermal_Transport/core/heat_conduction.py#L490)
**Class Source:** [Line 394](Python/Thermal_Transport/core/heat_conduction.py#L394)
### `HeatExchanger`
Heat exchanger modeling and design.
#### Methods
##### `__init__(self, exchanger_type)`
Initialize heat exchanger.
Args:
exchanger_type: Type of heat exchanger
**Source:** [Line 509](Python/Thermal_Transport/core/heat_conduction.py#L509)
##### `effectiveness_ntu(self, ntu, capacity_ratio)`
Calculate effectiveness using NTU method.
Args:
ntu: Number of transfer units
capacity_ratio: Capacity rate ratio (Cmin/Cmax)
Returns:
Heat exchanger effectiveness
**Source:** [Line 518](Python/Thermal_Transport/core/heat_conduction.py#L518)
##### `heat_transfer_rate(self, effectiveness, c_min, t_hot_in, t_cold_in)`
Calculate heat transfer rate.
Args:
effectiveness: Heat exchanger effectiveness
c_min: Minimum capacity rate
t_hot_in: Hot fluid inlet temperature
t_cold_in: Cold fluid inlet temperature
Returns:
Heat transfer rate
**Source:** [Line 549](Python/Thermal_Transport/core/heat_conduction.py#L549)
##### `pressure_drop(self, reynolds_number, friction_factor, length, diameter, density, velocity)`
Calculate pressure drop.
Args:
reynolds_number: Reynolds number
friction_factor: Friction factor
length: Flow length
diameter: Hydraulic diameter
density: Fluid density
velocity: Flow velocity
Returns:
Pressure drop
**Source:** [Line 568](Python/Thermal_Transport/core/heat_conduction.py#L568)
**Class Source:** [Line 506](Python/Thermal_Transport/core/heat_conduction.py#L506)
### `NanoscaleHeatTransfer`
Nanoscale and microscale heat transfer phenomena.
#### Methods
##### `__init__(self)`
Initialize nanoscale heat transfer model.
**Source:** [Line 593](Python/Thermal_Transport/core/heat_conduction.py#L593)
##### `ballistic_thermal_conductance(self, temperature, contact_area, transmission_coefficient)`
Calculate ballistic thermal conductance.
Args:
temperature: Temperature
contact_area: Contact area
transmission_coefficient: Transmission probability
Returns:
Thermal conductance
**Source:** [Line 599](Python/Thermal_Transport/core/heat_conduction.py#L599)
##### `kapitza_resistance(self, debye_temp_1, debye_temp_2, temperature, interface_area)`
Calculate Kapitza thermal boundary resistance.
Args:
debye_temp_1: Debye temperature of material 1
debye_temp_2: Debye temperature of material 2
temperature: Temperature
interface_area: Interface area
Returns:
Kapitza resistance
**Source:** [Line 621](Python/Thermal_Transport/core/heat_conduction.py#L621)
##### `near_field_radiation(self, temperature_1, temperature_2, gap_distance, area, material_properties)`
Calculate near-field radiative heat transfer.
Args:
temperature_1: Temperature of surface 1
temperature_2: Temperature of surface 2
gap_distance: Gap distance
area: Surface area
material_properties: Material optical properties
Returns:
Radiative heat flux
**Source:** [Line 643](Python/Thermal_Transport/core/heat_conduction.py#L643)
##### `phonon_tunneling(self, gap_distance, phonon_wavelength, transmission_probability)`
Calculate phonon tunneling probability.
Args:
gap_distance: Gap distance
phonon_wavelength: Phonon wavelength
transmission_probability: Base transmission probability
Returns:
Tunneling-enhanced transmission
**Source:** [Line 675](Python/Thermal_Transport/core/heat_conduction.py#L675)
**Class Source:** [Line 590](Python/Thermal_Transport/core/heat_conduction.py#L590)