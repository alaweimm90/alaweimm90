# thermal_mechanical
**Module:** `Python/Multiphysics/thermal_mechanical.py`
## Overview
Thermal-Mechanical Coupling Module.
This module implements thermal-mechanical coupling for problems involving
heat transfer and structural deformation, including thermal expansion,
thermoelastic analysis, and coupled heat conduction.
Classes:
ThermalMechanicalCoupling: Main coupled solver
ThermalExpansion: Thermal expansion models
ThermoelasticSolver: Coupled thermoelastic solver
HeatGenerationModel: Heat generation from deformation
Functions:
thermal_stress_analysis: Compute thermal stresses
coupled_heat_conduction: Solve coupled heat transfer
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `thermal_stress_analysis(geometry, temperature_field, material, constraint_type)`
Analyze thermal stresses in simple geometries.
Args:
geometry: Geometric parameters
temperature_field: Temperature distribution function
material: Material properties
constraint_type: Constraint condition
Returns:
Stress analysis results
**Source:** [Line 661](Python/Multiphysics/thermal_mechanical.py#L661)
### `coupled_heat_conduction(mesh, thermal_props, mech_props, heat_source, boundary_conditions, coupling_effects)`
Solve coupled heat conduction with mechanical effects.
Args:
mesh: Finite element mesh
thermal_props: Thermal properties
mech_props: Mechanical properties
heat_source: Heat source function
boundary_conditions: Combined BC
coupling_effects: Include coupling effects
Returns:
Coupled solution
**Source:** [Line 723](Python/Multiphysics/thermal_mechanical.py#L723)
## Classes
### `ThermalProperties`
Thermal material properties.
#### Methods
##### `thermal_diffusivity(self)`
Compute thermal diffusivity.
**Source:** [Line 41](Python/Multiphysics/thermal_mechanical.py#L41)
**Class Source:** [Line 32](Python/Multiphysics/thermal_mechanical.py#L32)
### `MechanicalProperties`
Mechanical material properties.
#### Methods
##### `shear_modulus(self)`
Compute shear modulus.
**Source:** [Line 55](Python/Multiphysics/thermal_mechanical.py#L55)
##### `bulk_modulus(self)`
Compute bulk modulus.
**Source:** [Line 60](Python/Multiphysics/thermal_mechanical.py#L60)
##### `lame_lambda(self)`
First Lamé parameter.
**Source:** [Line 65](Python/Multiphysics/thermal_mechanical.py#L65)
##### `lame_mu(self)`
Second Lamé parameter (shear modulus).
**Source:** [Line 71](Python/Multiphysics/thermal_mechanical.py#L71)
**Class Source:** [Line 47](Python/Multiphysics/thermal_mechanical.py#L47)
### `ThermalExpansion`
Thermal expansion models.
Provides various thermal expansion models including
isotropic, orthotropic, and temperature-dependent expansions.
#### Methods
##### `__init__(self, expansion_type)`
Initialize thermal expansion model.
Args:
expansion_type: Type of thermal expansion model
**Source:** [Line 83](Python/Multiphysics/thermal_mechanical.py#L83)
##### `compute_thermal_strain(self, temperature, properties, direction)`
Compute thermal strain.
Args:
temperature: Temperature field
properties: Material properties
direction: Direction for anisotropic expansion
Returns:
Thermal strain tensor/components
**Source:** [Line 92](Python/Multiphysics/thermal_mechanical.py#L92)
##### `_temperature_dependent_alpha(self, temperature)`
Temperature-dependent expansion coefficient.
**Source:** [Line 144](Python/Multiphysics/thermal_mechanical.py#L144)
##### `compute_thermal_stress(self, temperature, thermal_props, mech_props, constraint)`
Compute thermal stress.
Args:
temperature: Temperature field
thermal_props: Thermal properties
mech_props: Mechanical properties
constraint: Constraint type (free, constrained)
Returns:
Thermal stress tensor
**Source:** [Line 152](Python/Multiphysics/thermal_mechanical.py#L152)
**Class Source:** [Line 76](Python/Multiphysics/thermal_mechanical.py#L76)
### `ThermoelasticSolver`
Coupled thermoelastic solver.
Solves coupled heat conduction and elastic deformation
with thermal expansion effects.
#### Methods
##### `__init__(self, mesh, thermal_props, mech_props)`
Initialize thermoelastic solver.
Args:
mesh: Finite element mesh
thermal_props: Thermal properties
mech_props: Mechanical properties
**Source:** [Line 206](Python/Multiphysics/thermal_mechanical.py#L206)
##### `_setup_system(self)`
Setup finite element system.
**Source:** [Line 236](Python/Multiphysics/thermal_mechanical.py#L236)
##### `solve_steady_state(self, thermal_bc, mechanical_bc, heat_source, body_force)`
Solve steady-state thermoelastic problem.
Args:
thermal_bc: Thermal boundary conditions
mechanical_bc: Mechanical boundary conditions
heat_source: Volumetric heat source
body_force: Body force vector
Returns:
Solution fields
**Source:** [Line 260](Python/Multiphysics/thermal_mechanical.py#L260)
##### `solve_transient(self, time_span, dt, initial_temp, thermal_bc, mechanical_bc)`
Solve transient thermoelastic problem.
Args:
time_span: Time interval (t0, tf)
dt: Time step
initial_temp: Initial temperature
thermal_bc: Thermal boundary conditions
mechanical_bc: Mechanical boundary conditions
Returns:
Time history of solution fields
**Source:** [Line 325](Python/Multiphysics/thermal_mechanical.py#L325)
##### `_apply_thermal_bc(self, bc, matrix, rhs)`
Apply thermal boundary conditions.
**Source:** [Line 392](Python/Multiphysics/thermal_mechanical.py#L392)
##### `_apply_mechanical_bc(self, bc, matrix, rhs)`
Apply mechanical boundary conditions.
**Source:** [Line 407](Python/Multiphysics/thermal_mechanical.py#L407)
##### `_compute_thermal_load(self, thermal_strain)`
Compute equivalent thermal load vector.
**Source:** [Line 429](Python/Multiphysics/thermal_mechanical.py#L429)
##### `_compute_stress(self, displacement, thermal_strain)`
Compute total stress (mechanical + thermal).
**Source:** [Line 454](Python/Multiphysics/thermal_mechanical.py#L454)
**Class Source:** [Line 199](Python/Multiphysics/thermal_mechanical.py#L199)
### `HeatGenerationModel`
Heat generation from mechanical processes.
Models heat generation from plastic deformation,
friction, and viscoelastic dissipation.
#### Methods
##### `__init__(self, generation_type)`
Initialize heat generation model.
Args:
generation_type: Type of heat generation
**Source:** [Line 505](Python/Multiphysics/thermal_mechanical.py#L505)
##### `compute_heat_generation(self, stress, strain_rate, material_props)`
Compute volumetric heat generation rate.
Args:
stress: Stress tensor
strain_rate: Strain rate tensor
material_props: Material properties
Returns:
Heat generation rate (W/m³)
**Source:** [Line 514](Python/Multiphysics/thermal_mechanical.py#L514)
##### `_compute_von_mises_stress(self, stress)`
Compute von Mises equivalent stress.
**Source:** [Line 563](Python/Multiphysics/thermal_mechanical.py#L563)
**Class Source:** [Line 498](Python/Multiphysics/thermal_mechanical.py#L498)
### `ThermalMechanicalCoupling`
Main thermal-mechanical coupling solver.
Coordinates thermal and mechanical solvers with
appropriate coupling strategies.
#### Methods
##### `__init__(self, mesh, thermal_props, mech_props, coupling_scheme)`
Initialize coupled solver.
Args:
mesh: Finite element mesh
thermal_props: Thermal properties
mech_props: Mechanical properties
coupling_scheme: Coupling strategy
**Source:** [Line 589](Python/Multiphysics/thermal_mechanical.py#L589)
##### `setup_domains(self)`
Setup thermal and mechanical domains.
**Source:** [Line 617](Python/Multiphysics/thermal_mechanical.py#L617)
##### `setup_coupling(self)`
Setup thermal-mechanical coupling.
**Source:** [Line 622](Python/Multiphysics/thermal_mechanical.py#L622)
##### `solve_coupled_problem(self, time_span, thermal_bc, mechanical_bc, initial_temp)`
Solve coupled thermal-mechanical problem.
Args:
time_span: Time interval for transient analysis
thermal_bc: Thermal boundary conditions
mechanical_bc: Mechanical boundary conditions
initial_temp: Initial temperature field
Returns:
Solution data
**Source:** [Line 627](Python/Multiphysics/thermal_mechanical.py#L627)
**Class Source:** [Line 582](Python/Multiphysics/thermal_mechanical.py#L582)