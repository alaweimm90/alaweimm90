# electromagnetic
**Module:** `Python/Multiphysics/electromagnetic.py`
## Overview
Electromagnetic-Thermal Coupling Module.
This module implements electromagnetic-thermal coupling for problems involving
electromagnetic fields and heat generation, including Joule heating,
induction heating, and eddy current effects.
Classes:
ElectromagneticThermalCoupling: Main coupled solver
MaxwellSolver: Electromagnetic field solver
JouleHeating: Joule heating model
InductionHeating: Induction heating solver
EddyCurrentSolver: Eddy current analysis
Functions:
electromagnetic_heating: Compute EM heating
coupled_em_thermal: Solve coupled EM-thermal problem
Author: Berkeley SciComp Team
Date: 2024
## Constants
- **`MU_0`**
- **`EPSILON_0`**
- **`SIGMA_0`**
## Functions
### `electromagnetic_heating(geometry, current, frequency, material)`
Calculate electromagnetic heating for simple geometries.
Args:
geometry: Geometric parameters
current: Applied current (A)
frequency: Frequency (Hz)
material: Material properties
Returns:
Heating analysis results
**Source:** [Line 760](Python/Multiphysics/electromagnetic.py#L760)
### `coupled_em_thermal(mesh, em_source, boundary_conditions, materials, analysis_type)`
Solve coupled electromagnetic-thermal problem.
Args:
mesh: Finite element mesh
em_source: EM excitation
boundary_conditions: Combined BC
materials: Material properties
analysis_type: Analysis type (steady/transient)
Returns:
Coupled solution fields
**Source:** [Line 839](Python/Multiphysics/electromagnetic.py#L839)
## Classes
### `ElectromagneticProperties`
Electromagnetic material properties.
#### Methods
##### `skin_depth(self, frequency)`
Electromagnetic skin depth.
**Source:** [Line 46](Python/Multiphysics/electromagnetic.py#L46)
##### `impedance(self)`
Wave impedance.
**Source:** [Line 51](Python/Multiphysics/electromagnetic.py#L51)
**Class Source:** [Line 39](Python/Multiphysics/electromagnetic.py#L39)
### `MaxwellSolver`
Electromagnetic field solver.
Solves Maxwell's equations for electromagnetic fields
with support for various formulations.
#### Methods
##### `__init__(self, mesh, properties, formulation)`
Initialize Maxwell solver.
Args:
mesh: Finite element mesh
properties: EM properties
formulation: Solution formulation (A-phi, E-B, etc.)
**Source:** [Line 63](Python/Multiphysics/electromagnetic.py#L63)
##### `_setup_system(self)`
Setup finite element system for EM.
**Source:** [Line 96](Python/Multiphysics/electromagnetic.py#L96)
##### `_estimate_edges(self)`
Estimate number of edges for edge elements.
**Source:** [Line 117](Python/Multiphysics/electromagnetic.py#L117)
##### `solve_static(self, boundary_conditions, current_source)`
Solve static (DC) electromagnetic problem.
Args:
boundary_conditions: EM boundary conditions
current_source: Current density source
Returns:
EM field solution
**Source:** [Line 122](Python/Multiphysics/electromagnetic.py#L122)
##### `solve_frequency_domain(self, frequency, boundary_conditions, current_source)`
Solve frequency domain electromagnetic problem.
Args:
frequency: Frequency (Hz)
boundary_conditions: EM boundary conditions
current_source: Current source (complex)
Returns:
Complex EM field solution
**Source:** [Line 181](Python/Multiphysics/electromagnetic.py#L181)
##### `_apply_em_bc(self, bc, matrix, rhs)`
Apply electromagnetic boundary conditions.
**Source:** [Line 239](Python/Multiphysics/electromagnetic.py#L239)
##### `_apply_potential_bc(self, bc, matrix, rhs)`
Apply scalar potential boundary conditions.
**Source:** [Line 256](Python/Multiphysics/electromagnetic.py#L256)
##### `_compute_curl(self, vector_field)`
Compute curl of vector field (simplified).
**Source:** [Line 262](Python/Multiphysics/electromagnetic.py#L262)
##### `_compute_gradient(self, scalar_field)`
Compute gradient of scalar field.
**Source:** [Line 280](Python/Multiphysics/electromagnetic.py#L280)
##### `compute_power_density(self)`
Compute electromagnetic power density.
**Source:** [Line 293](Python/Multiphysics/electromagnetic.py#L293)
**Class Source:** [Line 56](Python/Multiphysics/electromagnetic.py#L56)
### `JouleHeating`
Joule heating model.
Computes heat generation from electrical current flow.
#### Methods
##### `__init__(self)`
Initialize Joule heating model.
**Source:** [Line 317](Python/Multiphysics/electromagnetic.py#L317)
##### `compute_heating(self, current_density, electric_field, conductivity)`
Compute Joule heating power density.
Args:
current_density: Current density J (A/m²)
electric_field: Electric field E (V/m)
conductivity: Electrical conductivity (S/m)
Returns:
Volumetric heat generation (W/m³)
**Source:** [Line 321](Python/Multiphysics/electromagnetic.py#L321)
##### `compute_heating_from_current(self, current, resistance)`
Compute total Joule heating from current and resistance.
Args:
current: Total current (A)
resistance: Total resistance (Ω)
Returns:
Total heating power (W)
**Source:** [Line 346](Python/Multiphysics/electromagnetic.py#L346)
**Class Source:** [Line 311](Python/Multiphysics/electromagnetic.py#L311)
### `InductionHeating`
Induction heating solver.
Solves coupled electromagnetic-thermal problem for
induction heating applications.
#### Methods
##### `__init__(self, mesh, em_props, thermal_props)`
Initialize induction heating solver.
Args:
mesh: Finite element mesh
em_props: Electromagnetic properties
thermal_props: Thermal properties
**Source:** [Line 368](Python/Multiphysics/electromagnetic.py#L368)
##### `solve_induction_heating(self, coil_current, frequency, thermal_bc, time_span)`
Solve induction heating problem.
Args:
coil_current: Coil current specification
frequency: Operating frequency (Hz)
thermal_bc: Thermal boundary conditions
time_span: Time interval for transient
Returns:
Solution fields
**Source:** [Line 392](Python/Multiphysics/electromagnetic.py#L392)
##### `_solve_steady_thermal(self, heat_source, bc)`
Solve steady-state heat conduction.
**Source:** [Line 439](Python/Multiphysics/electromagnetic.py#L439)
##### `_solve_transient_thermal(self, heat_source, bc, time_span)`
Solve transient heat conduction.
**Source:** [Line 459](Python/Multiphysics/electromagnetic.py#L459)
##### `optimize_coil_design(self, target_temperature, constraints)`
Optimize induction coil design.
Args:
target_temperature: Desired temperature
constraints: Design constraints
Returns:
Optimal coil parameters
**Source:** [Line 469](Python/Multiphysics/electromagnetic.py#L469)
**Class Source:** [Line 361](Python/Multiphysics/electromagnetic.py#L361)
### `EddyCurrentSolver`
Eddy current analysis solver.
Specialized solver for eddy current problems in
conducting materials.
#### Methods
##### `__init__(self, mesh, properties)`
Initialize eddy current solver.
Args:
mesh: Finite element mesh
properties: Material properties
**Source:** [Line 507](Python/Multiphysics/electromagnetic.py#L507)
##### `analyze_eddy_currents(self, applied_field, frequency, motion)`
Analyze eddy currents in conductor.
Args:
applied_field: Applied magnetic field function
frequency: Field frequency (Hz)
motion: Conductor motion specification
Returns:
Eddy current analysis results
**Source:** [Line 520](Python/Multiphysics/electromagnetic.py#L520)
##### `_compute_electromagnetic_forces(self, current_density, magnetic_field)`
Compute Lorentz forces.
**Source:** [Line 575](Python/Multiphysics/electromagnetic.py#L575)
##### `compute_impedance(self, frequency, geometry)`
Compute frequency-dependent impedance.
Args:
frequency: Operating frequency
geometry: Conductor geometry
Returns:
Complex impedance
**Source:** [Line 587](Python/Multiphysics/electromagnetic.py#L587)
**Class Source:** [Line 500](Python/Multiphysics/electromagnetic.py#L500)
### `ElectromagneticThermalCoupling`
Main electromagnetic-thermal coupling solver.
Coordinates EM and thermal solvers for coupled problems.
#### Methods
##### `__init__(self, mesh, em_props, thermal_props, coupling_scheme)`
Initialize coupled EM-thermal solver.
Args:
mesh: Finite element mesh
em_props: EM properties
thermal_props: Thermal properties
coupling_scheme: Coupling strategy
**Source:** [Line 634](Python/Multiphysics/electromagnetic.py#L634)
##### `setup_domains(self)`
Setup EM and thermal domains.
**Source:** [Line 660](Python/Multiphysics/electromagnetic.py#L660)
##### `setup_coupling(self)`
Setup EM-thermal coupling.
**Source:** [Line 665](Python/Multiphysics/electromagnetic.py#L665)
##### `solve_coupled_em_thermal(self, em_source, thermal_bc, frequency, max_iterations, tolerance)`
Solve coupled EM-thermal problem.
Args:
em_source: EM source specification
thermal_bc: Thermal boundary conditions
frequency: EM frequency
max_iterations: Coupling iterations
tolerance: Convergence tolerance
Returns:
Coupled solution
**Source:** [Line 670](Python/Multiphysics/electromagnetic.py#L670)
**Class Source:** [Line 628](Python/Multiphysics/electromagnetic.py#L628)