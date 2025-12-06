# beam_theory
**Module:** `Python/Elasticity/core/beam_theory.py`
## Overview
Structural Beam Analysis and Theory
Comprehensive beam analysis including Euler-Bernoulli, Timoshenko, and
nonlinear beam theories with various boundary conditions and loading scenarios.
## Functions
### `create_standard_beam_sections()`
Create database of standard beam cross-sections.
**Source:** [Line 657](Python/Elasticity/core/beam_theory.py#L657)
## Classes
### `BeamProperties`
Beam geometric and material properties.
#### Methods
##### `__post_init__(self)`
Calculate derived properties.
**Source:** [Line 28](Python/Elasticity/core/beam_theory.py#L28)
**Class Source:** [Line 18](Python/Elasticity/core/beam_theory.py#L18)
### `LoadCase`
Define loading conditions for beam analysis.
#### Methods
##### `__init__(self, load_type)`
Initialize load case.
Parameters:
load_type: 'point', 'distributed', 'moment', or 'function'
**kwargs: Load-specific parameters
**Source:** [Line 38](Python/Elasticity/core/beam_theory.py#L38)
##### `point_load(cls, force, position)`
Create point load.
**Source:** [Line 50](Python/Elasticity/core/beam_theory.py#L50)
##### `distributed_load(cls, intensity, start, end)`
Create distributed load.
**Source:** [Line 55](Python/Elasticity/core/beam_theory.py#L55)
##### `moment_load(cls, moment, position)`
Create applied moment.
**Source:** [Line 61](Python/Elasticity/core/beam_theory.py#L61)
##### `function_load(cls, load_function)`
Create load defined by function q(x).
**Source:** [Line 66](Python/Elasticity/core/beam_theory.py#L66)
##### `evaluate_load(self, x, beam_length)`
Evaluate load distribution at given positions.
**Source:** [Line 70](Python/Elasticity/core/beam_theory.py#L70)
**Class Source:** [Line 35](Python/Elasticity/core/beam_theory.py#L35)
### `BoundaryCondition`
Define boundary conditions for beam analysis.
#### Methods
##### `__init__(self, condition_type, position, value)`
Initialize boundary condition.
Parameters:
condition_type: 'displacement', 'slope', 'moment', 'shear'
position: 'left' or 'right'
value: Prescribed value
**Source:** [Line 105](Python/Elasticity/core/beam_theory.py#L105)
##### `simply_supported(cls)`
Create simply supported boundary conditions.
**Source:** [Line 119](Python/Elasticity/core/beam_theory.py#L119)
##### `clamped(cls)`
Create clamped (fixed) boundary conditions.
**Source:** [Line 129](Python/Elasticity/core/beam_theory.py#L129)
##### `cantilever(cls)`
Create cantilever boundary conditions.
**Source:** [Line 139](Python/Elasticity/core/beam_theory.py#L139)
**Class Source:** [Line 102](Python/Elasticity/core/beam_theory.py#L102)
### `EulerBernoulliBeam`
Euler-Bernoulli beam theory implementation.
Features:
- Linear and nonlinear analysis
- Multiple boundary conditions
- Static and dynamic analysis
- Buckling analysis
Examples:
>>> beam = EulerBernoulliBeam(properties, boundary_conditions)
>>> deflection = beam.static_analysis([point_load, distributed_load])
>>> frequencies = beam.natural_frequencies(n_modes=5)
#### Methods
##### `__init__(self, properties, boundary_conditions)`
Initialize Euler-Bernoulli beam.
**Source:** [Line 165](Python/Elasticity/core/beam_theory.py#L165)
##### `static_analysis(self, loads, n_elements)`
Perform static analysis using finite element method.
Parameters:
loads: List of load cases
n_elements: Number of finite elements
Returns:
Dictionary with results
**Source:** [Line 176](Python/Elasticity/core/beam_theory.py#L176)
##### `_assemble_stiffness_matrix(self, n_elements)`
Assemble global stiffness matrix.
**Source:** [Line 221](Python/Elasticity/core/beam_theory.py#L221)
##### `_assemble_load_vector(self, loads, x)`
Assemble global load vector.
**Source:** [Line 249](Python/Elasticity/core/beam_theory.py#L249)
##### `_apply_boundary_conditions(self, K, F, x)`
Apply boundary conditions using penalty method.
**Source:** [Line 280](Python/Elasticity/core/beam_theory.py#L280)
##### `_calculate_moment(self, u, x)`
Calculate bending moment from displacement field.
**Source:** [Line 307](Python/Elasticity/core/beam_theory.py#L307)
##### `_calculate_shear(self, u, x)`
Calculate shear force from displacement field.
**Source:** [Line 314](Python/Elasticity/core/beam_theory.py#L314)
##### `natural_frequencies(self, n_modes)`
Calculate natural frequencies using finite element method.
Parameters:
n_modes: Number of modes to calculate
Returns:
Natural frequencies (Hz)
**Source:** [Line 321](Python/Elasticity/core/beam_theory.py#L321)
##### `_assemble_mass_matrix(self, n_elements)`
Assemble global mass matrix.
**Source:** [Line 350](Python/Elasticity/core/beam_theory.py#L350)
##### `buckling_analysis(self, axial_load)`
Perform linear buckling analysis.
Parameters:
axial_load: Applied axial load (N)
Returns:
Critical buckling load factor
**Source:** [Line 376](Python/Elasticity/core/beam_theory.py#L376)
##### `_assemble_geometric_stiffness_matrix(self, n_elements, axial_load)`
Assemble geometric stiffness matrix for buckling analysis.
**Source:** [Line 407](Python/Elasticity/core/beam_theory.py#L407)
**Class Source:** [Line 149](Python/Elasticity/core/beam_theory.py#L149)
### `TimoshenkoBeam`
Timoshenko beam theory including shear deformation effects.
Features:
- Shear deformation effects
- Thick beam analysis
- Improved accuracy for short beams
- Dynamic analysis with rotatory inertia
#### Methods
##### `__init__(self, properties, boundary_conditions)`
Initialize Timoshenko beam.
**Source:** [Line 447](Python/Elasticity/core/beam_theory.py#L447)
##### `static_analysis(self, loads, n_elements)`
Perform static analysis using Timoshenko beam theory.
**Source:** [Line 461](Python/Elasticity/core/beam_theory.py#L461)
##### `_assemble_timoshenko_stiffness_matrix(self, n_elements)`
Assemble Timoshenko beam stiffness matrix.
**Source:** [Line 497](Python/Elasticity/core/beam_theory.py#L497)
##### `_calculate_shear_strain(self, u, x)`
Calculate shear strain γ = dw/dx - φ.
**Source:** [Line 538](Python/Elasticity/core/beam_theory.py#L538)
##### `_calculate_moment_timoshenko(self, u, x)`
Calculate bending moment M = EI dφ/dx.
**Source:** [Line 548](Python/Elasticity/core/beam_theory.py#L548)
##### `_calculate_shear_force_timoshenko(self, u, x)`
Calculate shear force V = kGA γ.
**Source:** [Line 554](Python/Elasticity/core/beam_theory.py#L554)
##### `natural_frequencies_timoshenko(self, n_modes)`
Calculate natural frequencies including rotatory inertia effects.
**Source:** [Line 559](Python/Elasticity/core/beam_theory.py#L559)
##### `_assemble_timoshenko_mass_matrix(self, n_elements)`
Assemble Timoshenko beam mass matrix with rotatory inertia.
**Source:** [Line 580](Python/Elasticity/core/beam_theory.py#L580)
##### `_assemble_load_vector(self, loads, x)`
Reuse load vector assembly from Euler-Bernoulli.
**Source:** [Line 609](Python/Elasticity/core/beam_theory.py#L609)
##### `_apply_boundary_conditions(self, K, F, x)`
Apply boundary conditions using penalty method.
**Source:** [Line 631](Python/Elasticity/core/beam_theory.py#L631)
**Class Source:** [Line 436](Python/Elasticity/core/beam_theory.py#L436)