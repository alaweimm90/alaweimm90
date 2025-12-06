# fluid_structure
**Module:** `Python/Multiphysics/fluid_structure.py`
## Overview
Fluid-Structure Interaction (FSI) Module.
This module implements fluid-structure interaction methods including
Arbitrary Lagrangian-Eulerian (ALE) formulations, mesh motion algorithms,
and FSI benchmarks.
Classes:
FluidStructureInteraction: Main FSI solver
FluidSolver: Fluid domain solver
StructuralSolver: Structural domain solver
MeshMotion: Mesh motion algorithms
ALE: Arbitrary Lagrangian-Eulerian framework
Functions:
fsi_benchmark: Standard FSI benchmark problems
vortex_induced_vibration: VIV analysis
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `fsi_benchmark(benchmark_name)`
Standard FSI benchmark problems.
Args:
benchmark_name: Name of benchmark problem
Returns:
Benchmark configuration and reference data
**Source:** [Line 758](Python/Multiphysics/fluid_structure.py#L758)
### `vortex_induced_vibration(reduced_velocity, mass_ratio, damping_ratio)`
Analyze vortex-induced vibration.
Args:
reduced_velocity: U/(f_n*D) where U is flow velocity
mass_ratio: m/(ρ*D²) where m is mass per unit length
damping_ratio: Structural damping ratio
Returns:
VIV response characteristics
**Source:** [Line 819](Python/Multiphysics/fluid_structure.py#L819)
## Classes
### `FluidProperties`
Fluid material properties.
**Class Source:** [Line 33](Python/Multiphysics/fluid_structure.py#L33)
### `StructuralProperties`
Structural material properties.
#### Methods
##### `shear_modulus(self)`
Compute shear modulus.
**Source:** [Line 49](Python/Multiphysics/fluid_structure.py#L49)
##### `bulk_modulus(self)`
Compute bulk modulus.
**Source:** [Line 54](Python/Multiphysics/fluid_structure.py#L54)
**Class Source:** [Line 41](Python/Multiphysics/fluid_structure.py#L41)
### `FluidSolver`
Incompressible fluid flow solver.
Implements finite element solution of Navier-Stokes equations
with support for moving meshes (ALE formulation).
#### Methods
##### `__init__(self, mesh, properties, formulation)`
Initialize fluid solver.
Args:
mesh: Mesh dictionary with 'nodes', 'elements'
properties: Fluid properties
formulation: Flow formulation type
**Source:** [Line 66](Python/Multiphysics/fluid_structure.py#L66)
##### `_setup_system(self)`
Setup finite element system.
**Source:** [Line 98](Python/Multiphysics/fluid_structure.py#L98)
##### `solve(self, dt, boundary_conditions, interface_data)`
Solve fluid flow for one time step.
Args:
dt: Time step size
boundary_conditions: Boundary condition data
interface_data: FSI interface data
Returns:
Dictionary with velocity and pressure fields
**Source:** [Line 112](Python/Multiphysics/fluid_structure.py#L112)
##### `_solve_incompressible_ns(self, bc)`
Solve incompressible Navier-Stokes equations.
**Source:** [Line 145](Python/Multiphysics/fluid_structure.py#L145)
##### `_apply_interface_conditions(self, interface_data)`
Apply FSI interface conditions.
**Source:** [Line 185](Python/Multiphysics/fluid_structure.py#L185)
##### `_apply_boundary_conditions(self, bc)`
Apply boundary conditions.
**Source:** [Line 202](Python/Multiphysics/fluid_structure.py#L202)
##### `_update_mesh(self, dt)`
Update mesh position for ALE.
**Source:** [Line 220](Python/Multiphysics/fluid_structure.py#L220)
##### `compute_interface_forces(self, interface_nodes)`
Compute forces on fluid-structure interface.
Args:
interface_nodes: List of interface node indices
Returns:
Force vector at interface nodes
**Source:** [Line 225](Python/Multiphysics/fluid_structure.py#L225)
**Class Source:** [Line 59](Python/Multiphysics/fluid_structure.py#L59)
### `StructuralSolver`
Structural dynamics solver.
Implements finite element solution for structural dynamics
with support for large deformations and FSI coupling.
#### Methods
##### `__init__(self, mesh, properties, formulation)`
Initialize structural solver.
Args:
mesh: Mesh dictionary
properties: Structural properties
formulation: Formulation type (linear/nonlinear)
**Source:** [Line 256](Python/Multiphysics/fluid_structure.py#L256)
##### `_setup_system(self)`
Setup finite element system.
**Source:** [Line 287](Python/Multiphysics/fluid_structure.py#L287)
##### `solve(self, dt, external_forces, interface_data)`
Solve structural dynamics for one time step.
Args:
dt: Time step size
external_forces: External force vector
interface_data: FSI interface data
Returns:
Dictionary with displacement, velocity, acceleration
**Source:** [Line 306](Python/Multiphysics/fluid_structure.py#L306)
##### `_newmark_linear(self, dt, forces)`
Linear Newmark time integration.
**Source:** [Line 343](Python/Multiphysics/fluid_structure.py#L343)
##### `_newmark_nonlinear(self, dt, forces)`
Nonlinear Newmark time integration.
**Source:** [Line 382](Python/Multiphysics/fluid_structure.py#L382)
**Class Source:** [Line 249](Python/Multiphysics/fluid_structure.py#L249)
### `MeshMotion`
Mesh motion algorithms for ALE formulation.
Provides various mesh motion strategies including
Laplacian smoothing, elastic analogy, and RBF interpolation.
#### Methods
##### `__init__(self, method)`
Initialize mesh motion solver.
Args:
method: Mesh motion method
**Source:** [Line 396](Python/Multiphysics/fluid_structure.py#L396)
##### `set_reference_mesh(self, mesh)`
Set reference mesh configuration.
**Source:** [Line 405](Python/Multiphysics/fluid_structure.py#L405)
##### `compute_mesh_displacement(self, boundary_displacement, mesh)`
Compute mesh displacement field.
Args:
boundary_displacement: Prescribed boundary displacements
mesh: Current mesh
Returns:
Displacement field for all nodes
**Source:** [Line 412](Python/Multiphysics/fluid_structure.py#L412)
##### `update_mesh_quality(self, mesh)`
Compute mesh quality metrics.
Args:
mesh: Current mesh
Returns:
Dictionary of quality metrics
**Source:** [Line 470](Python/Multiphysics/fluid_structure.py#L470)
**Class Source:** [Line 389](Python/Multiphysics/fluid_structure.py#L389)
### `ALE`
Arbitrary Lagrangian-Eulerian framework.
Handles the ALE formulation for moving mesh problems
including convective terms and geometric conservation law.
#### Methods
##### `__init__(self)`
Initialize ALE framework.
**Source:** [Line 519](Python/Multiphysics/fluid_structure.py#L519)
##### `compute_mesh_velocity(self, mesh_displacement, dt)`
Compute mesh velocity from displacement.
Args:
mesh_displacement: Mesh displacement field
dt: Time step
Returns:
Mesh velocity field
**Source:** [Line 524](Python/Multiphysics/fluid_structure.py#L524)
##### `compute_convective_velocity(self, material_velocity, mesh_velocity)`
Compute convective velocity for ALE formulation.
Args:
material_velocity: Material point velocity
mesh_velocity: Mesh velocity
Returns:
Convective velocity (u - u_mesh)
**Source:** [Line 547](Python/Multiphysics/fluid_structure.py#L547)
##### `check_geometric_conservation(self, mesh_old, mesh_new, dt)`
Check geometric conservation law.
Args:
mesh_old: Previous mesh configuration
mesh_new: Current mesh configuration
dt: Time step
Returns:
GCL error measure
**Source:** [Line 561](Python/Multiphysics/fluid_structure.py#L561)
##### `_compute_mesh_volume(self, mesh)`
Compute total mesh volume/area.
**Source:** [Line 589](Python/Multiphysics/fluid_structure.py#L589)
**Class Source:** [Line 512](Python/Multiphysics/fluid_structure.py#L512)
### `FluidStructureInteraction`
Main FSI solver class.
Coordinates fluid and structural solvers with
appropriate coupling strategies.
#### Methods
##### `__init__(self, fluid_mesh, structure_mesh, fluid_props, structure_props, coupling_scheme)`
Initialize FSI solver.
Args:
fluid_mesh: Fluid domain mesh
structure_mesh: Structure domain mesh
fluid_props: Fluid properties
structure_props: Structural properties
coupling_scheme: FSI coupling scheme
**Source:** [Line 612](Python/Multiphysics/fluid_structure.py#L612)
##### `setup_domains(self)`
Setup fluid and structural domains.
**Source:** [Line 641](Python/Multiphysics/fluid_structure.py#L641)
##### `setup_coupling(self)`
Setup FSI coupling interface.
**Source:** [Line 659](Python/Multiphysics/fluid_structure.py#L659)
##### `solve_fsi_step(self, dt, boundary_conditions)`
Solve one FSI time step.
Args:
dt: Time step size
boundary_conditions: BC for both domains
Returns:
FSI solution data
**Source:** [Line 678](Python/Multiphysics/fluid_structure.py#L678)
**Class Source:** [Line 605](Python/Multiphysics/fluid_structure.py#L605)