# coupling
**Module:** `Python/Multiphysics/coupling.py`
## Overview
General Coupling Framework for Multiphysics Problems.
This module provides the core infrastructure for coupling different
physics domains, including coupling schemes, interface management,
and field transfer mechanisms.
Classes:
CoupledSystem: Base class for coupled multiphysics systems
CouplingInterface: Interface between coupled domains
CouplingScheme: Enum for coupling strategies
FieldTransfer: Field interpolation and projection
Functions:
create_coupling_interface: Create interface between domains
monolithic_coupling: Monolithic coupling approach
partitioned_coupling: Partitioned coupling approach
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `create_coupling_interface(source_domain, target_domain, source_nodes, target_nodes, name)`
Create coupling interface between domains.
Args:
source_domain: Source domain name
target_domain: Target domain name
source_nodes: Source interface nodes
target_nodes: Target interface nodes
name: Interface name
Returns:
Configured coupling interface
**Source:** [Line 487](Python/Multiphysics/coupling.py#L487)
### `monolithic_coupling(physics_models, coupling_terms, dt)`
Monolithic coupling approach.
Args:
physics_models: Dictionary of physics models
coupling_terms: Coupling term functions
dt: Time step
Returns:
Updated states for all physics
**Source:** [Line 513](Python/Multiphysics/coupling.py#L513)
### `partitioned_coupling(physics_models, interfaces, dt, scheme, max_iterations, tolerance)`
Partitioned coupling approach.
Args:
physics_models: Dictionary of physics models
interfaces: List of coupling interfaces
dt: Time step
scheme: Coupling scheme (explicit/implicit)
max_iterations: Maximum iterations for implicit
tolerance: Convergence tolerance
Returns:
Updated states for all physics
**Source:** [Line 532](Python/Multiphysics/coupling.py#L532)
## Classes
### `CouplingScheme`
Enumeration of coupling schemes.
**Class Source:** [Line 32](Python/Multiphysics/coupling.py#L32)
### `CouplingData`
Data structure for coupling information.
**Class Source:** [Line 42](Python/Multiphysics/coupling.py#L42)
### `CouplingInterface`
Interface between coupled physics domains.
Manages data exchange, interpolation, and conservation
properties at the interface between different physics.
#### Methods
##### `__init__(self, name, source_domain, target_domain, interface_type)`
Initialize coupling interface.
Args:
name: Interface identifier
source_domain: Source physics domain
target_domain: Target physics domain
interface_type: Type of interface (surface, volume, point)
**Source:** [Line 60](Python/Multiphysics/coupling.py#L60)
##### `set_interface_geometry(self, source_nodes, target_nodes)`
Set interface node locations.
Args:
source_nodes: Node coordinates in source domain
target_nodes: Node coordinates in target domain
**Source:** [Line 86](Python/Multiphysics/coupling.py#L86)
##### `_build_mapping_matrix(self)`
Build interpolation/projection matrix between domains.
**Source:** [Line 101](Python/Multiphysics/coupling.py#L101)
##### `transfer_field(self, field_data, direction, conserve_integral)`
Transfer field data across interface.
Args:
field_data: Field values to transfer
direction: Transfer direction ('forward' or 'backward')
conserve_integral: Whether to conserve integral quantities
Returns:
Transferred field values
**Source:** [Line 125](Python/Multiphysics/coupling.py#L125)
##### `check_conservation(self, source_field, target_field, quantity)`
Check conservation of quantities across interface.
Args:
source_field: Source domain field
target_field: Target domain field
quantity: Quantity to check (mass, momentum, energy)
Returns:
Relative conservation error
**Source:** [Line 167](Python/Multiphysics/coupling.py#L167)
**Class Source:** [Line 53](Python/Multiphysics/coupling.py#L53)
### `FieldTransfer`
Advanced field transfer operations.
Provides various interpolation and projection methods
for transferring fields between non-matching meshes.
#### Methods
##### `__init__(self, method)`
Initialize field transfer.
Args:
method: Transfer method (linear, rbf, conservative)
**Source:** [Line 207](Python/Multiphysics/coupling.py#L207)
##### `setup_interpolation(self, source_points, target_points)`
Setup interpolation between point sets.
Args:
source_points: Source mesh points
target_points: Target mesh points
**Source:** [Line 216](Python/Multiphysics/coupling.py#L216)
##### `_setup_conservative_transfer(self)`
Setup conservative field transfer.
**Source:** [Line 242](Python/Multiphysics/coupling.py#L242)
##### `transfer(self, field_values)`
Transfer field values.
Args:
field_values: Values at source points
Returns:
Interpolated values at target points
**Source:** [Line 248](Python/Multiphysics/coupling.py#L248)
**Class Source:** [Line 200](Python/Multiphysics/coupling.py#L200)
### `CoupledSystem`
Abstract base class for coupled multiphysics systems.
Provides framework for implementing various multiphysics
coupling strategies and solution algorithms.
#### Methods
##### `__init__(self, name, physics_domains, coupling_scheme)`
Initialize coupled system.
Args:
name: System identifier
physics_domains: List of physics domain names
coupling_scheme: Coupling strategy to use
**Source:** [Line 289](Python/Multiphysics/coupling.py#L289)
##### `setup_domains(self)`
Setup individual physics domains.
**Source:** [Line 320](Python/Multiphysics/coupling.py#L320)
##### `setup_coupling(self)`
Setup coupling between domains.
**Source:** [Line 325](Python/Multiphysics/coupling.py#L325)
##### `add_domain_solver(self, domain, solver)`
Add solver for physics domain.
Args:
domain: Domain name
solver: Domain solver object
**Source:** [Line 329](Python/Multiphysics/coupling.py#L329)
##### `add_interface(self, interface)`
Add coupling interface.
Args:
interface: Coupling interface object
**Source:** [Line 341](Python/Multiphysics/coupling.py#L341)
##### `solve(self, time_span, dt, initial_conditions)`
Solve coupled system.
Args:
time_span: Time interval (t0, tf)
dt: Time step size
initial_conditions: Initial conditions for each domain
Returns:
Solution dictionary
**Source:** [Line 349](Python/Multiphysics/coupling.py#L349)
##### `_solve_monolithic(self, dt)`
Monolithic coupling solution.
**Source:** [Line 398](Python/Multiphysics/coupling.py#L398)
##### `_solve_partitioned_explicit(self, dt)`
Explicit partitioned coupling.
**Source:** [Line 404](Python/Multiphysics/coupling.py#L404)
##### `_solve_partitioned_implicit(self, dt)`
Implicit partitioned coupling with fixed-point iteration.
**Source:** [Line 417](Python/Multiphysics/coupling.py#L417)
##### `_get_coupling_data(self, target_domain)`
Get coupling data for target domain from other domains.
**Source:** [Line 467](Python/Multiphysics/coupling.py#L467)
**Class Source:** [Line 282](Python/Multiphysics/coupling.py#L282)