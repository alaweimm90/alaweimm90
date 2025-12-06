# boundary_conditions
**Module:** `Python/ODE_PDE/boundary_conditions.py`
## Overview
Boundary Conditions for PDE Solvers.
This module provides comprehensive boundary condition handling for PDEs
including Dirichlet, Neumann, Robin, and periodic boundary conditions.
Classes:
BoundaryCondition: Base class for boundary conditions
DirichletBC: Essential boundary conditions (prescribed values)
NeumannBC: Natural boundary conditions (prescribed derivatives)
RobinBC: Mixed boundary conditions (linear combination)
PeriodicBC: Periodic boundary conditions
BoundaryConditions: Container for multiple boundary conditions
Functions:
apply_boundary_conditions: Apply boundary conditions to system
create_boundary_condition: Factory function for BC creation
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `apply_boundary_conditions(matrix, rhs, boundary_conditions, mesh, time)`
Apply boundary conditions from dictionary specification.
Args:
matrix: System matrix
rhs: Right-hand side vector
boundary_conditions: BC specification dictionary
mesh: Mesh information
time: Current time
Returns:
Modified matrix and RHS
**Source:** [Line 454](Python/ODE_PDE/boundary_conditions.py#L454)
### `create_boundary_condition(bc_type, location, value)`
Factory function for creating boundary conditions.
Args:
bc_type: Type of BC ('dirichlet', 'neumann', 'robin', 'periodic')
location: Boundary location
value: Boundary value
**kwargs: Additional parameters
Returns:
BoundaryCondition object
**Source:** [Line 496](Python/ODE_PDE/boundary_conditions.py#L496)
### `validate_boundary_conditions(boundary_conditions, domain)`
Validate boundary condition specification.
Args:
boundary_conditions: BC specification
domain: Domain specification
Returns:
True if valid
**Source:** [Line 529](Python/ODE_PDE/boundary_conditions.py#L529)
### `homogeneous_dirichlet(nodes)`
Create homogeneous Dirichlet BC (u = 0).
**Source:** [Line 555](Python/ODE_PDE/boundary_conditions.py#L555)
### `homogeneous_neumann(nodes)`
Create homogeneous Neumann BC (∂u/∂n = 0).
**Source:** [Line 561](Python/ODE_PDE/boundary_conditions.py#L561)
### `time_dependent_dirichlet(nodes, time_function)`
Create time-dependent Dirichlet BC.
**Source:** [Line 567](Python/ODE_PDE/boundary_conditions.py#L567)
## Classes
### `BCData`
Data structure for boundary condition specification.
**Class Source:** [Line 31](Python/ODE_PDE/boundary_conditions.py#L31)
### `BoundaryCondition`
Abstract base class for boundary conditions.
#### Methods
##### `__init__(self, bc_data)`
Initialize boundary condition.
Args:
bc_data: Boundary condition data
**Source:** [Line 43](Python/ODE_PDE/boundary_conditions.py#L43)
##### `apply(self, matrix, rhs, mesh, time)`
Apply boundary condition to system.
Args:
matrix: System matrix
rhs: Right-hand side vector
mesh: Mesh information
time: Current time (for time-dependent BC)
Returns:
Modified matrix and RHS
**Source:** [Line 55](Python/ODE_PDE/boundary_conditions.py#L55)
##### `evaluate_value(self, mesh, time)`
Evaluate boundary condition value at given time.
**Source:** [Line 70](Python/ODE_PDE/boundary_conditions.py#L70)
**Class Source:** [Line 40](Python/ODE_PDE/boundary_conditions.py#L40)
### `DirichletBC`
Dirichlet (essential) boundary condition: u = g.
Prescribes the value of the solution at boundary points.
#### Methods
##### `apply(self, matrix, rhs, mesh, time)`
Apply Dirichlet boundary condition.
**Source:** [Line 91](Python/ODE_PDE/boundary_conditions.py#L91)
##### `_get_boundary_nodes(self, region, mesh)`
Get boundary nodes for named region.
**Source:** [Line 129](Python/ODE_PDE/boundary_conditions.py#L129)
**Class Source:** [Line 85](Python/ODE_PDE/boundary_conditions.py#L85)
### `NeumannBC`
Neumann (natural) boundary condition: ∂u/∂n = g.
Prescribes the normal derivative of the solution at boundary points.
#### Methods
##### `apply(self, matrix, rhs, mesh, time)`
Apply Neumann boundary condition.
**Source:** [Line 157](Python/ODE_PDE/boundary_conditions.py#L157)
##### `_get_boundary_nodes(self, region, mesh)`
Get boundary nodes for named region.
**Source:** [Line 196](Python/ODE_PDE/boundary_conditions.py#L196)
**Class Source:** [Line 151](Python/ODE_PDE/boundary_conditions.py#L151)
### `RobinBC`
Robin (mixed) boundary condition: α*u + β*∂u/∂n = g.
Linear combination of Dirichlet and Neumann conditions.
#### Methods
##### `__init__(self, bc_data, alpha, beta)`
Initialize Robin boundary condition.
Args:
bc_data: Boundary condition data
alpha: Coefficient for u term
beta: Coefficient for ∂u/∂n term
**Source:** [Line 225](Python/ODE_PDE/boundary_conditions.py#L225)
##### `apply(self, matrix, rhs, mesh, time)`
Apply Robin boundary condition.
**Source:** [Line 237](Python/ODE_PDE/boundary_conditions.py#L237)
##### `_get_boundary_nodes(self, region, mesh)`
Get boundary nodes for named region.
**Source:** [Line 291](Python/ODE_PDE/boundary_conditions.py#L291)
**Class Source:** [Line 219](Python/ODE_PDE/boundary_conditions.py#L219)
### `PeriodicBC`
Periodic boundary condition: u(left) = u(right), ∂u/∂n(left) = ∂u/∂n(right).
Enforces periodicity across domain boundaries.
#### Methods
##### `__init__(self, bc_data, paired_location)`
Initialize periodic boundary condition.
Args:
bc_data: Boundary condition data
paired_location: Location of paired boundary
**Source:** [Line 314](Python/ODE_PDE/boundary_conditions.py#L314)
##### `apply(self, matrix, rhs, mesh, time)`
Apply periodic boundary condition.
**Source:** [Line 324](Python/ODE_PDE/boundary_conditions.py#L324)
##### `_get_boundary_nodes(self, region, mesh)`
Get boundary nodes for named region.
**Source:** [Line 361](Python/ODE_PDE/boundary_conditions.py#L361)
**Class Source:** [Line 308](Python/ODE_PDE/boundary_conditions.py#L308)
### `BoundaryConditions`
Container for multiple boundary conditions.
#### Methods
##### `__init__(self)`
Initialize boundary conditions container.
**Source:** [Line 380](Python/ODE_PDE/boundary_conditions.py#L380)
##### `add_bc(self, bc)`
Add boundary condition.
**Source:** [Line 384](Python/ODE_PDE/boundary_conditions.py#L384)
##### `add_dirichlet(self, location, value)`
Add Dirichlet boundary condition.
**Source:** [Line 388](Python/ODE_PDE/boundary_conditions.py#L388)
##### `add_neumann(self, location, flux)`
Add Neumann boundary condition.
**Source:** [Line 394](Python/ODE_PDE/boundary_conditions.py#L394)
##### `add_robin(self, location, value, alpha, beta)`
Add Robin boundary condition.
**Source:** [Line 400](Python/ODE_PDE/boundary_conditions.py#L400)
##### `add_periodic(self, location1, location2)`
Add periodic boundary condition.
**Source:** [Line 407](Python/ODE_PDE/boundary_conditions.py#L407)
##### `apply_all(self, matrix, rhs, mesh, time)`
Apply all boundary conditions.
**Source:** [Line 412](Python/ODE_PDE/boundary_conditions.py#L412)
##### `get_dirichlet_nodes(self)`
Get all nodes with Dirichlet boundary conditions.
**Source:** [Line 423](Python/ODE_PDE/boundary_conditions.py#L423)
##### `is_well_posed(self, problem_type)`
Check if boundary conditions make problem well-posed.
**Source:** [Line 436](Python/ODE_PDE/boundary_conditions.py#L436)
**Class Source:** [Line 377](Python/ODE_PDE/boundary_conditions.py#L377)