# finite_element
**Module:** `Python/ODE_PDE/finite_element.py`
## Overview
Finite Element Method for PDEs.
This module provides comprehensive finite element method (FEM) implementations
for solving partial differential equations with support for various element types,
quadrature rules, and boundary conditions.
Classes:
FiniteElement: Base class for finite elements
LinearElement: Linear Lagrange elements
QuadraticElement: Quadratic Lagrange elements
FEMSolver: General FEM solver
FEMAssembler: Assembly of FEM matrices
Functions:
assemble_stiffness_matrix: Assemble global stiffness matrix
assemble_mass_matrix: Assemble global mass matrix
solve_fem_poisson: FEM solver for Poisson equation
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `create_1d_mesh(domain, n_elements, element_type)`
Create 1D finite element mesh.
Args:
domain: Domain interval (a, b)
n_elements: Number of elements
element_type: Type of elements
Returns:
Tuple of (nodes, elements)
**Source:** [Line 566](Python/ODE_PDE/finite_element.py#L566)
### `assemble_stiffness_matrix(mesh, elements, coefficient, element_type)`
Convenience function to assemble stiffness matrix.
Args:
mesh: Mesh nodes
elements: Element connectivity
coefficient: Diffusion coefficient
element_type: Type of finite element
Returns:
Global stiffness matrix
**Source:** [Line 604](Python/ODE_PDE/finite_element.py#L604)
### `assemble_mass_matrix(mesh, elements, density, element_type)`
Convenience function to assemble mass matrix.
Args:
mesh: Mesh nodes
elements: Element connectivity
density: Density coefficient
element_type: Type of finite element
Returns:
Global mass matrix
**Source:** [Line 622](Python/ODE_PDE/finite_element.py#L622)
### `solve_fem_poisson(domain, n_elements, boundary_conditions, source, diffusivity, element_type)`
Solve Poisson equation using FEM.
Args:
domain: Spatial domain
n_elements: Number of finite elements
boundary_conditions: Boundary conditions
source: Source term
diffusivity: Diffusion coefficient
element_type: Type of finite element
Returns:
FEMResult with solution
**Source:** [Line 640](Python/ODE_PDE/finite_element.py#L640)
### `compute_l2_error(u_exact, u_fem, mesh, elements, element_type)`
Compute L2 error between exact and FEM solution.
Args:
u_exact: Exact solution function
u_fem: FEM solution
mesh: Mesh nodes
elements: Element connectivity
element_type: Type of finite element
Returns:
L2 error norm
**Source:** [Line 668](Python/ODE_PDE/finite_element.py#L668)
### `fem_convergence_study(domain, u_exact, boundary_conditions, source, element_counts, element_type)`
Perform FEM convergence study.
Args:
domain: Spatial domain
u_exact: Exact solution function
boundary_conditions: Boundary conditions
source: Source term
element_counts: List of element counts to test
element_type: Type of finite element
Returns:
Dictionary with convergence results
**Source:** [Line 725](Python/ODE_PDE/finite_element.py#L725)
## Classes
### `FEMResult`
Result of FEM computation.
**Class Source:** [Line 34](Python/ODE_PDE/finite_element.py#L34)
### `Element`
Finite element definition.
**Class Source:** [Line 49](Python/ODE_PDE/finite_element.py#L49)
### `FiniteElement`
Abstract base class for finite elements.
#### Methods
##### `__init__(self, element_type, order)`
Initialize finite element.
Args:
element_type: Type of element ('linear', 'quadratic', etc.)
order: Polynomial order
**Source:** [Line 60](Python/ODE_PDE/finite_element.py#L60)
##### `shape_functions(self, xi)`
Evaluate shape functions at reference coordinates.
**Source:** [Line 72](Python/ODE_PDE/finite_element.py#L72)
##### `shape_derivatives(self, xi)`
Evaluate shape function derivatives at reference coordinates.
**Source:** [Line 77](Python/ODE_PDE/finite_element.py#L77)
##### `quadrature_rule(self, order)`
Get quadrature points and weights.
**Source:** [Line 82](Python/ODE_PDE/finite_element.py#L82)
##### `jacobian(self, xi, coordinates)`
Compute Jacobian of coordinate transformation.
Args:
xi: Reference coordinate
coordinates: Physical coordinates of element nodes
Returns:
Jacobian determinant
**Source:** [Line 86](Python/ODE_PDE/finite_element.py#L86)
##### `physical_coordinates(self, xi, coordinates)`
Map from reference to physical coordinates.
Args:
xi: Reference coordinate(s)
coordinates: Physical coordinates of element nodes
Returns:
Physical coordinate(s)
**Source:** [Line 100](Python/ODE_PDE/finite_element.py#L100)
**Class Source:** [Line 57](Python/ODE_PDE/finite_element.py#L57)
### `LinearElement`
Linear Lagrange element (1D).
#### Methods
##### `__init__(self)`
*No documentation available.*
**Source:** [Line 121](Python/ODE_PDE/finite_element.py#L121)
##### `shape_functions(self, xi)`
Linear shape functions N1 = (1-xi)/2, N2 = (1+xi)/2.
**Source:** [Line 124](Python/ODE_PDE/finite_element.py#L124)
##### `shape_derivatives(self, xi)`
Linear shape function derivatives dN/dxi.
**Source:** [Line 134](Python/ODE_PDE/finite_element.py#L134)
##### `quadrature_rule(self, order)`
Gauss-Legendre quadrature for interval [-1, 1].
**Source:** [Line 144](Python/ODE_PDE/finite_element.py#L144)
**Class Source:** [Line 118](Python/ODE_PDE/finite_element.py#L118)
### `QuadraticElement`
Quadratic Lagrange element (1D).
#### Methods
##### `__init__(self)`
*No documentation available.*
**Source:** [Line 168](Python/ODE_PDE/finite_element.py#L168)
##### `shape_functions(self, xi)`
Quadratic shape functions.
**Source:** [Line 171](Python/ODE_PDE/finite_element.py#L171)
##### `shape_derivatives(self, xi)`
Quadratic shape function derivatives.
**Source:** [Line 185](Python/ODE_PDE/finite_element.py#L185)
##### `quadrature_rule(self, order)`
Gauss-Legendre quadrature for quadratic elements.
**Source:** [Line 199](Python/ODE_PDE/finite_element.py#L199)
**Class Source:** [Line 165](Python/ODE_PDE/finite_element.py#L165)
### `FEMAssembler`
Finite element matrix assembler.
#### Methods
##### `__init__(self, mesh, elements, element_type)`
Initialize FEM assembler.
Args:
mesh: Mesh node coordinates
elements: Element connectivity
element_type: Type of finite element
**Source:** [Line 218](Python/ODE_PDE/finite_element.py#L218)
##### `assemble_stiffness_matrix(self, coefficient)`
Assemble global stiffness matrix.
Args:
coefficient: Diffusion coefficient (scalar or function)
Returns:
Global stiffness matrix
**Source:** [Line 240](Python/ODE_PDE/finite_element.py#L240)
##### `assemble_mass_matrix(self, density)`
Assemble global mass matrix.
Args:
density: Density coefficient (scalar or function)
Returns:
Global mass matrix
**Source:** [Line 294](Python/ODE_PDE/finite_element.py#L294)
##### `assemble_load_vector(self, source)`
Assemble global load vector.
Args:
source: Source term (scalar or function)
Returns:
Global load vector
**Source:** [Line 345](Python/ODE_PDE/finite_element.py#L345)
**Class Source:** [Line 215](Python/ODE_PDE/finite_element.py#L215)
### `FEMSolver`
General finite element solver.
#### Methods
##### `__init__(self, mesh, elements, boundary_conditions, element_type)`
Initialize FEM solver.
Args:
mesh: Mesh node coordinates
elements: Element connectivity
boundary_conditions: Boundary conditions
element_type: Type of finite element
**Source:** [Line 399](Python/ODE_PDE/finite_element.py#L399)
##### `apply_boundary_conditions(self, K, f)`
Apply boundary conditions to system.
Args:
K: Stiffness matrix
f: Load vector
Returns:
Modified matrix and vector
**Source:** [Line 418](Python/ODE_PDE/finite_element.py#L418)
##### `solve_poisson(self, source, diffusivity)`
Solve Poisson equation -∇·(k∇u) = f.
Args:
source: Source term
diffusivity: Diffusion coefficient
Returns:
FEMResult with solution
**Source:** [Line 454](Python/ODE_PDE/finite_element.py#L454)
##### `solve_heat_equation(self, initial_condition, time_span, dt, thermal_diffusivity, source)`
Solve transient heat equation using FEM.
Args:
initial_condition: Initial condition function
time_span: Time interval
dt: Time step
thermal_diffusivity: Thermal diffusivity
source: Source term
Returns:
FEMResult with time-dependent solution
**Source:** [Line 503](Python/ODE_PDE/finite_element.py#L503)
**Class Source:** [Line 396](Python/ODE_PDE/finite_element.py#L396)