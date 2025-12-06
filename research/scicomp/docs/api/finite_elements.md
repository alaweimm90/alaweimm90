# finite_elements
**Module:** `Python/FEM/core/finite_elements.py`
## Overview
Finite Element Method Core Implementation
Comprehensive finite element framework for structural analysis including
element formulations, assembly procedures, and solution algorithms.
## Functions
### `create_element_factory()`
Create factory for element types.
**Source:** [Line 756](Python/FEM/core/finite_elements.py#L756)
## Classes
### `Node`
Finite element node definition.
#### Methods
##### `__post_init__(self)`
Validate node coordinates.
**Source:** [Line 26](Python/FEM/core/finite_elements.py#L26)
**Class Source:** [Line 19](Python/FEM/core/finite_elements.py#L19)
### `Element`
Base finite element definition.
**Class Source:** [Line 33](Python/FEM/core/finite_elements.py#L33)
### `FiniteElementBase`
Abstract base class for finite elements.
Features:
- Element stiffness matrix computation
- Element mass matrix computation
- Shape function evaluation
- Numerical integration
#### Methods
##### `__init__(self, element_id, nodes, material)`
Initialize finite element.
Parameters:
element_id: Unique element identifier
nodes: List of element nodes
material: Material properties
**Source:** [Line 54](Python/FEM/core/finite_elements.py#L54)
##### `shape_functions(self, xi)`
Evaluate shape functions at natural coordinates.
Parameters:
xi: Natural coordinates (parametric space)
Returns:
Shape function values
**Source:** [Line 73](Python/FEM/core/finite_elements.py#L73)
##### `shape_function_derivatives(self, xi)`
Evaluate shape function derivatives at natural coordinates.
Parameters:
xi: Natural coordinates
Returns:
Shape function derivatives w.r.t. natural coordinates
**Source:** [Line 86](Python/FEM/core/finite_elements.py#L86)
##### `stiffness_matrix(self)`
Compute element stiffness matrix.
Returns:
Element stiffness matrix
**Source:** [Line 99](Python/FEM/core/finite_elements.py#L99)
##### `mass_matrix(self)`
Compute element mass matrix.
Returns:
Element mass matrix
**Source:** [Line 109](Python/FEM/core/finite_elements.py#L109)
##### `jacobian_matrix(self, xi)`
Compute Jacobian matrix for coordinate transformation.
Parameters:
xi: Natural coordinates
Returns:
Jacobian matrix dx/dxi
**Source:** [Line 118](Python/FEM/core/finite_elements.py#L118)
##### `jacobian_determinant(self, xi)`
Compute Jacobian determinant.
**Source:** [Line 136](Python/FEM/core/finite_elements.py#L136)
##### `global_coordinates(self, xi)`
Map natural coordinates to global coordinates.
Parameters:
xi: Natural coordinates
Returns:
Global coordinates
**Source:** [Line 147](Python/FEM/core/finite_elements.py#L147)
##### `gauss_quadrature_points(self, order)`
Get Gauss quadrature points and weights.
Parameters:
order: Quadrature order
Returns:
Tuple of (points, weights)
**Source:** [Line 162](Python/FEM/core/finite_elements.py#L162)
**Class Source:** [Line 43](Python/FEM/core/finite_elements.py#L43)
### `LinearBar1D`
1D linear bar element for truss analysis.
Features:
- Axial deformation only
- Linear shape functions
- 2 nodes, 1 DOF per node
#### Methods
##### `__init__(self, element_id, nodes, material, cross_section_area)`
Initialize 1D bar element.
Parameters:
element_id: Element ID
nodes: Two nodes
material: Material properties
cross_section_area: Cross-sectional area
**Source:** [Line 217](Python/FEM/core/finite_elements.py#L217)
##### `shape_functions(self, xi)`
Linear shape functions for bar element.
**Source:** [Line 238](Python/FEM/core/finite_elements.py#L238)
##### `shape_function_derivatives(self, xi)`
Shape function derivatives.
**Source:** [Line 246](Python/FEM/core/finite_elements.py#L246)
##### `stiffness_matrix(self)`
Compute element stiffness matrix using analytical integration.
**Source:** [Line 254](Python/FEM/core/finite_elements.py#L254)
##### `mass_matrix(self)`
Compute element mass matrix.
**Source:** [Line 265](Python/FEM/core/finite_elements.py#L265)
**Class Source:** [Line 207](Python/FEM/core/finite_elements.py#L207)
### `LinearTriangle2D`
2D linear triangular element for plane stress/strain analysis.
Features:
- Linear shape functions
- Constant strain field
- 3 nodes, 2 DOF per node
#### Methods
##### `__init__(self, element_id, nodes, material, thickness, plane_stress)`
Initialize 2D triangular element.
Parameters:
element_id: Element ID
nodes: Three nodes
material: Material properties
thickness: Element thickness
plane_stress: True for plane stress, False for plane strain
**Source:** [Line 287](Python/FEM/core/finite_elements.py#L287)
##### `shape_functions(self, xi)`
Linear shape functions for triangular element.
Parameters:
xi: Natural coordinates [xi, eta]
**Source:** [Line 313](Python/FEM/core/finite_elements.py#L313)
##### `shape_function_derivatives(self, xi)`
Shape function derivatives.
**Source:** [Line 330](Python/FEM/core/finite_elements.py#L330)
##### `strain_displacement_matrix(self)`
Compute strain-displacement matrix B.
For constant strain triangle, B is constant throughout element.
**Source:** [Line 345](Python/FEM/core/finite_elements.py#L345)
##### `constitutive_matrix(self)`
Compute constitutive matrix D.
**Source:** [Line 377](Python/FEM/core/finite_elements.py#L377)
##### `stiffness_matrix(self)`
Compute element stiffness matrix.
**Source:** [Line 401](Python/FEM/core/finite_elements.py#L401)
##### `mass_matrix(self)`
Compute element mass matrix.
**Source:** [Line 411](Python/FEM/core/finite_elements.py#L411)
**Class Source:** [Line 277](Python/FEM/core/finite_elements.py#L277)
### `LinearQuadrilateral2D`
2D linear quadrilateral element for plane stress/strain analysis.
Features:
- Bilinear shape functions
- Isoparametric formulation
- 4 nodes, 2 DOF per node
#### Methods
##### `__init__(self, element_id, nodes, material, thickness, plane_stress)`
Initialize 2D quadrilateral element.
Parameters:
element_id: Element ID
nodes: Four nodes (in counterclockwise order)
material: Material properties
thickness: Element thickness
plane_stress: True for plane stress, False for plane strain
**Source:** [Line 437](Python/FEM/core/finite_elements.py#L437)
##### `shape_functions(self, xi)`
Bilinear shape functions for quadrilateral element.
Parameters:
xi: Natural coordinates [xi, eta] in range [-1, 1]
**Source:** [Line 456](Python/FEM/core/finite_elements.py#L456)
##### `shape_function_derivatives(self, xi)`
Shape function derivatives w.r.t. natural coordinates.
**Source:** [Line 477](Python/FEM/core/finite_elements.py#L477)
##### `strain_displacement_matrix(self, xi)`
Compute strain-displacement matrix B at given natural coordinates.
Parameters:
xi: Natural coordinates [xi, eta]
**Source:** [Line 499](Python/FEM/core/finite_elements.py#L499)
##### `constitutive_matrix(self)`
Compute constitutive matrix D.
**Source:** [Line 532](Python/FEM/core/finite_elements.py#L532)
##### `stiffness_matrix(self)`
Compute element stiffness matrix using numerical integration.
**Source:** [Line 556](Python/FEM/core/finite_elements.py#L556)
##### `mass_matrix(self)`
Compute element mass matrix using numerical integration.
**Source:** [Line 577](Python/FEM/core/finite_elements.py#L577)
**Class Source:** [Line 427](Python/FEM/core/finite_elements.py#L427)
### `LinearTetrahedron3D`
3D linear tetrahedral element for solid mechanics.
Features:
- Linear shape functions
- Constant strain field
- 4 nodes, 3 DOF per node
#### Methods
##### `__init__(self, element_id, nodes, material)`
Initialize 3D tetrahedral element.
Parameters:
element_id: Element ID
nodes: Four nodes
material: Material properties
**Source:** [Line 615](Python/FEM/core/finite_elements.py#L615)
##### `shape_functions(self, xi)`
Linear shape functions for tetrahedral element.
Parameters:
xi: Natural coordinates [xi, eta, zeta]
**Source:** [Line 636](Python/FEM/core/finite_elements.py#L636)
##### `shape_function_derivatives(self, xi)`
Shape function derivatives.
**Source:** [Line 654](Python/FEM/core/finite_elements.py#L654)
##### `strain_displacement_matrix(self)`
Compute strain-displacement matrix B.
For constant strain tetrahedron, B is constant throughout element.
**Source:** [Line 667](Python/FEM/core/finite_elements.py#L667)
##### `constitutive_matrix(self)`
Compute 3D constitutive matrix D.
**Source:** [Line 710](Python/FEM/core/finite_elements.py#L710)
##### `stiffness_matrix(self)`
Compute element stiffness matrix.
**Source:** [Line 730](Python/FEM/core/finite_elements.py#L730)
##### `mass_matrix(self)`
Compute element mass matrix.
**Source:** [Line 740](Python/FEM/core/finite_elements.py#L740)
**Class Source:** [Line 605](Python/FEM/core/finite_elements.py#L605)