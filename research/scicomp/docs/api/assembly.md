# assembly
**Module:** `Python/FEM/core/assembly.py`
## Overview
Global Assembly Procedures for Finite Element Analysis
Comprehensive assembly algorithms for finite element systems including
global matrix assembly, boundary condition application, and solution procedures.
## Classes
### `GlobalAssembly`
Global assembly system for finite element analysis.
Features:
- Global stiffness and mass matrix assembly
- Boundary condition application
- Load vector assembly
- Degree of freedom management
#### Methods
##### `__init__(self, mesh, materials)`
Initialize global assembly system.
Parameters:
mesh: Finite element mesh
materials: Dictionary of material properties by material ID
**Source:** [Line 29](Python/FEM/core/assembly.py#L29)
##### `_initialize_dof_mapping(self)`
Initialize degree of freedom mapping.
**Source:** [Line 58](Python/FEM/core/assembly.py#L58)
##### `_get_dofs_per_node(self)`
Determine number of DOFs per node based on element types.
**Source:** [Line 77](Python/FEM/core/assembly.py#L77)
##### `assemble_global_stiffness(self)`
Assemble global stiffness matrix.
Returns:
Global stiffness matrix (sparse)
**Source:** [Line 88](Python/FEM/core/assembly.py#L88)
##### `assemble_global_mass(self)`
Assemble global mass matrix.
Returns:
Global mass matrix (sparse)
**Source:** [Line 125](Python/FEM/core/assembly.py#L125)
##### `_compute_element_stiffness(self, element)`
Compute element stiffness matrix.
**Source:** [Line 162](Python/FEM/core/assembly.py#L162)
##### `_compute_element_mass(self, element)`
Compute element mass matrix.
**Source:** [Line 192](Python/FEM/core/assembly.py#L192)
##### `_get_element_dofs(self, element)`
Get global DOF indices for an element.
**Source:** [Line 219](Python/FEM/core/assembly.py#L219)
##### `apply_boundary_conditions(self, displacement_bcs)`
Apply displacement boundary conditions.
Parameters:
displacement_bcs: Dictionary of {(node_id, dof): value} boundary conditions
where dof is 0=x, 1=y, 2=z
**Source:** [Line 226](Python/FEM/core/assembly.py#L226)
##### `apply_point_loads(self, point_loads)`
Apply point loads to create load vector.
Parameters:
point_loads: Dictionary of {(node_id, dof): force} point loads
Returns:
Global load vector
**Source:** [Line 252](Python/FEM/core/assembly.py#L252)
##### `solve_static(self, tolerance)`
Solve static equilibrium: K * u = f
Parameters:
tolerance: Solver tolerance
Returns:
Global displacement vector
**Source:** [Line 280](Python/FEM/core/assembly.py#L280)
##### `compute_element_stresses(self, displacement)`
Compute stresses in all elements.
Parameters:
displacement: Global displacement vector
Returns:
Dictionary of element stresses {element_id: stress_array}
**Source:** [Line 330](Python/FEM/core/assembly.py#L330)
##### `_compute_element_stress(self, element, element_displacement)`
Compute stress for a single element.
**Source:** [Line 353](Python/FEM/core/assembly.py#L353)
##### `compute_reaction_forces(self, displacement)`
Compute reaction forces at constrained nodes.
Parameters:
displacement: Global displacement vector
Returns:
Dictionary of reaction forces {node_id: force_vector}
**Source:** [Line 411](Python/FEM/core/assembly.py#L411)
##### `get_displacement_at_node(self, displacement, node_id)`
Get displacement vector at a specific node.
Parameters:
displacement: Global displacement vector
node_id: Node ID
Returns:
Node displacement vector
**Source:** [Line 446](Python/FEM/core/assembly.py#L446)
##### `compute_system_energy(self, displacement)`
Compute system energy quantities.
Parameters:
displacement: Global displacement vector
Returns:
Dictionary of energy quantities
**Source:** [Line 463](Python/FEM/core/assembly.py#L463)
**Class Source:** [Line 18](Python/FEM/core/assembly.py#L18)