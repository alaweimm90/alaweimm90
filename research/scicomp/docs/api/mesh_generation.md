# mesh_generation
**Module:** `Python/FEM/core/mesh_generation.py`
## Overview
Mesh Generation for Finite Element Analysis
Comprehensive mesh generation algorithms for various geometries and element types.
Includes structured and unstructured mesh generation with adaptive refinement.
## Classes
### `MeshParameters`
Parameters for mesh generation.
**Class Source:** [Line 19](Python/FEM/core/mesh_generation.py#L19)
### `Geometry`
Geometry definition for meshing.
**Class Source:** [Line 31](Python/FEM/core/mesh_generation.py#L31)
### `Mesh`
Finite element mesh representation.
Features:
- Node and element management
- Boundary condition application
- Mesh quality assessment
- Visualization capabilities
#### Methods
##### `__init__(self)`
Initialize empty mesh.
**Source:** [Line 51](Python/FEM/core/mesh_generation.py#L51)
##### `add_node(self, coordinates, node_id)`
Add node to mesh.
Parameters:
coordinates: Node coordinates
node_id: Optional node ID (auto-generated if None)
Returns:
Node ID
**Source:** [Line 60](Python/FEM/core/mesh_generation.py#L60)
##### `add_element(self, element_type, node_ids, material_id, element_id)`
Add element to mesh.
Parameters:
element_type: Type of element
node_ids: List of node IDs
material_id: Material identifier
element_id: Optional element ID (auto-generated if None)
**properties: Additional element properties
Returns:
Element ID
**Source:** [Line 78](Python/FEM/core/mesh_generation.py#L78)
##### `get_node_coordinates(self)`
Get all node coordinates as array.
**Source:** [Line 103](Python/FEM/core/mesh_generation.py#L103)
##### `get_element_connectivity(self)`
Get element connectivity as list of node ID lists.
**Source:** [Line 107](Python/FEM/core/mesh_generation.py#L107)
##### `find_boundary_nodes(self, tolerance)`
Find boundary nodes automatically.
Parameters:
tolerance: Tolerance for boundary detection
Returns:
Dictionary of boundary node lists by boundary name
**Source:** [Line 111](Python/FEM/core/mesh_generation.py#L111)
##### `apply_boundary_condition(self, boundary_name, dof, value)`
Apply boundary condition to boundary nodes.
Parameters:
boundary_name: Name of boundary
dof: Degree of freedom (0=x, 1=y, 2=z)
value: Prescribed value
**Source:** [Line 151](Python/FEM/core/mesh_generation.py#L151)
##### `mesh_quality_metrics(self)`
Compute mesh quality metrics.
Returns:
Dictionary of quality metrics
**Source:** [Line 167](Python/FEM/core/mesh_generation.py#L167)
##### `_triangle_quality(self, element)`
Calculate quality metrics for triangular element.
**Source:** [Line 208](Python/FEM/core/mesh_generation.py#L208)
##### `_quad_quality(self, element)`
Calculate quality metrics for quadrilateral element.
**Source:** [Line 236](Python/FEM/core/mesh_generation.py#L236)
##### `_quad_jacobian(self, coords, xi_eta)`
Calculate Jacobian matrix for quadrilateral element.
**Source:** [Line 259](Python/FEM/core/mesh_generation.py#L259)
##### `plot(self, show_node_ids, show_element_ids, figsize)`
Plot the mesh.
Parameters:
show_node_ids: Whether to show node IDs
show_element_ids: Whether to show element IDs
figsize: Figure size
Returns:
Matplotlib figure
**Source:** [Line 273](Python/FEM/core/mesh_generation.py#L273)
**Class Source:** [Line 40](Python/FEM/core/mesh_generation.py#L40)
### `StructuredMeshGenerator`
Structured mesh generation for regular geometries.
Features:
- Rectangular/box domains
- Uniform and graded spacing
- Boundary layer generation
#### Methods
##### `__init__(self)`
Initialize structured mesh generator.
**Source:** [Line 342](Python/FEM/core/mesh_generation.py#L342)
##### `generate_rectangle_mesh(self, width, height, nx, ny, element_type)`
Generate structured mesh for rectangular domain.
Parameters:
width: Domain width
height: Domain height
nx: Number of elements in x-direction
ny: Number of elements in y-direction
element_type: Element type ('quad2d' or 'triangle2d')
Returns:
Generated mesh
**Source:** [Line 346](Python/FEM/core/mesh_generation.py#L346)
##### `generate_box_mesh(self, width, height, depth, nx, ny, nz)`
Generate structured mesh for box domain.
Parameters:
width: Domain width (x-direction)
height: Domain height (y-direction)
depth: Domain depth (z-direction)
nx, ny, nz: Number of elements in each direction
Returns:
Generated mesh
**Source:** [Line 410](Python/FEM/core/mesh_generation.py#L410)
##### `generate_graded_mesh(self, domain, nx, grading_ratio, element_type)`
Generate 1D graded mesh.
Parameters:
domain: (start, end) coordinates
nx: Number of elements
grading_ratio: Ratio of largest to smallest element
element_type: Element type
Returns:
Generated mesh
**Source:** [Line 469](Python/FEM/core/mesh_generation.py#L469)
**Class Source:** [Line 332](Python/FEM/core/mesh_generation.py#L332)
### `UnstructuredMeshGenerator`
Unstructured mesh generation using Delaunay triangulation.
Features:
- Arbitrary domain shapes
- Automatic triangulation
- Boundary preservation
- Adaptive refinement
#### Methods
##### `__init__(self)`
Initialize unstructured mesh generator.
**Source:** [Line 527](Python/FEM/core/mesh_generation.py#L527)
##### `generate_delaunay_mesh(self, geometry, mesh_params)`
Generate unstructured mesh using Delaunay triangulation.
Parameters:
geometry: Domain geometry
mesh_params: Mesh generation parameters
Returns:
Generated mesh
**Source:** [Line 531](Python/FEM/core/mesh_generation.py#L531)
##### `_generate_boundary_points(self, geometry, element_size)`
Generate points along domain boundary.
**Source:** [Line 577](Python/FEM/core/mesh_generation.py#L577)
##### `_generate_interior_points(self, geometry, boundary_points, element_size)`
Generate interior points for mesh density control.
**Source:** [Line 599](Python/FEM/core/mesh_generation.py#L599)
##### `_point_in_domain(self, point, geometry)`
Check if point is inside domain using ray casting algorithm.
**Source:** [Line 629](Python/FEM/core/mesh_generation.py#L629)
##### `_point_in_polygon(self, point, polygon)`
Check if point is inside polygon.
**Source:** [Line 653](Python/FEM/core/mesh_generation.py#L653)
##### `_improve_mesh_quality(self, mesh, quality_threshold)`
Improve mesh quality through edge swapping and node smoothing.
**Source:** [Line 669](Python/FEM/core/mesh_generation.py#L669)
##### `_laplacian_smoothing(self, mesh)`
Apply Laplacian smoothing to improve mesh quality.
**Source:** [Line 675](Python/FEM/core/mesh_generation.py#L675)
**Class Source:** [Line 516](Python/FEM/core/mesh_generation.py#L516)