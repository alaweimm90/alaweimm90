"""
Mesh Generation for Finite Element Analysis
Comprehensive mesh generation algorithms for various geometries and element types.
Includes structured and unstructured mesh generation with adaptive refinement.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import warnings
from .finite_elements import Node, Element
@dataclass
class MeshParameters:
    """Parameters for mesh generation."""
    element_size: float = 1.0
    element_type: str = 'triangle2d'
    refinement_level: int = 0
    boundary_layers: int = 0
    boundary_layer_ratio: float = 1.2
    structured: bool = False
    quality_threshold: float = 0.5
@dataclass
class Geometry:
    """Geometry definition for meshing."""
    vertices: np.ndarray
    edges: List[Tuple[int, int]]
    regions: List[Dict] = field(default_factory=list)
    holes: List[np.ndarray] = field(default_factory=list)
    boundary_markers: Dict[int, str] = field(default_factory=dict)
class Mesh:
    """
    Finite element mesh representation.
    Features:
    - Node and element management
    - Boundary condition application
    - Mesh quality assessment
    - Visualization capabilities
    """
    def __init__(self):
        """Initialize empty mesh."""
        self.nodes: Dict[int, Node] = {}
        self.elements: Dict[int, Element] = {}
        self.dimension: int = 2
        self.boundary_nodes: Dict[str, List[int]] = {}
        self.node_counter: int = 0
        self.element_counter: int = 0
    def add_node(self, coordinates: np.ndarray, node_id: Optional[int] = None) -> int:
        """
        Add node to mesh.
        Parameters:
            coordinates: Node coordinates
            node_id: Optional node ID (auto-generated if None)
        Returns:
            Node ID
        """
        if node_id is None:
            node_id = self.node_counter
            self.node_counter += 1
        self.nodes[node_id] = Node(node_id, coordinates)
        return node_id
    def add_element(self, element_type: str, node_ids: List[int],
                   material_id: int = 0, element_id: Optional[int] = None,
                   **properties) -> int:
        """
        Add element to mesh.
        Parameters:
            element_type: Type of element
            node_ids: List of node IDs
            material_id: Material identifier
            element_id: Optional element ID (auto-generated if None)
            **properties: Additional element properties
        Returns:
            Element ID
        """
        if element_id is None:
            element_id = self.element_counter
            self.element_counter += 1
        self.elements[element_id] = Element(
            element_id, element_type, node_ids, material_id, **properties
        )
        return element_id
    def get_node_coordinates(self) -> np.ndarray:
        """Get all node coordinates as array."""
        return np.array([node.coordinates for node in self.nodes.values()])
    def get_element_connectivity(self) -> List[List[int]]:
        """Get element connectivity as list of node ID lists."""
        return [element.node_ids for element in self.elements.values()]
    def find_boundary_nodes(self, tolerance: float = 1e-10) -> Dict[str, List[int]]:
        """
        Find boundary nodes automatically.
        Parameters:
            tolerance: Tolerance for boundary detection
        Returns:
            Dictionary of boundary node lists by boundary name
        """
        coords = self.get_node_coordinates()
        boundary_nodes = {}
        if self.dimension == 2:
            # Find bounding box
            x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
            y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
            # Classify boundary nodes
            boundary_nodes['left'] = []
            boundary_nodes['right'] = []
            boundary_nodes['bottom'] = []
            boundary_nodes['top'] = []
            for node_id, node in self.nodes.items():
                x, y = node.coordinates[0], node.coordinates[1]
                if abs(x - x_min) < tolerance:
                    boundary_nodes['left'].append(node_id)
                elif abs(x - x_max) < tolerance:
                    boundary_nodes['right'].append(node_id)
                if abs(y - y_min) < tolerance:
                    boundary_nodes['bottom'].append(node_id)
                elif abs(y - y_max) < tolerance:
                    boundary_nodes['top'].append(node_id)
        self.boundary_nodes = boundary_nodes
        return boundary_nodes
    def apply_boundary_condition(self, boundary_name: str, dof: int, value: float):
        """
        Apply boundary condition to boundary nodes.
        Parameters:
            boundary_name: Name of boundary
            dof: Degree of freedom (0=x, 1=y, 2=z)
            value: Prescribed value
        """
        if boundary_name not in self.boundary_nodes:
            self.find_boundary_nodes()
        for node_id in self.boundary_nodes[boundary_name]:
            if node_id in self.nodes:
                self.nodes[node_id].boundary_conditions[dof] = value
    def mesh_quality_metrics(self) -> Dict[str, float]:
        """
        Compute mesh quality metrics.
        Returns:
            Dictionary of quality metrics
        """
        if not self.elements:
            return {}
        aspect_ratios = []
        skewnesses = []
        jacobians = []
        for element in self.elements.values():
            if element.element_type == 'triangle2d':
                quality = self._triangle_quality(element)
                aspect_ratios.append(quality['aspect_ratio'])
                skewnesses.append(quality['skewness'])
            elif element.element_type == 'quad2d':
                quality = self._quad_quality(element)
                aspect_ratios.append(quality['aspect_ratio'])
                jacobians.append(quality['min_jacobian'])
        metrics = {
            'num_nodes': len(self.nodes),
            'num_elements': len(self.elements),
            'min_aspect_ratio': min(aspect_ratios) if aspect_ratios else 0,
            'max_aspect_ratio': max(aspect_ratios) if aspect_ratios else 0,
            'avg_aspect_ratio': np.mean(aspect_ratios) if aspect_ratios else 0,
        }
        if skewnesses:
            metrics['max_skewness'] = max(skewnesses)
            metrics['avg_skewness'] = np.mean(skewnesses)
        if jacobians:
            metrics['min_jacobian'] = min(jacobians)
        return metrics
    def _triangle_quality(self, element: Element) -> Dict[str, float]:
        """Calculate quality metrics for triangular element."""
        node_coords = np.array([self.nodes[nid].coordinates for nid in element.node_ids])
        # Calculate edge lengths
        edges = [
            np.linalg.norm(node_coords[1] - node_coords[0]),
            np.linalg.norm(node_coords[2] - node_coords[1]),
            np.linalg.norm(node_coords[0] - node_coords[2])
        ]
        # Area
        area = 0.5 * abs(np.cross(node_coords[1] - node_coords[0],
                                 node_coords[2] - node_coords[0]))
        # Aspect ratio (longest edge / shortest edge)
        aspect_ratio = max(edges) / min(edges)
        # Skewness (deviation from equilateral triangle)
        optimal_area = (np.sqrt(3) / 4) * min(edges)**2
        skewness = abs(area - optimal_area) / optimal_area
        return {
            'aspect_ratio': aspect_ratio,
            'skewness': skewness,
            'area': area
        }
    def _quad_quality(self, element: Element) -> Dict[str, float]:
        """Calculate quality metrics for quadrilateral element."""
        node_coords = np.array([self.nodes[nid].coordinates for nid in element.node_ids])
        # Calculate edge lengths
        edges = []
        for i in range(4):
            j = (i + 1) % 4
            edges.append(np.linalg.norm(node_coords[j] - node_coords[i]))
        # Aspect ratio
        aspect_ratio = max(edges) / min(edges)
        # Jacobian calculation at element center
        xi_eta = np.array([0.0, 0.0])  # Element center in natural coordinates
        J = self._quad_jacobian(node_coords, xi_eta)
        min_jacobian = np.linalg.det(J)
        return {
            'aspect_ratio': aspect_ratio,
            'min_jacobian': min_jacobian
        }
    def _quad_jacobian(self, coords: np.ndarray, xi_eta: np.ndarray) -> np.ndarray:
        """Calculate Jacobian matrix for quadrilateral element."""
        xi, eta = xi_eta
        # Shape function derivatives
        dN_dxi = np.array([
            [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)],
            [-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)]
        ])
        # Jacobian matrix
        J = coords.T @ dN_dxi.T
        return J
    def plot(self, show_node_ids: bool = False, show_element_ids: bool = False,
             figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot the mesh.
        Parameters:
            show_node_ids: Whether to show node IDs
            show_element_ids: Whether to show element IDs
            figsize: Figure size
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        if self.dimension == 2:
            # Plot elements
            for element in self.elements.values():
                node_coords = np.array([self.nodes[nid].coordinates
                                      for nid in element.node_ids])
                if element.element_type in ['triangle2d', 'quad2d']:
                    # Close the polygon
                    plot_coords = np.vstack([node_coords, node_coords[0]])
                    ax.plot(plot_coords[:, 0], plot_coords[:, 1], 'b-', linewidth=1)
                    ax.fill(plot_coords[:, 0], plot_coords[:, 1], 'lightblue', alpha=0.3)
            # Plot nodes
            coords = self.get_node_coordinates()
            ax.scatter(coords[:, 0], coords[:, 1], c='red', s=20, zorder=5)
            # Show node IDs
            if show_node_ids:
                for node_id, node in self.nodes.items():
                    ax.annotate(str(node_id), node.coordinates[:2],
                              xytext=(3, 3), textcoords='offset points',
                              fontsize=8, color='red')
            # Show element IDs
            if show_element_ids:
                for element in self.elements.values():
                    node_coords = np.array([self.nodes[nid].coordinates
                                          for nid in element.node_ids])
                    centroid = np.mean(node_coords, axis=0)
                    ax.annotate(str(element.id), centroid[:2],
                              ha='center', va='center', fontsize=8, color='blue')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title('Finite Element Mesh')
        else:
            warnings.warn("3D mesh plotting not implemented yet")
        return fig
class StructuredMeshGenerator:
    """
    Structured mesh generation for regular geometries.
    Features:
    - Rectangular/box domains
    - Uniform and graded spacing
    - Boundary layer generation
    """
    def __init__(self):
        """Initialize structured mesh generator."""
        pass
    def generate_rectangle_mesh(self, width: float, height: float,
                              nx: int, ny: int, element_type: str = 'quad2d') -> Mesh:
        """
        Generate structured mesh for rectangular domain.
        Parameters:
            width: Domain width
            height: Domain height
            nx: Number of elements in x-direction
            ny: Number of elements in y-direction
            element_type: Element type ('quad2d' or 'triangle2d')
        Returns:
            Generated mesh
        """
        mesh = Mesh()
        mesh.dimension = 2
        # Generate nodes
        x = np.linspace(0, width, nx + 1)
        y = np.linspace(0, height, ny + 1)
        node_map = {}
        for j in range(ny + 1):
            for i in range(nx + 1):
                node_id = mesh.add_node(np.array([x[i], y[j]]))
                node_map[(i, j)] = node_id
        # Generate elements
        if element_type == 'quad2d':
            for j in range(ny):
                for i in range(nx):
                    # Quadrilateral element (counterclockwise ordering)
                    node_ids = [
                        node_map[(i, j)],     # bottom-left
                        node_map[(i+1, j)],   # bottom-right
                        node_map[(i+1, j+1)], # top-right
                        node_map[(i, j+1)]    # top-left
                    ]
                    mesh.add_element(element_type, node_ids)
        elif element_type == 'triangle2d':
            for j in range(ny):
                for i in range(nx):
                    # Split each quad into two triangles
                    # Triangle 1
                    node_ids1 = [
                        node_map[(i, j)],
                        node_map[(i+1, j)],
                        node_map[(i, j+1)]
                    ]
                    mesh.add_element(element_type, node_ids1)
                    # Triangle 2
                    node_ids2 = [
                        node_map[(i+1, j)],
                        node_map[(i+1, j+1)],
                        node_map[(i, j+1)]
                    ]
                    mesh.add_element(element_type, node_ids2)
        mesh.find_boundary_nodes()
        return mesh
    def generate_box_mesh(self, width: float, height: float, depth: float,
                         nx: int, ny: int, nz: int) -> Mesh:
        """
        Generate structured mesh for box domain.
        Parameters:
            width: Domain width (x-direction)
            height: Domain height (y-direction)
            depth: Domain depth (z-direction)
            nx, ny, nz: Number of elements in each direction
        Returns:
            Generated mesh
        """
        mesh = Mesh()
        mesh.dimension = 3
        # Generate nodes
        x = np.linspace(0, width, nx + 1)
        y = np.linspace(0, height, ny + 1)
        z = np.linspace(0, depth, nz + 1)
        node_map = {}
        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    node_id = mesh.add_node(np.array([x[i], y[j], z[k]]))
                    node_map[(i, j, k)] = node_id
        # Generate hexahedral elements (split into tetrahedra)
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # Get 8 corner nodes of hexahedron
                    hex_nodes = [
                        node_map[(i, j, k)],     # 0
                        node_map[(i+1, j, k)],   # 1
                        node_map[(i+1, j+1, k)], # 2
                        node_map[(i, j+1, k)],   # 3
                        node_map[(i, j, k+1)],   # 4
                        node_map[(i+1, j, k+1)], # 5
                        node_map[(i+1, j+1, k+1)], # 6
                        node_map[(i, j+1, k+1)]  # 7
                    ]
                    # Split hexahedron into 5 tetrahedra
                    tet_connectivity = [
                        [hex_nodes[0], hex_nodes[1], hex_nodes[2], hex_nodes[5]],
                        [hex_nodes[0], hex_nodes[2], hex_nodes[3], hex_nodes[7]],
                        [hex_nodes[0], hex_nodes[5], hex_nodes[2], hex_nodes[6]],
                        [hex_nodes[0], hex_nodes[5], hex_nodes[6], hex_nodes[7]],
                        [hex_nodes[0], hex_nodes[2], hex_nodes[6], hex_nodes[7]]
                    ]
                    for tet_nodes in tet_connectivity:
                        mesh.add_element('tetrahedron3d', tet_nodes)
        return mesh
    def generate_graded_mesh(self, domain: Tuple[float, float],
                           nx: int, grading_ratio: float = 1.0,
                           element_type: str = 'quad2d') -> Mesh:
        """
        Generate 1D graded mesh.
        Parameters:
            domain: (start, end) coordinates
            nx: Number of elements
            grading_ratio: Ratio of largest to smallest element
            element_type: Element type
        Returns:
            Generated mesh
        """
        start, end = domain
        length = end - start
        if abs(grading_ratio - 1.0) < 1e-10:
            # Uniform spacing
            x = np.linspace(start, end, nx + 1)
        else:
            # Geometric progression
            r = grading_ratio**(1/nx)
            x = np.zeros(nx + 1)
            x[0] = start
            # Calculate spacing
            h0 = length * (r - 1) / (r**nx - 1)
            for i in range(1, nx + 1):
                x[i] = x[i-1] + h0 * r**(i-1)
        mesh = Mesh()
        mesh.dimension = 1
        # Add nodes
        for i, xi in enumerate(x):
            mesh.add_node(np.array([xi]))
        # Add elements
        for i in range(nx):
            mesh.add_element('bar1d', [i, i+1])
        return mesh
class UnstructuredMeshGenerator:
    """
    Unstructured mesh generation using Delaunay triangulation.
    Features:
    - Arbitrary domain shapes
    - Automatic triangulation
    - Boundary preservation
    - Adaptive refinement
    """
    def __init__(self):
        """Initialize unstructured mesh generator."""
        pass
    def generate_delaunay_mesh(self, geometry: Geometry,
                             mesh_params: MeshParameters) -> Mesh:
        """
        Generate unstructured mesh using Delaunay triangulation.
        Parameters:
            geometry: Domain geometry
            mesh_params: Mesh generation parameters
        Returns:
            Generated mesh
        """
        mesh = Mesh()
        mesh.dimension = 2
        # Generate boundary points
        boundary_points = self._generate_boundary_points(geometry, mesh_params.element_size)
        # Generate interior points if needed
        interior_points = self._generate_interior_points(geometry, boundary_points,
                                                        mesh_params.element_size)
        # Combine all points
        all_points = np.vstack([boundary_points, interior_points])
        # Delaunay triangulation
        tri = Delaunay(all_points)
        # Add nodes to mesh
        for i, point in enumerate(all_points):
            mesh.add_node(point)
        # Add triangular elements
        for simplex in tri.simplices:
            # Check if triangle is inside domain
            centroid = np.mean(all_points[simplex], axis=0)
            if self._point_in_domain(centroid, geometry):
                mesh.add_element('triangle2d', simplex.tolist())
        # Apply mesh quality improvements
        if mesh_params.quality_threshold > 0:
            self._improve_mesh_quality(mesh, mesh_params.quality_threshold)
        mesh.find_boundary_nodes()
        return mesh
    def _generate_boundary_points(self, geometry: Geometry, element_size: float) -> np.ndarray:
        """Generate points along domain boundary."""
        boundary_points = []
        # Add vertices
        boundary_points.extend(geometry.vertices.tolist())
        # Add points along edges
        for edge in geometry.edges:
            start_vertex = geometry.vertices[edge[0]]
            end_vertex = geometry.vertices[edge[1]]
            edge_length = np.linalg.norm(end_vertex - start_vertex)
            num_points = max(1, int(edge_length / element_size))
            for i in range(1, num_points):
                t = i / num_points
                point = start_vertex + t * (end_vertex - start_vertex)
                boundary_points.append(point.tolist())
        return np.array(boundary_points)
    def _generate_interior_points(self, geometry: Geometry, boundary_points: np.ndarray,
                                element_size: float) -> np.ndarray:
        """Generate interior points for mesh density control."""
        # Find bounding box
        min_coords = np.min(geometry.vertices, axis=0)
        max_coords = np.max(geometry.vertices, axis=0)
        # Create regular grid of candidate points
        nx = int((max_coords[0] - min_coords[0]) / element_size) + 1
        ny = int((max_coords[1] - min_coords[1]) / element_size) + 1
        x = np.linspace(min_coords[0], max_coords[0], nx)
        y = np.linspace(min_coords[1], max_coords[1], ny)
        xx, yy = np.meshgrid(x, y)
        candidate_points = np.column_stack([xx.ravel(), yy.ravel()])
        # Filter points inside domain and away from boundary
        interior_points = []
        min_boundary_distance = element_size * 0.5
        for point in candidate_points:
            if self._point_in_domain(point, geometry):
                # Check distance to boundary
                distances = np.linalg.norm(boundary_points - point, axis=1)
                if np.min(distances) > min_boundary_distance:
                    interior_points.append(point)
        return np.array(interior_points) if interior_points else np.empty((0, 2))
    def _point_in_domain(self, point: np.ndarray, geometry: Geometry) -> bool:
        """Check if point is inside domain using ray casting algorithm."""
        x, y = point
        vertices = geometry.vertices
        inside = False
        j = len(vertices) - 1
        for i in range(len(vertices)):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        # Check if point is in any hole
        for hole in geometry.holes:
            if self._point_in_polygon(point, hole):
                inside = False
                break
        return inside
    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Check if point is inside polygon."""
        x, y = point
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    def _improve_mesh_quality(self, mesh: Mesh, quality_threshold: float):
        """Improve mesh quality through edge swapping and node smoothing."""
        # Simple Laplacian smoothing
        for iteration in range(5):  # Limited iterations
            self._laplacian_smoothing(mesh)
    def _laplacian_smoothing(self, mesh: Mesh):
        """Apply Laplacian smoothing to improve mesh quality."""
        # Build node-to-element connectivity
        node_elements = {node_id: [] for node_id in mesh.nodes.keys()}
        for element in mesh.elements.values():
            for node_id in element.node_ids:
                node_elements[node_id].append(element.id)
        # Smooth interior nodes
        for node_id, node in mesh.nodes.items():
            if not node.boundary_conditions:  # Don't move boundary nodes
                # Find neighboring nodes
                neighbors = set()
                for elem_id in node_elements[node_id]:
                    element = mesh.elements[elem_id]
                    for neighbor_id in element.node_ids:
                        if neighbor_id != node_id:
                            neighbors.add(neighbor_id)
                if neighbors:
                    # Calculate centroid of neighbors
                    neighbor_coords = np.array([mesh.nodes[nid].coordinates
                                              for nid in neighbors])
                    centroid = np.mean(neighbor_coords, axis=0)
                    # Move node towards centroid (damped)
                    damping = 0.1
                    node.coordinates = node.coordinates + damping * (centroid - node.coordinates)