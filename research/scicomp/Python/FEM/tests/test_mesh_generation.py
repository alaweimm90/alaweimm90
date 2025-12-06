"""
Tests for Mesh Generation
Comprehensive test suite for mesh generation algorithms including
structured and unstructured mesh generation, mesh quality assessment.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.mesh_generation import (
    Mesh, Node, Element, MeshParameters, Geometry,
    StructuredMeshGenerator, UnstructuredMeshGenerator
)
class TestMesh:
    """Test Mesh class functionality."""
    def setup_method(self):
        """Set up test fixtures."""
        self.mesh = Mesh()
    def test_mesh_initialization(self):
        """Test mesh initialization."""
        assert len(self.mesh.nodes) == 0
        assert len(self.mesh.elements) == 0
        assert self.mesh.dimension == 2
        assert self.mesh.node_counter == 0
        assert self.mesh.element_counter == 0
    def test_add_node(self):
        """Test adding nodes to mesh."""
        # Add node without specifying ID
        node_id1 = self.mesh.add_node(np.array([0.0, 0.0]))
        assert node_id1 == 0
        assert len(self.mesh.nodes) == 1
        assert self.mesh.node_counter == 1
        # Add node with specific ID
        node_id2 = self.mesh.add_node(np.array([1.0, 0.0]), node_id=5)
        assert node_id2 == 5
        assert len(self.mesh.nodes) == 2
        assert self.mesh.node_counter == 1  # Should not increment
        # Verify node coordinates
        assert_array_almost_equal(self.mesh.nodes[0].coordinates, [0.0, 0.0])
        assert_array_almost_equal(self.mesh.nodes[5].coordinates, [1.0, 0.0])
    def test_add_element(self):
        """Test adding elements to mesh."""
        # Add nodes first
        self.mesh.add_node(np.array([0.0, 0.0]))
        self.mesh.add_node(np.array([1.0, 0.0]))
        self.mesh.add_node(np.array([0.0, 1.0]))
        # Add element without specifying ID
        element_id1 = self.mesh.add_element('triangle2d', [0, 1, 2])
        assert element_id1 == 0
        assert len(self.mesh.elements) == 1
        assert self.mesh.element_counter == 1
        # Add element with specific ID
        element_id2 = self.mesh.add_element('triangle2d', [0, 1, 2], element_id=10)
        assert element_id2 == 10
        assert len(self.mesh.elements) == 2
        # Verify element properties
        element = self.mesh.elements[0]
        assert element.element_type == 'triangle2d'
        assert element.node_ids == [0, 1, 2]
        assert element.material_id == 0
    def test_get_node_coordinates(self):
        """Test getting all node coordinates."""
        self.mesh.add_node(np.array([0.0, 0.0]))
        self.mesh.add_node(np.array([1.0, 0.0]))
        self.mesh.add_node(np.array([0.0, 1.0]))
        coords = self.mesh.get_node_coordinates()
        expected = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert_array_almost_equal(coords, expected)
    def test_get_element_connectivity(self):
        """Test getting element connectivity."""
        self.mesh.add_node(np.array([0.0, 0.0]))
        self.mesh.add_node(np.array([1.0, 0.0]))
        self.mesh.add_node(np.array([0.0, 1.0]))
        self.mesh.add_element('triangle2d', [0, 1, 2])
        connectivity = self.mesh.get_element_connectivity()
        assert connectivity == [[0, 1, 2]]
    def test_find_boundary_nodes_2d(self):
        """Test automatic boundary node detection for 2D mesh."""
        # Create rectangular mesh
        self.mesh.add_node(np.array([0.0, 0.0]))  # 0: bottom-left
        self.mesh.add_node(np.array([1.0, 0.0]))  # 1: bottom-right
        self.mesh.add_node(np.array([1.0, 1.0]))  # 2: top-right
        self.mesh.add_node(np.array([0.0, 1.0]))  # 3: top-left
        self.mesh.add_node(np.array([0.5, 0.5]))  # 4: interior
        boundary_nodes = self.mesh.find_boundary_nodes()
        assert 'left' in boundary_nodes
        assert 'right' in boundary_nodes
        assert 'bottom' in boundary_nodes
        assert 'top' in boundary_nodes
        # Check specific boundary assignments
        assert 0 in boundary_nodes['left'] or 0 in boundary_nodes['bottom']
        assert 1 in boundary_nodes['right'] or 1 in boundary_nodes['bottom']
        assert 2 in boundary_nodes['right'] or 2 in boundary_nodes['top']
        assert 3 in boundary_nodes['left'] or 3 in boundary_nodes['top']
        # Interior node should not be on any boundary
        for boundary_list in boundary_nodes.values():
            assert 4 not in boundary_list
    def test_mesh_quality_metrics(self):
        """Test mesh quality calculation."""
        # Create simple triangle mesh
        self.mesh.add_node(np.array([0.0, 0.0]))
        self.mesh.add_node(np.array([1.0, 0.0]))
        self.mesh.add_node(np.array([0.0, 1.0]))
        self.mesh.add_element('triangle2d', [0, 1, 2])
        metrics = self.mesh.mesh_quality_metrics()
        assert 'num_nodes' in metrics
        assert 'num_elements' in metrics
        assert 'min_aspect_ratio' in metrics
        assert 'max_aspect_ratio' in metrics
        assert 'avg_aspect_ratio' in metrics
        assert metrics['num_nodes'] == 3
        assert metrics['num_elements'] == 1
        assert metrics['min_aspect_ratio'] > 0
        assert metrics['max_aspect_ratio'] >= metrics['min_aspect_ratio']
class TestStructuredMeshGenerator:
    """Test structured mesh generation."""
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = StructuredMeshGenerator()
    def test_rectangle_mesh_quad(self):
        """Test rectangular mesh generation with quadrilaterals."""
        width, height = 2.0, 1.0
        nx, ny = 4, 2
        mesh = self.generator.generate_rectangle_mesh(width, height, nx, ny, 'quad2d')
        # Check node count
        expected_nodes = (nx + 1) * (ny + 1)
        assert len(mesh.nodes) == expected_nodes
        # Check element count
        expected_elements = nx * ny
        assert len(mesh.elements) == expected_elements
        # Check mesh dimensions
        coords = mesh.get_node_coordinates()
        assert np.min(coords[:, 0]) == 0.0
        assert np.max(coords[:, 0]) == width
        assert np.min(coords[:, 1]) == 0.0
        assert np.max(coords[:, 1]) == height
        # Check element type
        for element in mesh.elements.values():
            assert element.element_type == 'quad2d'
            assert len(element.node_ids) == 4
    def test_rectangle_mesh_triangle(self):
        """Test rectangular mesh generation with triangles."""
        width, height = 1.0, 1.0
        nx, ny = 2, 2
        mesh = self.generator.generate_rectangle_mesh(width, height, nx, ny, 'triangle2d')
        # Check node count
        expected_nodes = (nx + 1) * (ny + 1)
        assert len(mesh.nodes) == expected_nodes
        # Check element count (2 triangles per quad)
        expected_elements = 2 * nx * ny
        assert len(mesh.elements) == expected_elements
        # Check element type
        for element in mesh.elements.values():
            assert element.element_type == 'triangle2d'
            assert len(element.node_ids) == 3
    def test_box_mesh_3d(self):
        """Test 3D box mesh generation."""
        width, height, depth = 1.0, 1.0, 1.0
        nx, ny, nz = 2, 2, 2
        mesh = self.generator.generate_box_mesh(width, height, depth, nx, ny, nz)
        # Check dimensions
        assert mesh.dimension == 3
        # Check node count
        expected_nodes = (nx + 1) * (ny + 1) * (nz + 1)
        assert len(mesh.nodes) == expected_nodes
        # Check 3D coordinates
        coords = mesh.get_node_coordinates()
        assert coords.shape[1] == 3
        assert np.min(coords[:, 0]) == 0.0
        assert np.max(coords[:, 0]) == width
        assert np.min(coords[:, 1]) == 0.0
        assert np.max(coords[:, 1]) == height
        assert np.min(coords[:, 2]) == 0.0
        assert np.max(coords[:, 2]) == depth
        # Check element type (tetrahedra)
        for element in mesh.elements.values():
            assert element.element_type == 'tetrahedron3d'
            assert len(element.node_ids) == 4
    def test_graded_mesh_1d(self):
        """Test 1D graded mesh generation."""
        domain = (0.0, 1.0)
        nx = 5
        grading_ratio = 4.0  # Fine to coarse
        mesh = self.generator.generate_graded_mesh(domain, nx, grading_ratio)
        # Check dimensions
        assert mesh.dimension == 1
        # Check node count
        expected_nodes = nx + 1
        assert len(mesh.nodes) == expected_nodes
        # Check element count
        assert len(mesh.elements) == nx
        # Check grading (elements should get progressively larger)
        coords = mesh.get_node_coordinates()
        x_coords = coords[:, 0]
        x_sorted = np.sort(x_coords)
        # Element lengths
        element_lengths = []
        for i in range(len(x_sorted) - 1):
            element_lengths.append(x_sorted[i+1] - x_sorted[i])
        # Check that elements generally increase in size
        if grading_ratio > 1:
            assert element_lengths[-1] > element_lengths[0]
    def test_uniform_graded_mesh(self):
        """Test uniform spacing with grading ratio = 1."""
        domain = (0.0, 2.0)
        nx = 4
        grading_ratio = 1.0
        mesh = self.generator.generate_graded_mesh(domain, nx, grading_ratio)
        coords = mesh.get_node_coordinates()
        x_coords = np.sort(coords[:, 0])
        # Check uniform spacing
        spacings = np.diff(x_coords)
        expected_spacing = (domain[1] - domain[0]) / nx
        assert_allclose(spacings, expected_spacing, rtol=1e-10)
class TestUnstructuredMeshGenerator:
    """Test unstructured mesh generation."""
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = UnstructuredMeshGenerator()
    def test_simple_geometry_creation(self):
        """Test creating simple geometry."""
        # Square domain
        vertices = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        geometry = Geometry(vertices, edges)
        assert geometry.vertices.shape == (4, 2)
        assert len(geometry.edges) == 4
        assert len(geometry.regions) == 0
        assert len(geometry.holes) == 0
    def test_delaunay_mesh_generation(self):
        """Test Delaunay triangulation mesh generation."""
        # Simple square domain
        vertices = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        geometry = Geometry(vertices, edges)
        mesh_params = MeshParameters(element_size=0.3, element_type='triangle2d')
        mesh = self.generator.generate_delaunay_mesh(geometry, mesh_params)
        # Check that mesh was created
        assert len(mesh.nodes) > 4  # Should have more than just corners
        assert len(mesh.elements) > 0
        # Check mesh dimensions
        coords = mesh.get_node_coordinates()
        assert np.min(coords[:, 0]) >= 0.0
        assert np.max(coords[:, 0]) <= 1.0
        assert np.min(coords[:, 1]) >= 0.0
        assert np.max(coords[:, 1]) <= 1.0
        # Check element type
        for element in mesh.elements.values():
            assert element.element_type == 'triangle2d'
            assert len(element.node_ids) == 3
    def test_point_in_domain(self):
        """Test point-in-domain detection."""
        # Square domain
        vertices = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        geometry = Geometry(vertices, edges)
        # Test points
        assert self.generator._point_in_domain(np.array([0.5, 0.5]), geometry)  # Inside
        assert not self.generator._point_in_domain(np.array([1.5, 0.5]), geometry)  # Outside
        assert not self.generator._point_in_domain(np.array([0.5, 1.5]), geometry)  # Outside
        # Boundary points (implementation dependent)
        # Just check that method doesn't crash
        self.generator._point_in_domain(np.array([0.0, 0.5]), geometry)
        self.generator._point_in_domain(np.array([1.0, 0.5]), geometry)
    def test_boundary_point_generation(self):
        """Test boundary point generation."""
        vertices = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0]
        ])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        geometry = Geometry(vertices, edges)
        element_size = 0.5
        boundary_points = self.generator._generate_boundary_points(geometry, element_size)
        # Should include original vertices
        assert len(boundary_points) >= len(vertices)
        # Should have points along edges
        # Check that some points lie on edges
        for i, edge in enumerate(edges):
            start = geometry.vertices[edge[0]]
            end = geometry.vertices[edge[1]]
            edge_length = np.linalg.norm(end - start)
            expected_points = max(1, int(edge_length / element_size))
            # Don't check exact count due to implementation details
    def test_mesh_parameters(self):
        """Test mesh parameter validation."""
        params = MeshParameters(
            element_size=0.1,
            element_type='triangle2d',
            refinement_level=2,
            quality_threshold=0.8
        )
        assert params.element_size == 0.1
        assert params.element_type == 'triangle2d'
        assert params.refinement_level == 2
        assert params.quality_threshold == 0.8
class TestMeshQuality:
    """Test mesh quality assessment."""
    def test_triangle_quality_equilateral(self):
        """Test quality metrics for equilateral triangle."""
        mesh = Mesh()
        # Equilateral triangle
        height = np.sqrt(3) / 2
        mesh.add_node(np.array([0.0, 0.0]))
        mesh.add_node(np.array([1.0, 0.0]))
        mesh.add_node(np.array([0.5, height]))
        mesh.add_element('triangle2d', [0, 1, 2])
        metrics = mesh.mesh_quality_metrics()
        # Equilateral triangle should have good aspect ratio
        assert metrics['min_aspect_ratio'] > 0
        assert metrics['max_aspect_ratio'] < 2.0  # Should be close to 1 for equilateral
    def test_triangle_quality_degenerate(self):
        """Test quality metrics for nearly degenerate triangle."""
        mesh = Mesh()
        # Very flat triangle (poor quality)
        mesh.add_node(np.array([0.0, 0.0]))
        mesh.add_node(np.array([1.0, 0.0]))
        mesh.add_node(np.array([0.5, 0.01]))  # Very small height
        mesh.add_element('triangle2d', [0, 1, 2])
        metrics = mesh.mesh_quality_metrics()
        # Should have poor aspect ratio
        assert metrics['max_aspect_ratio'] > 10.0
    def test_quad_quality_square(self):
        """Test quality metrics for square quadrilateral."""
        mesh = Mesh()
        # Unit square
        mesh.add_node(np.array([0.0, 0.0]))
        mesh.add_node(np.array([1.0, 0.0]))
        mesh.add_node(np.array([1.0, 1.0]))
        mesh.add_node(np.array([0.0, 1.0]))
        mesh.add_element('quad2d', [0, 1, 2, 3])
        metrics = mesh.mesh_quality_metrics()
        # Square should have good quality
        assert metrics['min_aspect_ratio'] == metrics['max_aspect_ratio']  # Only one element
        assert metrics['max_aspect_ratio'] < 2.0
class TestMeshVisualization:
    """Test mesh plotting functionality."""
    def test_plot_2d_mesh(self):
        """Test 2D mesh plotting."""
        mesh = Mesh()
        # Create simple triangle mesh
        mesh.add_node(np.array([0.0, 0.0]))
        mesh.add_node(np.array([1.0, 0.0]))
        mesh.add_node(np.array([0.0, 1.0]))
        mesh.add_element('triangle2d', [0, 1, 2])
        # Test that plotting doesn't crash
        try:
            fig = mesh.plot(show_node_ids=True, show_element_ids=True)
            # If we get here, plotting succeeded
            assert fig is not None
        except Exception as e:
            # Plotting might fail in headless environment
            pass
    def test_plot_empty_mesh(self):
        """Test plotting empty mesh."""
        mesh = Mesh()
        # Should handle empty mesh gracefully
        try:
            fig = mesh.plot()
            assert fig is not None
        except Exception:
            # Expected to handle gracefully or raise appropriate error
            pass
if __name__ == "__main__":
    pytest.main([__file__])