"""
Tests for Finite Element Implementations
Comprehensive test suite for finite element classes including element formulations,
shape functions, stiffness matrices, and numerical integration.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.finite_elements import (
    Node, Element, LinearBar1D, LinearTriangle2D,
    LinearQuadrilateral2D, LinearTetrahedron3D
)
from utils.material_properties import MaterialLibrary
class TestNode:
    """Test Node class functionality."""
    def test_node_creation_2d(self):
        """Test creating 2D node."""
        node = Node(1, np.array([1.0, 2.0]))
        assert node.id == 1
        assert_array_almost_equal(node.coordinates, [1.0, 2.0])
        assert len(node.dof_ids) == 0
        assert len(node.boundary_conditions) == 0
    def test_node_creation_3d(self):
        """Test creating 3D node."""
        node = Node(2, np.array([1.0, 2.0, 3.0]))
        assert node.id == 2
        assert_array_almost_equal(node.coordinates, [1.0, 2.0, 3.0])
    def test_invalid_coordinates(self):
        """Test invalid coordinate dimensions."""
        with pytest.raises(ValueError):
            Node(1, np.array([1.0, 2.0, 3.0, 4.0]))  # 4D not supported
class TestElement:
    """Test Element class functionality."""
    def test_element_creation(self):
        """Test creating element."""
        element = Element(1, 'triangle2d', [0, 1, 2], 0)
        assert element.id == 1
        assert element.element_type == 'triangle2d'
        assert element.node_ids == [0, 1, 2]
        assert element.material_id == 0
        assert element.thickness == 1.0
class TestLinearBar1D:
    """Test 1D linear bar element."""
    def setup_method(self):
        """Set up test fixtures."""
        self.nodes = [
            Node(0, np.array([0.0])),
            Node(1, np.array([1.0]))
        ]
        self.material = MaterialLibrary.steel_mild()
        self.area = 0.01  # 1 cm²
        self.element = LinearBar1D(0, self.nodes, self.material, self.area)
    def test_element_creation(self):
        """Test element creation."""
        assert self.element.element_id == 0
        assert len(self.element.nodes) == 2
        assert self.element.area == self.area
        assert self.element.length == 1.0
    def test_invalid_node_count(self):
        """Test error with wrong number of nodes."""
        with pytest.raises(ValueError):
            LinearBar1D(0, [self.nodes[0]], self.material, self.area)
    def test_shape_functions(self):
        """Test shape function evaluation."""
        # Test at element center (xi = 0)
        N = self.element.shape_functions(np.array([0.0]))
        expected = np.array([[0.5, 0.5]])
        assert_array_almost_equal(N, expected)
        # Test at nodes
        N1 = self.element.shape_functions(np.array([-1.0]))
        assert_array_almost_equal(N1, [[1.0, 0.0]])
        N2 = self.element.shape_functions(np.array([1.0]))
        assert_array_almost_equal(N2, [[0.0, 1.0]])
    def test_shape_function_derivatives(self):
        """Test shape function derivatives."""
        dN = self.element.shape_function_derivatives(np.array([0.0]))
        expected = np.array([[-0.5, 0.5]])
        assert_array_almost_equal(dN, expected)
    def test_stiffness_matrix(self):
        """Test stiffness matrix calculation."""
        K = self.element.stiffness_matrix()
        # Expected analytical result
        E = self.material.youngs_modulus
        A = self.area
        L = self.element.length
        k_expected = (E * A / L) * np.array([[1, -1], [-1, 1]])
        assert K.shape == (2, 2)
        assert_array_almost_equal(K, k_expected)
    def test_mass_matrix(self):
        """Test mass matrix calculation."""
        M = self.element.mass_matrix()
        # Expected consistent mass matrix
        rho = self.material.density
        A = self.area
        L = self.element.length
        m_expected = (rho * A * L / 6) * np.array([[2, 1], [1, 2]])
        assert M.shape == (2, 2)
        assert_array_almost_equal(M, m_expected)
    def test_jacobian_determinant(self):
        """Test Jacobian determinant."""
        det_J = self.element.jacobian_determinant(np.array([0.0]))
        expected = self.element.length / 2  # For linear element
        assert_allclose(det_J, expected)
class TestLinearTriangle2D:
    """Test 2D linear triangular element."""
    def setup_method(self):
        """Set up test fixtures."""
        # Unit triangle
        self.nodes = [
            Node(0, np.array([0.0, 0.0])),
            Node(1, np.array([1.0, 0.0])),
            Node(2, np.array([0.0, 1.0]))
        ]
        self.material = MaterialLibrary.aluminum_6061()
        self.thickness = 0.01
        self.element = LinearTriangle2D(0, self.nodes, self.material,
                                       self.thickness, plane_stress=True)
    def test_element_creation(self):
        """Test element creation."""
        assert self.element.element_id == 0
        assert len(self.element.nodes) == 3
        assert self.element.thickness == self.thickness
        assert self.element.plane_stress == True
        assert self.element.area == 0.5  # Unit triangle area
    def test_invalid_node_count(self):
        """Test error with wrong number of nodes."""
        with pytest.raises(ValueError):
            LinearTriangle2D(0, self.nodes[:2], self.material, self.thickness)
    def test_zero_area_triangle(self):
        """Test error with degenerate triangle."""
        # Collinear nodes
        bad_nodes = [
            Node(0, np.array([0.0, 0.0])),
            Node(1, np.array([1.0, 0.0])),
            Node(2, np.array([2.0, 0.0]))
        ]
        with pytest.raises(ValueError):
            LinearTriangle2D(0, bad_nodes, self.material, self.thickness)
    def test_shape_functions(self):
        """Test shape function evaluation."""
        # Test at triangle center
        xi = np.array([[1/3, 1/3]])
        N = self.element.shape_functions(xi)
        expected = np.array([[1/3, 1/3, 1/3]])
        assert_array_almost_equal(N, expected)
        # Test at nodes
        N1 = self.element.shape_functions(np.array([[0.0, 0.0]]))
        assert_array_almost_equal(N1, [[1.0, 0.0, 0.0]])
        N2 = self.element.shape_functions(np.array([[1.0, 0.0]]))
        assert_array_almost_equal(N2, [[0.0, 1.0, 0.0]])
        N3 = self.element.shape_functions(np.array([[0.0, 1.0]]))
        assert_array_almost_equal(N3, [[0.0, 0.0, 1.0]])
    def test_shape_function_derivatives(self):
        """Test shape function derivatives."""
        xi = np.array([[0.0, 0.0]])
        dN = self.element.shape_function_derivatives(xi)
        # For linear triangle, derivatives are constant
        expected = np.array([[-1, -1], [1, 0], [0, 1]])
        assert_array_almost_equal(dN[0], expected)
    def test_strain_displacement_matrix(self):
        """Test strain-displacement matrix."""
        B = self.element.strain_displacement_matrix()
        # B matrix should be 3x6 for 2D triangle
        assert B.shape == (3, 6)
        # Check that B matrix is constant (constant strain triangle)
        # No specific values to check without detailed calculation
    def test_constitutive_matrix_plane_stress(self):
        """Test plane stress constitutive matrix."""
        D = self.element.constitutive_matrix()
        E = self.material.youngs_modulus
        nu = self.material.poissons_ratio
        factor = E / (1 - nu**2)
        expected = factor * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1-nu)/2]
        ])
        assert D.shape == (3, 3)
        assert_array_almost_equal(D, expected)
    def test_constitutive_matrix_plane_strain(self):
        """Test plane strain constitutive matrix."""
        element = LinearTriangle2D(0, self.nodes, self.material,
                                  self.thickness, plane_stress=False)
        D = element.constitutive_matrix()
        E = self.material.youngs_modulus
        nu = self.material.poissons_ratio
        factor = E / ((1 + nu) * (1 - 2*nu))
        expected = factor * np.array([
            [1-nu, nu, 0],
            [nu, 1-nu, 0],
            [0, 0, (1-2*nu)/2]
        ])
        assert D.shape == (3, 3)
        assert_array_almost_equal(D, expected)
    def test_stiffness_matrix(self):
        """Test stiffness matrix calculation."""
        K = self.element.stiffness_matrix()
        # Should be 6x6 matrix (3 nodes × 2 DOF/node)
        assert K.shape == (6, 6)
        # Matrix should be symmetric
        assert_array_almost_equal(K, K.T)
        # Matrix should be positive semi-definite (for well-posed elements)
        eigenvals = np.linalg.eigvals(K)
        assert np.all(eigenvals >= -1e-10)  # Allow small numerical errors
    def test_mass_matrix(self):
        """Test mass matrix calculation."""
        M = self.element.mass_matrix()
        # Should be 6x6 matrix
        assert M.shape == (6, 6)
        # Matrix should be symmetric
        assert_array_almost_equal(M, M.T)
        # Matrix should be positive definite
        eigenvals = np.linalg.eigvals(M)
        assert np.all(eigenvals > 0)
class TestLinearQuadrilateral2D:
    """Test 2D linear quadrilateral element."""
    def setup_method(self):
        """Set up test fixtures."""
        # Unit square
        self.nodes = [
            Node(0, np.array([0.0, 0.0])),
            Node(1, np.array([1.0, 0.0])),
            Node(2, np.array([1.0, 1.0])),
            Node(3, np.array([0.0, 1.0]))
        ]
        self.material = MaterialLibrary.steel_mild()
        self.thickness = 0.02
        self.element = LinearQuadrilateral2D(0, self.nodes, self.material,
                                           self.thickness, plane_stress=True)
    def test_element_creation(self):
        """Test element creation."""
        assert self.element.element_id == 0
        assert len(self.element.nodes) == 4
        assert self.element.thickness == self.thickness
    def test_invalid_node_count(self):
        """Test error with wrong number of nodes."""
        with pytest.raises(ValueError):
            LinearQuadrilateral2D(0, self.nodes[:3], self.material, self.thickness)
    def test_shape_functions(self):
        """Test shape function evaluation."""
        # Test at element center
        xi = np.array([[0.0, 0.0]])
        N = self.element.shape_functions(xi)
        expected = np.array([[0.25, 0.25, 0.25, 0.25]])
        assert_array_almost_equal(N, expected)
        # Test at corners
        corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
        for i, (xi_val, eta_val) in enumerate(corners):
            xi_point = np.array([[xi_val, eta_val]])
            N = self.element.shape_functions(xi_point)
            expected_N = np.zeros(4)
            expected_N[i] = 1.0
            assert_array_almost_equal(N[0], expected_N)
    def test_shape_function_derivatives(self):
        """Test shape function derivatives."""
        xi = np.array([[0.0, 0.0]])
        dN = self.element.shape_function_derivatives(xi)
        # At element center
        expected = np.array([
            [-0.25, 0.25, 0.25, -0.25],  # dN/dxi
            [-0.25, -0.25, 0.25, 0.25]   # dN/deta
        ]).T
        assert dN.shape == (1, 4, 2)
        assert_array_almost_equal(dN[0], expected)
    def test_jacobian_matrix(self):
        """Test Jacobian matrix calculation."""
        xi = np.array([0.0, 0.0])
        J = self.element.jacobian_matrix(xi)
        # For unit square, Jacobian should be [0.5, 0; 0, 0.5]
        expected = np.array([[0.5, 0.0], [0.0, 0.5]])
        assert_array_almost_equal(J, expected)
    def test_jacobian_determinant(self):
        """Test Jacobian determinant."""
        xi = np.array([0.0, 0.0])
        det_J = self.element.jacobian_determinant(xi)
        # For unit square
        expected = 0.25
        assert_allclose(det_J, expected)
    def test_gauss_quadrature_points(self):
        """Test Gauss quadrature point generation."""
        points, weights = self.element.gauss_quadrature_points(2)
        # For 2x2 integration, should have 4 points
        assert points.shape == (4, 2)
        assert len(weights) == 4
        # Check that weights sum to expected value (area of natural element)
        assert_allclose(np.sum(weights), 4.0)  # 2×2 natural element area
    def test_stiffness_matrix(self):
        """Test stiffness matrix calculation."""
        K = self.element.stiffness_matrix()
        # Should be 8x8 matrix (4 nodes × 2 DOF/node)
        assert K.shape == (8, 8)
        # Matrix should be symmetric
        assert_array_almost_equal(K, K.T, decimal=10)
        # Matrix should be positive semi-definite
        eigenvals = np.linalg.eigvals(K)
        assert np.all(eigenvals >= -1e-10)
    def test_mass_matrix(self):
        """Test mass matrix calculation."""
        M = self.element.mass_matrix()
        # Should be 8x8 matrix
        assert M.shape == (8, 8)
        # Matrix should be symmetric
        assert_array_almost_equal(M, M.T)
        # Matrix should be positive definite
        eigenvals = np.linalg.eigvals(M)
        assert np.all(eigenvals > 0)
class TestLinearTetrahedron3D:
    """Test 3D linear tetrahedral element."""
    def setup_method(self):
        """Set up test fixtures."""
        # Unit tetrahedron
        self.nodes = [
            Node(0, np.array([0.0, 0.0, 0.0])),
            Node(1, np.array([1.0, 0.0, 0.0])),
            Node(2, np.array([0.0, 1.0, 0.0])),
            Node(3, np.array([0.0, 0.0, 1.0]))
        ]
        self.material = MaterialLibrary.titanium_ti6al4v()
        self.element = LinearTetrahedron3D(0, self.nodes, self.material)
    def test_element_creation(self):
        """Test element creation."""
        assert self.element.element_id == 0
        assert len(self.element.nodes) == 4
        assert self.element.volume == 1/6  # Unit tetrahedron volume
    def test_invalid_node_count(self):
        """Test error with wrong number of nodes."""
        with pytest.raises(ValueError):
            LinearTetrahedron3D(0, self.nodes[:3], self.material)
    def test_zero_volume_tetrahedron(self):
        """Test error with degenerate tetrahedron."""
        # Coplanar nodes
        bad_nodes = [
            Node(0, np.array([0.0, 0.0, 0.0])),
            Node(1, np.array([1.0, 0.0, 0.0])),
            Node(2, np.array([0.0, 1.0, 0.0])),
            Node(3, np.array([0.5, 0.5, 0.0]))  # In same plane
        ]
        with pytest.raises(ValueError):
            LinearTetrahedron3D(0, bad_nodes, self.material)
    def test_shape_functions(self):
        """Test shape function evaluation."""
        # Test at tetrahedron center
        xi = np.array([[0.25, 0.25, 0.25]])
        N = self.element.shape_functions(xi)
        expected = np.array([[0.25, 0.25, 0.25, 0.25]])
        assert_array_almost_equal(N, expected)
        # Test at nodes
        N1 = self.element.shape_functions(np.array([[0.0, 0.0, 0.0]]))
        assert_array_almost_equal(N1, [[1.0, 0.0, 0.0, 0.0]])
    def test_shape_function_derivatives(self):
        """Test shape function derivatives."""
        xi = np.array([[0.0, 0.0, 0.0]])
        dN = self.element.shape_function_derivatives(xi)
        # For linear tetrahedron, derivatives are constant
        expected = np.array([
            [-1, -1, -1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        assert dN.shape == (1, 4, 3)
        assert_array_almost_equal(dN[0], expected)
    def test_constitutive_matrix(self):
        """Test 3D constitutive matrix."""
        D = self.element.constitutive_matrix()
        assert D.shape == (6, 6)
        # Matrix should be symmetric
        assert_array_almost_equal(D, D.T)
        # Check diagonal terms
        E = self.material.youngs_modulus
        nu = self.material.poissons_ratio
        factor = E / ((1 + nu) * (1 - 2*nu))
        expected_diag = factor * (1 - nu)
        assert_allclose(D[0, 0], expected_diag)
        assert_allclose(D[1, 1], expected_diag)
        assert_allclose(D[2, 2], expected_diag)
    def test_stiffness_matrix(self):
        """Test stiffness matrix calculation."""
        K = self.element.stiffness_matrix()
        # Should be 12x12 matrix (4 nodes × 3 DOF/node)
        assert K.shape == (12, 12)
        # Matrix should be symmetric
        assert_array_almost_equal(K, K.T, decimal=10)
        # Matrix should be positive semi-definite
        eigenvals = np.linalg.eigvals(K)
        assert np.all(eigenvals >= -1e-10)
    def test_mass_matrix(self):
        """Test mass matrix calculation."""
        M = self.element.mass_matrix()
        # Should be 12x12 matrix
        assert M.shape == (12, 12)
        # Matrix should be diagonal (lumped mass)
        off_diagonal = M - np.diag(np.diag(M))
        assert_allclose(off_diagonal, 0, atol=1e-15)
        # Total mass should equal element mass
        total_mass = np.sum(np.diag(M))
        expected_mass = self.material.density * self.element.volume
        assert_allclose(total_mass, expected_mass)
def test_element_factory():
    """Test element factory function."""
    from core.finite_elements import create_element_factory
    factory = create_element_factory()
    assert 'bar1d' in factory
    assert 'triangle2d' in factory
    assert 'quad2d' in factory
    assert 'tetrahedron3d' in factory
    assert factory['bar1d'] == LinearBar1D
    assert factory['triangle2d'] == LinearTriangle2D
    assert factory['quad2d'] == LinearQuadrilateral2D
    assert factory['tetrahedron3d'] == LinearTetrahedron3D
if __name__ == "__main__":
    pytest.main([__file__])