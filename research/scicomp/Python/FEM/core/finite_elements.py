"""
Finite Element Method Core Implementation
Comprehensive finite element framework for structural analysis including
element formulations, assembly procedures, and solution algorithms.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from scipy import sparse, linalg
import warnings
from ..utils.material_properties import MaterialProperty
@dataclass
class Node:
    """Finite element node definition."""
    id: int
    coordinates: np.ndarray
    dof_ids: List[int] = field(default_factory=list)
    boundary_conditions: Dict[int, float] = field(default_factory=dict)
    def __post_init__(self):
        """Validate node coordinates."""
        if len(self.coordinates) not in [1, 2, 3]:
            raise ValueError("Node coordinates must be 1D, 2D, or 3D")
@dataclass
class Element:
    """Base finite element definition."""
    id: int
    element_type: str
    node_ids: List[int]
    material_id: int
    thickness: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
class FiniteElementBase(ABC):
    """
    Abstract base class for finite elements.
    Features:
    - Element stiffness matrix computation
    - Element mass matrix computation
    - Shape function evaluation
    - Numerical integration
    """
    def __init__(self, element_id: int, nodes: List[Node], material: MaterialProperty):
        """
        Initialize finite element.
        Parameters:
            element_id: Unique element identifier
            nodes: List of element nodes
            material: Material properties
        """
        self.element_id = element_id
        self.nodes = nodes
        self.material = material
        self.dimension = len(nodes[0].coordinates)
        # Validate nodes
        if not all(len(node.coordinates) == self.dimension for node in nodes):
            raise ValueError("All nodes must have same dimensionality")
    @abstractmethod
    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """
        Evaluate shape functions at natural coordinates.
        Parameters:
            xi: Natural coordinates (parametric space)
        Returns:
            Shape function values
        """
        pass
    @abstractmethod
    def shape_function_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """
        Evaluate shape function derivatives at natural coordinates.
        Parameters:
            xi: Natural coordinates
        Returns:
            Shape function derivatives w.r.t. natural coordinates
        """
        pass
    @abstractmethod
    def stiffness_matrix(self) -> np.ndarray:
        """
        Compute element stiffness matrix.
        Returns:
            Element stiffness matrix
        """
        pass
    @abstractmethod
    def mass_matrix(self) -> np.ndarray:
        """
        Compute element mass matrix.
        Returns:
            Element mass matrix
        """
        pass
    def jacobian_matrix(self, xi: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix for coordinate transformation.
        Parameters:
            xi: Natural coordinates
        Returns:
            Jacobian matrix dx/dxi
        """
        dN_dxi = self.shape_function_derivatives(xi)
        node_coords = np.array([node.coordinates for node in self.nodes])
        # J = sum(x_i * dN_i/dxi)
        jacobian = node_coords.T @ dN_dxi
        return jacobian
    def jacobian_determinant(self, xi: np.ndarray) -> float:
        """Compute Jacobian determinant."""
        J = self.jacobian_matrix(xi)
        if J.ndim == 1:
            return J[0]  # 1D case
        elif J.shape[0] == J.shape[1]:
            return np.linalg.det(J)
        else:
            # Non-square Jacobian (e.g., 2D element in 3D space)
            return np.sqrt(np.linalg.det(J @ J.T))
    def global_coordinates(self, xi: np.ndarray) -> np.ndarray:
        """
        Map natural coordinates to global coordinates.
        Parameters:
            xi: Natural coordinates
        Returns:
            Global coordinates
        """
        N = self.shape_functions(xi)
        node_coords = np.array([node.coordinates for node in self.nodes])
        return N @ node_coords
    def gauss_quadrature_points(self, order: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get Gauss quadrature points and weights.
        Parameters:
            order: Quadrature order
        Returns:
            Tuple of (points, weights)
        """
        if order == 1:
            points = np.array([[0.0]])
            weights = np.array([2.0])
        elif order == 2:
            points = np.array([[-1/np.sqrt(3)], [1/np.sqrt(3)]])
            weights = np.array([1.0, 1.0])
        elif order == 3:
            points = np.array([[-np.sqrt(3/5)], [0.0], [np.sqrt(3/5)]])
            weights = np.array([5/9, 8/9, 5/9])
        else:
            raise NotImplementedError(f"Quadrature order {order} not implemented")
        # For higher dimensions, use tensor products
        if self.dimension > 1:
            # Create tensor product grid
            points_1d = points.flatten()
            weights_1d = weights
            if self.dimension == 2:
                xi_grid, eta_grid = np.meshgrid(points_1d, points_1d)
                points = np.column_stack([xi_grid.flatten(), eta_grid.flatten()])
                w_xi, w_eta = np.meshgrid(weights_1d, weights_1d)
                weights = (w_xi * w_eta).flatten()
            elif self.dimension == 3:
                xi_grid, eta_grid, zeta_grid = np.meshgrid(points_1d, points_1d, points_1d)
                points = np.column_stack([xi_grid.flatten(), eta_grid.flatten(), zeta_grid.flatten()])
                w_xi, w_eta, w_zeta = np.meshgrid(weights_1d, weights_1d, weights_1d)
                weights = (w_xi * w_eta * w_zeta).flatten()
        return points, weights
class LinearBar1D(FiniteElementBase):
    """
    1D linear bar element for truss analysis.
    Features:
    - Axial deformation only
    - Linear shape functions
    - 2 nodes, 1 DOF per node
    """
    def __init__(self, element_id: int, nodes: List[Node], material: MaterialProperty,
                 cross_section_area: float):
        """
        Initialize 1D bar element.
        Parameters:
            element_id: Element ID
            nodes: Two nodes
            material: Material properties
            cross_section_area: Cross-sectional area
        """
        if len(nodes) != 2:
            raise ValueError("Bar element requires exactly 2 nodes")
        super().__init__(element_id, nodes, material)
        self.area = cross_section_area
        # Calculate element length
        coords = np.array([node.coordinates for node in nodes])
        self.length = np.linalg.norm(coords[1] - coords[0])
    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Linear shape functions for bar element."""
        xi = np.atleast_1d(xi)
        N = np.zeros((len(xi), 2))
        N[:, 0] = 0.5 * (1 - xi)  # N1
        N[:, 1] = 0.5 * (1 + xi)  # N2
        return N
    def shape_function_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Shape function derivatives."""
        xi = np.atleast_1d(xi)
        dN_dxi = np.zeros((len(xi), 2))
        dN_dxi[:, 0] = -0.5  # dN1/dxi
        dN_dxi[:, 1] = 0.5   # dN2/dxi
        return dN_dxi
    def stiffness_matrix(self) -> np.ndarray:
        """Compute element stiffness matrix using analytical integration."""
        E = self.material.youngs_modulus
        A = self.area
        L = self.length
        # Analytical stiffness matrix for bar element
        k = (E * A / L) * np.array([[1, -1], [-1, 1]])
        return k
    def mass_matrix(self) -> np.ndarray:
        """Compute element mass matrix."""
        rho = self.material.density
        A = self.area
        L = self.length
        # Consistent mass matrix
        m = (rho * A * L / 6) * np.array([[2, 1], [1, 2]])
        return m
class LinearTriangle2D(FiniteElementBase):
    """
    2D linear triangular element for plane stress/strain analysis.
    Features:
    - Linear shape functions
    - Constant strain field
    - 3 nodes, 2 DOF per node
    """
    def __init__(self, element_id: int, nodes: List[Node], material: MaterialProperty,
                 thickness: float = 1.0, plane_stress: bool = True):
        """
        Initialize 2D triangular element.
        Parameters:
            element_id: Element ID
            nodes: Three nodes
            material: Material properties
            thickness: Element thickness
            plane_stress: True for plane stress, False for plane strain
        """
        if len(nodes) != 3:
            raise ValueError("Triangle element requires exactly 3 nodes")
        super().__init__(element_id, nodes, material)
        self.thickness = thickness
        self.plane_stress = plane_stress
        # Calculate element area
        coords = np.array([node.coordinates for node in nodes])
        self.area = 0.5 * abs(np.cross(coords[1] - coords[0], coords[2] - coords[0]))
        if self.area <= 0:
            raise ValueError("Element has zero or negative area")
    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """
        Linear shape functions for triangular element.
        Parameters:
            xi: Natural coordinates [xi, eta]
        """
        xi = np.atleast_2d(xi)
        N = np.zeros((xi.shape[0], 3))
        # Area coordinates (barycentric coordinates)
        N[:, 0] = 1 - xi[:, 0] - xi[:, 1]  # N1
        N[:, 1] = xi[:, 0]                 # N2
        N[:, 2] = xi[:, 1]                 # N3
        return N
    def shape_function_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Shape function derivatives."""
        xi = np.atleast_2d(xi)
        dN_dxi = np.zeros((xi.shape[0], 3, 2))
        # Derivatives w.r.t. natural coordinates
        dN_dxi[:, 0, 0] = -1  # dN1/dxi
        dN_dxi[:, 0, 1] = -1  # dN1/deta
        dN_dxi[:, 1, 0] = 1   # dN2/dxi
        dN_dxi[:, 1, 1] = 0   # dN2/deta
        dN_dxi[:, 2, 0] = 0   # dN3/dxi
        dN_dxi[:, 2, 1] = 1   # dN3/deta
        return dN_dxi
    def strain_displacement_matrix(self) -> np.ndarray:
        """
        Compute strain-displacement matrix B.
        For constant strain triangle, B is constant throughout element.
        """
        # Get node coordinates
        coords = np.array([node.coordinates for node in self.nodes])
        x = coords[:, 0]
        y = coords[:, 1]
        # Calculate area coordinates derivatives
        A = 2 * self.area
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / A
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / A
        # Strain-displacement matrix
        B = np.zeros((3, 6))
        # Strain εxx
        B[0, 0::2] = b  # ∂N/∂x
        # Strain εyy
        B[1, 1::2] = c  # ∂N/∂y
        # Shear strain γxy
        B[2, 0::2] = c  # ∂N/∂y
        B[2, 1::2] = b  # ∂N/∂x
        return B
    def constitutive_matrix(self) -> np.ndarray:
        """Compute constitutive matrix D."""
        E = self.material.youngs_modulus
        nu = self.material.poissons_ratio
        if self.plane_stress:
            # Plane stress
            factor = E / (1 - nu**2)
            D = factor * np.array([
                [1,  nu, 0],
                [nu, 1,  0],
                [0,  0,  (1-nu)/2]
            ])
        else:
            # Plane strain
            factor = E / ((1 + nu) * (1 - 2*nu))
            D = factor * np.array([
                [1-nu, nu,    0],
                [nu,   1-nu,  0],
                [0,    0,     (1-2*nu)/2]
            ])
        return D
    def stiffness_matrix(self) -> np.ndarray:
        """Compute element stiffness matrix."""
        B = self.strain_displacement_matrix()
        D = self.constitutive_matrix()
        # K = ∫ B^T D B dV = B^T D B * (Area * thickness)
        K = B.T @ D @ B * self.area * self.thickness
        return K
    def mass_matrix(self) -> np.ndarray:
        """Compute element mass matrix."""
        rho = self.material.density
        # Lumped mass matrix (1/3 of total mass at each node)
        total_mass = rho * self.area * self.thickness
        node_mass = total_mass / 3
        M = np.zeros((6, 6))
        for i in range(3):
            M[2*i, 2*i] = node_mass      # x-direction
            M[2*i+1, 2*i+1] = node_mass  # y-direction
        return M
class LinearQuadrilateral2D(FiniteElementBase):
    """
    2D linear quadrilateral element for plane stress/strain analysis.
    Features:
    - Bilinear shape functions
    - Isoparametric formulation
    - 4 nodes, 2 DOF per node
    """
    def __init__(self, element_id: int, nodes: List[Node], material: MaterialProperty,
                 thickness: float = 1.0, plane_stress: bool = True):
        """
        Initialize 2D quadrilateral element.
        Parameters:
            element_id: Element ID
            nodes: Four nodes (in counterclockwise order)
            material: Material properties
            thickness: Element thickness
            plane_stress: True for plane stress, False for plane strain
        """
        if len(nodes) != 4:
            raise ValueError("Quadrilateral element requires exactly 4 nodes")
        super().__init__(element_id, nodes, material)
        self.thickness = thickness
        self.plane_stress = plane_stress
    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """
        Bilinear shape functions for quadrilateral element.
        Parameters:
            xi: Natural coordinates [xi, eta] in range [-1, 1]
        """
        xi = np.atleast_2d(xi)
        xi_coord = xi[:, 0]
        eta_coord = xi[:, 1]
        N = np.zeros((xi.shape[0], 4))
        # Bilinear shape functions
        N[:, 0] = 0.25 * (1 - xi_coord) * (1 - eta_coord)  # N1
        N[:, 1] = 0.25 * (1 + xi_coord) * (1 - eta_coord)  # N2
        N[:, 2] = 0.25 * (1 + xi_coord) * (1 + eta_coord)  # N3
        N[:, 3] = 0.25 * (1 - xi_coord) * (1 + eta_coord)  # N4
        return N
    def shape_function_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Shape function derivatives w.r.t. natural coordinates."""
        xi = np.atleast_2d(xi)
        xi_coord = xi[:, 0]
        eta_coord = xi[:, 1]
        dN_dxi = np.zeros((xi.shape[0], 4, 2))
        # Derivatives w.r.t. xi
        dN_dxi[:, 0, 0] = -0.25 * (1 - eta_coord)  # dN1/dxi
        dN_dxi[:, 1, 0] = 0.25 * (1 - eta_coord)   # dN2/dxi
        dN_dxi[:, 2, 0] = 0.25 * (1 + eta_coord)   # dN3/dxi
        dN_dxi[:, 3, 0] = -0.25 * (1 + eta_coord)  # dN4/dxi
        # Derivatives w.r.t. eta
        dN_dxi[:, 0, 1] = -0.25 * (1 - xi_coord)   # dN1/deta
        dN_dxi[:, 1, 1] = -0.25 * (1 + xi_coord)   # dN2/deta
        dN_dxi[:, 2, 1] = 0.25 * (1 + xi_coord)    # dN3/deta
        dN_dxi[:, 3, 1] = 0.25 * (1 - xi_coord)    # dN4/deta
        return dN_dxi
    def strain_displacement_matrix(self, xi: np.ndarray) -> np.ndarray:
        """
        Compute strain-displacement matrix B at given natural coordinates.
        Parameters:
            xi: Natural coordinates [xi, eta]
        """
        # Shape function derivatives w.r.t. natural coordinates
        dN_dxi = self.shape_function_derivatives(xi.reshape(1, -1))[0]
        # Jacobian matrix and its inverse
        J = self.jacobian_matrix(xi)
        J_inv = np.linalg.inv(J)
        # Shape function derivatives w.r.t. global coordinates
        dN_dx = dN_dxi @ J_inv
        # Strain-displacement matrix
        B = np.zeros((3, 8))
        for i in range(4):
            # Strain εxx
            B[0, 2*i] = dN_dx[i, 0]
            # Strain εyy
            B[1, 2*i+1] = dN_dx[i, 1]
            # Shear strain γxy
            B[2, 2*i] = dN_dx[i, 1]
            B[2, 2*i+1] = dN_dx[i, 0]
        return B
    def constitutive_matrix(self) -> np.ndarray:
        """Compute constitutive matrix D."""
        E = self.material.youngs_modulus
        nu = self.material.poissons_ratio
        if self.plane_stress:
            # Plane stress
            factor = E / (1 - nu**2)
            D = factor * np.array([
                [1,  nu, 0],
                [nu, 1,  0],
                [0,  0,  (1-nu)/2]
            ])
        else:
            # Plane strain
            factor = E / ((1 + nu) * (1 - 2*nu))
            D = factor * np.array([
                [1-nu, nu,    0],
                [nu,   1-nu,  0],
                [0,    0,     (1-2*nu)/2]
            ])
        return D
    def stiffness_matrix(self) -> np.ndarray:
        """Compute element stiffness matrix using numerical integration."""
        D = self.constitutive_matrix()
        # Gauss quadrature points and weights
        points, weights = self.gauss_quadrature_points(2)  # 2x2 integration
        K = np.zeros((8, 8))
        for i, (point, weight) in enumerate(zip(points, weights)):
            # Strain-displacement matrix at integration point
            B = self.strain_displacement_matrix(point)
            # Jacobian determinant
            det_J = self.jacobian_determinant(point)
            # Add contribution to stiffness matrix
            K += B.T @ D @ B * det_J * weight * self.thickness
        return K
    def mass_matrix(self) -> np.ndarray:
        """Compute element mass matrix using numerical integration."""
        rho = self.material.density
        # Gauss quadrature points and weights
        points, weights = self.gauss_quadrature_points(2)
        M = np.zeros((8, 8))
        for point, weight in zip(points, weights):
            # Shape functions at integration point
            N = self.shape_functions(point.reshape(1, -1))[0]
            # Create shape function matrix for 2D displacement
            N_matrix = np.zeros((2, 8))
            for i in range(4):
                N_matrix[0, 2*i] = N[i]      # x-displacement
                N_matrix[1, 2*i+1] = N[i]    # y-displacement
            # Jacobian determinant
            det_J = self.jacobian_determinant(point)
            # Add contribution to mass matrix
            M += rho * N_matrix.T @ N_matrix * det_J * weight * self.thickness
        return M
class LinearTetrahedron3D(FiniteElementBase):
    """
    3D linear tetrahedral element for solid mechanics.
    Features:
    - Linear shape functions
    - Constant strain field
    - 4 nodes, 3 DOF per node
    """
    def __init__(self, element_id: int, nodes: List[Node], material: MaterialProperty):
        """
        Initialize 3D tetrahedral element.
        Parameters:
            element_id: Element ID
            nodes: Four nodes
            material: Material properties
        """
        if len(nodes) != 4:
            raise ValueError("Tetrahedron element requires exactly 4 nodes")
        super().__init__(element_id, nodes, material)
        # Calculate element volume
        coords = np.array([node.coordinates for node in nodes])
        self.volume = abs(np.linalg.det(coords[1:] - coords[0])) / 6
        if self.volume <= 0:
            raise ValueError("Element has zero or negative volume")
    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """
        Linear shape functions for tetrahedral element.
        Parameters:
            xi: Natural coordinates [xi, eta, zeta]
        """
        xi = np.atleast_2d(xi)
        N = np.zeros((xi.shape[0], 4))
        # Volume coordinates
        N[:, 0] = 1 - xi[:, 0] - xi[:, 1] - xi[:, 2]  # N1
        N[:, 1] = xi[:, 0]                            # N2
        N[:, 2] = xi[:, 1]                            # N3
        N[:, 3] = xi[:, 2]                            # N4
        return N
    def shape_function_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Shape function derivatives."""
        xi = np.atleast_2d(xi)
        dN_dxi = np.zeros((xi.shape[0], 4, 3))
        # Derivatives w.r.t. natural coordinates
        dN_dxi[:, 0, :] = [-1, -1, -1]  # dN1/dxi, dN1/deta, dN1/dzeta
        dN_dxi[:, 1, :] = [1, 0, 0]     # dN2/dxi, dN2/deta, dN2/dzeta
        dN_dxi[:, 2, :] = [0, 1, 0]     # dN3/dxi, dN3/deta, dN3/dzeta
        dN_dxi[:, 3, :] = [0, 0, 1]     # dN4/dxi, dN4/deta, dN4/dzeta
        return dN_dxi
    def strain_displacement_matrix(self) -> np.ndarray:
        """
        Compute strain-displacement matrix B.
        For constant strain tetrahedron, B is constant throughout element.
        """
        # Get node coordinates
        coords = np.array([node.coordinates for node in self.nodes])
        # Calculate volume coordinate derivatives
        V = 6 * self.volume
        # Construct matrix for calculating derivatives
        coord_matrix = np.column_stack([np.ones(4), coords])
        # Calculate derivatives of volume coordinates
        derivs = np.linalg.inv(coord_matrix)
        b = derivs[1, :]  # ∂N/∂x
        c = derivs[2, :]  # ∂N/∂y
        d = derivs[3, :]  # ∂N/∂z
        # Strain-displacement matrix (6 strains, 12 DOFs)
        B = np.zeros((6, 12))
        for i in range(4):
            # Normal strains
            B[0, 3*i] = b[i]      # εxx
            B[1, 3*i+1] = c[i]    # εyy
            B[2, 3*i+2] = d[i]    # εzz
            # Shear strains
            B[3, 3*i+1] = d[i]    # γyz
            B[3, 3*i+2] = c[i]
            B[4, 3*i] = d[i]      # γxz
            B[4, 3*i+2] = b[i]
            B[5, 3*i] = c[i]      # γxy
            B[5, 3*i+1] = b[i]
        return B
    def constitutive_matrix(self) -> np.ndarray:
        """Compute 3D constitutive matrix D."""
        E = self.material.youngs_modulus
        nu = self.material.poissons_ratio
        # 3D isotropic elasticity matrix
        factor = E / ((1 + nu) * (1 - 2*nu))
        D = np.zeros((6, 6))
        # Diagonal terms
        D[0, 0] = D[1, 1] = D[2, 2] = factor * (1 - nu)
        D[3, 3] = D[4, 4] = D[5, 5] = factor * (1 - 2*nu) / 2
        # Off-diagonal terms
        off_diag = factor * nu
        D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = off_diag
        return D
    def stiffness_matrix(self) -> np.ndarray:
        """Compute element stiffness matrix."""
        B = self.strain_displacement_matrix()
        D = self.constitutive_matrix()
        # K = B^T D B * Volume
        K = B.T @ D @ B * self.volume
        return K
    def mass_matrix(self) -> np.ndarray:
        """Compute element mass matrix."""
        rho = self.material.density
        # Lumped mass matrix (1/4 of total mass at each node)
        total_mass = rho * self.volume
        node_mass = total_mass / 4
        M = np.zeros((12, 12))
        for i in range(4):
            for j in range(3):  # 3 DOF per node
                M[3*i+j, 3*i+j] = node_mass
        return M
def create_element_factory() -> Dict[str, type]:
    """Create factory for element types."""
    return {
        'bar1d': LinearBar1D,
        'triangle2d': LinearTriangle2D,
        'quad2d': LinearQuadrilateral2D,
        'tetrahedron3d': LinearTetrahedron3D
    }