"""Boundary Conditions for PDE Solvers.
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
"""
import numpy as np
from typing import Callable, Dict, Any, List, Union, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy import sparse
import warnings
@dataclass
class BCData:
    """Data structure for boundary condition specification."""
    location: Union[int, List[int], str]  # Node indices or region name
    value: Union[float, np.ndarray, Callable]  # BC value(s)
    normal: Optional[np.ndarray] = None  # Normal direction for Neumann BC
    coefficient: Optional[float] = None  # Coefficient for Robin BC
    time_dependent: bool = False  # Whether BC is time-dependent
class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions."""
    def __init__(self, bc_data: BCData):
        """Initialize boundary condition.
        Args:
            bc_data: Boundary condition data
        """
        self.bc_data = bc_data
        self.location = bc_data.location
        self.value = bc_data.value
        self.time_dependent = bc_data.time_dependent
    @abstractmethod
    def apply(self, matrix: sparse.spmatrix, rhs: np.ndarray,
              mesh: Optional[Dict] = None, time: float = 0.0) -> Tuple[sparse.spmatrix, np.ndarray]:
        """Apply boundary condition to system.
        Args:
            matrix: System matrix
            rhs: Right-hand side vector
            mesh: Mesh information
            time: Current time (for time-dependent BC)
        Returns:
            Modified matrix and RHS
        """
        pass
    def evaluate_value(self, mesh: Optional[Dict] = None, time: float = 0.0) -> Union[float, np.ndarray]:
        """Evaluate boundary condition value at given time."""
        if callable(self.value):
            if self.time_dependent:
                return self.value(time)
            else:
                # Assume spatial dependence
                if mesh is not None and 'nodes' in mesh:
                    return self.value(mesh['nodes'])
                else:
                    return self.value(0.0)  # Fallback
        else:
            return self.value
class DirichletBC(BoundaryCondition):
    """Dirichlet (essential) boundary condition: u = g.
    Prescribes the value of the solution at boundary points.
    """
    def apply(self, matrix: sparse.spmatrix, rhs: np.ndarray,
              mesh: Optional[Dict] = None, time: float = 0.0) -> Tuple[sparse.spmatrix, np.ndarray]:
        """Apply Dirichlet boundary condition."""
        # Get boundary value
        bc_value = self.evaluate_value(mesh, time)
        # Handle different location specifications
        if isinstance(self.location, int):
            nodes = [self.location]
        elif isinstance(self.location, list):
            nodes = self.location
        elif isinstance(self.location, str):
            nodes = self._get_boundary_nodes(self.location, mesh)
        else:
            raise ValueError(f"Invalid location specification: {self.location}")
        # Ensure bc_value is compatible with nodes
        if np.isscalar(bc_value):
            bc_values = [bc_value] * len(nodes)
        else:
            bc_values = np.asarray(bc_value)
            if len(bc_values) != len(nodes):
                bc_values = np.full(len(nodes), bc_values[0])
        # Modify matrix and RHS
        matrix_modified = matrix.copy()
        rhs_modified = rhs.copy()
        for node, value in zip(nodes, bc_values):
            if 0 <= node < matrix.shape[0]:
                # Set row to identity equation
                matrix_modified.data[matrix_modified.indptr[node]:matrix_modified.indptr[node+1]] = 0
                matrix_modified[node, node] = 1.0
                rhs_modified[node] = value
        return matrix_modified, rhs_modified
    def _get_boundary_nodes(self, region: str, mesh: Optional[Dict]) -> List[int]:
        """Get boundary nodes for named region."""
        if mesh is None:
            raise ValueError("Mesh required for named boundary regions")
        if 'boundary_nodes' in mesh and region in mesh['boundary_nodes']:
            return mesh['boundary_nodes'][region]
        elif region == 'left':
            return [0]
        elif region == 'right':
            return [mesh.get('nx', 100) - 1]
        elif region == 'bottom':
            return list(range(mesh.get('nx', 100)))  # Bottom row in 2D
        elif region == 'top':
            nx = mesh.get('nx', 100)
            ny = mesh.get('ny', 100)
            return list(range((ny-1)*nx, ny*nx))  # Top row in 2D
        else:
            warnings.warn(f"Unknown boundary region: {region}")
            return []
class NeumannBC(BoundaryCondition):
    """Neumann (natural) boundary condition: ∂u/∂n = g.
    Prescribes the normal derivative of the solution at boundary points.
    """
    def apply(self, matrix: sparse.spmatrix, rhs: np.ndarray,
              mesh: Optional[Dict] = None, time: float = 0.0) -> Tuple[sparse.spmatrix, np.ndarray]:
        """Apply Neumann boundary condition."""
        # Get boundary value (flux)
        bc_value = self.evaluate_value(mesh, time)
        # Handle different location specifications
        if isinstance(self.location, int):
            nodes = [self.location]
        elif isinstance(self.location, list):
            nodes = self.location
        elif isinstance(self.location, str):
            nodes = self._get_boundary_nodes(self.location, mesh)
        else:
            raise ValueError(f"Invalid location specification: {self.location}")
        # Ensure bc_value is compatible with nodes
        if np.isscalar(bc_value):
            bc_values = [bc_value] * len(nodes)
        else:
            bc_values = np.asarray(bc_value)
            if len(bc_values) != len(nodes):
                bc_values = np.full(len(nodes), bc_values[0])
        # Modify RHS (add flux contribution)
        rhs_modified = rhs.copy()
        # Get mesh spacing for flux scaling
        dx = mesh.get('dx', 1.0) if mesh else 1.0
        for node, flux in zip(nodes, bc_values):
            if 0 <= node < len(rhs):
                # Add flux contribution to RHS
                # For finite differences: -k * ∂u/∂n ≈ -k * flux
                rhs_modified[node] += flux * dx  # Scale by mesh size
        return matrix, rhs_modified
    def _get_boundary_nodes(self, region: str, mesh: Optional[Dict]) -> List[int]:
        """Get boundary nodes for named region."""
        # Same implementation as DirichletBC
        if mesh is None:
            raise ValueError("Mesh required for named boundary regions")
        if 'boundary_nodes' in mesh and region in mesh['boundary_nodes']:
            return mesh['boundary_nodes'][region]
        elif region == 'left':
            return [0]
        elif region == 'right':
            return [mesh.get('nx', 100) - 1]
        elif region == 'bottom':
            return list(range(mesh.get('nx', 100)))
        elif region == 'top':
            nx = mesh.get('nx', 100)
            ny = mesh.get('ny', 100)
            return list(range((ny-1)*nx, ny*nx))
        else:
            warnings.warn(f"Unknown boundary region: {region}")
            return []
class RobinBC(BoundaryCondition):
    """Robin (mixed) boundary condition: α*u + β*∂u/∂n = g.
    Linear combination of Dirichlet and Neumann conditions.
    """
    def __init__(self, bc_data: BCData, alpha: float = 1.0, beta: float = 1.0):
        """Initialize Robin boundary condition.
        Args:
            bc_data: Boundary condition data
            alpha: Coefficient for u term
            beta: Coefficient for ∂u/∂n term
        """
        super().__init__(bc_data)
        self.alpha = alpha
        self.beta = beta
    def apply(self, matrix: sparse.spmatrix, rhs: np.ndarray,
              mesh: Optional[Dict] = None, time: float = 0.0) -> Tuple[sparse.spmatrix, np.ndarray]:
        """Apply Robin boundary condition."""
        # Get boundary value
        bc_value = self.evaluate_value(mesh, time)
        # Handle different location specifications
        if isinstance(self.location, int):
            nodes = [self.location]
        elif isinstance(self.location, list):
            nodes = self.location
        elif isinstance(self.location, str):
            nodes = self._get_boundary_nodes(self.location, mesh)
        else:
            raise ValueError(f"Invalid location specification: {self.location}")
        # Ensure bc_value is compatible with nodes
        if np.isscalar(bc_value):
            bc_values = [bc_value] * len(nodes)
        else:
            bc_values = np.asarray(bc_value)
            if len(bc_values) != len(nodes):
                bc_values = np.full(len(nodes), bc_values[0])
        # Modify matrix and RHS for Robin BC
        matrix_modified = matrix.copy()
        rhs_modified = rhs.copy()
        # Get mesh spacing
        dx = mesh.get('dx', 1.0) if mesh else 1.0
        for node, value in zip(nodes, bc_values):
            if 0 <= node < matrix.shape[0]:
                # Robin BC: α*u + β*(∂u/∂n) = g
                # Discretized: α*u_i + β*(u_{i+1} - u_i)/dx = g
                # Rearranged: (α - β/dx)*u_i + (β/dx)*u_{i+1} = g
                # Modify matrix row for Robin BC
                matrix_modified.data[matrix_modified.indptr[node]:matrix_modified.indptr[node+1]] = 0
                # Set coefficients for Robin BC
                matrix_modified[node, node] = self.alpha - self.beta / dx
                # Add contribution from neighboring node (simplified for 1D)
                if node < matrix.shape[1] - 1:
                    matrix_modified[node, node + 1] = self.beta / dx
                elif node > 0:
                    matrix_modified[node, node - 1] = self.beta / dx
                rhs_modified[node] = value
        return matrix_modified, rhs_modified
    def _get_boundary_nodes(self, region: str, mesh: Optional[Dict]) -> List[int]:
        """Get boundary nodes for named region."""
        # Same implementation as DirichletBC
        if mesh is None:
            raise ValueError("Mesh required for named boundary regions")
        if 'boundary_nodes' in mesh and region in mesh['boundary_nodes']:
            return mesh['boundary_nodes'][region]
        elif region == 'left':
            return [0]
        elif region == 'right':
            return [mesh.get('nx', 100) - 1]
        else:
            warnings.warn(f"Unknown boundary region: {region}")
            return []
class PeriodicBC(BoundaryCondition):
    """Periodic boundary condition: u(left) = u(right), ∂u/∂n(left) = ∂u/∂n(right).
    Enforces periodicity across domain boundaries.
    """
    def __init__(self, bc_data: BCData, paired_location: Union[int, str]):
        """Initialize periodic boundary condition.
        Args:
            bc_data: Boundary condition data
            paired_location: Location of paired boundary
        """
        super().__init__(bc_data)
        self.paired_location = paired_location
    def apply(self, matrix: sparse.spmatrix, rhs: np.ndarray,
              mesh: Optional[Dict] = None, time: float = 0.0) -> Tuple[sparse.spmatrix, np.ndarray]:
        """Apply periodic boundary condition."""
        # Get boundary nodes
        if isinstance(self.location, int):
            left_nodes = [self.location]
        elif isinstance(self.location, str):
            left_nodes = self._get_boundary_nodes(self.location, mesh)
        else:
            left_nodes = self.location
        if isinstance(self.paired_location, int):
            right_nodes = [self.paired_location]
        elif isinstance(self.paired_location, str):
            right_nodes = self._get_boundary_nodes(self.paired_location, mesh)
        else:
            right_nodes = self.paired_location
        if len(left_nodes) != len(right_nodes):
            raise ValueError("Periodic BC requires equal number of paired nodes")
        # Modify matrix and RHS for periodicity
        matrix_modified = matrix.copy()
        rhs_modified = rhs.copy()
        for left_node, right_node in zip(left_nodes, right_nodes):
            if 0 <= left_node < matrix.shape[0] and 0 <= right_node < matrix.shape[1]:
                # Enforce u(left) = u(right)
                # Replace left equation with: u_left - u_right = 0
                matrix_modified.data[matrix_modified.indptr[left_node]:matrix_modified.indptr[left_node+1]] = 0
                matrix_modified[left_node, left_node] = 1.0
                matrix_modified[left_node, right_node] = -1.0
                rhs_modified[left_node] = 0.0
        return matrix_modified, rhs_modified
    def _get_boundary_nodes(self, region: str, mesh: Optional[Dict]) -> List[int]:
        """Get boundary nodes for named region."""
        if mesh is None:
            raise ValueError("Mesh required for named boundary regions")
        if 'boundary_nodes' in mesh and region in mesh['boundary_nodes']:
            return mesh['boundary_nodes'][region]
        elif region == 'left':
            return [0]
        elif region == 'right':
            return [mesh.get('nx', 100) - 1]
        else:
            warnings.warn(f"Unknown boundary region: {region}")
            return []
class BoundaryConditions:
    """Container for multiple boundary conditions."""
    def __init__(self):
        """Initialize boundary conditions container."""
        self.conditions: List[BoundaryCondition] = []
    def add_bc(self, bc: BoundaryCondition):
        """Add boundary condition."""
        self.conditions.append(bc)
    def add_dirichlet(self, location: Union[int, List[int], str],
                     value: Union[float, np.ndarray, Callable]):
        """Add Dirichlet boundary condition."""
        bc_data = BCData(location=location, value=value)
        self.conditions.append(DirichletBC(bc_data))
    def add_neumann(self, location: Union[int, List[int], str],
                   flux: Union[float, np.ndarray, Callable]):
        """Add Neumann boundary condition."""
        bc_data = BCData(location=location, value=flux)
        self.conditions.append(NeumannBC(bc_data))
    def add_robin(self, location: Union[int, List[int], str],
                 value: Union[float, np.ndarray, Callable],
                 alpha: float = 1.0, beta: float = 1.0):
        """Add Robin boundary condition."""
        bc_data = BCData(location=location, value=value)
        self.conditions.append(RobinBC(bc_data, alpha, beta))
    def add_periodic(self, location1: Union[int, str], location2: Union[int, str]):
        """Add periodic boundary condition."""
        bc_data = BCData(location=location1, value=0.0)
        self.conditions.append(PeriodicBC(bc_data, location2))
    def apply_all(self, matrix: sparse.spmatrix, rhs: np.ndarray,
                  mesh: Optional[Dict] = None, time: float = 0.0) -> Tuple[sparse.spmatrix, np.ndarray]:
        """Apply all boundary conditions."""
        current_matrix = matrix
        current_rhs = rhs
        for bc in self.conditions:
            current_matrix, current_rhs = bc.apply(current_matrix, current_rhs, mesh, time)
        return current_matrix, current_rhs
    def get_dirichlet_nodes(self) -> List[int]:
        """Get all nodes with Dirichlet boundary conditions."""
        dirichlet_nodes = []
        for bc in self.conditions:
            if isinstance(bc, DirichletBC):
                if isinstance(bc.location, int):
                    dirichlet_nodes.append(bc.location)
                elif isinstance(bc.location, list):
                    dirichlet_nodes.extend(bc.location)
        return dirichlet_nodes
    def is_well_posed(self, problem_type: str = 'elliptic') -> bool:
        """Check if boundary conditions make problem well-posed."""
        dirichlet_nodes = self.get_dirichlet_nodes()
        if problem_type == 'elliptic':
            # Elliptic problems need at least one Dirichlet BC
            return len(dirichlet_nodes) > 0
        elif problem_type == 'parabolic':
            # Parabolic problems are usually well-posed with any reasonable BC
            return True
        elif problem_type == 'hyperbolic':
            # Hyperbolic problems need appropriate characteristic BC
            return True
        else:
            return True
# Utility functions
def apply_boundary_conditions(matrix: sparse.spmatrix, rhs: np.ndarray,
                            boundary_conditions: Dict[str, Any],
                            mesh: Optional[Dict] = None,
                            time: float = 0.0) -> Tuple[sparse.spmatrix, np.ndarray]:
    """Apply boundary conditions from dictionary specification.
    Args:
        matrix: System matrix
        rhs: Right-hand side vector
        boundary_conditions: BC specification dictionary
        mesh: Mesh information
        time: Current time
    Returns:
        Modified matrix and RHS
    """
    # Create BoundaryConditions object from dictionary
    bc_container = BoundaryConditions()
    # Parse boundary conditions dictionary
    for bc_type, bc_specs in boundary_conditions.items():
        if bc_type == 'dirichlet':
            for location, value in bc_specs.items():
                bc_container.add_dirichlet(location, value)
        elif bc_type == 'neumann':
            for location, flux in bc_specs.items():
                bc_container.add_neumann(location, flux)
        elif bc_type == 'robin':
            for location, specs in bc_specs.items():
                value = specs.get('value', 0.0)
                alpha = specs.get('alpha', 1.0)
                beta = specs.get('beta', 1.0)
                bc_container.add_robin(location, value, alpha, beta)
        elif bc_type == 'periodic':
            for pair in bc_specs:
                location1, location2 = pair
                bc_container.add_periodic(location1, location2)
    # Apply all boundary conditions
    return bc_container.apply_all(matrix, rhs, mesh, time)
def create_boundary_condition(bc_type: str, location: Union[int, List[int], str],
                            value: Union[float, np.ndarray, Callable],
                            **kwargs) -> BoundaryCondition:
    """Factory function for creating boundary conditions.
    Args:
        bc_type: Type of BC ('dirichlet', 'neumann', 'robin', 'periodic')
        location: Boundary location
        value: Boundary value
        **kwargs: Additional parameters
    Returns:
        BoundaryCondition object
    """
    bc_data = BCData(location=location, value=value)
    if bc_type == 'dirichlet':
        return DirichletBC(bc_data)
    elif bc_type == 'neumann':
        return NeumannBC(bc_data)
    elif bc_type == 'robin':
        alpha = kwargs.get('alpha', 1.0)
        beta = kwargs.get('beta', 1.0)
        return RobinBC(bc_data, alpha, beta)
    elif bc_type == 'periodic':
        paired_location = kwargs.get('paired_location')
        if paired_location is None:
            raise ValueError("Periodic BC requires paired_location")
        return PeriodicBC(bc_data, paired_location)
    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")
def validate_boundary_conditions(boundary_conditions: Dict[str, Any],
                                domain: Dict[str, np.ndarray]) -> bool:
    """Validate boundary condition specification.
    Args:
        boundary_conditions: BC specification
        domain: Domain specification
    Returns:
        True if valid
    """
    # Check that all specified nodes exist
    max_node = domain.get('nx', 100) - 1
    for bc_type, bc_specs in boundary_conditions.items():
        if bc_type in ['dirichlet', 'neumann']:
            for location, value in bc_specs.items():
                if isinstance(location, int):
                    if not (0 <= location <= max_node):
                        warnings.warn(f"Node {location} outside domain")
                        return False
    return True
# Convenience functions for common boundary conditions
def homogeneous_dirichlet(nodes: Union[int, List[int]]) -> DirichletBC:
    """Create homogeneous Dirichlet BC (u = 0)."""
    bc_data = BCData(location=nodes, value=0.0)
    return DirichletBC(bc_data)
def homogeneous_neumann(nodes: Union[int, List[int]]) -> NeumannBC:
    """Create homogeneous Neumann BC (∂u/∂n = 0)."""
    bc_data = BCData(location=nodes, value=0.0)
    return NeumannBC(bc_data)
def time_dependent_dirichlet(nodes: Union[int, List[int]],
                           time_function: Callable) -> DirichletBC:
    """Create time-dependent Dirichlet BC."""
    bc_data = BCData(location=nodes, value=time_function, time_dependent=True)
    return DirichletBC(bc_data)